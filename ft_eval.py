"""Inference + evaluation, identical to the frozen-SAM3 runs of Method B (run 1 / b1_pool notebooks).

Per image and per prompt (one prompt at a time, as Sam3Processor):
  score  = sigmoid(query logit) * sigmoid(presence logit)
  keep   = queries with score > STORE_THR (0.05), at most STORE_TOPK (30) per prompt
  mask   = bilinear-upsampled mask logit > 0  (= sigmoid > 0.5)
  pixel  -> label of the covering mask with the highest score (ties -> earlier prompt, then earlier query)
  at threshold tau a pixel keeps that label iff the winner's score > tau, else label 0.
Masks are ranked once by score, so ONE pass gives the confusion matrix at every tau of the grid
(the trick of the diagnose notebook): hist[GT, winning label, score bin] -> conf_tau [NT, K, K].

Modes (CLI):
  val        : model <ds>_<cfg>_best.pt on the whole val split, at tau_A chosen on holdout;
               writes raw (run-1 format), pred PNG, conf_tau, report.
  preflight  : SAM3 as released on the 50 reference holdout images (assets/<ds>_ref_pred) -> agreement
               with the Kaggle run-1 prediction (environment check + hardware noise).
  reload     : best checkpoint reloaded from disk == model in memory (holdout images of the history).
  rawcheck   : (local, CPU) rebuild maps from the run-1 raw .npz of Method B and compare with its pred/
               and its tau sweep; checks the map painting / metric code without SAM3.
"""
from __future__ import annotations

import argparse
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ft_common as fc

RAW_FORMAT = {
    "orig_hw": "int32 [2]  original image height, width",
    "prompts": "str [P]  text prompt per class (prompt p -> label p+1)",
    "presence_logit": "float32 [P]  presence logit (image-level 'concept is present')",
    "query_logits": "float16 [P, Q]  logit of all Q=200 queries (before sigmoid, before presence)",
    "k_prompt": "int16 [M]  prompt index of each kept query",
    "k_query": "int16 [M]  query index (0..Q-1) of each kept query",
    "k_prob": "float16 [M]  score = sigmoid(query logit) * sigmoid(presence logit)",
    "k_box": "float16 [M, 4]  box (cx, cy, w, h), normalised to [0, 1]",
    "k_mask": "uint8 [M, h, w]  mask probability * 255 at SAM3's low mask resolution",
    "k_embed": "float16 [M, D]  decoder query embedding",
    "k_feat_in": "float16 [M, C]  backbone feature averaged inside the mask (weights = mask prob)",
    "k_feat_ring": "float16 [M, C]  backbone feature averaged in a ring around the mask (context)",
    "sem_seg": "uint8 [P, hs, ws]  semantic-head probability * 255 per prompt",
    "global_feat": "float16 [C]  backbone feature averaged over the whole image",
    "feat_hw": "int32 [2]  resolution of the backbone feature map used for pooling",
    "conf": "int64 [K, K]  confusion of this image at 0.5 (GT row, prediction column), as run 1",
    "conf_tau": "int64 [NT, K, K]  confusion of this image at every tau of the grid",
    "time_s": "float  seconds spent on this image",
}


def autocast_ctx(device):
    if str(device).startswith("cuda") and fc.PRECISION == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------------------------------
# Map painting at every tau (shared by the model path and the raw-rebuild path)
# ----------------------------------------------------------------------------------------------
def winner(scores, labels, logits, H, W, dev):
    """scores [M] float, labels [M] int (1..C), logits [M, h, w] mask logits (torch, any device).
    Kept order = prompt order then query order. -> (score of winner [H, W], label of winner [H, W])."""
    M = int(scores.numel())
    s_sorted_ext = torch.tensor([-1.0], device=dev)
    if M == 0:
        return s_sorted_ext.expand(H, W), torch.zeros((H, W), dtype=torch.long, device=dev)
    order = torch.sort(scores.float().cpu(), descending=True, stable=True).indices
    best = torch.full((H, W), M, dtype=torch.int32, device=dev)
    for s in range(0, M, fc.MASK_CHUNK):
        idx = order[s:s + fc.MASK_CHUNK]
        lg = logits[idx.to(logits.device)].to(dev).float()
        cov = F.interpolate(lg.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0
        r = torch.arange(s, s + len(idx), device=dev, dtype=torch.int32).view(-1, 1, 1)
        best = torch.minimum(best, torch.where(cov, r, torch.full_like(r, M)).amin(0))
    s_sorted = torch.cat([scores.float().cpu()[order], torch.tensor([-1.0])]).to(dev)
    l_sorted = torch.cat([labels.long().cpu()[order], torch.tensor([0])]).to(dev)
    b = best.long()
    return s_sorted[b], l_sorted[b]


def hist_of(s_w, l_w, gt, K):
    """-> hist [K, K, NB] int64 over valid GT pixels (gt < K)."""
    dev = s_w.device
    taus = torch.tensor(fc.TAUS, dtype=torch.float32, device=dev)
    b = torch.bucketize(s_w.float(), taus)                 # number of grid values strictly below the score
    g = torch.as_tensor(gt, device=dev).long()
    v = g < K
    key = (g[v] * K + l_w[v]) * fc.NB + b[v]
    return torch.bincount(key, minlength=K * K * fc.NB).view(K, K, fc.NB).cpu().numpy().astype(np.int64)


def label_at(s_w, l_w, tau):
    return torch.where(s_w > tau, l_w, torch.zeros_like(l_w))


def logits_from_uint8(k_mask):
    p = torch.from_numpy(k_mask).float().clamp(0.5, 254.5) / 255.0
    return torch.log(p) - torch.log1p(-p)


# ----------------------------------------------------------------------------------------------
# SAM3 inference on one image (code of the Kaggle runs, Sam3Runner.process)
# ----------------------------------------------------------------------------------------------
class Sam3Infer:
    def __init__(self, model, device, prompts):
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model, self.device = model, torch.device(device)
        self.proc = Sam3Processor(model, device=str(self.device), confidence_threshold=fc.BASE_THR)
        self.prompts = list(prompts)
        self._text = {}                                    # per evaluation run (the text resizer is trained in A2)

    def text(self, prompt):
        if prompt not in self._text:
            with autocast_ctx(self.device):
                self._text[prompt] = self.model.backbone.forward_text([prompt], device=self.device)
        return self._text[prompt]

    @torch.inference_mode()
    def run(self, image, want_raw):
        """image: PIL RGB. -> dict(H, W, scores [M], labels [M], logits [M,h,w] on device, raw or None)"""
        t0 = time.time()
        W, H = image.size
        with autocast_ctx(self.device):
            state = self.proc.set_image(image)
        bo = state["backbone_out"]
        if want_raw:
            fmap = bo["backbone_fpn"][-1] if "backbone_fpn" in bo else bo["vision_features"]
            feat = fmap[0].float()
            C, fh, fw = feat.shape
            feat_flat = feat.reshape(C, -1).T.contiguous()
        ks, kl, kq, kp, kmask = [], [], [], [], []
        presence, qlog, sem, kbox, kemb, kfin, kfring = [], [], [], [], [], [], []
        mask_hw, embed_dim = None, 0
        for p_idx, prompt in enumerate(self.prompts):
            self.proc.reset_all_prompts(state)
            state["backbone_out"].update(self.text(prompt))
            state["geometric_prompt"] = self.model._get_dummy_prompt()
            with autocast_ctx(self.device):
                out = self.model.forward_grounding(backbone_out=state["backbone_out"], find_input=self.proc.find_stage,
                                                   geometric_prompt=state["geometric_prompt"], find_target=None)
            probs = (out["pred_logits"].float().sigmoid() * out["presence_logit_dec"].float().sigmoid().unsqueeze(1)).squeeze(-1)[0]
            m_lr = out["pred_masks"][0].float()                    # [Q, h, w] logits
            mask_hw = tuple(m_lr.shape[-2:])
            order = torch.argsort(probs, descending=True)
            sel = order[probs[order] > fc.STORE_THR][: fc.STORE_TOPK]
            sel, _ = torch.sort(sel)
            if want_raw:
                presence.append(float(out["presence_logit_dec"].float().reshape(-1)[0]))
                qlog.append(out["pred_logits"][0, :, 0].float().half().cpu().numpy())
                if "semantic_seg" in out:
                    s_map = out["semantic_seg"]
                    s_map = s_map.reshape(-1, *s_map.shape[-2:])[0].float().sigmoid() * 255
                    sem.append(s_map.round().to(torch.uint8).cpu().numpy())
            if sel.numel() == 0:
                continue
            ks.append(probs[sel])
            kl.append(torch.full((sel.numel(),), p_idx + 1, dtype=torch.long, device=probs.device))
            kmask.append(m_lr[sel])
            if want_raw:
                mp = m_lr[sel].sigmoid()
                Mi = mp.shape[0]
                w_in = F.interpolate(mp.unsqueeze(1), size=(fh, fw), mode="area")[:, 0]
                core = (w_in > 0.5).float()
                ring = (F.max_pool2d(core.unsqueeze(1), 2 * fc.RING_RADIUS + 1, stride=1, padding=fc.RING_RADIUS)[:, 0] - core).clamp(min=0)
                w_in, ring = w_in.reshape(Mi, -1), ring.reshape(Mi, -1)
                kfin.append(((w_in @ feat_flat) / w_in.sum(1, keepdim=True).clamp(min=1e-6)).half().cpu().numpy())
                kfring.append(((ring @ feat_flat) / ring.sum(1, keepdim=True).clamp(min=1e-6)).half().cpu().numpy())
                kp.append(np.full(Mi, p_idx, np.int16))
                kq.append(sel.cpu().numpy().astype(np.int16))
                kbox.append(out["pred_boxes"][0, sel].float().half().cpu().numpy())
                if "queries" in out:
                    q = out["queries"][0, sel].float()
                    embed_dim = q.shape[-1]
                    kemb.append(q.half().cpu().numpy())
        dev = self.device
        scores = torch.cat(ks) if ks else torch.zeros(0, device=dev)
        labels = torch.cat(kl) if kl else torch.zeros(0, dtype=torch.long, device=dev)
        h, w = mask_hw
        logits = torch.cat(kmask) if kmask else torch.zeros((0, h, w), device=dev)
        raw = None
        if want_raw:
            def cat(lst, shape, dtype):
                return np.concatenate(lst) if lst else np.zeros(shape, dtype)
            raw = {"orig_hw": np.array([H, W], np.int32), "prompts": np.array(self.prompts),
                   "presence_logit": np.array(presence, np.float32), "query_logits": np.stack(qlog).astype(np.float16),
                   "k_prompt": cat(kp, (0,), np.int16), "k_query": cat(kq, (0,), np.int16),
                   "k_prob": scores.half().cpu().numpy(), "k_box": cat(kbox, (0, 4), np.float16),
                   "k_mask": (logits.sigmoid() * 255).round().to(torch.uint8).cpu().numpy(),
                   "k_embed": cat(kemb, (0, embed_dim), np.float16), "k_feat_in": cat(kfin, (0, C), np.float16),
                   "k_feat_ring": cat(kfring, (0, C), np.float16),
                   "sem_seg": np.stack(sem) if sem else np.zeros((0, 1, 1), np.uint8),
                   "global_feat": feat_flat.mean(0).half().cpu().numpy(), "feat_hw": np.array([fh, fw], np.int32)}
        return {"H": H, "W": W, "scores": scores, "labels": labels, "logits": logits, "raw": raw, "time_s": time.time() - t0}


def _loader(sp, paths, split, ids, workers=2):
    """Prefetch (image, GT) in background threads (PNG decoding of 2048x1024 images is slow)."""
    from PIL import Image

    def load(i):
        return i, Image.open(sp.image_path(paths, split, i)).convert("RGB"), sp.load_gt(paths, split, i)
    with ThreadPoolExecutor(workers) as ex:
        for k in range(0, len(ids), 8):
            yield from ex.map(load, ids[k:k + 8])


def evaluate(infer, sp, paths, split, ids, desc, raw_dir=None, pred_dir=None, pred_tau=None, ref_dir=None):
    """Run the model on ids. Returns dict(conf_tau [N, NT, K, K] int64, ids, seconds, [agree])."""
    from PIL import Image
    from tqdm import tqdm
    dev = infer.device
    conf_tau = np.zeros((len(ids), fc.NT, sp.K, sp.K), np.int64)
    agree, npix, t0 = 0, 0, time.time()
    todo = []
    for n, i in enumerate(ids):
        rp = Path(raw_dir) / f"{i}.npz" if raw_dir else None
        pp = Path(pred_dir) / f"{i}.png" if pred_dir else None
        if rp is not None and rp.exists() and (pp is None or pp.exists()):
            with np.load(rp) as z:                         # resume: image already done
                conf_tau[n] = z["conf_tau"]
        else:
            todo.append(n)
    pos = {ids[n]: n for n in todo}
    for i, image, gt in tqdm(_loader(sp, paths, split, [ids[n] for n in todo]), total=len(todo), desc=desc, mininterval=30):
        n = pos[i]
        r = infer.run(image, want_raw=raw_dir is not None)
        H, W = r["H"], r["W"]
        if gt.shape != (H, W):
            raise fc.PipelineError(f"{i}: GT {gt.shape} != image {(H, W)}")
        s_w, l_w = winner(r["scores"], r["labels"], r["logits"], H, W, dev)
        ct = fc.conf_tau_from_hist(hist_of(s_w, l_w, gt, sp.K))
        conf_tau[n] = ct
        if ref_dir is not None:
            ref = torch.from_numpy(np.array(Image.open(Path(ref_dir) / f"{i}.png"))).to(dev).long()
            agree += int((label_at(s_w, l_w, fc.BASE_THR) == ref).sum())
            npix += H * W
        if pred_dir is not None:
            Path(pred_dir).mkdir(parents=True, exist_ok=True)
            lab = label_at(s_w, l_w, pred_tau).to(torch.uint8).cpu().numpy()
            Image.fromarray(lab).save(Path(pred_dir) / f"{i}.png")
        if raw_dir is not None:
            raw = dict(r["raw"], conf=ct[fc.J_BASE], conf_tau=ct, time_s=np.float32(r["time_s"]))
            fc.save_npz_atomic(Path(raw_dir) / f"{i}.npz", **raw)
    out = {"conf_tau": conf_tau, "ids": list(ids), "seconds": time.time() - t0, "n_run": len(todo)}
    if ref_dir is not None:
        out["agree"] = agree / max(npix, 1)
    return out


def summarize(conf_tau, sp):
    """conf_tau [N, NT, K, K] -> mIoU per tau, chosen tau (rule of tau*), per-class IoU."""
    tot = conf_tau.sum(0)
    m = [fc.miou(tot[j], sp) for j in range(fc.NT)]
    j = fc.choose_tau(m)
    return {"miou_by_tau": m, "tau_index": j, "tau": float(fc.TAUS[j]), "miou": m[j], "miou_0.5": m[fc.J_BASE],
            "iou_at_tau": dict(zip(sp.names, fc.per_label_iou(tot[j]).tolist())),
            "iou_at_0.5": dict(zip(sp.names, fc.per_label_iou(tot[fc.J_BASE]).tolist()))}


# ----------------------------------------------------------------------------------------------
# CLI modes
# ----------------------------------------------------------------------------------------------
def mode_val(paths, ds, cfg, limit=None):
    import ft_model
    sp = fc.spec(ds)
    best_pt, best_js = fc.model_file(paths, ds, cfg, "best.pt"), fc.model_file(paths, ds, cfg, "best.json")
    fc.need(best_pt, "best checkpoint (train step not finished?)")
    best = fc.load_json(best_js)
    tau_j = int(best["tau_index"])
    ids = sp.list_ids(paths, "val")[:limit] if limit else sp.list_ids(paths, "val")
    dev = "cuda" if paths["device"] == "cuda" else "cpu"
    model, meta = ft_model.build_eval_model(cfg, paths["sam3_checkpoint"], best_pt, device=dev)
    assert meta["epoch"] == best["epoch"], (meta, best)
    infer = Sam3Infer(model, dev, sp.prompts)
    res = evaluate(infer, sp, paths, "val", ids, f"{ds} {cfg} val",
                   raw_dir=fc.model_file(paths, ds, cfg, "val_raw"), pred_dir=fc.model_file(paths, ds, cfg, "val_pred"),
                   pred_tau=float(fc.TAUS[tau_j]))
    s = summarize(res["conf_tau"], sp)
    fc.save_npz_atomic(fc.model_file(paths, ds, cfg, "val_conf_tau.npz"), ids=np.array(ids), taus=fc.TAUS,
                       conf=res["conf_tau"].astype(np.int32), label_names=np.array(sp.names))
    report = {"dataset": ds, "config": cfg, "split": "val", "n_images": len(ids), "limit": limit,
              "tau_A": float(fc.TAUS[tau_j]), "tau_A_from": "holdout (best.json)", "best_epoch": best["epoch"],
              "miou_at_tau_A": s["miou_by_tau"][tau_j], "miou_at_0.5": s["miou_0.5"],
              "miou_by_tau": dict(zip(map(str, fc.TAU_GRID), s["miou_by_tau"])),
              "iou_at_tau_A": dict(zip(sp.names, fc.per_label_iou(res["conf_tau"].sum(0)[tau_j]).tolist())),
              "iou_at_0.5": s["iou_at_0.5"], "eval_idx": sp.eval_idx,
              "note_val_best_tau": "tau maximising val mIoU is NOT used for anything (val is looked at once)",
              "seconds": res["seconds"], "raw_format": RAW_FORMAT, "created": fc.now()}
    fc.save_json_atomic(fc.model_file(paths, ds, cfg, "val_report.json"), report)
    fc.log(f"{ds} {cfg} val: mIoU {report['miou_at_tau_A']:.4f} at tau_A={report['tau_A']} "
           f"({report['miou_at_0.5']:.4f} at 0.5), {len(ids)} images")
    return report


def ref_ids(ds):
    meta = fc.load_json(fc.ASSETS_DIR / f"{ds}_ref_pred" / "ref_meta.json")
    if meta is None:
        raise fc.PipelineError(f"assets/{ds}_ref_pred/ref_meta.json missing")
    return meta["ids"], meta


def mode_preflight(paths, ds, out_json, limit=None):
    import ft_model
    sp = fc.spec(ds)
    ids, meta = ref_ids(ds)
    ids = ids[:limit] if limit else ids
    assert meta["prompts"] == sp.prompts, "reference prompts != prompts of this code"
    dev = "cuda" if paths["device"] == "cuda" else "cpu"
    model, _ = ft_model.build_eval_model("base", paths["sam3_checkpoint"], device=dev)
    infer = Sam3Infer(model, dev, sp.prompts)
    res = evaluate(infer, sp, paths, "train", ids, f"{ds} preflight (SAM3 as released)",
                   ref_dir=fc.ASSETS_DIR / f"{ds}_ref_pred")
    s = summarize(res["conf_tau"], sp)
    out = {"dataset": ds, "n_images": len(ids), "agreement_with_kaggle_pred": res["agree"],
           "miou_0.5_here": s["miou_0.5"], "reference": {k: meta[k] for k in ("source", "torch", "gpus", "sam3_commit")},
           "seconds_per_image": res["seconds"] / max(len(ids), 1), "created": fc.now()}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    fc.save_json_atomic(out_json, out)
    fc.log(f"{ds} preflight: {100 * out['agreement_with_kaggle_pred']:.3f}% pixels agree with the Kaggle run-1 "
           f"prediction ({len(ids)} images, {out['seconds_per_image']:.2f} s/image)")
    return out


def mode_reload(paths, ds, cfg, n_images=2):
    """The best checkpoint reloaded from disk gives the holdout confusion stored with it."""
    import ft_model
    sp = fc.spec(ds)
    best = fc.load_json(fc.model_file(paths, ds, cfg, "best.json"))
    with np.load(fc.model_file(paths, ds, cfg, "holdout_conf_tau.npz")) as z:
        ids, conf = list(z["ids"])[:n_images], z["conf"][:n_images].astype(np.int64)
    dev = "cuda" if paths["device"] == "cuda" else "cpu"
    model, meta = ft_model.build_eval_model(cfg, paths["sam3_checkpoint"], fc.model_file(paths, ds, cfg, "best.pt"), device=dev)
    assert meta["epoch"] == best["epoch"]
    res = evaluate(Sam3Infer(model, dev, sp.prompts), sp, paths, "train", ids, f"{ds} {cfg} reload check")
    j = int(best["tau_index"])
    a, b = conf[:, j], res["conf_tau"][:, j]
    diff = float(np.abs(a - b).sum() / max(2 * a.sum(), 1))
    return {"images": ids, "pixel_disagreement": diff, "identical": bool(np.array_equal(a, b))}


def mode_rawcheck(paths, ds, raw_dir, max_images=None):
    """Local check without SAM3: Method-B run-1 raw -> maps; compare with its pred/ and tau_sweep.csv."""
    from PIL import Image
    from tqdm import tqdm
    sp = fc.spec(ds)
    raw_dir = Path(raw_dir)
    meta = fc.load_json(raw_dir / "meta.json")
    assert meta["prompts"] == sp.prompts and meta["dataset"] == ds, "raw run does not match this dataset spec"
    fit, hold = sp.split_train()
    ids = fit + hold
    if max_images:
        ids = fit[:max_images] + hold[:max_images]
    nfit = len(fit) if not max_images else min(max_images, len(fit))
    conf_tau = np.zeros((len(ids), fc.NT, sp.K, sp.K), np.int64)
    agree = npix = 0
    dev = torch.device("cpu")
    for n, i in enumerate(tqdm(ids, desc=f"{ds} raw rebuild", mininterval=10)):
        with np.load(raw_dir / "raw" / f"{i}.npz") as z:
            H, W = (int(v) for v in z["orig_hw"])
            scores = torch.from_numpy(z["k_prob"].astype(np.float32))
            labels = torch.from_numpy(z["k_prompt"].astype(np.int64) + 1)
            logits = logits_from_uint8(z["k_mask"])
            conf05 = z["conf"]
        gt = sp.load_gt(paths, "train", i)
        s_w, l_w = winner(scores, labels, logits, H, W, dev)
        ct = fc.conf_tau_from_hist(hist_of(s_w, l_w, gt, sp.K))
        conf_tau[n] = ct
        ref = torch.from_numpy(np.array(Image.open(raw_dir / "pred" / f"{i}.png"))).long()
        agree += int((label_at(s_w, l_w, fc.BASE_THR) == ref).sum())
        npix += H * W
        assert np.array_equal(conf05.sum(1), ct[fc.J_BASE].sum(1)), f"{i}: GT pixel counts differ from run 1"
    fit_m = [fc.miou(conf_tau[:nfit, j].sum(0), sp) for j in range(fc.NT)]
    hold_m = [fc.miou(conf_tau[nfit:, j].sum(0), sp) for j in range(fc.NT)]
    return {"dataset": ds, "n_images": len(ids), "agreement_with_run1_pred": agree / npix,
            "miou_fit_by_tau": fit_m, "miou_holdout_by_tau": hold_m, "tau_star_rule": float(fc.TAUS[fc.choose_tau(fit_m)])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["val", "preflight", "reload", "rawcheck"])
    ap.add_argument("--paths", required=True)
    ap.add_argument("--dataset", required=True, choices=["voc2012", "cityscapes"])
    ap.add_argument("--cfg", choices=list(fc.FT_CONFIGS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--raw-dir", default=None, help="rawcheck: Method-B run-1 folder holding raw/, pred/, meta.json")
    args = ap.parse_args()
    paths = fc.load_paths(args.paths)
    if args.mode == "val":
        mode_val(paths, args.dataset, args.cfg, args.limit)
    elif args.mode == "preflight":
        mode_preflight(paths, args.dataset, args.out_json, args.limit)
    elif args.mode == "reload":
        r = mode_reload(paths, args.dataset, args.cfg)
        fc.save_json_atomic(args.out_json, r)
        fc.log(f"reload check: disagreement {r['pixel_disagreement']:.2e}, identical={r['identical']}")
    else:
        r = mode_rawcheck(paths, args.dataset, args.raw_dir, args.limit)
        if args.out_json:
            fc.save_json_atomic(args.out_json, r)
        fc.log(f"rawcheck {args.dataset}: agreement {100 * r['agreement_with_run1_pred']:.3f}%, "
               f"tau* rule -> {r['tau_star_rule']}")


if __name__ == "__main__":
    fc.run_main(main)
