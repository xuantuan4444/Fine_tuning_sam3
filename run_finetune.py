"""THE single command of Method A (one job, nothing in parallel):

  python run_finetune.py --config paths.yaml            run everything, start to end
  python run_finetune.py --config paths.yaml --dry-run  print the plan (what is done / what will run)
  python run_finetune.py --config paths.yaml --smoke    whole chain on a few images (outputs in <out_root>_smoke)

Steps (killed? run the SAME command again: finished steps are skipped, a model in training resumes):
  0 checks     python / torch / GPU, sam3 commit, sha256 of sam3.pt, dataset files, disk
  1 prepare    GT -> COCO + RLE for train-fit of Cityscapes and VOC (ft_prepare_data.py)
  2 preflight  SAM3 as released on 50 holdout images vs the Kaggle run-1 prediction (>= 98% pixels);
               30-step A2 and A1 runs (speed, memory, best.pt save/reload); hours estimate
  3 A2 Cityscapes  4 A2 VOC  5 A1 VOC, A1 Cityscapes
               each: train on fit, holdout after every epoch, early stop (ft_train.py), then val ONCE with
               the best checkpoint at tau_A chosen on holdout (ft_eval.py --mode val), then pack
  6 summary    sam3ft_summary.csv + sam3ft_results_pack.zip (small files to send back)
The pack is rebuilt after every model, so results of finished models are never lost.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import ft_common as fc

PY = sys.executable
HERE = fc.REPO_DIR
AGREE_MIN = 0.98                  # preflight: SAM3 here vs Kaggle run-1 prediction (T4 bf16 -> H200 bf16 flips
                                  # a few border pixels; a wrong commit / checkpoint / preprocessing drops far lower)
RELOAD_MAX = 1e-3                 # preflight: best.pt reloaded vs model in memory
DISK_MIN_GB = 25
PREFLIGHT = {"fit_limit": 30, "holdout_limit": 4, "epoch_cap": 1}
SMOKE = {"fit_limit": 4, "holdout_limit": 2, "epoch_cap": 2, "val_limit": 2}


class Runner:
    def __init__(self, paths, log_path):
        self.paths = paths
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_f = open(log_path, "a", encoding="utf-8", buffering=1)
        self.say(f"log file: {log_path}")

    def say(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")

    def banner(self, title):
        self.say("=" * 90)
        self.say(title)
        self.say("=" * 90)

    def run(self, args, what):
        """Run a step as a child process; its output goes to the screen and to the log."""
        cmd = [PY, "-u"] + [str(a) for a in args]
        self.say(f"$ {' '.join(cmd)}")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            txt = chunk.decode("utf-8", "replace")
            sys.stdout.write(txt)
            sys.stdout.flush()
            self.log_f.write(txt)
        rc = p.wait()
        self.say(f"{what}: exit code {rc} after {(time.time() - t0) / 60:.1f} min")
        if rc != 0:
            raise fc.PipelineError(f"{what} failed (exit code {rc}); see the log above. "
                                   f"Fix it and run the same command again: finished steps are skipped.")


# ----------------------------------------------------------------------------------------------
# Step 0: checks
# ----------------------------------------------------------------------------------------------
def sam3_commit():
    import importlib.metadata as md
    try:
        d = md.distribution("sam3")
    except md.PackageNotFoundError:
        return None, "sam3 is not installed"
    info = json.loads(d.read_text("direct_url.json") or "{}")
    if "vcs_info" in info:
        return info["vcs_info"].get("commit_id"), info.get("url")
    url = info.get("url", "")
    if url.startswith("file://"):
        src = url[len("file://"):].lstrip("/") if os.name == "nt" else url[len("file://"):]
        try:
            out = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
            if out.returncode == 0:
                return out.stdout.strip(), src
        except Exception:
            pass
    return None, f"cannot read the commit of the installed sam3 ({url or 'no direct_url.json'})"


def ckpt_sha(paths):
    p = Path(paths["sam3_checkpoint"])
    fc.need(p, "sam3.pt")
    cache_f = Path(paths["out_root"]) / "checks_sha_cache.json"
    key = f"{p.resolve()}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
    cache = fc.load_json(cache_f, {})
    if cache.get("key") != key:
        cache = {"key": key, "sha256": fc.sha256_file(p)}
        fc.save_json_atomic(cache_f, cache)
    return cache["sha256"]


def count_files(paths, rn):
    out, problems = {}, []
    v, c = fc.spec("voc2012"), fc.spec("cityscapes")
    for sp in (v, c):
        for split in ("train", "val"):
            ids = sp.list_ids(paths, split)                      # raises if the count is wrong
            miss_img = [i for i in ids if not sp.image_path(paths, split, i).exists()]
            gts = [sp.gt_paths(paths, split, i) for i in ids]
            keys = list(gts[0].keys()) if split == "train" else (["cls"] if sp.name == "voc2012" else ["label"])
            miss_gt = {k: sum(not g[k].exists() for g in gts) for k in keys}
            out[f"{sp.name}_{split}"] = {"images": len(ids), "missing_images": len(miss_img), "missing_gt": miss_gt}
            if miss_img:
                problems.append(f"{sp.name} {split}: {len(miss_img)} images missing, e.g. {sp.image_path(paths, split, miss_img[0])}")
            for k, n in miss_gt.items():
                if n:
                    hint = (" (Cityscapes *_gtFine_instanceIds.png are inside gtFine_trainvaltest.zip: copy the whole gtFine folder)"
                            if k == "inst" else "")
                    problems.append(f"{sp.name} {split}: {n} GT files '{k}' missing{hint}")
    return out, problems


def step_checks(paths, rn, smoke):
    rn.banner("STEP 0: checks")
    rep, problems = {"python": sys.version.split()[0], "platform": platform.platform(), "time": fc.now()}, []
    for mod in ("torch", "torchvision", "numpy", "scipy", "pycocotools", "hydra", "omegaconf", "submitit", "fvcore",
                "einops", "psutil", "iopath", "yaml", "PIL", "tqdm", "tensorboard", "sam3"):
        try:
            m = importlib.import_module(mod)
            rep[f"v_{mod}"] = getattr(m, "__version__", "ok")
        except Exception as e:
            problems.append(f"cannot import {mod}: {e}")
    if paths["device"] == "cuda":
        # the repo's focal losses (sam3/train/loss/sigmoid_focal_loss.py) are Triton GPU kernels;
        # triton ships with the Linux CUDA wheels of torch
        try:
            rep["v_triton"] = importlib.import_module("triton").__version__
        except Exception as e:
            problems.append(f"cannot import triton ({e}); it comes with the Linux CUDA build of torch")
    if problems:
        raise fc.PipelineError("missing packages:\n  " + "\n  ".join(problems) + "\nInstall them as in README.md")
    import psutil
    import torch
    rep["ram_gb"] = psutil.virtual_memory().total / 2**30
    if paths["device"] == "cuda":
        if not torch.cuda.is_available():
            raise fc.PipelineError("torch.cuda.is_available() is False (device: cuda in paths.yaml)")
        free, total = torch.cuda.mem_get_info()
        rep.update(gpu=torch.cuda.get_device_name(0), gpu_total_gb=total / 2**30, gpu_free_gb=free / 2**30,
                   torch_cuda=torch.version.cuda, bf16=torch.cuda.is_bf16_supported())
        if not rep["bf16"]:
            problems.append("GPU without bf16 support")
        if total / 2**30 < 30:
            rn.say(f"WARNING: GPU memory {total / 2**30:.1f} GB < 30 GB (planned for a 35 GB MIG slice)")
    commit, where = sam3_commit()
    rep["sam3_commit"], rep["sam3_source"] = commit, where
    if commit != fc.SAM3_COMMIT:
        problems.append(f"sam3 commit {commit} != {fc.SAM3_COMMIT} ({where}). Install the pinned commit (README.md)")
    sha = ckpt_sha(paths)
    rep["sam3_sha256"] = sha
    if sha != fc.SAM3_CKPT_SHA256:
        problems.append(f"sha256 of {paths['sam3_checkpoint']} = {sha} != {fc.SAM3_CKPT_SHA256} (another checkpoint?)")
    for ds in ("voc2012", "cityscapes"):
        fc.spec(ds).split_train()
        fc.need(fc.ASSETS_DIR / f"{ds}_ref_pred" / "ref_meta.json", "reference predictions")
    rep["files"], p2 = count_files(paths, rn)
    problems += p2
    du = shutil.disk_usage(Path(paths["out_root"]))
    rep["disk_free_gb"] = du.free / 2**30
    if du.free / 2**30 < DISK_MIN_GB and not smoke:
        problems.append(f"only {du.free / 2**30:.0f} GB free at {paths['out_root']} (need >= {DISK_MIN_GB} GB)")
    fc.save_json_atomic(Path(paths["out_root"]) / "checks.json", rep)
    for k, v in rep.items():
        if k != "files":
            rn.say(f"  {k:16s} {v}")
    for k, v in rep["files"].items():
        rn.say(f"  {k:16s} {v}")
    if problems:
        raise fc.PipelineError("checks failed:\n  " + "\n  ".join(problems))
    rn.say("checks OK")


# ----------------------------------------------------------------------------------------------
# Step 2: preflight
# ----------------------------------------------------------------------------------------------
def derived_yaml(paths, out_root, name):
    """paths.yaml with another out_root (COCO files still read from the main out_root)."""
    import yaml
    d = {k: paths[k] for k in fc.PATH_KEYS}
    d.update(out_root=str(out_root), device=paths["device"], num_workers=paths["num_workers"],
             coco_root=paths.get("coco_root") or str(Path(paths["out_root"]) / "data_coco"))
    p = Path(out_root) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return p


def preflight_fp():
    return fc.fingerprint({"code": fc.CODE_VERSION, "sam3": fc.SAM3_CKPT_SHA256, "agree": AGREE_MIN, "pf": PREFLIGHT})


def step_preflight(paths, rn):
    rn.banner("STEP 2: preflight")
    pf_dir = Path(paths["out_root"]) / "preflight"
    rep_f = pf_dir / "preflight_report.json"
    old = fc.load_json(rep_f)
    if old and old.get("passed") and old.get("fingerprint") == preflight_fp():
        rn.say("preflight already passed, skip")
        return old
    rep, problems = {"fingerprint": preflight_fp(), "created": fc.now()}, []
    for ds in ("cityscapes", "voc2012"):
        f = pf_dir / f"preflight_{ds}_base.json"
        if not (fc.load_json(f) or {}).get("agreement_with_kaggle_pred"):
            rn.run(["ft_eval.py", "--mode", "preflight", "--paths", paths["_yaml"], "--dataset", ds, "--out-json", f],
                   f"preflight {ds}: SAM3 as released vs Kaggle")
        r = fc.load_json(f)
        rep[f"{ds}_base"] = r
        if r["agreement_with_kaggle_pred"] < AGREE_MIN:
            problems.append(f"{ds}: only {100 * r['agreement_with_kaggle_pred']:.2f}% pixels agree with the Kaggle "
                            f"run-1 prediction (need >= {100 * AGREE_MIN}%): wrong sam3 commit / torch / checkpoint?")
    if problems:
        fc.save_json_atomic(rep_f, dict(rep, passed=False, problems=problems))
        raise fc.PipelineError("preflight failed:\n  " + "\n  ".join(problems))
    py = derived_yaml(paths, pf_dir / "runs", "paths_preflight.yaml")
    pp = fc.load_paths(py)
    speed = {}
    for cfg in ("A2", "A1"):
        rn.run(["ft_train.py", "--paths", py, "--dataset", "cityscapes", "--cfg", cfg,
                "--fit-limit", PREFLIGHT["fit_limit"], "--holdout-limit", PREFLIGHT["holdout_limit"],
                "--epoch-cap", PREFLIGHT["epoch_cap"]], f"preflight: {PREFLIGHT['fit_limit']}-step {cfg} run on Cityscapes")
        h = fc.load_json(fc.model_file(pp, "cityscapes", cfg, "holdout_history.json"))
        e0 = h["epochs"]["0"]
        speed[cfg] = {"s_per_step": e0["train_seconds"] / max(e0["steps_this_epoch"], 1),
                      "s_per_holdout_image": e0["eval_seconds"] / max(h["holdout_images"], 1),
                      "peak_gpu_gb": e0["peak_gpu_gb_last_step"]}
    rl = pf_dir / "preflight_reload_A2.json"
    rn.run(["ft_eval.py", "--mode", "reload", "--paths", py, "--dataset", "cityscapes", "--cfg", "A2", "--out-json", rl],
           "preflight: best.pt reload check")
    rep["reload"] = fc.load_json(rl)
    if rep["reload"]["pixel_disagreement"] > RELOAD_MAX:
        problems.append(f"best.pt reloaded differs from the model in memory ({rep['reload']['pixel_disagreement']:.2e})")
    rep["speed"] = speed
    est, total = {}, 0.0
    for ds, cfg in fc.MODEL_ORDER:
        sp = fc.spec(ds)
        n_fit, n_hold = (1171, 293) if ds == "voc2012" else (2380, 595)
        cap = fc.EPOCH_CAP[ds]
        s_img = speed[cfg]["s_per_holdout_image"] * (sp.C / 19)
        h_cap = (cap * n_fit * speed[cfg]["s_per_step"] + cap * n_hold * s_img + sp.n_val * s_img * 1.3) / 3600
        est[f"{ds}_{cfg}"] = {"hours_if_cap": h_cap}
        total += h_cap
    rep["estimate_hours"] = dict(est, total_if_all_reach_cap=total)
    rn.say(f"speed: {json.dumps(speed)}")
    for k, v in est.items():
        rn.say(f"  {k:16s} <= {v['hours_if_cap']:.1f} h (to the epoch cap; early stopping usually ends sooner)")
    rn.say(f"  total <= {total:.1f} h")
    rep["passed"] = not problems
    rep["problems"] = problems
    fc.save_json_atomic(rep_f, rep)
    if problems:
        raise fc.PipelineError("preflight failed:\n  " + "\n  ".join(problems))
    shutil.rmtree(pf_dir / "runs", ignore_errors=True)       # the 30-step models are not needed any more
    rn.say("preflight OK")
    return rep


# ----------------------------------------------------------------------------------------------
# Steps 3-5: one model; step 6: summary + pack
# ----------------------------------------------------------------------------------------------
def model_done(paths, ds, cfg, fp):
    d = fc.load_json(fc.model_file(paths, ds, cfg, "done.json"))
    return d is not None and d.get("fingerprint") == fp


def step_model(paths, rn, ds, cfg, limits):
    import ft_train
    fp = ft_train.train_fingerprint(paths, ds, cfg, limits.get("fit_limit"), limits.get("holdout_limit"),
                                    limits.get("epoch_cap"))
    if model_done(paths, ds, cfg, fp):
        rn.say(f"{ds} {cfg}: done, skip")
        return
    rn.banner(f"MODEL {ds} {cfg}: train -> holdout every epoch -> early stop -> val with the best checkpoint")
    args = ["ft_train.py", "--paths", paths["_yaml"], "--dataset", ds, "--cfg", cfg]
    for k in ("fit_limit", "holdout_limit", "epoch_cap"):
        if limits.get(k):
            args += [f"--{k.replace('_', '-')}", limits[k]]
    rn.run(args, f"{ds} {cfg} train")
    val = ["ft_eval.py", "--mode", "val", "--paths", paths["_yaml"], "--dataset", ds, "--cfg", cfg]
    if limits.get("val_limit"):
        val += ["--limit", limits["val_limit"]]
    rn.run(val, f"{ds} {cfg} val")
    rep = fc.load_json(fc.model_file(paths, ds, cfg, "val_report.json"))
    fc.save_json_atomic(fc.model_file(paths, ds, cfg, "done.json"),
                        {"fingerprint": fp, "val_miou_at_tau_A": rep["miou_at_tau_A"], "tau_A": rep["tau_A"], "finished": fc.now()})
    write_summary(paths)
    pack(paths, rn)


def write_summary(paths):
    rows = []
    for ds, cfg in fc.MODEL_ORDER:
        best = fc.load_json(fc.model_file(paths, ds, cfg, "best.json"))
        if best is None:
            continue
        rep = fc.load_json(fc.model_file(paths, ds, cfg, "val_report.json")) or {}
        td = fc.load_json(fc.model_file(paths, ds, cfg, "train_done.json")) or {}
        par = fc.load_json(fc.model_file(paths, ds, cfg, "params.json")) or {}
        rows.append({"dataset": ds, "config": cfg, "trainable_params": par.get("trainable"),
                     "best_epoch": best["epoch"], "epochs_run": best["epochs_run"], "epoch_cap": best["epoch_cap"],
                     "stopped_early": best["stopped_early"], "hit_cap": best["hit_cap"], "tau_A": best["tau"],
                     "holdout_miou_tau_A": best["holdout_miou"], "holdout_miou_0.5": best["holdout_miou_0.5"],
                     "val_miou_tau_A": rep.get("miou_at_tau_A"), "val_miou_0.5": rep.get("miou_at_0.5"),
                     "train_and_holdout_hours": td.get("hours_train_and_holdout"),
                     "val_hours": (rep.get("seconds") or 0) / 3600 or None, "status": "done" if rep else "training"})
    if rows:
        f = Path(paths["out_root"]) / "sam3ft_summary.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


PACK_SUFFIXES = (".json", ".csv", ".yaml", ".log", ".txt")
PACK_NPZ = ("holdout_conf_tau.npz", "val_conf_tau.npz")


def pack(paths, rn):
    """Small files only (no .pt, no raw/pred folders, no tensorboard): the part to send back."""
    root = Path(paths["out_root"])
    out = root / "sam3ft_results_pack.zip"
    tmp = out.with_name(out.name + ".tmp")
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            parts = set(rel.parts)
            if not p.is_file() or any(x.endswith(("_val_raw", "_val_pred", "_resume", "tensorboard")) for x in parts):
                continue
            if "runs" in rel.parts[:2] or p.name.endswith((".tmp", ".zip")) or "fit_coco" in p.name:
                continue
            if p.suffix in PACK_SUFFIXES or p.name.endswith(PACK_NPZ) or (p.suffix == ".jpg" and "samples" in parts):
                z.write(p, rel.as_posix())
                n += 1
    os.replace(tmp, out)
    rn.say(f"pack: {out} ({n} files, {out.stat().st_size / 2**20:.1f} MB)")


# ----------------------------------------------------------------------------------------------
def plan(paths, smoke, models):
    import ft_prepare_data
    import ft_train
    lim = SMOKE if smoke else {}
    rows = [("0 checks", "always")]
    rows.append(("1 prepare", " ".join(f"{ds}:{'done' if ft_prepare_data.is_done(paths, ds) else 'TODO'}"
                                      for ds in ("cityscapes", "voc2012"))))
    if not smoke:
        r = fc.load_json(Path(paths["out_root"]) / "preflight" / "preflight_report.json") or {}
        rows.append(("2 preflight", "done" if r.get("passed") and r.get("fingerprint") == preflight_fp() else "TODO"))
    for ds, cfg in models:
        fp = ft_train.train_fingerprint(paths, ds, cfg, lim.get("fit_limit"), lim.get("holdout_limit"), lim.get("epoch_cap"))
        if model_done(paths, ds, cfg, fp):
            st = "done"
        elif (fc.load_json(fc.model_file(paths, ds, cfg, "train_done.json")) or {}).get("fingerprint") == fp:
            st = "trained, val TODO"
        else:
            h = fc.load_json(fc.model_file(paths, ds, cfg, "holdout_history.json"))
            st = f"training resumes after epoch {max(map(int, h['epochs']))}" if h and h["epochs"] else "TODO"
        rows.append((f"  {ds} {cfg}", st))
    rows.append(("6 summary + pack", "always"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="paths.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run of the whole chain (outputs in <out_root>_smoke)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset, e.g. cityscapes:A2 voc2012:A2 (default: all four in the planned order)")
    ap.add_argument("--skip-preflight", action="store_true", help="(debug) do not run step 2")
    args = ap.parse_args()
    paths = fc.load_paths(args.config)
    models = fc.MODEL_ORDER
    if args.models:
        want = [tuple(m.split(":")) for m in args.models]
        bad = [m for m in want if m not in fc.MODEL_ORDER]
        if bad:
            raise fc.PipelineError(f"unknown model(s) {bad}; choose from {fc.MODEL_ORDER}")
        models = [m for m in fc.MODEL_ORDER if m in want]
    if args.smoke:
        main_out = Path(paths["out_root"])
        smoke_out = main_out.with_name(main_out.name + "_smoke")
        y = derived_yaml(paths, smoke_out, "paths_smoke.yaml")
        paths = fc.load_paths(y)
    Path(paths["out_root"]).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for a, b in plan(paths, args.smoke, models):
            print(f"{a:28s} {b}")
        return
    rn = Runner(paths, Path(paths["out_root"]) / "logs" / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log")
    t0 = time.time()
    rn.banner(f"SAM3 fine-tuning (Method A) -- {'SMOKE ' if args.smoke else ''}run, models {models}")
    for a, b in plan(paths, args.smoke, models):
        rn.say(f"  {a:28s} {b}")
    step_checks(paths, rn, args.smoke)
    rn.banner("STEP 1: prepare data (GT -> COCO + RLE, train-fit)")
    rn.run(["ft_prepare_data.py", "--paths", paths["_yaml"], "--datasets", "cityscapes", "voc2012"], "prepare data")
    if not args.smoke and not args.skip_preflight:
        step_preflight(paths, rn)
    for ds, cfg in models:
        step_model(paths, rn, ds, cfg, SMOKE if args.smoke else {})
    rn.banner("STEP 6: summary")
    for r in write_summary(paths):
        rn.say("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    pack(paths, rn)
    rn.say(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Send back {Path(paths['out_root']) / 'sam3ft_results_pack.zip'}")


if __name__ == "__main__":
    fc.run_main(main)
