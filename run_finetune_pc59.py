"""THE single command of the PASCAL-Context 59 job (separate from run_finetune.py, decided 18/11):

  python run_finetune_pc59.py --config paths.yaml            run everything, start to end
  python run_finetune_pc59.py --config paths.yaml --dry-run  print the plan (what is done / what will run)
  python run_finetune_pc59.py --config paths.yaml --smoke    whole chain on a few images (<out_root>_smoke/pc59)

paths.yaml needs pc59_root, sam3_checkpoint, out_root (voc_root is optional, see step 2).
Everything is written to <out_root>/pc59/ (own logs, preflight, models, summary, pack), so the outputs and the
pack of the main job (VOC + Cityscapes) are untouched; the two jobs can run one after the other in any order.

Steps (killed? run the SAME command again: finished steps are skipped, a model in training resumes):
  0 checks     python / torch / GPU, sam3 commit, sha256 of sam3.pt, PC59 files (4998 train / 5105 val), disk
  1 prepare    GT -> COCO + RLE for PC59 train-fit (3998 images, ft_prepare_data.py)
  2 preflight  environment: SAM3 as released vs the Kaggle run-1 prediction. There is no Kaggle run on PC59, so
               the passed preflight of the main job is reused if it is in <out_root>/preflight/; otherwise the
               VOC check (50 images, >= 98% pixels) runs if voc_root is in paths.yaml; otherwise a WARNING
               (commit and sha256 are still checked in step 0).
               Then 30-step A2 and A1 runs ON PC59 (59 queries per image: speed, GPU memory, holdout speed),
               best.pt reload check, hours estimate.
  3 base       SAM3 as released on val (5105 images) at 0.5 = BL0 of PC59, to reproduce the SAM3 paper
               (ft_eval.py --mode base_val; decided 18/11, Q-P1). The same pass gives the mIoU at every tau
               (curve only, no tau chosen on val). Outputs in pc59_base/.
  4 A2 PC59    5 A1 PC59: train on fit, holdout after every epoch, early stop (cap 4 epochs), best.pt and tau_A
               chosen on holdout (same protocol as VOC/City), then val (5105 images) ONCE, then pack
  6 summary    sam3ft_pc59_summary.csv (base + A2 + A1) + sam3ft_pc59_results_pack.zip (small files to send back)
Val outputs (Q-P3, decided 18/11): no val_raw on PC59; val_pred/ (PNG) is kept for the before/after figures and a
small per-image confusion file (val_conf/<id>.npz) lets an interrupted val resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import ft_common as fc
import run_finetune as rf

DS = "pc59"
DISK_MIN_GB = 10                  # no val raw (Q-P3): outputs ~0.5 GB (best.pt, 3 x val_pred ~30 MB,
                                  # 3 x val_conf ~35 MB, COCO json) + the resume checkpoint while training
SUMMARY_CSV = "sam3ft_pc59_summary.csv"
PACK_ZIP = "sam3ft_pc59_results_pack.zip"
BASE = fc.BASE_CFG


def pc59_out(main_out, smoke):
    base = Path(main_out)
    if smoke:
        base = base.with_name(base.name + "_smoke")
    return base / DS


def write_summary_pc59(paths):
    """rf.write_summary (A2, A1 rows) + a first row for SAM3 as released (BL0 at 0.5)."""
    rows = rf.write_summary(paths, fc.PC59_MODEL_ORDER, SUMMARY_CSV)
    rep = fc.load_json(fc.model_file(paths, DS, BASE, "val_report.json"))
    if rep is None:
        return rows
    keys = list(rows[0]) if rows else ["dataset", "config", "trainable_params", "best_epoch", "epochs_run", "epoch_cap",
                                       "stopped_early", "hit_cap", "tau_A", "holdout_miou_tau_A", "holdout_miou_0.5",
                                       "val_miou_tau_A", "val_miou_0.5", "train_and_holdout_hours", "val_hours", "status"]
    base_row = dict.fromkeys(keys)
    base_row.update({"dataset": DS, "config": "base (SAM3 as released)", "trainable_params": 0,
                     "val_miou_0.5": rep["miou_at_0.5"], "val_hours": rep["seconds"] / 3600, "status": "done"})
    rows = [base_row] + rows
    with open(Path(paths["out_root"]) / SUMMARY_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return rows


def finish_pc59(paths, rn):
    write_summary_pc59(paths)
    rf.pack(paths, rn, PACK_ZIP, skip_top=())


def base_fp(paths, limit):
    return fc.fingerprint({"code": fc.CODE_VERSION, "ds": DS, "sam3": fc.SAM3_CKPT_SHA256, "thr": fc.BASE_THR,
                           "store": [fc.STORE_THR, fc.STORE_TOPK], "taus": fc.TAU_GRID, "limit": limit,
                           "prompts": fc.spec(DS).prompts})


def base_done(paths, limit):
    d = fc.load_json(fc.model_file(paths, DS, BASE, "done.json"))
    return d is not None and d.get("fingerprint") == base_fp(paths, limit)


def step_base(paths, rn, limit=None):
    """SAM3 as released on PC59 val at 0.5 (BL0). Killed? the same command resumes from the per-image files."""
    if base_done(paths, limit):
        rn.say(f"{DS} base: done, skip")
        return
    rn.banner(f"STEP 3: SAM3 as released on {DS} val at {fc.BASE_THR} (BL0, reproduces the SAM3 paper setting)")
    args = ["ft_eval.py", "--mode", "base_val", "--paths", paths["_yaml"], "--dataset", DS]
    if limit:
        args += ["--limit", limit]
    rn.run(args, f"{DS} SAM3 base val")
    rep = fc.load_json(fc.model_file(paths, DS, BASE, "val_report.json"))
    fc.save_json_atomic(fc.model_file(paths, DS, BASE, "done.json"),
                        {"fingerprint": base_fp(paths, limit), "val_miou_at_0.5": rep["miou_at_0.5"], "finished": fc.now()})
    finish_pc59(paths, rn)


def plan_pc59(paths, smoke, models):
    rows = rf.plan(paths, smoke, models, [DS], preflight_done(paths))
    limit = rf.SMOKE["val_limit"] if smoke else None
    at = len(rows) - len(models) - 1                       # before the model rows
    rows.insert(at, (f"  {DS} base (SAM3, val, 0.5)", "done" if base_done(paths, limit) else "TODO"))
    return rows


def preflight_fp():
    return fc.fingerprint({"code": fc.CODE_VERSION, "sam3": fc.SAM3_CKPT_SHA256, "agree": rf.AGREE_MIN,
                           "pf": rf.PREFLIGHT, "ds": DS})


def preflight_done(paths):
    r = fc.load_json(Path(paths["out_root"]) / "preflight" / "preflight_report.json") or {}
    return bool(r.get("passed") and r.get("fingerprint") == preflight_fp())


def voc_available(paths):
    if not paths.get("voc_root"):
        return False
    try:
        fc.spec("voc2012").list_ids(paths, "train")
        return True
    except fc.PipelineError:
        return False


def step_environment(paths, main_out, rn, pf_dir):
    """SAM3 as released here == the Kaggle run-1 prediction (no Kaggle run exists on PC59)."""
    main_rep = fc.load_json(Path(main_out) / "preflight" / "preflight_report.json") or {}
    if main_rep.get("passed") and main_rep.get("fingerprint") == rf.preflight_fp():
        rn.say("environment: the main job's preflight passed in this out_root, reused")
        return {"source": "main preflight " + str(main_rep.get("created")),
                **{ds: main_rep[f"{ds}_base"]["agreement_with_kaggle_pred"] for ds in fc.MAIN_DATASETS}}, []
    if voc_available(paths):
        f = pf_dir / "preflight_voc2012_base.json"
        if not (fc.load_json(f) or {}).get("agreement_with_kaggle_pred"):
            rn.run(["ft_eval.py", "--mode", "preflight", "--paths", paths["_yaml"], "--dataset", "voc2012",
                    "--out-json", f], "preflight voc2012: SAM3 as released vs Kaggle (environment check)")
        r = fc.load_json(f)
        problems = []
        if r["agreement_with_kaggle_pred"] < rf.AGREE_MIN:
            problems.append(f"voc2012: only {100 * r['agreement_with_kaggle_pred']:.2f}% pixels agree with the Kaggle "
                            f"run-1 prediction (need >= {100 * rf.AGREE_MIN}%): wrong sam3 commit / torch / checkpoint?")
        return {"source": "voc2012 check", "voc2012": r["agreement_with_kaggle_pred"], "voc2012_base": r}, problems
    rn.say("WARNING: environment NOT compared with Kaggle (no passed main preflight in out_root and no voc_root in "
           "paths.yaml); the sam3 commit and the sha256 of sam3.pt were checked in step 0")
    return {"source": "not checked"}, []


def step_preflight(paths, main_out, rn):
    rn.banner("STEP 2: preflight (PC59)")
    pf_dir = Path(paths["out_root"]) / "preflight"
    rep_f = pf_dir / "preflight_report.json"
    old = fc.load_json(rep_f)
    if old and old.get("passed") and old.get("fingerprint") == preflight_fp():
        rn.say("preflight already passed, skip")
        return old
    rep = {"fingerprint": preflight_fp(), "created": fc.now()}
    rep["environment"], problems = step_environment(paths, main_out, rn, pf_dir)
    if problems:
        fc.save_json_atomic(rep_f, dict(rep, passed=False, problems=problems))
        raise fc.PipelineError("preflight failed:\n  " + "\n  ".join(problems))
    py = rf.derived_yaml(paths, pf_dir / "runs", "paths_preflight.yaml")
    pp = fc.load_paths(py, [DS])
    speed = {}
    for cfg in ("A2", "A1"):
        rn.run(["ft_train.py", "--paths", py, "--dataset", DS, "--cfg", cfg,
                "--fit-limit", rf.PREFLIGHT["fit_limit"], "--holdout-limit", rf.PREFLIGHT["holdout_limit"],
                "--epoch-cap", rf.PREFLIGHT["epoch_cap"]], f"preflight: {rf.PREFLIGHT['fit_limit']}-step {cfg} run on PC59")
        h = fc.load_json(fc.model_file(pp, DS, cfg, "holdout_history.json"))
        e0 = h["epochs"]["0"]
        speed[cfg] = {"s_per_step": e0["train_seconds"] / max(e0["steps_this_epoch"], 1),
                      "s_per_holdout_image": e0["eval_seconds"] / max(h["holdout_images"], 1),
                      "peak_gpu_gb": e0["peak_gpu_gb_last_step"]}
    rl = pf_dir / "preflight_reload_A2.json"
    rn.run(["ft_eval.py", "--mode", "reload", "--paths", py, "--dataset", DS, "--cfg", "A2", "--out-json", rl],
           "preflight: best.pt reload check (PC59)")
    rep["reload"] = fc.load_json(rl)
    if rep["reload"]["pixel_disagreement"] > rf.RELOAD_MAX:
        problems.append(f"best.pt reloaded differs from the model in memory ({rep['reload']['pixel_disagreement']:.2e})")
    rep["speed"] = speed
    sp = fc.spec(DS)
    n_fit, n_hold = (len(x) for x in sp.split_train())
    est, total = {}, 0.0
    for ds, cfg in fc.PC59_MODEL_ORDER:
        cap = fc.EPOCH_CAP[ds]
        s_img = speed[cfg]["s_per_holdout_image"]
        h_cap = (cap * n_fit * speed[cfg]["s_per_step"] + cap * n_hold * s_img + sp.n_val * s_img * 1.3) / 3600
        est[f"{ds}_{cfg}"] = {"hours_if_cap": h_cap, "hours_val": sp.n_val * s_img * 1.3 / 3600}
        total += h_cap
    s_base = speed["A2"]["s_per_holdout_image"]            # same inference path as the holdout eval
    est[f"{DS}_{BASE}"] = {"hours_val": sp.n_val * s_base * 1.3 / 3600}
    total += est[f"{DS}_{BASE}"]["hours_val"]
    rep["estimate_hours"] = dict(est, total_if_all_reach_cap=total)
    rn.say(f"speed: {json.dumps(speed)}")
    for k, v in est.items():
        if "hours_if_cap" in v:
            rn.say(f"  {k:16s} <= {v['hours_if_cap']:.1f} h (to the epoch cap, of which val {v['hours_val']:.1f} h; "
                   f"early stopping usually ends sooner)")
        else:
            rn.say(f"  {k:16s} ~  {v['hours_val']:.1f} h (SAM3 as released on val)")
    rn.say(f"  total <= {total:.1f} h")
    rep["passed"] = not problems
    rep["problems"] = problems
    fc.save_json_atomic(rep_f, rep)
    if problems:
        raise fc.PipelineError("preflight failed:\n  " + "\n  ".join(problems))
    shutil.rmtree(pf_dir / "runs", ignore_errors=True)       # the 30-step models are not needed any more
    rn.say("preflight OK")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="paths.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run of the whole chain (outputs in <out_root>_smoke/pc59)")
    ap.add_argument("--models", nargs="+", default=None, help="subset, e.g. pc59:A2 (default: A2 then A1)")
    ap.add_argument("--skip-preflight", action="store_true", help="(debug) do not run step 2")
    args = ap.parse_args()
    main_paths = fc.load_paths(args.config, [DS])
    models = fc.PC59_MODEL_ORDER
    if args.models:
        want = [tuple(m.split(":")) for m in args.models]
        bad = [m for m in want if m not in fc.PC59_MODEL_ORDER]
        if bad:
            raise fc.PipelineError(f"unknown model(s) {bad}; choose from {fc.PC59_MODEL_ORDER}")
        models = [m for m in fc.PC59_MODEL_ORDER if m in want]
    out = pc59_out(main_paths["out_root"], args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    y = rf.derived_yaml(main_paths, out, "paths_pc59.yaml", coco_root=out / "data_coco")
    paths = fc.load_paths(y, [DS])
    if args.dry_run:
        for a, b in plan_pc59(paths, args.smoke, models):
            print(f"{a:28s} {b}")
        return
    rn = rf.Runner(paths, out / "logs" / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log")
    t0 = time.time()
    rn.banner(f"SAM3 fine-tuning (Method A) on PASCAL-Context 59 -- {'SMOKE ' if args.smoke else ''}run, "
              f"models {models}, outputs in {out}")
    for a, b in plan_pc59(paths, args.smoke, models):
        rn.say(f"  {a:28s} {b}")
    rf.step_checks(paths, rn, args.smoke, [DS], DISK_MIN_GB)
    rn.banner("STEP 1: prepare data (GT -> COCO + RLE, PC59 train-fit)")
    rn.run(["ft_prepare_data.py", "--paths", paths["_yaml"], "--datasets", DS], "prepare data (PC59)")
    if not args.smoke and not args.skip_preflight:
        step_preflight(paths, main_paths["out_root"], rn)
    step_base(paths, rn, rf.SMOKE["val_limit"] if args.smoke else None)
    for ds, cfg in models:
        rf.step_model(paths, rn, ds, cfg, rf.SMOKE if args.smoke else {}, finish=finish_pc59)
    rn.banner("STEP 6: summary (PC59)")
    for r in write_summary_pc59(paths):
        rn.say("  " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    rf.pack(paths, rn, PACK_ZIP, skip_top=())
    rn.say(f"ALL DONE in {(time.time() - t0) / 3600:.2f} h. Send back {out / PACK_ZIP}")


if __name__ == "__main__":
    fc.run_main(main)
