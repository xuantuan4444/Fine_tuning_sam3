"""Train ONE model (dataset x configuration) with the official SAM3 Trainer.

  python ft_train.py --paths paths.yaml --dataset cityscapes --cfg A2
         [--fit-limit N --holdout-limit N --epoch-cap E]      (smoke / preflight only)

Launch path = sam3/train/train.py without submitit: single_proc_run(local_rank=0, world_size=1), which sets
the torch.distributed env variables, instantiates trainer._target_ (ft_trainer.FtTrainer) and calls
trainer.run(). Killed? Run the same command again: the Trainer resumes from <ds>_<cfg>_resume/checkpoint.pt
and FtTrainer skips the epochs already in the holdout history.

Output (outputs/<ds>_<cfg>/): <ds>_<cfg>_best.pt / _best.json / _holdout_history.json / _holdout_conf_tau.npz /
_params.json / _train_log/ / _train_config.yaml / _train_done.json. The resume folder is deleted at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import ft_common as fc

sys.path.insert(0, str(fc.REPO_DIR))          # Hydra targets ft_trainer.FtTrainer / ft_model.build_ft_model


def train_fingerprint(paths, ds, cfg, fit_limit=None, holdout_limit=None, epoch_cap=None):
    import ft_prepare_data
    return fc.fingerprint({"code": fc.CODE_VERSION, "ds": ds, "cfg": cfg, "cap": epoch_cap or fc.EPOCH_CAP[ds],
                           "patience": fc.PATIENCE, "min_gain": fc.MIN_GAIN, "seed": fc.SEED,
                           "data": ft_prepare_data.prepare_fingerprint(paths, ds), "sam3": fc.SAM3_CKPT_SHA256,
                           "fit_limit": fit_limit, "holdout_limit": holdout_limit,
                           "config_yaml": fc.sha256_file(fc.CONFIG_YAML)})


def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def subset_coco(src, dst, n):
    coco = json.load(open(src))
    keep = {im["id"] for im in coco["images"][:n]}
    coco["images"] = [im for im in coco["images"] if im["id"] in keep]
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in keep]
    fc.save_json_atomic(dst, coco)


def build_config(paths, ds, cfg, fit_limit=None, holdout_limit=None, epoch_cap=None):
    import ft_model
    import ft_prepare_data
    from omegaconf import OmegaConf
    sp = fc.spec(ds)
    run_dir = fc.model_dir(paths, ds, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    ann, _, _ = ft_prepare_data.coco_paths(paths, ds)
    fc.need(ann, f"{ds} COCO file (run ft_prepare_data.py first)")
    if fit_limit:
        sub = run_dir / f"{ds}_{cfg}_fit_coco_first{fit_limit}.json"
        subset_coco(ann, sub, fit_limit)
        ann = sub
    c = OmegaConf.load(fc.CONFIG_YAML)
    c.paths.img_folder = str(sp.img_folder(paths, "train"))
    c.paths.fit_ann_file = str(ann)
    c.paths.sam3_checkpoint = paths["sam3_checkpoint"]
    c.paths.run_dir = str(run_dir)
    cpu = paths["device"] == "cpu"
    c.ft.update({"dataset": ds, "config": cfg, "paths_yaml": paths["_yaml"], "out_root": paths["out_root"],
                 "holdout_limit": holdout_limit, "epoch_cap": int(epoch_cap or fc.EPOCH_CAP[ds]),
                 "accelerator": "cpu" if cpu else "cuda", "amp": not cpu, "num_workers": paths["num_workers"],
                 "seed": fc.SEED})
    c.trainer.checkpoint.save_dir = str(run_dir / f"{ds}_{cfg}_resume")
    c.trainer.checkpoint.skip_saving_parameters = ft_model.frozen_patterns(cfg)
    c.trainer.logging.log_dir = str(run_dir / f"{ds}_{cfg}_train_log")
    c.trainer.logging.tensorboard_writer.log_dir = str(run_dir / f"{ds}_{cfg}_train_log" / "tensorboard")
    return c, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--dataset", required=True, choices=["voc2012", "cityscapes"])
    ap.add_argument("--cfg", required=True, choices=list(fc.FT_CONFIGS))
    ap.add_argument("--fit-limit", type=int, default=None)
    ap.add_argument("--holdout-limit", type=int, default=None)
    ap.add_argument("--epoch-cap", type=int, default=None)
    args = ap.parse_args()
    paths = fc.load_paths(args.paths)
    ds, cfg = args.dataset, args.cfg

    import torch
    from omegaconf import OmegaConf
    from sam3.train.utils.train_utils import register_omegaconf_resolvers
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass
    fp = train_fingerprint(paths, ds, cfg, args.fit_limit, args.holdout_limit, args.epoch_cap)
    done_f = fc.model_file(paths, ds, cfg, "train_done.json")
    done = fc.load_json(done_f)
    if done is not None:
        if done.get("fingerprint") == fp:
            fc.log(f"{ds} {cfg}: training already done ({done_f.name}), skip")
            return
        raise fc.PipelineError(f"{done_f} was made with other settings (fingerprint {done.get('fingerprint')} != {fp}). "
                               f"Move the folder {done_f.parent} away to retrain.")
    c, run_dir = build_config(paths, ds, cfg, args.fit_limit, args.holdout_limit, args.epoch_cap)
    run_fp_f = fc.model_file(paths, ds, cfg, "train_fingerprint.json")
    old = fc.load_json(run_fp_f)
    if old is not None and old["fingerprint"] != fp:
        raise fc.PipelineError(f"{run_dir} holds a partial run with other settings; move it away to start again.")
    fc.save_json_atomic(run_fp_f, {"fingerprint": fp, "created": fc.now()})
    with open(fc.model_file(paths, ds, cfg, "train_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(c, resolve=True))

    if paths["device"] == "cpu":
        # LOCAL CPU TEST ONLY: the repo's MemMeter calls CUDA memory functions unconditionally
        from sam3.train.utils import train_utils
        train_utils.MemMeter.update = lambda self, n=1, reset_peak_usage=True: None
    else:
        if not torch.cuda.is_available():
            raise fc.PipelineError("device: cuda in paths.yaml but torch.cuda.is_available() is False")

    from sam3.train.train import single_proc_run
    t0 = time.time()
    fc.log(f"{ds} {cfg}: training (cap {c.ft.epoch_cap} epochs, patience {fc.PATIENCE}, min gain {fc.MIN_GAIN}) -> {run_dir}")
    single_proc_run(local_rank=0, main_port=free_port(), cfg=c, world_size=1)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    best = fc.load_json(fc.model_file(paths, ds, cfg, "best.json"))
    hist = fc.load_json(fc.model_file(paths, ds, cfg, "holdout_history.json"))
    if not best or not best.get("finished") or not fc.model_file(paths, ds, cfg, "best.pt").exists():
        raise fc.PipelineError(f"{ds} {cfg}: trainer returned but the run is not finished (see {run_dir})")
    hours = sum(e["train_seconds"] + e["eval_seconds"] for e in hist["epochs"].values()) / 3600
    shutil.rmtree(Path(c.trainer.checkpoint.save_dir), ignore_errors=True)        # temporary resume file
    fc.save_json_atomic(done_f, {"fingerprint": fp, "best": best, "hours_train_and_holdout": hours,
                                 "this_session_hours": (time.time() - t0) / 3600, "finished": fc.now(),
                                 "torch": torch.__version__,
                                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    fc.log(f"{ds} {cfg}: done. best epoch {best['epoch']} / {best['epochs_run']} run, holdout mIoU "
           f"{best['holdout_miou']:.4f} at tau {best['tau']}; stopped_early={best['stopped_early']} hit_cap={best['hit_cap']}")


if __name__ == "__main__":
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    fc.run_main(main)
