"""FtTrainer = the official sam3.train.trainer.Trainer with ONE method replaced: run_val.

Training (epoch loop, forward, losses, matcher, AdamW, bf16 AMP, gradient clipping, lr schedule, saving /
loading the resume checkpoint) is the repo code, untouched. The Trainer calls run_val after every epoch
(val_epoch_freq 1, skip_first_val False; the last epoch's val runs after the loop). Ours:
  1. evaluates the model being trained on the HOLDOUT images with the val inference of ft_eval.py
     (mIoU at every tau of the grid; epoch score = mIoU at the tau chosen by the tau* rule);
  2. if the score is the best so far: overwrites <ds>_<cfg>_best.pt (trained weights only),
     <ds>_<cfg>_best.json and <ds>_<cfg>_holdout_conf_tau.npz;
  3. appends the epoch to <ds>_<cfg>_holdout_history.json (also used to skip epochs already evaluated
     when a killed job is resumed);
  4. early stopping: after PATIENCE epochs without a gain > MIN_GAIN it sets max_epochs = epoch + 1,
     so the repo's own loop ends. The lr at step s depends only on s (apart from the 20 cooldown steps
     before the cap), so stopping at epoch e gives the same model as training to the cap and taking e.
"""
from __future__ import annotations

import gc
import logging
import time

import os

import numpy as np
import sam3.train.trainer as repo_trainer
import torch
from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped

import ft_common as fc
import ft_eval
import ft_model

# The repo's Trainer.__init__ logs EVERY environment variable (train_utils.log_env_variables) into
# train_log/log.txt and stdout, i.e. into the logs we pack and send back. The server environment may hold
# tokens / passwords, so only the variables useful for debugging a run are logged.
_ENV_PREFIXES = ("CUDA", "NVIDIA", "TORCH", "NCCL", "PYTORCH", "SLURM_JOB", "SLURM_GPUS", "SLURM_CPUS", "OMP_",
                 "MKL_", "HYDRA", "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "HOSTNAME")
_ENV_SECRET = ("KEY", "TOKEN", "SECRET", "PASS", "CREDENTIAL", "AUTH")


def _log_env_variables_filtered():
    keep = sorted(k for k in os.environ if k.startswith(_ENV_PREFIXES) and not any(s in k.upper() for s in _ENV_SECRET))
    logging.info("Logging ENV_VARIABLES (filtered by ft_trainer: %d of %d shown)", len(keep), len(os.environ))
    logging.info("".join(f"{k}={os.environ[k]}\n" for k in keep))


repo_trainer.log_env_variables = _log_env_variables_filtered


class FtTrainer(Trainer):
    def __init__(self, *, ft, **kwargs):
        self.ft = dict(ft)
        self._t_mark = time.time()
        self._steps_mark = 0
        super().__init__(**kwargs)
        self.ds, self.cfg = self.ft["dataset"], self.ft["config"]
        self.paths = fc.load_paths(self.ft["paths_yaml"], [self.ft["dataset"]])
        self.paths["out_root"] = self.ft["out_root"]
        self.sp = fc.spec(self.ds)
        _, hold = self.sp.split_train()
        self.holdout_ids = hold[: self.ft["holdout_limit"]] if self.ft.get("holdout_limit") else hold
        self.f_hist = fc.model_file(self.paths, self.ds, self.cfg, "holdout_history.json")
        self.cap = int(self.max_epochs)
        hist = self._history()
        if hist["stopped_at"] is not None:                 # resumed after early stopping was decided
            self.max_epochs = int(hist["stopped_at"]) + 1
            logging.info(f"[FtTrainer] early stop already decided at epoch {hist['stopped_at']}")
        self._steps_mark = int(self.steps["train"])
        rep = getattr(unwrap_ddp_if_wrapped(self.model), "ft_report", None)
        if rep is not None and self.distributed_rank == 0:
            fc.save_json_atomic(fc.model_file(self.paths, self.ds, self.cfg, "params.json"), rep)

    # ------------------------------------------------------------------------------------------
    def _history(self):
        h = fc.load_json(self.f_hist)
        if h is None:
            h = {"dataset": self.ds, "config": self.cfg, "epoch_cap": self.cap, "patience": fc.PATIENCE,
                 "min_gain": fc.MIN_GAIN, "taus": fc.TAU_GRID, "holdout_images": len(self.holdout_ids),
                 "epochs": {}, "best_epoch": None, "best_score": None, "ref_score": None, "wait": 0,
                 "stopped_at": None}
        return h

    def run_val(self):
        epoch = int(self.epoch)
        hist = self._history()
        if str(epoch) in hist["epochs"]:
            logging.info(f"[FtTrainer] holdout of epoch {epoch} already evaluated, skip")
            return
        steps_now = int(self.steps["train"])
        train_sec = time.time() - self._t_mark
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        model = unwrap_ddp_if_wrapped(self.model)
        was_training = model.training
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        infer = ft_eval.Sam3Infer(model, self.device, self.sp.prompts)
        res = ft_eval.evaluate(infer, self.sp, self.paths, "train", self.holdout_ids,
                               f"{self.ds} {self.cfg} holdout epoch {epoch}")
        s = ft_eval.summarize(res["conf_tau"], self.sp)
        eval_sec = time.time() - t0
        del infer
        model.train(was_training)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        score = s["miou"]
        rec = {"score": score, "tau": s["tau"], "tau_index": s["tau_index"], "miou_0.5": s["miou_0.5"],
               "miou_by_tau": s["miou_by_tau"], "steps_total": steps_now,
               "steps_this_epoch": steps_now - self._steps_mark, "train_seconds": train_sec,
               "eval_seconds": eval_sec, "peak_gpu_gb_last_step": peak, "time": fc.now()}
        is_best = hist["best_score"] is None or score > hist["best_score"]      # ties -> earlier epoch
        if is_best:
            meta = {"dataset": self.ds, "config": self.cfg, "epoch": epoch, "tau": s["tau"],
                    "tau_index": s["tau_index"], "holdout_miou": score, "steps": steps_now,
                    "sam3_sha256": fc.SAM3_CKPT_SHA256, "code_version": fc.CODE_VERSION}
            ft_model.save_best(model, fc.model_file(self.paths, self.ds, self.cfg, "best.pt"), meta)
            fc.save_npz_atomic(fc.model_file(self.paths, self.ds, self.cfg, "holdout_conf_tau.npz"),
                               ids=np.array(self.holdout_ids), taus=fc.TAUS, conf=res["conf_tau"].astype(np.int32, copy=False),
                               label_names=np.array(self.sp.names), epoch=np.int32(epoch))
            hist["best_epoch"], hist["best_score"] = epoch, score
        if hist["ref_score"] is None or score > hist["ref_score"] + fc.MIN_GAIN:
            hist["ref_score"], hist["wait"] = score, 0
        else:
            hist["wait"] += 1
        rec["is_best"], rec["wait"] = is_best, hist["wait"]
        hist["epochs"][str(epoch)] = rec
        stop = hist["wait"] >= fc.PATIENCE and epoch + 1 < self.max_epochs
        if stop:
            hist["stopped_at"] = epoch
            self.max_epochs = epoch + 1                    # the repo loop ends after this epoch
        last = epoch + 1 >= self.max_epochs
        best = {"dataset": self.ds, "config": self.cfg, "epoch": hist["best_epoch"],
                "tau": hist["epochs"][str(hist["best_epoch"])]["tau"],
                "tau_index": hist["epochs"][str(hist["best_epoch"])]["tau_index"],
                "holdout_miou": hist["best_score"],
                "holdout_miou_0.5": hist["epochs"][str(hist["best_epoch"])]["miou_0.5"],
                "epochs_run": len(hist["epochs"]), "epoch_cap": self.cap,
                "stopped_early": hist["stopped_at"] is not None,
                # reached the cap while the last epoch still gained > MIN_GAIN: more epochs might help
                "hit_cap": bool(last and hist["stopped_at"] is None and hist["wait"] == 0),
                "finished": bool(last)}
        if self.distributed_rank == 0:
            fc.save_json_atomic(self.f_hist, hist)
            fc.save_json_atomic(fc.model_file(self.paths, self.ds, self.cfg, "best.json"), best)
        fc.log(f"{self.ds} {self.cfg} epoch {epoch}: holdout mIoU {score:.4f} at tau {s['tau']} "
               f"(0.5: {s['miou_0.5']:.4f}); best epoch {hist['best_epoch']} ({hist['best_score']:.4f}); "
               f"wait {hist['wait']}/{fc.PATIENCE}{'; EARLY STOP' if stop else ''}; "
               f"train {train_sec / 60:.1f} min, holdout {eval_sec / 60:.1f} min")
        self._t_mark, self._steps_mark = time.time(), steps_now
