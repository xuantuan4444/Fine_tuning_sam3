"""Make (or re-check) assets/<ds>_split_train.json: the fixed train-fit / train-holdout split.

  python tools/make_split.py --paths paths.yaml --dataset pc59                 write assets/pc59_split_train.json
  python tools/make_split.py --paths paths.yaml --dataset cityscapes --check   recompute, compare with the asset

The algorithm is make_split of the Method-B raw notebooks (run_on_kaggle/sam3_raw_*_train.ipynb), copied
line by line: among SPLIT_TRIES random permutations (seed SPLIT_SEED) keep the one whose holdout share of every
class is closest to HOLDOUT_FRAC, asking each class for >= min(30, 20% of its images) holdout images.
Class presence = at least one GT pixel of the class in the evaluation label space (ft_common.load_gt).
Ids in sorted order (= list-file order for VOC and PC59, = city/file order for Cityscapes, as the notebooks).
The VOC and Cityscapes assets were copied from the Kaggle runs; --check reproduces them exactly (checked 18/11).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ft_common as fc  # noqa: E402

HOLDOUT_FRAC = 0.2
SPLIT_SEED = 0
SPLIT_TRIES = 500


def make_split(ids, counts, sp):
    """= make_split of the Method-B notebooks (EVAL_IDX, NAMES of the dataset)."""
    present = counts[:, sp.eval_idx] > 0
    n_img = present.sum(0)
    target = np.minimum(np.ceil(HOLDOUT_FRAC * n_img), 30)
    n_hold = int(round(HOLDOUT_FRAC * len(ids)))
    rng = np.random.default_rng(SPLIT_SEED)
    best = None
    for t in range(SPLIT_TRIES):
        hold = rng.permutation(len(ids))[:n_hold]
        n_c = present[hold].sum(0)
        frac = np.where(n_img > 0, n_c / np.maximum(n_img, 1), HOLDOUT_FRAC)
        score = (np.clip(target - n_c, 0, None).sum(), np.abs(frac - HOLDOUT_FRAC).max())
        if best is None or score < best[0]:
            best = (score, t, np.sort(hold))
    (short, dev), t_best, hold = best
    hold_set = set(hold.tolist())
    return {"seed": SPLIT_SEED, "try": int(t_best), "holdout_frac": HOLDOUT_FRAC,
            "max_share_deviation": float(dev), "holdout_shortfall": int(short),
            "fit": [ids[i] for i in range(len(ids)) if i not in hold_set],
            "holdout": [ids[i] for i in hold],
            "per_label": {sp.names[c]: {"n_images": int(n_img[j]), "n_holdout": int(present[hold, j].sum())}
                          for j, c in enumerate(sp.eval_idx)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--dataset", required=True, choices=list(fc.DATASETS))
    ap.add_argument("--check", action="store_true", help="compare with the existing asset instead of writing it")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = fc.load_paths(args.paths, [args.dataset])
    sp = fc.spec(args.dataset)
    ids = sp.list_ids(paths, "train")

    def count(i):
        gt = sp.load_gt(paths, "train", i)
        return np.bincount(gt[gt < sp.K].ravel(), minlength=sp.K)[: sp.K]

    from tqdm import tqdm
    with ThreadPoolExecutor(args.workers) as ex:
        counts = np.stack(list(tqdm(ex.map(count, ids), total=len(ids), desc=f"{sp.name} GT pixel counts")))
    s = make_split(ids, counts, sp)
    out = fc.ASSETS_DIR / f"{sp.name}_split_train.json"
    fc.log(f"{sp.name}: try {s['try']}, fit {len(s['fit'])}, holdout {len(s['holdout'])}, "
           f"max share deviation {s['max_share_deviation']:.5f}, shortfall {s['holdout_shortfall']}")
    if args.check:
        old = fc.load_json(out)
        if old is None:
            raise fc.PipelineError(f"{out} does not exist")
        same = {k: old.get(k) == s[k] for k in ("try", "fit", "holdout", "per_label", "holdout_shortfall")}
        fc.log(f"compare with {out.name}: {same}")
        if not all(same.values()):
            raise fc.PipelineError("the recomputed split differs from the asset")
        fc.log("identical")
        return
    if out.exists():
        raise fc.PipelineError(f"{out} exists; the split is fixed once made (delete it by hand to remake it)")
    s["note"] = ("made by tools/make_split.py (= make_split of the Method-B raw notebooks); "
                 "class presence = GT pixels in the eval label space")
    fc.save_json_atomic(out, s)
    fc.log(f"wrote {out}")


if __name__ == "__main__":
    fc.run_main(main)
