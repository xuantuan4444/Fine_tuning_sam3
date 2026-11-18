"""Shared definitions for Method A (fine-tuning SAM3 on VOC 2012 / Cityscapes, and PASCAL-Context 59).

Everything that is specific to a dataset (classes, prompts, label space, file layout) lives in
DATASET SPEC below. Every other file imports it from here, so train / holdout / val all use the
same label space and the same prompts as the frozen-SAM3 runs of Method B (run_on_kaggle/).

Label space for evaluation (identical to Method B):
  VOC : 0 = background (also what an un-predicted pixel becomes), 1..20 = classes, 255 = void;
        mIoU over the 21 labels.
  City: 0 = no prediction (counts as a miss), 1..19 = trainId + 1, 255 = ignore; mIoU over 19 classes.
  PC59: 0 = no prediction (counts as a miss), 1..59 = classes in alphabetical order (= the PNG value of
        SegmentationClassContext), PNG 0 (background / the ~400 other Context classes) -> 255 = ignore;
        mIoU over 59 classes (= mmsegmentation PascalContextDataset59, reduce_zero_label).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = REPO_DIR / "assets"
CONFIG_YAML = REPO_DIR / "configs" / "ft_sam3_seg.yaml"

# ----------------------------------------------------------------------------------------------
# Fixed settings (same values as Method B; changing any of them changes the fingerprint)
# ----------------------------------------------------------------------------------------------
SAM3_COMMIT = "f1647e8d0c1d280991f4d1b7d974c2704537fe66"
SAM3_CKPT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CODE_VERSION = "ft-1.0"                 # bump when a change alters any output

PRECISION = "bf16"                      # autocast dtype on GPU (as the Kaggle runs)
TAU_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TAUS = np.array(TAU_GRID, np.float64)
NT = len(TAUS)
NB = NT + 1                             # score bin b = number of grid values strictly below the score
BASE_THR = 0.5                          # Sam3Processor default (BL0)
J_BASE = int(np.argmin(np.abs(TAUS - BASE_THR)))
TAU_TIE = 0.001                         # taus within 0.001 mIoU of the best are tied -> closest to 0.5
STORE_THR = 0.05                        # raw keeps queries with score > STORE_THR ...
STORE_TOPK = 30                         # ... at most STORE_TOPK per prompt (as the Kaggle raw)
RING_RADIUS = 2                         # context ring for raw features (backbone cells)
MASK_CHUNK = 32                         # masks upsampled at a time

# Training schedule (decided 17/11): early stopping on holdout mIoU
EPOCH_CAP = {"voc2012": 12, "cityscapes": 6,   # ~14k steps each (VOC 12 x 1171, City 6 x 2380)
             "pc59": 4}                         # PC59 4 x 3998 = 16k steps (3 epochs would leave early
                                                # stopping no room: it needs 1 + PATIENCE epochs)
PATIENCE = 2                            # stop after 2 epochs without a significant gain
MIN_GAIN = 0.001                        # significant gain = +0.1 mIoU point on holdout
SEED = 0

# Label conversion for training (decided 17/11)
VOID_FILL_PX = 3                        # VOC void <= 3 px from a labelled pixel takes its label
CITY_MIN_PIECE = 64                     # City pole/light/sign pieces < 64 px join the nearest piece
PC59_MIN_PIECE = 64                     # PC59 "thing" pieces < 64 px join the nearest piece (same rule)
MAX_OBJ_PER_QUERY = 200                 # repo filter FilterFindQueriesWithTooManyOut (max_ann_per_img)

# Order of the models in the single job = priority order (decided 17/11: A2 first, A1 last)
MODEL_ORDER = [("cityscapes", "A2"), ("voc2012", "A2"), ("voc2012", "A1"), ("cityscapes", "A1")]
FT_CONFIGS = ("A2", "A1")
MAIN_DATASETS = ("cityscapes", "voc2012")             # run_finetune.py
PC59_MODEL_ORDER = [("pc59", "A2"), ("pc59", "A1")]   # run_finetune_pc59.py (separate job, decided 18/11)
DATASETS = ("voc2012", "cityscapes", "pc59")
# val outputs (Q-P3, decided 18/11): VOC/City keep the per-image raw .npz (run-1 format, error analysis, resume);
# PC59 keeps no raw (16-21 GB per model, not needed: val only reports the fine-tune result) but a small per-image
# confusion file <ds>_<cfg>_val_conf/<id>.npz instead, so an interrupted val resumes. val_pred/ is kept everywhere.
VAL_SAVE_RAW = {"voc2012": True, "cityscapes": True, "pc59": False}
BASE_CFG = "base"                       # SAM3 as released on val at 0.5 (BL0), PC59 job only (Q-P1, decided 18/11)

# ----------------------------------------------------------------------------------------------
# DATASET SPEC
# ----------------------------------------------------------------------------------------------
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
# VOC prompts are split into words (finding of Method B, 09/11): +3.3 mIoU on val
VOC_PROMPTS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa",
               "train", "tv monitor"]
CITY_CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
                "motorcycle", "bicycle"]
# Cityscapes labelId -> trainId (official table, 19 evaluated classes; everything else -> 255)
CITY_ID2TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
CITY_LUT = np.full(256, 255, np.uint8)
for _k, _v in CITY_ID2TRAIN.items():
    CITY_LUT[_k] = _v
# labelIds of the classes that have instance ids in *_gtFine_instanceIds.png (value = labelId*1000 + k;
# a "group" region keeps value = labelId)
CITY_INSTANCE_LABELIDS = {24, 25, 26, 27, 28, 31, 32, 33}   # person rider car truck bus train motorcycle bicycle
CITY_THING_NO_INSTANCE_LABELIDS = {17, 19, 20}               # pole, traffic light, traffic sign
# all other evaluated classes are regions ("stuff"): one mask per class per image

# PASCAL-Context 59 (VOC2010 images). ABSOLUTE RULE: alphabetical order = mmsegmentation
# PascalContextDataset59 = value v of SegmentationClassContext/<id>.png is PC59_CLASSES[v - 1].
# Never derive it from 59_labels.txt (that file lists the 20 VOC classes first). Checked 18/11 on 400
# VOC2010 images: each of the 20 VOC classes of SegmentationClass falls on its alphabetical index.
PC59_CLASSES = ["aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book", "bottle",
                "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth", "computer", "cow", "cup",
                "curtain", "dog", "door", "fence", "floor", "flower", "food", "grass", "ground", "horse",
                "keyboard", "light", "motorbike", "mountain", "mouse", "person", "plate", "platform",
                "pottedplant", "road", "rock", "sheep", "shelves", "sidewalk", "sign", "sky", "snow", "sofa",
                "table", "track", "train", "tree", "truck", "tvmonitor", "wall", "water", "window", "wood"]
# prompts: class names; the two glued VOC names are split into words as for VOC (finding of Method B)
PC59_PROMPTS = [{"pottedplant": "potted plant", "tvmonitor": "tv monitor"}.get(c, c) for c in PC59_CLASSES]
# PC59 has no instance ids. "Things" (countable objects: the 20 VOC classes + bag bed bench book computer cup
# keyboard mouse plate sign truck) -> one object per 8-connected piece (pieces < PC59_MIN_PIECE px join the
# nearest piece), as City pole / light / sign. Every other class is a region ("stuff"): one mask per class.
PC59_THINGS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "table", "dog",
               "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
               "bag", "bed", "bench", "book", "computer", "cup", "keyboard", "mouse", "plate", "sign", "truck"]
assert len(PC59_CLASSES) == 59 and PC59_CLASSES == sorted(PC59_CLASSES) and set(PC59_THINGS) <= set(PC59_CLASSES)


class DatasetSpec:
    def __init__(self, name):
        self.name = name
        if name == "voc2012":
            self.classes, self.prompts = VOC_CLASSES, VOC_PROMPTS
            self.K, self.zero_name = 21, "background"
            self.eval_idx = list(range(0, 21))
            self.n_train, self.n_val = 1464, 1449
        elif name == "cityscapes":
            self.classes, self.prompts = CITY_CLASSES, list(CITY_CLASSES)
            self.K, self.zero_name = 20, "no_prediction"
            self.eval_idx = list(range(1, 20))
            self.n_train, self.n_val = 2975, 500
        elif name == "pc59":
            self.classes, self.prompts = PC59_CLASSES, PC59_PROMPTS
            self.K, self.zero_name = 60, "no_prediction"
            self.eval_idx = list(range(1, 60))
            self.n_train, self.n_val = 4998, 5105
        else:
            raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
        self.C = len(self.classes)
        self.names = [self.zero_name] + self.classes
        assert len(self.prompts) == self.C and self.K == self.C + 1

    # --- file layout (roots come from paths.yaml) ---
    def img_folder(self, paths, split):
        """Folder the COCO file_name is relative to."""
        if self.name == "voc2012":
            return Path(paths["voc_root"]) / "JPEGImages"
        if self.name == "pc59":
            return Path(paths["pc59_root"]) / "JPEGImages"
        return Path(paths["cityscapes_leftImg8bit"]) / split

    def rel_image(self, img_id):
        if self.name in ("voc2012", "pc59"):
            return f"{img_id}.jpg"
        return f"{img_id.split('_')[0]}/{img_id}_leftImg8bit.png"

    def image_path(self, paths, split, img_id):
        return self.img_folder(paths, split) / self.rel_image(img_id)

    def gt_paths(self, paths, split, img_id):
        if self.name == "voc2012":
            r = Path(paths["voc_root"])
            return {"cls": r / "SegmentationClass" / f"{img_id}.png", "obj": r / "SegmentationObject" / f"{img_id}.png"}
        if self.name == "pc59":
            return {"cls": Path(paths["pc59_root"]) / "SegmentationClassContext" / f"{img_id}.png"}
        d = Path(paths["cityscapes_gtFine"]) / split / img_id.split("_")[0]
        return {"label": d / f"{img_id}_gtFine_labelIds.png", "inst": d / f"{img_id}_gtFine_instanceIds.png"}

    def load_gt(self, paths, split, img_id):
        """Evaluation label map (uint8 [H, W]) in the label space of the module docstring."""
        from PIL import Image
        g = self.gt_paths(paths, split, img_id)
        if self.name == "voc2012":
            return np.array(Image.open(g["cls"]), dtype=np.uint8)
        if self.name == "pc59":
            raw = load_pc59_png(g["cls"])
            return np.where(raw == 0, 255, raw).astype(np.uint8)
        train_id = CITY_LUT[np.array(Image.open(g["label"]), dtype=np.uint8)]
        return np.where(train_id == 255, 255, train_id + 1).astype(np.uint8)

    def list_ids(self, paths, split):
        """All image ids of a split, sorted (train = 1464 / 2975 / 4998, val = 1449 / 500 / 5105)."""
        if self.name == "voc2012":
            lst = Path(paths["voc_root"]) / "ImageSets" / "Segmentation" / f"{split}.txt"
            need(lst, f"VOC {split} list")
            ids = sorted(x.strip() for x in lst.read_text().splitlines() if x.strip())
        elif self.name == "pc59":
            # one line per image: 'JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png'
            lst = Path(paths["pc59_root"]) / f"pascal_context_{split}.txt"
            need(lst, f"PC59 {split} list")
            ids = sorted(Path(x.split()[0]).stem for x in lst.read_text().splitlines() if x.strip())
        else:
            d = Path(paths["cityscapes_gtFine"]) / split
            need(d, f"Cityscapes gtFine/{split}")
            ids = sorted(p.name[: -len("_gtFine_labelIds.png")] for p in d.glob("*/*_gtFine_labelIds.png"))
        n = self.n_train if split == "train" else self.n_val
        if len(ids) != n:
            raise PipelineError(f"{self.name} {split}: found {len(ids)} images, expected {n} (check the dataset root)")
        return ids

    def split_train(self):
        """fit / holdout ids, the exact split of Method B (assets/<ds>_split_train.json)."""
        p = ASSETS_DIR / f"{self.name}_split_train.json"
        need(p, "split file (copied from the Method-B raw run)")
        s = json.load(open(p))
        fit, hold = list(s["fit"]), list(s["holdout"])
        assert len(set(fit) & set(hold)) == 0 and len(fit) + len(hold) == self.n_train, (len(fit), len(hold))
        return fit, hold


def spec(name) -> DatasetSpec:
    return DatasetSpec(name)


def load_pc59_png(path):
    """SegmentationClassContext PNG: 0 = background (other Context classes), v in 1..59 = PC59_CLASSES[v-1]."""
    from PIL import Image
    raw = np.array(Image.open(path), dtype=np.uint8)
    if raw.ndim != 2 or raw.max() > 59:
        raise PipelineError(f"{path}: not a PC59 label map (shape {raw.shape}, max {raw.max()}); "
                            f"expected values 0..59 (SegmentationClassContext)")
    return raw


# ----------------------------------------------------------------------------------------------
# paths.yaml
# ----------------------------------------------------------------------------------------------
PATH_KEYS = {
    "voc_root": "folder holding JPEGImages/, SegmentationClass/, SegmentationObject/, ImageSets/Segmentation/",
    "cityscapes_leftImg8bit": "Cityscapes folder holding train/<city>/*_leftImg8bit.png and val/...",
    "cityscapes_gtFine": "Cityscapes folder holding train/<city>/*_gtFine_{labelIds,instanceIds}.png and val/...",
    "pc59_root": "PC59 folder holding JPEGImages/ (VOC2010), SegmentationClassContext/, pascal_context_{train,val}.txt",
    "sam3_checkpoint": "sam3.pt (the official SAM3 image checkpoint)",
    "out_root": "output folder (created if missing)",
}
CORE_PATH_KEYS = ("sam3_checkpoint", "out_root")
DATASET_PATH_KEYS = {"voc2012": ("voc_root",), "cityscapes": ("cityscapes_leftImg8bit", "cityscapes_gtFine"),
                     "pc59": ("pc59_root",)}


def load_paths(yaml_path, datasets=MAIN_DATASETS):
    """Read paths.yaml. Required: sam3_checkpoint, out_root and the keys of `datasets` (the main run needs
    VOC + Cityscapes, the PC59 run only pc59_root). Every known key that is present is resolved."""
    import yaml
    p = Path(yaml_path)
    need(p, "paths file (copy paths.example.yaml to paths.yaml and edit it)")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    required = list(CORE_PATH_KEYS) + [k for ds in datasets for k in DATASET_PATH_KEYS[ds]]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise PipelineError(f"{p}: missing keys {missing}. Expected:\n" + "\n".join(f"  {k}: {PATH_KEYS[k]}" for k in required))
    base = p.resolve().parent
    out = dict(cfg)
    for k in [k for k in PATH_KEYS if cfg.get(k)]:   # relative paths are relative to the yaml file
        q = Path(os.path.expanduser(str(cfg[k])))
        out[k] = str(q if q.is_absolute() else (base / q).resolve())
    out.setdefault("device", "cuda")
    out.setdefault("num_workers", 4)
    assert out["device"] in ("cuda", "cpu"), out["device"]
    out["num_workers"] = int(out["num_workers"])
    out["_yaml"] = str(p.resolve())
    return out


# ----------------------------------------------------------------------------------------------
# Metrics and threshold choice (same definitions as Method B, diagnose notebook)
# ----------------------------------------------------------------------------------------------
def per_label_iou(conf):
    conf = np.asarray(conf, np.float64)
    tp = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def miou(conf, sp: DatasetSpec):
    return float(np.nanmean(per_label_iou(conf)[sp.eval_idx]))


def conf_tau_from_hist(hist):
    """hist [K, K, NB] (GT x winning label x score bin) -> confusion [NT, K, K] at every tau of the grid.
    At tau_j a pixel keeps its winning label iff the winner's score > tau_j (bin > j); otherwise label 0."""
    K = hist.shape[0]
    out = np.zeros((NT, K, K), np.int64)
    for j in range(NT):
        keep = hist[:, :, j + 1:].sum(-1)
        drop = hist[:, :, : j + 1].sum((1, 2))
        out[j] = keep
        out[j][:, 0] += drop
    return out


def choose_tau(miou_by_tau):
    """Index of the chosen tau: best mIoU, ties (within TAU_TIE) -> closest to 0.5, then the larger tau.
    Same rule as tau* of Method B (diagnose, step 1)."""
    m = np.asarray(miou_by_tau, np.float64)
    best = np.nanmax(m)
    tied = [j for j in range(NT) if m[j] >= best - TAU_TIE]
    tied.sort(key=lambda j: (abs(TAUS[j] - BASE_THR), -TAUS[j]))
    return int(tied[0])


# ----------------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------------
class PipelineError(RuntimeError):
    """Expected, explained failure (bad path, failed check): printed without a traceback."""


def tree(root, depth=2, limit=60):
    root = Path(root)
    if not root.exists():
        return f"{root} does not exist"
    items = sorted(root.glob("*")) + (sorted(root.glob("*/*")) if depth > 1 else [])
    lines = [str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in items[:limit]]
    return "\n  ".join(lines) or "(empty)"


def need(path, what):
    path = Path(path)
    if path.exists():
        return path
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    raise PipelineError(f"{what} not found: {path}\nContent of {parent} (2 levels):\n  {tree(parent)}")


def save_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def model_dir(paths, ds, cfg):
    return Path(paths["out_root"]) / f"{ds}_{cfg}"


def model_file(paths, ds, cfg, what):
    """outputs/<ds>_<cfg>/<ds>_<cfg>_<what>  (naming rule of section 8.6 of thongtin_finetuning.md)"""
    return model_dir(paths, ds, cfg) / f"{ds}_{cfg}_{what}"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_main(fn):
    """Run a CLI main; PipelineError -> clean message + exit code 2."""
    try:
        fn()
    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(2)
