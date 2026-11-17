"""Step 1: ground truth -> COCO json + RLE masks for the TRAIN-FIT images (the only images SAM3 trains on).

  python ft_prepare_data.py --paths paths.yaml [--datasets cityscapes voc2012] [--check]

Output: <out_root>/data_coco/<ds>_fit_coco.json, <ds>_fit_stats.json, samples/<ds>_*.png

One annotation = one object; its category name is the PROMPT of the class (VOC spaced: "dining table").
The repo loader (sam3.train.data.coco_json_loaders.COCO_FROM_JSON, include_negatives=True) turns each
(image, class) pair into one query; classes absent from the image become negative queries.

Objects:
  VOC  : one object per id of SegmentationObject; its class from SegmentationClass.
         Void (255) pixels <= 3 px from a labelled pixel take class AND object id of the nearest labelled
         pixel (background included); deeper void becomes background (negative for every query).
  City : person rider car truck bus train motorcycle bicycle -> one object per instance id (>= 1000);
         the "group" region of a class (id < 1000) is one more object (iscrowd = 0, the repo drops crowds).
         pole / traffic light / traffic sign -> 8-connected pieces of the class; pieces < 64 px join the
         nearest piece >= 64 px (all pieces form one object if none is that large).
         road sidewalk building wall fence vegetation terrain sky -> one mask per class per image.
         trainId 255 pixels belong to no object (negative for every query); eval still ignores them.
Every image is checked: objects are disjoint, each labelled pixel belongs to exactly one object of its
class, every RLE decodes back to its mask, bbox/area match the mask.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import ft_common as fc

PREPARE_VERSION = "prep-1.0"


def coco_paths(paths, ds):
    d = Path(paths.get("coco_root") or Path(paths["out_root"]) / "data_coco")
    return d / f"{ds}_fit_coco.json", d / f"{ds}_fit_stats.json", d / "samples"


def prepare_fingerprint(paths, ds):
    sp = fc.spec(ds)
    fit, _ = sp.split_train()
    return fc.fingerprint({"v": PREPARE_VERSION, "ds": ds, "fit": fit, "prompts": sp.prompts,
                           "void_px": fc.VOID_FILL_PX, "min_piece": fc.CITY_MIN_PIECE})


# ----------------------------------------------------------------------------------------------
# Object maps (pure numpy / scipy; one image at a time)
# ----------------------------------------------------------------------------------------------
def voc_objects(cls, obj):
    """cls, obj: uint8 [H, W] (SegmentationClass, SegmentationObject).
    -> (objects [(label 1..20, bool mask)], label map after void filling, void stats)"""
    from scipy import ndimage
    void = cls == 255                                   # the class map defines void (the object map follows)
    stats = {"void": int(void.sum()), "filled": 0, "deep": 0}
    cls_f, obj_f = cls.copy(), obj.copy()
    if void.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(void, return_indices=True)
        near = void & (dist <= fc.VOID_FILL_PX)
        deep = void & ~near
        cls_f[near] = cls[iy[near], ix[near]]
        obj_f[near] = obj[iy[near], ix[near]]
        cls_f[deep] = 0
        obj_f[deep] = 0
        stats["filled"], stats["deep"] = int(near.sum()), int(deep.sum())
    assert cls_f.max() <= 20, "class map still has void after filling"
    # object id 255 can survive only where the object map had void but the class map did not
    obj_f[obj_f == 255] = 0
    fg = cls_f > 0
    bad = fg & (obj_f == 0)
    if bad.any():
        raise fc.PipelineError(f"{int(bad.sum())} labelled pixels have no object id")
    objects = []
    for o in np.unique(obj_f[fg]):
        m = (obj_f == o) & fg
        labs = np.unique(cls_f[m])
        if len(labs) != 1 or labs[0] == 0:
            raise fc.PipelineError(f"object {o}: classes {labs.tolist()} (expected exactly one class)")
        objects.append((int(labs[0]), m))
    stray = (obj_f > 0) & ~fg
    stats["obj_on_bg"] = int(stray.sum())               # object pixels whose class is background: ignored
    return objects, cls_f, stats


def city_objects(label_ids, inst_ids):
    """label_ids uint8, inst_ids int32 [H, W] (*_gtFine_labelIds.png, *_gtFine_instanceIds.png).
    -> (objects [(label 1..19, bool mask)], eval label map, stats)"""
    from scipy import ndimage
    train_id = fc.CITY_LUT[label_ids]
    lab = np.where(train_id == 255, 0, train_id + 1).astype(np.uint8)     # 0 = no object here
    objects, stats = [], {"merged_pieces": 0, "pieces": 0}
    for lid, tid in fc.CITY_ID2TRAIN.items():
        cm = label_ids == lid
        if not cm.any():
            continue
        c = tid + 1
        if lid in fc.CITY_INSTANCE_LABELIDS:
            vals = np.unique(inst_ids[cm])
            for v in vals:
                if v >= 1000 and v // 1000 != lid:
                    raise fc.PipelineError(f"instance id {v} inside labelId {lid}")
                if v < 1000 and v != lid:
                    raise fc.PipelineError(f"region value {v} inside labelId {lid}")
                objects.append((c, cm & (inst_ids == v)))
        elif lid in fc.CITY_THING_NO_INSTANCE_LABELIDS:
            comp, n = ndimage.label(cm, structure=np.ones((3, 3), bool))
            sizes = np.bincount(comp.ravel(), minlength=n + 1)
            big = [k for k in range(1, n + 1) if sizes[k] >= fc.CITY_MIN_PIECE]
            stats["pieces"] += n
            if not big:
                objects.append((c, cm))
                stats["merged_pieces"] += max(n - 1, 0)
                continue
            if len(big) < n:
                big_map = np.where(np.isin(comp, big), comp, 0)
                _, (iy, ix) = ndimage.distance_transform_edt(big_map == 0, return_indices=True)
                small = cm & (big_map == 0)
                big_map[small] = big_map[iy[small], ix[small]]
                comp = big_map
                stats["merged_pieces"] += n - len(big)
            for k in big:
                objects.append((c, comp == k))
        else:
            objects.append((c, cm))
    return objects, lab, stats


# ----------------------------------------------------------------------------------------------
# One image -> COCO annotations (+ checks)
# ----------------------------------------------------------------------------------------------
def encode(mask):
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    area = int(mu.area(rle))
    bbox = [float(x) for x in mu.toBbox(rle)]
    rle["counts"] = rle["counts"].decode("ascii")
    return rle, area, bbox


def check_objects(objects, lab, H, W):
    """Disjoint objects; every pixel with label c > 0 belongs to exactly one object of class c."""
    cover = np.zeros((H, W), np.int16)
    cls_of = np.zeros((H, W), np.uint8)
    for c, m in objects:
        cover += m
        cls_of[m] = c
    if cover.max() > 1:
        raise fc.PipelineError(f"{int((cover > 1).sum())} pixels belong to two objects")
    want = lab > 0
    if not np.array_equal(cover == 1, want) or not np.array_equal(cls_of[want], lab[want]):
        raise fc.PipelineError("objects do not cover the labelled pixels exactly")


def process_image(job):
    from PIL import Image
    from pycocotools import mask as mu
    ds, paths, img_id, check_images = job
    sp = fc.spec(ds)
    g = sp.gt_paths(paths, "train", img_id)
    if ds == "voc2012":
        cls = np.array(Image.open(g["cls"]), np.uint8)
        obj = np.array(Image.open(g["obj"]), np.uint8)
        assert cls.shape == obj.shape
        objects, lab, st = voc_objects(cls, obj)
    else:
        label_ids = np.array(Image.open(g["label"]), np.uint8)
        inst_ids = np.array(Image.open(g["inst"]), np.int32)
        assert label_ids.shape == inst_ids.shape
        objects, lab, st = city_objects(label_ids, inst_ids)
    H, W = lab.shape
    if check_images:
        ip = sp.image_path(paths, "train", img_id)
        with Image.open(ip) as im:
            if im.size != (W, H):
                raise fc.PipelineError(f"{img_id}: image {im.size} != GT {(W, H)}")
    check_objects(objects, lab, H, W)
    anns = []
    for c, m in objects:
        rle, area, bbox = encode(m)
        assert area == int(m.sum()) and area > 0
        back = mu.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)
        if not np.array_equal(back, m):
            raise fc.PipelineError(f"{img_id}: RLE does not decode back to the mask")
        ys, xs = np.nonzero(m)
        assert bbox == [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)], bbox
        anns.append({"category_id": c, "segmentation": rle, "area": area, "bbox": bbox, "iscrowd": 0})
    st["pixels"] = H * W
    return {"id": img_id, "H": H, "W": W, "anns": anns, "stats": st}


# ----------------------------------------------------------------------------------------------
def build(paths, ds, workers, check_images=True, limit=None):
    sp = fc.spec(ds)
    fit, _ = sp.split_train()
    all_train = set(sp.list_ids(paths, "train"))
    missing = [i for i in fit if i not in all_train]
    if missing:
        raise fc.PipelineError(f"{ds}: {len(missing)} fit ids not in the dataset, e.g. {missing[:3]}")
    ids = fit[:limit] if limit else fit
    for i in ids[:1]:
        for k, p in sp.gt_paths(paths, "train", i).items():
            fc.need(p, f"{ds} GT ({k})")
    jobs = [(ds, paths, i, check_images) for i in ids]
    results = []
    from tqdm import tqdm
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            for r in tqdm(ex.map(process_image, jobs, chunksize=8), total=len(jobs), desc=f"{ds} GT->COCO"):
                results.append(r)
    else:
        for j in tqdm(jobs, desc=f"{ds} GT->COCO"):
            results.append(process_image(j))

    images, anns = [], []
    per_class = {c: {"n_images": 0, "n_objects": 0, "max_objects_in_image": 0} for c in sp.classes}
    tot = {"void": 0, "filled": 0, "deep": 0, "pixels": 0, "obj_on_bg": 0, "pieces": 0, "merged_pieces": 0}
    too_many = []
    for k, r in enumerate(results, start=1):
        images.append({"id": k, "file_name": sp.rel_image(r["id"]), "height": r["H"], "width": r["W"], "sid": r["id"]})
        cnt = np.zeros(sp.K, np.int64)
        for a in r["anns"]:
            anns.append(dict(a, id=len(anns) + 1, image_id=k))
            cnt[a["category_id"]] += 1
        for c in range(1, sp.K):
            pc = per_class[sp.classes[c - 1]]
            if cnt[c]:
                pc["n_images"] += 1
                pc["n_objects"] += int(cnt[c])
                pc["max_objects_in_image"] = max(pc["max_objects_in_image"], int(cnt[c]))
            if cnt[c] > fc.MAX_OBJ_PER_QUERY:
                too_many.append((r["id"], sp.classes[c - 1], int(cnt[c])))
        for key in tot:
            tot[key] += r["stats"].get(key, 0)
    coco = {"info": {"dataset": ds, "split": "train-fit", "version": PREPARE_VERSION,
                     "note": "category name = SAM3 prompt; one annotation per object (see ft_prepare_data.py)"},
            "images": images, "annotations": anns,
            "categories": [{"id": c, "name": sp.prompts[c - 1], "class": sp.classes[c - 1]} for c in range(1, sp.K)]}
    stats = {"dataset": ds, "n_images": len(images), "n_annotations": len(anns), "per_class": per_class,
             "pixels": tot, "queries_per_image": sp.C,
             "positive_queries_per_image": float(np.mean([len({a["category_id"] for a in r["anns"]}) for r in results])),
             "queries_over_max_objects": too_many, "fingerprint": prepare_fingerprint(paths, ds), "limit": limit}
    if ds == "voc2012" and tot["void"]:
        stats["void_share_of_pixels"] = tot["void"] / tot["pixels"]
        stats["void_filled_share"] = tot["filled"] / tot["void"]
        stats["void_deep_share"] = tot["deep"] / tot["void"]
    return coco, stats, results


def save_samples(paths, ds, coco, out_dir, n=3):
    """Overlay of the objects on a few images (to check by eye)."""
    from PIL import Image
    from pycocotools import mask as mu
    sp = fc.spec(ds)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco["images"][:: max(1, len(coco["images"]) // n)][:n]:
        p = sp.img_folder(paths, "train") / im["file_name"]
        if not p.exists():
            continue
        rgb = np.array(Image.open(p).convert("RGB"), np.float32)
        ov = rgb.copy()
        for a in by_img.get(im["id"], []):
            m = mu.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode("ascii")}).astype(bool)
            col = rng.integers(40, 255, 3)
            ov[m] = 0.45 * rgb[m] + 0.55 * col
            edge = m ^ np.pad(m, 1)[2:, 1:-1] | m ^ np.pad(m, 1)[1:-1, 2:]
            ov[edge & m] = 255
        Image.fromarray(ov.clip(0, 255).astype(np.uint8)).save(out_dir / f"{ds}_{im['sid']}_objects.jpg", quality=85)


def is_done(paths, ds):
    js, st, _ = coco_paths(paths, ds)
    s = fc.load_json(st)
    return js.exists() and s is not None and s.get("fingerprint") == prepare_fingerprint(paths, ds) and not s.get("limit")


def check_existing(paths, ds):
    """Re-read an existing COCO file: ids == fit split, every image has its queries, categories = prompts."""
    sp = fc.spec(ds)
    js, st, _ = coco_paths(paths, ds)
    fc.need(js, f"{ds} COCO file")
    coco = json.load(open(js))
    fit, _ = sp.split_train()
    sids = [im["sid"] for im in coco["images"]]
    assert sids == fit, "image list != fit split"
    assert [c["name"] for c in coco["categories"]] == sp.prompts, "categories != prompts"
    n_ann = np.bincount([a["image_id"] for a in coco["annotations"]], minlength=len(sids) + 1)
    assert (n_ann[1:] > 0).all(), "an image without any object"
    fc.log(f"{ds}: COCO file OK ({len(sids)} images, {len(coco['annotations'])} objects)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--datasets", nargs="+", default=["cityscapes", "voc2012"])
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--check", action="store_true", help="only re-check existing COCO files")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output is up to date")
    ap.add_argument("--limit", type=int, default=None, help="first N fit images only (testing)")
    ap.add_argument("--skip-image-check", action="store_true", help="do not open the images (local test without images)")
    args = ap.parse_args()
    paths = fc.load_paths(args.paths)
    for ds in args.datasets:
        if args.check:
            check_existing(paths, ds)
            continue
        js, st, samples = coco_paths(paths, ds)
        if is_done(paths, ds) and not args.force and not args.limit:
            fc.log(f"{ds}: {js.name} up to date, skip")
            continue
        coco, stats, _ = build(paths, ds, args.workers, check_images=not args.skip_image_check, limit=args.limit)
        if args.limit:
            js = js.with_name(js.stem + f"_limit{args.limit}.json")
            st = st.with_name(st.stem + f"_limit{args.limit}.json")
        fc.save_json_atomic(st, stats)
        js.parent.mkdir(parents=True, exist_ok=True)
        tmp = js.with_name(js.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(coco, f)
        os.replace(tmp, js)
        fc.log(f"{ds}: {stats['n_images']} images, {stats['n_annotations']} objects, "
               f"{stats['positive_queries_per_image']:.1f}/{fc.spec(ds).C} positive queries per image -> {js}")
        if stats["queries_over_max_objects"]:
            fc.log(f"{ds}: WARNING {len(stats['queries_over_max_objects'])} queries have > {fc.MAX_OBJ_PER_QUERY} "
                   f"objects (the repo drops them): {stats['queries_over_max_objects'][:5]}")
        if ds == "voc2012" and "void_filled_share" in stats:
            fc.log(f"voc2012: void {100 * stats['void_share_of_pixels']:.2f}% of pixels; "
                   f"{100 * stats['void_filled_share']:.1f}% filled (<= {fc.VOID_FILL_PX} px), "
                   f"{100 * stats['void_deep_share']:.1f}% -> background")
        if not args.skip_image_check:
            save_samples(paths, ds, coco, samples)


if __name__ == "__main__":
    fc.run_main(main)
