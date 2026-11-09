"""
Stage 2: CLEVRTex (Karazija et al. 2021). Textured objects on textured ground, where
appearance is misleading; CAMO goes further and gives objects the ground's material.

Raw layout (after extracting the official archives): files named
    CLEVRTEX_<variant>_<6-digit index>.png        image
    CLEVRTEX_<variant>_<index>_flat.png           object-id mask (0 = background, i = object with "index" i)
    CLEVRTEX_<variant>_<index>.json               metadata: objects[*].shape / size / material / index
Preprocessing follows the official loader: centre crop 80% of the short side, resize
(bilinear for images, nearest for masks). Splits follow the official index fractions:
test = first 10%, val = next 10%, train = last 80%.

Questions use shape, size and material, with (shape, material) as the binding pair. Each
material has one held-out shape (25% of combinations). CLEVRTex images are fixed, so the
hold-out is at the question level: training never asks about held-out objects, test_ood
asks only about them. outd (unseen materials) and camo are kept for grouping metrics only.
"""

import glob
import json
import os
import re
import zlib
from multiprocessing import get_context

import numpy as np

from tasks.qa import SceneQA, scene_stats

BASE_URL = "https://thor.robots.ox.ac.uk/datasets/clevrtex"
ARCHIVES = [f"clevrtex_full_part{i}.tar.gz" for i in range(1, 6)] + ["clevrtex_outd.tar.gz", "clevrtex_camo.tar.gz"]
SHAPES = ["cube", "cylinder", "sphere", "monkey"]
SIZES = ["small", "medium", "large"]
MAX_OBJECTS = 10
DEFAULTS = dict(res=128, min_visible=60, n_questions=4, seed=7)


def index_variant(raw_dir, variant):
    """Sorted list of (index, image, mask, json) for one variant, found anywhere under raw_dir."""
    pat = re.compile(rf"CLEVRTEX_{variant}_(\d{{6}})\.png$")
    rows = []
    for path in glob.glob(os.path.join(raw_dir, "**", f"CLEVRTEX_{variant}_*.png"), recursive=True):
        m = pat.search(os.path.basename(path))
        if m:
            stem = path[:-4]
            rows.append((int(m.group(1)), path, stem + "_flat.png", stem + ".json"))
    return sorted(rows)


def load_example(img_path, mask_path, res):
    from PIL import Image
    img, msk = Image.open(img_path).convert("RGB"), Image.open(mask_path)
    c = int(0.8 * float(min(img.width, img.height)))
    box = ((img.width - c) // 2, (img.height - c) // 2, (img.width + c) // 2, (img.height + c) // 2)
    img = img.crop(box).resize((res, res), resample=Image.BILINEAR)
    msk = msk.crop(box).resize((res, res), resample=Image.NEAREST)
    return np.asarray(img, np.uint8), np.asarray(msk, np.uint8)


def object_attrs(meta, materials):
    """attrs[k] for mask id k+1, as (shape, size, material) indices (-1 = unknown)."""
    attrs = np.full((MAX_OBJECTS, 3), -1, np.int16)
    for pos, obj in enumerate(meta["objects"]):
        k = int(obj.get("index", pos + 1)) - 1
        if 0 <= k < MAX_OBJECTS:
            attrs[k] = (SHAPES.index(obj["shape"]) if obj["shape"] in SHAPES else -1,
                        SIZES.index(obj["size"]) if obj["size"] in SIZES else -1,
                        materials.get(obj["material"], -1))
    return attrs


def make_qa(materials, cfg=DEFAULTS):
    order = np.random.default_rng(cfg["seed"]).permutation(len(materials))
    heldout = [(i % len(SHAPES), int(order[i])) for i in range(len(materials))]  # one shape per material
    return SceneQA({"shape": SHAPES, "size": SIZES, "material": list(materials)}, ("shape", "material"),
                   heldout, min_visible=cfg["min_visible"], margin=cfg["res"] / 16)


def _work(args):
    (variant, idx, img_path, mask_path, json_path), modes, materials, cfg = args
    image, mask = load_example(img_path, mask_path, cfg["res"])
    with open(json_path) as f:
        attrs = object_attrs(json.load(f), materials)
    K = int(min(MAX_OBJECTS, mask.max()))
    out = dict(image=image, mask=mask, attrs=attrs, n_objects=K, questions={})
    if modes:
        qa = make_qa(materials, cfg)
        cent, vis = scene_stats(mask, K)
        a = attrs[:K].astype(np.int64)
        ok = np.all(a >= 0, axis=1)
        vis = np.where(ok, vis, 0)  # objects with unknown attributes are never asked about
        a = np.where(a >= 0, a, 0)
        for split, mode in modes:
            rng = np.random.default_rng([cfg["seed"], idx, zlib.crc32(split.encode())])
            out["questions"][split] = qa.make_questions(rng, a, cent, vis, cfg["n_questions"], mode)
    return out


def convert(raw_dir, out_dir, cfg=DEFAULTS, workers=None):
    os.makedirs(out_dir, exist_ok=True)
    full = index_variant(raw_dir, "full")
    if not full:
        raise FileNotFoundError(f"no CLEVRTEX_full_*.png under {raw_dir}")
    materials = set()
    for _, _, _, jp in full:
        with open(jp) as f:
            materials.update(o["material"] for o in json.load(f)["objects"])
    materials = {m: i for i, m in enumerate(sorted(materials))}
    n = len(full)
    plan = {  # split -> (rows, [(question split, mode), ...])
        "train": (full[int(0.2 * n):], [("train", "train")]),
        "val": (full[int(0.1 * n):int(0.2 * n)], [("val", "iid")]),
        "test": (full[:int(0.1 * n)], [("test", "iid"), ("test_ood", "ood")]),
        "outd": (index_variant(raw_dir, "outd"), []),
        "camo": (index_variant(raw_dir, "camo"), []),
    }
    workers = workers or max(1, min(32, os.cpu_count() or 1))
    res, Q = cfg["res"], cfg["n_questions"]
    written = []
    for split, (rows, modes) in plan.items():
        if not rows:
            print(f"  clevrtex/{split}: no files found, skipped")
            continue
        variant = "full" if split in ("train", "val", "test") else split
        jobs = [((variant, *row), modes, materials, cfg) for row in rows]
        N = len(rows)
        images = np.zeros((N, res, res, 3), np.uint8)
        masks = np.zeros((N, res, res), np.uint8)
        attrs = np.full((N, MAX_OBJECTS, 3), -1, np.int16)
        n_obj = np.zeros(N, np.int8)
        qs = {s: [np.zeros((N, Q, 4), np.int16), np.full((N, Q), -1, np.int16),
                  np.zeros((N, Q), np.int8), np.zeros((N, Q), np.int8)] for s, _ in modes}
        with get_context("spawn").Pool(workers) as pool:  # spawn: safe even if threads are running
            for i, r in enumerate(pool.imap(_work, jobs, chunksize=64)):
                images[i], masks[i], attrs[i], n_obj[i] = r["image"], r["mask"], r["attrs"], r["n_objects"]
                for s in qs:
                    for arr, val in zip(qs[s], r["questions"][s]):
                        arr[i] = val
                if (i + 1) % 5000 == 0:
                    print(f"    {split}: {i + 1}/{N}")
        targets = [(s, *qs[s]) for s, _ in modes] or [(split, np.zeros((N, Q, 4), np.int16),
                                                         np.full((N, Q), -1, np.int16),
                                                         np.zeros((N, Q), np.int8), np.zeros((N, Q), np.int8))]
        for s, toks, ans, kinds, flags in targets:
            keep = np.arange(N) if s in ("outd", "camo") else np.nonzero((ans >= 0).any(1))[0]
            np.savez(os.path.join(out_dir, f"{s}.npz"), images=images[keep], masks=masks[keep],
                     attrs=attrs[keep], n_objects=n_obj[keep], q_tokens=toks[keep], answers=ans[keep],
                     q_kinds=kinds[keep], q_flags=flags[keep])
            print(f"  clevrtex/{s}: {len(keep)} images, {int((ans[keep] >= 0).sum())} questions")
            written.append(s)
    meta = dict(task="clevrtex", kind="image", img_size=res, max_objects=MAX_OBJECTS, n_questions=Q,
                splits=written, config=cfg, **make_qa(materials, cfg).meta())
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
