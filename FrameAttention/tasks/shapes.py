"""
Stage 1: synthetic scenes with known objects (generated locally, no download).

64x64 images with 3-5 objects. Each object has a shape, a color and a texture, and objects
may touch or overlap, so neither position nor a single patch is enough to bind attributes.
One color per shape is held out of training entirely (6 of 36 shape-color combinations);
the OOD test asks about objects with exactly those combinations.

Splits: train (no held-out objects), val and test (same distribution, for iid accuracy),
test_ood (every image has one held-out object; questions ask about it).
"""

import json
import os
from multiprocessing import get_context

import numpy as np

from tasks.qa import SceneQA, pick_heldout, scene_stats

SHAPES = ["square", "circle", "triangle", "diamond", "cross", "ring"]
COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan"]
RGB = np.array([[220, 45, 45], [50, 170, 60], [50, 90, 225], [225, 200, 40], [200, 60, 200], [50, 190, 200]],
               np.float32)
TEXTURES = ["solid", "hstripes", "vstripes", "checker"]
HELDOUT = pick_heldout(len(SHAPES), len(COLORS), per_a=1, seed=0)  # one unseen color per shape

DEFAULTS = dict(img_size=64, k_min=3, k_max=5, r_min=6.0, r_max=10.0, min_visible=30, n_questions=4,
                sizes=dict(train=100_000, val=5_000, test=5_000, test_ood=5_000))


def make_qa(cfg=DEFAULTS):
    return SceneQA({"shape": SHAPES, "color": COLORS, "texture": TEXTURES}, ("shape", "color"), HELDOUT,
                   min_visible=cfg["min_visible"], margin=cfg["img_size"] / 16)


def shape_mask(shape, dx, dy, r):
    if shape == 0:  # square
        return (np.abs(dx) <= 0.85 * r) & (np.abs(dy) <= 0.85 * r)
    if shape == 1:  # circle
        return dx * dx + dy * dy <= r * r
    if shape == 2:  # triangle, apex up
        return (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2)
    if shape == 3:  # diamond
        return np.abs(dx) + np.abs(dy) <= r
    if shape == 4:  # cross
        return ((np.abs(dx) <= r / 3) & (np.abs(dy) <= r)) | ((np.abs(dy) <= r / 3) & (np.abs(dx) <= r))
    if shape == 5:  # ring
        d2 = dx * dx + dy * dy
        return (d2 <= r * r) & (d2 >= (0.5 * r) ** 2)
    raise ValueError(shape)


def texture_dark(texture, dx, dy, r):
    u, v = np.floor((dx + r) / 2), np.floor((dy + r) / 2)  # 2-pixel bands
    if texture == 0:
        return np.zeros(dx.shape, bool)
    if texture == 1:
        return v % 2 == 1
    if texture == 2:
        return u % 2 == 1
    return (u + v) % 2 == 1


def render(rng, objs, S):
    """objs: list of (shape, color, texture, cx, cy, r). Returns image, id mask, full areas."""
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
    img = np.full((S, S, 3), rng.uniform(15, 45), np.float32) + rng.normal(0, 4, (S, S, 3))
    mask = np.zeros((S, S), np.uint8)
    areas = []
    for i, (shape, color, texture, cx, cy, r) in enumerate(objs):
        dx, dy = xx - cx, yy - cy
        m = shape_mask(shape, dx, dy, r)
        col = RGB[color] + rng.uniform(-12, 12, 3)
        shade = np.where(texture_dark(texture, dx, dy, r), 0.45, 1.0)
        img[m] = col[None, :] * shade[m][:, None]
        mask[m] = i + 1
        areas.append(int(m.sum()))
    return np.clip(img, 0, 255).astype(np.uint8), mask, np.array(areas)


def sample_scene(rng, cfg, ood):
    S, held = cfg["img_size"], set(HELDOUT)
    for _ in range(500):
        K = int(rng.integers(cfg["k_min"], cfg["k_max"] + 1))
        combos = []
        if ood:  # one held-out object whose shape and color are unique in the scene
            combos.append(HELDOUT[rng.integers(len(HELDOUT))])
        while len(combos) < K:
            s, c = int(rng.integers(len(SHAPES))), int(rng.integers(len(COLORS)))
            if (s, c) in held or (ood and (s == combos[0][0] or c == combos[0][1])):
                continue
            combos.append((s, c))
        objs = []
        for s, c in combos:
            r = rng.uniform(cfg["r_min"], cfg["r_max"])
            cx, cy = rng.uniform(r, S - r, 2)
            objs.append((s, c, int(rng.integers(len(TEXTURES))), cx, cy, r))
        objs = [objs[i] for i in rng.permutation(K)]  # random drawing (occlusion) order
        img, mask, areas = render(rng, objs, S)
        cent, vis = scene_stats(mask, K)
        if np.all(vis >= 0.5 * areas) and np.all(vis >= cfg["min_visible"]):
            attrs = np.array([[s, c, t] for s, c, t, *_ in objs], np.int16)
            return img, mask, attrs, cent, vis
    raise RuntimeError("could not place objects; lower k_max or r_max")


def _chunk(args):
    seed, n, mode, cfg = args
    rng = np.random.default_rng(seed)
    qa = make_qa(cfg)
    S, Q, Kmax = cfg["img_size"], cfg["n_questions"], cfg["k_max"]
    out = dict(images=np.zeros((n, S, S, 3), np.uint8), masks=np.zeros((n, S, S), np.uint8),
               attrs=np.full((n, Kmax, 3), -1, np.int16), n_objects=np.zeros(n, np.int8),
               q_tokens=np.zeros((n, Q, 4), np.int16), answers=np.full((n, Q), -1, np.int16),
               q_kinds=np.zeros((n, Q), np.int8), q_flags=np.zeros((n, Q), np.int8))
    i = 0
    while i < n:
        img, mask, attrs, cent, vis = sample_scene(rng, cfg, ood=(mode == "ood"))
        toks, ans, kinds, flags = qa.make_questions(rng, attrs, cent, vis, Q, mode)
        if (ans >= 0).sum() == 0:
            continue
        K = len(attrs)
        out["images"][i], out["masks"][i] = img, mask
        out["attrs"][i, :K], out["n_objects"][i] = attrs, K
        out["q_tokens"][i], out["answers"][i], out["q_kinds"][i], out["q_flags"][i] = toks, ans, kinds, flags
        i += 1
    return out


def generate(out_dir, cfg=DEFAULTS, workers=None, seed=1234, chunk=1000):
    os.makedirs(out_dir, exist_ok=True)
    modes = dict(train="train", val="iid", test="iid", test_ood="ood")
    workers = workers or max(1, min(16, os.cpu_count() or 1))
    for si, (split, n) in enumerate(cfg["sizes"].items()):
        jobs = [(seed + 1000 * si + c, min(chunk, n - c * chunk), modes[split], cfg)
                for c in range((n + chunk - 1) // chunk)]
        with get_context("spawn").Pool(workers) as pool:  # spawn: safe even if threads are running
            parts = pool.map(_chunk, jobs)
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(out_dir, f"{split}.npz"), **data)
        n_q = int((data["answers"] >= 0).sum())
        print(f"  shapes/{split}: {n} images, {n_q} questions")
    meta = dict(task="shapes", kind="image", img_size=cfg["img_size"], max_objects=cfg["k_max"],
                n_questions=cfg["n_questions"], splits=list(cfg["sizes"]), config=cfg, **make_qa(cfg).meta())
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
