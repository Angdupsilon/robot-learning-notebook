"""
Stage 2 (common fate): MOVi-A from Kubric (Greff et al. 2022). Short videos of CLEVR-style
objects that fall and collide. Each object has shape, color, size and material labels, and
every frame has ground-truth optical flow, which gives the "motion fingerprints" for variant E.

Steps (download_data.py runs all three):
  1. download  the TFDS files over plain HTTPS from the public bucket kubric-public
               (same files tfds.load("movi_a/128x128", data_dir="gs://kubric-public/tfds") reads)
  2. convert   read them with tensorflow-datasets and keep only what we use, per video:
               frame `frame` (image + instance mask), forward flow for `flow_steps` frames pooled
               to a `grid` x `grid` grid, and the label of every object       -> raw_<split>.npz
  3. prepare   make questions (binding pair: shape, color; 2 held-out colors per shape),
               split train/val/test, and save train/val/test/test_ood .npz

Step 2 needs TensorFlow; everything else needs only numpy. Run as a module for step 2:
    python -m tasks.movi convert --src data/raw/movi_a/128x128/1.0.0 --out data/movi_a
MOVi-C / MOVi-E have no attribute labels (only a category), so step 3 supports MOVi-A and
MOVi-B; steps 1-2 work for any MOVi variant.
"""

import argparse
import glob
import json
import os
import zlib

import numpy as np

GCS = "https://storage.googleapis.com/kubric-public/tfds"
MAX_OBJECTS = 24
DEFAULTS = dict(frame=0, flow_steps=8, grid=32, min_visible=40, n_questions=4, seed=11,
                fractions=dict(train=0.85, val=0.05))


# ----------------------------------------------------------------------------- 1. download

def shard_names(info, max_shards=None):
    """TFDS shard file names from dataset_info.json, per split."""
    name = info["name"]
    fmt = str(info.get("fileFormat", "tfrecord")).lower()
    out = {}
    for split in info.get("splits", []):
        n = len(split.get("shardLengths", [])) or int(split.get("numShards", 0))
        if not n:
            continue
        template = split.get("filepathTemplate") or "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}"
        keep = n if not max_shards else min(n, max_shards if split["name"] == "train" else 1)
        out[split["name"]] = [template.replace("{DATASET}", name).replace("{SPLIT}", split["name"])
                              .replace("{FILEFORMAT}", fmt).replace("{SHARD_X_OF_Y}", f"{i:05d}-of-{n:05d}")
                              for i in range(keep)]
    return out


def label_files(features):
    """Class-label name files TFDS keeps next to the shards (e.g. instances-shape_label.labels.txt)."""
    out = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "classLabel" in node:
            out.append("-".join(path) + ".labels.txt")
            return
        for key, val in node.items():
            if key == "features" and isinstance(val, dict):
                for name, spec in val.items():
                    walk(spec, path + [name])
            else:
                walk(val, path)

    walk(features, [])
    return out


def list_bucket(prefix):
    """Fallback: list object names under a prefix with the public GCS JSON API."""
    from tasks.fetch import fetch_json
    names, token = [], None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/kubric-public/o?prefix={prefix}&fields=items(name),nextPageToken"
        page = fetch_json(url + (f"&pageToken={token}" if token else ""))
        names += [it["name"] for it in page.get("items", [])]
        token = page.get("nextPageToken")
        if not token:
            return names


def download(name="movi_a", res="128x128", raw_root="data/raw", max_shards=None):
    from tasks.fetch import download as fetch, human, remote_size
    rel = f"{name}/{res}/1.0.0"
    vdir = os.path.join(raw_root, rel)
    for f in ("dataset_info.json", "features.json"):
        fetch(f"{GCS}/{rel}/{f}", os.path.join(vdir, f), quiet=True)
    with open(os.path.join(vdir, "dataset_info.json")) as f:
        info = json.load(f)
    with open(os.path.join(vdir, "features.json")) as f:
        for fn in label_files(json.load(f)):  # class names live in small text files
            try:
                fetch(f"{GCS}/{rel}/{fn}", os.path.join(vdir, fn), quiet=True)
            except Exception as e:
                print(f"    warning: could not fetch {fn} ({e})")
    shards = shard_names(info, max_shards)
    if not shards or remote_size(f"{GCS}/{rel}/{next(iter(shards.values()))[0]}") is None:
        print("    shard names not in dataset_info.json; listing the bucket instead")
        listing = [os.path.basename(n) for n in list_bucket(f"tfds/{rel}/") if ".tfrecord" in n]
        shards = {}
        for fn in sorted(listing):
            split = fn.split("-")[1].split(".")[0]
            shards.setdefault(split, []).append(fn)
        if max_shards:
            shards = {s: v[:max_shards if s == "train" else 1] for s, v in shards.items()}
    total = sum(int(s.get("numBytes", 0)) for s in info.get("splits", []))
    n_files = sum(len(v) for v in shards.values())
    print(f"    {n_files} shard files ({', '.join(f'{k}: {len(v)}' for k, v in shards.items())})"
          + (f", full dataset {human(total)}" if total and not max_shards else ""))
    for i, fn in enumerate([fn for v in shards.values() for fn in v]):
        fetch(f"{GCS}/{rel}/{fn}", os.path.join(vdir, fn), quiet=True)
        if (i + 1) % 10 == 0 or i + 1 == n_files:
            print(f"    downloaded {i + 1}/{n_files} shards")
    return vdir


# ----------------------------------------------------------------------------- 2. convert

def extract(ex, label_keys, frame, flow_steps, grid):
    """Keep what the experiments use from one decoded TFDS example (numpy dict)."""
    video, seg = ex["video"], ex["segmentations"][..., 0]
    T, H, W = seg.shape
    assert frame + flow_steps <= T and H % grid == 0, (frame, flow_steps, T, H, grid)
    lo, hi = np.asarray(ex["metadata"]["forward_flow_range"], np.float32)
    flow = ex["forward_flow"][frame:frame + flow_steps].astype(np.float32) / 65535.0 * (hi - lo) + lo  # pixels
    p = H // grid
    flow = flow.reshape(flow_steps, grid, p, grid, p, 2).mean((2, 4)).transpose(1, 2, 0, 3)  # (g, g, K, 2)
    labels = np.stack([np.asarray(ex["instances"][k]) for k in label_keys], -1).astype(np.int16)
    attrs = np.full((MAX_OBJECTS, len(label_keys)), -1, np.int16)
    attrs[:len(labels)] = labels[:MAX_OBJECTS]
    return dict(image=video[frame].astype(np.uint8), mask=seg[frame].astype(np.uint8),
                flow=flow.astype(np.float16), attrs=attrs, n_objects=min(len(labels), MAX_OBJECTS))


def save_raw(rows, path):
    np.savez(path, images=np.stack([r["image"] for r in rows]), masks=np.stack([r["mask"] for r in rows]),
             flow=np.stack([r["flow"] for r in rows]), attrs=np.stack([r["attrs"] for r in rows]),
             n_objects=np.array([r["n_objects"] for r in rows], np.int16))


def convert(src, out, frame=0, flow_steps=8, grid=32, max_examples=0):
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(src)
    feats = builder.info.features
    inst = feats["instances"]
    inst = getattr(inst, "feature", inst)  # Sequence -> its FeaturesDict
    label_keys = sorted(k for k in inst.keys() if k.endswith("_label"))
    names = {k[:-len("_label")]: list(inst[k].names) for k in label_keys}
    os.makedirs(out, exist_ok=True)
    res = None
    for split in builder.info.splits.keys():
        files = sorted(glob.glob(os.path.join(src, f"{builder.info.name}-{split}.tfrecord-*")))
        if not files:
            continue
        ds = tf.data.TFRecordDataset(files).map(feats.deserialize_example, num_parallel_calls=tf.data.AUTOTUNE)
        rows = []
        for ex in tfds.as_numpy(ds):
            rows.append(extract(ex, label_keys, frame, flow_steps, grid))
            if len(rows) % 500 == 0:
                print(f"    {split}: {len(rows)} videos")
            if max_examples and len(rows) >= max_examples:
                break
        res = rows[0]["image"].shape[0]
        save_raw(rows, os.path.join(out, f"raw_{split}.npz"))
        print(f"  {os.path.basename(out)}/raw_{split}: {len(rows)} videos")
    with open(os.path.join(out, "raw_meta.json"), "w") as f:
        json.dump(dict(name=builder.info.name, labels=names, label_keys=label_keys, frame=frame,
                       flow_steps=flow_steps, grid=grid, res=res), f, indent=1)


# ----------------------------------------------------------------------------- 3. prepare

def prepare(out, cfg=DEFAULTS):
    from tasks.qa import SceneQA, pick_heldout, scene_stats
    with open(os.path.join(out, "raw_meta.json")) as f:
        raw = json.load(f)
    attrs_names = [k[:-len("_label")] for k in raw["label_keys"]]
    if not {"shape", "color"} <= set(attrs_names):
        raise SystemExit(f"{raw['name']} has labels {attrs_names}; questions need shape and color (use movi_a)")
    attributes = {a: raw["labels"][a] for a in attrs_names}
    n_shape, n_color = len(attributes["shape"]), len(attributes["color"])
    per_shape = 2 if n_color >= 2 * n_shape else 1  # MOVi-A: 2 of 8 colors held out per shape
    heldout = pick_heldout(n_shape, n_color, per_a=per_shape, seed=cfg["seed"])
    res = raw["res"]
    qa = SceneQA(attributes, ("shape", "color"), heldout, min_visible=cfg["min_visible"], margin=res / 16)

    pools = {s: np.load(os.path.join(out, f"raw_{s}.npz")) for s in ("train", "validation")
             if os.path.exists(os.path.join(out, f"raw_{s}.npz"))}
    tr = pools["train"]
    n = len(tr["images"])
    a, b = int(cfg["fractions"]["train"] * n), int((cfg["fractions"]["train"] + cfg["fractions"]["val"]) * n)
    parts = {"train": [("train", np.arange(0, a))], "val": [("train", np.arange(a, b))],
             "test": [("train", np.arange(b, n))]}
    if "validation" in pools:
        parts["test"].append(("validation", np.arange(len(pools["validation"]["images"]))))
    modes = {"train": [("train", "train")], "val": [("val", "iid")], "test": [("test", "iid"), ("test_ood", "ood")]}
    Q = cfg["n_questions"]
    flow_std = None
    for split, sources in parts.items():
        arrays = {k: np.concatenate([pools[src][k][idx] for src, idx in sources]) for k in tr.files}
        N = len(arrays["images"])
        for qsplit, mode in modes[split]:
            toks, ans = np.zeros((N, Q, 4), np.int16), np.full((N, Q), -1, np.int16)
            kinds, flags = np.zeros((N, Q), np.int8), np.zeros((N, Q), np.int8)
            for i in range(N):
                K = int(arrays["n_objects"][i])
                cent, vis = scene_stats(arrays["masks"][i], K)
                at = arrays["attrs"][i, :K].astype(np.int64)
                rng = np.random.default_rng([cfg["seed"], i, zlib.crc32(qsplit.encode())])
                toks[i], ans[i], kinds[i], flags[i] = qa.make_questions(rng, at, cent, vis, Q, mode)
            keep = np.nonzero((ans >= 0).any(1))[0]
            if flow_std is None:
                flow_std = float(arrays["flow"].astype(np.float32).std()) or 1.0
            np.savez(os.path.join(out, f"{qsplit}.npz"), images=arrays["images"][keep], masks=arrays["masks"][keep],
                     flow=arrays["flow"][keep], attrs=arrays["attrs"][keep], n_objects=arrays["n_objects"][keep],
                     q_tokens=toks[keep], answers=ans[keep], q_kinds=kinds[keep], q_flags=flags[keep])
            print(f"  {os.path.basename(out)}/{qsplit}: {len(keep)} videos, {int((ans[keep] >= 0).sum())} questions")
    meta = dict(task=os.path.basename(out.rstrip("/")), kind="image", img_size=res, max_objects=MAX_OBJECTS,
                n_questions=Q, splits=["train", "val", "test", "test_ood"], config=cfg,
                flow=dict(grid=raw["grid"], steps=raw["flow_steps"], frame=raw["frame"], std=flow_std),
                **qa.meta())
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["convert", "prepare"])
    p.add_argument("--src")
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=DEFAULTS["frame"])
    p.add_argument("--flow-steps", type=int, default=DEFAULTS["flow_steps"])
    p.add_argument("--grid", type=int, default=DEFAULTS["grid"])
    p.add_argument("--max-examples", type=int, default=0)
    args = p.parse_args()
    if args.step == "convert":
        convert(args.src, args.out, args.frame, args.flow_steps, args.grid, args.max_examples)
    else:
        prepare(args.out)
