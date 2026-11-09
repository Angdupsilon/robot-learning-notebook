"""
Checks for the data, metrics, model and training pipeline.

Every generated question is re-answered by an independent interpreter, held-out combinations
are checked to be absent from training, and each variant trains for a few steps end to end.
Run:  python -m pytest tests -q
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import metrics as M  # noqa: E402
from model import VARIANTS, FrameTransformer, matched_layers  # noqa: E402
from tasks import boxes, clevrtex, movi, shapes  # noqa: E402
from tasks.qa import FLAG_OOD, FLAG_PAIR, RELATIONS, SceneQA, scene_stats  # noqa: E402


# ----------------------------------------------------------------------------- an independent interpreter

def answer_scene_question(meta, tokens, attrs, cent, vis):
    """Re-answer an encoded question from the scene, without using SceneQA."""
    names = list(meta["attributes"])
    sizes = [len(v) for v in meta["attributes"].values()]
    offs = np.cumsum([0] + sizes)[:-1]
    A, n_ans = len(names), sum(sizes)
    target = tokens[0] - 1
    desc = []
    for t in tokens[1:3]:
        if t:
            g = t - (1 + A)
            d = int(np.searchsorted(offs, g, side="right") - 1)
            desc.append((d, g - offs[d]))
    rel = tokens[3] - (1 + A + n_ans)
    present = [j for j in range(len(attrs)) if vis[j] > 0]
    match = [j for j in present if all(attrs[j][d] == v for d, v in desc)]
    assert len(match) == 1, f"description matches {len(match)} objects"
    o = match[0]
    if rel:
        m, (x, y) = meta["margin"], cent[o]
        cands = [j for j in present if j != o and vis[j] >= meta["min_visible"]]
        side = {1: lambda j: cent[j][0] < x - m, 2: lambda j: cent[j][0] > x + m,
                3: lambda j: cent[j][1] < y - m, 4: lambda j: cent[j][1] > y + m}[rel]
        dist = sorted((np.hypot(cent[j][0] - x, cent[j][1] - y), j) for j in cands if side(j))
        assert dist and (len(dist) == 1 or dist[1][0] - dist[0][0] >= m / 2)
        o = dist[0][1]
    assert vis[o] >= meta["min_visible"]
    return int(offs[target] + attrs[o][target]), o


def check_scene_split(meta, z, mode):
    held = {tuple(p) for p in meta["heldout"]}
    pa, pb = [list(meta["attributes"]).index(n) for n in meta["binding_pair"]]
    n_checked = 0
    for i in range(len(z["answers"])):
        K = int(z["n_objects"][i])
        attrs = z["attrs"][i, :K].astype(int)
        cent, vis = scene_stats(z["masks"][i], K)
        for q in range(z["answers"].shape[1]):
            if z["answers"][i, q] < 0:
                continue
            ans, o = answer_scene_question(meta, z["q_tokens"][i, q], attrs, cent, vis)
            assert ans == z["answers"][i, q]
            is_held = (attrs[o][pa], attrs[o][pb]) in held
            flags = int(z["q_flags"][i, q])
            if mode == "ood":
                assert is_held and flags & FLAG_OOD and flags & FLAG_PAIR
            else:
                assert not is_held and not flags & FLAG_OOD
            n_checked += 1
    return n_checked


# ----------------------------------------------------------------------------- datasets

@pytest.fixture(scope="module")
def shapes_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("shapes"))
    cfg = dict(shapes.DEFAULTS, sizes=dict(train=300, val=60, test=60, test_ood=60))
    shapes.generate(d, cfg, workers=2, chunk=100)
    return d


def test_shapes_questions_are_correct_and_holdout_is_respected(shapes_dir):
    meta = json.load(open(os.path.join(shapes_dir, "meta.json")))
    held = {tuple(p) for p in meta["heldout"]}
    for split, mode in (("train", "train"), ("val", "iid"), ("test", "iid"), ("test_ood", "ood")):
        z = np.load(os.path.join(shapes_dir, f"{split}.npz"))
        assert check_scene_split(meta, z, mode) > 0
        for i in range(len(z["answers"])):  # training scenes never even show a held-out object
            combos = {(int(s), int(c)) for s, c, _ in z["attrs"][i, :z["n_objects"][i]]}
            assert bool(combos & held) == (mode == "ood")
    assert len(held) == 6 and len({c for _, c in held}) == 6  # one unseen color per shape


def test_scene_qa_on_random_scenes():
    """Property test of the question generator on random scenes with repeats and occlusion."""
    rng = np.random.default_rng(0)
    qa = SceneQA({"a": list("pqrs"), "b": list("xyz"), "c": list("uv")}, ("a", "b"),
                 [(0, 1), (2, 2)], min_visible=5, margin=4.0)
    meta = qa.meta()
    for _ in range(300):
        K = int(rng.integers(2, 8))
        attrs = np.stack([rng.integers(0, 4, K), rng.integers(0, 3, K), rng.integers(0, 2, K)], 1)
        cent = rng.uniform(0, 64, (K, 2))
        vis = rng.integers(0, 40, K)
        for mode in ("train", "ood"):
            toks, ans, kinds, flags = qa.make_questions(rng, attrs, cent, vis, 4, mode)
            for t, a_, f in zip(toks, ans, flags):
                if a_ < 0:
                    continue
                got, o = answer_scene_question(meta, t, attrs, cent, vis)
                assert got == a_
                assert ((attrs[o][0], attrs[o][1]) in qa.heldout) == (mode == "ood")


def replay_boxes(words):
    """Independent re-simulation of a boxes story from its words."""
    where, i = {}, 0
    stmt = []
    for w in words:
        stmt.append(w)
        if w in (".", "?"):
            s, stmt = stmt, []
            if s[0] == "box" and s[3] == "empty":
                continue
            if s[0] == "the":
                b = int(s[-2])
                for k, x in enumerate(s):
                    if x == "the":
                        where[s[k + 1]] = b
            elif s[0] == "move":
                assert where[s[2]] == int(s[5])
                where[s[2]] = int(s[8])
            elif s[0] == "remove":
                assert where.pop(s[2]) == int(s[5])
            elif s[0] == "put":
                assert s[2] not in where
                where[s[2]] = int(s[5])
            elif s[0] == "where":
                return f"box {where[s[3]]}" if s[3] in where else "nowhere"
            elif s[0] == "what":
                inside = [o for o, b in where.items() if b == int(s[4])]
                assert len(inside) <= 1
                return inside[0] if inside else "nothing"
    raise AssertionError("no question")


def test_boxes_answers_replay(tmp_path):
    cfg = dict(boxes.DEFAULTS, sizes=dict(train=400, val=50, test=50, test_ood=200))
    boxes.generate(str(tmp_path), cfg)
    for split in ("train", "test_ood"):
        z = np.load(tmp_path / f"{split}.npz")
        for i in range(len(z["text"])):
            words = [boxes.WORDS[t] for t in z["text"][i] if t]
            assert replay_boxes(words) == boxes.ANSWERS[z["answers"][i, 0]]
        lo, hi = cfg["ops"][split]
        assert z["n_ops"].min() >= lo and z["n_ops"].max() <= hi


def make_mock_clevrtex(root, n_full=60, n_extra=8, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    mats = ["Bricks", "Wood", "Marble", "Rust", "Fabric", "Leather", "Tiles", "Stone"]
    for variant, n in (("full", n_full), ("outd", n_extra), ("camo", n_extra)):
        for i in range(n):
            d = os.path.join(root, f"clevrtex_{variant}", str(i // 25))
            os.makedirs(d, exist_ok=True)
            H, W = 240, 320
            img = np.zeros((H, W, 4), np.uint8)
            img[..., 3] = 255
            msk = np.zeros((H, W), np.uint8)
            objs = []
            for k in range(1, int(rng.integers(3, 7)) + 1):
                cx, cy, r = rng.integers(70, 250), rng.integers(50, 190), int(rng.integers(12, 22))
                yy, xx = np.mgrid[0:H, 0:W]
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
                msk[m], img[m, :3] = k, rng.integers(0, 255, 3)
                objs.append(dict(shape=clevrtex.SHAPES[rng.integers(4)], size=clevrtex.SIZES[rng.integers(3)],
                                 material=mats[rng.integers(len(mats))] if variant != "outd" else "Novel",
                                 index=k))
            stem = os.path.join(d, f"CLEVRTEX_{variant}_{i:06d}")
            Image.fromarray(img).save(stem + ".png")
            Image.fromarray(msk).save(stem + "_flat.png")
            json.dump(dict(objects=objs), open(stem + ".json", "w"))


def test_clevrtex_conversion(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_mock_clevrtex(str(raw))
    clevrtex.convert(str(raw), str(out), workers=2)
    meta = json.load(open(out / "meta.json"))
    assert set(meta["splits"]) >= {"train", "val", "test", "outd", "camo"}
    mats = meta["attributes"]["material"]
    held = {tuple(p) for p in meta["heldout"]}
    assert len(held) == len(mats) and {m for _, m in held} == set(range(len(mats)))  # one shape per material
    for split, mode in (("train", "train"), ("val", "iid"), ("test", "iid"), ("test_ood", "ood")):
        p = out / f"{split}.npz"
        if p.exists():
            check_scene_split(meta, np.load(p), mode)
    z = np.load(out / "train.npz")
    assert z["images"].shape[1:] == (128, 128, 3) and z["masks"].max() <= 10
    # the mask/attribute mapping: every visible object id has attributes
    for i in range(len(z["images"])):
        ids = set(np.unique(z["masks"][i])) - {0}
        assert all(z["attrs"][i, k - 1, 0] >= 0 for k in ids)


def test_movi_prepare_and_extract(tmp_path):
    rng = np.random.default_rng(0)
    keys = ["color_label", "material_label", "shape_label", "size_label"]
    names = {"color": ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"],
             "material": ["metal", "rubber"], "shape": ["cube", "cylinder", "sphere"], "size": ["small", "large"]}

    def example():
        T, H = 24, 128
        n = int(rng.integers(3, 9))
        seg = np.zeros((T, H, H, 1), np.uint8)
        yy, xx = np.mgrid[0:H, 0:H]
        for k in range(1, n + 1):
            cx, cy, r = rng.integers(15, 113), rng.integers(15, 113), int(rng.integers(6, 14))
            seg[:, (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, 0] = k
        return {"video": rng.integers(0, 255, (T, H, H, 3), dtype=np.uint8), "segmentations": seg,
                "forward_flow": rng.integers(0, 65535, (T, H, H, 2)).astype(np.uint16),
                "metadata": {"forward_flow_range": np.array([-3.0, 4.0], np.float32)},
                "instances": {k: rng.integers(0, len(names[k[:-6]]), n) for k in keys}}

    ex = example()
    r = movi.extract(ex, keys, 0, 8, 32)
    flow = ex["forward_flow"][:8].astype(np.float32) / 65535 * 7 - 3
    assert np.allclose(r["flow"][1, 2].astype(np.float32), flow[:, 4:8, 8:12].mean((1, 2)), atol=1e-2)
    assert (r["mask"] == ex["segmentations"][0, ..., 0]).all()
    for split, n in (("train", 60), ("validation", 10)):
        movi.save_raw([movi.extract(example(), keys, 0, 8, 32) for _ in range(n)], str(tmp_path / f"raw_{split}.npz"))
    json.dump(dict(name="movi_a", labels=names, label_keys=keys, frame=0, flow_steps=8, grid=32, res=128),
              open(tmp_path / "raw_meta.json", "w"))
    movi.prepare(str(tmp_path))
    meta = json.load(open(tmp_path / "meta.json"))
    assert meta["binding_pair"] == ["shape", "color"] and len(meta["heldout"]) == 6
    for split, mode in (("train", "train"), ("val", "iid"), ("test", "iid"), ("test_ood", "ood")):
        check_scene_split(meta, np.load(tmp_path / f"{split}.npz"), mode)


def test_movi_label_files_and_shard_names():
    feats = {"featuresDict": {"features": {"instances": {"sequence": {"feature": {"featuresDict": {"features": {
        "shape_label": {"classLabel": {"numClasses": "3"}}, "positions": {"tensor": {}}}}}}},
        "video": {"video": {}}}}}
    assert movi.label_files(feats) == ["instances-shape_label.labels.txt"]
    info = {"name": "movi_a", "splits": [{"name": "train", "shardLengths": ["1"] * 4},
                                         {"name": "validation", "shardLengths": ["1"] * 2}]}
    names = movi.shard_names(info, max_shards=2)
    assert names == {"train": ["movi_a-train.tfrecord-00000-of-00004", "movi_a-train.tfrecord-00001-of-00004"],
                     "validation": ["movi_a-validation.tfrecord-00000-of-00002"]}


# ----------------------------------------------------------------------------- metrics

def test_ari_matches_sklearn_and_basic_cases():
    rng = np.random.default_rng(0)
    assert M.ari(np.array([0, 0, 1, 1]), np.array([5, 5, 7, 7])) == 1.0
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        pytest.skip("scikit-learn not installed")
    for _ in range(50):
        a, b = rng.integers(0, 4, 30), rng.integers(0, 5, 30)
        assert abs(M.ari(a, b) - adjusted_rand_score(a, b)) < 1e-12


def test_grouping_metrics_on_perfect_and_collapsed_frames():
    masks = torch.zeros(2, 16, 16, dtype=torch.uint8)
    masks[:, :8, :8], masks[:, 8:, 8:], masks[:, :8, 8:] = 1, 2, 3
    lab = M.patch_labels(masks, 4)
    assert lab.shape == (2, 16) and set(lab.unique().tolist()) == {0, 1, 2, 3}
    tag = (lab.clamp_min(0).float() * 2.0)  # each object its own angle
    dials = tag[:, None, :, None].expand(2, 2, 16, 3).clone()
    gyros = torch.eye(3).expand(2, 2, 16, 1, 3, 3).clone()
    g = M.tag_gaps(dials, gyros, lab)
    assert abs(g["dial_same"] - 1) < 1e-6 and g["tag_gap"] > 0.5
    assert M.fg_ari(M.frame_features(dials, gyros), lab, 4) == 1.0
    collapsed = M.tag_gaps(torch.zeros_like(dials), gyros, lab)
    assert abs(collapsed["tag_gap"]) < 1e-6
    assert abs(M.sync_order(torch.zeros_like(dials), gyros, lab)["dial_sync"] - 1) < 1e-6


# ----------------------------------------------------------------------------- model

@pytest.mark.parametrize("variant", ["A", "B", "C", "D", "E"])
def test_model_variants_forward_backward(variant):
    torch.manual_seed(0)
    fp = 16 if variant == "E" else 0
    model = FrameTransformer(variant, n_answers=10, n_heads=2, n_layers=3, img_size=32, patch=8,
                             vocab_size=20, fingerprint_dim=fp)
    x = dict(images=torch.randn(3, 3, 32, 32), q_tokens=torch.randint(0, 20, (3, 4)))
    if variant == "E":
        x["fingerprints"] = torch.randn(3, 16, 16)
    logits, internals = model(**x, return_internals=True)
    assert logits.shape == (3, 10) and len(internals["frames"]) == 3
    logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if model.n_gyro:
        g = internals["frames"][-1].gyros
        assert (g.transpose(-1, -2) @ g - torch.eye(3)).abs().max() < 1e-5


def test_text_model_and_param_counts():
    counts = {}
    for v in ("A", "B", "C", "D"):
        m = FrameTransformer(v, n_answers=33, n_heads=4, n_layers=2, kind="text", vocab_size=50, max_len=40)
        text = torch.randint(1, 50, (2, 40))
        text[0, 30:] = 0
        assert m(text=text).shape == (2, 33)
        counts[v] = sum(p.numel() for p in m.parameters())
    assert max(counts.values()) / min(counts.values()) < 1.15  # same head width; frames add little
    assert all(VARIANTS[v]["n_free"] + 16 + 2 * VARIANTS[v]["n_dial"] + 3 * VARIANTS[v]["n_gyro"] == 64
               for v in VARIANTS)


def test_compute_matched_baseline_has_at_least_d_flops():
    def make(v, L):
        return FrameTransformer(v, n_answers=10, n_heads=2, n_layers=L, img_size=32, patch=8, vocab_size=20)
    x = dict(images=torch.randn(2, 3, 32, 32), q_tokens=torch.randint(0, 20, (2, 4)))
    from model import forward_flops
    L, target = matched_layers(make, 4, x)
    assert L >= 4 and forward_flops(make("A", L), x) >= target


# ----------------------------------------------------------------------------- end to end

def test_train_and_summarize_end_to_end(shapes_dir, tmp_path):
    runs = tmp_path / "runs"
    for v in ("A", "A_prime", "B", "C", "D"):
        for seed in (0, 1):
            subprocess.run([sys.executable, "train.py", "--task", "shapes", "--variant", v, "--seed", str(seed),
                            "--preset", "quick", "--steps", "6", "--eval-every", "3", "--data", shapes_dir,
                            "--out", str(runs), "--device", "cpu"], cwd=ROOT, check=True, capture_output=True)
            r = json.load(open(runs / "shapes" / v / f"seed{seed}" / "results.json"))
            assert 0 <= r["final"]["test_ood"]["acc"] <= 1 and r["final"]["test_ood"]["n"] > 0
            if v in ("B", "C", "D"):
                assert "fg_ari_frames" in r["final"]["grouping"]["val"]
    out = subprocess.run([sys.executable, "summarize.py", "--runs", str(runs)], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    assert "| D |" in out and "D beats A" in out
