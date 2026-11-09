"""
Train and evaluate one model.

    python train.py --task shapes --variant D --seed 0                  # full run
    python train.py --task shapes --variant D --seed 0 --preset quick   # ~1 minute smoke test

tasks:    shapes | clevrtex | movi_a | boxes          (prepare data first: python download_data.py)
variants: A | A_prime | B | C | D | E | K             (see model.py; E is for movi_a only)
ablations of D: D_nosync | D_nomoves | D_frozen

Writes runs/<task>/<variant>/seed<seed>/
    metrics.jsonl   one line per evaluation (accuracy, tag gap, FG-ARI, sync) for learning curves
    results.json    config, size, compute and final metrics, read by summarize.py
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import metrics as M
from model import FrameTransformer, count_params, forward_flops, matched_layers, param_groups
from tasks.qa import FLAG_OOD, FLAG_PAIR

TASKS = {
    "shapes":   dict(data="data/shapes", img_size=64, patch=8, grouping=["val"]),
    "clevrtex": dict(data="data/clevrtex", img_size=128, patch=8, grouping=["val", "outd", "camo"]),
    "movi_a":   dict(data="data/movi_a", img_size=64, patch=8, grouping=["val"], fingerprints=True),
    "boxes":    dict(data="data/boxes", grouping=[]),
}
PRESETS = {
    "full": {
        "shapes":   dict(heads=4, layers=6, batch=256, steps=30_000, lr=1e-3, warmup=1_000, eval_every=2_500),
        "clevrtex": dict(heads=4, layers=6, batch=128, steps=40_000, lr=5e-4, warmup=2_000, eval_every=4_000),
        "movi_a":   dict(heads=4, layers=6, batch=128, steps=20_000, lr=5e-4, warmup=1_000, eval_every=2_000),
        "boxes":    dict(heads=4, layers=6, batch=256, steps=30_000, lr=5e-4, warmup=1_000, eval_every=2_500),
    },
    "quick": {t: dict(heads=2, layers=2, batch=32, steps=60, lr=1e-3, warmup=10, eval_every=30) for t in TASKS},
}
COMMON = dict(weight_decay=0.05, grad_clip=1.0, eval_questions=4096, final_eval_questions=0,
              grouping_images=256, final_grouping_images=512)
QUICK_COMMON = dict(eval_questions=256, final_eval_questions=256, grouping_images=32, final_grouping_images=32)


# ----------------------------------------------------------------------------- data

class Split:
    """One .npz split held on the device, plus the list of (example, question) pairs."""

    def __init__(self, path, device):
        z = np.load(path)
        self.device = device
        put = lambda k: torch.from_numpy(z[k]).to(device) if k in z.files else None
        self.images, self.masks, self.text = put("images"), put("masks"), put("text")
        self.flow = put("flow")
        self.n_objects = z["n_objects"] if "n_objects" in z.files else None
        self.q_tokens = put("q_tokens")
        self.answers, self.kinds, self.flags = put("answers").long(), put("q_kinds").long(), put("q_flags").long()
        ex, q = np.nonzero(z["answers"] >= 0)
        self.pairs = torch.from_numpy(np.stack((ex, q), 1)).to(device)
        self.n = len(z["answers"])


class Data:
    def __init__(self, task, device, img_size=None, data_dir=None):
        cfg = TASKS[task]
        data_dir = data_dir or cfg["data"]
        with open(os.path.join(data_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.kind = self.meta["kind"]
        self.splits = {}
        for s in sorted(set(["train", "val", "test", "test_ood"] + cfg["grouping"])):
            p = os.path.join(data_dir, f"{s}.npz")
            if os.path.exists(p):
                self.splits[s] = Split(p, device)
        self.img_size = img_size or cfg.get("img_size")
        self.patch = cfg.get("patch")
        self.use_fp = cfg.get("fingerprints", False)
        if self.use_fp:
            fl = self.meta["flow"]
            self.fp_dim, self.fp_std = 2 * fl["steps"], fl["std"]
        else:
            self.fp_dim = 0

    def inputs(self, split, ex, q=None):
        """Model inputs for examples ex (and question slots q) of a split."""
        s = self.splits[split]
        if self.kind == "text":
            return dict(text=s.text[ex].long())
        img = s.images[ex].permute(0, 3, 1, 2).float().div(255.0).sub(0.5).div(0.25)
        if img.shape[-1] != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size), mode="area")
        q = torch.zeros_like(ex) if q is None else q
        out = dict(images=img, q_tokens=s.q_tokens[ex, q].long())
        if self.use_fp:
            g = self.img_size // self.patch
            fl = s.flow[ex].float()  # (B, grid, grid, steps, 2)
            B, grid = fl.shape[:2]
            fl = fl.reshape(B, grid, grid, -1).permute(0, 3, 1, 2)
            fl = F.adaptive_avg_pool2d(fl, g).flatten(2).transpose(1, 2) / self.fp_std  # (B, g*g, 2*steps)
            out["fingerprints"] = fl
        return out

    def masks(self, split, ex):
        m = self.splits[split].masks[ex]
        if m.shape[-1] != self.img_size:
            m = F.interpolate(m[:, None].float(), size=(self.img_size, self.img_size), mode="nearest")[:, 0]
        return m.long()


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, data, split, batch, cap, amp):
    s = data.splits[split]
    pairs = s.pairs if not cap else s.pairs[:cap]
    stats = {}

    def add(name, ok):
        c, n = stats.get(name, (0, 0))
        stats[name] = (c + int(ok.sum()), n + int(ok.numel()))

    for i in range(0, len(pairs), batch):
        ex, q = pairs[i:i + batch, 0], pairs[i:i + batch, 1]
        with amp:
            logits = model(**data.inputs(split, ex, q))
        ok = logits.argmax(-1) == s.answers[ex, q]
        flags, kinds = s.flags[ex, q], s.kinds[ex, q]
        add("acc", ok)
        if (flags & FLAG_PAIR).any():
            add("acc_pair", ok[(flags & FLAG_PAIR) > 0])
        if (flags & FLAG_OOD).any():
            add("acc_ood", ok[(flags & FLAG_OOD) > 0])
        for k, name in ((0, "acc_attr"), (1, "acc_rel")):
            if (kinds == k).any():
                add(name, ok[kinds == k])
    out = {k: c / max(n, 1) for k, (c, n) in stats.items()}
    out["n"] = stats.get("acc", (0, 0))[1]
    return out


@torch.no_grad()
def grouping(model, data, split, n_images, amp, chunk=64):
    s = data.splits[split]
    n = min(n_images, s.n)
    k = int(s.n_objects[:n].max()) + 1
    feats_f, feats_c, labels, gaps, syncs, layer_gaps = [], [], [], [], [], []
    for i in range(0, n, chunk):
        ex = torch.arange(i, min(n, i + chunk), device=s.device)
        with amp:
            _, internals = model(**data.inputs(split, ex), return_internals=True)
        P = model.n_prefix  # metrics run on the CPU in float32
        lab = M.patch_labels(data.masks(split, ex).cpu(), data.patch)
        labels.append(lab)
        feats_c.append(F.normalize(internals["hidden"].float().cpu(), dim=-1))
        if model.has_frames:
            frames = [(d[:, :, P:].float().cpu(), g[:, :, P:].float().cpu()) for d, g in internals["frames"]]
            dials, gyros = frames[-1]
            feats_f.append(M.frame_features(dials, gyros))
            gaps.append(M.tag_gaps(dials, gyros, lab))
            syncs.append(M.sync_order(dials, gyros, lab))
            layer_gaps.append([M.tag_gaps(d, g, lab) for d, g in frames])
    labels = torch.cat(labels)
    out = dict(images=n, k=k, fg_ari_content=M.fg_ari(torch.cat(feats_c), labels, k))
    if feats_f:
        out["fg_ari_frames"] = M.fg_ari(torch.cat(feats_f), labels, k)
        for key in gaps[0]:
            out[key] = float(np.mean([g[key] for g in gaps]))
        for key in syncs[0]:
            out[key] = float(np.mean([g[key] for g in syncs]))
        for key in ("tag_gap", "gyro_gap"):  # how the grouping sharpens layer by layer
            if key in gaps[0]:
                out[f"{key}_per_layer"] = np.mean([[lg[key] for lg in img] for img in layer_gaps], 0).tolist()
    return out


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="shapes", choices=list(TASKS))
    p.add_argument("--variant", default="D", choices=["A", "A_prime", "B", "C", "D", "E", "K", "D_nosync", "D_nomoves", "D_frozen"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preset", default="full", choices=["full", "quick"])
    for name, typ in (("steps", int), ("batch", int), ("lr", float), ("layers", int), ("heads", int),
                      ("eval-every", int), ("img-size", int)):
        p.add_argument(f"--{name}", type=typ, default=None, help="override the preset")
    p.add_argument("--frame-init-std", type=float, default=1.0, help="spread of the starting frame angles (radians)")
    p.add_argument("--data", default=None, help="data folder (default: data/<task>)")
    p.add_argument("--out", default="runs")
    p.add_argument("--device", default=None)
    p.add_argument("--compile", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    cfg = dict(COMMON, **PRESETS[args.preset][args.task], frame_init_std=args.frame_init_std)
    if args.preset == "quick":
        cfg.update(QUICK_COMMON)
    for k in ("steps", "batch", "lr", "layers", "heads", "eval_every"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.variant == "E" and args.task != "movi_a":
        raise SystemExit("variant E uses motion fingerprints: only --task movi_a has them")

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    data = Data(args.task, device, args.img_size, args.data)
    meta = data.meta
    run_dir = os.path.join(args.out, args.task, args.variant, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    def build(variant, layers):
        kw = dict(kind=data.kind, n_answers=meta["n_answers"], n_heads=cfg["heads"], n_layers=layers,
                  vocab_size=meta["vocab_size"], frame_init_std=cfg["frame_init_std"])
        if data.kind == "image":
            kw.update(img_size=data.img_size, patch=data.patch, n_question=4, fingerprint_dim=data.fp_dim)
        else:
            kw.update(max_len=meta["max_len"])
        return FrameTransformer(variant, **kw)

    layers, target_flops = cfg["layers"], None
    probe = data.inputs("train", torch.arange(2, device=device), None)
    if args.variant == "A_prime":  # deepen A until its compute matches D
        probe_cpu = {k: v.cpu() for k, v in probe.items()}
        layers, target_flops = matched_layers(lambda v, L: build(v, L), cfg["layers"], probe_cpu)
        print(f"A_prime: {layers} layers to match variant D's {target_flops / 1e9:.3f} GFLOPs/sample")
    model = build(args.variant, layers).to(device)
    flops = forward_flops(model, probe)
    n_params = count_params(model)
    print(f"{args.task} / {args.variant} / seed {args.seed}: {n_params / 1e6:.2f}M params, {layers} layers, "
          f"{flops / 1e9:.3f} GFLOPs/sample forward, device {device}")

    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=cfg["lr"], betas=(0.9, 0.98),
                            **({"fused": True} if device == "cuda" else {}))
    run = model
    if args.compile == "on" or (args.compile == "auto" and device == "cuda"):
        try:
            run = torch.compile(model)
        except Exception as e:  # pragma: no cover
            print(f"torch.compile unavailable ({e}); running eagerly")

    def lr_at(step):
        if step < cfg["warmup"]:
            return cfg["lr"] * (step + 1) / cfg["warmup"]
        t = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
        return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    train = data.splits["train"]
    gen = torch.Generator().manual_seed(args.seed)
    log = open(os.path.join(run_dir, "metrics.jsonl"), "w")
    t0, seen, losses = time.time(), 0, []

    def eval_all(final):
        model.eval()
        cap = cfg["final_eval_questions"] if final else cfg["eval_questions"]
        res = {s: evaluate(model, data, s, 512, cap, amp) for s in ("val", "test", "test_ood") if s in data.splits}
        if not final:
            res.pop("test", None)
        n_img = cfg["final_grouping_images"] if final else cfg["grouping_images"]
        gsplits = TASKS[args.task]["grouping"] if final else TASKS[args.task]["grouping"][:1]
        res["grouping"] = {s: grouping(model, data, s, n_img, amp) for s in gsplits if s in data.splits}
        model.train()
        return res

    model.train()
    for step in range(cfg["steps"] + 1):
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            res = eval_all(final=False)
            row = dict(step=step, train_loss=float(np.mean(losses)) if losses else None,
                       time=time.time() - t0, **res)
            log.write(json.dumps(row) + "\n")
            log.flush()
            g = res.get("grouping", {}).get("val", {})
            print(f"step {step:6d} | loss {row['train_loss'] or float('nan'):.4f} | "
                  f"val {res['val']['acc']:.3f} | ood {res.get('test_ood', {}).get('acc', float('nan')):.3f} | "
                  f"tag gap {g.get('tag_gap', float('nan')):.3f} | ari(frames) {g.get('fg_ari_frames', float('nan')):.3f}"
                  f" | {time.time() - t0:.0f}s")
            losses = []
        if step == cfg["steps"]:
            break
        pick = torch.randint(len(train.pairs), (cfg["batch"],), generator=gen).to(device)
        ex, q = train.pairs[pick, 0], train.pairs[pick, 1]
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        with amp:
            logits = run(**data.inputs("train", ex, q))
        loss = F.cross_entropy(logits.float(), train.answers[ex, q])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        losses.append(loss.item())
        seen += cfg["batch"]
        if step == 20:
            t_warm = time.time()
            seen_warm = seen
        if step > 20 and step % 200 == 0:
            rate = (seen - seen_warm) / (time.time() - t_warm)
            eta = (cfg["steps"] - step) * cfg["batch"] / rate
            print(f"  step {step} loss {np.mean(losses[-200:]):.4f}  {rate:,.0f} samples/s  eta {eta / 60:.1f} min")

    train_time = time.time() - t0
    final = eval_all(final=True)
    log.close()
    results = dict(task=args.task, variant=args.variant, seed=args.seed, preset=args.preset, config=cfg,
                   layers=layers, n_params=n_params, flops_per_sample=flops, matched_target_flops=target_flops,
                   train_time_s=train_time, samples_per_s=seen / train_time, device=device, final=final)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    print(json.dumps({k: final[k] for k in final if k != "grouping"}, indent=1))
    print("grouping:", json.dumps(final.get("grouping", {}), indent=1))
    print(f"saved {run_dir}/results.json")


if __name__ == "__main__":
    main()
