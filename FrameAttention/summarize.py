"""
Collect runs/*/*/seed*/results.json into one table per task (mean ± std over seeds) and
check the pass/fail criteria that were fixed before running.

    python summarize.py                 # all tasks
    python summarize.py --task shapes   # one task
Writes runs/summary.md as well as printing it.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

ORDER = ["A", "A_prime", "B", "C", "D", "E", "K", "D_nosync", "D_nomoves", "D_frozen", "D_random"]
COLUMNS = [  # (label, path into results["final"], scale)
    ("iid acc", ("test", "acc"), 100),
    ("iid binding-pair acc", ("test", "acc_pair"), 100),
    ("OOD acc", ("test_ood", "acc"), 100),
    ("tag gap", ("grouping", "val", "tag_gap"), 1),
    ("FG-ARI frames", ("grouping", "val", "fg_ari_frames"), 100),
    ("FG-ARI content", ("grouping", "val", "fg_ari_content"), 100),
    ("dial sync", ("grouping", "val", "dial_sync"), 1),
]


def get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def stats(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd, len(xs)


def fmt(s, scale):
    if s is None:
        return "–"
    m, sd, _ = s
    return f"{m * scale:.1f} ± {sd * scale:.1f}" if scale == 100 else f"{m:.3f} ± {sd:.3f}"


def criteria(rows, metric=("test_ood", "acc")):
    """The pre-registered checks, on OOD accuracy."""
    def s(v):
        return stats([get(r["final"], metric) for r in rows.get(v, [])])
    A, Ap, B, C, D, K = (s(v) for v in ("A", "A_prime", "B", "C", "D", "K"))
    lines = []

    def beats(x, y, name_x, name_y):
        if not x or not y:
            return f"- {name_x} vs {name_y}: not enough runs"
        diff, pooled = x[0] - y[0], math.sqrt((x[1] ** 2 + y[1] ** 2) / 2)
        verdict = "PASS" if diff > 2 * pooled and x[2] >= 2 and y[2] >= 2 else "no"
        return (f"- {name_x} beats {name_y} by more than 2x the seed spread: **{verdict}** "
                f"(difference {100 * diff:+.1f} points, pooled std {100 * pooled:.1f}, seeds {x[2]}/{y[2]})")

    lines.append(beats(D, A, "D", "A (RoPE baseline)"))
    lines.append(beats(D, Ap, "D", "A_prime (compute-matched)"))
    lines.append(beats(D, K, "D", "K (published KoPE)"))
    if B and D:
        pooled = math.sqrt((B[1] ** 2 + D[1] ** 2) / 2)
        close = abs(D[0] - B[0]) <= pooled
        lines.append(f"- B vs D: {'within one std: gyros are not earning their cost' if close else 'different'}"
                     f" ({100 * (D[0] - B[0]):+.1f} points)")
    if C and D:
        lines.append(f"- C vs D: {100 * (D[0] - C[0]):+.1f} points (what the dials add on top of gyros)")
    for v, off in (("D_nosync", "sync off"), ("D_nomoves", "moves off"), ("D_frozen", "all frame updates off"),
                   ("D_random", "random fixed frames")):
        if rows.get(v):  # a PASS means the switched-off force was contributing
            lines.append(beats(D, s(v), "D", f"{v} ({off})"))
    weak = [v for v in ORDER if rows.get(v) and (stats([get(r["final"], ("test", "acc")) for r in rows[v]]) or (1,))[0] < 0.9]
    if weak:
        lines.append(f"- **Below 90% iid, so comparisons with these are confounded by training:** {', '.join(weak)}")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--task", default=None)
    args = p.parse_args()
    found = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(args.runs, "*", "*", "seed*", "results.json")):
        with open(path) as f:
            r = json.load(f)
        if args.task in (None, r["task"]):
            found[r["task"]][r["variant"]].append(r)
    if not found:
        raise SystemExit(f"no results under {args.runs}/")
    out = []
    for task, rows in sorted(found.items()):
        out.append(f"## {task}\n")
        head = ["variant", "seeds", "params (M)", "layers", "GFLOPs/sample"] + [c[0] for c in COLUMNS]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
        for v in [v for v in ORDER if v in rows]:
            rs = rows[v]
            cells = [v, str(len(rs)), f"{rs[0]['n_params'] / 1e6:.2f}", str(rs[0]["layers"]),
                     f"{rs[0]['flops_per_sample'] / 1e9:.3f}"]
            cells += [fmt(stats([get(r["final"], path) for r in rs]), scale) for _, path, scale in COLUMNS]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        if task in ("shapes", "clevrtex", "movi_a", "boxes"):
            out.append("Pass/fail (OOD accuracy):")
            out += criteria(rows)
            if task == "movi_a" and "E" in rows and "D" in rows:
                e = stats([get(r["final"], ("test_ood", "acc")) for r in rows["E"]])
                d = stats([get(r["final"], ("test_ood", "acc")) for r in rows["D"]])
                out.append(f"- E vs D (common fate): {100 * (e[0] - d[0]):+.1f} points OOD")
            out.append("")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.runs, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
