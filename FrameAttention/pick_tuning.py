"""
Apply the tuning rules in PREDICTIONS.md to the runs from run_tune.sh.

    python pick_tuning.py --steps 30000            # pick each variant's learning rate, check the step budget
    python pick_tuning.py --steps 30000 --reveal   # then show held-out results for the chosen settings only

Without --reveal this reads iid validation accuracy only and never prints OOD numbers, so the choice
can't be influenced by them. Pick the settings first and reveal afterwards.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

ORDER = ["A", "A_prime", "K", "B", "C", "D", "D_nosync", "D_nomoves", "D_frozen", "D_random"]
LRS = ["3e-4", "1e-3", "3e-3"]
IID_FLOOR = 0.90    # a baseline below this is confounded by training
PLATEAU = 0.01      # the budget is enough if every variant gains less than this over the last 10%


def late_gain(curve, steps):
    """Gain in iid validation accuracy over the last 10% of training."""
    earlier = [row for row in curve if row["step"] <= 0.9 * steps]
    return curve[-1]["val"]["acc"] - earlier[-1]["val"]["acc"]


def load_runs(root):
    """root/lr<lr>/<task>/<variant>/seed0 -> {variant: {lr: run}}"""
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(root, "lr*", "*", "*", "seed0", "results.json")):
        with open(path) as f:
            result = json.load(f)
        with open(os.path.join(os.path.dirname(path), "metrics.jsonl")) as f:
            curve = [json.loads(line) for line in f if line.strip()]
        lr = path.split(os.sep)[-5][len("lr"):]
        runs[result["variant"]][lr] = dict(val=result["final"]["val"]["acc"],
                                           gain=late_gain(curve, result["config"]["steps"]), result=result)
    return runs


def choose(by_lr, lrs):
    """The learning rate with the best final iid validation accuracy (earlier in the sweep wins ties)."""
    return max((lr for lr in lrs if lr in by_lr), key=lambda lr: by_lr[lr]["val"])


def report(runs, lrs, steps):
    lines = [f"## Tuning at {steps} steps (seed 0, iid validation only)", "",
             "| variant | lr | val acc | gain over last 10% | chosen |", "|---|---|---|---|---|"]
    chosen, incomplete = {}, []
    for v in [v for v in ORDER if v in runs]:
        best = choose(runs[v], lrs)
        chosen[v] = best
        if any(lr not in runs[v] for lr in lrs):
            incomplete.append(v)
        for lr in [lr for lr in lrs if lr in runs[v]]:
            r = runs[v][lr]
            lines.append(f"| {v} | {lr} | {100 * r['val']:.1f} | {100 * r['gain']:+.1f} | {'*' if lr == best else ''} |")
    lines.append("")
    if incomplete:
        lines.append(f"- Sweep incomplete for: {', '.join(incomplete)}. The step budget can't be decided yet.")
    still_rising = [v for v in chosen if runs[v][chosen[v]]["gain"] >= PLATEAU]
    weak = [v for v in chosen if runs[v][chosen[v]]["val"] < IID_FLOOR]
    if not incomplete:
        verdict = f"still rising for {', '.join(still_rising)}" if still_rising else "every variant has plateaued"
        lines.append(f"- Step budget of {steps}: {'not enough' if still_rising else 'enough'} ({verdict}).")
    if weak:
        lines.append(f"- Below {100 * IID_FLOOR:.0f}% iid at their best learning rate, so confounded: {', '.join(weak)}.")
    return lines, chosen


def reveal(runs, chosen):
    lines = ["", "## Held-out results for the chosen settings (seed 0, exploratory)", "",
             "| variant | lr | iid acc | iid binding-pair acc | OOD acc | OOD gap | status |", "|---|---|---|---|---|---|---|"]
    for v, lr in chosen.items():
        final = runs[v][lr]["result"]["final"]
        pair, ood = final["test"]["acc_pair"], final["test_ood"]["acc"]
        status = "clean" if runs[v][lr]["val"] >= IID_FLOOR else "confounded"
        lines.append(f"| {v} | {lr} | {100 * final['test']['acc']:.1f} | {100 * pair:.1f} | {100 * ood:.1f} "
                     f"| {100 * (pair - ood):.1f} | {status} |")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lrs", nargs="+", default=LRS)
    p.add_argument("--root", default="runs_tune")
    p.add_argument("--reveal", action="store_true", help="also show held-out results for the chosen settings")
    args = p.parse_args()
    runs = load_runs(os.path.join(args.root, f"steps{args.steps}"))
    if not runs:
        raise SystemExit(f"no finished runs under {args.root}/steps{args.steps}/")
    lines, chosen = report(runs, args.lrs, args.steps)
    if args.reveal:
        lines += reveal(runs, chosen)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
