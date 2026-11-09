"""
Binding in language: an entity-tracking "boxes" task in the style of Kim & Schuster (2023),
generated locally (no download), so a small model can be trained from scratch.

    the key and the cup are in box 1 . box 2 is empty . ...
    move the key from box 1 to box 4 . remove the cup from box 1 . put the pen into box 2 .
    where is the key ?          -> box 4
    what is in box 2 ?          -> pen

Splits: train / val / test use 0-4 operations; test_ood uses 5-8 operations (longer chains
of rebinding than anything seen in training).
"""

import json
import os

import numpy as np

OBJECTS = ["apple", "bell", "book", "bottle", "brick", "cake", "candle", "card", "clock", "coin", "cup", "disk",
           "egg", "fan", "fork", "key", "lamp", "leaf", "map", "pen", "ring", "shoe", "sock", "watch"]
N_BOXES = 7
WORDS = ["<pad>", "the", "is", "are", "in", "box", "empty", "and", ",", ".", "move", "from", "to", "remove",
         "put", "into", "where", "what", "?"] + [str(i) for i in range(N_BOXES)] + OBJECTS
ANSWERS = [f"box {i}" for i in range(N_BOXES)] + ["nowhere"] + OBJECTS + ["nothing"]
WID = {w: i for i, w in enumerate(WORDS)}
AID = {a: i for i, a in enumerate(ANSWERS)}

DEFAULTS = dict(max_len=144, ops=dict(train=(0, 4), val=(0, 4), test=(0, 4), test_ood=(5, 8)),
                sizes=dict(train=200_000, val=5_000, test=5_000, test_ood=5_000))


def describe(contents, box):
    if not contents:
        return ["box", str(box), "is", "empty", "."]
    words = []
    for i, obj in enumerate(contents):
        if i > 0:
            words += ["and"] if i == len(contents) - 1 else [","]
        words += ["the", obj]
    return words + (["is"] if len(contents) == 1 else ["are"]) + ["in", "box", str(box), "."]


def sample_story(rng, n_ops):
    """Returns (words, answer, kind) with kind 0 = 'where is X', 1 = 'what is in box k'."""
    m = int(rng.integers(3, 7))
    objs = list(rng.choice(OBJECTS, m, replace=False))
    where = {o: int(rng.integers(N_BOXES)) for o in objs}  # object -> box
    words = []
    for b in range(N_BOXES):
        words += describe([o for o in objs if where[o] == b], b)
    mentioned, touched_objs, touched_boxes = set(objs), [], []
    for _ in range(n_ops):
        present = [o for o in where]
        absent = [o for o in OBJECTS if o not in where]
        ops = (["move"] * 5 + ["remove"] * 2 if present else []) + (["put"] * 3 if absent else [])
        op = ops[rng.integers(len(ops))]
        if op == "move":
            o = present[rng.integers(len(present))]
            src = where[o]
            dst = int(rng.choice([b for b in range(N_BOXES) if b != src]))
            words += ["move", "the", o, "from", "box", str(src), "to", "box", str(dst), "."]
            where[o] = dst
            touched_boxes += [src, dst]
        elif op == "remove":
            o = present[rng.integers(len(present))]
            words += ["remove", "the", o, "from", "box", str(where[o]), "."]
            touched_boxes.append(where.pop(o))
        else:
            o = absent[rng.integers(len(absent))]
            dst = int(rng.integers(N_BOXES))
            words += ["put", "the", o, "into", "box", str(dst), "."]
            where[o] = dst
            touched_boxes.append(dst)
            mentioned.add(o)
        touched_objs.append(o)

    contents = {b: [o for o in where if where[o] == b] for b in range(N_BOXES)}
    small_boxes = [b for b in range(N_BOXES) if len(contents[b]) <= 1]
    if rng.random() < 0.5 and small_boxes:
        pref = [b for b in small_boxes if b in touched_boxes]
        pool = pref if pref and rng.random() < 0.7 else small_boxes
        b = int(pool[rng.integers(len(pool))])
        answer = contents[b][0] if contents[b] else "nothing"
        return words + ["what", "is", "in", "box", str(b), "?"], AID[answer], 1
    pool = touched_objs if touched_objs and rng.random() < 0.7 else sorted(mentioned)
    o = pool[rng.integers(len(pool))]
    answer = f"box {where[o]}" if o in where else "nowhere"
    return words + ["where", "is", "the", o, "?"], AID[answer], 0


def generate(out_dir, cfg=DEFAULTS, seed=4321):
    os.makedirs(out_dir, exist_ok=True)
    L = cfg["max_len"]
    for si, (split, n) in enumerate(cfg["sizes"].items()):
        rng = np.random.default_rng(seed + si)
        lo, hi = cfg["ops"][split]
        text = np.zeros((n, L), np.int16)
        answers = np.zeros((n, 1), np.int16)
        kinds = np.zeros((n, 1), np.int8)
        n_ops = np.zeros(n, np.int8)
        for i in range(n):
            k = int(rng.integers(lo, hi + 1))
            words, ans, kind = sample_story(rng, k)
            assert len(words) <= L, len(words)
            text[i, :len(words)] = [WID[w] for w in words]
            answers[i, 0], kinds[i, 0], n_ops[i] = ans, kind, k
        np.savez(os.path.join(out_dir, f"{split}.npz"), text=text, answers=answers, q_kinds=kinds,
                 q_flags=np.zeros_like(kinds), n_ops=n_ops)
        print(f"  boxes/{split}: {n} stories, {lo}-{hi} operations")
    meta = dict(task="boxes", kind="text", max_len=L, vocab_size=len(WORDS), words=WORDS,
                n_answers=len(ANSWERS), answer_names=ANSWERS, splits=list(cfg["sizes"]), config=cfg)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def decode(tokens):
    return " ".join(WORDS[t] for t in tokens if t != 0)
