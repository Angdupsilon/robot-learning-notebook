"""
Question generator shared by the scene tasks (shapes, CLEVRTex, MOVi).

A scene is a set of objects. Each object has one value per attribute (e.g. shape, color),
a centroid (x, y) in pixels, and a count of visible pixels. Questions are

    kind 0 (attribute): "what <target> is the <description> object?"
    kind 1 (relation):  "what <target> is the object <left of|right of|above|below> the <description> object?"

where <description> is one or two attribute values that pick out exactly one object.
A question is encoded as 4 tokens: [target, value, value or PAD, relation].

One pair of attributes is the "binding pair" (e.g. shape and color), and some value
combinations of that pair are held out:
    train / iid questions never involve an object with a held-out combination
    ood questions ask only about held-out objects, using exactly the binding pair
        ("what color is the triangle?" when this triangle's color never appeared on a triangle)
"""

import numpy as np

RELATIONS = ["none", "left", "right", "above", "below"]
FLAG_PAIR = 1  # target is one attribute of the binding pair and the description uses the other
FLAG_OOD = 2   # the answered object has a held-out combination


class SceneQA:
    def __init__(self, attributes, binding_pair, heldout, min_visible, margin):
        """attributes: dict attribute name -> list of value names
        binding_pair: two attribute names
        heldout: iterable of (value index of pair[0], value index of pair[1])"""
        self.attr_names = list(attributes)
        self.values = [list(v) for v in attributes.values()]
        self.offsets = np.cumsum([0] + [len(v) for v in self.values])[:-1]
        self.n_answers = int(sum(len(v) for v in self.values))
        self.pair = tuple(self.attr_names.index(n) for n in binding_pair)
        self.heldout = {(int(x), int(y)) for x, y in heldout}
        self.min_visible, self.margin = min_visible, margin
        A = len(self.attr_names)
        self.tok_target0 = 1                       # 0 is PAD
        self.tok_value0 = 1 + A
        self.tok_rel0 = 1 + A + self.n_answers
        self.vocab_size = self.tok_rel0 + len(RELATIONS)

    # ---- names, for meta.json and for reading questions back
    def answer_names(self):
        return [f"{n}={v}" for n, vals in zip(self.attr_names, self.values) for v in vals]

    def token_names(self):
        return (["<pad>"] + [f"what-{n}?" for n in self.attr_names] + self.answer_names()
                + [f"rel={r}" for r in RELATIONS])

    def meta(self):
        return dict(attributes={n: v for n, v in zip(self.attr_names, self.values)},
                    binding_pair=[self.attr_names[i] for i in self.pair],
                    heldout=sorted(self.heldout), n_answers=self.n_answers, vocab_size=self.vocab_size,
                    answer_names=self.answer_names(), token_names=self.token_names(),
                    min_visible=self.min_visible, margin=self.margin)

    # ---- helpers
    def answer_id(self, attr, value):
        return int(self.offsets[attr] + value)

    def is_heldout(self, obj_attrs):
        return (int(obj_attrs[self.pair[0]]), int(obj_attrs[self.pair[1]])) in self.heldout

    def encode(self, target, desc, obj_attrs, rel=0):
        toks = [self.tok_target0 + target, 0, 0, self.tok_rel0 + rel]
        for i, d in enumerate(desc):
            toks[1 + i] = self.tok_value0 + int(self.offsets[d]) + int(obj_attrs[d])
        return toks

    def descriptions(self, attrs, present, o, exclude=None, must_include=None):
        """Minimal descriptions (1 or 2 attributes, not `exclude`) that match only object o."""
        others = [j for j in present if j != o]
        cand = [d for d in range(attrs.shape[1]) if d != exclude]
        singles = [(d,) for d in cand if all(attrs[j, d] != attrs[o, d] for j in others)]
        unique1 = {s[0] for s in singles}
        pairs = []
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                d1, d2 = cand[i], cand[j]
                if d1 in unique1 or d2 in unique1:
                    continue  # one of them alone already works: the pair would be redundant
                if all((attrs[x, d1], attrs[x, d2]) != (attrs[o, d1], attrs[o, d2]) for x in others):
                    pairs.append((d1, d2))
        out = singles + pairs
        if must_include is not None:
            out = [d for d in out if must_include in d]
        return out

    def relation_target(self, cent, candidates, r, rel):
        """Nearest object to r in direction rel (1 left, 2 right, 3 above, 4 below), or None
        if there is none or the nearest two are too close to call."""
        x, y = cent[r]
        m = self.margin
        test = {1: lambda j: cent[j, 0] < x - m, 2: lambda j: cent[j, 0] > x + m,
                3: lambda j: cent[j, 1] < y - m, 4: lambda j: cent[j, 1] > y + m}[rel]
        hits = sorted((float(np.hypot(*(cent[j] - cent[r]))), j) for j in candidates if j != r and test(j))
        if not hits or (len(hits) > 1 and hits[1][0] - hits[0][0] < m / 2):
            return None
        return hits[0][1]

    # ---- main entry
    def make_questions(self, rng, attrs, cent, vis, n_q, mode, p_relation=0.3, max_tries=60):
        """attrs (K, A) ints, cent (K, 2) pixels, vis (K,) visible pixels.
        mode: 'train' or 'iid' (never touch held-out objects) or 'ood' (only binding-pair
        questions about held-out objects).
        Returns tokens (n_q, 4) int16, answers (n_q,) int16 (-1 = no question), kinds, flags."""
        K, A = attrs.shape
        present = [j for j in range(K) if vis[j] > 0]
        visible = [j for j in present if vis[j] >= self.min_visible]
        held = {j for j in present if self.is_heldout(attrs[j])}
        tokens = np.zeros((n_q, 4), np.int16)
        answers = np.full(n_q, -1, np.int16)
        kinds = np.zeros(n_q, np.int8)
        flags = np.zeros(n_q, np.int8)
        seen, n = set(), 0
        for _ in range(max_tries):
            if n == n_q:
                break
            if mode == "ood":
                cands = [j for j in visible if j in held]
                if not cands:
                    break
                o = cands[rng.integers(len(cands))]
                t = self.pair[rng.integers(2)]
                other = self.pair[1] if t == self.pair[0] else self.pair[0]
                descs = self.descriptions(attrs, present, o, exclude=t, must_include=other)
                if not descs:
                    continue
                d = descs[rng.integers(len(descs))]
                q = (tuple(self.encode(t, d, attrs[o])), self.answer_id(t, attrs[o, t]), 0, FLAG_PAIR | FLAG_OOD)
            else:
                pool = [j for j in visible if j not in held]
                if not pool:
                    break
                t = int(rng.integers(A))
                if rng.random() < p_relation:
                    r = pool[rng.integers(len(pool))]
                    rel = int(rng.integers(1, 5))
                    o = self.relation_target(cent, visible, r, rel)
                    if o is None or o in held:
                        continue
                    descs = self.descriptions(attrs, present, r)
                    if not descs:
                        continue
                    d = descs[rng.integers(len(descs))]
                    q = (tuple(self.encode(t, d, attrs[r], rel)), self.answer_id(t, attrs[o, t]), 1, 0)
                else:
                    o = pool[rng.integers(len(pool))]
                    descs = self.descriptions(attrs, present, o, exclude=t)
                    if not descs:
                        continue
                    d = descs[rng.integers(len(descs))]
                    other = self.pair[1] if t == self.pair[0] else self.pair[0]
                    flag = FLAG_PAIR if (t in self.pair and other in d) else 0
                    q = (tuple(self.encode(t, d, attrs[o])), self.answer_id(t, attrs[o, t]), 0, flag)
            if q[0] in seen:
                continue
            seen.add(q[0])
            tokens[n], answers[n], kinds[n], flags[n] = q
            n += 1
        return tokens, answers, kinds, flags


def scene_stats(mask, K):
    """Centroids (K, 2) as (x, y) and visible pixel counts (K,) from an object-id mask (0 = background)."""
    H, W = mask.shape
    ids = mask.ravel().astype(np.int64)
    vis = np.bincount(ids, minlength=K + 1)[1:K + 1]
    ys, xs = np.divmod(np.arange(H * W), W)
    sx = np.bincount(ids, weights=xs + 0.5, minlength=K + 1)[1:K + 1]
    sy = np.bincount(ids, weights=ys + 0.5, minlength=K + 1)[1:K + 1]
    cent = np.stack((sx, sy), -1) / np.maximum(vis, 1)[:, None]
    return cent, vis


def pick_heldout(n_a, n_b, per_a, seed):
    """Choose held-out (a, b) value pairs: `per_a` values of b for each value of a, spreading
    b as evenly as possible so every single value still appears in training."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_b)
    out, k = [], 0
    for a in range(n_a):
        for _ in range(per_a):
            out.append((a, int(order[k % n_b])))
            k += 1
    return out
