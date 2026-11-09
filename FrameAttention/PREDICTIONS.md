# Predictions and analysis rules

Written 2026-11-09, before the runs and analyses listed below. Do not edit entries after results
exist. Changes go under **Amendments** at the bottom, dated, with the reason and what had been seen.

## What had already been seen when this was written

Anything below is exploratory and cannot count as a confirmed prediction:

- Shapes at lr 1e-3 and 30,000 steps: A, A_prime, B, C, D with 5 seeds.
- Shapes, seed 0 only: K, D_nosync, D_nomoves, D_frozen.
- Seed 0 learning curves for A, A_prime and D, and D's per-layer tag gap.

Everything from Stage 1 is a pilot. The paper's main table comes from runs under the rules below.

## Analysis rules

- **Primary metric, OOD gap:** iid binding-pair accuracy minus OOD accuracy. Raw OOD is also reported.
- **Seeds:** 5 per variant in the main table, with mean, standard deviation and 95% CI.
- **SEP:** the means differ by more than 2 × the pooled standard deviation, as in `summarize.py`.
  **NOT SEP** means that test failed. It does not prove the two are equal.
- **Equal tuning:** every main variant gets the same sweep of learning rates {3e-4, 1e-3, 3e-3} on
  seed 0. Pick by final iid validation accuracy, never by OOD. The chosen rate is then fixed for all seeds.
- **One step budget for every variant:** the smallest of {30k, 60k, 120k} at which every variant's
  iid validation accuracy gains less than 1 point over the last 10% of training. Decided once,
  using iid validation only, before final runs.
- **Comparable baselines:** a comparison counts as clean only if the baseline reaches at least 90%
  iid at the final budget. Otherwise it is reported as confounded by training.
- **Reporting:** every run is reported. No seed is dropped or rerun because of its result. Interim
  OOD numbers in `metrics.jsonl` are never used for selection.
- **K:** `K` matches Microsoft's `vit_kope.py` at commit `1722dcf`. With weights copied across and
  the x/y phase channel order aligned, patch outputs agree to 1e-8. The other differences are task
  adaptations: question tokens start at zero phase, position embeddings cover them, token-type
  embeddings are added, and the unread last phase update is skipped.

## Layer 1: behavior

| ID | Prediction | If it fails |
|---|---|---|
| B1 | On Shapes, D's OOD gap is SEP smaller than A's and A_prime's. | Claim 1 is not supported on Shapes. |
| B2 | On Shapes, D's OOD gap is SEP smaller than K's. | D adds nothing measurable beyond KoPE here. |
| B3 | D's OOD gap is not SEP larger than a slot baseline's. | Frames do not match slot models. Claim 1 fails as worded. |
| B4 | On CLEVRTex or MOVi under the Kapl protocol, D's OOD gap is SEP smaller than A's. | The Shapes effect does not transfer to richer data. |
| B5 | Within the trained range, accuracy falls less per added object for D than for A (slope SEP shallower, 3 to 5 objects). | Frames do not reduce the cost of crowding. |
| B6 | Tested on 1 to 8 objects with visibility held fixed, D still has a measurable breaking point at or below 8 objects. | Report the range tested. No capacity limit was found within it. |

## Layer 2: predicted pattern

| ID | Prediction | If it fails |
|---|---|---|
| P1 | CLEVRTex: D vs B is NOT SEP. In-plane rotations commute. | Gyros help on a task where theory says they should not. |
| P2 | 3D rigid-parts task with parts interleaved: D is SEP better than B and SEP better than D_dials. | Claims 2 and 3 fail. |
| P3 | Boxes: every variant is at least 10 points worse past the training length than within it, and D vs B is NOT SEP. | Revisit the state-tracking theory. |
| P4 | D is not SEP better than B on both CLEVRTex and Boxes. | Read D's gains as extra rotation capacity, not non-commutativity. |

Shapes also predicts D ≈ B, but that was seen before this was written, so it does not count.

## Layer 3: mechanism

"Question token" means the shape or color descriptor tokens in the question. "Flip" means the
answer changes to the distractor object's attribute.

| ID | Prediction | If it fails |
|---|---|---|
| M1 | In D, a question token's frame is closer to its target object's patches than to distractor patches, at some layer, with the 95% CI over seeds above zero. | Frames are not acting as pointers from question to object. |
| M2 | Giving the question token a distractor patch's frame flips the answer at least 30% of the time at the best layer, and at least 3× as often as a random-frame swap or a same-object swap. | Frames are not sufficient to redirect binding. |
| M3 | Moving a growing fraction of an object's patches to another object's frame degrades the answer monotonically. | Frames are not necessary at the object level. |
| M4 | Swaps at earlier layers have less effect than swaps at later layers in models with sync on, because later updates partly undo them. | Frames do not recover after a swap. |
| M5 | D's flip rate is SEP higher than D_frozen's. B's is NOT SEP from D's. | B difference: binding depends on gyros. No frozen difference: updates are not needed for redirection. |
| M6 | FG-ARI from frames rises by at least 5 points from the first layer to the last in D. | Frames do not sharpen into object groups with depth. |
| M7 | On the 3D task, the relative rotation g_i^T g_j predicts the true part-to-whole rotation with median geodesic error at most half that of a constant-rotation baseline, and lower than a probe on content embeddings. | Claim 3 fails. |

## Layer 4: controls

| ID | Prediction | If it fails |
|---|---|---|
| C1 | D_nosync vs D on OOD is NOT SEP, and D is SEP better than D_frozen. Informed by seed 0, so a confirmation only. | The ablation story from seed 0 was noise. |
| C2 | Dials-only (`n_dial=20, n_gyro=0`, channel-matched, 9 fewer degrees of freedom than D) vs D on Shapes is NOT SEP. | Gyros help even on a task where rotations commute. |
| C3 | Random fixed frames, independent of content, have an OOD gap SEP larger than D's. | Gain is generic geometry, not binding. |
| C4 | Repel-off has final dial sync SEP higher than D's and is not SEP better on OOD. | The repel term is not preventing collapse. |
| C5 | Every D run ends with dial sync below 0.9. | Frames are collapsing in some runs. |

## Amendments

None yet.
