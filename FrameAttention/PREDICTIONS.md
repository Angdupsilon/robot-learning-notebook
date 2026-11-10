# Predictions and analysis rules

Written 2026-11-09, before the runs and analyses listed below. Do not edit entries after results
exist. Changes go under **Amendments** at the bottom, dated, with the reason and what had been seen.

## What had already been seen when this was written

Anything below is exploratory and cannot count as a confirmed prediction:

- Shapes at lr 1e-3 and 30,000 steps: A, A_prime, B, C, D with 5 seeds.
- Shapes, seed 0 only: K, D_nosync, D_nomoves, D_frozen.
- Seed 0 learning curves for A, A_prime and D, and D's per-layer tag gap.

Everything from Stage 1 is a pilot. The paper's main table comes from runs under the rules below.
Amendment 1 lists what was seen after this list was written.

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

### Amendment 1 (2026-11-09): results seen after writing, and registration timing

**Reason.** More Stage 1 runs finished and were looked at after the list at the top was written.
No prediction, threshold or analysis rule is changed by this amendment.

**Also seen before this amendment:**

- The Shapes summary table below, at lr 1e-3 and 30,000 steps.
- B's seed 0 per-layer tag gap (0.445, 0.444, 0.455, 0.464, 0.485, 0.449) and B's seed 0 learning curve.

| variant | seeds | iid acc | iid binding-pair acc | OOD acc | OOD gap | tag gap | FG-ARI frames | FG-ARI content | dial sync |
|---|---|---|---|---|---|---|---|---|---|
| A | 5 | 60.7 ± 6.7 | 80.5 ± 10.0 | 5.4 ± 6.8 | 75.1 | – | – | 52.9 ± 8.4 | – |
| A_prime | 5 | 54.9 ± 2.6 | 70.7 ± 5.2 | 0.2 ± 0.2 | 70.5 | – | – | 51.6 ± 4.1 | – |
| B | 5 | 95.4 ± 0.9 | 97.9 ± 0.3 | 60.9 ± 2.3 | 37.0 | 0.443 ± 0.029 | 60.0 ± 2.6 | 54.1 ± 5.8 | 0.577 ± 0.018 |
| C | 5 | 95.0 ± 2.8 | 98.1 ± 0.3 | 61.8 ± 5.6 | 36.3 | – | 56.5 ± 2.8 | 53.8 ± 4.8 | – |
| D | 5 | 96.5 ± 0.4 | 98.2 ± 0.2 | 66.5 ± 6.2 | 31.7 | 0.415 ± 0.049 | 55.3 ± 4.8 | 51.6 ± 2.9 | 0.609 ± 0.019 |
| K | 3 | 60.8 ± 4.0 | 79.7 ± 7.7 | 1.9 ± 2.7 | 77.8 | 0.145 ± 0.017 | 46.6 ± 1.6 | 42.7 ± 15.3 | 0.922 ± 0.009 |
| D_nosync | 3 | 96.3 ± 0.7 | 97.9 ± 0.5 | 59.9 ± 4.3 | 38.0 | 0.396 ± 0.030 | 47.5 ± 1.0 | 48.5 ± 1.6 | 0.567 ± 0.016 |
| D_nomoves | 3 | 95.1 ± 1.5 | 98.0 ± 0.6 | 59.3 ± 7.5 | 38.7 | 0.428 ± 0.024 | 55.8 ± 4.6 | 50.4 ± 2.0 | 0.604 ± 0.018 |
| D_frozen | 3 | 95.2 ± 1.3 | 97.5 ± 0.4 | 51.4 ± 12.8 | 46.1 | 0.393 ± 0.009 | 46.5 ± 2.1 | 45.7 ± 1.8 | 0.545 ± 0.016 |

OOD gap is iid binding-pair accuracy minus OOD accuracy, computed from the means.

**What this means for the predictions:**

- **Baselines are confounded in this pilot.** A, A_prime and K are all below the 90% iid requirement,
  so their comparisons with D say nothing clean yet about binding.
- **B1, B2:** informed by this table. They count only if the runs under the analysis rules confirm them.
- **C1:** D vs D_frozen is NOT SEP here (+15.1 points OOD, pooled std 10.0, seeds 5/3), against the
  prediction. C1 is left as written and will be tested with 5 seeds under the analysis rules.
- **C5:** mean D dial sync is visible above, so C5 is informed by this table.
- **M6:** FG-ARI by layer has not been seen, but per-layer tag gap has for B and D.
- **The other predictions** concern runs or analyses that had not been done at this date.

**Registration timing.** Registration on OSF is deferred until Stage 1 is finished, so the decision
to pursue publication can use the pilot. All Shapes runs finished before registration are therefore
exploratory. Only runs started after registration can confirm a prediction. Registration comes
before any CLEVRTex, MOVi, Boxes or 3D task result, and before any frame-swap analysis (M1 to M5).

### Amendment 2 (2026-11-09): how the control and the tuning are implemented

**Reason.** The random-frame control and the tuning sweep were built. This fixes their details
before any result from them exists. No prediction, threshold or analysis rule is changed.

- **C3 is the variant `D_random`:** D_frozen's layout, with no content-based frame initialization.
  Each token position gets its own random frame, drawn once per model from the same 1 radian spread
  as `--frame-init-std`, and the frame stays the same for every image.
- **The sweep is `run_tune.sh`:** learning rates {3e-4, 1e-3, 3e-3}, seed 0, with 20 evaluations per
  run so that 90% of training falls exactly on one.
- **The choice is `pick_tuning.py`:** without `--reveal` it reads only iid validation accuracy and
  prints no OOD numbers. The learning rate is chosen per variant.
- **The plateau rule can pass while a model is still undertrained,** because the learning rate has
  decayed by the last 10% of training. The 90% iid requirement is checked separately for that reason.
