# FrameAttention

Testing whether **Frame Attention** binds features to objects better than ordinary attention.

Each token carries a frame (2D "dials" and 3D "gyros"). Attention compares tokens in each
other's frame, and every layer updates the frames: tokens that attend to each other align
(sync), and tokens pass each other small turns (moves).

The claim under test: at the same size, Frame Attention binds better on combinations it never
saw in training, and its dials form object tags.

## What's here

| file | what it does |
|---|---|
| `frame_gpt.py` | short Karpathy-style version, for reading (Tiny Shakespeare) |
| `frame_attention.py` | the layer used in experiments (batched, one attention call per layer) |
| `model.py` | the transformer and the variants A, A_prime, B, C, D, E, K |
| `tasks/` | data: synthetic shapes, boxes (text), CLEVRTex, MOVi-A, and the shared question generator |
| `download_data.py` | **one command that downloads or generates every dataset** into `data/` |
| `train.py` | trains and evaluates one model, writes `runs/<task>/<variant>/seed<k>/` |
| `metrics.py` | tag gap, FG-ARI from clustering frames, collapse monitor |
| `summarize.py` | mean ± std table over seeds, plus the pass/fail checks |
| `run_stage1.sh`, `run_stage2.sh` | the full experiment grids |
| `spark_setup.sh` | opens NVIDIA's PyTorch container on the DGX Spark |
| `tests/` | 29 checks of the math and the pipeline |

## Variants

Every head has 64 channels. Except in K, 16 of them carry ordinary RoPE (position), and the
rest are split as below.

| variant | per head | question it answers |
|---|---|---|
| A | RoPE + 48 plain | the control: an ordinary RoPE transformer |
| A_prime | A, with more layers so its FLOPs ≥ D's | is any gain just extra compute? |
| B | + 11 dials (sync + moves) | does the update rule help? |
| C | + 6 gyros (sync + moves) | do gyros alone help? |
| D | + 11 dials + 6 gyros | full Frame Attention |
| E | D + motion fingerprints (MOVi only) | does common fate help? |
| K | published KoPE: 32 phases, no RoPE channels | does D beat the published method? |
| D_nosync | D, sync force off | does sync contribute? |
| D_nomoves | D, moves force off | do moves contribute? |
| D_frozen | D, both off: frames stay as initialized from content | do the frame updates matter at all? |

K is ported from the reference `vit_kope.py` in
[microsoft/Neuro-inspired_Phase_Encoding](https://github.com/microsoft/Neuro-inspired_Phase_Encoding)
([paper](https://arxiv.org/abs/2604.07904)) with that file's defaults: phases start from 2D position
(base 20), one coupling module shared by all layers, step 0.05, learned phase mixing, values rotated,
and absolute position embeddings. The only change is skipping the last layer's phase update, which
nothing reads.

## Setup

**On the Mac (your uv environment).** PyTorch, numpy and pillow are already in
`robot-learning-notebook/uv.lock`, so nothing gets added to `pyproject.toml`:

```bash
cd robot-learning-notebook/FrameAttention
uv run --with pytest pytest tests -q          # 29 tests, about 1 minute (1 skips without scikit-learn)
uv run python download_data.py --only shapes boxes tinyshakespeare
uv run python train.py --task shapes --variant D --preset quick
```

**On the DGX Spark.** Use NVIDIA's container, which has a PyTorch build for the GB10 GPU.
Commit and push `FrameAttention/` first. `data/` is gitignored, so download the data on the
Spark itself.

```bash
git clone <your robot-learning-notebook repo> && cd robot-learning-notebook
bash FrameAttention/spark_setup.sh            # prints the torch version and GPU, then gives a shell
python -m pytest tests -q                     # inside the container: 29 tests
```

Files the container creates in `data/` and `runs/` are owned by root; run
`sudo chown -R $USER FrameAttention` on the host if that gets in the way.

## Step 1: data

```bash
python download_data.py                                   # everything
python download_data.py --only shapes boxes               # generated only, no internet
python download_data.py --only clevrtex --clevrtex-parts 1   # smoke test: 1 of 5 parts
python download_data.py --only movi_a --movi-shards 4        # smoke test: 4 shards
```

| dataset | source | size on disk | notes |
|---|---|---|---|
| shapes | generated locally | 1.8 GB | 110k scenes, about 1 min with 2 cores |
| boxes | generated locally | 60 MB | 215k stories, about 15 s |
| tinyshakespeare | copied from `../nanoGPT/ShakespeareText.txt` | 1 MB | only for `frame_gpt.py` |
| clevrtex | 7 archives from Oxford | printed before download | 5 parts + OOD + CAMO; converted to 128×128 |
| movi_a | TFDS shards from Google's public bucket | printed before download | keeps 1 frame + 8 steps of flow per video |

- The script prints download sizes and checks free disk space first. Re-running skips
  anything already finished, and interrupted downloads resume.
- MOVi conversion needs TensorFlow. If it isn't importable, the script creates a private
  environment in `data/.tfds-venv`; training never needs it. TensorFlow has wheels for both
  the Spark (Linux ARM64) and Apple Silicon, so the conversion can also run on the Mac before
  you copy `data/movi_a/` over.
- **Run the two smoke-test commands first on the Spark.** The download code was tested against
  local servers serving files in the exact formats of these sources (TFDS layout built with
  Kubric's own feature definitions; CLEVRTex archives with the official naming). The real
  Oxford and Google servers couldn't be reached from where this was built, so the first real
  download is the first contact with them.

## Step 2: smoke test (a few minutes)

```bash
for v in A A_prime B C D; do python train.py --task shapes --variant $v --preset quick; done
python summarize.py --runs runs
```

Training prints `samples/s` and an ETA after 200 steps. Use that to adjust `--steps` before
starting long runs. The step counts in the `full` presets are estimates.

## Step 3: Stage 1 (the decisive test)

```bash
bash run_stage1.sh          # A, A_prime, B, C, D, K x 5 seeds; finished runs are skipped
python summarize.py --task shapes
```

On the Mac, prefix with uv: `PY="uv run python" bash run_stage1.sh`.

The synthetic task:

- 64×64 scenes with 3–5 touching or overlapping objects, each with a shape, a color and a texture.
- One color per shape (6 of 36 shape-color combinations) never appears in training.
- `test_ood` asks only about those unseen objects, for example *"what color is the triangle?"*
  when that triangle's color has never been seen on a triangle.

### Pass/fail, fixed before running

`summarize.py` checks these:

1. **Primary:** D beats A on OOD accuracy by more than 2× the seed spread.
2. **Not just compute:** D also beats A_prime by more than 2× the seed spread.
3. **Beyond published work:** D also beats K by more than 2× the seed spread. Added on
   2026-11-08, after Stage 1 had started but before any D run had finished.
4. **Mechanism:** D's tag gap is clearly above zero and rises during training
   (`metrics.jsonl`), and FG-ARI from frames matches the true objects.

How to read the outcomes:

| result | meaning |
|---|---|
| B ≈ D | gyros aren't earning their cost |
| D ≈ A_prime | the gain is compute |
| D ≈ K | the gain is what KoPE already showed |
| A ≈ D | the idea doesn't work here (still a reportable result) |

### Ablations of D

Added on 2026-11-09, after Stage 1 finished, because D's tag gap stayed flat across layers and
after step 2,500. These are follow-up diagnostics, not part of the pass/fail checks above.

```bash
VARIANTS="D_nosync D_nomoves D_frozen" SEEDS="0 1 2" bash run_stage1.sh
```

| result | meaning |
|---|---|
| D_frozen ≈ D | the benefit comes from the content-set rotations, not from frames evolving across layers |
| D beats D_frozen, D_nosync ≈ D | moves carry the benefit; sync isn't needed |
| D beats D_frozen, D_nomoves ≈ D | sync carries the benefit |

## Step 4: Stage 2 (only if Stage 1 passes)

```bash
bash run_stage2.sh          # clevrtex: A A_prime D | movi_a: A D E | boxes: A A_prime D
```

## What the metrics mean

| metric | meaning |
|---|---|
| iid acc | accuracy on held-in data |
| iid binding-pair acc | the same question types as OOD (shape↔color), on seen combinations: the fair comparison for OOD |
| OOD acc | questions about held-out combinations only |
| tag gap | mean frame similarity of patch pairs on the same object minus pairs on different objects. Larger means sharper tags. `tag_gap_per_layer` shows it sharpening with depth. |
| FG-ARI frames | cluster patches by their frames, score against the true objects (1 = perfect) |
| FG-ARI content | the same, using the final embeddings; the baseline every variant has |
| dial sync | how aligned all object patches' dials are within an image. Near 1 in every image means all objects share one tag (collapse), and the layer has quietly become plain attention. |

## Caveats worth knowing

- **What A_prime matches.** A_prime matches matmul FLOPs, rounded up to whole layers. Frame
  Attention also spends time on small element-wise rotations that FLOP counters ignore, so
  compare `samples_per_s` in `results.json` too.
- **CLEVRTex hold-out is question-level.** The images are fixed, so training never *asks* about
  held-out shape-material objects, but it does see them. FG-ARI here comes from a model trained
  to answer questions, so it isn't directly comparable to AKOrN's unsupervised numbers.
- **Why MOVi-A, not MOVi-C/E.** C and E label objects only by category, which leaves no pair
  of attributes to bind. MOVi-A has shape, color, size and material.
- **The last layer doesn't update frames,** since nothing reads them afterwards. Starting frames
  are spread about 1 radian by content (`--frame-init-std`), so the sync rule is active from
  step 0.
