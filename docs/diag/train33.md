# train33 — Generalization Diagnosis (read-only)

Date: 2026-09-20
Checkpoint: `output/f1-train/train33/weights/best.pt` (recast experiment; trained fold1, 600 ep)
Scope: **read-only** diagnostics. No model, config, or checkpoint was modified.

## Question

The model underperforms because many nuclei are not found on val/test but are found on
train. Assignment is not the problem (correct predictions and GT are matched). Is this a
generalization failure, and where does it live?

## Method

Matched-positive audit (extends `output/f1-train/train33/generalization-audit/` to a
3-way split). For every GT matched to a query by the original Hungarian matcher we record
whether the head still predicts `∅` (BG error), and decompose the logit margin into the
learned-feature part and the direct seed-channel part:

```
margin          = z[gt_cls]   - z[∅]
contrib         = head.weight[:, -1] * head_input[-1]      # direct seed channel
feature_margin  = (z - contrib)[gt_cls] - (z - contrib)[∅]
```

Splits: train = fold1 (2656 tiles), val = fold2 (2523), test = fold3 (2722).
384 uniformly spaced, unaugmented tiles per split; eval mode; no optimizer updates.

Scripts: `/tmp/opencode/gen_diag2.py`, `/tmp/opencode/bn_probe2.py`,
`/tmp/opencode/feat_probe.py`. Results: `/tmp/opencode/{gen_diag2_train33,bn_probe2_train33,feat_probe_train33}/`.

## Headline result

| split | n | BG/∅ error | feature margin | seed margin |
|---|---|---|---|---|
| train (fold1) | 8491 | **7.3%** | +7.46 | +0.020 |
| val (fold2) | 9107 | **25.4%** | +2.85 | +0.018 |
| test (fold3) | 9171 | **25.7%** | +3.25 | +0.019 |

val ≈ test ⇒ fold-agnostic. The collapse is entirely in the **learned-feature** part of the
head input; the direct seed-channel contribution is identical across splits.

## BG error by nucleus size (px²)

| size bin | train | val | test |
|---|---|---|---|
| < 100 | 58.7% | 58.6% | 69.9% |
| 100–249 | **5.4%** | **39.5%** | **45.1%** |
| 250–499 | 0.5% | 17.7% | 18.6% |
| ≥ 500 | 0.2% | 9.4% | 9.2% |

Two distinct problems:
1. **Tiny nuclei (< 100 px²) fail even on train** — a resolution/shape limitation, not a
   generalization issue.
2. **The generalization gap lives in the 100–249 px² bin** (train 5.4% → held-out
   39.5%/45.1%), and remains large even for ≥ 500 px² (0.2% → 9%).

## Hypotheses ruled out by measurement

| hypothesis | test | result |
|---|---|---|
| direct seed/∅ cue | zero the seed contribution | val BG 25.4 → 25.5% (negligible) |
| frozen FG prior | oracle seg vs ray targets | Dice 0.850 train vs 0.825/0.827 held-out (near gap-free) |
| class/tissue/size/coverage composition | reweight val strata to train proportions | 7.5% reweighted-train vs 24.1% val |
| BatchNorm running-stat mismatch | per-batch BN; per-split recalibration | batch val 26.6%/test 29.1% (no better); recalib catastrophic even on train (80%) |
| head miscalibration | fresh ridge probe on frozen features | ridge val 80.3%/test 81.5% ≈ model head 81.1%/82.7% |

The ridge-probe result is decisive: a **fresh linear classifier trained on the frozen train
features transfers at the same accuracy as the model's own head**. The bottleneck is the
features themselves, not the classifier.

## Feature-transfer evidence

| probe | train | val | test |
|---|---|---|---|
| model class acc (non-BG preds) | 99.2% | 81.1% | 82.7% |
| ridge probe (fit on train features) | 99.0% | 80.3% | 81.5% |
| nearest-centroid cosine | 95.2% (cos 0.674) | 69.0% (0.406) | 69.6% (0.412) |

Val/test features are far less aligned to their class centroids (cosine 0.674 → ~0.41;
cosine is scale-invariant, so this is a genuine directional shift). Class separability in
the features falls 99% → ~80% held-out.

### Model-head confusion (rows = pred, cols = gt; Neoplastic / Inflammatory / Connective / Dead / Epithelial)

```
train [3750,  12,  12,  1,   0]      val  [3076, 104, 278,  23,  56]      test [3837, 113, 325, 22, 117]
      [   6,1224,  11,  0,   2]           [  76,1181, 293,  10,  37]           [  94, 958, 243, 15,  32]
      [   4,  14,2152,  0,   1]           [ 195, 234,1998,  22,  73]           [ 192, 179,1636, 19,  58]
      [   3,   0,   2,166,   0]           [  53,  12,  30,  87,   0]           [  26,   1,  24, 97,   0]
      [   0,   0,   3,  0,1128]           [ 116,  33,  74,   0,1046]           [  26,  30,  66,  0,1061]
```

Train is essentially diagonal; held-out confusion is systematic and concentrated in the
morphologically similar **Neoplastic ↔ Connective ↔ Inflammatory** classes (plus some
Epithelial ↔ Neoplastic).

## Supporting measurements

**Input appearance** (per-split means) — small but systematic fold drift:

| split | R | G | B | stain_ratio (R/G) |
|---|---|---|---|---|
| train | 0.6014 | 0.4254 | 0.5575 | 1.512 |
| val | 0.5978 | 0.4181 | 0.5511 | 1.530 |
| test | 0.5885 | 0.4075 | 0.5428 | 1.551 |

**Det-stream encoder features** (hook on `m.encoder_decoder`):

| level | train std / absmean | val std / absmean | test std / absmean |
|---|---|---|---|
| b3 (st4) | 0.566 / 0.405 | 0.543 / 0.393 | 0.555 / 0.398 |
| b5 (st16) | **0.420 / 0.324** | **0.330 / 0.257** | **0.328 / 0.256** |

Coarse-level b5 activations are ~24% weaker held-out. BN experiments show this is
content-driven, not a BatchNorm-statistics artifact.

**Oracle FG segmentation** (frozen prior, pixel-level) — transfers cleanly:

| split | recall | precision | Dice |
|---|---|---|---|
| train | 0.8589 | 0.8417 | 0.8502 |
| val | 0.8335 | 0.8175 | 0.8254 |
| test | 0.8328 | 0.8210 | 0.8269 |

## Best-supported diagnosis

The held-out gap is a **transfer limitation of the DETR head's learned query features**,
which are supervised only through the sparse 1:1 matched queries. Those features fit fold1
(class separability 99%) but do not transfer (≈80%), while the **densely supervised, frozen
oracle transfers well** (Dice 0.85 → 0.825). This is not explained by the seed cue, the FG
prior, category composition, BatchNorm statistics, or head miscalibration.

Separately, tiny nuclei (< 100 px²) are a resolution/shape limitation present even on train.

## Actionable next steps (training-side only; no post-processing)

1. **Priority — auxiliary dense supervision of the trainable det stream**: add a per-pixel
   auxiliary loss on the det decoder features (b3/b4/b5) against GT masks/classes, so the
   encoder is not shaped solely by matched-query signal. Targets the 100–249 px² bin.
2. **Input domain generalization**: stain normalization (e.g. Macenko) and/or augmentation
   matched to the measured fold-to-fold appearance drift.
3. **Reduce sparse-head overfit**: dropout / weight decay / heavier augmentation, or train the
   final model on 2 folds (relaxes the single-fold held-out requirement).

## Artifacts

- Scripts: `/tmp/opencode/gen_diag2.py`, `/tmp/opencode/bn_probe2.py`, `/tmp/opencode/feat_probe.py`
- Results: `/tmp/opencode/gen_diag2_train33/`, `/tmp/opencode/bn_probe2_train33/`, `/tmp/opencode/feat_probe_train33/`
- Concurrent-session audits: `output/f1-train/train33/{generalization-audit,generalization-audit-train23,gt-centered-diagnostic,cls-audit}/`

> These are validation diagnostics, not literature PQ. The only literature-comparable
> metric remains `raycasted/evals/eval_lsp.py` on a genuinely held-out fold.
