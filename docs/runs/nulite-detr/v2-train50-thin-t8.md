# Train50 — thin FastViT-T8 baseline

Held-out Fold3 result that establishes the light weight baseline. This run
replaces the earlier FastViT-T12 reference series as the project baseline.

## Configuration

`main/pannuke-thin-t8.yaml`, derived from the Train44 control
(`main/pannuke-frozen-thaw-full.yaml`) by changing the backbone and thinning the
DETR head. Three mechanisms are inherited unchanged from Train44: the frozen
NuLite encoder-decoder (`detr_backbone: frozen`), the depthwise-pointwise
fusion adapters on the U-Net skip concats (`detr_frozen_adapter: true`), and the
staged full-U-Net thaw at epoch 50
(`detr_frozen_oracle_unfreeze_decoder: true`,
`detr_frozen_thaw_warmup_epochs: 50`). The score head is not shared across
layers (`detr_shared_layer_groups: 3` groups the *transformer* layers only).

Deltas from Train44:

| setting | Train44 | Train50 |
|---|---|---|
| `nulite_variant` / `det_variant` | `fastvit_t12` | `fastvit_t8` |
| `nulite_ckpt` | `nuLiteT-fold1-sep-t12.pt` | `nuLiteT-fold1-sep-t8.pt` |
| `detr_ndl` | 6 | 5 |
| `detr_hd` | 256 | 192 |
| `detr_d_ffn` | 1024 | 512 |
| `detr_shared_layer_groups` | 0 | 3 |

The T8 oracle was reused from disk, not retrained. It loads into
`NuLiteSegOracle` with 554/554 keys matched (3.640M params, versus 7.206M for
T12).

The three head cuts are justified by measurements, not guesswork:

- `detr_shared_layer_groups: 3` — on T12, sharing to 2 groups (Train45) cost
  only -0.0017 bPQ for -2.1M parameters.
- `detr_ndl: 5` — a truncation probe on Train44 gave centroid F1 of L1 0.7602,
  L2 0.7945, L3 0.8048, L4 0.8051, L5 0.8064, L6 0.8310. Layers 2-5 are nearly
  redundant with each other, so dropping the sixth is the cheapest cut.
- `detr_hd: 192` and `detr_d_ffn: 512` — the transformer layers were 53.6% of
  the Train44 parameter count (6.123M of 15.27M at six layers), and FFN is
  roughly two thirds of each layer. This is the only lever that moves total size
  substantially.

`detr_nq: 300` is inert in grid mode: `_grid_decoder_input` derives
`num_queries = round(1 / grid_size) ** 2` = 400 at `grid_size: 0.05`, as in
every prior run.

## Parameters

| component | Train44 | Train50 |
|---|---:|---:|
| total | 15.27M | **6.16M** |
| frozen oracle U-Net | 7.206M | 3.640M |
| detection head | 8.066M | 2.519M |

Train50 is 40% of the Train44 parameter count, and 46% of the un-thinned T8
build (11.43M, six distinct layers at `hd` 256 / `d_ffn` 1024).

## Held-out Fold3 results

`eval_lsp.py`, 2,722 images, 66,654 GT instances, raw `nc+1` argmax, no NMS.

| checkpoint | params | bPQ | bDQ | bSQ | mPQ | macro bPQ | macro mPQ | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **train50 last** | **6.16M** | 0.6337 | 0.7755 | 0.8025 | **0.4546** | 0.6532 | 0.4609 | 0.8222 | 0.7717 |
| train50 best | 6.16M | 0.6314 | 0.7758 | 0.8001 | 0.4547 | 0.6488 | 0.4580 | 0.8111 | 0.7758 |
| train44 last | 15.27M | 0.6374 | 0.7795 | 0.8044 | 0.4544 | 0.6578 | 0.4617 | 0.8440 | 0.7616 |
| train34 control | 24.7M | 0.6325 | 0.7759 | 0.8009 | 0.4476 | 0.6521 | 0.4578 | 0.8467 | 0.7583 |
| train36 reference | 24.7M | 0.6274 | 0.7690 | 0.8007 | 0.4445 | 0.6458 | 0.4498 | 0.8555 | 0.7463 |

Against Train44, Train50 gives up 0.0037 bPQ and 0.0046 macro bPQ, gains 0.0002
mPQ (flat), and trades -0.022 precision for +0.010 recall. It still beats the
full-size Train34 (24.7M) and Train36 controls on bPQ and macro bPQ.

The validator and held-out readings disagreed in the expected direction: the
last-20 validator window compared Train50 unfavourably to Train44 by -0.0089
bPQ, but the held-out gap is less than half that. Validator PQ remains a
trajectory signal, not a result.

Train50 peaked on the validator at epoch 251 (bPQ 0.6574) and decayed through
the cosine tail to 0.6522 at epoch 332, so `last.pt` is retained per protocol
rather than `best.pt`; the two differ by only 0.0023 held-out bPQ.

## Cohort audit

| cohort | train50 | train44 | delta |
|---|---:|---:|---:|
| matched | 51,066 | 50,438 | +628 |
| good_suppressed_bg | **8,782** | 9,731 | **-949** |
| no_good_polygon | 6,403 | 6,255 | +148 |
| wrong_type | 7,729 | 7,945 | -216 |
| unmatched_predictions | **11,491** | 9,711 | **+1,780** |
| duplicate_like_raw | 40 | 50 | -10 |
| raw_predictions | 62,590 | 60,158 | +2,432 |

The thinning recovers 949 nuclei from the good-polygon/background suppression
cohort, the dominant held-out failure mode tracked since Train34, and reduces
wrong-type errors by 216. It pays for that with 1,780 additional unmatched
predictions, which is the source of the precision drop. This is the same
recall-for-precision trade seen in Train45, but larger, and achieved at a third
of the parameter count.

Suppression recovery is therefore the main non-obvious effect of the thinning.
Whether it comes from the smaller hidden width (a less saturated score
distribution) or from fewer shared layers could not be separated in this run,
because all three head cuts were made together.

## Reproduction

Training (the user runs training; agents must not launch it):

```bash
SKIP_SEG=1 DET_NAME=train50_thin-t8-aggressive scripts/train_thin_t8.sh
```

`scripts/train_thin_t8.sh` has two stages. Stage 1 trains the segmentation
oracle with `raycasted.model.train_nulite_seg`; `SKIP_SEG=1` skips it and reuses
`output/oracle/nuLiteT-fold1-sep-t8.pt`. Stage 2 runs the detector. Both stages
honour `PY`, `DEVICE`, `WORKERS`, `BATCH`, `SEG_EPOCHS`, `DET_EPOCHS`,
`DET_END`, `OUTPUT`, `DET_NAME`, `RECIPE`, and `LOG_DIR`.

Evaluation:

```bash
uv run python -m raycasted.evals.eval_lsp \
  --weights output/v2/train50_thin-t8-aggressive/weights/last.pt \
  --data-dir output/v2/transformed/test \
  --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
  --batch 8 --workers 0 --device 0 --save-predictions /tmp/opencode/rc_full/pred_train50_last.npz
```

## Status

Single-fold held-out result. Three-fold cross-validation is still required
before any comparison with published LSP-DETR (3-fold macro bPQ 0.675, mPQ
0.482) or LKCell numbers.
