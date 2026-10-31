# RayCastED handover — 2026-10-26

## Current objective

Improve RayCastED on PanNuke under the LSP-DETR evaluation protocol while
keeping end-to-end, NMS-free inference. LSP-DETR's published 3-fold macro
reference is bPQ 0.675 and mPQ 0.482. We do not yet have a comparable 3-fold
result; current measurements are single-fold held-out diagnostics.

NuLite-DETR is the current query-based version of RayCastED. It retains the
RayCastED radial polygon representation and replaces the earlier dense FCN
prediction head with a Hungarian one-to-one query decoder; it is not a separate
project or competing architecture.

## Ground truth for evaluation

Use `raycasted/evals/eval_lsp.py` only for literature-facing claims. It uses
the original dense parquet masks, raw no-object argmax decisions, PIL polygon
filling, and LSP-DETR's distance-based overlap ownership. It reports
per-image mean PQ and tissue macro metrics.

The evaluator was corrected on 2026-10-25. Old documents that report OpenCV
filling, largest-polygon overlap ownership, confidence thresholds, polygon GT,
or training-fold "test" results are not literature comparable. See
`docs/diag/lsp-evaluator-correction-2026-10-25.md`.

## Held-out reference results

All figures below use Fold3 dense GT, 2,722 images and 66,654 instances, and
`eval_lsp.py` with raw `nc+1` argmax and no NMS.

**The current baseline is Train50**, a thin FastViT-T8 model at 6.16M
parameters. Full documentation is in
`docs/runs/nulite-detr/v2-train50-thin-t8.md`.

| checkpoint | params | bPQ | bDQ | bSQ | mPQ | macro bPQ | macro mPQ | P / R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`output/v2/train50_thin-t8-aggressive/weights/last.pt`** | **6.16M** | 0.6337 | 0.7755 | 0.8025 | 0.4546 | 0.6532 | 0.4609 | 0.8222 / 0.7717 |
| `output/v2/train44_frozen-thaw-full/weights/last.pt` | 15.27M | 0.6374 | 0.7795 | 0.8044 | 0.4544 | 0.6578 | 0.4617 | 0.8440 / 0.7616 |
| `output/v2/train34_viable-owner-foreground/weights/last.pt` | 24.7M | 0.6325 | 0.7759 | 0.8009 | 0.4476 | 0.6521 | 0.4578 | 0.8467 / 0.7583 |
| `output/f1-train/train36/weights/last.pt` | 24.7M | 0.6274 | 0.7690 | 0.8007 | 0.4445 | 0.6458 | 0.4498 | 0.8555 / 0.7463 |
| `output/v2/train30/weights/last.pt` appearance | 24.7M | 0.6274 | 0.7666 | 0.8023 | 0.4404 | 0.6462 | 0.4444 | 0.8546 / 0.7440 |
| `output/v2/train31/weights/last.pt` appearance + presence | 24.7M | 0.6269 | 0.7678 | 0.8006 | 0.4434 | 0.6446 | 0.4458 | 0.8584 / 0.7442 |

Train50 gives up 0.0037 bPQ against Train44, the previous best, while using 40%
of the parameters; it still beats the full-size Train34 and Train36 controls.
Train36 stopped at epoch 246. Train30, Train31, Train34, Train44, and Train50
all stopped at epoch 332 under a 600-epoch learning-rate schedule, so Train36
is a provisional rather than matched control for them.

## Architecture of the Train50 baseline

Two streams, one forward pass. A frozen NuLite-T8 U-Net produces the
foreground map, and the same U-Net supplies `b3`/`b4`/`b5` to the DETR decoder
through small depthwise-pointwise fusion adapters. The oracle is fully thawed
from epoch 50 onward except for `np_head`. The decoder has five layers sharing
three distinct transformer weights, `hd` 192, `d_ffn` 512, and 400 grid
queries. Inference remains raw `nc+1` argmax with no NMS, clustering, or
confidence threshold.

## What the diagnosis established

The main recall failure has two parts:

1. **Viable polygon suppressed as empty.** Train36 has 10,626 held-out GT
   nuclei with a raw query polygon at IoU above 0.5 but no foreground firing.
   Their median best polygon IoU is 0.708; their median best
   foreground-minus-empty margin is -2.325. They are concentrated in small
   nuclei.
2. **No viable raw polygon.** 6,764 GT nuclei have no raw query crossing IoU
   0.5. This cohort is smaller still and is primarily a coverage/geometry
   problem.

The first cohort is not commonly caused by another GT stealing its owner. A
256-image decoder trace found that good-polygon/background query slots are
almost always already empty at the encoder stage and stay empty while polygon
IoU improves. The final Hungarian owner is viable for about 88% of that cohort;
best candidates assigned to another GT are rare.

Frozen feature tests found some identity information, but foreground-map
weighted predicted-region pooling barely improved it. A held-out linear probe
could add roughly 20 matches only by increasing duplicate-like outputs from 4
to 25; constraining duplicates removed the recall gain. This does not support
an inference-time probe or more generic foreground pooling as the next change.

Later probes narrowed this considerably and are recorded in
`docs/diag/undocumented-findings-2026-10-30.md` and
`docs/diag/query-classification-failures-2026-10-28.md`. The decisive result:
on the exact 257-d `enc_score_head` input, a balanced linear probe reaches AUC
0.912 (0.892 with nucleus size fully controlled), while the trained head
operates at foreground recall 0.650 and background recall 0.926. More capacity
does not help. The constraint is the decision rule that one-to-one Hungarian
training over roughly 93% negative queries buys, not the representation or the
head's capacity.

## Mechanisms tried and their outcomes

All figures are Fold3 held-out `eval_lsp` bPQ after a full 332-epoch run unless
noted. The Train44 configuration is the common control for this series.

| mechanism | config flag | bPQ | verdict |
|---|---:|---:|---|
| Train44 control | — | 0.6374 | strongest accuracy, 15.27M |
| nucleus-CDN | `detr_nucleus_cdn` | 0.6302 | learned shortcut: score head read the answer off a private class embedding |
| CDN token-only | `detr_nucleus_cdn` (no lookup) | 0.6251 | confirmatory null |
| CDN real-to-DN attention | `detr_nucleus_cdn_real_to_dn` | collapsed | real queries depended on anchors absent at eval |
| RoPE | `detr_self_attention_rope` | 0.6301 | null; positions load-bearing but redundant with absolute PE |
| sliding-tile attention | `detr_self_attention_sta` | 0.6294 | best bSQ of the series, no compute saving, worse bDQ |
| shared single stream | `detr_backbone: shared` | 0.6259 | cheaper, but the map drifted and precision fell |
| map modulation | `detr_map_modulation` | stopped ~ep133 | live but ~1-2% of content norm; within noise |
| balanced proposal loss | `detr_proposal_bal_weight` | stopped ~ep117 | recalled +0.006 stable, far short of what is needed |
| score-size bias | `detr_score_size_bias` | stopped ~ep73 | directionally correct, but the size proxy carries too little information |
| **thin T8** | `main/pannuke-thin-t8.yaml` | **0.6337** | **new baseline at 6.16M** |

Full-U-Net thaw with fusion adapters (Train44) and the thin T8 baseline
(Train50) are the two survivors of this line. Everything else was either a
measured null or a learned shortcut.

## Current state and next steps

Train50 is the baseline. Its remaining failure profile, from the cohort audit:

| cohort | count |
|---|---:|
| matched | 51,066 |
| good_suppressed_bg | 8,782 |
| no_good_polygon | 6,403 |
| wrong_type | 7,729 |
| unmatched_predictions | 11,491 |

The thinning recovered 949 nuclei from the suppression cohort but added 1,780
unmatched predictions. If that recall gain is real rather than churn, the
follow-up is to recover the precision: the surviving unmatched predictions
should be characterised by size, crowding, and class before any further
mechanism is built.

Three-fold cross-validation remains required for any comparison with published
LSP-DETR (3-fold macro bPQ 0.675, mPQ 0.482) or LKCell numbers. All results
here are single-fold held-out diagnostics.

## Reproduction commands

The user runs training; agents must not launch it. The baseline is produced by:

```bash
SKIP_SEG=1 DET_NAME=train50_thin-t8-aggressive scripts/train_thin_t8.sh
```

The script runs the segmentation oracle first and the detector second;
`SKIP_SEG=1` reuses `output/oracle/nuLiteT-fold1-sep-t8.pt`. Without the
override it retrains the oracle with `raycasted.model.train_nulite_seg`.

For final held-out evaluation:

```bash
uv run python -m raycasted.evals.eval_lsp \
  --weights output/v2/train50_thin-t8-aggressive/weights/last.pt \
  --data-dir output/v2/transformed/test \
  --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
  --batch 8 --workers 0 --device 0 --save-predictions
```

Then run `raycasted/evals/audit_lsp_cohorts.py` on the saved predictions to
compare matched nuclei, good-polygon/background suppression, no-good-polygon
misses, size, crowding, class, and duplicate proxies.
