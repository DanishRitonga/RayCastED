# RayCastED handover — 2026-10-26 (updated 2026-11-15)

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

## Seed map used by every reported result (added 2026-11-13)

**This version of the manuscript uses ONLY the fg−bg (binary plateau) seed map.
The centroid-potential map is unused for now.**

Every checkpoint behind a reported number — Train50/T8, the FastViT backbone
sweep, the edgeNeXt variants, the T8 and T12 decoder grids, and all four
matcher arms — was trained with `nuLiteT-fold1-sep-*.pt` and no
`nulite_oracle_kind`, i.e. `oracle_kind: binary` where seed =
`logit(fg) − logit(bg)`. This was verified from the checkpoint attributes, not
inferred from recipes.

The centroid-potential oracles (`nuLiteT-fold1-center-t12.pt`,
`nuLiteT-fold1-fg-center-t12.pt`) were built on 2026-10-02 / 2026-10-05 and no
recipe referenced them before 2026-11-13. The single centroid arm
(`train60_fgmap-fgcentroid`, held-out bPQ 0.6387 / mPQ 0.4542 / F1 0.8034) is
below the adoption bar, is not adopted, and is not part of any manuscript
number. It is retained only as a directional hint that a properly calibrated
centroid map may improve the detection operating point. Details:
`docs/runs/nulite-detr/fg-map-ablation-results.md`.

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

## Lines closed since Train50 (2026-11-15)

Post-Train50 ablations run on a FastViT-T12 control,
`output/v2/train52_fastvit-t12/weights/last.pt`: bPQ 0.6372, bDQ 0.7798,
bSQ 0.8032, mPQ 0.4561, macro bPQ 0.6566, macro mPQ 0.4618, F1 0.7987,
P 0.8270, R 0.7724. Every arm is a single-fold, single-seed held-out Fold3
`eval_lsp` run with raw `nc+1` argmax and no NMS. The pre-registered rule for
each arm was: adopt only if bPQ **and** macro bPQ both gain more than +0.002;
reject if bPQ drops more than 0.002.

### Decoder geometry — exhausted

T8 and T12 grids over `detr_ndl` {4,5,6}, `detr_hd` {128,192,256} and
`detr_shared_layer_groups` {0,2,3}: total spread 0.0032 bPQ, no variant cleared
the band, and the T8 ranking did not transfer to T12. The 5-layer / 192-width /
3-group thinning is confirmed as adequate. Doc:
`docs/runs/nulite-detr/t12-decoder-grid-results.md`.

### Assignment (matcher) — closed

| arm | bPQ | macro bPQ | cohort move | verdict |
|---|---:|---:|---|---|
| soft inside gate | 0.6345 | 0.6548 | `good_suppressed_bg` −14, `no_good_polygon` +30, `unmatched` +73 | rejected (−0.0027) |
| additive empty cost λ=1 | 0.6391 | 0.6578 | `good_suppressed_bg` 1,163 → 1,045 (−118), `matched` +102 | best point, below the +0.002 bar |
| additive empty cost λ=2 | 0.6360 | 0.6554 | suppression only −12, `wrong_type` +37 | rejected; λ sweep closed |
| conditional empty cost | 0.6365 | 0.6565 | `good_suppressed_bg` +78, `matched` −40 | rejected |

The one-to-one Hungarian decision rule is not fixable through match cost.
`wrong_type` (974 of 8,917 audited GT) is untouched because
`p(correct) − p(best_wrong)` has median 0.000 on this head, and the best arm
still leaves ~1,045 suppressed nuclei. Docs:
`docs/runs/nulite-detr/empty-column-cost-results.md`,
`.../soft-inside-gate-results.md`, `.../empty-column-cost-conditional-results.md`.

### Foreground map — closed

Consumer 1, `detr_seed_in_content` off: 0.6365 (−0.0007), null — the decision
path does not use the map. Consumer 2, `detr_seed_level` off: 0.6332
(−0.0040) — the 4th deformable level is load-bearing for geometry. Consumer 3,
`detr_hybrid_peaks`, is dead config under `query_selection: grid`. Consumer 4,
`seed_map_target` + `lambda_seg`, is inert because `lambda_seg` is forced to 0;
enabling it as `detr_seed_map_loss` scored 0.6343 (−0.0029). The current
binary plateau map is therefore not the binding constraint, but improving the
map representation is still the most promising direction left (see the
`fgcentroid` result below). Doc:
`docs/runs/nulite-detr/fg-map-ablation-results.md`.

### Recast and cohort-margin losses (2026-11-15)

Removal arms, both a copy of `main/pannuke-thin-t12.yaml`:

| arm | recipe (one mechanism removed) | bPQ | bDQ | bSQ | mPQ | macro bPQ | macro mPQ | F1 | P | R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | `main/pannuke-thin-t12.yaml` | 0.6372 | 0.7798 | 0.8032 | 0.4561 | 0.6566 | 0.4618 | 0.7987 | 0.8270 | 0.7724 |
| norecast | `detr_recast: false` | 0.6380 | 0.7808 | 0.8038 | 0.4590 | 0.6581 | 0.4648 | 0.7975 | 0.8221 | 0.7743 |
| nomargin | `detr_seg_fg_margin_weight: 0`, `detr_viable_owner_fg_weight: 0` | 0.6289 | 0.7647 | 0.8068 | 0.4524 | 0.6507 | 0.4576 | 0.7945 | 0.8756 | 0.7271 |

Cohort audit (384 images, 8,917 GT):

| cohort | control | norecast | nomargin |
|---|---:|---:|---:|
| matched | 6,929 | 6,955 | 6,477 |
| good_suppressed_bg | 1,163 | 1,143 | 1,620 |
| no_good_polygon | 798 | 779 | 808 |
| wrong_type | 974 | 972 | 916 |
| unmatched_predictions | 1,262 | 1,304 | 860 |
| raw predictions | 8,196 | 8,261 | 7,339 |

**Recast (`train62_norecast`) is harmful as configured.** Removing the
inter-layer query correction is inside ±0.002 on bPQ and macro bPQ, and it
improves the metrics that actually separate nucleus types: mPQ +0.0029 and
macro mPQ +0.0030. It also saves 37k parameters (10.031M → 9.994M construction
count; 6.51 h vs 6.55 h). It is not an inert gate: on the control checkpoint
the projection weight norm is 8.35 (zero at init) and the mean correction is
2.21 against a query norm of 12.65, i.e. ~17.5% of the query magnitude. So the
module is doing substantial work that actively shifts queries without buying
accuracy; the decoder can re-derive the same information from later layers.
Recast should be treated as a removal candidate from the baseline and must not
be presented as a contributor. This is the paper's namesake mechanism, which
makes the negative result important to state explicitly rather than omit.

**The cohort-margin losses (`train63_nomargin`) are strongly load-bearing.**
Removing `detr_seg_fg_margin_weight` and `detr_viable_owner_fg_weight` costs
0.0083 bPQ and 0.0059 macro bPQ, roughly doubles `good_suppressed_bg`
(1,163 → 1,620, +457), drops matched by 452, emits 857 fewer predictions, and
raises precision 0.8270 → 0.8756 while dropping recall 0.7724 → 0.7271. That is
an emit-fewer signature: without the margin terms the head reverts to
suppressing real nuclei. These two 0.5-weight terms account for about 457 of
the 1,163 suppressed nuclei — roughly four times the best matcher arm — and
must stay. The arm also trains 17% faster (58.9 vs 70.9 s/epoch) because the
removed terms do per-layer stencil sampling and polygon-IoU work.

Artefacts: `main/pannuke-thin-t12-norecast.yaml`,
`main/pannuke-thin-t12-nomargin.yaml`, `scripts/train_recast_margin.sh`;
runs `output/v2/train62_norecast`, `output/v2/train63_nomargin`.

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

### Where this leaves the project (2026-11-15)

Four lever families are closed: decoder geometry, matcher cost, foreground-map
quality, and the recast correction. The only active mechanisms still without a
controlled A/B are `detr_per_layer_match` and `detr_matcher_lsp`. The
margin/viable-owner losses are the most effective component measured for the
dominant failure cohort and stay at 0.5/0.5.

Two constraints remain that no matcher or map change has moved: the class head
does not separate nucleus types (`p(correct) − p(best_wrong)` median 0.000;
`wrong_type` is about 11% of audited GT), and one-to-one Hungarian training
keeps foreground recall near 0.65 while a balanced linear probe on the exact
257-d head input reaches AUC 0.912 at recall 0.931.

The next mechanism to pursue is a better foreground-map representation, which
the `fgcentroid` arm points at. That arm (`train60_fgmap-fgcentroid`, an
`fg_centroid` oracle carrying foreground and centroid-potential channels) is
the best detection operating point measured in the sweep even though it sits
below the +0.002 adoption bar: bPQ 0.6387 (+0.0015), macro bPQ 0.6584
(+0.0018), F1 0.8034, P 0.8313, R 0.7773, and 36 fewer `good_suppressed_bg`
nuclei. It is not yet adoptable because two things changed at once (the map
target and the oracle initialisation) and because the saved centroid channel is
sharp (σ = 2.5–4 px), leaving it nearly dead at the 12.8 px query grid — only
2.9–3.5% of cell centres exceed 0.5, against 74% for the plateau map. The
prepared follow-up is the wide-σ centroid oracle
(`scripts/train_oracle_fgcenter_wide.sh`, σ scale 0.6 / min 4.0 / max 8.0, 120
epochs, output `output/oracle/nuLiteT-fold1-fg-center-wide-t12.pt`) followed by
the already-prepared `main/pannuke-thin-t12-fgmap-fgcentroid-wide.yaml`.

A three-fold rotation on the chosen configuration is the next required step
before any literature claim.

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
