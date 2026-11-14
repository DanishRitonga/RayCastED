# Foreground/seed-map ablation sweep (T12, single fold)

Status: **complete** — four arms trained and evaluated (2026-11-14: consumer 4,
the direct seed-map loss, added as an opt-in arm `train61_fgmap-seedloss`); see
`docs/runs/nulite-detr/fg-map-ablation-results.md`. None cleared the adoption
bar, so the control stays the baseline. **The current manuscript uses only the
fg−bg binary plateau map; the centroid-potential map is unused for now** (the
`fgcentroid` arm is kept as a future-work pointer, not as a result). Control is
`train52_fastvit-t12` (9.55M params, held-out Fold3 bPQ 0.6372 / macro bPQ
0.6566 / mPQ 0.4561 / F1 0.7987).

## Why this sweep

The assignment line is closed: four Hungarian-matcher experiments (soft inside
gate, additive empty-column cost at λ=1 and λ=2, conditional empty-column cost)
all landed inside ±0.002 or below it, and the best arm (additive λ=1, bPQ
0.6391) still misses the pre-registered +0.002 adoption bar. `wrong_type`
(974 / 8,917 GT in the audit sample) cannot be moved by any matcher cost because
`p(correct) − p(best_wrong)` has median 0.000 on this head.

`docs/diag/foreground-map-not-used-2026-11-01.md` claims the seed map is
informative only at the exact nucleus centre and is noise at the grid cell
centre (AUC 0.503), and that the head already over-weights the seed column
(0.131 / 0.221 of a content row vs a 0.062 fair share). The previous
interventions that tried to *add* the map to the decision were firewalled or
flat. This sweep does the opposite: it **removes** each map consumer one at a
time and measures what breaks.

## The map has four consumers

An ablation is only meaningful once a consumer is named. From
`raycasted/model/blocks/nulite_detr.py` and
`raycasted/model/nulite_detr_model.py`:

| # | Knob | Path | Touches the decision? |
|---|---|---|---|
| 1 | `detr_seed_in_content: true` | seed logit concatenated to the score head (`score_in = hd + 1`, sampled at the query at `blocks/nulite_detr.py:850`) | **yes — this is the only one** |
| 2 | `detr_seed_level: true` | `centroid_seed` projected by `Conv2d(1, hd, 3, stride=4)` as a 4th deformable-attention level (`:695`) | no (geometry) |
| 3 | `detr_hybrid_peaks: 100` | local maxima of the seed map become queries | no (proposals) |
| 4 | `seed_map_target: true` + `lambda_seg` | L1 to the InstanSeg per-instance EDT target built from GT in the collate fn (`train.py:148-208`) | no (supervision) |

The map *kind* (`binary` plateau, `center_potential`, `fg_centroid`) determines
what channels 1-4 actually receive.

## Arms

All recipes are copies of `main/pannuke-thin-t12.yaml` with the diff below and
nothing else (verified with `ETLConfig`).

| Arm | Recipe | Diff vs control | Tests |
|---|---|---|---|
| control | `main/pannuke-thin-t12.yaml` | — | baseline |
| **nocontent** | `main/pannuke-thin-t12-fgmap-nocontent.yaml` | `detr_seed_in_content: True → False` | does the map drive the fg/empty decision at all? |
| **noseedlevel** | `main/pannuke-thin-t12-fgmap-noseedlevel.yaml` | `detr_seed_level: True → False` | is the 4th attention level load-bearing? |
| **fgcentroid** | `main/pannuke-thin-t12-fgmap-fgcentroid.yaml` | `nulite_ckpt → nuLiteT-fold1-fg-center-t12.pt` **and** `nulite_oracle_kind: binary → fg_centroid` | does a clean, validated two-channel map beat the plateau? |
| fgcentroid-wide *(optional)* | `main/pannuke-thin-t12-fgmap-fgcentroid-wide.yaml` | as above but with the wide-sigma oracle | is the failure map *semantics* or map *sharpness*? |

`fgcentroid` changes two keys but is one mechanism: `nulite_detr_model.py:1520-1522`
raises if the checkpoint's `oracle_kind` disagrees with the model's, so the pair
cannot be separated.

Run names: `train60_fgmap-nocontent`, `train60_fgmap-noseedlevel`,
`train60_fgmap-fgcentroid`, `train60_fgmap-fgcentroid-wide`.

All runs use `--epochs 600 --end 332`, matching the whole train52/55/56/57/58/59
series. Cost is ~6.6 h per detector run (71 s/epoch), so ~20 h for the three
primary arms.

## Oracles

The three primary arms need **no new oracle**:

- control / nocontent / noseedlevel → `output/oracle/nuLiteT-fold1-sep-t12.pt`
- fgcentroid → `output/oracle/nuLiteT-fold1-fg-center-t12.pt`
  (`oracle_kind: fg_centroid`, `fg_dice 0.8103`, `center_f1 0.7487`)

The optional wide arm needs a new checkpoint, prepared by
`scripts/train_oracle_fgcenter_wide.sh`
(`--fg-center-target --center-sigma-scale 0.6 --center-sigma-min 4.0 --center-sigma-max 8.0`,
120 epochs, out `output/oracle/nuLiteT-fold1-fg-center-wide-t12.pt`). Rationale:
the saved centroid maps use σ = clamp(0.25·mean_radius, 2.5, 4.0) px, while a
query grid cell centre can be 12.8 px away; measured on Fold3, the sharp
centroid channels exceed 0.5 at only 2.9-3.5% of cell centres (plateau map:
74%). The wide oracle changes one axis — kernel width — and nothing else.

## Measured prior (why expectations are set this way)

64 held-out Fold3 tiles, 1,934 GT, sigmoid values, sampled at the exact GT
centroid vs the query grid cell centre:

| map | exact centroid median / >0.5 | cell centre median / >0.5 |
|---|---|---|
| plateau (binary, control) | 0.940 / 88.9% | 0.882 / 74.0% |
| center_potential | 0.319 / 27.1% | 0.054 / 2.9% |
| fg_centroid ch0 (fg) | 0.968 / 87.3% | 0.938 / 72.6% |
| fg_centroid ch1 (centroid) | 0.349 / 32.5% | 0.059 / 3.5% |

Local maxima (3×3 max-pool, >0.5): plateau 123 peaks/image with 65.7% of GT
within 3 px; center_potential 10 peaks/image with 49.3% within 3 px.

Prior expectation, stated before the runs: `nocontent` is flat or slightly
worse; `noseedlevel` drops (the geometry consumer is the plausible load-bearing
one); `fgcentroid` is flat or negative because its centroid channel is nearly
dead at cell centres.

## Pre-registered decision rules

Mechanistic reading, not adoption:

1. **`nocontent` within ±0.002 on bPQ and macro bPQ** → the map is not
   load-bearing for the foreground/empty decision. Then no map-quality work at
   the decision level is justified, and the remaining headroom is the assignment
   ceiling and `wrong_type` class separation.
2. **`nocontent` drops > 0.002** → the map is load-bearing; the `fgcentroid`
   delta then says whether the map's target shape can be improved, and the wide
   arm says whether sharpness is the limiter.
3. **`noseedlevel` drop > 0.002** → the 4th attention level matters; report it
   as a geometry result, not a decision result.
4. **`fgcentroid` (or wide) adopted only if bPQ AND macro bPQ both gain
   > +0.002** over the control; reject if bPQ drops > 0.002. This is the same
   bar used for the four matcher arms, so no threshold shopping.
5. Every arm reports the cohort shift for `matched`, `good_suppressed_bg`,
   `no_good_polygon`, `wrong_type`, `unmatched_predictions`, `raw predictions`.

## Confounders to state in the results doc

1. `fgcentroid` also changes the initial weights of the oracle decoder, which is
   thawed after epoch 50; it is not purely "a different input map".
2. `seed_map_target` supervision keeps the InstanSeg EDT target (computed from
   GT masks, not from the checkpoint) in every arm, so the map is not trained
   toward the new oracle's own semantics.
3. Single fold, single seed. Any delta inside ±0.002 is unresolved until the
   3-fold rotation.

## Commands

```bash
# three primary arms (~20 h), one at a time, resumable/skip-if-done
scripts/train_fgmap_sweep.sh

# each arm individually
VARIANTS="nocontent" scripts/train_fgmap_sweep.sh

# optional fourth arm (train the oracle first)
scripts/train_oracle_fgcenter_wide.sh
VARIANTS="fgcentroid-wide" scripts/train_fgmap_sweep.sh

# held-out fold3 evaluation
for v in control nocontent noseedlevel fgcentroid; do
  if [[ $v == control ]]; then W=output/v2/train52_fastvit-t12/weights/last.pt
  else W=output/v2/train60_fgmap-$v/weights/last.pt; fi
  .venv/bin/python -m raycasted.evals.eval_lsp \
    --weights $W \
    --data-dir output/v2/transformed/test \
    --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
    --batch 8 --device 0 --workers 0 \
    --save-predictions output/evals/fgmap/$v.npz
done
```

Run `eval_lsp.py` as the sole evaluator: held-out Fold3 parquet, raw `nc+1`
all-class argmax, no confidence threshold, no NMS, overlap resolved by greatest
Euclidean interior distance. Validator bPQ is trajectory signal only.
