# Undocumented diagnostic findings (2026-10-29 / 10-30)

These findings were established by read-only probes whose caches lived in
`/tmp/opencode/rc_full` (a RAM-backed tmpfs). The caches are disposable; the
numbers below are the record. Companion to
`query-classification-failures-2026-10-28.md`, which covers the suppression
wall itself.

All numbers are held-out Fold3 (`eval_lsp.py`, 2,722 tiles / 66,654 GT) unless
stated. `train34` = `output/v2/train34_viable-owner-foreground/weights/last.pt`
(two-stream control, 24.7 M); `t44` = `output/v2/train44_frozen-thaw-full`
(frozen U-Net + fusion adapters + full thaw, 11.7 M).

## 1. Oracle-off ablation (train34 weights, oracle zeroed at inference)

Approach: wrap `model.seg_encoder` in an `nn.Module` returning `zeros_like`
(keeps shapes valid; `seed_level` degrades via the existing zero-substitution
in `_get_encoder_input`, `seed_in_content` scalar becomes 0). Caveat: the model
never saw a zero map, so magnitude conflates dependency with distribution
shift; direction is informative.

| | bPQ | bDQ | bSQ | mPQ |
|---|---:|---:|---:|---:|
| train34 baseline | 0.6325 | 0.7759 | 0.8009 | 0.4476 |
| oracle zeroed | 0.6044 | 0.7523 | 0.7849 | 0.4231 |
| delta | **−0.0281** | −0.0236 | −0.0160 | −0.0245 |

Cohort delta: `no_good_polygon` **+887** (6,654 → 7,541), `matched` −1,216,
suppression only **+342** (9,764 → 10,106), `wrong_type` −255.

Conclusion: the oracle is load-bearing, but its contribution is **geometric**
(4th deformable level supplying spatial evidence), not foreground detection.
The suppression wall is independent of it.

## 2. Feature ceiling: the suppression wall is a decision-rule problem

Capture: hook `detr.enc_score_head` on t44, 240 tiles, the exact 257-d proposal
input (256 content + 1 seed scalar). Cohorts matched vs `good_suppressed_bg`,
nearest grid query to GT centroid. N = 4,896 (4,136 matched / 760 suppressed);
median area 550.8 vs 255.3 px.

| probe | AUC | balAcc | recall | tnr |
|---|---:|---:|---:|---:|
| trained head (margin > 0) | 0.8489 | 0.788 | 0.650 | **0.926** |
| linear probe, 257-d | **0.9124** | 0.785 | 0.931 | 0.638 |
| MLP 512-512, 257-d | 0.8967 | **0.810** | 0.939 | 0.681 |
| linear, content only (256) | 0.9122 | 0.780 | 0.929 | 0.630 |
| MLP + log-area + ref-dist | 0.9028 | 0.802 | — | — |

Size confound controls: log-area alone AUC 0.7784 on the full cohort; within
size buckets 0.578 / 0.461 / 0.651 / 0.543 / 0.688. **Size-matched subset**
(760 vs 760, median 178 vs 183 px, log-area AUC 0.5056 = chance): linear 257
**0.8918**, MLP **0.8914**.

Conclusion: neither parameters nor feature discernment is limiting. The
features hold AUC ≈ 0.91 (≈0.89 with size fully controlled), more capacity does
not help (MLP ≤ linear), and extra geometry features add nothing. The trained
head has good AUC but a badly chosen operating point — recall 0.650 /
tnr 0.926, i.e. over-eager to say empty — while a balanced probe on the same
features reaches recall ≈ 0.93. Root cause: one-to-one Hungarian training
rewards precision on the majority matched class. **The binding constraint is the
decision rule / loss calibration.** This motivated `detr_proposal_bal_weight`
(class-balanced focal at the encoder proposal layer, implemented separately).

Also from the same capture: the seed scalar contributes literally nothing
(content+seed 0.663 vs content-only 0.663 in the earlier clustering probe).

## 3. Missed nuclei are inside the sampling footprint (t44)

Probe: hook every decoder layer's `cross_attn`, replicate
`_ScheduledMSDeformAttn.forward` sampling math
(`loc = refer_bbox + sampling_offsets / shapes`), 200 tiles / 4,395 GT.
Final layer = 8 heads × 4 levels × 4 points = 128 sampling points per query.
Each GT assigned the nearest grid query.

| cohort | n | median area | any point inside | mean frac of 128 inside | ref→centroid |
|---|---:|---:|---:|---:|---:|
| matched | 3,336 | 462 px | **100%** | 0.361 | 1.78 px |
| good_suppressed_bg | 617 | 183 px | **100%** | 0.310 | 2.33 px |
| no_good_polygon | 428 | **45.5 px** | 98.8% | **0.164** | **4.72 px** |

Per-size mean frac-inside (suppressed): <100 px 0.254, 100–200 0.294,
200–400 0.330, 400–800 0.356, >800 0.472 — **at small sizes suppressed nuclei
are sampled more densely than matched ones** (matched <100 px = 0.145).

Conclusion: suppression is **not** a visibility or sampling problem (100% of
suppressed GTs have a point inside; density is if anything higher than for
matched nuclei). Centroid distance is ~1.2 px everywhere, so placement is fine.
`no_good_polygon` *is* under-sampled, but as a **scale** effect: median GT is
45.5 px, so a ~3 px sampling spread straddles the boundary — a coverage
precondition failure, not misplacement.

## 4. Proposal re-scoring features: map-derived features are at chance

Capture: hook `detr.enc_score_head` (train34), 220 tiles. Compared the 256-d
content against map-derived features sampled at grid cell centres: the seed
scalar, 3×3 max-pool local max, and multi-scale seed patches (raw + avg-pool
3/5/9/13). Usable subset N = 4,291 (3,631 matched / 660 suppressed; note the
size mismatch, median 474 vs 192 px, so treat AUCs as relative).

| features | AUC (single) |
|---|---:|
| seed scalar at cell centre | **0.503** |
| 3×3 local max | 0.515 |
| patch raw / avg3 / avg5 / avg9 / avg13 | 0.504 / 0.494 / 0.497 / 0.510 / 0.526 |
| seed + patch5 + local max (linear) | 0.544 |
| content 256 (linear) | **0.722** |
| content + seed | 0.722 |
| content + patch5 + local max | 0.725 |

MLP (256-256, 70/30, two seeds): content 0.622–0.637; content+patch+lmax
0.616–0.619; content+log-area 0.715–0.724.

Conclusion: **the oracle map carries no information about this decision at the
location the model consumes it** (AUC 0.50–0.54; MLP 0.57 even with five
features jointly). Contrast the earlier finding that sampled at the *exact GT
centroid* the same seed has median 0.768 (73% > 0.5): the signal is localised
precisely at the nucleus centre, and a grid query up to 12.8 px away loses it.
This is the direct explanation for the harmful union-admission sweep and for the
optimiser zeroing `seed_in_content`. Content carries the real signal.

## 5. Centroid regression from the seed map: bounded by peak coverage

Data: 120 tiles, t44; 3,136 GTs. Fit a small MLP (64-64, GELU, Adam 1e-3) to
predict GT centroid − local peak offset from a 7×7 map patch, 70/30 split,
N = 1,384 GTs that have a peak within 6 px.

| features | mean err px | median | p90 | frac <0.5 px |
|---|---:|---:|---:|---:|
| patch only | 1.345 | 0.816 | 3.198 | 0.315 |
| patch + log-area | 1.209 | 0.713 | 2.856 | 0.368 |
| patch + log-area + peak distance | **0.797** | **0.502** | 1.715 | 0.495 |
| do nothing (use the peak) | 1.490 | 0.517 | 4.519 | 0.483 |

Coverage is the binding constraint — fraction of GTs with a local max (3×3
max-pool, τ>2) within d px: 2 px 0.307, 4 px 0.375, 6 px **0.441**, 8 px 0.497,
12 px 0.613. By size (median nearest-peak distance, ≤4 px, ≤8 px): <100 px
10.53 / 0.239 / 0.395; 100–200 8.55 / 0.328 / 0.470; 200–400 6.03 / 0.441 /
0.553; 400–800 7.56 / 0.398 / 0.506; >800 5.43 / 0.454 / 0.556.

Conclusion: regressing the exact centroid from a map patch works well
(median error 0.50–0.80 px), but the map has a local maximum within 6 px of
only 44% of nuclei and small nuclei are the worst (median 10.5 px for <100 px).
A peak-anchored regressor therefore cannot reach the nuclei that currently
fail. A dense per-pixel regressor would need its own dedup, i.e. the same
one-to-one problem again.

## 6. Seed-map drift in the shared-stream variant (train41, epoch ~4)

96 tiles; compared train41's trainable seed map against train34's frozen
oracle map.

| measure | value |
|---|---|
| raw MAE / correlation | 3.01 / 0.622 |
| shape corr after per-image z-norm | 0.574 |
| fg IoU at τ>0 | **0.359** |
| cur fg inside ref fg | 0.679 |
| ref fg covered by cur | 0.491 |
| fg fraction (cur vs ref) | 0.127 vs 0.176 (shrunk 28%) |
| peaks τ>2 per image | **1,025 vs 126 (8×)** |
| reference vs pixel-shuffled MAE (noise floor) | 4.301 (cur-vs-ref 3.01 is below it) |

Conclusion: not decorrelation to noise (spatial smoothness is unchanged and the
MAE is below the shuffle floor), but a systematic contraction toward background
plus fragmentation into ~8× more, weaker peaks. This is exactly why the shared
variant needed `lambda_seg` restored to 1.0 (L1 anchor against the oracle
target) rather than being left unsupervised.

## 7. Why shared decoder layers help suppression (t44 vs t45)

Per-GT final-layer and per-layer margins, 250 tiles. t45 = 2 shared layer
groups (11.256 M) vs t44 = 6 distinct layers (11.7 M). Both have identical
architecture otherwise.

Per-layer fire rate (margin > 0) and median margin:

| layer | t44 fire | t45 fire | t44 med | t45 med |
|---|---:|---:|---:|---:|
| L1 | 0.3808 | 0.3802 | −1.242 | −1.175 |
| L2 | 0.4120 | 0.4302 | −1.074 | −0.871 |
| L3 | 0.4239 | 0.4417 | −1.026 | −0.787 |
| L4 | 0.4276 | 0.4511 | −1.091 | −0.665 |
| L5 | 0.4477 | 0.4571 | −0.932 | −0.631 |
| L6 | 0.4945 | **0.5132** | −0.127 | **+0.214** |

Mechanism: **L6 margin std 8.282 → 5.601 (−32%)**, IQR 15.16 → 10.04 — t45's
logits are compressed toward zero while firing more GTs. This is a
**calibration** change, not new evidence: L1 (proposal) fire rates are
statistically identical (0.3808 vs 0.3802) and size-stratified L1 deltas are
noise.

Size-stratified final-layer fire rate (decisive):

| size | n44 | t44 | n45 | t45 | delta |
|---|---:|---:|---:|---:|---:|
| 0–100 px | 680 | 0.2647 | 682 | 0.2566 | −0.008 |
| 100–200 | 678 | 0.4012 | 682 | 0.3915 | −0.010 |
| 200–400 | 1,346 | 0.5431 | 1,348 | 0.5593 | +0.016 |
| 400–800 | 1,858 | 0.5571 | 1,855 | **0.5898** | +0.033 |
| >800 | 892 | 0.5370 | 891 | **0.5735** | +0.037 |

Conclusion: t45's cohort gain (suppression −670, matched +619) is **entirely in
large nuclei**; small nuclei (<200 px) got slightly worse. The 670 recovered GTs
are mostly large, and the gain is paid for with +1,402 unmatched predictions
(precision −0.020). The dominant small-nucleus wall is unchanged. Shared layers
act as score-head compression/regularisation, not as a small-nucleus fix.

Truncation probe (128 tiles, centroid F1) explains why the shared weights still
work: L1→L2 is the biggest jump (+0.034 / +0.029), L5→L6 second (+0.025 /
+0.021), L2–L5 together ≤ +0.008, and t45 is uniformly −0.004..−0.009 at every
truncation point (a mild global degradation, not one failing layer). Repeated
weights are not no-ops because their input changed. Adjacent full-layer weight
cosine: t44 L0–L1 0.6146, L1–L2 0.6622 (mean ≈ 0.64); t45's two distinct bases
cosine 0.6050 — so the shared groups did not collapse into near-duplicates.
`dec_bbox_head` cosine vs L0 ≈ 0.005–0.019 in both (independent, zero-init).

## Disposition

Caches for all of the above (`rescore_capture.npz`, `missed_deform_t44.pkl`,
`seed_drift_ep4.pt`, `centroid_maps.npz`, `feat_ceiling.npz`,
`layermarg_t4{4,5}.json`, `pred_t34_nooracle.npz`) are deleted after this record.
The evaluation prediction caches (`pred_train*.npz`) and cohort CSVs are kept
only as long as their metrics are still needed for comparison; the metrics
themselves are in `HANDOVER.md` and the run documents.
