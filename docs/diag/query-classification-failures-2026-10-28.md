# Why queries stay empty — diagnosis log, 2026-10-28

All numbers are the corrected `eval_lsp.py` protocol on the genuinely held-out
Fold3 fold (2,722 tiles, 66,654 dense GT instances), single fold, `last.pt`.
These are diagnostics, not paper claims.

## The failure being explained

Good polygons whose query stays empty dominate the error budget:

| cohort (train34 last) | count |
|---|---:|
| matched | 50,112 |
| **good_suppressed_bg** | **9,764** |
| no_good_polygon | 6,654 |
| wrong_type | 8,172 |

Suppressed GTs have a median raw polygon IoU of 0.70. No row is below 0.50, and
the distribution has no bad tail: p5 0.549, p50 0.705, p95 0.870, with 21% at
IoU >= 0.8. Suppression is 63% below 100 px, 32% at 100-200 px, 15% at
200-400 px, 9% at 400-800 px, 5% above 800 px.

## What it is not

- **Not duplicate ownership.** Only 8.4% of suppressed GTs have any other
  foreground query within 8 px; matched GTs have one 99% of the time.
- **Not query placement.** Forcing a query exactly onto the GT centroid raises
  firing from 8.1% to only 14.3% (train34) / 21.0% (train36).
- **Not missing oracle evidence.** At those centroids the seed value has median
  0.768 and exceeds 0.5 for 73.4% of suppressed GTs.
- **Not gradient starvation.** The loss already produces large gradients for
  misclassified queries.
- **Not cross-attention starvation.** Hooked every decoder layer's deformable
  cross-attention and recomputed its 4x4 sampling points (240 images, 4,520 GT
  rows). **100.0%** of rows in both cohorts have at least one of their 16
  samples inside the GT nucleus; none have zero. Per size, suppressed nuclei
  see a *higher* inside-fraction than matched at the same size. Query distance
  to the GT centroid is identical (5.08 vs 5.09 px).
- **Not additive size confounding.** Regressing log nucleus area out of the
  score-head input makes clustering worse (balanced 0.665 -> 0.644) and halves
  small-nucleus recall (0.644 -> 0.450). A shifted-size control tracks the
  baseline, and an apparent 0.976 was traced to every background row carrying
  a `-1` size sentinel.

## Where the decision is made

Decoder trace with per-layer margins: suppressed queries are already about
-3.0 at the **encoder proposal layer** and never recover across six layers.
The empty decision is committed before any decoder refinement.

## Feature separability (clustering probe)

Captured the exact tensor the proposal score head consumes (`enc_score_head`
input, 256-d content + 1 seed scalar) for 160 Fold3 images, 45,144 queries
(3,561 fg / 41,583 bg), balanced subsample of 7,122.

| measurement | value |
|---|---:|
| k-means (k=2, 32-d PCA) balanced accuracy | 0.665 (shuffled-label null 0.506) |
| k-NN(10) same-class purity | 0.62 |
| trained score head, same queries | 0.589 |
| head fg recall / bg recall | 0.205 / 0.972 |
| head 1-D margin AUC | 0.759 |

The features are moderately separable (+0.16 over chance), and the head's own
scalar margin already ranks fg above bg reasonably (AUC 0.759). The head
discards most of that evidence: it sits at an extreme operating point where
background recall is 0.972 and foreground recall is 0.205.

The seed scalar contributes **nothing** to separability: 0.663 with it versus
0.663 without, and identical k-NN purity. We pay 29% of parameters for an
oracle stream whose signal the classifier ignores.

The size profile is inverted — k-means finds small nuclei better than the head
does, while the head prefers large ones:

| fg area | k-means recall | head recall |
|---|---:|---:|
| 0-100 px | 0.645 | 0.285 |
| 100-200 px | 0.588 | 0.271 |
| 200-400 px | 0.342 | 0.189 |
| 400-800 px | 0.278 | 0.177 |
| >800 px | 0.176 | 0.127 |

## Foreground-map admission test (cached predictions, no training)

Rule tested: keep a query if the head keeps it **or** the oracle seed exceeds
tau. Sweep on the full fold with the cached train34 predictions.

| rule | bPQ | precision | recall |
|---|---:|---:|---:|
| head only (baseline) | 0.6325 | 0.8467 | 0.7583 |
| seed > 6.0 | 0.5393 | 0.6485 | 0.7544 |
| seed > 4.0 | 0.4307 | 0.4762 | 0.7810 |
| seed > 3.0 | 0.3717 | 0.4031 | 0.8033 |
| seed > 0.0 | 0.2413 | 0.2647 | 0.8708 |

The map is not complementary to the head. Even the strictest gate adds 19.5k
predictions, moves recall by -0.004, and costs 0.20 precision. Relaxing tau buys
recall only by collapsing precision. The head's willingness to ignore the seed
scalar is therefore a rational response to a high-recall, low-precision input,
not a bug.

Note the oracle is not a raw binary mask: with `oracle_kind: binary` it emits
`[fg, centroid]` channels trained with an InstemSeg-style centre-weighted
distance-to-boundary target, and the model uses `fg - centroid`. It is already
a peak-shaped field; the defect is specificity, not softness.

## Mechanisms tried and rejected on this contract

| mechanism | held-out Fold3 bPQ | macro bPQ | verdict |
|---|---:|---:|---|
| viable-owner-fg (train34, control) | 0.6325 | 0.6521 | best to date |
| nucleus-CDN with shared class embedding (train35) | 0.6302 | 0.6413 | learned shortcut: reads the answer off `denoising_class_embed` (probe: swapping the class row costs -7.9 logits; DN token and seed exactly 0.000) |
| nucleus-CDN token-only (train37) | 0.6251 | 0.6353 | confirmatory null, no transfer |
| nucleus-CDN real->DN attention open (train38) | — | — | collapsed by epoch 20: real queries depend on a block absent at inference |
| Cayley-STRING RoPE, LSP port (train39) | 0.6301 | 0.6414 | null vs control; mechanism is load-bearing (zeroing positions costs -0.033 bPQ) but replaces absolute PE rather than adding information |
| sliding-tile attention, kernel 3 (train40) | in progress | — | — |
| dense auxiliary head (`aux_dense`) | earlier run | — | no held-out gain; the head was training-only and never fed the decision logits |

Two recurring failure modes explain most of these: **firewalled supervision**
(auxiliary branches whose output never reaches `predict`) and **train/eval
structure mismatch** (train38's DN scaffold).

## Conclusion

The bottleneck is the proposal-layer foreground/empty decision on small nuclei.
The features carry enough signal for roughly 0.66 balanced accuracy and the
head extracts 0.59 by operating at a threshold near the background extreme.
Evidence is present, visible to cross-attention, and not used.

Dead ends to stop revisiting: more margin variants on matched owners, seed
amplification or admission, additive size normalisation, denoising-query
scaffolds, positional-encoding variants, and dense supervision on a detached
side head.

## Oracle map peak counts (closes the local-maximum route)

300 Fold3 images, 3x3 local maxima of the raw seed field, versus 22.8 GT nuclei
per image:

| threshold | peaks/img | ratio to GT | images above 3x GT |
|---|---:|---:|---:|
| > 0.0 | 162.8 | 7.15 | 0.96 |
| > 2.0 | 125.3 | 5.51 | 0.90 |
| > 4.0 | 89.4 | 3.93 | 0.64 |
| > 6.0 | 52.9 | 2.33 | 0.34 |

The field is non-specific: it produces several times more maxima than there are
nuclei even at high thresholds. Local-maximum query selection would spend the
400 query slots on noise, so `query_selection='seed'`/`hybrid'` is not viable
on this map as trained.

Open, untested levers: a proposal-layer re-scoring head trained with
class-balanced sampling, and retraining the oracle map against a centroid
target (`oracle_kind: center_potential`) so nuclei are forced apart.
