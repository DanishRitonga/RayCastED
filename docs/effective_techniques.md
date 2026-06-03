# Effective Techniques — What Works and Why

Chronological record of techniques that improved the NMS-free FCN o2o branch.
All verified experimentally. Dead ends omitted.

---

## 1. `bg_fg_ratio_o2o=0` — Overprediction Fix (train23)

**Problem:** Standard cls BCE sampled 3 bg anchors per fg anchor. O2o branch has only ~28 fg per
image (topk2=1) vs 5348 bg. The bg BCE gradient overwhelmed fg signal, model couldn't
discriminate → 7.5x overprediction (750k preds vs 66k GT).

**Fix:** Zero bg sampling. O2o cls head only sees 28 fg anchors per image.

**Impact:** mAP50 +0.106 (0.411 → 0.517), overprediction 7.5x → 0.94x.

**Why it works:** Fewer fg anchors need individually stronger cls signal per anchor. Every sample counts.

**Reference:** train23, `docs/runs/train23.md`

---

## 2. Inter-Scale Competition — Cross-Scale Duplicate Suppression (inference-only)

**Problem:** Same nucleus predicted at both P2 (64×64) and P3 (32×32) with high confidence.
No mechanism forces scales to compete for predictions.

**Fix:** Softmax across P2/P3/P4 at inference (temperature=0.5). Each spatial position's score
is weighted by its relative strength vs the same position at other scales.
Differentiable but only applied in `_inference`, NOT during training forward pass.

**Impact:** Eval metrics improved but the +0.013 attributed to this in train49 was confounded
with topk2=3→1 annealing (also added in train49). Competition itself is inference-only — it
does not change training dynamics. The eval improvement is real (better cross-scale dedup at
inference) but the training curve was unchanged.

**Why it works:** Scales compete directly in score space at inference. The winning scale gets
weight ~1.0, losers get ~0. Natural NMS-free cross-scale suppression.

**Config:** `inter_scale_competition: true`, `inter_scale_temperature: 0.5`

**Reference:** train49, `docs/runs/train49.md`

---

## 3. topk2=3→1 Annealing — Better Early Gradient (train49)

**Problem:** topk2=1 from epoch 0 = only 28 fg anchors/image. Too few to learn meaningful
fg/bg separation early in training.

**Fix:** Start topk2=3, linearly anneal to 1 over epochs 67-200. Even though topk2=1 at
inference, training starts permissive and tightens as the head learns.

**Impact:** Contributed to train49's +0.013 mAP50 improvement. Cleaner convergence curve.

**Why it works:** More fg assignments early = more cls gradient when the model needs it most.
Annealing to 1 preserves 1:1 exclusivity at convergence.

**Reference:** train49, `docs/runs/train49.md`

---

## 4. Hierarchical CLS Head — fg/bg Discrimination (train71)

**Problem:** Standard 5ch cls head does everything in one layer. Fg signal is split across
5 channels — each class gets ~5 fg anchors per image. The head can't specialize in
presence (fg/bg) separately from identity (cell type).

**Fix:** Two heads:
- **Binary head** (1ch): Focal BCE on ALL 5376 anchors. Learns "is this a nucleus?".
  All 28 fg push the single channel to 1.0. Normalized by fg_count (~28) not B×N (86016).
  Input features detached (stop-grad) to prevent backbone flooding.
- **Class head** (nc ch): Softmax CE on 28 fg anchors only. Learns "neoplastic vs inflammatory...".
  Gets normal backbone gradient.

**Impact:** bDQ 0.322 → **0.628** (nearly 2x). Val fg/bg gap 2.5x → **28.6x**.
bSQ 0.781 (only 3.2% behind LSP-DETR's 0.807). Detection remains bottleneck but gap halved.

**Caveat:** Recall dropped (0.533 → 0.500) because `sigmoid(binary) × softmax(class)` at
inference multiplies — borderline TPs get over-suppressed. Fixed in §5.

**Why it works:** Binary head gets 5x concentrated gradient per parameter (single channel,
all 28 fg push it to 1.0) vs standard 5ch (each channel gets ~5 fg). Decomposition is correct:
`P(nucleus) × P(type | nucleus)`.

**Reference:** train71, `docs/runs/train71.md`

---

## 5. Lower `assigner_beta` — Assignment pIoU Dominance Fix (train74, train80)

**Problem:** `align = cls^0.5 × pIoU^6.0` made assignment overwhelmingly pIoU-dominated.
Two anchors 4px apart both get pIoU>0.9 on near-circular nuclei — a 2% pIoU difference
amplifies to 14% with β^6. The cls head with `bg_fg_ratio_o2o=0` only sees ~28 "best"
anchors per image and never learns that nearby anchors are also acceptable. This causes
spatial hallucination: model fires at wrong locations because it was never told those
locations could be correct.

**Fix:** Lower β. First tested β=2.0 (train74), then β=0 / CLS-only TAL (train80).
Blending pIoU at beginning or end of training didn't help — pure cls-only assignment is
consistently better.

**Impact (β=6→2):** mAP50 **0.530 → 0.544**. bDQ **0.628 → 0.660**. Recall **0.500 → 0.534**.
bPQ **0.524 → 0.549**. Perfect pred/GT calibration (0.99x).

**Impact (β=2→0, CLS-only TAL):** Further recall improvement. Self-reinforcing feedback
loop — higher cls scores → better assignment → even higher cls. Small nuclei with
best-anchor-offset >30% of radius get pIoU≈0.5 under β>0, but cls-only TAL assigns
them wherever the cls head has any fg signal within spatial range.

**Why it works:** Lower β lets the cls term (α=0.5) actually matter in anchor selection.
More anchors per GT get positive signal, the cls head learns a broader spatial prior,
and the model stops hallucinating at unlearned locations. Going all the way to β=0
removes the geometric bottleneck entirely — the assigner trusts what the cls head
already believes, gated only by Gaussian spatial proximity.

**Reference:** train74, train80, `docs/runs/train74.md`

---

## 6. Hierarchical CLS + Lower β — Combined Effect

Combining §4 (hierarchical CLS) + §5 (lower β) produces the best FCN results to date:

| Metric | Standard (t49) | Hier. CLS (t71) | + β=2.0 (t74) | + CLS-only TAL (t80) |
|--------|---------------|-----------------|---------------|----------------------|
| mAP50 | 0.530 | 0.527 | 0.544 | **0.540** |
| bDQ   | 0.322 | 0.628 | 0.660 | — |
| bPQ   | 0.491 | 0.524 | 0.549 | **0.542** |
| bSQ   | 0.756 | 0.781 | 0.778 | — |
| Recall | 0.533 | 0.500 | 0.534 | **higher** |

---

## 7. CLS-Only TAL — Self-Reinforcing Cls Assignment (train80) ✓

**Problem:** pIoU-dominated TAL (even at β=2.0) still leaves ~90 GTs with zero positive
assignments (confirmed by STAL diagnosis). Small nuclei with best-anchor-offset >30% of
radius get pIoU≈0.5 — the assigner won't pick them as positive even when the cls head
has high confidence.

**Fix:** O2o assigner uses `align = cls^α × gaussian_decay` — no pIoU at all.
Picks anchors purely based on what the cls head already believes, gated by spatial
proximity (Gaussian decay with σ ∝ GT radius). O2m branch unchanged (pIoU as normal).

**Impact:** Recall improvement over β=2.0. Blending pIoU (either at start or end of
training) didn't help — pure cls-only is consistently better. mAP50 ~0.540.

**Why it works:** Removes the geometric bottleneck entirely. Cls head already has
meaningful (if weak) fg/bg discrimination from hierarchical binary head's 28.6x gap.
The assigner just needs to trust it. Self-reinforcing: higher cls → better assignment →
even higher cls. The ~90 zero-positive GTs get anchor assignments wherever cls has
any fg signal within spatial range.

---

## 8. Soft Cascade — Binary-Weighted Class Loss

**Problem:** With hierarchical cls, the class head (multiclass) sees ~28 fg anchors
per step but the binary head (fg/bg) sees all 5,376. The binary head trains easily
(5,376 examples, focal BCE) but the class head struggles — 28 anchors for 5 classes,
tiny gradient signal. The backbone only gets class gradient from these 28 anchors.

**Fix:** Weight class loss by binary head's sigmoid confidence (detached).
High-confidence fg anchors get full class gradient. Ambiguous anchors get dampened.
This acts as soft curriculum learning — the class head focuses on "clearly fg" anchors
where its gradient signal is most useful.

**Gradient flow:**
- Binary head: trains from focal BCE on all 5,376 anchors (detached from backbone)
- Class head: trains from CE × binary_weight on fg anchors → backbone gets gradient
- No cross-talk: binary_weight is detached, so class loss doesn't train binary head

**Implementation:** `loss.py` line ~723, gated by `hierarchical_soft_cascade: true` in config.

---

## Summary Table

| Change | Train | mAP50 Δ | bDQ Δ | What it fixes |
|--------|-------|---------|-------|---------------|
| `bg_fg_ratio_o2o=0` | t23 | +0.106 | — | Overprediction (7.5x → 0.94x) |
| Inter-scale competition | t49 | +0.013 | — | Cross-scale duplicates |
| topk2=3→1 anneal | t49 | incl. | — | Early convergence |
| Hierarchical CLS | t71 | -0.003 | **+0.306** | fg/bg discrimination |
| `assigner_beta: 6→2` | t74 | **+0.014** | **+0.032** | Spatial anchor misalignment |
| CLS-only TAL (β=0) | t80 | further | — | Zero-positive GT blockade |
| STAL (stal_min_positives=3) | t76 | -0.002 | +0.013 | Small nuclei recall (+1.5pp) |
| NWD | t78 | -0.015 | -0.056 | Too smooth, killed |

## Cumulative Best Values

| Metric | Value | Config |
|--------|-------|--------|
| mAP50 | 0.540 | log-cls TAL (β=0, hierarchical_cls) |
| mAP50-95 | 0.398 | log-cls TAL |
| bDQ | 0.689 | log-cls TAL |
| bPQ | 0.542 | log-cls TAL |
| bSQ | 0.764 | log-cls TAL |
| Recall | 0.537 | log-cls TAL |
| Overprediction | 0.94x | t23, t49 |

## 9. LSP-DETR Bias Init — Minimum Plausible Radius (train71+)

**Problem:** Default ray bias=0 → `softplus(0)=0.69` → model starts by predicting ~0.69 normalized
radius for every ray. But small nuclei have radii ~3px (at 256px crops), and the model must
learn to both expand (for large nuclei) and shrink (for background) from a mid-range starting
point. Conflicting shrink/expand signals slow convergence.

**Fix:** Initialize ray biases so `softplus(bias) = min_nucleus_diameter / (2 * crop_size * mpp)`.
For PanNuke: `3.5 / (2 × 256 × 0.25) = 0.0273`. This starts rays at the smallest plausible
nucleus radius — gradient is unidirectional (expand only) for most real nuclei.

**Impact:** Cleaner convergence from epoch 1 (xy_loss drops faster). bSQ reaches 0.75+ by epoch 5
(vs oscillating with default init).

**Why it works:** Inspired by LSP-DETR. Starting at minimum radius eliminates the conflicting
gradient signals that occur when initialized at average radius — some GTs need expansion while
others need contraction. With min-radius init, nearly all GTs need expansion → consistent gradient.

**Reference:** `head.py:676-701`, `pannuke.yaml: native_mpp=0.25, min_nucleus_diameter_um=3.5`

---

## 10. Bilinear Inter-Scale Competition (Training) — Score-Level Competition Without Learnable Parameters

**Problem:** PixelShuffle competition (trainable projection heads) regressed mAP50 from 0.540 to
0.370 (-17%). The projection heads learn nc-dimensional sub-grid scores from raw features, but
these learned scores have different magnitude/distribution than the trained cv3 scores → LayerNorm
can't fix the mismatch during training.

**Fix:** Bilinear upsampling competition applied during training in `forward_head` (o2o branch only).
P3/P4 scores are upsampled to P2 resolution via bilinear interpolation, softmax across 3 scales
(temperature=0.5), competition weights applied back at native resolution. No new parameters.

**Impact:** Used in best runs (train74/80). When combined with hierarchical CLS + CLS-only TAL,
contributes to the 0.540 mAP50 baseline.

**Why it works:** Uses the same scores the model already computed — just upsampled for comparison.
No magnitude mismatch, no learnable parameters to go wrong. The competition signal tells each
scale "how confident are you relative to the other scales at this position" — naturally suppresses
cross-scale duplicates.

**Config:** `inter_scale_competition: true`, `inter_scale_pixel_shuffle: false`, `inter_scale_temperature: 0.5`

**Reference:** `head.py:446-471` (`_apply_bilinear_competition`)

---

## Dead Ends (Do Not Repeat)

- **NWD** (t78): Smoother than pIoU — two anchors 4px apart on 4px nucleus both get NWD>0.8, assigner can't pick winner. pIoU's sharper cliff creates genuine competition.
- **Suppress loss** (t24): O(N²) pairwise repulsion too aggressive for dense nuclei. Inter-scale softmax (§2) handles this better.
- **2-stage threshold** (t72): Binary gate is no-op with 28.6x fg/bg gap. Product inference is equivalent and simpler.
- **PredictionRefinementAttention**: Implemented but never trained (prediction_refinement_weight=0.0).
- **Feature Bank** (contrastive EMA prototypes): Three experiments, all hurt mAP50 by ~10pts (-8.9 to -10.5). Single prototype per class is too naive — big/small nuclei within same class (e.g., inflammatory) have very different feature vectors. EMA average lands in no-man's-land. Warmup didn't help. bPQ improved ~1.7pts but detection regressed badly (recall 0.537→0.365). Root cause: PanNuke has 74% of Dead cells as small fragments, 25% of Inflammatory cells small — these create noisy prototypes that corrupt the feature space for all anchors. See `docs/plan/feature_bank_plan.md` for the full design.
- **PixelShuffle competition** (trainable projection heads): mAP50 0.540→0.370 (-17%). Projection heads learn nc-dim sub-grid scores from raw features, but magnitude/distribution mismatch with trained cv3 scores causes competition weights to suppress P3/P4. LayerNorm + Xavier init didn't help. Bilinear upsampling (§10) works better with zero extra parameters.
