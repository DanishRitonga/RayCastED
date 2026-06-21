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

## 2. Inter-Scale Competition — Cross-Scale Duplicate Suppression (train49)

**Problem:** Same nucleus predicted at both P2 (64×64) and P3 (32×32) with high confidence.
No mechanism forces scales to compete for predictions.

**Fix:** Softmax across P2/P3/P4 at inference (temperature=0.5). Each spatial position's score
is weighted by its relative strength vs the same position at other scales.
Differentiable, no post-processing.

**Impact:** mAP50 +0.013 (0.517 → 0.530).

**Why it works:** Scales compete directly in score space. The winning scale gets weight ~1.0,
losers get ~0. Natural NMS-free cross-scale suppression.

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

## 5. `assigner_beta=2.0` — Assignment pIoU Dominance Fix (train74)

**Problem:** `align = cls^0.5 × pIoU^6.0` made assignment overwhelmingly pIoU-dominated.
Two anchors 4px apart both get pIoU>0.9 on near-circular nuclei — a 2% pIoU difference
amplifies to 14% with β^6. The cls head with `bg_fg_ratio_o2o=0` only sees ~28 "best"
anchors per image and never learns that nearby anchors are also acceptable. This causes
spatial hallucination: model fires at wrong locations because it was never told those
locations could be correct.

**Fix:** `assigner_beta: 6.0 → 2.0`. Single line in pannuke.yaml. Both o2m and o2o branches.

**Impact:** mAP50 **0.530 → 0.544** (new FCN record). bDQ **0.628 → 0.660**. Recall **0.500 → 0.534**.
bPQ **0.524 → 0.549**. Perfect pred/GT calibration (0.99x). Small nuclei recall +1.5pp (38.3% →
39.8%), medium +4.8pp, large +5.9pp.

**Why it works:** Lower β lets the cls term (α=0.5) actually matter in anchor selection.
More anchors per GT get positive signal, the cls head learns a broader spatial prior,
and the model stops hallucinating at unlearned locations.

**Reference:** train74, `docs/runs/train74.md`

---

## 6. Hierarchical CLS + β=2.0 — Combined Effect

Combining §4 (hierarchical CLS) + §5 (β=2.0) produces the best FCN results to date:

| Metric | Standard (t49) | Hier. CLS (t71) | Hier. CLS + β=2.0 (t74) |
|--------|---------------|-----------------|--------------------------|
| mAP50 | 0.530 | 0.527 | **0.544** |
| bDQ   | 0.322 | 0.628 | **0.660** |
| bPQ   | 0.491 | 0.524 | **0.549** |
| bSQ   | 0.756 | 0.781 | 0.778 |
| Recall | 0.533 | 0.500 | **0.534** |

---

## 7. CLS-Only TAL — Self-Reinforcing Cls Assignment (train80, pending)

**Problem:** pIoU-dominated TAL (even at β=2.0) still leaves ~90 GTs with zero positive
assignments (confirmed by STAL diagnosis). Small nuclei with best-anchor-offset >30% of
radius get pIoU≈0.5 — the assigner won't pick them as positive even when the cls head
has high confidence.

**Fix:** O2o assigner uses `align = cls^α × gaussian_decay` — no pIoU at all.
Picks anchors purely based on what the cls head already believes, gated by spatial
proximity (Gaussian decay with σ ∝ GT radius). O2m branch unchanged (pIoU as normal).

**Expected:** Self-reinforcing feedback loop — higher cls scores → better assignment →
even higher cls. The ~90 zero-positive GTs get anchor assignments wherever cls has
any fg signal within spatial range. Small nuclei recall should increase.

**Why it works:** Removes the geometric bottleneck entirely. Cls head already has
meaningful (if weak) fg/bg discrimination from hierarchical binary head's 28.6x gap.
The assigner just needs to trust it.

---

## Summary Table

| Change | Train | mAP50 Δ | bDQ Δ | What it fixes |
|--------|-------|---------|-------|---------------|
| `bg_fg_ratio_o2o=0` | t23 | +0.106 | — | Overprediction (7.5x → 0.94x) |
| Inter-scale competition | t49 | +0.013 | — | Cross-scale duplicates |
| topk2=3→1 anneal | t49 | incl. | — | Early convergence |
| Hierarchical CLS | t71 | -0.003 | **+0.306** | fg/bg discrimination |
| `assigner_beta=2.0` | t74 | **+0.014** | **+0.032** | Spatial anchor misalignment |
| STAL (stal_min_positives=3) | t76 | -0.002 | +0.013 | Small nuclei recall (+1.5pp) |
| NWD [REMOVED] | t78 | -0.015 | -0.056 | Too smooth, killed |
| CLS-only TAL | t80 | TBD | TBD | Zero-positive GT blockade |

## Cumulative Best Values

| Metric | Value | Train |
|--------|-------|-------|
| mAP50 | 0.544 | t74 |
| bDQ | 0.673 | t76 |
| bPQ | 0.550 | t75 |
| bSQ | 0.781 | t71 |
| Recall | 0.534 | t74 |
| Overprediction | 0.94x | t23, t49 |

## Dead Ends (Do Not Repeat)

- **NWD** [REMOVED] (t78): Smoother than pIoU — two anchors 4px apart on 4px nucleus both get NWD>0.8, assigner can't pick winner. pIoU's sharper cliff creates genuine competition. Code removed in cleanup.
- **Suppress loss** (t24): O(N²) pairwise repulsion too aggressive for dense nuclei. Inter-scale softmax (§2) handles this better.
- **2-stage threshold** (t72): Binary gate is no-op with 28.6x fg/bg gap. Product inference is equivalent and simpler.
- **PredictionRefinementAttention**: Removed in cleanup. Implemented but never trained successfully.
- See `docs/effective_techniques.md` for the full list of dead ends.
