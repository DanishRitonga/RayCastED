# Train74 — assigner_beta: 6.0 → 2.0 (Hierarchical CLS + Train71 Setup)

**Date:** 2026-05-29
**Status:** SUCCESS — new FCN mAP50 record, medium/large nuclei fixed
**max_epochs:** 400 (stopped at 329, best epoch 229)

## Hypothesis

`assigner_beta=6.0` made the TAL assignment overwhelmingly dominated by pIoU geometry. With near-circular PanNuke nuclei, two anchors 4px apart both get pIoU>0.9 — a 2% pIoU difference amplifies to 14% assignment gap with β^6. The cls head with `bg_fg_ratio_o2o=0` only sees ~28 "best" anchors per image and never learns that nearby anchors are also acceptable. This causes the spatial hallucination problem: model fires at wrong locations because it was never told those locations could be correct.

Lowering β=2.0 makes assignment ~4% gap instead of 14% for the same anchors — the cls term (α=0.5) actually matters, and more anchors per GT get positive signal.

## Config Changes from Train71

| Parameter | Train71 | Train74 |
|-----------|---------|---------|
| `assigner_beta` | 6.0 | **2.0** |
| `hierarchical_cls` | true | true |
| `hierarchical_cls_detach` | true | true |
| Everything else | same | same |

## Key Hyperparameters

- `assigner_beta: 2.0` (was 6.0) — single line change in pannuke.yaml
- Affects both o2m and o2o branches (shared assigner parameter)
- Hierarchical cls: sigmoid(binary) × softmax(class), detach=true
- Inter-scale competition temp=0.5, topk2=3→1

## Validation Results (best epoch 229)

| Metric | Train74 | Train71 | Train49 | LSP-DETR |
|--------|---------|---------|---------|----------|
| mAP50 | **0.544** | 0.527 | 0.530 | 0.691 |
| mAP50-95 | **0.413** | — | 0.403 | — |
| Precision | 0.554 | — | 0.550 | — |
| Recall | 0.534 | 0.500 | 0.533 | 0.791 |
| bPQ | **0.549** | 0.524 | 0.491 | 0.657 |
| bSQ | 0.778 | 0.781 | 0.756 | 0.807 |
| bDQ | **0.660** | 0.628 | 0.322 | 0.803 |
| o2o fg/bg gap | **66x** | 28.6x | 2.5x | — |

mAP50=0.544 is the new FCN record (+0.014 over train49, +0.017 over train71).

## Eval Results (main/eval_pannuke.py, conf=0.24)

| Metric | Value |
|--------|-------|
| AJI | 0.5302 |
| AP@0.5 | 0.3689 |
| AP@0.7 | 0.2766 |
| AP@0.9 | 0.0084 |
| AP@0.5:0.05:0.95 | 0.2071 |
| bPQ | 0.5294 |
| bMPQ | 0.6696 |
| mPQ | 0.3915 |
| mMPQ | 0.4537 |
| F1 | 0.6549 |
| Precision | 0.6576 |
| Recall | 0.6522 |
| Preds vs GT | 65,261 / 65,848 (0.99x) |

Perfect overprediction calibration at 0.99x.

## Recall Diagnosis by Size (conf=0.24)

| Size Bin | Train74 | Train71 | Gap |
|----------|---------|---------|-----|
| Small (<228px²) | 39.8% | 38.3% | +1.5pp |
| Medium (228-475px²) | 74.2% | 69.4% | +4.8pp |
| Large (>475px²) | 81.4% | 75.5% | +5.9pp |

## Recall Diagnosis by Class (conf=0.24)

| Class | Recall |
|-------|--------|
| Neoplastic | 71.1% |
| Inflammatory | 68.0% |
| Connective | 67.1% |
| Dead (Necrosis) | 49.5% |
| Epithelial | 66.3% |

## Unmatched GT Breakdown (conf=0.24)

- <5px (near miss): 10.2%
- 5-12px (drifted): 5.0%
- >12px (truly missed): 84.1% — no prediction fires near these GTs
- No predictions at all: 0.7%
- Mean conf within 12px for unmatched: 0.413

## Training Dynamics

- Step change at epoch 102: topk2 annealing 3→1 triggered, mAP50 jumped 0.29→0.38
- Gradual improvement from epoch 168 to 229
- o2o val fg/bg gap reached ~66x (fg=1.319 vs bg=0.020) — binary head discrimination continues improving
- Plateaued after epoch 229, stopped at 329

## Diagnosis

**β=2.0 is a clear win.** mAP50, bPQ, bDQ, and recall all improved. The spatial hallucination problem is dramatically reduced — prediction count perfectly calibrated at 0.99x GT (vs 1.01x train71 with 40% FP rate).

**Medium and large nuclei are largely solved** (74-81% recall). The remaining gap is small nuclei (39.8%). Their best anchor is fundamentally poorly aligned: 8px radius / 2-4px anchor offset = 25-50% misalignment ratio. Lowering β helps but doesn't fix the geometric reality that no anchor is "close enough" for a 4px nucleus on a 4px-stride grid.

**bDQ now 0.660 vs LSP-DETR's 0.803** — gap narrowed from 0.175 to 0.143. Most of the remaining gap is small nuclei (39.8% vs likely 60%+ for LSP-DETR's deformable attention).

## Key Insight

Assignment tuning (β) can only optimize anchor SELECTION from the existing grid. It cannot create anchors that don't exist. Small nuclei fundamentally need either:
1. More anchors near them (P1 detection scale, rejected due to fg/bg ratio)
2. Ways to aggregate signal from distant anchors (NWD, which provides smooth similarity even for misaligned boxes)
3. Forced positive assignments (STAL) to at least give them some gradient signal

## Next Steps

- STAL (`stal_min_positives=3`) — force minimum positive anchors for small nuclei
- NWD — replace pIoU with Wasserstein distance for smooth small-object similarity
