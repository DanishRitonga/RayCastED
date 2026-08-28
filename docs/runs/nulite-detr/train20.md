# Train20 — Pure-DETR no-seed ablation

## Run
- Date: 2026-08-29
- Branch: `feature/nulite-raycast-seg` (HEAD `e5f48ee`)
- Output: `output/nulite/train20` (600 epochs, amp=false, patience=500, plain cosine)
- Config change from train19: `lambda_seg: 1.0 → 0.0`, `detr_seed_feature: true → false`, `detr_seed_in_content: true → false` — the NP/seed head is computed in the forward pass but receives ZERO gradient (verified: np_head 0/5 grads). `detr_mds: false` (stripped after train19 wash).

## Hypothesis
The seed branch (seg loss + feature channel + seed_q objectness read) may be entirely redundant — the DETR head learns detection from the FCN-parity loss + cost_inside + focal-∅ alone. If train20 matches train19, the whole NP/seed branch is strippable.

## Results

### Val (train set, 2656 imgs; 0 zero-metric epochs — first stable DETR baseline)
| metric | best | epoch |
|---|---|---|
| bPQ | 0.5874 | 599 (still climbing) |
| bSQ | 0.7816 | 599 |
| bDQ | 0.7139 | 599 |
| mAP50 | 0.4455 | 579 |
| P / R | 0.446 / 0.657 | 599 |

### Test (fold3, 2722 tiles, conf=0.20, corrected instance-level metrics)
| metric | train19 (seed) | train20 (no seed) | Δ |
|---|---|---|---|
| bPQ | 0.6728 | **0.6565** | −0.016 |
| mPQ | 0.6294 | **0.6160** | −0.013 |
| AP@0.5 | 0.8758 | **0.8682** | −0.008 |
| AP@0.5:0.95 | 0.5023 | **0.4803** | −0.022 |
| AJI | 0.7521 | **0.7388** | −0.013 |
| F1 (centroid) | 0.8823 | **0.8743** | −0.008 |
| Prec / Rec | 0.7937 / 0.9933 | 0.7812 / 0.9925 | −0.012 / −0.001 |
| preds | 82,409 (1.25x) | 83,658 (1.27x) | +1,249 |
| ms/img | 5.10 | 5.28 | +0.18 |

Class F1: Neoplastic 0.8553 / Inflammatory 0.9137 / Connective 0.8628 / Necrosis 0.7562 / Epithelial 0.8855. Size recall: small 0.9802 / medium 0.9980 / large 0.9990. Worst tissue: Colorectal (bPQ 0.5447). Polygon-level (area-ratio IoU, relative ref only): bPQ 0.8327, centroid L2 87.6px, ray L1 1.74px.

## Diagnosis
Detection **fully holds** without any seed signal (bPQ 0.657 still above FCN train6 0.631 and roughly at literature LSP-DETR 0.675 / LKCell 0.684), but the seed branch is a **small, consistent, genuine contributor** (~+1.6 bPQ, +1.3 mPQ, +2.2 AP@50-95) — not a wash. Overprediction stays solved (1.27x).

The contributing seed component is unknown: it could be (a) the seed_q objectness read in the score heads, (b) the seed_feature channel in input_proj, or (c) the explicit seg loss regularizing the NP head.

## Next
- train21 (config committed `869b662`): keep ONLY seed_q (`detr_seed_in_content: true`, `lambda_seg: 0.0`, `detr_seed_feature: false`). If it recovers to train19 levels → seed_q is the sole contributor and the seg loss + feature channel are strippable. If it stays at train20 levels → seed_q contributes nothing; the gain came from seg loss / feature channel (further ablation then needed).
