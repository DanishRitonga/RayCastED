# Train21 — seed_q-only ablation

## Run
- Date: 2026-08-29
- Branch: `feature/nulite-raycast-seg` (HEAD `869b662`)
- Output: `output/nulite/train21` (600 epochs, amp=false, patience=500, plain cosine)
- Config change from train20: `detr_seed_in_content: false → true` only. `lambda_seg: 0.0`, `detr_seed_feature: false` unchanged → the only seed signal is the per-query objectness read (raw seed logit concat to score-head inputs). Verified np_head 5/5 grads (objectness gradient flows via seed_q with lambda_seg=0).

## Hypothesis
Isolate the seed_q objectness-read contribution. If train21 ≈ train19 (0.673), seed_q is the whole seed-branch contribution and the seg loss + feature channel are strippable.

## Results

### Val (train set; 0 zero-metric epochs)
Best fitness ep600: bPQ 0.5744, bSQ 0.7824, bDQ 0.6966, mAP50 0.4351, P 0.4283, R 0.6690.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
| metric | train19 S+F+Q | train20 none | train21 Q only |
|---|---|---|---|
| bPQ | **0.6728** | 0.6565 | 0.6474 |
| mPQ | **0.6294** | 0.6160 | 0.6046 |
| AP@0.5 | **0.8758** | 0.8682 | 0.8495 |
| AP@0.5:0.95 | **0.5023** | 0.4803 | 0.4883 |
| AJI | 0.7521 | 0.7388 | 0.7426 |
| F1 | 0.8823 | 0.8743 | 0.8540 |
| Prec / Rec | 0.7937 / 0.9933 | 0.7812 / 0.9925 | 0.7502 / 0.9911 |
| preds | 82,409 (1.25x) | 83,658 (1.27x) | 86,992 (1.32x) |

## Diagnosis
**seed_q in isolation is actively harmful** — bPQ 0.6474 < pure-DETR 0.6565, more preds (1.32x), lower precision. Concatenating the raw seed logit into the score-head input perturbs the class/objectness features without a corresponding training benefit at lambda_seg=0 (the NP head's objectness gradient is weak/noisy vs. the dense seg-supervised signal). Therefore train19's +1.6 bPQ margin comes from the **seg loss + feature channel** (which must also overcome the seed_q harm), not from seed_q.

## Next
- train22 (config committed `eafcb00`): seg loss + feature channel, seed_q OFF (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: false`). If ≥ train19 → seed_q is pure harm and S+F is the minimal recipe. If ≈ train19+ → S+F is the contributor and the recipe is complete. If ≈ train20 → seg loss/feature channel also contribute little and the whole seed branch is strippable.
