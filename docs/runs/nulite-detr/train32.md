# Train32 — ladder B: FFN dim d_ffn 1024 → 512 (shared-layers base)

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `2b550bc`)
- Output: `output/nulite/train32` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep589/600 still climbing)
- Config: train31 shared-layers base S+F+Q (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, `detr_per_layer_match: true`, `detr_shared_layers: true`, `nulite_variant: fastvit_s12`) **+ `detr_d_ffn: 1024→512`** — the ONLY change from train31. Model: 13.51M params (−0.26M).

## Hypothesis
Ladder B cuts FFN capacity (53% of each decoder layer's params). Expectation: mild regression risk but a genuine latency/param win for a Pareto point.

## Results

### Val (train set; still climbing at ep600)
Best fitness ep589 (0.5543): mAP50 0.4589, bPQ 0.5952, bSQ 0.7797, bDQ 0.7246; ep600 row: mAP50 0.4584, bPQ 0.5953. ≈ train31 val.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.7606 · AP@0.5 0.8833 · AP@0.7 0.6830 · AP@0.9 0.0230 · AP@0.5:0.05:0.95 0.5097 · **bPQ 0.6843** · **mPQ 0.6463** · F1 0.8850 · Prec 0.8024 · Rec 0.9866 · **13.51M** · 5.09 ms/img · 82,535 preds (1.25x).

**vs train31 (d_ffn 1024):** bPQ −0.011, mPQ −0.010, AP@50-95 −0.010, AJI −0.003, F1 −0.011, Prec −1.9pt, Rec ~0, preds +0.05x (1.25x vs 1.20x), params −0.26M, ms −0.15.

## Diagnosis
**Small regression, consistent with the head-capacity pattern.** Cutting the FFN (the bulk of decoder-layer capacity) slightly hurts winner-selection: more duplicate preds, lower precision. 0.26M params / 0.15ms is not worth −1.1pt bPQ. Like ndl 3→2 (train26), plain capacity cuts don't pay; only weight-shared layers (train31, ladder D) gave a free win via recurrence. Config reverted to `detr_d_ffn: 1024`; train31 remains the submission candidate.

## Next
- Ladder C (hd 256→128) — last head-slimming knob; riskiest (−75% FLOPs).

---
