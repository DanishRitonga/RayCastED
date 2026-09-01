# Train30 — yolo11s backbone + FPN top-down

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `3d9d5a7`)
- Output: `output/nulite/train30` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep599/600 still climbing)
- Config: train19 S+F+Q recipe (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, `detr_per_layer_match: true`) **+ `detr_backbone: yolo11`, `yolo11_scale: s`, `detr_use_seed: true`** — the ONLY change from train29's recipe (FastViT s12 backbone swapped for YOLO11s + FPN top-down; stride-4 seed head on P2).
- Model: **9.97M params** (backbone+FPN 6.05M, P2/P3/P4 = 64/128/256ch → DETR head geometrically identical to s12). `3d9d5a7` added scale-aware pretrained loading (`yolo11{yolo11_scale}.pt`) so the s-scale weights actually load.

## Hypothesis
Same 64/128/256 head input channels as s12 → expect near-s12 quality at ~37% fewer params, plus a second backbone axis (C3k2 CNN vs FastViT) for the paper's modularity claim. T8 had lost −6.2pt bPQ mostly via shrunken head input channels (48/96/192); yolo11s keeps them full.

## Results

### Val (train set; still climbing at ep600)
Best fitness ep599/600: mAP50 0.4081, bPQ 0.5301, bSQ 0.7945, bDQ 0.6341.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.6993 · AP@0.5 0.8109 · AP@0.7 0.6121 · AP@0.9 0.0217 · AP@0.5:0.05:0.95 0.4611 · **bPQ 0.5743** · **mPQ 0.5419** · F1 0.7708 · Prec 0.6321 · Rec 0.9875 · **9.97M** · **4.48 ms/img** (fastest yet) · 102,861 preds (1.56x). Class F1 all ~0.78-0.86; small-nucleus recall 0.9662. Tissue macro bPQ 0.5975 (worst Colorectal 0.4540).

**vs s12 runs:** bPQ −0.096 vs train29 parity, −0.106 vs train28 old-dec. Pattern: recall holds (0.9875) but precision collapses (0.6321 vs 0.78-0.81) → more FPs (1.56x preds).

## Diagnosis
**yolo11s does NOT match s12 quality** — well below the 0.660 decision threshold. Identical head channels (64/128/256) yet far worse winner-selection: the FastViT/NuLite feature pipeline carries the quality, not the head input width. CNN backbone features are less discriminative for finding nuclei. s12 stays the submission candidate; yolo11s/T8 remain secondary lightweight paper data points (9.97M / 4.48ms is the fastest latency measured).

## Next
- Head-slimming ladder on the s12 parity base instead (→ train31 shared-layers, the eventual best).

---
