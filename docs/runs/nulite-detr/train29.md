# Train29 — s12 NuLite-parity decoder retrain

## Run
- Date: 2026-09-01
- Branch: `feature/nulite-raycast-seg` (HEAD `207e338`)
- Output: `output/nulite/train29` (600 epochs, amp=false, patience=300, plain cosine, 0 zero-metric epochs, best = ep599/600 still climbing)
- Config: train19 S+F+Q recipe (`lambda_seg: 1.0`, `detr_seed_feature: true`, `detr_seed_in_content: true`, `detr_no_object_weight: 2.0`, `detr_cost_inside: 10.0`, `detr_mds: false`, `detr_per_layer_match: true`, `nulite_variant: fastvit_s12`) on the **NuLite-parity decoder** (`207e338`): exact `Conv2DBlock` (Conv2d bias=True→BN→ReLU→Dropout), NuLite channel ladders (bottleneck 1 conv, decoder4/3 two-conv ladders, decoder2/1 one conv), concat order `[z<enc>, b<dec>]`, decoder0 = ReLU `_ConvBlock`. Model: 15.76M params (+0.296M vs old decoder).

## Hypothesis
The old decoder deviated from the NuLite reference (extra conv blocks, reversed concat order, SiLU decoder0). The parity fix makes our encoder-decoder load NuLite's official pretrained weights and match the paper's architecture exactly. Question: does parity change quality (should be ~neutral)?

## Results

### Val (train set; still climbing at ep600)
Best fitness ep599/600: mAP50 0.4460, bPQ 0.5760, bSQ 0.7795, bDQ 0.7010.

### Test (fold3, 2722 tiles, conf=0.20, corrected metrics)
AJI 0.7548 · AP@0.5 0.8743 · AP@0.7 0.6765 · AP@0.9 0.0245 · AP@0.5:0.05:0.95 0.5040 · **bPQ 0.6698** · **mPQ 0.6284** · F1 0.8694 · Prec 0.7768 · Rec 0.9870 · **15.76M** · 5.16 ms/img · 83,669 preds (1.27x).

**vs train28 (OLD decoder):** bPQ −0.010, mPQ −0.016, AP@0.5 −0.008, AP50-95 −0.003, AJI +0.003, F1 −0.018, Prec −3.0pt, Rec ~0, preds +3073 (1.27x vs 1.22x), ms −0.39.

## Diagnosis
Parity-corrected decoder is a **consistent small regression** on test (bPQ −1pt, mPQ −1.6pt, precision −3pt), matching the val direction. Recall holds at 0.987. The parity decoder is architecturally correct (NuLite-exact, needed for official-weight compatibility) but the old decoder's extra refinement capacity happened to help slightly. The gap is small enough that parity stays the right call for the paper; a head change (train31 shared-layers) recovered and exceeded the loss.

## Next
- train31 (weight-shared decoder on this base) → new best.

---
