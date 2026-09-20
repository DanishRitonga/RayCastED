# Inter-layer query recast experiment — 2026-09-19

Status: implemented; training not launched. No quality results yet.

## Mechanism

The active recipe keeps the frozen foreground segmentation oracle, trainable
FastViT-T12 detection stream, 400 grid queries, and six width-256 decoder layers.
Respawn is disabled. A single content correction follows decoder layer 3, before
layers 4–6. It leaves reference points and polygon refinement unchanged.

At each current predicted center, sample a 3×3 stencil spanning ±4 input pixels
from the projected finest detection feature map and sigmoid foreground map.
Foreground-weighted pooling produces a 256-dimensional local descriptor z.
The correction is q' = q + g Wz. The scalar gate is a learned sigmoid of
LayerNorm([q, P(BG), mean local FG]), multiplied by P(BG) and mean local FG.
P(BG) uses softmax over the five classes plus no-object as a routing heuristic;
it does not change the existing focal classification loss or inference decision.
Positions and probability cues are detached for routing. Gradients flow through
sampled detection features and the correction module. W starts at zero: exact
identity initially, with a trainable projection. The gate learns once W moves.

All queries receive a soft gate. There is no top-K, peak extraction, relocation,
second decoder pass, additional prediction layer, or residual matching loss.
Ordinary detection and auxiliary losses supervise the existing decoder outputs.
The new module has 66,311 parameters at width 256. Local feature sampling stores
B×256×400×9 values (about 56 MiB at batch 16 FP32, before backward overhead).
This is not a measurement of total GPU memory or a guarantee that batch 16 fits.

## Configuration and comparison

- `detr_recast: true`
- `detr_recast_after_layer: 3` (one-based; must precede the final/eval layer)
- `detr_recast_radius_px: 4.0`
- `detr_respawn: false`

Baseline: use the identical recipe with `detr_recast: false` and respawn still
false. Train each arm from a fresh initialization/output directory under the
same fold split and seed. Do not resume the broken respawn checkpoint as the
controlled baseline. Training remains user-launched.

Before full training, verify GPU peak memory with a representative forward/backward
batch. CPU checks cover train/eval shapes, decoder invocation count, identity
initialization, gradients, and invalid settings. GPU profiling was unavailable
in the implementation session because the NVIDIA driver could not be reached.

Use validation data for tuning and for examining whether high-disagreement queries
represent missed nuclei or duplicates. Compare recall, precision, duplicate errors,
and PQ. For final literature comparisons use eval_lsp.py on genuinely held-out
folds, dense parquet GT, final checkpoints, and three-fold reporting.

No claim that segmentation disagreement uniquely identifies a missed instance:
queries suppressed by one-to-one competition may also sit on foreground.
