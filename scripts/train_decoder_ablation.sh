#!/usr/bin/env bash
# Decoder ablation on the Train50 (FastViT-T8) thinning recipe.
#
# Six runs, each differing from main/pannuke-thin-t8.yaml in EXACTLY ONE key:
#   ndl6   detr_ndl 5 -> 6            depth up
#   ndl4   detr_ndl 5 -> 4            depth down   (measured -6% latency)
#   g2     shared_layer_groups 3 -> 2 more sharing (5.66M, param+latency win)
#   g0     shared_layer_groups 3 -> 0 no sharing   (7.15M, control)
#   hd128  detr_hd 192 -> 128         width down   (5.16M, -16% params)
#   hd256  detr_hd 192 -> 256         width up     (7.43M)
#
# The baseline itself is output/v2/train50_thin-t8-aggressive (no re-run needed).
# Folds: fold1 train / fold2 val / fold3 held out. Schedule matches the
# Train50 series: --epochs 600 --end 332.
#
# Usage:
#   scripts/train_decoder_ablation.sh                       # all six
#   VARIANTS="ndl4 hd128" scripts/train_decoder_ablation.sh # subset
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-16}
DET_EPOCHS=${DET_EPOCHS:-600}
DET_END=${DET_END:-332}
OUTPUT=${OUTPUT:-/mnt/data/projects/RayCastED/output/v2}
LOG_DIR=${LOG_DIR:-output/logs}

VARIANTS=${VARIANTS:-"ndl6 ndl4 g2 g0 hd128 hd256"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR"

for v in $VARIANTS; do
  recipe="main/pannuke-thin-t8-$v.yaml"
  run="train53_ablation-$v"

  log "=== $v -> $run ==="
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  "$PY" -m raycasted.pipeline \
    --config "$recipe" \
    --output "$OUTPUT" \
    --stage train \
    --epochs "$DET_EPOCHS" \
    --end "$DET_END" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --name "$run" 2>&1 | tee "$LOG_DIR/ablation_${v}.log"

  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all decoder ablations complete"
for v in $VARIANTS; do
  echo "  train53_ablation-$v: $OUTPUT/train53_ablation-$v/weights/last.pt"
done

cat <<'EOF'

Held-out fold3 evaluation (baseline = train50_thin-t8-aggressive):
  for v in ndl6 ndl4 g2 g0 hd128 hd256; do
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights output/v2/train53_ablation-$v/weights/last.pt \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0 \
      --save-predictions /tmp/opencode/rc_full/pred_t53_$v.npz
  done
EOF
