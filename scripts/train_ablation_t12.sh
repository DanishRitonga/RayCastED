#!/usr/bin/env bash
# T12 decoder grid search (single fold), symmetric with the T8 batch.
#
# Every recipe differs from main/pannuke-thin-t12.yaml in EXACTLY ONE key, so
# each run isolates one mechanism. Seg stage is skipped: the fold1-only T12
# oracle already exists.
#
# Device agnostic: relative paths, so the tree can be rsync'd to another box.
#
# Usage:
#   scripts/train_ablation_t12.sh                  # all six configs
#   VARIANTS="g2 hd128" scripts/train_ablation_t12.sh
#   DEVICE=1 scripts/train_ablation_t12.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-16}

OUTPUT=${OUTPUT:-output/v2}
LOG_DIR=${LOG_DIR:-output/logs}
ORACLE_DIR=${ORACLE_DIR:-output/oracle}

DET_EPOCHS=${DET_EPOCHS:-600}
DET_END=${DET_END:-332}

VARIANTS=${VARIANTS:-"ndl6 ndl4 g2 g0 hd128 hd256"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

CKPT="$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"
[[ -f "$CKPT" ]] || { log "ERROR: missing oracle $CKPT"; exit 1; }

for v in $VARIANTS; do
  recipe="main/pannuke-thin-t12-$v.yaml"
  run="train55_ablation-t12-$v"

  log "=== $v -> $run (epochs $DET_EPOCHS, end $DET_END) ==="
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  if [[ -f "$OUTPUT/$run/weights/last.pt" ]]; then
    log "already complete, skipping: $OUTPUT/$run/weights/last.pt"
    continue
  fi

  "$PY" -m raycasted.pipeline \
    --config "$recipe" \
    --output "$OUTPUT" \
    --stage train \
    --epochs "$DET_EPOCHS" \
    --end "$DET_END" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --name "$run" 2>&1 | tee "$LOG_DIR/ablation_t12_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all T12 configs complete"
for v in $VARIANTS; do
  echo "  train55_ablation-t12-$v: $OUTPUT/train55_ablation-t12-$v/weights/last.pt"
done
cat <<'EOF'

Held-out fold3 evaluation:
  for v in ndl6 ndl4 g2 g0 hd128 hd256; do
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights output/v2/train55_ablation-t12-$v/weights/last.pt \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0
  done
EOF
