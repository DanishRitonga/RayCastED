#!/usr/bin/env bash
# Thin FastViT-T8 pipeline: segmentation oracle first, then detection.
#
# Stage 1 trains the T8 NuLite U-Net oracle on fold1 (val fold2) with the
# standard oracle trainer. Stage 2 trains the detector against that oracle as a
# frozen backbone (pannuke-thin-t8.yaml: frozen backbone + FusionAdapter + full
# U-Net thaw at epoch 50 + 6 distinct decoder layers).
#
# Queries are NOT post-processed anywhere: stage 2 keeps the raw nc+1 argmax
# contract (no NMS, no clustering, no confidence threshold).
#
# Usage:
#   scripts/train_thin_t8.sh                # full pipeline
#   SKIP_SEG=1 scripts/train_thin_t8.sh     # reuse the existing T8 oracle
#   SEG_EPOCHS=60 DET_EPOCHS=600 scripts/train_thin_t8.sh
#
# Everything is logged under $LOG_DIR; the script stops on the first failure.

set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-16}

DATA_DIR=${DATA_DIR:-raycasted/data/dataset/PanNuke/data}
TRAIN_TILES=${TRAIN_TILES:-output/v2/transformed/train}
VAL_TILES=${VAL_TILES:-output/v2/transformed/val}
TEST_TILES=${TEST_TILES:-output/v2/transformed/test}

ORACLE_DIR=${ORACLE_DIR:-output/oracle}
ORACLE_CKPT=${ORACLE_CKPT:-$ORACLE_DIR/nuLiteT-fold1-sep-t8.pt}

SEG_EPOCHS=${SEG_EPOCHS:-120}
DET_EPOCHS=${DET_EPOCHS:-600}
DET_END=${DET_END:-332}

OUTPUT=${OUTPUT:-output/v2}
DET_NAME=${DET_NAME:-train49_thin-t8}
RECIPE=${RECIPE:-main/pannuke-thin-t8.yaml}

LOG_DIR=${LOG_DIR:-output/logs}
mkdir -p "$LOG_DIR" "$ORACLE_DIR"

log() { printf '[thin-t8] %s\n' "$*"; }

# ---------------------------------------------------------------- stage 1: seg
if [[ "${SKIP_SEG:-0}" == "1" ]]; then
  log "SKIP_SEG=1: reusing $ORACLE_CKPT"
  [[ -f "$ORACLE_CKPT" ]] || { log "ERROR: $ORACLE_CKPT missing"; exit 1; }
else
  log "stage 1/2: training T8 segmentation oracle (${SEG_EPOCHS} epochs)"
  "$PY" -m raycasted.model.train_nulite_seg \
    --train-tiles "$TRAIN_TILES" \
    --train-parquet "$DATA_DIR/fold1-00000-of-00001.parquet" \
    --val-tiles "$VAL_TILES" \
    --val-parquet "$DATA_DIR/fold2-00000-of-00001.parquet" \
    --variant fastvit_t8 \
    --separable-decoder \
    --epochs "$SEG_EPOCHS" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --out "$ORACLE_CKPT" \
    2>&1 | tee "$LOG_DIR/thin_t8_seg.log"
  log "stage 1/2 done: $ORACLE_CKPT"
fi

# ---------------------------------------------------------------- stage 2: det
log "stage 2/2: training detector on the frozen T8 oracle"
"$PY" -m raycasted.pipeline \
  --config "$RECIPE" \
  --output "$OUTPUT" \
  --stage train \
  --epochs "$DET_EPOCHS" \
  --end "$DET_END" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$DET_NAME" \
  2>&1 | tee "$LOG_DIR/thin_t8_det.log"

CKPT="$OUTPUT/$DET_NAME/weights/last.pt"
log "stage 2/2 done: $CKPT"
log "evaluate with:"
log "  $PY -m raycasted.evals.eval_lsp \\"
log "    --weights $CKPT \\"
log "    --data-dir $TEST_TILES \\"
log "    --parquet $DATA_DIR/fold3-00000-of-00001.parquet \\"
log "    --batch 8 --workers 0 --device $DEVICE"
