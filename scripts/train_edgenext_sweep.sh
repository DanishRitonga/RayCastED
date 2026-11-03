#!/usr/bin/env bash
# Single-fold backbone sweep: EdgeNeXt family vs the FastViT-T8 (Train50) baseline.
#
# For each variant this trains the fold1-only segmentation oracle first (reusing
# an existing checkpoint when present), then the detector on the same
# Train50 thinning recipe. Folds: fold1 train / fold2 val / fold3 held out.
#
# Usage:
#   scripts/train_edgenext_sweep.sh                    # all variants in VARIANTS
#   VARIANTS="xx_small" scripts/train_edgenext_sweep.sh
#   SKIP_SEG=1 scripts/train_edgenext_sweep.sh         # reuse existing oracles
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-16}

DATA_DIR=${DATA_DIR:-raycasted/data/dataset/PanNuke/data}
TRAIN_TILES=${TRAIN_TILES:-output/v2/transformed/train}
VAL_TILES=${VAL_TILES:-output/v2/transformed/val}

ORACLE_DIR=${ORACLE_DIR:-output/oracle}
SEG_EPOCHS=${SEG_EPOCHS:-120}
DET_EPOCHS=${DET_EPOCHS:-600}
DET_END=${DET_END:-332}
OUTPUT=${OUTPUT:-output/v2}
LOG_DIR=${LOG_DIR:-output/logs}

VARIANTS=${VARIANTS:-"xx_small x_small small_rw"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

timm_name() {
  case "$1" in
    xx_small) echo "edgenext_xx_small.in1k" ;;
    x_small)  echo "edgenext_x_small.in1k" ;;
    small_rw) echo "edgenext_small_rw.sw_in1k" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

for v in $VARIANTS; do
  name=$(timm_name "$v")
  ckpt="$ORACLE_DIR/nuLiteT-fold1-sep-$v.pt"
  recipe="main/pannuke-edgenext-$v.yaml"
  run="train51_edgenext-$v"

  log "=== $v ($name) ==="
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  if [[ "${SKIP_SEG:-0}" == "1" ]]; then
    log "SKIP_SEG=1: reusing $ckpt"
    [[ -f "$ckpt" ]] || { log "ERROR: $ckpt missing"; exit 1; }
  elif [[ -f "$ckpt" ]]; then
    log "oracle exists, skipping seg stage: $ckpt"
  else
    log "stage 1/2: seg oracle -> $ckpt"
    "$PY" -m raycasted.model.train_nulite_seg \
      --train-tiles "$TRAIN_TILES" \
      --train-parquet "$DATA_DIR/fold1-00000-of-00001.parquet" \
      --val-tiles "$VAL_TILES" \
      --val-parquet "$DATA_DIR/fold2-00000-of-00001.parquet" \
      --variant "$name" \
      --separable-decoder \
      --epochs "$SEG_EPOCHS" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --out "$ckpt" 2>&1 | tee "$LOG_DIR/edgenext_${v}_seg.log"
    log "stage 1/2 done: $ckpt"
  fi

  log "stage 2/2: detector -> $OUTPUT/$run"
  "$PY" -m raycasted.pipeline \
    --config "$recipe" \
    --output "$OUTPUT" \
    --stage train \
    --epochs "$DET_EPOCHS" \
    --end "$DET_END" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --name "$run" 2>&1 | tee "$LOG_DIR/edgenext_${v}_det.log"
  log "stage 2/2 done: $OUTPUT/$run/weights/last.pt"
done

log "all variants complete"
for v in $VARIANTS; do
  run="train51_edgenext-$v"
  echo "  $run: $OUTPUT/$run/weights/last.pt"
done
cat <<'EOF'

Evaluate each checkpoint on the held-out fold3:
  .venv/bin/python -m raycasted.evals.eval_lsp \
    --weights output/v2/train51_edgenext-<variant>/weights/last.pt \
    --data-dir output/v2/transformed/test \
    --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
    --batch 8 --device 0 --workers 0
EOF
