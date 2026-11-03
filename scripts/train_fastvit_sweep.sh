#!/usr/bin/env bash
# FastViT backbone scaling ablation (single fold): T8 / T12 / S12 / SA12.
#
# Every variant uses the identical Train50 thinning recipe, so the backbone is
# the only changed mechanism. Folds: fold1 train / fold2 val / fold3 held out.
#
# For each variant this trains the fold1-only segmentation oracle first (reusing
# an existing checkpoint when present), then the detector.
#
# Usage:
#   scripts/train_fastvit_sweep.sh                    # t12 s12 sa12 (t8 = Train50, done)
#   VARIANTS="s12" scripts/train_fastvit_sweep.sh
#   SKIP_SEG=1 scripts/train_fastvit_sweep.sh         # reuse existing oracles
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

# t8 is omitted: it is already the Train50 baseline (train50_thin-t8-aggressive).
VARIANTS=${VARIANTS:-"t12 s12 sa12"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

timm_name() {
  case "$1" in
    t8)   echo "fastvit_t8" ;;
    t12)  echo "fastvit_t12" ;;
    s12)  echo "fastvit_s12" ;;
    sa12) echo "fastvit_sa12" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

for v in $VARIANTS; do
  name=$(timm_name "$v")
  ckpt="$ORACLE_DIR/nuLiteT-fold1-sep-$v.pt"
  recipe="main/pannuke-thin-$v.yaml"
  run="train52_fastvit-$v"

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
      --out "$ckpt" 2>&1 | tee "$LOG_DIR/fastvit_${v}_seg.log"
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
    --name "$run" 2>&1 | tee "$LOG_DIR/fastvit_${v}_det.log"
  log "stage 2/2 done: $OUTPUT/$run/weights/last.pt"
done

log "all variants complete"
for v in $VARIANTS; do
  echo "  train52_fastvit-$v: $OUTPUT/train52_fastvit-$v/weights/last.pt"
done
cat <<'EOF'

Held-out fold3 evaluation for every point on the curve (t8 = the baseline run):
  for v in t8 t12 s12 sa12; do
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights output/v2/train52_fastvit-$v/weights/last.pt \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0
  done
EOF
