#!/usr/bin/env bash
# Extended training for the larger FastViT backbones (S12 / SA12).
#
# Motivation: the 332-epoch series stopped at 42% of lr0 (cosine configured for
# 600 epochs), so those numbers are not converged. This script keeps the SAME
# 600-epoch schedule as every prior run (so the LR curve is identical) and just
# lets the run reach a later point on it, --end 500, by which the LR has decayed
# to 0.000763. It is therefore a converged-er point on the same schedule, NOT a
# differently scheduled run.
#
# No segmentation stage by default: the fold1-only oracles already exist for
# every variant handled here (output/oracle/nuLiteT-fold1-sep-<v>.pt).
#
# First run on a fresh machine: the dataset parquets, the transformed tiles and
# the oracle checkpoints are all gitignored, so this script bootstraps them
# itself. It ingests + transforms the dataset once when $OUTPUT/transformed/train
# is absent, and trains a missing oracle before the detector stage.
#
# Device agnostic: all paths are relative to the repo root, so the tree can be
# rsync'd to another machine (second GPU) and run unchanged.
#
# Usage:
#   scripts/train_extended.sh                     # s12 then sa12
#   VARIANTS="s12" scripts/train_extended.sh
#   DEVICE=1 scripts/train_extended.sh            # second GPU box
#   FORCE_INGEST=1 scripts/train_extended.sh      # rebuild tiles from raw data
#   SKIP_SEG=1 scripts/train_extended.sh          # fail instead of training oracles
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
DET_END=${DET_END:-500}

SEG_EPOCHS=${SEG_EPOCHS:-120}
DATA_DIR=${DATA_DIR:-raycasted/data/dataset/PanNuke/data}
TRANSFORMED=${TRANSFORMED:-$OUTPUT/transformed}
FORCE_INGEST=${FORCE_INGEST:-0}
SKIP_SEG=${SKIP_SEG:-0}

# t8 is excluded (user decision: no extended training for T8).
VARIANTS=${VARIANTS:-"s12 sa12"}

# Any recipe carrying the PanNuke ETL block works for ingest/transform; the
# backbone choice is irrelevant to those stages.
RECIPE_BOOTSTRAP=${RECIPE_BOOTSTRAP:-main/pannuke-thin-t12.yaml}

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

# ---------------------------------------------------------------------------
# Stage 0: make this machine able to train at all.
# ---------------------------------------------------------------------------
if [[ ! -f "$DATA_DIR/fold1-00000-of-00001.parquet" ]]; then
  log "ERROR: missing $DATA_DIR/fold1-00000-of-00001.parquet"
  log "       The dataset is gitignored; copy the PanNuke parquets onto this box first."
  exit 1
fi

if [[ "$FORCE_INGEST" == "1" || ! -d "$TRANSFORMED/train" || -z "$(ls -A "$TRANSFORMED/train" 2>/dev/null)" ]]; then
  log "stage 0/2: ingest + transform dataset -> $TRANSFORMED"
  "$PY" -m raycasted.pipeline \
    --config "$RECIPE_BOOTSTRAP" \
    --output "$OUTPUT" \
    --stage ingest \
    --device "$DEVICE" 2>&1 | tee "$LOG_DIR/extended_ingest.log"
  "$PY" -m raycasted.pipeline \
    --config "$RECIPE_BOOTSTRAP" \
    --output "$OUTPUT" \
    --stage transform \
    --device "$DEVICE" 2>&1 | tee "$LOG_DIR/extended_transform.log"
  log "stage 0/2 done: $TRANSFORMED"
else
  log "stage 0/2: reusing existing tiles at $TRANSFORMED (FORCE_INGEST=1 to rebuild)"
fi

if [[ ! -d "$TRANSFORMED/val" || -z "$(ls -A "$TRANSFORMED/val" 2>/dev/null)" ]]; then
  log "ERROR: $TRANSFORMED/val is empty; cannot train the oracle"
  exit 1
fi

for v in $VARIANTS; do
  name=$(timm_name "$v")
  ckpt="$ORACLE_DIR/nuLiteT-fold1-sep-$v.pt"
  recipe="main/pannuke-thin-$v.yaml"
  run="train54_${v}-ext${DET_END}"

  log "=== $v ($name) -> $run (epochs $DET_EPOCHS, end $DET_END) ==="
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  if [[ -f "$ckpt" ]]; then
    log "stage 1/2: oracle exists, reusing $ckpt"
  elif [[ "$SKIP_SEG" == "1" ]]; then
    log "ERROR: missing oracle $ckpt and SKIP_SEG=1"
    exit 1
  else
    log "stage 1/2: training seg oracle -> $ckpt"
    "$PY" -m raycasted.model.train_nulite_seg \
      --train-tiles "$TRANSFORMED/train" \
      --train-parquet "$DATA_DIR/fold1-00000-of-00001.parquet" \
      --val-tiles "$TRANSFORMED/val" \
      --val-parquet "$DATA_DIR/fold2-00000-of-00001.parquet" \
      --variant "$name" \
      --separable-decoder \
      --epochs "$SEG_EPOCHS" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --cache-root "$ORACLE_DIR" \
      --out "$ckpt" 2>&1 | tee "$LOG_DIR/extended_${v}_seg.log"
    log "stage 1/2 done: $ckpt"
  fi

  if [[ -f "$OUTPUT/$run/weights/last.pt" ]]; then
    log "already complete, skipping: $OUTPUT/$run/weights/last.pt"
    continue
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
    --name "$run" 2>&1 | tee "$LOG_DIR/extended_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all extended runs complete"
for v in $VARIANTS; do
  echo "  train54_${v}-ext${DET_END}: $OUTPUT/train54_${v}-ext${DET_END}/weights/last.pt"
done
cat <<EOF

Held-out fold3 evaluation:
  for v in $VARIANTS; do
    .venv/bin/python -m raycasted.evals.eval_lsp \\
      --weights output/v2/train54_\${v}-ext${DET_END}/weights/last.pt \\
      --data-dir output/v2/transformed/test \\
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \\
      --batch 8 --device 0 --workers 0
  done
EOF
