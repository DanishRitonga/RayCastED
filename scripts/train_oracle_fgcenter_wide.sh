#!/usr/bin/env bash
# Train a WIDE-sigma fg+centroid oracle for the seed-map ablation.
#
# Why a new oracle at all: the existing centroid oracles use
# center_sigma = clamp(0.25 * mean_radius, 2.5, 4.0), i.e. 2.5-4 px. The query
# grid is 20x20 over a 256 px tile, so a grid cell centre sits up to 12.8 px
# from a centroid. Measured on held-out fold3, the existing sharp centroid
# channels are alive at only 2.9-3.5% of cell centres (vs 74% for the plateau
# map) - they are clean maps that the grid queries cannot see. This oracle keeps
# the same dual-target objective (foreground + centroid potential) and only
# widens the centroid kernel, so that arm separates "centroid semantics" from
# "map too sharp to be sampled".
#
# The detector arm that consumes it:
#   main/pannuke-thin-t12-fgmap-fgcentroid-wide.yaml
#
# Full training is required: the detector validates the checkpoint's
# oracle_kind against the model, so the existing checkpoint cannot be reused
# with different sigma settings.
#
# Usage:
#   scripts/train_oracle_fgcenter_wide.sh
#   EPOCHS=150 scripts/train_oracle_fgcenter_wide.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-16}

OUTPUT=${OUTPUT:-output/v2}
LOG_DIR=${LOG_DIR:-output/logs}
ORACLE_DIR=${ORACLE_DIR:-output/oracle}
DATA_DIR=${DATA_DIR:-raycasted/data/dataset/PanNuke/data}
TRAIN_TILES=${TRAIN_TILES:-$OUTPUT/transformed/train}
VAL_TILES=${VAL_TILES:-$OUTPUT/transformed/val}

EPOCHS=${EPOCHS:-120}
SIGMA_SCALE=${SIGMA_SCALE:-0.6}
SIGMA_MIN=${SIGMA_MIN:-4.0}
SIGMA_MAX=${SIGMA_MAX:-8.0}

OUT_CKPT=${OUT_CKPT:-$ORACLE_DIR/nuLiteT-fold1-fg-center-wide-t12.pt}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

for f in "$DATA_DIR/fold1-00000-of-00001.parquet" "$DATA_DIR/fold2-00000-of-00001.parquet"; do
  [[ -f "$f" ]] || { log "ERROR: missing $f"; exit 1; }
done
for d in "$TRAIN_TILES" "$VAL_TILES"; do
  [[ -d "$d" ]] || { log "ERROR: missing tile dir $d"; exit 1; }
done

if [[ -f "$OUT_CKPT" ]]; then
  log "oracle already exists, skipping: $OUT_CKPT"
  exit 0
fi

log "=== fg_centroid wide oracle -> $OUT_CKPT"
log "    sigma = clamp(${SIGMA_SCALE} * mean_radius, ${SIGMA_MIN}, ${SIGMA_MAX}) px, epochs $EPOCHS"

"$PY" -m raycasted.model.train_nulite_seg \
  --train-tiles "$TRAIN_TILES" \
  --train-parquet "$DATA_DIR/fold1-00000-of-00001.parquet" \
  --val-tiles "$VAL_TILES" \
  --val-parquet "$DATA_DIR/fold2-00000-of-00001.parquet" \
  --variant fastvit_t12 \
  --separable-decoder \
  --fg-center-target \
  --center-sigma-scale "$SIGMA_SCALE" \
  --center-sigma-min "$SIGMA_MIN" \
  --center-sigma-max "$SIGMA_MAX" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --cache-root "$ORACLE_DIR" \
  --out "$OUT_CKPT" 2>&1 | tee "$LOG_DIR/oracle_fgcenter_wide.log"

log "done: $OUT_CKPT"
cat <<EOF

Then run the optional fourth arm:
  VARIANTS="fgcentroid-wide" scripts/train_fgmap_sweep.sh
EOF
