#!/usr/bin/env bash
# Recast and cohort-margin removal ablations (single fold, T12).
#
# Both mechanisms are ON in every nulite_detr recipe ever committed, so neither
# has ever had a controlled A/B. Each arm here is a removal test: the scattered
# question is "is this mechanism actually carrying anything?", not "can we tune
# it".
#
#   norecast   detr_recast: true -> false
#              removes the local embedding correction applied after decoder
#              layer 3 (SegmentationQueryRecast: 3x3 stencil over the projected
#              detection feature + sigmoid foreground map, foreground-weighted
#              pooling, gated residual into the query). This is the mechanism
#              the project is named after. The module is not constructed when
#              the flag is false, so the decoder is a plain DETR stack.
#              Note: docs/runs/nulite-detr/query-recast-experiment.md (2026-09-19)
#              specifies exactly this control but records no quality results.
#
#   nomargin   detr_seg_fg_margin_weight 0.5 -> 0.0
#              detr_viable_owner_fg_weight 0.5 -> 0.0
#              one mechanism (cohort foreground-margin supervision): the two
#              auxiliary losses that exist solely to stop a matched or
#              viable query from losing to the empty logit. These are the only
#              modules in the model written specifically for the
#              good_suppressed_bg cohort, and they are on at 0.5 in every run.
#              If the two move in different directions, split them afterwards.
#
# Controls: output/v2/train52_fastvit-t12/weights/last.pt (held-out fold3
# bPQ 0.6372 / macro bPQ 0.6566, 9.55M params).
#
# Every run is `--epochs 600 --end 332` so the LR schedule matches the whole
# train52/train55/train56/train57/train58/train59/train60/train61 series.
#
# Decision rule for a REMOVAL arm (pre-registered, both bPQ and macro bPQ):
#   |delta| <= 0.002 on both   -> mechanism is NOT load-bearing
#   removal drops bPQ > 0.002  -> mechanism is load-bearing, keep it
#   removal gains bPQ > 0.002  -> drop it from the baseline
# Always report cohort shifts: matched, good_suppressed_bg, no_good_polygon,
# wrong_type, unmatched_predictions, raw predictions.
#
# Device agnostic: relative paths; works after an rsync to the second box.
#
# Usage:
#   scripts/train_recast_margin.sh
#   VARIANTS="norecast" scripts/train_recast_margin.sh
#   DEVICE=1 scripts/train_recast_margin.sh
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

VARIANTS=${VARIANTS:-"norecast nomargin"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

recipe_for() {
  case "$1" in
    norecast) echo "main/pannuke-thin-t12-norecast.yaml" ;;
    nomargin) echo "main/pannuke-thin-t12-nomargin.yaml" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

run_for() {
  case "$1" in
    norecast) echo "train62_norecast" ;;
    nomargin) echo "train63_nomargin" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

[[ -f "$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt" ]] \
  || { log "ERROR: missing control oracle $ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"; exit 1; }

for v in $VARIANTS; do
  recipe=$(recipe_for "$v")
  run=$(run_for "$v")

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
    --name "$run" 2>&1 | tee "$LOG_DIR/recast_margin_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all recast/margin arms complete"
for v in $VARIANTS; do echo "  $(run_for "$v"): $OUTPUT/$(run_for "$v")/weights/last.pt"; done
cat <<'EOF'

Held-out fold3 evaluation (control first, then each arm):
  for v in control norecast nomargin; do
    if [[ $v == control ]]; then W=output/v2/train52_fastvit-t12/weights/last.pt
    elif [[ $v == norecast ]]; then W=output/v2/train62_norecast/weights/last.pt
    else W=output/v2/train63_nomargin/weights/last.pt; fi
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights $W \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0 \
      --save-predictions output/evals/recast_margin/$v.npz
  done
EOF
