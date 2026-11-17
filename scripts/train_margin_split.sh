#!/usr/bin/env bash
# Margin-term split ablation (single fold, T12): which of the two cohort
# foreground-margin losses carries the effect?
#
# Context: main/pannuke-thin-t12-nomargin.yaml turned BOTH terms off at once
# (train63_nomargin) and lost 0.0083 bPQ / 0.0059 macro bPQ, with
# good_suppressed_bg 1,163 -> 1,620 and matched -452. That removal is
# confounded: the two terms have different shapes and the effect cannot be
# attributed to either.
#
#   segonly      detr_viable_owner_fg_weight 0.5 -> 0.0
#                keeps L_seg_fg_margin (evidence-gated: 3x3 grid_sample of
#                sigmoid(seed) at the predicted centroid, radius 4 px, applied
#                to matched queries only).
#
#   viableonly   detr_seg_fg_margin_weight 0.5 -> 0.0
#                keeps L_viable_owner_fg (ungated softplus(1 + z_empty - z_true)
#                on every matched query whose detached sampled pIoU >= 0.5).
#
# Both are final-decoder-layer only and both are guarded by `> 0` in
# nulite_detr_model.py, so 0.0 removes the term cleanly (no dead module, no
# extra compute). Each recipe differs from main/pannuke-thin-t12.yaml in
# exactly ONE key.
#
# Controls: output/v2/train52_fastvit-t12/weights/last.pt (held-out fold3
# bPQ 0.6372 / macro bPQ 0.6566, 9.55M params) and the joint removal
# train63_nomargin (0.6289 / 0.6507).
#
# Every run is `--epochs 600 --end 332` so the LR schedule matches the whole
# train52/train55/train56/train57/train58/train59/train60/train61/train62/
# train63 series.
#
# Decision rule (pre-registered, both bPQ and macro bPQ, vs control):
#   |delta| <= 0.002 on both   -> that term is NOT load-bearing
#   removal drops bPQ > 0.002  -> that term is load-bearing, keep it
#   removal gains bPQ > 0.002  -> drop it from the baseline
# Then check additivity against the joint arm: if each single removal costs
# roughly half of the joint -0.0083, the terms are redundant rather than
# complementary. Always report cohort shifts: matched, good_suppressed_bg,
# no_good_polygon, wrong_type, unmatched_predictions, raw predictions.
#
# CSV caveat: a results.csv column only exists while its weight is > 0. These
# arms therefore have 32 columns (not 33) and bPQ/bSQ/bDQ sit at fields
# 13/14/15 instead of 14/15/16.
#
# Device agnostic: relative paths; works after an rsync to the second box.
#
# Usage:
#   scripts/train_margin_split.sh
#   VARIANTS="segonly" scripts/train_margin_split.sh
#   DEVICE=1 scripts/train_margin_split.sh
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

VARIANTS=${VARIANTS:-"segonly viableonly"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

recipe_for() {
  case "$1" in
    segonly)    echo "main/pannuke-thin-t12-margin-segonly.yaml" ;;
    viableonly) echo "main/pannuke-thin-t12-margin-viableonly.yaml" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

run_for() {
  case "$1" in
    segonly)    echo "train64_margin-segonly" ;;
    viableonly) echo "train65_margin-viableonly" ;;
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
    --name "$run" 2>&1 | tee "$LOG_DIR/margin_split_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all margin-split arms complete"
for v in $VARIANTS; do echo "  $(run_for "$v"): $OUTPUT/$(run_for "$v")/weights/last.pt"; done
cat <<'EOF'

Held-out fold3 evaluation (control, joint removal, then each split arm):
  eval_one() {
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights "$1" \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0 \
      --save-predictions "output/evals/margin_split/$2.npz"
  }
  eval_one output/v2/train52_fastvit-t12/weights/last.pt        ctrl
  eval_one output/v2/train63_nomargin/weights/last.pt           nomargin
  eval_one output/v2/train64_margin-segonly/weights/last.pt     segonly
  eval_one output/v2/train65_margin-viableonly/weights/last.pt  viableonly
EOF
