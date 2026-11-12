#!/usr/bin/env bash
# Empty-column cost, lambda sweep follow-up: lambda = 2.0.
#
# Control is main/pannuke-thin-t12.yaml (train52_fastvit-t12: held-out Fold3
# bPQ 0.6372, macro bPQ 0.6566). The treated arm differs from the control in
# two config keys that together are ONE mechanism at a chosen strength:
#   detr_matcher_empty_cost: true
#   detr_matcher_empty_cost_lambda: 2.0
# The lambda=1.0 arm (train57_empty-cost) is the dose-response reference:
# bPQ 0.6391 (+0.0019), macro bPQ 0.6578 (+0.0012), good_suppressed_bg -118.
#
# Mechanism: every query's match cost gains lambda * -log(1 - p_empty). The term
# is constant across GT columns, so it cannot reorder a fixed-size assignment;
# it changes which queries the solver leaves unused, so a query that claims the
# explicit no-object class stops absorbing a real nucleus.
#
# Seg stage is skipped: the fold1-only T12 oracle already exists.
# Device agnostic: relative paths, so the tree can be rsync'd to another box.
#
# Pre-registered decision rule (eval_lsp.py, raw nc+1 argmax, no NMS):
#   adopt  only if bPQ AND macro bPQ both gain > +0.002 over 0.6372 / 0.6566
#   reject if bPQ drops > 0.002
#   always report good_suppressed_bg and unmatched_predictions cohort shifts
#   stop the sweep if lambda=2 reverses the lambda=1 direction
#
# Usage:
#   scripts/train_empty_cost_l2.sh
#   DEVICE=1 scripts/train_empty_cost_l2.sh
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

RECIPE=${RECIPE:-main/pannuke-thin-t12-empty-cost-l2.yaml}
RUN=${RUN:-train58_empty-cost-l2}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

CKPT="$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"
[[ -f "$CKPT" ]] || { log "ERROR: missing oracle $CKPT"; exit 1; }
[[ -f "$RECIPE" ]] || { log "ERROR: missing recipe $RECIPE"; exit 1; }

if [[ -f "$OUTPUT/$RUN/weights/last.pt" ]]; then
  log "already complete, nothing to do: $OUTPUT/$RUN/weights/last.pt"
  exit 0
fi

log "=== empty-cost lambda=2 -> $RUN (epochs $DET_EPOCHS, end $DET_END) ==="
"$PY" -m raycasted.pipeline \
  --config "$RECIPE" \
  --output "$OUTPUT" \
  --stage train \
  --epochs "$DET_EPOCHS" \
  --end "$DET_END" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$RUN" 2>&1 | tee "$LOG_DIR/empty_cost_l2_det.log"

log "done: $OUTPUT/$RUN/weights/last.pt"
cat <<EOF

Held-out fold3 evaluation:
  .venv/bin/python -m raycasted.evals.eval_lsp \\
    --weights output/v2/$RUN/weights/last.pt \\
    --data-dir output/v2/transformed/test \\
    --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \\
    --batch 8 --device 0 --workers 0
EOF
