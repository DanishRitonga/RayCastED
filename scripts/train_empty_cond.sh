#!/usr/bin/env bash
# Single-mechanism ablation: conditional (logit-renormalised) empty-column cost.
#
# Control is the lambda=1 additive arm, train57_empty-cost (held-out Fold3 bPQ
# 0.6391, macro bPQ 0.6578), which itself differs from train52_fastvit-t12 in
# exactly one key. This script trains ONLY the treated arm, which differs from
# train57 in exactly one config key: detr_matcher_empty_cost_mode:
# additive -> conditional. lambda is unused in conditional mode.
#
# Mechanism: instead of adding lambda * -log(1 - p_empty), the class probability
# fed to the focal term is renormalised over {true class, no-object}:
#     p_cond = p_class / (p_class + p_empty + eps)
# So a query that splits its probability mass with the empty column scores worse
# than one that is confidently foreground, at the same raw p_class. Unlike the
# additive term this is NOT row-constant: it changes the relative cost of queries
# for a given GT, not just which rows are left unused.
#
# Seg stage is skipped: the fold1-only T12 oracle already exists.
# Device agnostic: relative paths, so the tree can be rsync'd to another box.
#
# Pre-registered decision rule (eval_lsp.py, raw nc+1 argmax, no NMS):
#   adopt  only if bPQ AND macro bPQ both gain > +0.002 over train52 0.6372 / 0.6566
#   reject if bPQ drops > 0.002
#   always report good_suppressed_bg and unmatched_predictions cohort shifts
#
# Usage:
#   scripts/train_empty_cond.sh
#   DEVICE=1 scripts/train_empty_cond.sh
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

RECIPE=${RECIPE:-main/pannuke-thin-t12-empty-cost-cond.yaml}
RUN=${RUN:-train59_empty-cond}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

CKPT="$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"
[[ -f "$CKPT" ]] || { log "ERROR: missing oracle $CKPT"; exit 1; }
[[ -f "$RECIPE" ]] || { log "ERROR: missing recipe $RECIPE"; exit 1; }

if [[ -f "$OUTPUT/$RUN/weights/last.pt" ]]; then
  log "already complete, nothing to do: $OUTPUT/$RUN/weights/last.pt"
  exit 0
fi

log "=== empty-cond -> $RUN (epochs $DET_EPOCHS, end $DET_END) ==="
"$PY" -m raycasted.pipeline \
  --config "$RECIPE" \
  --output "$OUTPUT" \
  --stage train \
  --epochs "$DET_EPOCHS" \
  --end "$DET_END" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$RUN" 2>&1 | tee "$LOG_DIR/empty_cond_det.log"

log "done: $OUTPUT/$RUN/weights/last.pt"
cat <<EOF

Held-out fold3 evaluation:
  .venv/bin/python -m raycasted.evals.eval_lsp \\
    --weights output/v2/$RUN/weights/last.pt \\
    --data-dir output/v2/transformed/test \\
    --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \\
    --batch 8 --device 0 --workers 0
EOF
