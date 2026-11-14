#!/usr/bin/env bash
# FG-map ablation, consumer 4: the direct seed-map loss.
#
# Mechanism: in the frozen-prior layout `lambda_seg` was hard-zeroed, so the
# oracle's seed map was never supervised directly -- it only saw DETR gradients
# after the epoch-50 thaw. This arm sets `detr_seed_map_loss: true`, which keeps
# the existing `lambda_seg: 1.0` L1(seed_logits, batch['seed_map'] /
# InstanSeg centre-weighted EDT target) term live.
#
# Expected shape of the result: epochs 0-49 the oracle is frozen, so the term is
# a constant with no gradient (wasted compute, no effect). From epoch 50 the
# thawed U-Net gets a direct anchor to the segmentation target on top of the
# detector gradients. `np_head` stays frozen (detr_frozen_oracle_unfreeze_decoder
# thaws the U-Net but not np_head), so the anchor acts on encoder/decoder
# features, not on the final map projection.
#
# Control: main/pannuke-thin-t12.yaml (one-key diff).
# Pre-registered (same bar as every other arm in this series): adopt only if
# held-out Fold3 bPQ AND macro bPQ both gain > +0.002; reject if bPQ < -0.002.
#
# Usage:
#   scripts/train_fgmap_seedloss.sh
#   DEVICE=1 scripts/train_fgmap_seedloss.sh
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
RECIPE=${RECIPE:-main/pannuke-thin-t12-fgmap-seedloss.yaml}
RUN=${RUN:-train61_fgmap-seedloss}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

[[ -f "$RECIPE" ]] || { log "ERROR: missing $RECIPE"; exit 1; }
[[ -f "$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt" ]] || {
  log "ERROR: missing oracle $ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"; exit 1; }

if [[ -f "$OUTPUT/$RUN/weights/last.pt" ]]; then
  log "already complete: $OUTPUT/$RUN/weights/last.pt"
  exit 0
fi

log "=== seed-map loss -> $RUN (epochs $DET_EPOCHS, end $DET_END) ==="
"$PY" -m raycasted.pipeline \
  --config "$RECIPE" \
  --output "$OUTPUT" \
  --stage train \
  --epochs "$DET_EPOCHS" \
  --end "$DET_END" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --workers "$WORKERS" \
  --name "$RUN" 2>&1 | tee "$LOG_DIR/fgmap_seedloss_det.log"

log "done: $OUTPUT/$RUN/weights/last.pt"
cat <<EOF

Held-out fold3 evaluation (control first, same session so both are comparable):
  for v in ctrl arm; do
    w=output/v2/train52_fastvit-t12/weights/last.pt
    [[ \$v == arm ]] && w=output/v2/$RUN/weights/last.pt
    .venv/bin/python -m raycasted.evals.eval_lsp \\
      --weights \$w \\
      --data-dir output/v2/transformed/test \\
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \\
      --batch 8 --device 0 --workers 0 --save-predictions output/evals/fgmap_seedloss/\$v.npz
  done
EOF
