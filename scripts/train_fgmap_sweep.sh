#!/usr/bin/env bash
# Foreground/seed-map ablation sweep (single fold, T12).
#
# The seed map has four distinct consumers; each arm here detaches exactly one
# so the map's contribution can be attributed instead of guessed at:
#
#   nocontent    detr_seed_in_content: false
#                the seed logit is no longer concatenated to the score head.
#                This is the ONLY path by which the map reaches the
#                foreground/empty decision, so this arm is the decisive null
#                test: if it is flat, improving the map cannot help the
#                decision (docs/diag/foreground-map-not-used-2026-11-01.md
#                predicts this).
#
#   noseedlevel  detr_seed_level: false
#                the projected centroid map no longer serves as the 4th
#                deformable-attention level. Geometry path only.
#
#   fgcentroid   nulite_oracle_kind: fg_centroid + nulite_ckpt fg-center oracle
#                map TARGET change: the validated two-channel oracle
#                (fg_dice 0.8103 / center_f1 0.7487) replaces the binary
#                plateau map. Two keys, one mechanism: the checkpoint kind is
#                validated against the model, so they cannot be separated.
#
#   fgcentroid-wide  (optional, only when the wide oracle exists)
#                same as fgcentroid but with a wide-sigma centroid oracle, to
#                separate "centroid map semantics" from "map too sharp for a
#                12.8 px query grid" (probe: 3.5% of cell centres > 0.5).
#                Train the oracle first with scripts/train_oracle_fgcenter_wide.sh
#
# Every detector run is `--epochs 600 --end 332` so the LR schedule matches the
# whole train52/train55/train56/train57/train58/train59 series.
#
# Device agnostic: relative paths; works after an rsync to the second box.
#
# Usage:
#   scripts/train_fgmap_sweep.sh
#   VARIANTS="nocontent" scripts/train_fgmap_sweep.sh
#   DEVICE=1 scripts/train_fgmap_sweep.sh
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

VARIANTS=${VARIANTS:-"nocontent noseedlevel fgcentroid"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

recipe_for() {
  case "$1" in
    nocontent)       echo "main/pannuke-thin-t12-fgmap-nocontent.yaml" ;;
    noseedlevel)     echo "main/pannuke-thin-t12-fgmap-noseedlevel.yaml" ;;
    fgcentroid)      echo "main/pannuke-thin-t12-fgmap-fgcentroid.yaml" ;;
    fgcentroid-wide) echo "main/pannuke-thin-t12-fgmap-fgcentroid-wide.yaml" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

# The plateau map is the control's map; no arm here may run without it.
[[ -f "$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt" ]] \
  || { log "ERROR: missing control oracle $ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"; exit 1; }

for v in $VARIANTS; do
  recipe=$(recipe_for "$v")
  run="train60_fgmap-$v"

  log "=== $v -> $run (epochs $DET_EPOCHS, end $DET_END) ==="
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  case "$v" in
    fgcentroid)
      [[ -f "$ORACLE_DIR/nuLiteT-fold1-fg-center-t12.pt" ]] \
        || { log "ERROR: missing $ORACLE_DIR/nuLiteT-fold1-fg-center-t12.pt"; exit 1; }
      ;;
    fgcentroid-wide)
      [[ -f "$ORACLE_DIR/nuLiteT-fold1-fg-center-wide-t12.pt" ]] \
        || { log "ERROR: missing wide oracle; run scripts/train_oracle_fgcenter_wide.sh first"; exit 1; }
      ;;
  esac

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
    --name "$run" 2>&1 | tee "$LOG_DIR/fgmap_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "all fg-map arms complete"
for v in $VARIANTS; do
  echo "  train60_fgmap-$v: $OUTPUT/train60_fgmap-$v/weights/last.pt"
done
cat <<'EOF'

Held-out fold3 evaluation (control first, then each arm):
  for v in control nocontent noseedlevel fgcentroid; do
    if [[ $v == control ]]; then W=output/v2/train52_fastvit-t12/weights/last.pt
    else W=output/v2/train60_fgmap-$v/weights/last.pt; fi
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights $W \
      --data-dir output/v2/transformed/test \
      --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
      --batch 8 --device 0 --workers 0 \
      --save-predictions output/evals/fgmap/$v.npz
  done
EOF
