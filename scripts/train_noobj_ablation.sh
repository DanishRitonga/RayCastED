#!/usr/bin/env bash
# Empty-column (no-object) weight ablation, single fold, T12.
#
#   noobj1   detr_no_object_weight 2.0 -> 1.0
#            The focal-empty column currently receives a 2x up-weight while it
#            takes ~93% of the Hungarian assignment (one-to-one over 400 grid
#            queries). 1.0 is the ETLConfig default and removes the double
#            weight. On 4 tiles the total loss drops 405.8 -> 373.3 at init, so
#            this halves the empty-column gradient relative to real classes:
#            expect a precision/recall rebalance, not a no-op.
#
# Prior: every recipe carries the comment "train18 base: 2.0; train20 1.0 was a
# fold3 regression", but it is unverified -- the only documented train20
# (output/nulite/train20, 600 ep) ablated the seed branch, not this weight, and
# records no magnitude. The current T12 contract has never had this knob
# ablated. Treat the prior as a hunch.
#
# Control: output/v2/train52_fastvit-t12/weights/last.pt
#   held-out fold3 bPQ 0.6372 / bDQ 0.7798 / bSQ 0.8032 / mPQ 0.4561 /
#   macro bPQ 0.6566 / macro mPQ 0.4618 / F1 0.7987 / P 0.8270 / R 0.7724
#   and good_suppressed_bg 1,163 / unmatched_predictions 1,262 (384 imgs,
#   8,917 GT), the cohorts this arm targets.
#
# Same `--epochs 600 --end 332` as the whole train52/train55/.../train65 series,
# so only the mechanism differs. Recipe differs from the control in exactly one
# key (verified with ETLConfig).
#
# Decision rule (pre-registered, both bPQ and macro bPQ):
#   |delta| <= 0.002 on both   -> the 2x empty prior is not load-bearing
#   gain  > 0.002              -> adopt 1.0 and revisit 0.5 / 1.5
#   drop  > 0.002              -> the 2x up-weight is load-bearing, keep 2.0
# Read the cohort audit before the PQ verdict: recall up / precision down with
# good_suppressed_bg falling is the designed trade; unmatched_predictions
# exploding is the failure mode. Also report F1, since PQ can be flat while the
# operating point moves a long way.
#
# Device agnostic: relative paths; works after an rsync to the second box.
#
# Usage:
#   scripts/train_noobj_ablation.sh
#   DEVICE=1 scripts/train_noobj_ablation.sh
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

VARIANTS=${VARIANTS:-"noobj1"}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

recipe_for() {
  case "$1" in
    noobj1) echo "main/pannuke-thin-t12-noobj1.yaml" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

run_for() {
  case "$1" in
    noobj1) echo "train66_noobj1" ;;
    *) log "ERROR: unknown variant '$1'"; exit 1 ;;
  esac
}

for v in $VARIANTS; do
  case "$v" in
    noobj1) ;;
    *) log "ERROR: unknown variant '$v' (known: noobj1)"; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

[[ -f "$ORACLE_DIR/nuLiteT-fold1-sep-t12.pt" ]] \
  || { log "ERROR: missing control oracle $ORACLE_DIR/nuLiteT-fold1-sep-t12.pt"; exit 1; }

for v in $VARIANTS; do
  recipe=$(recipe_for "$v")
  run=$(run_for "$v")

  log "=== $v -> $run (epochs $DET_EPOCHS, end $DET_END, device $DEVICE) ==="
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
    --name "$run" 2>&1 | tee "$LOG_DIR/noobj_${v}_det.log"
  log "done: $OUTPUT/$run/weights/last.pt"
done

log "no-object arm complete"
for v in $VARIANTS; do echo "  $(run_for "$v"): $OUTPUT/$(run_for "$v")/weights/last.pt"; done
cat <<'EOF'

Held-out fold3 evaluation (control first, then the arm):
  OUT=output/evals/noobj_ablation
  mkdir -p "$OUT"
  PQ=raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet
  for v in ctrl noobj1; do
    if [[ $v == ctrl ]]; then W=output/v2/train52_fastvit-t12/weights/last.pt
    else W=output/v2/train66_noobj1/weights/last.pt; fi
    .venv/bin/python -m raycasted.evals.eval_lsp \
      --weights "$W" --data-dir output/v2/transformed/test --parquet "$PQ" \
      --batch 8 --device 0 --workers 0 --save-predictions "$OUT/$v.npz" \
      > "$OUT/$v.log" 2>&1
    .venv/bin/python -m raycasted.evals.audit_lsp_cohorts \
      --cache "$OUT/$v.npz" --parquet "$PQ" --output-prefix "$OUT/${v}_cohort" \
      > "$OUT/${v}_audit.log" 2>&1
  done

CSV caveat: results.csv columns exist only while their weight is > 0, but this
arm keeps both margin terms, so the column count is unchanged (33) and
bPQ/bSQ/bDQ stay at fields 14/15/16.
EOF
