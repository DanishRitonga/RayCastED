#!/usr/bin/env bash
# 3-fold CV, oracle stage only, for every FastViT backbone in the sweep
# (t8 / t12 / s12 / sa12). Detector training is NOT part of this script.
#
# Rotation map (each fold is train in one rotation, val in another, test in the
# third; no fold is ever both train and test, so the frozen oracle never sees
# the held-out fold):
#
#   rotation  | train | val   | test  | output dir    | oracle prefix
#   ----------|-------|-------|-------|---------------|---------------------------
#   r1 (done) | fold1 | fold2 | fold3 | output/v2     | nuLiteT-fold1-sep-<v>.pt
#   r2        | fold2 | fold3 | fold1 | output/v2-r2  | nuLiteT-fold2-sep-<v>.pt
#   r3        | fold3 | fold1 | fold2 | output/v2-r3  | nuLiteT-fold3-sep-<v>.pt
#
# r1 already exists for all four backbones (t8/t12/s12/sa12, trained Sep-Nov),
# so this script builds 2 rotations x 4 backbones = 6 new oracles, reusing the
# r2/r3 tile sets that are already built. Backbones are the four FastViT points
# kept by the backbone sweep; the edgeNeXt variants are excluded because every
# one of them lost macro bPQ to the FastViT family.
#
# Why the oracle must be retrained per rotation: the detector loads the oracle
# frozen and its foreground map is an input. A fold1 oracle is legitimate for r1
# (fold1 is r1's training fold) but would leak for r2, whose held-out fold IS
# fold1.
#
# Recipes: main/pannuke-cv-<fold2|fold3>-<v>.yaml, one per rotation x backbone.
# Each is a copy of main/pannuke-thin-<v>.yaml with exactly two keys changed:
# datasets.PanNuke.split_args.split_map (rotation) and training.nulite_ckpt
# (that rotation's oracle). Architecture is untouched, so a CV model is the same
# model every ablation measured for that backbone.
#
# The memmap caches under output/oracle/cache-<fold> are shared by all four
# backbones on purpose: they hold decoded tiles and masks, which are identical
# across backbones, and the cache filename does not encode the variant.
#
# Tile counts are PanNuke's true per-fold counts and are checked after the
# transform, because the oracle trainer requires sorted glob order to match
# parquet order and a short tile set silently mispairs images and masks.
#
# Usage:
#   scripts/train_cv_oracles.sh                        # both rotations, all 4 backbones
#   SEG_VARIANTS="t8 s12" scripts/train_cv_oracles.sh  # subset
#   ROTATIONS="r2" DEVICE=1 scripts/train_cv_oracles.sh
#   SKIP_INGEST=1 scripts/train_cv_oracles.sh          # reuse an existing ingested tree
#                                                      # (only if built with the same recipe)
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
DEVICE=${DEVICE:-0}
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-4}

DATA_DIR=${DATA_DIR:-raycasted/data/dataset/PanNuke/data}
ORACLE_DIR=${ORACLE_DIR:-output/oracle}
LOG_DIR=${LOG_DIR:-output/logs}

SEG_EPOCHS=${SEG_EPOCHS:-120}
SEG_VARIANTS=${SEG_VARIANTS:-"t8 t12 s12 sa12"}
ROTATIONS=${ROTATIONS:-"r2 r3"}

mkdir -p "$LOG_DIR" "$ORACLE_DIR"

log() { printf '[cv-oracle] %s\n' "$*"; }

for r in $ROTATIONS; do
  case "$r" in
    r2|r3) ;;
    *) log "ERROR: unknown rotation '$r' (known: r2 r3)"; exit 2 ;;
  esac
done
for v in $SEG_VARIANTS; do
  case "$v" in
    t8|t12|s12|sa12) ;;
    *) log "ERROR: unknown backbone '$v' (known: t8 t12 s12 sa12)"; exit 2 ;;
  esac
done

recipe_for() {
  case "$1" in
    r2) echo "main/pannuke-cv-fold2-$2.yaml" ;;
    r3) echo "main/pannuke-cv-fold3-$2.yaml" ;;
  esac
}

out_for() {
  case "$1" in
    r2) echo "${OUTPUT_R2:-output/v2-r2}" ;;
    r3) echo "${OUTPUT_R3:-output/v2-r3}" ;;
  esac
}

train_fold_for() {
  case "$1" in
    r2) echo fold2 ;;
    r3) echo fold3 ;;
  esac
}

val_fold_for() {
  case "$1" in
    r2) echo fold3 ;;
    r3) echo fold1 ;;
  esac
}

test_fold_for() {
  case "$1" in
    r2) echo fold1 ;;
    r3) echo fold2 ;;
  esac
}

count_for() {
  case "$1" in
    fold1) echo 2656 ;;
    fold2) echo 2523 ;;
    fold3) echo 2722 ;;
  esac
}

check_tiles() {
  local dir=$1 want=$2 label=$3
  [[ -d "$dir" ]] || { log "ERROR: missing tile dir $dir"; return 1; }
  local got
  got=$(find "$dir" -maxdepth 1 -type f | wc -l)
  if [[ "$got" != "$want" ]]; then
    log "WARNING: $label has $got tiles, expected $want — verify split_map and tiling before training"
  else
    log "ok: $label $got tiles"
  fi
}

for r in $ROTATIONS; do
  out=$(out_for "$r")
  tr_fold=$(train_fold_for "$r")
  va_fold=$(val_fold_for "$r")
  # every recipe of a rotation shares the same split map, so any one works
  recipe=$(recipe_for "$r" "${SEG_VARIANTS%% *}")

  log "================ rotation $r: train $tr_fold, val $va_fold, test $(test_fold_for "$r") ================"
  log "tiles $out/transformed, backbones: $SEG_VARIANTS"
  [[ -f "$recipe" ]] || { log "ERROR: missing $recipe"; exit 1; }

  mkdir -p "$out"

  # ---------------------------------------------------------------- ingest
  # NOTE: split_map is applied at INGEST time (etl/ingestors/_base.py:90-93), not
  # at transform time. An ingested tree built with a different rotation's recipe
  # carries the wrong split labels and the transform would silently honour them,
  # so a reused tree is only safe if it was ingested with this same rotation.
  if [[ -d "$out/ingested/PanNuke" ]]; then
    log "ingest: reusing $out/ingested/PanNuke (must have been ingested with recipe-based split_map for $r)"
  elif [[ "${SKIP_INGEST:-0}" == "1" ]]; then
    log "ERROR: SKIP_INGEST=1 but $out/ingested/PanNuke does not exist"; exit 1
  else
    log "ingest: $out/ingested ($(date +%H:%M:%S))"
    "$PY" -m raycasted.pipeline \
      --config "$recipe" --output "$out" --stage ingest \
      2>&1 | tee "$LOG_DIR/cv_${r}_ingest.log"
    log "ingest done ($(date +%H:%M:%S))"
  fi

  # ------------------------------------------------------------- transform
  if [[ -n "$(ls -A "$out/transformed/train" 2>/dev/null || true)" ]]; then
    log "transform: reusing $out/transformed"
  else
    log "transform: $out/transformed ($(date +%H:%M:%S))"
    "$PY" -m raycasted.pipeline \
      --config "$recipe" --output "$out" --stage transform \
      2>&1 | tee "$LOG_DIR/cv_${r}_transform.log"
    log "transform done ($(date +%H:%M:%S))"
  fi

  check_tiles "$out/transformed/train" "$(count_for "$tr_fold")" "$r train ($tr_fold)"
  check_tiles "$out/transformed/val"   "$(count_for "$va_fold")" "$r val ($va_fold)"

  # --------------------------------------------------------------- oracles
  for v in $SEG_VARIANTS; do
    oracle="$ORACLE_DIR/nuLiteT-${tr_fold}-sep-${v}.pt"
    if [[ -f "$oracle" ]]; then
      log "oracle $v: exists, skipping $oracle"
      continue
    fi
    log "oracle $v: training fastvit_$v on $tr_fold (val $va_fold), ${SEG_EPOCHS} epochs ($(date +%H:%M:%S))"
    "$PY" -m raycasted.model.train_nulite_seg \
      --train-tiles "$out/transformed/train" \
      --train-parquet "$DATA_DIR/${tr_fold}-00000-of-00001.parquet" \
      --val-tiles "$out/transformed/val" \
      --val-parquet "$DATA_DIR/${va_fold}-00000-of-00001.parquet" \
      --variant "fastvit_$v" \
      --separable-decoder \
      --epochs "$SEG_EPOCHS" \
      --batch "$BATCH" \
      --device "$DEVICE" \
      --cache-root "$ORACLE_DIR/cache-${tr_fold}" \
      --out "$oracle" \
      2>&1 | tee "$LOG_DIR/cv_${r}_${v}_seg.log"
    log "oracle $v: done $oracle"
  done
done

log "oracle stage complete"
for r in $ROTATIONS; do
  for v in $SEG_VARIANTS; do
    echo "  $r/$v: $(out_for "$r")/transformed + $ORACLE_DIR/nuLiteT-$(train_fold_for "$r")-sep-$v.pt"
  done
done
cat <<'EOF'

Detector stage is deliberately NOT run here. Each rotation x backbone has its
own recipe that already points nulite_ckpt at that rotation's oracle:

  # ~6.6 h per run at --epochs 600 --end 332; 8 runs = ~53 h. Choose which
  # backbones actually get the full rotation before starting.
  for v in t8 t12 s12 sa12; do
    .venv/bin/python -m raycasted.pipeline \
      --config "main/pannuke-cv-fold2-$v.yaml" --output output/v2-r2 \
      --stage train --epochs 600 --end 332 --batch 16 --device 0 --workers 4 \
      --name "train67_cv-fold2-$v" 2>&1 | tee "output/logs/train67_cv-fold2-${v}_det.log"
    .venv/bin/python -m raycasted.pipeline \
      --config "main/pannuke-cv-fold3-$v.yaml" --output output/v2-r3 \
      --stage train --epochs 600 --end 332 --batch 16 --device 0 --workers 4 \
      --name "train68_cv-fold3-$v" 2>&1 | tee "output/logs/train68_cv-fold3-${v}_det.log"
  done

Then evaluate each rotation on its own held-out fold:

  r1: output/v2/train52_fastvit-t12 (and train50 for t8, train52_* for s12/sa12) -> fold3
  r2: output/v2-r2/train67_cv-fold2-<v>/weights/last.pt -> fold1 parquet
  r3: output/v2-r3/train68_cv-fold3-<v>/weights/last.pt -> fold2 parquet

Report mean +/- std over the three rotations per backbone.
EOF
