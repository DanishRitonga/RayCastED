#!/usr/bin/env bash
set -euo pipefail

# RayCastED — ONNX Export Script
#
# Usage:
#   bash raycasted/export/export_onnx.sh [--weights path] [--output path] [options]
#
# Examples:
#   bash raycasted/export/export_onnx.sh
#   bash raycasted/export/export_onnx.sh --weights docs/runs/v1.1/best.pt --output export/model.onnx --validate

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

WEIGHTS="${WEIGHTS:-docs/runs/v1.1/best.pt}"
OUTPUT="${OUTPUT:-export/raycast_model.onnx}"
IMGSZ="${IMGSZ:-256}"
OPSET="${OPSET:-17}"
VALIDATE="${VALIDATE:-}"

ARGS="--weights $WEIGHTS --output $OUTPUT --imgsz $IMGSZ --opset $OPSET"

if [ -n "$VALIDATE" ]; then
    ARGS="$ARGS --validate"
fi

exec uv run python -m raycasted.export.onnx_export $ARGS "$@"
