#!/usr/bin/env bash
# Build a TensorRT engine for a NuLite-DETR ONNX graph on the target device.
#
# TensorRT engines are device- and version-specific: ALWAYS build on the
# machine that will run them (Jetson Orin Nano), never copy an engine from
# x86.
#
# Usage (on the Jetson):
#   BATCH=8 scripts/build_trt_engine.sh deploy/train52_fastvit-t12.onnx
#   # -> deploy/train52_fastvit-t12.fp16.engine
set -euo pipefail

ONNX=$1
BATCH=${BATCH:-8}
IMGSZ=${IMGSZ:-256}
MAX_BATCH=$(( BATCH > 8 ? BATCH : 8 ))
OUT=${2:-${ONNX%.onnx}.fp16.engine}

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$OUT" \
  --fp16 \
  --minShapes=images:1x3x${IMGSZ}x${IMGSZ} \
  --optShapes=images:${BATCH}x3x${IMGSZ}x${IMGSZ} \
  --maxShapes=images:${MAX_BATCH}x3x${IMGSZ}x${IMGSZ} \
  "$@"

echo "Engine written: $OUT"
