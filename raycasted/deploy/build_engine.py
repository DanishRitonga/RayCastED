r"""RayCastED — TensorRT Engine Builder (Phase 9).

Builds a TensorRT FP16 engine from an ONNX model file.
Must run on the target Jetson device — engines are GPU-architecture-specific.

Usage:
    python -m raycasted.deploy.build_engine \
        --onnx polygon_yolo.onnx \
        --engine polygon_yolo.engine \
        --fp16 \
        --workspace 4096

Requires: TensorRT installed on the device.
"""

import argparse
import subprocess
import sys
from pathlib import Path

TRTEXEC_DEFAULT = '/usr/src/tensorrt/bin/trtexec'


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    workspace: int = 4096,
    trtexec_path: str = TRTEXEC_DEFAULT,
) -> str:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to input ONNX model.
        engine_path: Path to output TensorRT engine.
        fp16: Enable FP16 precision. Default True.
        int8: Enable INT8 precision (requires calibration). Default False.
        workspace: GPU workspace in MB. Default 4096.
        trtexec_path: Path to trtexec binary.

    Returns:
        Path to the built engine file.

    Raises:
        FileNotFoundError: If trtexec or ONNX file not found.
        RuntimeError: If engine build fails.
    """
    onnx_path = Path(onnx_path).resolve()
    engine_path = Path(engine_path).resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')
    if not Path(trtexec_path).exists():
        raise FileNotFoundError(f'trtexec not found at: {trtexec_path}')

    cmd = [
        str(trtexec_path),
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--memPoolSize=workspace:{workspace}',
        '--verbose',
    ]

    if fp16:
        cmd.append('--fp16')
    if int8:
        cmd.append('--int8')

    print(f'Building TensorRT engine: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'stderr: {result.stderr[-2000:]}', file=sys.stderr)
        raise RuntimeError(f'trtexec failed with exit code {result.returncode}')

    if not engine_path.exists():
        raise RuntimeError(f'Engine file not created: {engine_path}')

    size_mb = engine_path.stat().st_size / 1024**2
    print(f'Engine built: {engine_path} ({size_mb:.1f} MB)')
    return str(engine_path)


def main():
    """CLI entry point for building TensorRT engines."""
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX')
    parser.add_argument('--onnx', required=True, help='Path to ONNX model')
    parser.add_argument('--engine', required=True, help='Path to output engine')
    parser.add_argument('--fp16', action='store_true', default=True, help='Enable FP16 (default)')
    parser.add_argument('--int8', action='store_true', help='Enable INT8')
    parser.add_argument('--workspace', type=int, default=4096, help='GPU workspace (MB)')
    parser.add_argument('--trtexec', default=TRTEXEC_DEFAULT, help='Path to trtexec')
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace=args.workspace,
        trtexec_path=args.trtexec,
    )


if __name__ == '__main__':
    main()
