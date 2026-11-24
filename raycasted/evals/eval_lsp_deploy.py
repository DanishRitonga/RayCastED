r"""LSP-protocol evaluation on deployed runtimes (ONNX Runtime / TensorRT).

This is the Jetson-facing counterpart of ``raycasted.evals.eval_lsp``. The
metric machinery, GT loading, tile ordering, and the raw ``nc+1`` argmax
decode are imported directly from ``eval_lsp`` so the numbers are the same
protocol: original dense parquet masks, PIL rasterization, greatest-Euclidean-
interior-distance overlap ownership, no NMS and no confidence threshold. Only
the forward pass is swapped: the PyTorch model is replaced by an ONNX graph
(x86 validation) or a TensorRT engine (Jetson).

The ONNX graph contains the raw decoder outputs (see
``raycasted.export.export_nulite_detr``):
    dec_polygons [B, nq, 66] normalized polygons
    dec_logits   [B, nq, nc+1] raw logits, last column = no-object

Usage (x86, ONNX Runtime — parity check against eval_lsp):
    uv run python -m raycasted.evals.eval_lsp_deploy \
        --onnx output/v2/train52_fastvit-t12/deploy/train52_fastvit-t12.onnx \
        --meta output/v2/train52_fastvit-t12/deploy/train52_fastvit-t12.meta.json \
        --data-dir output/v2/transformed/test \
        --parquet raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet \
        --batch 16 --max-images 64

Usage (Jetson, TensorRT engine built on-device with trtexec — local .npz tiles):
    python -m raycasted.evals.eval_lsp_deploy \
        --engine output/v2/train52_fastvit-t12/deploy/train52_fastvit-t12.fp16.engine \
        --batch 8 --workers 2

Defaults load the transformed test tiles (.npz) from ``output/v2/transformed/test``
and the fold3 dense-GT parquet from ``raycasted/data/dataset/PanNuke/data/``;
if the tiles are missing they are generated from the raw fold parquet via the
ETL pipeline (ingest + transform, fold3-only workspace). Override with
``--data-dir`` / ``--parquet``, set ``--hf-dataset`` to fetch them from a
HuggingFace repo instead, or rely on the default parquet fetch: when the local
fold parquet is absent it is downloaded from ``--parquet-repo``
(default: RationAI/PanNuke, file ``data/fold3-00000-of-00001.parquet``).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.evals.eval_lsp import (
    TISSUE_NAMES,
    ParquetGT,
    TissueStats,
    collate_fn,
    rasterize_polys,
    roi_id_from_path,
)
from raycasted.model.register import register_raycast_head  # noqa: F401


def decode_deploy_outputs(polys, logits, crop_size):
    """Decode raw decoder outputs exactly like eval_lsp.run_inference_raw.

    polys: (B, nq, raycast_dim) normalized; logits: (B, nq, nc+1).
    Returns a list of per-image dicts {polys(px), labels, scores}.
    """
    logits = np.asarray(logits)
    polys = np.asarray(polys)
    keep = logits.argmax(-1) != (logits.shape[-1] - 1)
    scores = 1.0 - (1.0 / (1.0 + np.exp(-logits[..., -1])))
    results = []
    for si in range(polys.shape[0]):
        k = keep[si]
        poly_n = polys[si][k]
        labels = logits[si][k].argmax(-1)
        results.append(
            {
                'polys': poly_n * crop_size,
                'labels': labels.astype(int),
                'scores': scores[si][k],
            }
        )
    return results


# ---------------------------------------------------------------------------
# HuggingFace test-data fetch
# ---------------------------------------------------------------------------


def fetch_hf_testdata(repo_id: str, cache_dir: str | None = None) -> tuple[Path, Path | None]:
    """Download the tiles of a HF dataset repo (and its fold parquet, if any).

    Expected layout: ``test/<tile>.npz`` plus an optional ``*.parquet`` at the
    repo root or under ``data/``. The parquet is skipped for tile-only repos
    (use ``--gt-tiles`` there). Repeated calls are cache hits.
    """
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type='dataset',
            cache_dir=cache_dir,
            allow_patterns=['test/*', '*.parquet', 'data/*.parquet'],
        )
    )
    test_dir = local / 'test'
    if not test_dir.is_dir():
        raise SystemExit(f'{repo_id}: no test/ directory with .npz tiles at {local}')
    parquets = sorted(local.glob('*.parquet')) or sorted((local / 'data').glob('*.parquet'))
    if not parquets:
        return test_dir, None
    if len(parquets) > 1:
        print(f'{repo_id}: multiple parquets, using {parquets[0].name}', file=sys.stderr)
    return test_dir, parquets[0]


def fetch_hf_parquet(repo_id: str, filename: str, cache_dir: str | None = None) -> Path:
    """Download one raw fold parquet (``data/<filename>``) from a HF dataset repo."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=f'data/{filename}',
            cache_dir=cache_dir,
        )
    )


# ---------------------------------------------------------------------------
# ETL generation of the held-out test tiles (no pre-transformed data needed)
# ---------------------------------------------------------------------------


def tile_gt_from_npz(path: Path, crop: int) -> tuple[list, np.ndarray, int]:
    """Render GT for one tile from its .npz raycast annotations.

    The annotation rows are ``[class_id, cx, cy, d_1..d_n]`` (class 255 =
    ignore). Polygons are rasterized with ``rasterize_polys`` — the same PIL
    fill + greatest-Euclidean-distance overlap ownership the evaluator uses
    for predictions — so the GT masks follow the same construction rule.

    Returns (gt_masks, gt_labels, tissue) like ``ParquetGT.get``.
    """
    with np.load(path) as data:
        anns = np.asarray(data['annotations'], dtype=np.float64)
        tissue = int(data['tissue'])
    if anns.size == 0:
        return [], np.zeros((0,), dtype=int), tissue
    anns = anns[anns[:, 0] != 255]
    if len(anns) == 0:
        return [], np.zeros((0,), dtype=int), tissue
    masks, labels = rasterize_polys(anns[:, 1:], crop, crop, anns[:, 0].astype(int))
    return masks, labels, tissue


def generate_test_tiles(
    parquet_path: str,
    etl_config: str,
    final_output_dir: Path,
    stain_profile: str | None,
) -> Path:
    """Run the ETL pipeline (ingest -> transform) on one held-out fold parquet.

    Builds a temporary ETL workspace with only the given parquet (symlinked in
    as ``data/<name>.parquet``), derives a config whose split_map sends it to
    ``test``, then runs IngestionOrchestrator + TransformOrchestrator and the
    split re-organisation, producing ``<final_output_dir>/test/*.npz``.

    Args:
        parquet_path: Raw held-out fold parquet (also used as GT by the caller).
        etl_config: Base ETL yaml (e.g. main/pannuke.yaml).
        final_output_dir: Where transformed tiles are organised (test/ inside).
        stain_profile: Pre-computed population stain profile json; if None the
            profile is re-estimated from the ingested fold (slightly different
            from the full-dataset reference run).

    Returns:
        Path to the generated test tiles directory.
    """
    import yaml as _yaml

    from raycasted.data.etl.ingestors.ingestion_orchestrator import IngestionOrchestrator
    from raycasted.data.etl.transform.transform_orchestrator import TransformOrchestrator
    from raycasted.data.etl.utils.config import ETLConfig

    workspace = final_output_dir / 'etl-workspace'
    workspace.mkdir(parents=True, exist_ok=True)

    src_parquet = Path(parquet_path).resolve()
    data_dir = workspace / 'data'
    data_dir.mkdir(exist_ok=True)
    linked = data_dir / src_parquet.name
    if not linked.exists():
        linked.symlink_to(src_parquet)

    with open(etl_config) as f:
        cfg = _yaml.safe_load(f)
    cfg['global_settings']['root_dir'] = str(workspace)
    cfg['datasets']['PanNuke']['root_dir'] = 'data'
    cfg['datasets']['PanNuke']['split_args']['split_map'] = {'fold3': 'test'}
    cfg.pop('training', None)
    cfg_path = workspace / 'pannuke-fold3-eval.yaml'
    with open(cfg_path, 'w') as f:
        _yaml.dump(cfg, f, default_flow_style=False)

    ingested_dir = workspace / 'ingested'
    if not any(ingested_dir.rglob('*.npz')):
        print(f'ETL ingest: {src_parquet.name} -> {ingested_dir} ...', file=sys.stderr, flush=True)
        IngestionOrchestrator(str(cfg_path), output_dir=str(ingested_dir)).run('PanNuke')

    print(f'ETL transform -> {final_output_dir} ...', file=sys.stderr, flush=True)
    t = TransformOrchestrator(
        config_manager=ETLConfig(str(cfg_path)),
        ingested_dir=str(ingested_dir),
        final_output_dir=str(final_output_dir),
        use_gpu=False,
        stain_profile=stain_profile,
    )
    t.run_pipeline()

    # TransformOrchestrator saves tiles flat with the split in its registry;
    # move each tile into <final_output_dir>/<split>/ like RayCastPipeline does.
    moved = 0
    for row in t.registry.iter_rows(named=True):
        src = final_output_dir / Path(row['path']).name
        if not src.exists():
            continue
        dst_dir = final_output_dir / row['split']
        dst_dir.mkdir(exist_ok=True)
        src.rename(dst_dir / src.name)
        moved += 1
    print(f'Reorganised {moved} tiles into {final_output_dir}/<split>/ subdirectories.')

    return final_output_dir / 'test'


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def make_ort_session(onnx_path: str, use_cuda: bool, device_id: int = 0):
    """ONNX Runtime session; CUDA EP when available, else CPU."""
    import onnxruntime as ort

    providers = ['CPUExecutionProvider']
    if use_cuda:
        avail = ort.get_available_providers()
        if 'CUDAExecutionProvider' in avail:
            providers = [('CUDAExecutionProvider', {'device_id': device_id}), 'CPUExecutionProvider']
        else:
            print('onnxruntime built without CUDAExecutionProvider; running on CPU', file=sys.stderr)
    return ort.InferenceSession(onnx_path, providers=providers), providers


def load_trt_runner(engine_path: str):
    """TensorRT runner with dynamic-batch support (Jetson pycuda + tensorrt)."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = cuda.Stream()

    input_name = None
    outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            outputs.append((name, engine.get_tensor_dtype(name)))
    trt_dtype = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
    }
    return engine, context, stream, input_name, outputs, cuda, trt_dtype


class TrtRunner:
    """Runs the engine with per-call batch shape; allocates max-profile buffers."""

    def __init__(self, engine_path: str):
        (
            self.engine,
            self.context,
            self.stream,
            self.input_name,
            self.outputs,
            cuda,
            trt_dtype,
        ) = load_trt_runner(engine_path)
        self.cuda = cuda
        self.trt_dtype_map = trt_dtype
        self.out_names = [name for name, _ in self.outputs]

        # Buffer sizes from the (max) profile so any batch <= max works.
        profile = 0
        max_shape = self.engine.get_tensor_profile_shape(self.input_name, profile)[-1]
        self.max_batch = int(max_shape[0])
        d_input = cuda.mem_alloc(int(np.prod(max_shape)) * np.dtype(np.float32).itemsize)
        self.in_host = cuda.pagelocked_empty(int(np.prod(max_shape)), np.float32)
        self.in_dev = d_input
        self.host_out = {}
        self.dev_out = {}
        for name, dtype in self.outputs:
            shape = self.engine.get_tensor_shape(name)
            # Dynamic non-batch dims do not occur in this graph; only batch varies.
            if shape[0] == -1:
                pshape = self.engine.get_tensor_profile_shape(name, profile)[-1]
                shape = pshape
            n = int(np.prod(shape))
            self.host_out[name] = cuda.pagelocked_empty(n, self.trt_dtype_map[dtype])
            self.dev_out[name] = cuda.mem_alloc(n * self.host_out[name].itemsize)
            self.context.set_tensor_address(name, int(self.dev_out[name]))
        self.ctx_shapes = None

    def run(self, blob: np.ndarray) -> dict:
        """blob: (B, 3, H, W) float32. Returns {name: (B, ...) float32}."""
        b = blob.shape[0]
        shape = tuple(blob.shape)
        if self.ctx_shapes != shape:
            self.context.set_input_shape(self.input_name, shape)
            self.ctx_shapes = shape
        np.copyto(self.in_host[: blob.size], blob.ravel())
        self.cuda.memcpy_htod_async(self.in_dev, self.in_host[: blob.size], self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for name in self.out_names:
            self.cuda.memcpy_dtoh_async(self.host_out[name], self.dev_out[name], self.stream)
        self.stream.synchronize()
        out = {}
        for name in self.out_names:
            full = self.engine.get_tensor_shape(name)
            if full[0] == -1:
                pshape = self.engine.get_tensor_profile_shape(name, 0)[-1]
                full = list(pshape)
                full[0] = b
            out[name] = self.host_out[name][: int(np.prod(full))].reshape(full).astype(np.float32)
        return out


def make_trt_engine(onnx_path: str, batch: int, imgsz: int = 256):
    """Build a TensorRT engine on-device with trtexec and return its path."""
    import shutil
    import subprocess

    trtexec = shutil.which('trtexec')
    if trtexec is None:
        raise SystemExit('trtexec not found; build the engine manually (see scripts/build_trt_engine.sh)')
    out = str(Path(onnx_path).with_suffix('.engine'))
    cmd = [
        trtexec,
        f'--onnx={onnx_path}',
        f'--saveEngine={out}',
        '--fp16',
        f'--minShapes=images:1x3x{imgsz}x{imgsz}',
        f'--optShapes=images:{batch}x3x{imgsz}x{imgsz}',
        f'--maxShapes=images:{max(batch, 8)}x3x{imgsz}x{imgsz}',
    ]
    print('Building TensorRT engine:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the LSP protocol with an ONNX/TensorRT forward pass."""
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--onnx', help='Path to the exported .onnx graph')
    group.add_argument('--engine', help='Path to a TensorRT .engine built on this device')
    ap.add_argument(
        '--meta',
        help='Companion .meta.json from the export (default: auto-locate the '
        'single *.meta.json next to --onnx/--engine)',
    )
    ap.add_argument(
        '--hf-dataset',
        help='HF dataset repo id holding test/*.npz + the fold parquet; '
        'overrides local tile loading when set',
    )
    ap.add_argument(
        '--data-dir',
        default='output/v2/transformed/test',
        help='Held-out transformed test tiles (.npz) (default: output/v2/transformed/test)',
    )
    ap.add_argument(
        '--parquet',
        default='raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet',
        help='Held-out fold parquet with dense GT '
        '(default: raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet); '
        'also the raw input when tiles are generated with the ETL pipeline; '
        'fetched from --parquet-repo on HuggingFace when missing locally',
    )
    ap.add_argument(
        '--etl-config',
        default='main/pannuke.yaml',
        help='Base ETL yaml used when test tiles must be generated (default: main/pannuke.yaml)',
    )
    ap.add_argument(
        '--stain-profile',
        default='main/pannuke-v2-stain-profile.json',
        help='Pre-computed population stain profile for tile generation '
        '(default: main/pannuke-v2-stain-profile.json; pass "" to re-estimate from the fold)',
    )
    ap.add_argument('--hf-cache-dir', default=None, help='Custom HF download cache dir (default: HF default)')
    ap.add_argument(
        '--gt-tiles',
        action='store_true',
        help='Render GT from the .npz annotations (raycast polygons rasterized with '
        'the same PIL+EDT rule) instead of the dense parquet masks; when the tiles '
        'already exist the parquet is not required at all. WARNING: this deviates '
        'from the eval_lsp literature protocol (dense parquet GT), so the numbers '
        'are NOT comparable to eval_lsp results',
    )
    ap.add_argument(
        '--parquet-repo',
        default='RationAI/PanNuke',
        help='HF dataset repo to fetch the raw fold parquet from when the local '
        '--parquet is missing (default: RationAI/PanNuke)',
    )
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', default='0', help='CUDA device id for ORT (ignored on CPU/CPU EP)')
    ap.add_argument('--cpu', action='store_true', help='Force ONNX Runtime CPU EP')
    ap.add_argument('--max-images', type=int, help='Evaluate the first N sorted tiles (diagnostic)')
    ap.add_argument('--save-predictions', type=Path, help='Save raw predictions (eval_lsp npz format)')
    ap.add_argument('--build-engine', action='store_true', help='Build a TRT engine from --onnx via trtexec first')
    args = ap.parse_args()
    if args.save_predictions is not None and args.engine is not None:
        ap.error('--save-predictions is only supported with --onnx (ORT backend)')

    if args.hf_dataset is None:
        tiles_dir = Path(args.data_dir)
        have_tiles = tiles_dir.is_dir() and any(tiles_dir.glob('*.npz'))
        parquet = Path(args.parquet)
        if not have_tiles:
            # Tiles must be generated, and that needs the raw parquet.
            if not parquet.is_file():
                cands = sorted(tiles_dir.glob('*.parquet')) or sorted(tiles_dir.parent.glob('*.parquet'))
                if len(cands) == 1:
                    args.parquet = str(cands[0])
                    parquet = cands[0]
                    print(f'Using parquet found next to tiles: {args.parquet}', file=sys.stderr)
                else:
                    print(
                        f'--parquet not found: {parquet}; fetching fold parquet '
                        f'data/{parquet.name} from HuggingFace repo {args.parquet_repo} ...',
                        file=sys.stderr,
                        flush=True,
                    )
                    parquet = fetch_hf_parquet(args.parquet_repo, parquet.name, args.hf_cache_dir)
                    args.parquet = str(parquet)
                    print(f'  parquet: {args.parquet}', file=sys.stderr)
            print(
                f'No tiles at {tiles_dir}: generating them from {args.parquet} via the ETL pipeline ...',
                file=sys.stderr,
                flush=True,
            )
            test_dir = generate_test_tiles(
                args.parquet,
                args.etl_config,
                tiles_dir.parent,
                args.stain_profile or None,
            )
            args.data_dir = str(test_dir)
            tiles_dir = test_dir
        elif args.gt_tiles:
            print(
                'GT source: .npz annotations (--gt-tiles); the dense parquet is not required',
                file=sys.stderr,
            )
        elif not parquet.is_file():
            cands = sorted(tiles_dir.glob('*.parquet')) or sorted(tiles_dir.parent.glob('*.parquet'))
            if len(cands) == 1:
                args.parquet = str(cands[0])
                parquet = cands[0]
                print(f'Using parquet found next to tiles: {args.parquet}', file=sys.stderr)
            else:
                print(
                    f'--parquet not found: {parquet}; fetching fold parquet '
                    f'data/{parquet.name} from HuggingFace repo {args.parquet_repo} ...',
                    file=sys.stderr,
                    flush=True,
                )
                parquet = fetch_hf_parquet(args.parquet_repo, parquet.name, args.hf_cache_dir)
                args.parquet = str(parquet)
                print(f'  parquet: {args.parquet}', file=sys.stderr)

    import json

    if args.meta is None:
        artifact = args.onnx or args.engine
        metas = sorted(Path(artifact).parent.glob('*.meta.json'))
        if len(metas) == 1:
            args.meta = str(metas[0])
        else:
            ap.error(f'--meta required: found {len(metas)} *.meta.json next to {artifact}')

    with open(args.meta) as f:
        meta = json.load(f)
    crop = int(meta.get('crop_size', 256))
    if meta.get('arch') != 'nulite-detr':
        raise SystemExit(f'meta is for arch={meta.get("arch")!r}; expected nulite-detr')
    if int(meta.get('n_rays', 64)) != 64:
        raise SystemExit(f"meta n_rays={meta.get('n_rays')} but eval_lsp protocol assumes 64")

    register_raycast_head()

    from raycasted.data.etl.utils import constants as _const

    _const.configure_rays(64)

    default_data_dir = 'output/v2/transformed/test'
    default_parquet = 'raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet'
    if args.hf_dataset is not None and (args.data_dir == default_data_dir or args.parquet == default_parquet):
        print(f'Fetching test data from HuggingFace: {args.hf_dataset} ...', file=sys.stderr, flush=True)
        hf_dir, hf_parquet = fetch_hf_testdata(args.hf_dataset, args.hf_cache_dir)
        args.data_dir = str(hf_dir) if args.data_dir == default_data_dir else args.data_dir
        if args.parquet == default_parquet:
            if hf_parquet is not None:
                args.parquet = str(hf_parquet)
            elif args.gt_tiles:
                args.parquet = None
                print(
                    f'  parquet: none in {args.hf_dataset} (OK with --gt-tiles)',
                    file=sys.stderr,
                )
            else:
                print(
                    f'  parquet: none in {args.hf_dataset}; fetching '
                    f'{Path(default_parquet).name} from {args.parquet_repo}',
                    file=sys.stderr,
                    flush=True,
                )
                args.parquet = str(
                    fetch_hf_parquet(args.parquet_repo, Path(default_parquet).name, args.hf_cache_dir)
                )
        print(f'  tiles:   {args.data_dir}', file=sys.stderr)
        if args.parquet is not None:
            print(f'  parquet: {args.parquet}', file=sys.stderr)

    tile_paths = sorted(Path(args.data_dir).glob('*.npz'))
    if args.gt_tiles:
        print(
            'WARNING: --gt-tiles renders GT from the npz raycast annotations '
            '(not the dense parquet) — NOT comparable to eval_lsp literature numbers',
            file=sys.stderr,
        )
        gt = None
    else:
        gt = ParquetGT(args.parquet)
        assert len(tile_paths) == len(gt), (len(tile_paths), len(gt))
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError('--max-images must be positive')
        tile_paths = tile_paths[: args.max_images]

    ds = RayCastTileDataset(str(args.data_dir), crop_size=crop, augment=False)
    if args.max_images is not None:
        from torch.utils.data import Subset

        ds = Subset(ds, range(len(tile_paths)))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn
    )

    if args.build_engine:
        if args.onnx is None:
            ap.error('--build-engine requires --onnx')
        args.engine = make_trt_engine(args.onnx, args.batch, crop)
        args.onnx = None

    if args.onnx is not None:
        session, providers = make_ort_session(args.onnx, use_cuda=not args.cpu, device_id=int(args.device))
        print(f'Loaded ONNX: {args.onnx} (providers: {providers})')

        def infer(images):
            blob = images.numpy()
            outputs = session.run(None, {'images': blob})
            return {'dec_polygons': outputs[0], 'dec_logits': outputs[1]}

    else:
        runner = TrtRunner(args.engine)
        print(f'Loaded TensorRT engine: {args.engine} (max_batch={runner.max_batch})')
        if args.batch > runner.max_batch:
            raise SystemExit(f'--batch {args.batch} exceeds engine max batch {runner.max_batch}')

        def infer(images):
            blob = images.numpy()
            return runner.run(blob)

    per_tissue = {}
    total = TissueStats()
    preds = []
    n_pred_raw_total = n_gt_total = 0
    t0 = time.perf_counter()
    n_done = 0
    for batch in dl:
        images = batch['img']
        outputs = infer(images)
        decoded = decode_deploy_outputs(outputs['dec_polygons'], outputs['dec_logits'], crop)
        preds.extend(decoded)
        n_done += images.shape[0]
        if n_done % 500 < images.shape[0]:
            print(f'  {n_done}/{len(ds)} images ({time.perf_counter() - t0:.1f}s)', file=sys.stderr, flush=True)
    infer_ms = (time.perf_counter() - t0) / max(len(ds), 1) * 1000.0

    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.r_[0, np.cumsum([len(result['polys']) for result in preds])]
        np.savez_compressed(
            args.save_predictions,
            paths=np.asarray([path.name for path in tile_paths]),
            offsets=offsets,
            polygons=np.concatenate([result['polys'] for result in preds], axis=0),
            labels=np.concatenate([result['labels'] for result in preds], axis=0),
        )

    for path, res in zip(tile_paths, preds):
        if gt is None:
            gt_masks, gt_labels, tissue = tile_gt_from_npz(path, crop)
        else:
            gt_masks, gt_labels, tissue = gt.get(roi_id_from_path(path))
        pred_polys = res['polys']
        pred_labels = res['labels']
        n_pred_raw_total += len(pred_polys)
        if len(pred_polys) == 0:
            pm, pl = [], np.zeros((0,), dtype=int)
        else:
            pm, pl = rasterize_polys(pred_polys, crop, crop, pred_labels)
        n_gt_total += len(gt_masks)
        ts = per_tissue.setdefault(tissue, TissueStats())
        ts.update(pm, pl, gt_masks, gt_labels)
        ts.update_f1(pm, gt_masks)
        total.update(pm, pl, gt_masks, gt_labels)
        total.update_f1(pm, gt_masks)

    print(f'GT instances: {n_gt_total}   Raw pred instances (all non-∅): {n_pred_raw_total}')
    print(f'Inference: {infer_ms:.1f} ms/img ({len(ds)} images, batch {args.batch})')
    print()
    print(f'{"tissue":<15}{"bDQ":>8}{"bSQ":>8}{"bPQ":>8}{"mPQ":>8}{"F1":>8}{"P":>8}{"R":>8}')
    bpq_list, mpq_list = [], []
    for t in sorted(per_tissue):
        ts = per_tissue[t]
        s = ts.summary()
        f1, p, r = ts.f1()
        bpq_list.append(s['bPQ'])
        mpq_list.append(s['mPQ'])
        name = TISSUE_NAMES[t] if t < len(TISSUE_NAMES) else str(t)
        print(f'{name:<15}{s["bDQ"]:>8.4f}{s["bSQ"]:>8.4f}{s["bPQ"]:>8.4f}{s["mPQ"]:>8.4f}{f1:>8.4f}{p:>8.4f}{r:>8.4f}')
    if bpq_list:
        print(
            f'{"Average":<15}{"":>8}{"":>8}{np.mean(bpq_list):>8.4f}'
            f'{np.mean(mpq_list):>8.4f}   (macro over {len(bpq_list)} tissues)'
        )
    print()
    st = total.summary()
    f1, p, r = total.f1()
    print('Whole-fold totals:')
    for k in ['bDQ', 'bSQ', 'bPQ', 'mDQ', 'mSQ', 'mPQ']:
        print(f'  {k}: {st[k]:.4f}')
    print(f'  F1: {f1:.4f}  P: {p:.4f}  R: {r:.4f}  (centroid, r=12, global)')


if __name__ == '__main__':
    main()
