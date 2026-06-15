"""RayCastED — Visual Inference on PanNuke test tiles.

Runs inference and overlays predicted + GT polygons on images.
Saves annotated images to disk for inspection.

Usage:
    uv run python main/visual_inference.py \
        --weights output/run25/weights/best.pt \
        --data-dir output/pannuke/transformed/test \
        --out-dir output/visual_inference \
        --n-images 50 \
        --conf 0.2

    # Auto-transform from config
    uv run python main/visual_inference.py \
        --config main/pannuke.yaml \
        --output output/pannuke \
        --weights output/run25/weights/best.pt \
        --out-dir output/visual_inference \
        --n-images 50
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from raycasted.data.etl.loader.raycast_dataset import RayCastTileDataset
from raycasted.data.etl.ops.convert import decode_to_vertices
from raycasted.data.etl.utils.constants import configure_rays
from raycasted.model.register import register_raycast_head

CLASS_COLORS_GT = [(0, 255, 0)]
CLASS_COLORS_PRED = [(255, 0, 0)]


def load_model(weights_path: str, device: torch.device):
    """Load trained RayCastED model from checkpoint."""
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    model = model.float().to(device)
    model.eval()
    return model


def run_inference(model, images: torch.Tensor, device: torch.device, conf_threshold: float, raycast_dim: int):
    """Run model inference on a batch, returning per-image predictions."""
    with torch.no_grad():
        raw_out = model(images)
        decoded = raw_out[0] if isinstance(raw_out, tuple) else raw_out

    batch_results = []
    for si in range(decoded.shape[0]):
        det = decoded[si].cpu().numpy()
        if det.ndim == 2 and det.shape[1] == raycast_dim + 2:
            conf_mask = det[:, raycast_dim] > conf_threshold
            det = det[conf_mask]

        if det.shape[0] > 0:
            pred_poly = det[:, :raycast_dim]
            pred_cls = det[:, raycast_dim + 1].astype(int)
            pred_conf = det[:, raycast_dim]
        else:
            pred_poly = np.zeros((0, raycast_dim), dtype=np.float32)
            pred_cls = np.array([], dtype=int)
            pred_conf = np.array([], dtype=np.float32)

        batch_results.append({'pred_polys': pred_poly, 'pred_cls': pred_cls, 'pred_conf': pred_conf})

    return batch_results


def draw_polygons(
    img: np.ndarray,
    polys: np.ndarray,
    colors: list[tuple[int, int, int]],
    thickness: int = 1,
) -> np.ndarray:
    """Draw polygon outlines on an image (no labels, no centroids)."""
    if polys.shape[0] == 0:
        return img

    cx = polys[:, 0]
    cy = polys[:, 1]
    rays = polys[:, 2:]
    vertices = decode_to_vertices(rays, cx, cy)

    for i in range(polys.shape[0]):
        color = colors[0]  # single color for all
        pts = vertices[i].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return img


def _simple_collate(batch):
    import torch as _torch

    images = _torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]

    target_list = []
    for batch_idx, labels in enumerate(labels_list):
        if labels.shape[0] == 0:
            continue
        batch_col = np.full((labels.shape[0], 1), batch_idx, dtype=np.float32)
        target_list.append(np.concatenate([batch_col, labels], axis=1))

    if target_list:
        targets = _torch.from_numpy(np.concatenate(target_list, axis=0))
    else:
        ann_width = next((lbl.shape[1] for lbl in labels_list if lbl.ndim == 2 and lbl.shape[1] > 0), 35)
        targets = _torch.zeros((0, 2 + ann_width), dtype=_torch.float32)

    return {
        'img': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],
    }


def main():
    """Parse args and run visual inference."""
    parser = argparse.ArgumentParser(description='RayCastED Visual Inference')
    parser.add_argument('--weights', required=True, help='Path to trained .pt checkpoint')
    parser.add_argument('--data-dir', default=None, help='Path to test tiles directory (.npz files)')
    parser.add_argument('--config', default=None, help='ETL config YAML (for auto-transform)')
    parser.add_argument('--output', default=None, help='Output dir (required with --config)')
    parser.add_argument('--out-dir', required=True, help='Directory to save annotated images')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='0', help='Device (cpu, 0, 0,1)')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    parser.add_argument('--n-images', type=int, default=50, help='Number of images to visualise')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument(
        '--mode',
        choices=['overlay', 'side_by_side', 'pred_only', 'gt_only'],
        default='overlay',
        help='Visualisation mode',
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None and args.config is not None and args.output is not None:
        from raycasted.data.etl.transform.transform_orchestrator import TransformOrchestrator
        from raycasted.data.etl.utils.config import ETLConfig

        config = ETLConfig(args.config)
        transformed_dir = Path(args.output) / 'transformed'
        test_dir = transformed_dir / 'test'

        if not test_dir.exists() or not any(test_dir.glob('*.npz')):
            print(f'Test tiles not found at {test_dir}. Running transform...')
            t = TransformOrchestrator(
                config_manager=config,
                ingested_dir=str(Path(args.output) / 'ingested'),
                final_output_dir=str(transformed_dir),
            )
            t.run_pipeline()
            from raycasted.pipeline import RayCastPipeline

            RayCastPipeline._organize_by_split(None, t.registry)
        data_dir = str(test_dir)

    if data_dir is None:
        parser.error('Either --data-dir or both --config and --output must be provided')

    data_dir = Path(data_dir)
    if not data_dir.exists():
        parser.error(f'Data directory not found: {data_dir}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)

    print(f'Loading model: {args.weights}')
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    n_rays = training_args.get('n_rays', 32)
    raycast_dim = 2 + n_rays
    print(f'  crop_size={crop_size}, n_rays={n_rays}, raycast_dim={raycast_dim}')

    configure_rays(n_rays)

    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    n_total = min(args.n_images, len(dataset))
    print(f'Visualising {n_total} images from {len(dataset)} tiles')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=_simple_collate,
    )

    saved = 0
    global_idx = 0
    for batch in dataloader:
        if saved >= n_total:
            break

        images = batch['img'].to(device)
        batch_size = images.shape[0]

        batch_results = run_inference(model, images, device, args.conf, raycast_dim)

        for si in range(batch_size):
            if saved >= n_total:
                break

            tile_path = dataset.tile_paths[global_idx]
            tile_data = np.load(tile_path)
            raw_img = tile_data['image']
            if raw_img.ndim == 2:
                raw_img = np.stack([raw_img] * 3, axis=-1)
            vis_img = raw_img.copy()

            mask = batch['batch_idx'] == si
            gt_cls = batch['cls'][mask].numpy().flatten().astype(int)
            gt_poly = batch['bboxes'][mask].numpy()

            if gt_poly.shape[0] > 0:
                gt_poly_denorm = gt_poly.copy()
                gt_poly_denorm[:, 0] *= crop_size
                gt_poly_denorm[:, 1] *= crop_size
                gt_poly_denorm[:, 2:] *= crop_size
            else:
                gt_poly_denorm = np.zeros((0, raycast_dim), dtype=np.float32)

            pred = batch_results[si]
            pred_poly = pred['pred_polys']
            pred_cls = pred['pred_cls']
            pred_conf = pred['pred_conf']

            n_gt = gt_poly_denorm.shape[0]
            n_pred = pred_poly.shape[0]

            if args.mode == 'overlay':
                if n_gt > 0:
                    vis_img = draw_polygons(vis_img, gt_poly_denorm, CLASS_COLORS_GT, thickness=1)
                if n_pred > 0:
                    vis_img = draw_polygons(vis_img, pred_poly, CLASS_COLORS_PRED, thickness=2)
                out_path = out_dir / f'{global_idx:04d}_overlay_gt{n_gt}_pred{n_pred}.png'
                cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

            elif args.mode == 'side_by_side':
                gt_img = vis_img.copy()
                pred_img = vis_img.copy()
                if n_gt > 0:
                    gt_img = draw_polygons(gt_img, gt_poly_denorm, CLASS_COLORS_GT, thickness=1)
                if n_pred > 0:
                    pred_img = draw_polygons(pred_img, pred_poly, CLASS_COLORS_PRED, thickness=2)
                combined = np.concatenate([gt_img, pred_img], axis=1)
                label_h = 20
                header = np.zeros((label_h, combined.shape[1], 3), dtype=np.uint8)
                cv2.putText(header, f'GT ({n_gt})', (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(
                    header,
                    f'Pred ({n_pred})',
                    (combined.shape[1] // 2 + 10, 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    1,
                )
                combined = np.concatenate([header, combined], axis=0)
                out_path = out_dir / f'{global_idx:04d}_side_gt{n_gt}_pred{n_pred}.png'
                cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            elif args.mode == 'pred_only':
                if n_pred > 0:
                    vis_img = draw_polygons(vis_img, pred_poly, CLASS_COLORS_PRED, thickness=2)
                out_path = out_dir / f'{global_idx:04d}_pred{n_pred}.png'
                cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

            elif args.mode == 'gt_only':
                if n_gt > 0:
                    vis_img = draw_polygons(vis_img, gt_poly_denorm, CLASS_COLORS_GT, thickness=1)
                out_path = out_dir / f'{global_idx:04d}_gt{n_gt}.png'
                cv2.imwrite(str(out_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

            saved += 1
            global_idx += 1

            if saved % 10 == 0:
                print(f'  Saved {saved}/{n_total}')

    print(f'\nDone. {saved} images saved to {out_dir}')


if __name__ == '__main__':
    main()
