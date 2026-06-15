"""RayCastED — Visual Inference on PanNuke test tiles.

Runs inference and saves class-colored GT + prediction polygon outlines
as separate PNG files (no labels, no centroids, clean outlines only).

Usage:
    uv run python main/visual_inference.py \
        --weights output/v1.1/weights/best.pt \
        --data-dir output/pannuke/transformed/test \
        --out-dir output/vis --n-images 10 --conf 0.5
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

CLASS_COLORS = [
    (0, 255, 0),      # Green  — Neoplastic
    (0, 255, 255),    # Cyan   — Inflammatory
    (255, 255, 0),    # Yellow — Connective
    (255, 0, 255),    # Magenta — Necrosis
    (0, 165, 255),    # Orange — Epithelial
]


def load_model(weights_path: str, device: torch.device):
    register_raycast_head()
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) else ckpt
    return model.float().to(device).eval()


def run_inference(model, images, device, conf_threshold, raycast_dim):
    with torch.no_grad():
        decoded = model(images)[0]
    results = []
    for si in range(decoded.shape[0]):
        det = decoded[si].cpu().numpy()
        if det.ndim == 2 and det.shape[1] == raycast_dim + 2:
            det = det[det[:, raycast_dim] > conf_threshold]
        if det.shape[0]:
            pred_poly = det[:, :raycast_dim]
            pred_cls = det[:, raycast_dim + 1].astype(int)
        else:
            pred_poly = np.zeros((0, raycast_dim), dtype=np.float32)
            pred_cls = np.array([], dtype=int)
        results.append({'pred_polys': pred_poly, 'pred_cls': pred_cls})
    return results


def draw_polygons(img, polys, cls_labels, thickness=1):
    if polys.shape[0] == 0:
        return img
    vertices = decode_to_vertices(polys[:, 2:], polys[:, 0], polys[:, 1])
    for i in range(polys.shape[0]):
        cls_id = int(cls_labels[i]) if i < len(cls_labels) else 0
        color = CLASS_COLORS[min(cls_id, len(CLASS_COLORS) - 1)]
        pts = vertices[i].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    return img


def _simple_collate(batch):
    import torch as _torch
    images = _torch.stack([item[0] for item in batch])
    labels_list = [item[1] for item in batch]
    target_list = []
    for bi, labels in enumerate(labels_list):
        if labels.shape[0] == 0: continue
        bc = np.full((labels.shape[0], 1), bi, dtype=np.float32)
        target_list.append(np.concatenate([bc, labels], axis=1))
    if target_list:
        targets = _torch.from_numpy(np.concatenate(target_list, 0))
    else:
        w = next((lbl.shape[1] for lbl in labels_list if lbl.ndim==2 and lbl.shape[1]>0), 35)
        targets = _torch.zeros((0, 1+w), dtype=_torch.float32)
    return dict(img=images, batch_idx=targets[:,0], cls=targets[:,1], bboxes=targets[:,2:])


def main():
    parser = argparse.ArgumentParser(description='RayCastED Visual Inference')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.20)
    parser.add_argument('--n-images', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        parser.error(f'Not found: {data_dir}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.device}' if args.device.isdigit() else args.device)
    model = load_model(args.weights, device)
    training_args = getattr(model, 'training_args', {})
    crop_size = training_args.get('crop_size', 640)
    n_rays = training_args.get('n_rays', 32)
    raycast_dim = 2 + n_rays
    configure_rays(n_rays)

    dataset = RayCastTileDataset(data_dir=str(data_dir), crop_size=crop_size, augment=False)
    n_total = min(args.n_images, len(dataset))
    print(f'Visualising {n_total}/{len(dataset)} images')

    dl = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, collate_fn=_simple_collate)

    saved = 0; idx = 0
    for batch in dl:
        if saved >= n_total: break
        images = batch['img'].to(device)
        batch_results = run_inference(model, images, device, args.conf, raycast_dim)

        for si in range(images.shape[0]):
            if saved >= n_total: break

            tile_data = np.load(dataset.tile_paths[idx])
            raw_img = tile_data['image']
            if raw_img.ndim == 2:
                raw_img = np.stack([raw_img]*3, axis=-1)

            mask = batch['batch_idx'] == si
            gt_cls = batch['cls'][mask].numpy().flatten().astype(int)
            gt_poly = batch['bboxes'][mask].numpy()
            if gt_poly.shape[0]:
                gt_poly = gt_poly.copy()
                gt_poly[:, 0] *= crop_size; gt_poly[:, 1] *= crop_size; gt_poly[:, 2:] *= crop_size
            else:
                gt_poly = np.zeros((0, raycast_dim), dtype=np.float32)

            pred = batch_results[si]

            # GT
            gt_img = draw_polygons(raw_img.copy(), gt_poly, gt_cls, thickness=1)
            cv2.imwrite(str(out_dir / f'{idx:04d}_gt.png'), cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))

            # Pred
            pred_img = draw_polygons(raw_img.copy(), pred['pred_polys'], pred['pred_cls'], thickness=1)
            cv2.imwrite(str(out_dir / f'{idx:04d}_pred.png'), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

            saved += 1; idx += 1

    print(f'Done. {saved} images → {out_dir}')


if __name__ == '__main__':
    main()
