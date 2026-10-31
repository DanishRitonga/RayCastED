"""Check vectorized alignment against the original per-owner definition."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from raycasted.model.nulite_detr_model import NuLiteRayCastDETRModel


def _reference(model, queries, features, masks, polygons, scores, targets, groups):
    queries = F.normalize(model.instance_alignment_projection(queries), dim=-1)
    masks = F.interpolate(masks[:, None].float(), size=features.shape[-2:], mode='nearest')[:, 0].long()
    terms, offset = [], 0
    for b, ((qs, gs), count) in enumerate(zip(model.criterion.matcher(), groups)):
        centers = targets['bboxes'][offset : offset + count, :2]
        prototypes = {
            g: F.normalize(features[b, :, masks[b] == g + 1].mean(-1), dim=0)
            for g in range(count)
            if (masks[b] == g + 1).any()
        }
        distance = torch.cdist(centers, centers)
        for q, global_g in zip(qs.tolist(), gs.tolist()):
            g = global_g - offset
            if g not in prototypes:
                continue
            neighbors = [j for j in prototypes if 0 < distance[g, j] <= model.instance_alignment_radius_px / 64]
            neighbors.sort(key=lambda j: float(distance[g, j]))
            neighbors = neighbors[: model.instance_alignment_max_negatives]
            if neighbors:
                p = torch.stack([prototypes[j] for j in [g] + neighbors])
                logits = queries[b, q] @ p.T / model.instance_alignment_temperature
                terms.append(F.cross_entropy(logits[None], torch.zeros(1, dtype=torch.long)))
        offset += count
    return torch.stack(terms).mean() if terms else queries.sum() * 0


@pytest.mark.parametrize('radius', [0.0, 12.0, 64.0])
@pytest.mark.parametrize('cap', [1, 8])
def test_alignment_matches_reference_value_and_gradients(radius, cap):
    """Preserve ties, missing instances, empty images, global IDs and all gradients."""
    torch.manual_seed(17)
    model = object.__new__(NuLiteRayCastDETRModel)
    torch.nn.Module.__init__(model)
    model.instance_alignment_projection = torch.nn.Linear(6, 4).double()
    model.instance_alignment_temperature = 0.1
    model.instance_alignment_radius_px = radius
    model.instance_alignment_max_negatives = cap
    groups = [4, 0, 2]
    owners = [
        (torch.tensor([3, 1, 2, 0]), torch.tensor([0, 1, 2, 3])),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        (torch.tensor([1, 3]), torch.tensor([4, 5])),
    ]
    model.criterion = SimpleNamespace(matcher=lambda *args: owners)
    queries = torch.randn(3, 4, 6, dtype=torch.double, requires_grad=True)
    features = torch.randn(3, 4, 8, 8, dtype=torch.double, requires_grad=True)
    masks = torch.zeros(3, 16, 16, dtype=torch.long)
    masks[0, :8, :8] = 1
    masks[0, :8, 8:] = 2
    masks[0, 8:, :8] = 3  # instance 4 has no remaining pixels
    masks[2, :8] = 1
    masks[2, 8:] = 2
    gt = torch.zeros(6, 10, dtype=torch.double)
    gt[:, :2] = torch.tensor([[0.5, 0.5], [0.375, 0.5], [0.625, 0.5], [0.9, 0.9], [0.2, 0.2], [0.3, 0.2]])
    targets = {'bboxes': gt, 'cls': torch.zeros(6, dtype=torch.long), 'imgsz': 64}
    polygons, scores = torch.zeros(3, 4, 10), torch.zeros(3, 4, 3)
    args = (queries, features, masks, polygons, scores, targets, groups)
    expected = _reference(model, *args)
    actual = model._instance_alignment_loss(*args)
    torch.testing.assert_close(actual, expected)
    params = (queries, features, *model.instance_alignment_projection.parameters())
    old_grads = torch.autograd.grad(expected, params, allow_unused=True)
    new_grads = torch.autograd.grad(actual, params, allow_unused=True)
    for param, old, new in zip(params, old_grads, new_grads):
        torch.testing.assert_close(
            torch.zeros_like(param) if old is None else old, torch.zeros_like(param) if new is None else new
        )
