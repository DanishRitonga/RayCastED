from copy import deepcopy

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from raycasted.model.builder import raycasted_parse_model


class HybridDetectionModel(DetectionModel):
    def __init__(self, cfg='yolo26-hybrid-p234.yaml', ch=3, nc=None, verbose=True):
        from ultralytics.nn.tasks import yaml_model_load
        from ultralytics.utils import LOGGER

        yaml_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        yaml_dict['channels'] = ch
        if nc and nc != yaml_dict['nc']:
            LOGGER.info(f'Overriding model.yaml nc={yaml_dict["nc"]} with nc={nc}')
            yaml_dict['nc'] = nc

        super(DetectionModel, self).__init__()

        self.yaml = yaml_dict
        self.model, self.save = raycasted_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        self.nc = self.yaml['nc']

        self.stride = torch.tensor([4.0, 8.0, 16.0])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def init_criterion(self):
        from raycasted.model.hybrid_loss import HybridHungarianMatcher, HybridSetCriterion

        matcher = HybridHungarianMatcher(
            cost_class=1.0,
            cost_centroid=1.0,
            cost_radial=1.0,
            cost_inner=1.0,
        )
        return HybridSetCriterion(
            nc=self.nc,
            matcher=matcher,
        )

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.predict(batch['img'])

        bs = batch['img'].shape[0]
        batch_idx = batch.get('batch_idx', torch.zeros_like(batch['cls']))
        targets = []
        for i in range(bs):
            mask_i = batch_idx == i
            tgt = {'labels': batch['cls'][mask_i].long()}
            if 'bboxes' in batch:
                tgt['boxes'] = batch['bboxes'][mask_i][:, :2]
            targets.append(tgt)

        loss_dict = self.criterion(preds, targets)
        return loss_dict['total'], torch.as_tensor(
            [loss_dict.get(k, torch.tensor(0.0)).detach() for k in ['loss_ce', 'loss_centroid', 'loss_radial']],
            device=batch['img'].device,
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        y = []
        prev_out = x
        for m in self.model[:-1]:
            if m.f != -1:
                x_in = y[m.f] if isinstance(m.f, int) else [prev_out if j == -1 else y[j] for j in m.f]
            else:
                x_in = prev_out
            out = m(x_in)
            y.append(out if m.i in self.save else None)
            prev_out = out

        head = self.model[-1]
        features = [feat for feat in (y[j] for j in head.f) if feat is not None]
        return head(features)
