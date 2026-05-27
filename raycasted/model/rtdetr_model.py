"""RayCast RT-DETR Detection Model — integrates RayCastRTDETRDecoder with training pipeline.

Subclass of DetectionModel (via RTDETRDetectionModel) that:
- Uses raycasted_parse_model (custom builder) for model construction
- Uses RayCastRTDETRDecoder as the detection head
- Uses RayCastRTDETRDetectionLoss as the criterion
- Handles ray polygon batch format (raycast_dim instead of 4)
"""

from copy import deepcopy

import torch
from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel
from ultralytics.utils.torch_utils import initialize_weights

from raycasted.model.builder import raycasted_parse_model


def _yaml_has_custom_modules(cfg_dict):
    """Check if YAML uses any RayCastED custom modules."""
    custom_names = {
        'ResoConv',
        'ResoConvDS',
        'ResoConvDS_Hybrid',
        'ResoConvHybrid',
        'C3k2_LK',
        'DWT_LL',
        'DWT_HF',
        'HFResidual',
        'RayCastRTDETRDecoder',
        'RayCastDetect',
    }
    for section in ('backbone', 'head'):
        for layer in cfg_dict.get(section, []):
            if len(layer) > 2 and layer[2].__name__ if hasattr(layer[2], '__name__') else str(layer[2]) in custom_names:
                return True
    return False


class RayCastRTDETRDetectionModel(RTDETRDetectionModel):
    """RT-DETR Detection Model with ray polygon output.

    Uses raycasted_parse_model for YAMLs with custom modules (ResoConv, etc.)
    and standard ultralytics parse_model for stock RT-DETR YAMLs (HGNetV2, etc.).
    """

    def __init__(self, cfg='yolo26s-rtdetr-p234.yaml', ch=3, nc=None, verbose=True):
        from ultralytics.nn.tasks import yaml_model_load
        from ultralytics.utils import LOGGER

        yaml_dict = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        yaml_dict['channels'] = ch
        if nc and nc != yaml_dict['nc']:
            LOGGER.info(f'Overriding model.yaml nc={yaml_dict["nc"]} with nc={nc}')
            yaml_dict['nc'] = nc

        # BaseModel.__init__ only — skip DetectionModel.__init__
        super(DetectionModel, self).__init__()

        self.yaml = yaml_dict
        self.model, self.save = raycasted_parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}
        self.inplace = self.yaml.get('inplace', True)
        self.nc = self.yaml['nc']

        # Stride — RT-DETR decoder doesn't use strides like Detect heads
        m = self.model[-1]
        if hasattr(m, 'stride'):
            s = 256
            self.model.eval()
            m.training = True
            with torch.no_grad():
                _out = self.forward(torch.zeros(1, ch, s, s))
            m.stride = (
                torch.tensor([s / x.shape[-2] for x in _out])
                if isinstance(_out, (list, tuple))
                else torch.tensor([4.0, 8.0, 16.0])
            )
            self.stride = m.stride
            self.model.train()
        else:
            self.stride = torch.tensor([4.0, 8.0, 16.0])

        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def init_criterion(self):
        """Create RayCast RT-DETR detection loss."""
        from raycasted.model.rtdetr_loss import RayCastRTDETRDetectionLoss

        return RayCastRTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute training loss for ray polygon predictions."""
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        bs = img.shape[0]
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)

        dec_polygons, dec_scores, enc_polygons, enc_scores, dn_meta = preds if self.training else preds[1]

        if dn_meta is None:
            dn_polygons, dn_scores = None, None
        else:
            dn_polygons, dec_polygons = torch.split(dec_polygons, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_polygons = torch.cat([enc_polygons.unsqueeze(0), dec_polygons])
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_polygons, dec_scores), targets, dn_polygons=dn_polygons, dn_scores=dn_scores, dn_meta=dn_meta
        )

        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ['loss_class', 'loss_ray', 'loss_piou']],
            device=img.device,
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Forward pass through backbone/neck then RT-DETR decoder."""
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)
        return x
