"""RayCastED — RayCast Detection Loss (Phase 4 stub).

Provides a minimal RayCastDetectionLoss that overrides self.no and use_dfl.
Full 5-term loss implementation deferred to Phase 6.
"""

from ultralytics.utils.loss import v8DetectionLoss


class RayCastDetectionLoss(v8DetectionLoss):
    """Phase 4 stub — overrides self.no and disables DFL.

    Full loss computation (L_PolarIoU, L_L1, L_cls, L_smooth, L_dfl=0)
    and RayCastAssigner integration deferred to Phase 6.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize polygon detection loss with correct output dimension.

        Overrides:
            self.no = nc + 34 (parent sets nc + reg_max*4 — wrong for polygon head)
            self.use_dfl = False (DFL not applicable to polygon regression)
        """
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        m = model.model[-1]
        self.no = m.nc + 34  # BUG-02 fix
        self.use_dfl = False
