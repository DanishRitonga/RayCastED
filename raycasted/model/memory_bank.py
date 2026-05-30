"""Feature Bank with EMA Prototypes for classification feature refinement.

Implements the feature bank mechanism described in:

    @article{chen2026featurebank,
      title   = {Feature Bank with EMA Prototypes for Classification Feature Refinement},
      author  = {Chen, Lianghui and Li, Shuyang and Liu, Wenwen and Zheng, Haijiang and Zheng, Shun and Yang, Yifei},
      journal = {Journal of Computer Science and Technology},
      year    = {2026}
    }

The feature bank stores per-class exponential moving average (EMA) prototypes
extracted from the classification head's intermediate features (c3 layer).
During training, foreground anchor features are compared against all class
prototypes to compute a contrastive loss that pushes each fg feature toward its
assigned class prototype and away from others.

The prototypes are initialised on the first batch and updated with EMA after each
forward pass.  All feature vectors are **detached** before entering the bank, so
the bank is a pure momentum-based statistics store — no gradient flows through it.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureBank(nn.Module):
    """EMA prototype bank for intermediate cls head features (c3).

    Maintains per-class running-mean prototypes updated via exponential moving
    average.  Provides a contrastive loss that pulls foreground features toward
    their assigned class prototype and pushes them away from others.

    Args:
        num_classes: Number of foreground classes (nc).
        feat_dim: Dimensionality of the c3 intermediate features.
        momentum: EMA momentum for prototype updates (0–1).
            Higher = slower updates, more stable prototypes.
        temperature: InfoNCE temperature for the contrastive loss.
            Lower = sharper distribution, stronger pull toward target.
        weight: Scalar multiplier for the contrastive loss contribution.
        device: Device to allocate prototype tensors on.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        momentum: float = 0.9,
        temperature: float = 0.07,
        weight: float = 0.5,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.temperature = temperature
        self.weight = weight
        self._initialised = False
        self._num_updates = 0

        # Prototypes: [num_classes, feat_dim], not registered as buffer because
        # they are plain float state not tied to model architecture.
        # Will be created lazily on first forward pass (need device info).
        self.prototypes: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Update prototypes with EMA from current-batch foreground features.

        Args:
            features: [K, feat_dim] — detached foreground feature vectors.
            labels: [K] — integer class labels for each foreground anchor.
        """
        if features.numel() == 0:
            return

        if self.prototypes is None:
            self._init_prototypes(features.device)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_feats = features[mask]  # [K_c, D]
                class_mean = class_feats.mean(dim=0)  # [D]
                self.prototypes[c].lerp_(class_mean, 1.0 - self.momentum)

        self._num_updates += 1

    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE-style contrastive loss against class prototypes.

        For each foreground feature f_i with assigned class k_i:
            L_i = -log( exp(sim(f_i, p_{k_i}) / τ)
                       / Σ_j exp(sim(f_i, p_j) / τ) )

        All prototypes are detached before comparison so no gradient flows
        through the bank.  Foreground features keep their gradients.

        Args:
            features: [K, feat_dim] — fg anchor features (with grad).
            labels: [K] — integer class labels (0..nc-1).

        Returns:
            Scalar contrastive loss tensor (requires_grad=True).
        """
        if self.prototypes is None or features.numel() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # [K, D] × [D, C] = [K, C] — cosine similarity scaled by temperature
        features_norm = F.normalize(features, dim=-1)
        prototypes_detached = F.normalize(self.prototypes.detach(), dim=-1)  # [C, D]
        logits = features_norm @ prototypes_detached.T  # [K, C]
        logits = logits / self.temperature

        # Standard cross-entropy: target is the assigned class for each feature
        loss = F.cross_entropy(logits, labels)
        return loss

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs) -> dict:
        """Serialise prototype state for checkpointing."""
        state = super().state_dict(*args, **kwargs)
        state['prototypes'] = self.prototypes
        state['_initialised'] = self._initialised
        state['_num_updates'] = self._num_updates
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Restore prototype state from checkpoint."""
        self.prototypes = state_dict.pop('prototypes', None)
        self._initialised = state_dict.pop('_initialised', False)
        self._num_updates = state_dict.pop('_num_updates', 0)
        super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_prototypes(self, device: torch.device) -> None:
        """Lazily initialise prototype tensor with zeros."""
        self.prototypes = torch.zeros(
            self.num_classes,
            self.feat_dim,
            device=device,
            dtype=torch.float32,
        )
        self._initialised = True
        logger.info(
            'FeatureBank: initialised %d prototypes (dim=%d, momentum=%.2f, τ=%.4f)',
            self.num_classes,
            self.feat_dim,
            self.momentum,
            self.temperature,
        )

    def __repr__(self) -> str:
        return (
            f'FeatureBank(nc={self.num_classes}, dim={self.feat_dim}, '
            f'momentum={self.momentum}, τ={self.temperature}, '
            f'weight={self.weight}, updates={self._num_updates})'
        )
