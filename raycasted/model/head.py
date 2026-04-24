"""Backward-compatible import path for torch.load checkpoint unpickling.

The head module was moved from raycasted.model.head to raycasted.model.blocks.head.
Old checkpoints reference the former path. This stub allows torch.load to resolve it.
"""

from raycasted.model.blocks.head import *  # noqa: F401,F403
