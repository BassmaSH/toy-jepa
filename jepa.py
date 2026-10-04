"""A toy JEPA on BouncingShapes.

  context = frames 0..3   -> online encoder -> z_ctx (B, D)
  target  = frames 8..11  -> target encoder -> z_tgt (B, D), no gradient
  predictor(z_ctx) is trained to match z_tgt.
"""

import copy

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Small CNN video encoder: (B, T=4, 1, 32, 32) -> (B, D).
    The 4 frames are stacked on the channel dimension."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, dim),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.net(clip.squeeze(2))  # (B,T,1,H,W) -> (B,T,H,W)


class Predictor(nn.Module):
    """MLP that predicts the target embedding from the context embedding."""

    def __init__(self, dim: int = 64, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z)


class ToyJEPA(nn.Module):
    def __init__(self, dim: int = 64, ema_momentum: float = 0.99, use_ema: bool = True):
        super().__init__()
        self.encoder = Encoder(dim)
        self.predictor = Predictor(dim)
        self.ema_momentum = ema_momentum
        self.use_ema = use_ema
        # The target encoder starts as a frozen copy of the online encoder.
        # With use_ema=False it is not used: targets are encoded by the online
        # encoder and detached.
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)

    @torch.no_grad()
    def ema_update(self):
        """tgt = m * tgt + (1 - m) * online, with m = self.ema_momentum."""
        m = self.ema_momentum
        for i,j in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            i.mul_(m).add_(j,alpha=1-m)
        for i,j in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            i.copy_(j)

    def forward(self, context: torch.Tensor, target: torch.Tensor):
        """Return (pred, z_tgt). z_tgt never carries a gradient."""
        pred = self.predictor(self.encoder(context))
        if self.use_ema:
            with torch.no_grad():
              z_tgt = self.target_encoder(target)
        else:
            z_tgt = self.encoder(target).detach() 
        return (pred,z_tgt)


def jepa_loss(pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Smooth L1 between the prediction and the target embedding."""
    l = nn.SmoothL1Loss()
    return l(pred,z_tgt)


def variance_regularizer(z: torch.Tensor, floor: float = 1.0) -> torch.Tensor:
    """Variance term (as in VICReg) on the online embeddings: penalise each
    dimension whose standard deviation across the batch is below `floor`."""
    r = nn.ReLU()
    return r(floor - torch.sqrt(z.var(dim=0)+1e-6)).mean()


@torch.no_grad()
def embedding_std(z: torch.Tensor) -> float:
    """Collapse metric: mean over dimensions of the standard deviation across
    the batch. Near 0 means every clip gets the same embedding."""
    return z.std(dim=0).mean().item()
