"""Synthetic video dataset for the toy JEPA: short clips of shapes bouncing in a box.

Each clip is 12 grayscale frames of 32 x 32 pixels with two shapes (discs or squares) that move in
straight lines and bounce off the walls. The world has dynamics a model can learn (position,
velocity, bounces) and no visual noise. train_jepa.py trains on these clips.

Preview:  python bouncing_shapes.py          # writes preview.png next to this file
Import:   from bouncing_shapes import BouncingShapes

Code generated with Claude for the author. The preview's main(), the type hints and the named
constants were added by Claude Code on Nov 15, 2026; the clips it generates are unchanged.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PREVIEW = Path(__file__).parent / "preview.png"
PREVIEW_CLIPS = 4

# The frame whose positions are returned as the probe's ground truth: the last context frame.
STATE_FRAME = 3
# Shapes start at least this many pixels away from every wall.
START_MARGIN = 6
# Largest speed along one axis, in pixels per frame.
MAX_SPEED = 2.5
# Radii run from the smallest shape to the largest, each moved by up to RADIUS_JITTER.
MIN_RADIUS, MAX_RADIUS, RADIUS_JITTER = 2.6, 4.4, 0.3
# Each clip has its own random generator, seeded with seed * SEED_STRIDE + index.
SEED_STRIDE = 1_000_003
DISC = 0  # the other kind of shape is a square


class BouncingShapes(Dataset):
    """Clips of shape (T, 1, size, size), values in [0, 1].

    Each clip: `n_shapes` squares/discs with random position, velocity, and
    size, bouncing elastically off the walls for T frames.

    With return_state=True, also returns the shapes' (x, y) positions at
    frame 3 (the last context frame) as a (n_shapes, 2) float tensor —
    ground truth for the linear probe in train_jepa.py.
    """

    def __init__(  # noqa: PLR0913  (six settings of one dataset, all with defaults)
        self,
        num_clips: int = 10_000,
        T: int = 12,
        size: int = 32,
        n_shapes: int = 2,
        seed: int = 0,
        return_state: bool = False,
    ) -> None:
        self.num_clips, self.T, self.size, self.n_shapes = num_clips, T, size, n_shapes
        self.seed = seed
        self.return_state = return_state

    def __len__(self) -> int:
        return self.num_clips

    def _render(self, frame: np.ndarray, cx: float, cy: float, r: float, kind: int) -> None:
        """Draw one shape into `frame`, in place."""
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        if kind == DISC:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        else:  # square
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        frame[mask] = 1.0

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Generate clip `idx`. The same index always gives the same clip."""
        rng = np.random.default_rng(self.seed * SEED_STRIDE + idx)
        s = self.size
        pos = rng.uniform(START_MARGIN, s - START_MARGIN, size=(self.n_shapes, 2))
        vel = rng.uniform(-MAX_SPEED, MAX_SPEED, size=(self.n_shapes, 2))
        # Sizes are separated by index (shape 0 smallest ... last one biggest)
        # so each shape is VISUALLY identifiable — the probe asks "where is
        # the BIG one?", which is answerable from pixels. With identical size
        # ranges the probe target ("object 0") would be visually ill-posed.
        radius = np.linspace(MIN_RADIUS, MAX_RADIUS, self.n_shapes) + rng.uniform(
            -RADIUS_JITTER, RADIUS_JITTER, size=self.n_shapes
        )
        kind = rng.integers(0, 2, size=self.n_shapes)

        clip = np.zeros((self.T, 1, s, s), dtype=np.float32)
        state = None
        for t in range(self.T):
            for i in range(self.n_shapes):
                self._render(clip[t, 0], pos[i, 0], pos[i, 1], radius[i], kind[i])
            if t == STATE_FRAME:
                state = pos.astype(np.float32).copy()
            pos += vel
            for i in range(self.n_shapes):  # elastic bounce off walls
                for d in range(2):
                    if pos[i, d] < radius[i] or pos[i, d] > s - radius[i]:
                        vel[i, d] *= -1
                        pos[i, d] = np.clip(pos[i, d], radius[i], s - radius[i])
        if self.return_state:
            return torch.from_numpy(clip), torch.from_numpy(state)
        return torch.from_numpy(clip)


def main() -> None:
    """Write preview.png: a few clips, one row each, to check the motion by eye."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BouncingShapes(num_clips=PREVIEW_CLIPS)
    fig, axes = plt.subplots(PREVIEW_CLIPS, ds.T, figsize=(14, 5))
    for row in range(PREVIEW_CLIPS):
        clip = ds[row]
        for t in range(ds.T):
            axes[row, t].imshow(clip[t, 0], cmap="gray")
            axes[row, t].axis("off")
    fig.suptitle(f"BouncingShapes: {PREVIEW_CLIPS} clips x {ds.T} frames")
    fig.savefig(PREVIEW, dpi=100, bbox_inches="tight")
    print(f"wrote {PREVIEW} — eyeball that the physics looks right")


if __name__ == "__main__":
    main()
