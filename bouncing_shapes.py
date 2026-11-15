"""Synthetic video dataset for the toy JEPA: short clips of shapes bouncing in a box.

Each clip is 12 grayscale frames of 32 x 32 pixels with two shapes (discs or squares) that move in
straight lines and bounce off the walls. The world has dynamics a model can learn (position,
velocity, bounces) and no visual noise. train_jepa.py trains on these clips.

Preview:  python bouncing_shapes.py          # writes preview.png next to this file
Import:   from bouncing_shapes import BouncingShapes
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BouncingShapes(Dataset):
    """Clips of shape (T, 1, size, size), values in [0, 1].

    Each clip: `n_shapes` squares/discs with random position, velocity, and
    size, bouncing elastically off the walls for T frames.

    With return_state=True, also returns the shapes' (x, y) positions at
    frame 3 (the last context frame) as a (n_shapes, 2) float tensor —
    ground truth for the linear probe in train_jepa.py.
    """

    def __init__(self, num_clips=10_000, T=12, size=32, n_shapes=2, seed=0, return_state=False):
        self.num_clips, self.T, self.size, self.n_shapes = num_clips, T, size, n_shapes
        self.seed = seed
        self.return_state = return_state

    def __len__(self):
        return self.num_clips

    def _render(self, frame, cx, cy, r, kind):
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s]
        if kind == 0:  # disc
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        else:  # square
            mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        frame[mask] = 1.0

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        s = self.size
        pos = rng.uniform(6, s - 6, size=(self.n_shapes, 2))
        vel = rng.uniform(-2.5, 2.5, size=(self.n_shapes, 2))
        # Sizes are separated by index (shape 0 smallest ... last one biggest)
        # so each shape is VISUALLY identifiable — the probe asks "where is
        # the BIG one?", which is answerable from pixels. With identical size
        # ranges the probe target ("object 0") would be visually ill-posed.
        radius = np.linspace(2.6, 4.4, self.n_shapes) + rng.uniform(
            -0.3, 0.3, size=self.n_shapes
        )
        kind = rng.integers(0, 2, size=self.n_shapes)

        clip = np.zeros((self.T, 1, s, s), dtype=np.float32)
        state = None
        for t in range(self.T):
            for i in range(self.n_shapes):
                self._render(clip[t, 0], pos[i, 0], pos[i, 1], radius[i], kind[i])
            if t == 3:
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


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BouncingShapes(num_clips=4)
    fig, axes = plt.subplots(4, 12, figsize=(14, 5))
    for row in range(4):
        clip = ds[row]
        for t in range(12):
            axes[row, t].imshow(clip[t, 0], cmap="gray")
            axes[row, t].axis("off")
    fig.suptitle("BouncingShapes: 4 clips x 12 frames")
    preview = Path(__file__).parent / "preview.png"
    fig.savefig(preview, dpi=100, bbox_inches="tight")
    print(f"wrote {preview} — eyeball that the physics looks right")
