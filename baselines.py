"""Measure the two baselines the README quotes next to its table, with the probe of train_jepa.py.

    python baselines.py   # about 2 minutes on an Apple M4 Pro; prints one line per measurement

Mean predictor: the error of always answering the mean position of the largest shape. It is
measured over all clips, and through the probe's own split (fit on 6,348 rows, score on the 1,588
held out) for several shuffles, because a training run shuffles the clips before the probe reads
them.

Untrained encoder: the probe's error on a freshly initialised encoder, before any training step,
for the seed the published runs use and four others.

Code written by Claude Code on Nov 16, 2026, for the author.
"""

import torch
from torch.utils.data import DataLoader

from bouncing_shapes import BouncingShapes
from jepa import ToyJEPA
from train_jepa import NUM_CLIPS, NUM_WORKERS, SEED, get_device, probe_eval, probe_split

BATCH_SIZE = 256  # the default of train_jepa.py, used by every published run
DIM = 64  # likewise
UNTRAINED_SEEDS = (SEED, 0, 1, 2, 3)
SHUFFLES = 20  # number of shuffles the mean predictor is scored on


def largest_shape_positions(ds: BouncingShapes) -> torch.Tensor:
    """The (x, y) the probe is asked to read, one row per clip: the last shape's position."""
    return torch.stack([ds[i][1][-1] for i in range(len(ds))]).double()


def mean_predictor_error(y_fit: torch.Tensor, y_test: torch.Tensor) -> float:
    """Mean absolute error on y_test of always answering the mean of y_fit."""
    return (y_test - y_fit.mean(0)).abs().mean().item()


def mean_predictor_split_errors(
    y: torch.Tensor, rows: int, shuffles: int = SHUFFLES
) -> list[float]:
    """Score the mean predictor the way a run scores the probe, once per shuffle.

    Args:
        y: positions, one row per clip.
        rows: number of clips an epoch reads (the last partial batch is dropped).
        shuffles: number of random orders; shuffle i uses a generator seeded with i.

    Returns:
        One error per shuffle: the mean is taken on the rows that fit the probe and scored on the
        held-out rows.
    """
    n_fit = probe_split(rows)
    errors = []
    for i in range(shuffles):
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(i))[:rows]
        errors.append(mean_predictor_error(y[order[:n_fit]], y[order[n_fit:]]))
    return errors


def untrained_probe_error(ds: BouncingShapes, seed: int, device: torch.device) -> float:
    """The probe's error on an encoder that has not been trained, set up as in train_jepa.main()."""
    torch.manual_seed(seed)
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
    )
    model = ToyJEPA(dim=DIM).to(device)
    return probe_eval(model, loader, device)


def main() -> None:
    """Print the mean predictor's error, then the untrained encoder's, one line each."""
    ds = BouncingShapes(num_clips=NUM_CLIPS, return_state=True)
    y = largest_shape_positions(ds)
    print(f"mean predictor, all {NUM_CLIPS} clips: {mean_predictor_error(y, y):.2f} px")
    rows = (NUM_CLIPS // BATCH_SIZE) * BATCH_SIZE
    errors = mean_predictor_split_errors(y, rows)
    print(
        f"mean predictor, through the probe's split, {SHUFFLES} shuffles: "
        f"{min(errors):.2f} to {max(errors):.2f} px"
    )
    device = get_device()
    for seed in UNTRAINED_SEEDS:
        err = untrained_probe_error(ds, seed, device)
        print(f"untrained encoder, seed {seed}: {err:.2f} px", flush=True)


if __name__ == "__main__":
    main()
