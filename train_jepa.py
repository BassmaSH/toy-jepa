"""Train the toy JEPA and print one row per epoch.

  python train_jepa.py --no-ema --no-var-reg     # 1: neither
  python train_jepa.py --no-var-reg              # 2: EMA only
  python train_jepa.py                           # 3: EMA + variance term
  python train_jepa.py --no-var-reg --normalize  # 4: EMA + normalize

Columns:
  enc_std   -> standard deviation of the online encoder's embeddings across a
               batch. Near 0 = collapse.
  probe_err -> pixels of error when a ridge probe on the frozen embeddings
               reads the largest shape's (x, y). On this dataset: always
               predicting the mean gives about 5.5 px, an untrained encoder ~3.0 px
               (measured by baselines.py; the closing lines of a run print 5.62,
               kept so that the output stays identical to the logs).

Who wrote what: the argument parsing, the data loading and the docstrings started as an exercise
skeleton generated with Claude. The author wrote the training step, the probe and the --normalize
variant. The split into functions, type hints and named constants were added by Claude Code on
Nov 15, 2026, without changing what a run prints.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bouncing_shapes import BouncingShapes
from jepa import ToyJEPA, embedding_std, jepa_loss, variance_regularizer

SEED = 42
NUM_CLIPS = 8_000
NUM_WORKERS = 4
CONTEXT_FRAMES = slice(0, 4)
TARGET_FRAMES = slice(8, 12)
# The probe reads at most PROBE_TRAIN_BATCHES + PROBE_HELD_OUT_BATCHES batches and splits the rows
# it collected in that ratio. See probe_eval for what this gives when an epoch is shorter.
PROBE_TRAIN_BATCHES = 40
PROBE_HELD_OUT_BATCHES = 10
PROBE_RIDGE = 1e-3


def get_device() -> torch.device:
    """Apple's GPU (MPS) when it is available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_clip(clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 12, 1, 32, 32) -> (context frames 0..3, target frames 8..11)."""
    return (clip[:, CONTEXT_FRAMES], clip[:, TARGET_FRAMES])


def probe_split(n_rows: int, train_batches: int = PROBE_TRAIN_BATCHES) -> int:
    """Number of rows that fit the probe; the remaining rows are held out.

    The rows are split in the ratio train_batches : PROBE_HELD_OUT_BATCHES, whatever the number of
    batches actually read.
    """
    return (n_rows * train_batches) // (train_batches + PROBE_HELD_OUT_BATCHES)


def ridge_probe_error(Z: torch.Tensor, Y: torch.Tensor, n_train: int) -> float:
    """Fit a ridge probe on the first n_train rows and score it on the rest.

    Args:
        Z: embeddings, one row per clip.
        Y: the position to read from each embedding, one (x, y) row per clip.
        n_train: number of leading rows that fit the probe.

    Returns:
        The mean absolute error on the held-out rows, in the units of Y.
    """
    Ztr, Ytr, Zte, Yte = Z[:n_train], Y[:n_train], Z[n_train:], Y[n_train:]
    Ztr = torch.cat([Ztr, torch.ones(len(Ztr), 1)], 1)
    Zte = torch.cat([Zte, torch.ones(len(Zte), 1)], 1)
    ridge = PROBE_RIDGE * torch.eye(Ztr.shape[1])
    W = torch.linalg.solve(Ztr.T @ Ztr + ridge, Ztr.T @ Ytr)
    return (Zte @ W - Yte).abs().mean().item()


@torch.no_grad()
def probe_eval(
    model: ToyJEPA,
    loader: DataLoader,
    device: torch.device,
    train_batches: int = PROBE_TRAIN_BATCHES,
) -> float:
    """Ridge-regression probe, closed form: frozen embeddings -> the largest shape's (x, y).

    Returns the mean absolute error in pixels.

    At most train_batches + 10 batches are read from the loader, and the rows
    collected are split in the ratio train_batches : 10 (4 : 1 by default):
    the first part fits the probe, the rest is held out. With the settings in
    main() an epoch has only 31 batches (8,000 clips, batch size 256, last
    partial batch dropped), so the loader ends before the limit of 50: 7,936
    rows are collected, 6,348 fit the probe and 1,588 are held out.
    """
    Z, Y = [], []
    for i, (clip, state) in enumerate(loader):
        if i >= train_batches + PROBE_HELD_OUT_BATCHES:
            break
        ctx, _ = split_clip(clip.to(device))
        z = model.encoder(ctx)
        Z.append(z.cpu())
        Y.append(state[:, -1, :2])
    Z, Y = torch.cat(Z).double(), torch.cat(Y).double()
    return ridge_probe_error(Z, Y, probe_split(len(Y), train_batches))


def train_epoch(
    model: ToyJEPA,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one pass over the loader.

    Returns:
        The mean loss over the batches, and enc_std measured on the last batch of the epoch.
    """
    loss_sum, n = 0.0, 0
    for clip, _ in loader:
        ctx, tgt = split_clip(clip.to(device))
        pred, z_tgt = model(ctx, tgt)
        if args.normalize:
            pred, z_tgt = F.normalize(pred, dim=-1), F.normalize(z_tgt, dim=-1)
        loss = jepa_loss(pred, z_tgt)
        if not args.no_var_reg:
            loss += args.var_weight * variance_regularizer(model.encoder(ctx))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not args.no_ema:
            model.ema_update()
        loss_sum += loss.item()
        n += 1

    # collapse metric on the online encoder, last batch of the epoch
    with torch.no_grad():
        enc_std = embedding_std(model.encoder(ctx))
    return loss_sum / n, enc_std


def parse_args() -> argparse.Namespace:
    """Read the command line."""
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-var-reg", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--var-weight", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    """Train one variant and print a row per epoch."""
    args = parse_args()

    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device} | ema={not args.no_ema} var_reg={not args.no_var_reg}")

    ds = BouncingShapes(num_clips=NUM_CLIPS, return_state=True)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
    )
    model = ToyJEPA(dim=args.dim, use_ema=not args.no_ema).to(device)
    # Only parameters that require a gradient: the target encoder is frozen and
    # moves by EMA, not by the optimizer.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"{'epoch':>5} {'loss':>8} {'enc_std':>8} {'probe_err':>9}  (px)")
    for epoch in range(1, args.epochs + 1):
        mean_loss, enc_std = train_epoch(model, loader, opt, args, device)
        err = probe_eval(model, loader, device)
        print(f"{epoch:>5} {mean_loss:>8.4f} {enc_std:>8.3f} {err:>9.2f}  (px)")

    print(
        "\ncollapse: enc_std -> ~0, loss -> 0.0000, "
        "probe_err WORSE than an untrained encoder (~3.0px)"
    )
    print("healthy:  enc_std O(1) AND probe_err -> ~1.8px (mean-predictor floor is 5.62px)")


if __name__ == "__main__":
    main()
