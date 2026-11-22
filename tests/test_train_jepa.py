"""Tests for train_jepa.py: the probe behind the probe_err column, and one training step.

Code written by Claude Code on Nov 15, 2026, for the author.
"""

import argparse

import torch

from jepa import ToyJEPA
from train_jepa import (
    NUM_CLIPS,
    probe_eval,
    probe_split,
    ridge_probe_error,
    split_clip,
    train_epoch,
)

BATCH_SIZE = 256  # the default of --batch-size, used by every published run


def test_probe_split_of_a_published_run():
    # An epoch drops the last partial batch: 31 batches of 256 clips.
    rows = (NUM_CLIPS // BATCH_SIZE) * BATCH_SIZE
    assert rows == 7936
    n_train = probe_split(rows)
    assert (n_train, rows - n_train) == (6348, 1588)


def test_probe_split_when_all_50_batches_are_read():
    assert probe_split(50 * BATCH_SIZE) == 40 * BATCH_SIZE


def test_split_clip_takes_frames_0_to_3_and_8_to_11():
    clip = torch.arange(12.0).reshape(1, 12, 1, 1, 1).expand(2, 12, 1, 32, 32)
    ctx, tgt = split_clip(clip)
    assert ctx[0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert tgt[0, :, 0, 0, 0].tolist() == [8.0, 9.0, 10.0, 11.0]


def test_ridge_probe_reads_a_linear_target_almost_exactly():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(500, 8, generator=gen, dtype=torch.double)
    Y = Z @ torch.randn(8, 2, generator=gen, dtype=torch.double) + 3.0
    assert ridge_probe_error(Z, Y, n_train=400) < 1e-3


def test_ridge_probe_on_constant_embeddings_predicts_the_training_mean():
    # A collapsed encoder: every clip has the same embedding, so the probe learns a constant.
    Z = torch.ones(6, 3, dtype=torch.double)
    Y = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [1.0, 1.0], [5.0, 5.0]])
    # Fit on the first 4 rows (mean 3), held-out rows are 1 and 5: the error is 2.
    assert abs(ridge_probe_error(Z, Y.double(), n_train=4) - 2.0) < 1e-2


def test_ridge_probe_is_scored_on_held_out_rows_only():
    gen = torch.Generator().manual_seed(0)
    Z = torch.randn(100, 4, generator=gen, dtype=torch.double)
    Y = Z[:, :2].clone()
    Y[80:] += 10.0  # only the held-out rows are moved
    assert abs(ridge_probe_error(Z, Y, n_train=80) - 10.0) < 1e-2


class _Clips(torch.utils.data.Dataset):
    """A few random clips with a position for each of two shapes."""

    def __init__(self, n: int) -> None:
        gen = torch.Generator().manual_seed(0)
        self.clips = torch.rand(n, 12, 1, 32, 32, generator=gen)
        self.states = torch.rand(n, 2, 2, generator=gen) * 32

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clips[idx], self.states[idx]


def test_probe_eval_stops_at_50_batches_and_returns_an_error_in_pixels():
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(_Clips(120), batch_size=2)  # 60 batches
    seen = []
    model = ToyJEPA(dim=8)
    model.encoder.register_forward_hook(lambda *_: seen.append(1))
    err = probe_eval(model, loader, torch.device("cpu"))
    assert len(seen) == 50
    assert 0.0 < err < 32.0


def test_train_epoch_moves_the_weights_and_returns_finite_numbers():
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(_Clips(16), batch_size=8)
    model = ToyJEPA(dim=8)
    before = model.encoder.net[0].weight.clone()
    target_before = model.target_encoder.net[0].weight.clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    args = argparse.Namespace(normalize=False, no_var_reg=False, no_ema=False, var_weight=1.0)
    loss, enc_std = train_epoch(model, loader, opt, args, torch.device("cpu"))
    assert loss == loss and enc_std == enc_std  # neither is NaN
    assert enc_std > 0
    assert not torch.equal(model.encoder.net[0].weight, before)
    assert not torch.equal(model.target_encoder.net[0].weight, target_before)
