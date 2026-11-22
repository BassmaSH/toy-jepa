"""Tests for jepa.py: the collapse metric behind the enc_std column, the loss terms, the EMA update.

Code written by Claude Code on Nov 15, 2026, for the author.
"""

import torch

from jepa import ToyJEPA, embedding_std, jepa_loss, variance_regularizer


def test_embedding_std_is_zero_when_every_row_is_the_same():
    assert embedding_std(torch.ones(16, 4)) == 0.0


def test_embedding_std_is_the_mean_of_the_per_dimension_std():
    # Column 0 has standard deviation 1 across the batch, column 1 has 3.
    z = torch.tensor([[-1.0, -3.0], [0.0, 0.0], [1.0, 3.0]])
    assert abs(embedding_std(z) - 2.0) < 1e-6


def test_variance_term_is_zero_above_the_floor_and_near_the_floor_at_collapse():
    spread = torch.tensor([[-2.0, -3.0], [0.0, 0.0], [2.0, 3.0]])
    assert variance_regularizer(spread).item() == 0.0
    assert abs(variance_regularizer(torch.ones(16, 4)).item() - 1.0) < 1e-2


def test_loss_is_zero_when_the_prediction_equals_the_target():
    z = torch.randn(8, 4, generator=torch.Generator().manual_seed(0))
    assert jepa_loss(z, z).item() == 0.0
    assert jepa_loss(z, z + 1.0).item() > 0.0


def test_target_never_carries_a_gradient():
    torch.manual_seed(0)
    ctx, tgt = torch.rand(2, 4, 1, 32, 32), torch.rand(2, 4, 1, 32, 32)
    for use_ema in (True, False):
        pred, z_tgt = ToyJEPA(dim=8, use_ema=use_ema)(ctx, tgt)
        assert pred.requires_grad
        assert not z_tgt.requires_grad


def test_target_encoder_starts_as_a_copy_and_is_left_out_of_the_optimizer():
    torch.manual_seed(0)
    model = ToyJEPA(dim=8)
    for t, o in zip(model.target_encoder.parameters(), model.encoder.parameters(), strict=True):
        assert torch.equal(t, o)
        assert not t.requires_grad


def test_ema_update_moves_the_target_one_hundredth_of_the_way():
    torch.manual_seed(0)
    model = ToyJEPA(dim=8, ema_momentum=0.99)
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)  # the online encoder is now exactly 1 above the target
    before = [t.clone() for t in model.target_encoder.parameters()]
    model.ema_update()
    for t, b in zip(model.target_encoder.parameters(), before, strict=True):
        assert torch.allclose(t - b, torch.full_like(t, 0.01), atol=1e-6)
