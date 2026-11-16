"""Tests for baselines.py: the mean predictor, which needs no model and runs in a moment.

The untrained-encoder baseline is not tested here: it reads all 8,000 clips through the encoder.

Code written by Claude Code on Nov 16, 2026, for the author.
"""

import torch

from baselines import largest_shape_positions, mean_predictor_error, mean_predictor_split_errors
from bouncing_shapes import BouncingShapes


def test_mean_predictor_error_on_values_worked_by_hand():
    y_fit = torch.tensor([[0.0, 0.0], [4.0, 8.0]]).double()  # mean (2, 4)
    y_test = torch.tensor([[3.0, 4.0], [2.0, 10.0]]).double()  # errors 1, 0, 0, 6
    assert abs(mean_predictor_error(y_fit, y_test) - 1.75) < 1e-12


def test_positions_are_the_last_shape_of_each_clip():
    ds = BouncingShapes(num_clips=5, return_state=True)
    y = largest_shape_positions(ds)
    assert y.shape == (5, 2)
    assert torch.equal(y[3], ds[3][1][-1].double())


def test_split_errors_give_one_value_per_shuffle_and_repeat():
    y = torch.rand(100, 2, generator=torch.Generator().manual_seed(0)).double() * 32
    a = mean_predictor_split_errors(y, rows=90, shuffles=4)
    assert len(a) == 4
    assert len(set(a)) == 4  # each shuffle holds out other rows
    assert a == mean_predictor_split_errors(y, rows=90, shuffles=4)
