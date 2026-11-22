"""Tests for bouncing_shapes.py: the clips every run trains on and the positions the probe reads.

Code written by Claude Code on Nov 15, 2026, for the author.
"""

import torch

from bouncing_shapes import MAX_RADIUS, MIN_RADIUS, RADIUS_JITTER, STATE_FRAME, BouncingShapes


def test_clip_shape_and_values():
    clip = BouncingShapes(num_clips=4)[0]
    assert clip.shape == (12, 1, 32, 32)
    assert clip.dtype == torch.float32
    assert set(clip.unique().tolist()) == {0.0, 1.0}


def test_state_is_one_xy_row_per_shape():
    clip, state = BouncingShapes(num_clips=4, return_state=True)[0]
    assert clip.shape == (12, 1, 32, 32)
    assert state.shape == (2, 2)
    assert state.dtype == torch.float32


def test_same_index_gives_the_same_clip():
    a, state_a = BouncingShapes(num_clips=4, return_state=True)[3]
    b, state_b = BouncingShapes(num_clips=4, return_state=True)[3]
    assert torch.equal(a, b)
    assert torch.equal(state_a, state_b)


def test_another_index_or_seed_gives_another_clip():
    ds = BouncingShapes(num_clips=4)
    assert not torch.equal(ds[0], ds[1])
    assert not torch.equal(ds[0], BouncingShapes(num_clips=4, seed=1)[0])


def test_state_is_where_the_shapes_are_drawn_at_the_state_frame():
    # One shape, so the lit pixels of the state frame are that shape and their centre is its
    # position, to within a pixel.
    for idx in range(20):
        clip, state = BouncingShapes(num_clips=20, n_shapes=1, return_state=True)[idx]
        ys, xs = torch.nonzero(clip[STATE_FRAME, 0], as_tuple=True)
        assert abs(xs.float().mean().item() - state[0, 0].item()) < 1.0
        assert abs(ys.float().mean().item() - state[0, 1].item()) < 1.0


def test_the_last_shape_is_always_the_largest():
    # The probe reads state[-1] as "the largest shape". That holds when the radius ranges of the
    # two shapes cannot overlap, whatever the jitter.
    assert MIN_RADIUS + RADIUS_JITTER < MAX_RADIUS - RADIUS_JITTER
