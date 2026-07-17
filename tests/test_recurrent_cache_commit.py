# SPDX-License-Identifier: MIT
"""Focused correctness checks for accepted verify-trajectory commits."""

import pytest
import torch

from fni8serve.models.cache import RecurrentStateCache


@pytest.mark.parametrize(
    ("slots", "row_last"),
    [([2], [1]), ([3, 1], [2, 0])],
)
def test_commit_verify_selects_each_rows_accepted_state(slots, row_last):
    cache = RecurrentStateCache()
    cache.enable_static_buffers(num_slots=4)
    cache.bind(slots)
    cache.set_state(7, torch.zeros(len(slots), 2))
    cache.set_conv_tail(7, torch.zeros(len(slots), 3))

    batch_bucket, verify_tokens = 4, 3
    state = torch.arange(batch_bucket * verify_tokens * 2, dtype=torch.float32).reshape(
        batch_bucket, verify_tokens, 2
    )
    conv = torch.arange(batch_bucket * verify_tokens * 3, dtype=torch.float32).reshape(
        batch_bucket, verify_tokens, 3
    )
    cache.begin_verify_capture()
    cache.record_verify_traj(7, state, conv)
    cache.commit_verify(row_last)

    for row, slot in enumerate(slots):
        assert torch.equal(cache._state_buf[7][slot], state[row, row_last[row]])
        assert torch.equal(cache._conv_buf[7][slot], conv[row, row_last[row]])
    assert not cache.capturing_verify
    assert cache.verify_captured_layers() == 0


def test_batch_one_pick_is_a_view():
    trajectory = torch.arange(24).reshape(2, 3, 4)
    selected = RecurrentStateCache._pick(trajectory, [2])

    assert torch.equal(selected, trajectory[0, 2].unsqueeze(0))
    assert (
        selected.untyped_storage().data_ptr() == trajectory.untyped_storage().data_ptr()
    )
