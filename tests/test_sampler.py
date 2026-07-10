# SPDX-License-Identifier: MIT
"""Sampler unit tests: temperature/top-p behavior, plus the per-step logit-processor
hook (issue #38) -- a list of (input_ids, logits) -> logits callables applied
before sampling, the seam structured-output grammars (#39) plug into. Pure torch,
no CUDA/fni8 needed."""

import torch

from fni8serve.layers.sampler import Sampler


def test_greedy_matches_argmax():
    s = Sampler()
    logits = torch.tensor([[0.1, 0.9, 0.2], [0.5, 0.1, 0.3]])
    temps = torch.tensor([0.0, 0.0])
    out = s(logits, temps)
    assert out.tolist() == [1, 0]


def test_greedy_short_circuit_equals_full_path():
    """Issue #183: the all-greedy fast path (skips softmax/top-p sort/Gumbel) must
    be BIT-IDENTICAL to the old full path, which selected `argmax` for every
    temp==0 row via `torch.where` anyway. Check the fast path, the unhinted
    tensor-reduction path, and forcing the full sampled path all agree with the
    raw argmax."""
    s = Sampler()
    torch.manual_seed(0)
    logits = torch.randn(5, 151936)  # full-vocab width, like real decode
    temps = torch.zeros(5)
    top_p = torch.ones(5)
    ref = logits.float().argmax(dim=-1)

    fast = s(logits, temps, top_p=top_p, all_greedy=True)  # explicit hint
    unhinted = s(logits, temps, top_p=top_p)  # tensor reduction
    forced_full = s(logits, temps, top_p=top_p, all_greedy=False)  # runs the full path

    assert torch.equal(fast, ref)
    assert torch.equal(unhinted, ref)
    assert torch.equal(forced_full, ref)


def test_mixed_batch_greedy_rows_unaffected_by_short_circuit():
    """A batch mixing greedy and sampled rows must NOT take the all-greedy fast
    path, and the greedy rows must still match argmax."""
    s = Sampler()
    torch.manual_seed(1)
    logits = torch.randn(2, 64)
    temps = torch.tensor([0.0, 1.0])  # one greedy, one sampled
    out = s(logits, temps, top_p=torch.ones(2))
    assert out[0].item() == logits[0].float().argmax().item()


def test_top_p_one_skip_matches_full_sort():
    """top_p == 1.0 keeps the whole distribution, so skipping the nucleus sort
    must be identical to running it. Same RNG seed for the Gumbel noise -> same
    sampled token either way."""
    s = Sampler()
    logits = torch.randn(4, 2048)
    temps = torch.full((4,), 0.8)
    top_p = torch.ones(4)

    torch.manual_seed(123)
    skipped = s(logits, temps, top_p=top_p, any_top_p=False)  # skips the sort
    torch.manual_seed(123)
    sorted_path = s(logits, temps, top_p=top_p, any_top_p=True)  # runs the sort
    assert torch.equal(skipped, sorted_path)


def _force(token: int):
    def proc(input_ids, row):
        row = row.clone()
        row[:] = float("-inf")
        row[token] = 0.0
        return row

    return proc


def test_logit_processor_forces_a_token():
    """A processor that masks every token but one must make that token win even
    though it isn't the raw argmax."""
    s = Sampler()
    logits = torch.tensor([[5.0, 1.0, 1.0]])  # token 0 is the greedy pick
    temps = torch.tensor([0.0])
    out = s(logits, temps, logit_processors=[[_force(2)]], input_ids=[[7, 8, 9]])
    assert out.tolist() == [2]


def test_logit_processor_receives_the_sequences_own_input_ids():
    s = Sampler()
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    temps = torch.tensor([0.0, 0.0])
    seen = []

    def record(input_ids, row):
        seen.append(list(input_ids))
        return row

    s(logits, temps, logit_processors=[[record], [record]], input_ids=[[1, 2], [3, 4, 5]])
    assert seen == [[1, 2], [3, 4, 5]]


def test_logit_processors_none_is_a_no_op():
    s = Sampler()
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    temps = torch.tensor([0.0])
    out_with_none = s(logits, temps, logit_processors=None)
    out_default = s(logits, temps)
    assert out_with_none.tolist() == out_default.tolist()


def test_only_some_sequences_have_processors():
    """A batch can mix sequences with and without processors -- an empty list for a
    given row must leave that row's logits untouched."""
    s = Sampler()
    logits = torch.tensor([[5.0, 1.0, 1.0], [1.0, 5.0, 1.0]])
    temps = torch.tensor([0.0, 0.0])
    out = s(logits, temps, logit_processors=[[_force(2)], []], input_ids=[[1], [2]])
    assert out.tolist() == [2, 1]


def test_logit_processors_chain_in_order():
    """Multiple processors on one sequence apply in list order."""
    s = Sampler()
    logits = torch.tensor([[5.0, 1.0, 1.0]])
    temps = torch.tensor([0.0])
    out = s(logits, temps, logit_processors=[[_force(1), _force(2)]], input_ids=[[0]])
    assert out.tolist() == [2]
