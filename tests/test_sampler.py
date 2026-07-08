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
    logits = torch.tensor([[5.0, 1.0, 1.0]])   # token 0 is the greedy pick
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
