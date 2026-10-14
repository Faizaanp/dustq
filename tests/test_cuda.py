"""CUDA-only smoke (Plan v3.1 T1; review-2 M4): one tiny training step per precision on the
GPU. This is the direct B4-class check — a QLinear built on the wrong device crashes here
before any benchmark or Tier R time is spent. Skips entirely without CUDA."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from dustq import runner  # noqa: E402


@pytest.mark.parametrize("precision", ["fp", "w1.58a4"])
def test_one_dust_step_on_gpu(precision):
    losses = runner.run_dust(precision=precision, steps=1, device="cuda")
    assert losses and all(math.isfinite(l) for l in losses)


@pytest.mark.parametrize("precision", ["fp", "w1.58a4"])
def test_one_bp_step_on_gpu(precision):
    losses = runner.run_backprop(precision=precision, steps=1, device="cuda")
    assert losses and all(math.isfinite(l) for l in losses)
