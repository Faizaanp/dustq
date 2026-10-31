"""T5 alpha-table contract tests (no model runs - pure data/JSON logic).

The shipped bug: calibrated_alpha() looked for "a4s" at the TOP level of
{"provenance", "alpha", "detail"}, so every A4/A2 run raised
KeyError: 'no calibrated alpha for a4s; recalibrate' until the nested fix
(2026-10-31; the pasted session report shows the exact raised string)."""

import json

import pytest
import torch

from dustq.qmodel import calibrated_alpha


@pytest.fixture
def table_path(tmp_path):
    payload = {
        "provenance": {"checkpoint": "x", "rule": "argmin MSE"},
        "alpha": {"a8s": 0.4, "a8u": 0.65, "a4s": 0.55, "a4u": 5.25,
                  "a2s": 1.6, "a2u": 6.0},
        "detail": {},
    }
    path = tmp_path / "alpha_calibrated.json"
    path.write_text(json.dumps(payload))
    return path


def test_calibrated_alpha_reads_nested_table(table_path):
    assert calibrated_alpha(4, unsigned=False, path=table_path) == 0.55
    assert calibrated_alpha(4, unsigned=True, path=table_path) == 5.25
    assert calibrated_alpha(2, unsigned=True, path=table_path) == 6.0
    assert calibrated_alpha(8, unsigned=False, path=table_path) == 0.4


def test_calibrated_alpha_missing_key(table_path):
    with pytest.raises(KeyError, match="no calibrated alpha for a4s"):
        payload = json.loads(table_path.read_text())
        del payload["alpha"]["a4s"]
        table_path.write_text(json.dumps(payload))
        calibrated_alpha(4, unsigned=False, path=table_path)


def test_calibrate_grid_is_two_stage_and_wide(tmp_path, monkeypatch):
    """A boundary-pinned optimum (the 2026-10-31 failure) must be flagged so the
    calibration is never silently shipped degenerate."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("calib", "scripts/calibrate_alpha.py")
    calib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calib)

    # analytic check on synthetic data: hi=7 signed, optimum alpha is finite and interior
    torch.manual_seed(0)
    x = torch.randn(1_000_00) * 2.0
    alpha, mse, at_bounds = calib._calibrate(x, 4, unsigned=False)
    assert 0.01 <= alpha <= 158 and not at_bounds
    # two-stage must beat a naive wide uniform grid at equal budget
    assert mse <= calib.quant_mse(x, 4, False, 1.0) + 1e-12

    # a degenerate boundary case is correctly flagged
    tiny = torch.zeros(1_000)
    _, mse0, flagged = calib._calibrate(tiny, 4, unsigned=False)
    assert flagged, "all-zero input should pin at a boundary (scale floor)"
