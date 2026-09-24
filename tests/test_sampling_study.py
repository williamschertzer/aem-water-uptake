import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_uptake_convergence", Path(__file__).parents[1] / "scripts/check_uptake_convergence.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def write_campaign(path, values, usable=True):
    path.mkdir()
    (path / "morphology_results.json").write_text(json.dumps([
        {"seed": i, "water_uptake_wt_pct": x, "usable": usable}
        for i, x in enumerate(values)]))


def test_paired_convergence_requires_equivalence_not_just_overlapping_intervals(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    write_campaign(a, [9, 10, 11])
    write_campaign(b, [9.1, 10.1, 11.1])
    write_campaign(c, [8, 10, 12])
    assert study.compare(a, b)["passed"]
    assert not study.compare(a, c)["passed"]  # zero mean, unresolved spread


def test_incomplete_campaign_is_not_assessable(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_campaign(a, [9, 10, 11])
    write_campaign(b, [9, 10, 11], usable=False)
    with pytest.raises(ValueError, match="all thermodynamically saturated"):
        study.compare(a, b)
