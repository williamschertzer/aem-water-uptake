"""examples/run_e2e_fep.py must stay callable against the real API.

The script is the documented way to run the pipeline end to end, and it chains
three real entry points whose signatures it cannot see. A wrong keyword or a
renamed result field surfaces only when that stage is reached -- for the
hydration loop, an hour into a run and after the dry membrane has been built.
That is an expensive way to learn about a typo.

Every stub here returns the *real* dataclass rather than a stand-in, so the
field names the script reads are checked against the current definitions. A
dict shaped to match the script would agree with it by construction and prove
nothing.

The engine itself is not exercised -- tests/test_fep_membrane_smoke.py does
that with real LAMMPS. This is about the wiring.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "examples" / "e2e_fep_smoke.yaml"

sys.path.insert(0, str(REPO / "examples"))


@pytest.fixture(scope="module")
def script():
    return importlib.import_module("run_e2e_fep")


@pytest.fixture
def stubs():
    """Fake stage returns built from the production dataclasses."""
    from aemwater.bulk import BulkReference, WidomEstimate
    from aemwater.driver import Iteration, UptakeResult, bulk_settings_for

    def fake_bulk(config, workdir, ranks=None):
        estimate = WidomEstimate(
            mu_ex=-6.8, stderr=0.4, temperature=298.15, n_blocks=3,
            block_values=np.array([-6.7, -6.8, -6.9]), mean_boltzmann=1e5,
            effective_samples=25.0, volume=8000.0,
        )
        return BulkReference(bulk_settings_for(config), estimate, 0.997,
                             8000.0, Path(workdir), "fep")

    def fake_dry(config, workdir, resume=True):
        return [object(), object()], False

    def fake_uptake(config, workdir, typed_chains, bulk_reference=None,
                    resume=True):
        iteration = Iteration(
            index=0, n_waters_before=0, n_requested=8, n_inserted=8,
            n_waters_after=8, density=1.02, volume=9000.0, lambda_value=4.0,
            water_uptake_pct=12.5, mu_ex=-5.9, mu_ex_stderr=0.5, mu_gap=0.9,
            saturated=False, geometrically_saturated=False,
            free_volume_fraction=0.05, wall_seconds=12.0,
        )
        return UptakeResult(
            iterations=[iteration], n_waters=8, lambda_value=4.0,
            water_uptake_pct=12.5, hydrated_density=1.02, dry_density=1.08,
            stop_reason="iteration cap", converged=False, bulk_mu_ex=-6.8,
            workdir=Path(workdir), composition=None, dry_convergence=None,
        )

    return fake_bulk, fake_dry, fake_uptake


def test_the_script_runs_all_three_stages(script, stubs, tmp_path, monkeypatch):
    fake_bulk, fake_dry, fake_uptake = stubs
    monkeypatch.chdir(tmp_path)

    with mock.patch.object(script, "obtain_bulk_reference", fake_bulk), \
         mock.patch.object(script, "obtain_dry_membrane", fake_dry), \
         mock.patch.object(script, "run_uptake", fake_uptake):
        rc = script.main(["--config", str(CONFIG)])

    assert rc == 0

    workdir = tmp_path / "runs" / "e2e_fep_smoke"
    summary = json.loads((workdir / "e2e_summary.json").read_text())
    assert summary["mu_ex_method"] == "fep"
    assert summary["bulk"] == {"mu_ex": -6.8, "stderr": 0.4, "method": "fep"}
    assert summary["result"]["stop_reason"] == "iteration cap"
    assert summary["result"]["converged"] is False
    assert set(summary["timings_min"]) == {
        "bulk reference", "dry membrane", "hydration loop"}
    assert (workdir / "uptake_trajectory.csv").exists()


def test_the_script_reports_the_gap_between_membrane_and_reservoir(
        script, stubs, tmp_path, monkeypatch, capsys):
    """The comparison is the whole point of the run.

    A script that ran three stages and printed no gap would look like a
    success while answering nothing.
    """
    fake_bulk, fake_dry, fake_uptake = stubs
    monkeypatch.chdir(tmp_path)

    with mock.patch.object(script, "obtain_bulk_reference", fake_bulk), \
         mock.patch.object(script, "obtain_dry_membrane", fake_dry), \
         mock.patch.object(script, "run_uptake", fake_uptake):
        script.main(["--config", str(CONFIG)])

    out = capsys.readouterr().out
    assert "mu_ex(bulk)     = -6.800 +/- 0.400" in out
    assert "mu_ex(membrane) = -5.900 +/- 0.500" in out
    assert "gap             = +0.900" in out
    # Smoke-scale numbers must never be presented as a prediction.
    assert "Not an uptake prediction" in out


def test_the_shipped_config_is_valid_and_selects_fep():
    """A broken example config is a broken tutorial."""
    from aemwater.config import RunConfig

    config = RunConfig.from_yaml(CONFIG)
    config.validate()
    assert config.mu_ex_method == "fep"

    # The screening preset is what the loop actually runs, so the cheap
    # settings have to survive it -- see test_fep_config.py for why.
    screening = config.fep.at_screening_resolution()
    assert screening.production_steps == config.fep.production_steps
    assert len(screening.lj_lambdas) == len(config.fep.lj_lambdas)
    screening.validate()
