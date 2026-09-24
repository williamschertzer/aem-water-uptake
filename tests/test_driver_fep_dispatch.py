"""The uptake loop must measure the membrane with the configured estimator.

Before this was wired, `mu_ex_method: fep` computed the *bulk reference* by FEP
but still read the membrane number from Widom insertion. The saturation test is
a difference, and the Widom design only works because both halves carry the same
insertion bias (see the driver module docstring). Pairing a converged FEP
reference with an under-converged Widom membrane estimate differs by that bias --
several kcal/mol -- so the loop would have stopped many waters early while every
logged number looked plausible. These tests exist to keep that from returning.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from aemwater import driver
from aemwater.config import PolymerSpec, RunConfig


def _config(method: str) -> RunConfig:
    cfg = RunConfig(polymer=PolymerSpec(smiles="CC"))
    return replace(cfg, mu_ex_method=method)


def test_fep_is_the_default_membrane_estimator():
    # If this flips, the loop silently returns to the biased pairing above.
    assert RunConfig(polymer=PolymerSpec(smiles="CC")).mu_ex_method == "fep"


def test_widom_method_still_reads_the_insertion_file():
    cfg = _config("widom")
    assert cfg.mu_ex_method == "widom"
    assert cfg.widom.enabled, "widom method with sampling off would measure nothing"


@pytest.mark.parametrize("production_steps,equil_steps,n_states", [
    (500_000, 50_000, 11),
    (2_000, 1_000, 3),
])
def test_helper_preserves_configured_fep_settings(
        tmp_path, monkeypatch, production_steps, equil_steps, n_states):
    """The helper must hand run_membrane_campaign exactly one cell, with a spec
    whose n_morphologies matches, and write a per-iteration report."""
    seen = {}

    class _Est:
        mu_ex, stderr = -6.41, 0.55
        per_morphology, diagnostics = (), {}

        def summary(self):
            return {"mu_ex": self.mu_ex}

    def fake_campaign(cfg, workdir, systems, ranks=1, **kw):
        seen["n_cells"] = len(systems)
        seen["n_morphologies"] = cfg.fep.n_morphologies
        seen["spec"] = cfg.fep
        return _Est()

    monkeypatch.setattr("aemwater.fep.campaign.run_membrane_campaign",
                        fake_campaign)
    monkeypatch.setattr("aemwater.assembly.assemble",
                        lambda *a, **k: object())

    cfg = _config("fep").with_overrides(**{
        "fep.lj_lambdas": [i / (n_states - 1) for i in range(n_states)],
        "fep.coul_lambdas": [i / (n_states - 1) for i in range(n_states)],
        "fep.production_steps": production_steps,
        "fep.equil_steps": equil_steps,
        "fep.sample_every": 100,
        "fep.max_stderr": 0.2,
    })
    est = driver._membrane_mu_ex_fep(cfg, tmp_path, object(), None, 30.0, 3)

    assert est.mu_ex == pytest.approx(-6.41)
    assert seen["n_cells"] == 1
    # the check inside run_membrane_campaign compares these two; a mismatch is
    # the CampaignError that would abort every iteration.
    assert seen["n_morphologies"] == seen["n_cells"]
    # All user settings survive dispatch, including ladders, step counts and
    # strict precision thresholds; only the single-cell count is specialized.
    assert seen["spec"] == replace(cfg.fep, n_morphologies=1)
    assert (tmp_path / "fep_membrane.json").exists()


def test_helper_does_not_mutate_the_caller_config(tmp_path, monkeypatch):
    # The loop reuses `config` at every iteration and the production spec is
    # what the final campaign needs; forcing one morphology must be local.
    monkeypatch.setattr("aemwater.fep.campaign.run_membrane_campaign",
                        lambda *a, **k: type("E", (), {
                            "mu_ex": -6.4, "stderr": 0.5, "per_morphology": (),
                            "diagnostics": {},
                            "summary": lambda self: {}})())
    monkeypatch.setattr("aemwater.assembly.assemble", lambda *a, **k: object())
    cfg = _config("fep")
    before = cfg.fep.n_morphologies
    driver._membrane_mu_ex_fep(cfg, tmp_path, object(), None, 30.0, 0)
    assert cfg.fep.n_morphologies == before
