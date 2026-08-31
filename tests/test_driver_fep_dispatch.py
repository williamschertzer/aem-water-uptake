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


def test_screening_spec_is_forced_to_one_cell():
    # run_membrane_campaign refuses fewer cells than fep.n_morphologies, and the
    # loop has exactly one. Forcing the spec is what makes the two agree; the
    # between-morphology term comes from replicating the loop instead.
    cfg = _config("fep")
    spec = replace(cfg.fep.at_screening_resolution(), n_morphologies=1)
    assert spec.n_morphologies == 1
    # only the sampling is reduced -- the Hamiltonian must be untouched
    assert spec.soft_core_n == cfg.fep.soft_core_n
    assert spec.alpha_lj == cfg.fep.alpha_lj
    assert spec.kspace_accuracy == cfg.fep.kspace_accuracy


def test_widom_method_still_reads_the_insertion_file():
    cfg = _config("widom")
    assert cfg.mu_ex_method == "widom"
    assert cfg.widom.enabled, "widom method with sampling off would measure nothing"

def test_helper_forces_one_cell_and_reports(tmp_path, monkeypatch):
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
        seen["states"] = len(cfg.fep.lj_lambdas)
        return _Est()

    monkeypatch.setattr("aemwater.fep.campaign.run_membrane_campaign",
                        fake_campaign)
    monkeypatch.setattr("aemwater.assembly.assemble",
                        lambda *a, **k: object())

    cfg = _config("fep")
    est = driver._membrane_mu_ex_fep(cfg, tmp_path, object(), None, 30.0, 3)

    assert est.mu_ex == pytest.approx(-6.41)
    assert seen["n_cells"] == 1
    # the check inside run_membrane_campaign compares these two; a mismatch is
    # the CampaignError that would abort every iteration.
    assert seen["n_morphologies"] == seen["n_cells"]
    # screening resolution, not production
    assert seen["states"] == len(cfg.fep.at_screening_resolution().lj_lambdas)
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
