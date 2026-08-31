"""Tests for the campaign diagnostic figures.

The point of these is not that the pictures look right -- that needs eyes.
It is that (a) the figures are actually produced from a real estimate object,
so an interface drift in FEPEstimate breaks a test rather than a cluster run,
and (b) a plotting failure can never take down a completed campaign, since
the numbers are the deliverable and the figures are a convenience.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from aemwater.fep import diagnostics as diag


def _estimate(n_morph: int = 2, *, with_diagnostics: bool = True):
    """A stand-in shaped like a real FEPEstimate.

    The per-state curves and overlaps deliberately hang off
    ``per_morphology[i].legs[leg].diagnostics``, which is where the estimators
    actually put them. An earlier version of this fixture invented a flattened
    top-level dict; the figures were written to match the fixture, and both
    agreed with each other and with nothing the pipeline produces.
    """
    lj_lam = [0.0, 0.2, 0.5, 0.8, 1.0]
    coul_lam = [0.0, 0.25, 0.5, 0.75, 1.0]
    legs = {}
    if with_diagnostics:
        legs = {
            "lj": SimpleNamespace(diagnostics={
                "lambdas": lj_lam,
                "dudl_mean": [0.5, 3.0, 3.5, 1.5, 0.5],
                "dudl_sd": [1.0, 6.0, 3.0, 2.0, 1.0],
                "neighbour_overlap": [0.31, 0.02, 0.14, 0.28],
            }),
            "coul": SimpleNamespace(diagnostics={
                "lambdas": coul_lam,
                "dudl_mean": [-5.9, -6.8, -10.4, -10.9, -11.9],
                "dudl_sd": [4.0, 4.3, 5.0, 5.5, 8.0],
                "neighbour_overlap": [0.34, 0.26, 0.24, 0.21],
            }),
        }
    per = [
        SimpleNamespace(index=i, mu_ex=-6.8 - 0.1 * (i % 3), stderr=0.5 + 0.05 * i,
                        legs=legs)
        for i in range(n_morph)
    ]
    return SimpleNamespace(
        mu_ex=-6.83, stderr=0.30, ci95=0.59, n_morphologies=n_morph,
        per_morphology=per, var_between=0.02, var_within=0.30,
        between_unmeasured=(n_morph < 2), between_clamped=False,
        diagnostics={},
    )


@pytest.mark.parametrize("n_morph", [1, 2, 3, 6])
def test_all_three_figures_are_written(tmp_path, n_morph):
    written = diag.write_campaign_figures(_estimate(n_morph), tmp_path)
    names = {Path(p).name for p in written}
    assert names == {"fep_dudl.png", "fep_overlap.png", "fep_morphologies.png"}
    for p in written:
        assert Path(p).stat().st_size > 1000


def test_missing_diagnostics_still_yields_the_morphology_figure(tmp_path):
    # A TI-only run has no overlap matrix and a finite-difference run may have
    # no per-state fluctuations. Losing those two panels must not cost the
    # third, which is the one that says whether to run longer or pack more.
    written = diag.write_campaign_figures(
        _estimate(3, with_diagnostics=False), tmp_path)
    assert [Path(p).name for p in written] == [
        "fep_morphologies.png"]


def test_a_plotting_failure_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    # The numbers are the deliverable; figures are a convenience. A campaign
    # that finished must not be lost to a matplotlib problem.
    def boom(*_a, **_k):
        raise RuntimeError("no font cache")

    monkeypatch.setattr(diag, "plot_dudl", boom)
    with caplog.at_level(logging.WARNING):
        written = diag.write_campaign_figures(_estimate(2), tmp_path)
    assert not any("dudl" in str(p) for p in written)
    assert any("fep_dudl" in r.message or "no font cache" in str(r.message)
               for r in caplog.records)
    # the other two survived
    assert len(written) == 2


def test_single_morphology_says_between_cell_spread_is_unmeasured(tmp_path):
    # With M=1 there is no between-cell variance to report. The figure must
    # not imply a converged spread, since that is the number a reader would
    # quote when deciding whether one cell was enough.
    est = _estimate(1)
    path = diag.plot_morphologies(est, tmp_path / "one.png")
    assert Path(path).exists()


def test_overlap_flags_the_pair_below_threshold(tmp_path):
    # The below-threshold marker is the actionable part of this figure: a pair
    # under the acceptance floor makes dF across it unidentifiable regardless
    # of run length, so it must be drawn distinctly rather than as one of many.
    _, overlaps = diag._figure_data(_estimate(2))
    assert min(overlaps["lj"][1]) < 0.03, "fixture must contain a failing pair"
    fig_path = diag.plot_overlap({"lj": overlaps["lj"]},
                                 tmp_path / "ov.png", min_overlap=0.03)
    assert Path(fig_path).exists()


def _leg(estimator, **diag):
    from aemwater.fep.estimators import LegEstimate
    from aemwater.fep.schedule import FEPLeg

    return LegEstimate(estimator=estimator, leg=FEPLeg.LJ, delta_f=2.0,
                       stderr=0.3, n_effective=500.0, diagnostics=diag)


LADDER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_figure_data_reads_the_structures_a_real_campaign_produces():
    """The curves live on per_morphology[i].legs[leg].diagnostics.

    Written against the real dataclasses rather than a flattened dict: an
    earlier draft of the figures read keys that no estimator ever writes, and
    the only symptom was the figures quietly degrading to one panel on the
    cluster. This test fails loudly if that shape drifts again.
    """
    from aemwater.fep.campaign import FEPEstimate, MorphologyEstimate
    from aemwater.fep import diagnostics as diag

    legs = {
        "lj": _leg("mbar", lambdas=LADDER,
                   dudl_mean=[0.5, 3.0, 3.5, 1.5, 0.9, 0.5],
                   dudl_sd=[1.0, 6.0, 3.0, 2.0, 1.4, 1.0],
                   neighbour_overlap=[0.3, 0.02, 0.14, 0.28, 0.22]),
    }
    est = FEPEstimate(
        mu_ex=-6.8, stderr=0.5, temperature=298.0, n_morphologies=1,
        per_morphology=[MorphologyEstimate(index=0, mu_ex=-6.8, stderr=0.5,
                                          legs=legs)],
        between_unmeasured=True,
    )
    curves, overlaps = diag._figure_data(est)
    assert set(curves) == {"lj"}, "the LJ curve must be found"
    assert [len(x) for x in curves["lj"]] == [6, 6, 6]
    # One midpoint per neighbouring pair, i.e. one fewer than states.
    assert len(overlaps["lj"][0]) == len(LADDER) - 1
    assert overlaps["lj"][0][0] == pytest.approx(0.1)


def test_a_standard_error_is_never_drawn_as_the_fluctuation_band():
    """dudl_stderr must not stand in for dudl_sd.

    They differ by sqrt(N). The band is documented as the per-state
    fluctuation -- the quantity that governs neighbour overlap -- so filling it
    with the error on the mean would draw a band far too narrow while still
    looking like the thing a reader uses to judge the ladder.
    """
    from aemwater.fep.campaign import FEPEstimate, MorphologyEstimate
    from aemwater.fep import diagnostics as diag

    legs = {"lj": _leg("ti", lambdas=LADDER,
                       dudl_mean=[0.5, 3.0, 3.5, 1.5, 0.9, 0.5],
                       dudl_stderr=[0.1, 0.2, 0.15, 0.1, 0.1, 0.1])}
    est = FEPEstimate(
        mu_ex=-6.8, stderr=0.5, temperature=298.0, n_morphologies=1,
        per_morphology=[MorphologyEstimate(index=0, mu_ex=-6.8, stderr=0.5,
                                          legs=legs)],
        between_unmeasured=True,
    )
    curves, _ = diag._figure_data(est)
    assert curves == {}, "no sd means no curve, not a stderr band"


def test_select_reported_keeps_the_loser_s_unique_diagnostics():
    """MBAR measures overlap, TI measures the dU/dlambda profile.

    Only the reported estimate survives past select_reported, so without this
    carrying the unreported estimator's measurements are lost -- and they cost
    the entire leg to produce. Own keys must still win over carried ones.
    """
    from aemwater.fep.campaign import select_reported

    mbar = _leg("mbar", lambdas=LADDER,
                neighbour_overlap=[0.3, 0.2, 0.2, 0.3, 0.3])
    ti = _leg("ti", lambdas=[9.0] * 6,
              dudl_mean=[0.5, 3.0, 3.5, 1.5, 0.9, 0.5],
              dudl_sd=[1.0, 6.0, 3.0, 2.0, 1.4, 1.0])
    rep = select_reported({"mbar": mbar, "ti": ti})

    assert rep.estimator == "mbar"
    assert "dudl_sd" in rep.diagnostics, "TI's profile must survive"
    assert "neighbour_overlap" in rep.diagnostics
    assert rep.diagnostics["lambdas"] == LADDER, \
        "the reported estimator's own ladder must not be overwritten"


def test_mbar_records_the_ladder_so_overlap_is_attributable():
    """A pair overlap below threshold is only actionable with its lambdas."""
    import numpy as np
    from aemwater.fep.estimators import EnergyMatrix, mbar_estimate
    from aemwater.fep.schedule import FEPLeg

    rng = np.random.default_rng(0)
    lambdas = [0.0, 0.5, 1.0]
    n = 200
    u_kn = np.vstack([rng.normal(2.0 * lam, 1.0, 3 * n) for lam in lambdas])
    matrix = EnergyMatrix(u_kn=u_kn, N_k=np.array([n, n, n]),
                          lambdas=lambdas, leg=FEPLeg.LJ, kT=0.5922)
    est = mbar_estimate(matrix)
    assert est.diagnostics["lambdas"] == lambdas
    assert len(est.diagnostics["neighbour_overlap"]) == len(lambdas) - 1
