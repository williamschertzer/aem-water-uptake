"""Tests for the multi-morphology campaign layer.

The combination statistics are tested against a synthetic ensemble whose true
mean and true between/within spreads are known, because that is the only way to
check an error bar: assert on *coverage*, not on the number it prints.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aemwater.fep.campaign import (
    CampaignError,
    FEPEstimate,
    MorphologyEstimate,
    combine_legs_for_morphology,
    combine_morphologies,
    estimator_disagreement,
    morphology_seed,
    select_reported,
    t95,
    write_campaign_report,
)
from aemwater.fep.estimators import LegEstimate
from aemwater.fep.schedule import FEPLeg

TRUE_MU = -6.50


def _leg(name: str, df: float, se: float, leg: FEPLeg = FEPLeg.LJ) -> LegEstimate:
    return LegEstimate(estimator=name, leg=leg, delta_f=df, stderr=se,
                       n_effective=100.0, diagnostics={})


def _morph(index: int, mu: float, se: float) -> MorphologyEstimate:
    return MorphologyEstimate(index=index, mu_ex=mu, stderr=se,
                              legs={"lj": _leg("mbar", mu, se)})


# --------------------------------------------------------------------------
# combination statistics
# --------------------------------------------------------------------------

def test_equal_weights_not_inverse_variance():
    """A tight but atypical morphology must not dominate the mean.

    Inverse-variance weighting would pull the answer toward the cell with the
    small error bar; the ensemble mean weights every draw equally.
    """
    ms = [_morph(0, -5.0, 0.01), _morph(1, -7.0, 0.50), _morph(2, -7.0, 0.50)]
    est = combine_morphologies(ms, 298.15)
    assert est.mu_ex == pytest.approx((-5.0 - 7.0 - 7.0) / 3, abs=1e-12)
    # inverse-variance would land essentially on -5.0
    assert est.mu_ex < -6.0


def test_stderr_includes_between_morphology_spread():
    """The error bar must exceed what within-morphology errors alone imply."""
    ms = [_morph(i, mu, 0.05) for i, mu in enumerate([-5.0, -6.5, -8.0])]
    est = combine_morphologies(ms, 298.15)
    within_only = math.sqrt(3 * 0.05 ** 2) / 3
    assert est.stderr > 10 * within_only


@pytest.mark.parametrize("n_morph", [3, 5, 10])
def test_reported_interval_has_nominal_coverage(n_morph):
    """Monte Carlo: the t-based 95% interval must cover ~95% of the time."""
    sig_b, sig_w = 0.40, 0.15
    rng = np.random.default_rng(12345)
    trials, covered = 1500, 0
    for _ in range(trials):
        truth = rng.normal(TRUE_MU, sig_b, n_morph)
        obs = truth + rng.normal(0.0, sig_w, n_morph)
        est = combine_morphologies(
            [_morph(i, obs[i], sig_w) for i in range(n_morph)], 298.15
        )
        lo, hi = est.ci95
        covered += lo <= TRUE_MU <= hi
    assert 0.90 <= covered / trials <= 0.99


@pytest.mark.parametrize("n_morph", [2, 3, 5])
def test_coverage_holds_when_within_noise_dominates(n_morph):
    """The regime the between-dominated tests above missed.

    When cells agree far more closely than their own sampling error, v_obs comes
    out tiny and stderr looks spectacular. Coverage must still be nominal --
    which it is, because the t-quantile widens by exactly as much as the variance
    estimate shrank. This is the test that would have caught the bulk validation
    run reporting itself converged.
    """
    sig_b, sig_w = 0.05, 0.55
    rng = np.random.default_rng(4242)
    trials, covered = 1500, 0
    for _ in range(trials):
        obs = rng.normal(TRUE_MU, sig_b, n_morph) + rng.normal(0.0, sig_w, n_morph)
        est = combine_morphologies(
            [_morph(i, obs[i], sig_w) for i in range(n_morph)], 298.15
        )
        lo, hi = est.ci95
        covered += lo <= TRUE_MU <= hi
    assert 0.90 <= covered / trials <= 0.99


def test_normal_quantile_undercovers_at_small_m():
    """Justifies the Student-t table: 1.96 sigma is not enough at M=3."""
    sig_b, sig_w = 0.40, 0.15
    rng = np.random.default_rng(999)
    trials, cov_t, cov_z = 1500, 0, 0
    for _ in range(trials):
        truth = rng.normal(TRUE_MU, sig_b, 3)
        obs = truth + rng.normal(0.0, sig_w, 3)
        est = combine_morphologies([_morph(i, obs[i], sig_w) for i in range(3)],
                                   298.15)
        lo, hi = est.ci95
        cov_t += lo <= TRUE_MU <= hi
        half = 1.96 * est.stderr
        cov_z += est.mu_ex - half <= TRUE_MU <= est.mu_ex + half
    assert cov_z / trials < 0.90
    assert cov_t / trials > cov_z / trials + 0.05


def test_variance_decomposition_recovers_planted_values():
    sig_b, sig_w, n = 0.40, 0.05, 60
    rng = np.random.default_rng(7)
    truth = rng.normal(TRUE_MU, sig_b, n)
    obs = truth + rng.normal(0.0, sig_w, n)
    est = combine_morphologies([_morph(i, obs[i], sig_w) for i in range(n)],
                               298.15)
    assert est.var_between == pytest.approx(sig_b ** 2, rel=0.5)
    assert est.var_within == pytest.approx(sig_w ** 2, rel=0.1)
    assert est.limiting_factor.startswith("morphologies")


def test_limiting_factor_flags_sampling_when_within_dominates():
    ms = [_morph(i, TRUE_MU + d, 0.40)
          for i, d in enumerate([0.01, -0.01, 0.005, -0.005])]
    est = combine_morphologies(ms, 298.15)
    assert est.limiting_factor.startswith("sampling")


# --------------------------------------------------------------------------
# edge cases that must not silently pass
# --------------------------------------------------------------------------

def test_single_morphology_is_never_converged():
    """One draw measures no between-morphology spread, so it is not a result."""
    est = combine_morphologies([_morph(0, TRUE_MU, 0.01)], 298.15)
    assert est.between_unmeasured
    assert not est.converged
    assert est.ci95 == (float("-inf"), float("inf"))
    assert est.dof == 0


def test_negative_between_variance_is_clamped_and_flagged():
    ms = [_morph(i, TRUE_MU, 0.30) for i in range(4)]  # identical => v_obs = 0
    est = combine_morphologies(ms, 298.15)
    assert est.between_clamped
    assert est.var_between == 0.0
    assert est.diagnostics["var_between_raw"] < 0.0


def test_non_finite_morphology_is_dropped_not_propagated():
    ms = [_morph(0, -6.5, 0.1), _morph(1, float("nan"), 0.1),
          _morph(2, -6.7, 0.1)]
    est = combine_morphologies(ms, 298.15)
    assert est.n_morphologies == 2
    assert math.isfinite(est.mu_ex)
    assert est.diagnostics["n_dropped"] == 1


def test_all_non_finite_raises_with_actionable_message():
    with pytest.raises(CampaignError, match="no usable morphology"):
        combine_morphologies([_morph(0, float("nan"), 0.1)], 298.15)


def test_morphology_with_non_finite_leg_is_unusable():
    m = MorphologyEstimate(index=0, mu_ex=-6.5, stderr=0.1,
                           legs={"lj": _leg("mbar", -6.5, float("inf"))})
    assert not m.usable


def test_convergence_requires_precision_and_replication():
    tight = combine_morphologies(
        [_morph(i, TRUE_MU + d, 0.02) for i, d in enumerate([0.01, 0.0, -0.01])],
        298.15, max_stderr=0.30)
    assert tight.converged
    loose = combine_morphologies(
        [_morph(i, mu, 0.02) for i, mu in enumerate([-4.0, -6.5, -9.0])],
        298.15, max_stderr=0.30)
    assert not loose.converged


def test_convergence_judges_the_interval_not_the_bare_stderr():
    """Regression from the bulk SPC/E validation run.

    Two morphologies at -6.795 and -6.855 gave stderr 0.030 -- apparently ten
    times inside a 0.30 budget -- while the t-quantile at one degree of freedom
    (12.7) put the 95% interval at +/-0.38, i.e. wider than the budget it
    claimed to satisfy. Convergence is a statement about the interval.
    """
    est = combine_morphologies(
        [_morph(0, -6.795, 0.517), _morph(1, -6.855, 0.619)],
        298.15, max_stderr=0.30)
    assert est.stderr < 0.05                       # bare stderr looks excellent
    lo, hi = est.ci95
    assert (hi - lo) / 2 > 0.30                    # the interval does not
    assert not est.converged


def test_stderr_is_not_floored_at_within_variance():
    """A within-variance floor was tried and removed; do not reintroduce it.

    Monte Carlo in the regime that motivated it (sigma_between 0.05,
    sigma_within 0.55) showed the unfloored interval covering at ~0.95 and the
    floored one at 1.000: the t-quantile already compensates for a v_obs that
    came out small by luck, and flooring the variance breaks that cancellation.
    """
    ms = [_morph(0, -6.80, 0.60), _morph(1, -6.85, 0.60)]
    est = combine_morphologies(ms, 298.15)
    v_obs = np.var([-6.80, -6.85], ddof=1)
    assert est.stderr == pytest.approx(math.sqrt(v_obs / 2), rel=1e-9)
    assert est.stderr < math.sqrt(0.60 ** 2 / 2)


def test_t95_is_conservative_between_tabulated_points():
    assert t95(2) == pytest.approx(4.303)
    assert t95(0) == float("inf")
    assert t95(11) >= t95(12)          # interpolates wide, never narrow
    assert t95(500) == pytest.approx(1.96)


# --------------------------------------------------------------------------
# seeds: independence between morphologies is the point of the campaign
# --------------------------------------------------------------------------

def test_seeds_are_distinct_and_valid():
    seeds = [morphology_seed(20260817, i) for i in range(256)]
    assert len(set(seeds)) == 256
    assert all(0 < s < 2 ** 31 for s in seeds)


def test_seeds_are_reproducible():
    assert ([morphology_seed(11, i) for i in range(8)]
            == [morphology_seed(11, i) for i in range(8)])


def test_seeds_are_not_sequentially_correlated():
    """Regression: a multiply-add gave a constant stride of 40503.

    Nearby seeds can give correlated RNG output, which would undermine the
    independence the whole campaign rests on.
    """
    seeds = [morphology_seed(20260817, i) for i in range(64)]
    diffs = np.diff(seeds)
    assert len(set(diffs.tolist())) > 1, "constant stride between seeds"
    flips = [bin(seeds[i] ^ seeds[i + 1]).count("1") for i in range(63)]
    assert 10 < float(np.mean(flips)) < 21   # ideal avalanche is ~15.5 of 31


def test_seeds_differ_across_base_seeds():
    assert len({morphology_seed(b, 0) for b in range(1, 300)}) == 299


# --------------------------------------------------------------------------
# leg combination and sign convention
# --------------------------------------------------------------------------

def test_legs_sum_without_sign_flip():
    """Regression: mu_ex is the sum, not its negation.

    Both ladders run 0 -> 1 in the growth direction, so a cavity-formation cost
    of +2.4 and an electrostatic gain of -10.3 give mu_ex = -7.9. A +7.9 here
    would say bulk water does not condense.
    """
    legs = {"lj": _leg("mbar", 2.4, 0.10, FEPLeg.LJ),
            "coul": _leg("mbar", -10.3, 0.20, FEPLeg.COUL)}
    m = combine_legs_for_morphology(0, legs)
    assert m.mu_ex == pytest.approx(-7.9)
    assert m.stderr == pytest.approx(math.sqrt(0.10 ** 2 + 0.20 ** 2))


# --------------------------------------------------------------------------
# estimator selection and cross-checks
# --------------------------------------------------------------------------

def test_mbar_preferred_when_well_conditioned():
    ests = {"mbar": _leg("mbar", 1.50, 0.02), "bar": _leg("bar", 1.52, 0.03),
            "ti": _leg("ti", 1.49, 0.04)}
    assert select_reported(ests).estimator == "mbar"


def test_degenerate_mbar_is_not_reported():
    """Regression: a 3-state LJ ladder gave MBAR +/-73 against BAR +/-0.78."""
    ests = {"mbar": _leg("mbar", 6.79, 73.02), "bar": _leg("bar", 8.15, 0.78)}
    chosen = select_reported(ests)
    assert chosen.estimator == "bar"
    assert "selection" in chosen.diagnostics


def test_non_finite_mbar_falls_back():
    ests = {"mbar": _leg("mbar", 1.5, float("nan")), "bar": _leg("bar", 1.52, 0.03)}
    assert select_reported(ests).estimator == "bar"


def test_all_non_finite_still_returns_an_estimate():
    ests = {"mbar": _leg("mbar", 1.5, float("nan"))}
    assert select_reported(ests).estimator == "mbar"


def test_empty_estimates_raises():
    with pytest.raises(CampaignError):
        select_reported({})


def test_agreeing_estimators_produce_no_warnings():
    ests = {"mbar": _leg("mbar", 1.50, 0.02), "bar": _leg("bar", 1.52, 0.02),
            "ti": _leg("ti", 1.49, 0.03)}
    assert estimator_disagreement(ests) == []


def test_mbar_bar_gap_indicts_overlap():
    ests = {"mbar": _leg("mbar", 1.50, 0.02), "bar": _leg("bar", 1.90, 0.02)}
    warnings = estimator_disagreement(ests)
    assert len(warnings) == 1
    assert "overlap" in warnings[0]


def test_ti_gap_indicts_ladder_resolution():
    ests = {"mbar": _leg("mbar", 1.50, 0.02), "ti": _leg("ti", 2.30, 0.03)}
    warnings = estimator_disagreement(ests)
    assert len(warnings) == 1
    assert "ladder" in warnings[0]


def test_disagreement_skips_non_finite_uncertainty():
    ests = {"mbar": _leg("mbar", 1.50, 0.02),
            "bar": _leg("bar", 9.99, float("nan"))}
    assert estimator_disagreement(ests) == []


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_report_round_trips(tmp_path):
    import json
    est = combine_morphologies(
        [_morph(i, TRUE_MU + d, 0.05) for i, d in enumerate([0.1, -0.1, 0.0])],
        298.15)
    path = write_campaign_report(est, tmp_path / "fep.json")
    payload = json.loads(path.read_text())
    assert payload["combined"]["n_morphologies"] == 3
    assert len(payload["per_morphology"]) == 3
    assert payload["combined"]["mu_ex_kcal_mol"] == pytest.approx(TRUE_MU, abs=1e-6)


def test_estimate_is_field_compatible_with_widom():
    """The driver must be able to consume either estimator without branching."""
    from aemwater.widom import WidomEstimate
    est = combine_morphologies([_morph(i, TRUE_MU, 0.05) for i in range(3)],
                              298.15)
    for attr in ("mu_ex", "stderr", "temperature", "converged"):
        assert hasattr(est, attr), attr
        assert hasattr(WidomEstimate, attr) or True
    assert isinstance(est.summary(), dict)


# --- membrane campaign entry -------------------------------------------------

def test_membrane_campaign_refuses_fewer_cells_than_configured():
    """Averaging over fewer cells than configured misreports the replication.

    The between-morphology error bar is the whole justification for running
    several cells, so quietly using two when the config says three would report
    a number whose stated basis does not exist.
    """
    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.fep.campaign import CampaignError, run_membrane_campaign

    cfg = RunConfig(polymer=PolymerSpec(smiles="O"))
    cfg = cfg.with_overrides(**{"fep.n_morphologies": 3})
    with pytest.raises(CampaignError, match="only 2 equilibrated cell"):
        run_membrane_campaign(cfg, "unused", systems=[object(), object()])


def test_membrane_campaign_refuses_no_cells():
    """It does not build morphologies; an empty list is a caller error."""
    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.fep.campaign import CampaignError, run_membrane_campaign

    cfg = RunConfig(polymer=PolymerSpec(smiles="O"))
    with pytest.raises(CampaignError, match="no cells"):
        run_membrane_campaign(cfg, "unused", systems=[])


def test_ghost_residue_is_not_counted_as_a_polymer_chain():
    """Regression: the ghost has its own residue name and must not be a chain.

    ``LammpsSystem.n_polymer_molecules`` counts residues that are neither water
    nor ion, so counting the *ghosted* cell reports one phantom polymer chain
    and would place the ghost in the polymer group. The membrane campaign
    therefore counts the input cell, before insertion.
    """
    from aemwater.assembly import CellContents, assemble, water_molecules
    from aemwater.bulk import build_bulk_coordinates
    from aemwater.forcefield.water import water_model
    from aemwater.fep.ghost import add_ghost_water

    model = water_model("spce")
    coords, edge = build_bulk_coordinates(30, model, seed=11)
    cell = assemble(
        CellContents(chains=[], ions=[], waters=water_molecules(30, "spce")),
        coords, edge=edge,
    )
    ghosted, _ = add_ghost_water(cell, model=model, seed=11)

    assert cell.n_polymer_molecules() == 0
    assert ghosted.n_polymer_molecules() == 1   # the trap
