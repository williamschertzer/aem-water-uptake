"""The saturation criterion compares total water chemical potentials.

mu_w,m - mu_w,b = (mu_ex,m - mu_ex,b) + kT ln(rho_w,m / rho_w,b)

These tests pin the density term, the unbiased stop rule (total gap >= 0,
no tolerance band), the interpolated endpoint and the migration of
excess-only checkpoints.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aemwater import driver
from aemwater.bulk import BulkReference, BulkSettings
from aemwater.config import PolymerSpec, RunConfig
from aemwater.saturation import estimate_saturation_point
from aemwater.widom import (KB_KCAL, SaturationTest, WidomEstimate,
                            water_number_density)

T = 300.0
KT = KB_KCAL * T
RHO_BULK = water_number_density(500, 500 / 0.03342)


def est(mu, stderr=0.05):
    return WidomEstimate(mu, stderr, T, 5, np.zeros(5), 1.0, 1e4)


def test_number_density_counts_the_ghost():
    assert water_number_density(0, 8000.0) == pytest.approx(1 / 8000.0)
    assert water_number_density(99, 3000.0) == pytest.approx(100 / 3000.0)
    with pytest.raises(ValueError):
        water_number_density(10, 0.0)
    with pytest.raises(ValueError):
        water_number_density(-1, 1000.0)


def test_bulk_against_itself_gives_exactly_zero_gap():
    bulk = est(-6.8)
    test = SaturationTest(est(-6.8), bulk, membrane_converged=True,
                          rho_membrane=RHO_BULK, rho_bulk=RHO_BULK, temperature=T)
    assert test.density_term == 0.0
    assert test.difference == 0.0
    assert test.crossed is test.trustworthy


def test_density_term_is_kT_ln_ratio_and_negative_in_a_membrane():
    rho_m = 0.5 * RHO_BULK
    test = SaturationTest(est(-6.8), est(-6.8), membrane_converged=True,
                          rho_membrane=rho_m, rho_bulk=RHO_BULK, temperature=T)
    assert test.density_term == pytest.approx(KT * math.log(0.5))
    assert test.density_term < 0
    assert test.excess_difference == pytest.approx(0.0)
    assert test.difference == pytest.approx(KT * math.log(0.5))
    s = test.summary()
    assert s["density_term_included"] is True
    assert s["difference_kcal_mol"] == pytest.approx(round(KT * math.log(0.5), 4))


def test_excess_only_is_the_legacy_behaviour_when_densities_absent():
    test = SaturationTest(est(-6.0), est(-6.8), membrane_converged=True)
    assert not test.density_term_included
    assert test.difference == pytest.approx(0.8)


def test_crossing_needs_total_gap_nonnegative_not_a_sigma_band():
    # Excess gap +0.2 would have 'saturated' under the old excess-only
    # tolerance rule; the density term (-0.41 at rho_m = rho_b/2) says no.
    test = SaturationTest(est(-6.6, 0.2), est(-6.8, 0.2), tolerance_sigma=1.0,
                          membrane_converged=True, rho_membrane=0.5 * RHO_BULK,
                          rho_bulk=RHO_BULK, temperature=T)
    assert test.excess_difference > 0
    assert test.difference < 0
    assert not test.crossed
    # ... but -0.21 is within one combined sigma (0.28): noise-consistent,
    # which is reported and must not stop the loop.
    assert test.saturated
    assert test.summary()["crossed"] is False


def test_untrustworthy_estimate_never_crosses():
    test = SaturationTest(est(-3.0), est(-6.8), membrane_converged=False,
                          rho_membrane=RHO_BULK, rho_bulk=RHO_BULK, temperature=T)
    assert test.difference > 0
    assert not test.crossed


def test_bulk_reference_density_uses_n_plus_one_over_cell_volume():
    settings = BulkSettings("spce", T, 1.0, 500, 12.0, 1e-5, 1, 1, 1, 1)
    ref = BulkReference(settings, est(-6.8), 0.997, 15000.0, None, method="fep")
    assert ref.water_number_density == pytest.approx(501 / 15000.0)


def rows(points, adequate=True):
    out = []
    for k, (n, gap, se) in enumerate(points):
        out.append(driver.Iteration(
            index=k, n_waters_before=0, n_requested=0, n_inserted=0,
            n_waters_after=n, density=1.0, volume=1.0, lambda_value=0.0,
            water_uptake_pct=0.0, mu_ex=gap, mu_ex_stderr=se, mu_gap=gap,
            saturated=adequate and gap >= 0, sampling_adequate=adequate))
    return out


def test_endpoint_is_the_zero_of_a_linear_gap_not_the_last_state():
    # gap = 0.01 * (n - 250) exactly; loading overshoots to 400 waters.
    ns = [0, 100, 200, 300, 350, 400]
    its = rows([(n, 0.01 * (n - 250), 0.05) for n in ns])
    point = estimate_saturation_point(its, bulk_stderr=0.05)
    assert point.bracketed and point.method == "fit"
    assert point.n_waters == pytest.approx(250.0, abs=1e-6)
    assert point.n_waters_low < 250 < point.n_waters_high
    # Bulk error is common-mode: it alone contributes 1.96*0.05/0.01 = 9.8
    # waters of half-width, so the interval must be at least that wide.
    assert point.n_waters_high - point.n_waters_low >= 2 * 9.8 - 1e-6
    assert point.fit_indices == (1, 2, 3, 4)


def test_endpoint_without_a_lower_point_is_an_upper_bound():
    its = rows([(0, 0.5, 0.05), (50, 0.8, 0.05)])
    point = estimate_saturation_point(its)
    assert not point.bracketed
    assert point.method == "first_crossing"
    assert point.n_waters == 0


def test_no_crossing_no_endpoint():
    assert estimate_saturation_point(rows([(0, -3, .1), (10, -2, .1)])) is None


def test_untrustworthy_points_are_excluded_from_the_fit():
    its = rows([(0, -2.5, .05), (100, -1.5, .05), (200, -0.5, .05),
                (300, 0.5, .05), (400, 1.5, .05)])
    from dataclasses import replace
    its[1] = replace(its[1], mu_gap=+9.0, sampling_adequate=False, saturated=False)
    point = estimate_saturation_point(its)
    assert 1 not in point.fit_indices
    assert point.n_waters == pytest.approx(250.0, abs=1e-6)


def test_non_monotone_fit_falls_back_to_bracketing_pair():
    # Noisy points whose local 4-point slope is negative; the bracketing pair
    # (-0.1 at 200, +0.1 at 300) still defines a root at 250. The point at
    # 400 fell back below zero after the crossing, so it enters the fit.
    its = rows([(0, -3.0, .05), (100, -0.05, .05), (200, -0.1, .05),
                (300, 0.1, .05), (400, -1.0, .05)])
    point = estimate_saturation_point(its)
    assert point.bracketed
    assert point.method == "bracket"
    assert point.n_waters == pytest.approx(250.0)
    assert point.fit_indices == (2, 3)


def test_old_checkpoint_is_migrated_to_total_gap(tmp_path):
    """An excess-only 'saturated' iteration must be re-judged, not trusted."""
    cfg = RunConfig(polymer=PolymerSpec(smiles="CC"))
    settings = BulkSettings("spce", cfg.md.temperature, 1.0, 500, 12.0, 1e-5,
                            1, 1, 1, 1)
    ref = BulkReference(settings, est(-6.8, 0.05), 0.997, 15000.0, None,
                        method="fep")
    # 100 waters in a 5000 A^3 cell: rho_m = 101/5000 ~ 0.60 rho_b.
    it = driver.Iteration(
        index=1, n_waters_before=90, n_requested=10, n_inserted=10,
        n_waters_after=100, density=1.2, volume=5100.0, lambda_value=5.0,
        water_uptake_pct=10.0, mu_ex=-6.6, mu_ex_stderr=0.05, mu_gap=0.2,
        saturated=True, sampling_adequate=True)
    d = tmp_path / "iter_001"
    d.mkdir()
    edge = 5000.0 ** (1 / 3)
    (d / "relaxed.data").write_text(
        "LAMMPS data\n\n1 atoms\n1 atom types\n\n"
        f"0 {edge} xlo xhi\n0 {edge} ylo yhi\n0 {edge} zlo zhi\n\n"
        "Masses\n\n1 12.01\n\nAtoms # full\n\n1 1 1 0.0 1.0 1.0 1.0 0 0 0\n")
    (migrated,) = driver._migrate_gap_definition([it], tmp_path, ref, cfg)
    rho_m = 101 / 5000.0
    term = KB_KCAL * cfg.md.temperature * math.log(rho_m / (501 / 15000.0))
    assert migrated.mu_volume == pytest.approx(5000.0, rel=1e-6)
    assert migrated.mu_gap_excess == pytest.approx(0.2)
    assert migrated.density_term == pytest.approx(term, rel=1e-6)
    assert migrated.mu_gap == pytest.approx(0.2 + term, rel=1e-6)
    assert migrated.saturated is False
