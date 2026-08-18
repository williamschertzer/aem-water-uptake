"""Widom estimator: correctness against an analytic case, and its guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aemwater.widom import (
    KB_KCAL,
    SaturationTest,
    WidomError,
    WidomEstimate,
    effective_sample_size,
    estimate_from_series,
    mu_ex_from_boltzmann,
    read_widom_file,
)

T = 300.0


def test_mu_ex_inverts_the_boltzmann_factor():
    """A known mu_ex must round-trip through the estimator exactly."""
    mu = -5.0
    mean_exp = math.exp(-mu / (KB_KCAL * T))
    assert mu_ex_from_boltzmann(mean_exp, T) == pytest.approx(mu, abs=1e-9)


def test_unit_boltzmann_average_gives_zero():
    assert mu_ex_from_boltzmann(1.0, T) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_nonpositive_boltzmann_average_is_rejected(bad):
    with pytest.raises(WidomError):
        mu_ex_from_boltzmann(bad, T)


def test_gaussian_insertion_energies_recover_the_analytic_result():
    """For dU ~ N(m, s), mu_ex = m - s^2 / (2kT) exactly.

    This is the only closed-form check available for a Widom estimator, so it is
    the test that says the estimator is right rather than merely self-consistent.
    """
    m, s = -2.0, 1.0          # small sigma keeps the average well-sampled
    rng = np.random.default_rng(0)
    dU = rng.normal(m, s, 400_000)
    w = np.exp(-dU / (KB_KCAL * T))
    series = w.reshape(400, 1000).mean(axis=1)
    est = estimate_from_series(series, T, n_blocks=8)
    analytic = m - s ** 2 / (2 * KB_KCAL * T)
    assert est.mu_ex == pytest.approx(analytic, abs=0.05)


def test_effective_sample_size_equals_n_for_equal_weights():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50.0)


def test_effective_sample_size_collapses_when_one_weight_dominates():
    w = np.concatenate([[1e6], np.ones(999)])
    assert effective_sample_size(w) < 2.0


def test_effective_sample_size_of_nothing_is_zero():
    assert effective_sample_size(np.zeros(10)) == 0.0


def test_skewed_average_is_flagged_unconverged():
    """A mu_ex carried by a couple of lucky insertions must not be trusted."""
    series = np.concatenate([[1e8], np.full(199, 1e-12)])
    est = estimate_from_series(series, T, n_blocks=4)
    assert not est.converged


def test_all_zero_series_raises_actionable_error():
    with pytest.raises(WidomError, match="insertions_per_call"):
        estimate_from_series(np.zeros(100), T, n_blocks=4)


def test_empty_series_is_rejected():
    with pytest.raises(WidomError, match="no finite"):
        estimate_from_series(np.array([np.nan, np.inf]), T)


def test_block_count_is_capped_by_sample_count():
    est = estimate_from_series(np.full(3, 0.5), T, n_blocks=10)
    assert est.n_blocks <= 3


def test_stderr_shrinks_with_more_blocks_of_consistent_data():
    rng = np.random.default_rng(1)
    series = np.exp(-rng.normal(-2.0, 0.5, 8000) / (KB_KCAL * T))
    few = estimate_from_series(series[:1000], T, n_blocks=4)
    many = estimate_from_series(series, T, n_blocks=8)
    assert many.stderr < few.stderr


def _estimate(mu, err):
    return WidomEstimate(mu_ex=mu, stderr=err, temperature=T, n_blocks=5,
                         block_values=np.full(5, mu), mean_boltzmann=1.0,
                         effective_samples=100.0)


def test_membrane_more_favourable_than_bulk_is_not_saturated():
    t = SaturationTest(_estimate(-9.0, 0.1), _estimate(-6.0, 0.1))
    assert t.difference == pytest.approx(-3.0)
    assert not t.saturated
    assert t.trustworthy


def test_equal_potentials_count_as_saturated():
    t = SaturationTest(_estimate(-6.0, 0.1), _estimate(-6.0, 0.1))
    assert t.saturated


def test_marginal_difference_within_noise_counts_as_saturated():
    """The stop criterion must be resolvable, not a coin flip on noise."""
    t = SaturationTest(_estimate(-6.2, 0.2), _estimate(-6.0, 0.2), tolerance_sigma=2.0)
    assert t.combined_stderr == pytest.approx(math.hypot(0.2, 0.2))
    assert t.saturated


def test_difference_beyond_the_noise_threshold_is_not_saturated():
    t = SaturationTest(_estimate(-7.0, 0.1), _estimate(-6.0, 0.1), tolerance_sigma=2.0)
    assert not t.saturated


def test_unconverged_inputs_make_the_test_untrustworthy():
    bad = WidomEstimate(-6.0, 0.1, T, 5, np.full(5, -6.0), 1.0, effective_samples=2.0)
    t = SaturationTest(bad, _estimate(-6.0, 0.1))
    assert not t.trustworthy


def test_read_widom_file_recomputes_from_boltzmann_factors(tmp_path):
    """mu_ex must come from log<exp>, not from averaging the per-row mu column."""
    p = tmp_path / "mu.dat"
    # Two rows whose Boltzmann factors differ by orders of magnitude: averaging
    # the mu column and taking -kT log of the mean factor give different answers.
    b1, b2 = 1.0, 100.0
    mu1 = -KB_KCAL * T * math.log(b1)
    mu2 = -KB_KCAL * T * math.log(b2)
    p.write_text(
        "# Time-averaged data\n"
        "# TimeStep v_mu v_boltz v_vol\n"
        f"1000 {mu1} {b1} 8000.0\n"
        f"2000 {mu2} {b2} 8000.0\n"
    )
    est = read_widom_file(p, T, n_blocks=2)
    expected = -KB_KCAL * T * math.log((b1 + b2) / 2)
    assert est.mu_ex == pytest.approx(expected, abs=1e-9)
    assert est.mu_ex != pytest.approx((mu1 + mu2) / 2, abs=1e-3)
    assert est.volume == pytest.approx(8000.0)


def test_read_widom_file_rejects_an_empty_file(tmp_path):
    p = tmp_path / "empty.dat"
    p.write_text("# only a header\n")
    with pytest.raises(WidomError, match="no data rows"):
        read_widom_file(p, T)
