"""MBAR, BAR and TI.

These estimators are tested against a system with an *analytic* answer rather
than against each other. Coupled harmonic oscillators give an exactly known free
energy difference, so a bug that biases all three consistently -- a sign error, a
kT factor, a mis-set beta -- is caught, which a mutual-consistency test would
miss entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from aemwater.fep.estimators import (
    bar_estimate,
    combine_legs,
    dudl_from_finite_differences,
    mbar_estimate,
    read_fep_columns,
    ti_estimate,
)
from aemwater.fep.rerun import EnergyMatrix
from aemwater.fep.schedule import FEPLeg

pytest.importorskip("pymbar")

N_SAMPLES = 4000
SPRINGS = np.array([1.0, 2.0, 4.0, 8.0, 16.0])


def _harmonic_matrix(springs=SPRINGS, n=N_SAMPLES, seed=0):
    """U_k(x) = 0.5*K_k*x^2 in reduced units, sampled exactly.

    The reduced free energy of a 1D harmonic well is -0.5*ln(2*pi/K), so the
    ladder's total dF is known in closed form.
    """
    rng = np.random.default_rng(seed)
    lambdas = tuple(np.linspace(0.0, 1.0, len(springs)))
    u_kn = np.zeros((len(springs), len(springs) * n))
    for j, kj in enumerate(springs):
        x = rng.normal(0.0, 1.0 / np.sqrt(kj), n)
        for k, kk in enumerate(springs):
            u_kn[k, j * n : (j + 1) * n] = 0.5 * kk * x**2
    exact = -0.5 * np.log(2 * np.pi / springs)
    matrix = EnergyMatrix(
        u_kn=u_kn, N_k=np.full(len(springs), n), lambdas=lambdas,
        leg=FEPLeg.LJ, kT=1.0,
    )
    return matrix, float(exact[-1] - exact[0])


# ----------------------------------------------------------- analytic accuracy --


def test_mbar_recovers_the_analytic_free_energy():
    matrix, exact = _harmonic_matrix()
    est = mbar_estimate(matrix)
    assert est.delta_f == pytest.approx(exact, abs=5 * est.stderr)
    assert est.stderr < 0.05
    assert est.diagnostics["min_overlap"] > 0.1


def test_bar_recovers_the_analytic_free_energy():
    matrix, exact = _harmonic_matrix()
    est = bar_estimate(matrix)
    assert est.delta_f == pytest.approx(exact, abs=5 * est.stderr)
    assert est.diagnostics["usable"]


def test_bar_uncertainty_is_a_lower_bound_not_a_conservative_one():
    """BAR's quadrature sum under-counts, and that must stay documented.

    Neighbouring pairs share samples -- state k enters both (k-1,k) and (k,k+1)
    -- so adding per-pair errors in quadrature assumes an independence that does
    not hold. The observable consequence is that BAR reports a *tighter* error
    than MBAR while using strictly less information, which is why MBAR is the
    reported estimator. Pinned here so nobody 'fixes' the tighter number by
    trusting it.
    """
    matrix, _ = _harmonic_matrix()
    bar, mbar = bar_estimate(matrix), mbar_estimate(matrix)
    assert bar.stderr < mbar.stderr
    # Both must still bracket the truth; it is precision that is understated,
    # not accuracy.
    _, exact = _harmonic_matrix()
    assert bar.delta_f == pytest.approx(exact, abs=5 * mbar.stderr)


def test_mbar_is_unbiased_across_seeds():
    """Averaging over independent campaigns must converge on the exact answer."""
    errs = []
    for seed in range(6):
        matrix, exact = _harmonic_matrix(seed=seed)
        errs.append(mbar_estimate(matrix).delta_f - exact)
    # Mean error must be far smaller than the individual scatter -- i.e. random,
    # not systematic.
    assert abs(np.mean(errs)) < 0.5 * np.std(errs) + 0.01


# --------------------------------------------------------------- TI quadrature --


def test_ti_uses_spacing_aware_quadrature():
    """The production ladder is deliberately non-uniform.

    Integrating 3x^2 over the default LJ spacing: trapezoid with explicit
    lambdas errs by 4.6e-3, a naive mean of the integrand by 3.7e-2. This test
    pins the eight-fold difference so a refactor cannot quietly drop the
    lambda argument.
    """
    lam = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    integrand = 3 * lam**2                      # exact integral over 0..1 is 1
    est = ti_estimate(lam, integrand, np.zeros_like(lam), leg=FEPLeg.LJ)
    assert est.delta_f == pytest.approx(1.0, abs=1e-2)
    naive = float(np.mean(integrand))
    assert abs(est.delta_f - 1.0) < abs(naive - 1.0) / 4
    assert est.diagnostics["uniform_spacing"] is False


def test_ti_error_falls_with_ladder_density():
    """TI's error is resolution, not sampling: it must shrink as n grows."""
    errs = []
    for n in (5, 9, 17, 33):
        lam = np.linspace(0, 1, n)
        springs = 1.0 + 15.0 * lam
        # <dU/dlambda> for U = 0.5*K(lam)*x^2 with K linear in lambda.
        dudl = 0.5 * 15.0 / springs
        exact = float(-0.5 * np.log(2 * np.pi / springs[-1])
                      + 0.5 * np.log(2 * np.pi / springs[0]))
        est = ti_estimate(lam, dudl, np.zeros_like(lam), leg=FEPLeg.LJ)
        errs.append(abs(est.delta_f - exact))
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < errs[0] / 5


def test_ti_reports_curvature():
    """Curvature is what trapezoid gets wrong; a coarse ladder must be visible."""
    lam = np.linspace(0, 1, 9)
    smooth = ti_estimate(lam, lam, np.zeros_like(lam), leg=FEPLeg.LJ)
    kinked = ti_estimate(lam, np.exp(8 * lam), np.zeros_like(lam), leg=FEPLeg.LJ)
    assert smooth.diagnostics["max_abs_second_difference"] < 1e-9
    assert kinked.diagnostics["max_abs_second_difference"] > 100


def test_ti_propagates_per_state_errors_with_trapezoid_weights():
    lam = np.array([0.0, 0.5, 1.0])
    est = ti_estimate(lam, [1.0, 1.0, 1.0], [0.1, 0.1, 0.1], leg=FEPLeg.LJ)
    # Weights are 0.25, 0.5, 0.25; quadrature sum of (w*err)^2.
    expected = np.sqrt((0.25 * 0.1) ** 2 + (0.5 * 0.1) ** 2 + (0.25 * 0.1) ** 2)
    assert est.stderr == pytest.approx(expected)


def test_ti_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="matching shapes"):
        ti_estimate([0.0, 1.0], [1.0], [0.1, 0.1], leg=FEPLeg.LJ)
    with pytest.raises(ValueError, match="at least two"):
        ti_estimate([0.0], [1.0], [0.1], leg=FEPLeg.LJ)


# ---------------------------------------------------- finite-difference dU/dl --


def test_central_difference_beats_one_sided():
    """Both are supported, but the central form must be the more accurate one."""
    rng = np.random.default_rng(1)
    # dU(lambda+d) - dU(lambda-d) for a quadratic U(lambda): exact derivative 2.
    delta = 0.02
    noise = rng.normal(0, 1e-9, 500)
    plus = 2 * delta + delta**2 + noise
    minus = -2 * delta + delta**2 + noise
    central, _ = dudl_from_finite_differences(plus, minus, delta)
    one_sided, _ = dudl_from_finite_differences(plus, None, delta)
    assert abs(central - 2.0) < abs(one_sided - 2.0)
    assert central == pytest.approx(2.0, abs=1e-6)


def test_one_sided_reverse_difference_has_the_right_sign():
    delta = 0.05
    minus = np.full(100, -0.5)          # dU going down in lambda
    val, _ = dudl_from_finite_differences(None, minus, delta)
    assert val == pytest.approx(0.5 / delta)


def test_finite_difference_rejects_bad_input():
    a = np.zeros(10)
    with pytest.raises(ValueError, match="delta must be positive"):
        dudl_from_finite_differences(a, a, 0.0)
    with pytest.raises(ValueError, match="at least one"):
        dudl_from_finite_differences(None, None, 0.01)
    with pytest.raises(ValueError, match="same shape"):
        dudl_from_finite_differences(a, np.zeros(9), 0.01)


# ----------------------------------------------------------------- decorrelation --


def test_correlated_samples_widen_the_uncertainty():
    """Correlated frames fed in as independent give an optimistic error bar.

    The free energy barely moves; the *uncertainty* is what decorrelation fixes,
    so that is what this asserts.
    """
    matrix, _ = _harmonic_matrix(n=2000)
    # Duplicate every frame: information unchanged, naive N doubled.
    dup = EnergyMatrix(
        u_kn=np.repeat(matrix.u_kn, 2, axis=1),
        N_k=matrix.N_k * 2,
        lambdas=matrix.lambdas, leg=matrix.leg, kT=matrix.kT,
    )
    honest = mbar_estimate(dup, subsample=True)
    naive = mbar_estimate(dup, subsample=False)
    assert naive.stderr < honest.stderr
    assert honest.n_effective < naive.n_effective
    assert max(honest.diagnostics["statistical_inefficiency"]) > 1.5


# ------------------------------------------------------------------- guards --


def test_bar_flags_pairs_whose_uncertainty_underflows():
    """One absurd work value must not silently poison the total.

    exp(-w) is exactly zero above ~709 kT, and BAR's variance formula then
    returns nan. The estimate must report that rather than pass a nan onward as
    though it were an error bar.
    """
    matrix, _ = _harmonic_matrix(n=200)
    poisoned = matrix.u_kn.copy()
    # The bump must land in the block BAR actually reads for that pair: w_F for
    # pair (k, k+1) is state k+1's energy evaluated on state k's *own* samples.
    # Poisoning an arbitrary column is invisible -- it belongs to another state's
    # block and no neighbouring pair consults it.
    offsets = np.concatenate([[0], np.cumsum(matrix.N_k)])
    last = len(matrix.N_k) - 1
    poisoned[last, offsets[last - 1]] += 5000.0
    bad = EnergyMatrix(
        u_kn=poisoned, N_k=matrix.N_k, lambdas=matrix.lambdas,
        leg=matrix.leg, kT=matrix.kT,
    )
    est = bar_estimate(bad)
    assert est.diagnostics["usable"] is False
    assert est.diagnostics["pairs_without_uncertainty"] == [(last - 1, last)]
    assert any(p["n_extreme_work"] > 0 for p in est.diagnostics["per_pair"])
    # The healthy pairs must be unaffected -- one bad pair, not a poisoned run.
    healthy = [p for p in est.diagnostics["per_pair"] if p["pair"] != (last - 1, last)]
    assert all(np.isfinite(p["stderr"]) for p in healthy)


def test_read_fep_columns_drops_frame_zero(tmp_path):
    path = tmp_path / "fep.dat"
    path.write_text(
        "# fixed-lambda FEP: lj leg\n"
        "# step dU_fwd dU_ti_plus\n"
        " 0 1.0 -1.0\n 100 2.0 -2.0\n 200 3.0 -3.0\n"
    )
    cols = read_fep_columns(path)
    assert set(cols) == {"step", "dU_fwd", "dU_ti_plus"}
    assert cols["dU_fwd"].tolist() == [2.0, 3.0]      # frame 0 gone


def test_read_fep_columns_strips_lammps_default_prefix(tmp_path):
    """A title2/title3 regression yields `v_`-prefixed names; degrade, not crash."""
    path = tmp_path / "fep.dat"
    path.write_text("# hdr\n# TimeStep v_dU_fwd\n 0 1.0\n 100 2.0\n")
    assert "dU_fwd" in read_fep_columns(path)


def test_read_fep_columns_rejects_a_header_mismatch(tmp_path):
    path = tmp_path / "fep.dat"
    path.write_text("# hdr\n# step dU_fwd\n 0 1.0 2.0\n 100 1.0 2.0\n")
    with pytest.raises(ValueError, match="columns"):
        read_fep_columns(path)


def test_read_fep_columns_rejects_a_single_frame(tmp_path):
    path = tmp_path / "fep.dat"
    path.write_text("# hdr\n# step dU_fwd\n 0 1.0\n")
    with pytest.raises(ValueError, match="frame"):
        read_fep_columns(path)


# ------------------------------------------------------------- leg combination --


def test_legs_add_and_errors_add_in_quadrature():
    from aemwater.fep.estimators import LegEstimate, LegResult

    def leg(name, value, err):
        e = LegEstimate(estimator="mbar", leg=name, delta_f=value, stderr=err,
                        n_effective=100.0)
        return LegResult(leg=name, estimates=(e,), reported=e)

    total, err = combine_legs([leg(FEPLeg.LJ, 6.0, 0.3), leg(FEPLeg.COUL, -12.0, 0.4)])
    assert total == pytest.approx(-6.0)
    assert err == pytest.approx(0.5)


def test_leg_result_spread_reports_estimator_disagreement():
    from aemwater.fep.estimators import LegEstimate, LegResult

    def e(name, value):
        return LegEstimate(estimator=name, leg=FEPLeg.LJ, delta_f=value,
                           stderr=0.1, n_effective=100.0)

    mbar, bar, ti = e("mbar", 6.0), e("bar", 6.1), e("ti", 6.9)
    res = LegResult(leg=FEPLeg.LJ, estimates=(mbar, bar, ti), reported=mbar)
    assert res.spread == pytest.approx(0.9)
    assert res.by_name("ti") is ti
    assert res.by_name("nonexistent") is None
