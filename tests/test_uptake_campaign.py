"""Water uptake averaged over independently equilibrated morphologies.

The failure this module exists to prevent is a confident-looking error bar on a
single packing. The tests below therefore concentrate on two things: that the
interval covers at the two or three morphologies anyone will actually run, and
that a trajectory which never saturated cannot enter the average.
"""

import math

import numpy as np
import pytest

from aemwater.uptake_campaign import (
    MorphologyUptake,
    UptakeCampaignError,
    combine_uptake,
    morphology_box_seed,
)


def _morph(index, pct, lam=10.0, stop="saturated", failure=None):
    return MorphologyUptake(
        index=index, seed=1000 + index, workdir=".", n_waters=120,
        lambda_value=lam, water_uptake_pct=pct, hydrated_density=1.12,
        stop_reason=stop, converged=True, n_iterations=7, failure=failure,
    )


@pytest.mark.parametrize("m_count", [2, 3, 5])
def test_uptake_interval_covers_at_small_morphology_counts(m_count):
    """95% must mean 95% at the M anyone will actually run.

    The between-morphology scatter is estimated from M-1 degrees of freedom, so
    a normal quantile would undercover badly at M=2. This is the same reasoning
    as the mu_ex campaign's interval and the same fix.
    """
    truth, sigma, trials = 25.0, 3.0, 4000
    rng = np.random.default_rng(99)
    covered = 0
    for _ in range(trials):
        vals = rng.normal(truth, sigma, m_count)
        camp = combine_uptake([_morph(i, v) for i, v in enumerate(vals)], -6.8)
        lo, hi = camp.ci95
        covered += lo <= truth <= hi
    assert 0.93 <= covered / trials <= 0.97


def test_normal_quantile_would_undercover_at_two_morphologies():
    """Justifies the Student-t quantile rather than 1.96."""
    truth, sigma, trials = 25.0, 3.0, 4000
    rng = np.random.default_rng(5)
    covered = 0
    for _ in range(trials):
        vals = rng.normal(truth, sigma, 2)
        camp = combine_uptake([_morph(i, v) for i, v in enumerate(vals)], -6.8)
        half = 1.96 * camp.stderr
        covered += abs(camp.water_uptake_pct - truth) <= half
    assert covered / trials < 0.75


def test_unsaturated_trajectory_is_excluded_from_the_average():
    """A max_iterations stop is a lower bound, not a measurement.

    Averaging it in biases the campaign low with no sign of it in the error bar,
    which is precisely the silent failure this layer is meant to prevent.
    """
    camp = combine_uptake(
        [_morph(0, 25.0), _morph(1, 12.0, stop="max_iterations"), _morph(2, 26.0)],
        -6.8,
    )
    assert camp.n_usable == 2
    assert camp.water_uptake_pct == pytest.approx(25.5)


def test_failed_trajectory_does_not_discard_the_successful_ones():
    """One crashed packing must not throw away the CPU spent on the others."""
    camp = combine_uptake(
        [_morph(0, 24.0), _morph(1, float("nan"), failure="LAMMPS died"),
         _morph(2, 26.0)],
        -6.8,
    )
    assert camp.n_usable == 2
    assert camp.water_uptake_pct == pytest.approx(25.0)


def test_single_usable_morphology_reports_no_uncertainty():
    """With one packing there is no spread, and the result must say so.

    Reporting 0.0 here would be far worse than reporting nan: it would look like
    an extremely precise measurement.
    """
    camp = combine_uptake([_morph(0, 25.0)], -6.8)
    assert math.isnan(camp.stderr)
    assert camp.ci95 == (float("-inf"), float("inf"))
    assert "note" in camp.diagnostics


def test_no_usable_morphology_raises_rather_than_returning_a_number():
    camp = [_morph(0, 5.0, stop="max_iterations"),
            _morph(1, float("nan"), failure="boom")]
    with pytest.raises(UptakeCampaignError, match="no usable morphology"):
        combine_uptake(camp, -6.8)


def test_geometric_saturation_counts_as_usable():
    """Running out of cavities is a real endpoint, not a failure to converge."""
    camp = combine_uptake(
        [_morph(0, 25.0, stop="geometric_saturation"),
         _morph(1, 27.0, stop="saturated")],
        -6.8,
    )
    assert camp.n_usable == 2


def test_packing_seeds_are_decorrelated_not_merely_distinct():
    """Regression: a first version had a constant stride of 40503.

    This seed chooses where the chains go, so it defines the morphology. Nearby
    integer seeds can give correlated early RNG output, which would make the
    'independent' packings less independent than the error bar assumes.
    """
    seeds = [morphology_box_seed(20260817, i) for i in range(64)]
    assert len(set(seeds)) == 64
    strides = {b - a for a, b in zip(seeds, seeds[1:])}
    assert len(strides) > 32


def test_spread_is_reported_so_a_reader_can_see_morphology_dependence():
    """The point of replication is visible disagreement, not just a mean."""
    camp = combine_uptake([_morph(0, 20.0), _morph(1, 30.0)], -6.8)
    assert camp.diagnostics["spread_wt_pct"] == pytest.approx(10.0)
    assert camp.diagnostics["water_uptake_per_morphology"] == [20.0, 30.0]
