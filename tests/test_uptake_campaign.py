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


def test_uncharged_morphologies_still_average():
    """An undefined lambda must not disqualify a good wt %% measurement.

    A neutral polymer has no IEC, so every morphology reports lambda = NaN. The
    usability gate used to test lambda for finiteness, which made all of them
    unusable and raised "no usable morphology" after the sampling had already
    succeeded. The mass uptake is the measurement here and it averages normally;
    only lambda comes back undefined.
    """
    camp = combine_uptake(
        [_morph(0, 2.1, lam=float("nan")), _morph(1, 2.5, lam=float("nan"))],
        -6.8,
    )
    assert camp.n_usable == 2
    assert camp.water_uptake_pct == pytest.approx(2.3)
    assert math.isnan(camp.lambda_value)
    assert camp.summary()["lambda_waters_per_ionic_group"] is None


def test_undefined_lambda_serialises_as_null_not_nan():
    """`json.dumps` writes a bare `NaN`, which strict JSON readers reject.

    These summaries land in result.json at the end of a multi-hour run, so an
    unparseable file there is the whole run lost to a reporting detail.
    """
    import json

    camp = combine_uptake(
        [_morph(0, 2.1, lam=float("nan")), _morph(1, 2.5, lam=float("nan"))],
        -6.8,
    )
    text = json.dumps(camp.summary())
    assert "NaN" not in text
    assert json.loads(text)["lambda_waters_per_ionic_group"] is None


def test_a_genuinely_failed_morphology_is_still_excluded():
    """Loosening the lambda gate must not let a failure through."""
    camp = [_morph(0, float("nan"), lam=float("nan"), failure="boom"),
            _morph(1, float("nan"), lam=float("nan"))]
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


# --- orchestration -----------------------------------------------------------
#
# These stub prepare_dry_membrane and run_uptake rather than skipping without
# LAMMPS. What is being tested is the wiring -- that each morphology gets a
# different packing seed, that the reference is computed once, that one failure
# does not abort the campaign -- and every one of those is a plain bug that
# would otherwise surface only after hours of cluster time.

import types

import pytest

from aemwater.config import PolymerSpec, RunConfig


@pytest.fixture
def stubbed(monkeypatch):
    """Record what the orchestrator passes to each expensive call."""
    import aemwater.driver as driver
    import aemwater.prepare as prepare
    import aemwater.uptake_campaign as campaign_mod

    calls = {"seeds": [], "references": 0, "fep": [], "reference_resume": []}

    def fake_prepare(config, workdir):
        calls["seeds"].append(config.box.seed)
        calls["fep"].append(
            (len(config.fep.lj_lambdas), len(config.fep.coul_lambdas),
             config.fep.production_steps)
        )
        return types.SimpleNamespace(typed_chains=["chain"])

    def fake_reference(config, workdir, ranks=None, resume=True):
        # `resume` is recorded rather than absorbed by **kwargs: --force must
        # reach the reference, and a stub that swallowed the argument would let
        # that wiring rot while these tests kept passing.
        calls["references"] += 1
        calls["reference_resume"].append(resume)
        estimate = types.SimpleNamespace(mu_ex=-6.83)
        return types.SimpleNamespace(
            mu_ex=estimate, sanity=lambda: [], settings=None, method="fep")

    def fake_uptake(config, workdir, typed_chains, bulk_reference=None, resume=True):
        assert bulk_reference is not None, "trajectory ran without a reference"
        index = len(calls["seeds"]) - 1
        return types.SimpleNamespace(
            n_waters=100 + index, lambda_value=10.0 + index,
            water_uptake_pct=25.0 + index, hydrated_density=1.1,
            stop_reason="saturated", converged=True, iterations=[1, 2, 3],
            bulk_mu_ex=-6.83,
        )

    monkeypatch.setattr(prepare, "prepare_dry_membrane", fake_prepare)
    monkeypatch.setattr(driver, "obtain_bulk_reference", fake_reference)
    monkeypatch.setattr(driver, "run_uptake", fake_uptake)
    return calls, campaign_mod


def _config():
    return RunConfig(polymer=PolymerSpec(smiles="O"))


def test_each_morphology_gets_a_different_packing_seed(stubbed, tmp_path):
    """Same seed twice would give two copies of one packing and a fake spread."""
    calls, mod = stubbed
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=3)
    assert len(calls["seeds"]) == 3
    assert len(set(calls["seeds"])) == 3


def test_bulk_reference_is_computed_once_for_the_whole_campaign(stubbed, tmp_path):
    """Per-morphology references would put reference noise into the spread."""
    calls, mod = stubbed
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=3)
    assert calls["references"] == 1


def test_screening_resolution_reaches_the_trajectories(stubbed, tmp_path):
    """The preset is pointless if the loop still runs the production ladder."""
    calls, mod = stubbed
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=2, screening=True)
    assert all(n_lj == 7 and n_co == 7 and steps == 150_000
               for n_lj, n_co, steps in calls["fep"])


def test_production_resolution_is_passed_through_unchanged(stubbed, tmp_path):
    calls, mod = stubbed
    prod = _config()
    mod.run_uptake_campaign(prod, tmp_path, n_morphologies=2, screening=False)
    assert all(n_lj == len(prod.fep.lj_lambdas) and steps == prod.fep.production_steps
               for n_lj, _, steps in calls["fep"])


def test_one_failed_morphology_does_not_abort_the_campaign(
        stubbed, tmp_path, monkeypatch):
    """Hours of successful trajectories must not be lost to one crash."""
    calls, mod = stubbed
    import aemwater.driver as driver

    real = driver.run_uptake

    def flaky(config, workdir, typed_chains, **kw):
        if len(calls["seeds"]) == 2:
            raise RuntimeError("LAMMPS segfault")
        return real(config, workdir, typed_chains, **kw)

    monkeypatch.setattr(driver, "run_uptake", flaky)
    result = mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=3)

    assert len(result.per_morphology) == 3
    assert result.n_usable == 2
    failed = [m for m in result.per_morphology if not m.usable]
    assert "LAMMPS segfault" in failed[0].failure


def test_campaign_writes_a_report_with_every_morphology(stubbed, tmp_path):
    """A reader must be able to see the spread, not only the mean."""
    import json

    calls, mod = stubbed
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=3)
    payload = json.loads((tmp_path / "uptake_campaign.json").read_text())
    assert len(payload["per_morphology"]) == 3
    assert payload["n_morphologies_usable"] == 3


def test_zero_morphologies_is_rejected(stubbed, tmp_path):
    calls, mod = stubbed
    with pytest.raises(mod.UptakeCampaignError, match="must be >= 1"):
        mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=0)


def test_completed_morphology_does_not_rebuild_its_dry_membrane(
        stubbed, tmp_path, monkeypatch):
    """A requeued campaign must not re-anneal finished morphologies.

    The anneal plus GAFF2 charge derivation dominates per-morphology cost. An
    earlier version called prepare_dry_membrane unconditionally, so a requeue
    paid that cost again for every already-finished morphology -- silently,
    because the uptake loop underneath *did* resume and reported success. On the
    preemptible queue the campaign script targets, that is the difference
    between resumable and unable to finish.
    """
    calls, mod = stubbed
    import aemwater.prepare as prepare

    builds = []
    real_obtain = prepare.obtain_dry_membrane

    def counting_obtain(config, workdir, *, resume=True):
        from pathlib import Path
        dry_data = Path(workdir) / "dry" / "dry.data"
        rebuilt = not (resume and dry_data.exists())
        builds.append(rebuilt)
        # Materialise the checkpoint the way a real run would.
        dry_data.parent.mkdir(parents=True, exist_ok=True)
        dry_data.touch()
        return ["chain"], not rebuilt

    monkeypatch.setattr(prepare, "obtain_dry_membrane", counting_obtain)

    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=2)
    assert builds == [True, True], "first pass must build both"

    builds.clear()
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=2)
    assert builds == [False, False], "second pass must reuse, not rebuild"


def test_force_rebuilds_even_when_a_checkpoint_exists(stubbed, tmp_path, monkeypatch):
    """resume=False must actually reach the dry-membrane decision."""
    calls, mod = stubbed
    import aemwater.prepare as prepare

    rebuilt = []

    def counting_obtain(config, workdir, *, resume=True):
        from pathlib import Path
        dry_data = Path(workdir) / "dry" / "dry.data"
        rebuilt.append(not (resume and dry_data.exists()))
        dry_data.parent.mkdir(parents=True, exist_ok=True)
        dry_data.touch()
        return ["chain"], False

    monkeypatch.setattr(prepare, "obtain_dry_membrane", counting_obtain)

    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=1)
    rebuilt.clear()
    calls["reference_resume"].clear()
    mod.run_uptake_campaign(_config(), tmp_path, n_morphologies=1, resume=False)
    assert rebuilt == [True]

    # ...and the same flag must reach the bulk reference. It is the most
    # expensive stage and the one where reusing stale work is worst: a membrane
    # recomputed under new settings against a cached reservoir measured under
    # the old ones compares two protocols, and the saturation criterion is the
    # difference between them.
    assert calls["reference_resume"] == [False], (
        "--force did not reach the bulk reference; it would be read from cache "
        "while the membrane was recomputed"
    )
