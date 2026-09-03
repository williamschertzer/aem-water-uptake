"""Uptake loop: batch sizing, hydration accounting, state readback."""

from __future__ import annotations

import dataclasses
import math
import pathlib

import numpy as np
import pytest

from aemwater.config import MDSpec, WidomSpec
from aemwater.driver import (
    M_WATER,
    DriverError,
    Iteration,
    UptakeResult,
    _read_final_state,
    _resume_data_file,
    bulk_n_waters,
    hydration_number,
    next_batch_size,
    update_failed_batches,
    settle_steps,
    water_uptake_percent,
)


# ------------------------------------------------------------- batch sizing --
def test_first_batch_is_sized_from_the_ionic_group_count():
    """With no measurement yet, lambda ~ 1 is the safe first step."""
    assert next_batch_size(0, 8, mu_gap=None, stderr=0.0) == 8


def test_batch_scales_with_current_content():
    small = next_batch_size(40, 8, mu_gap=-5.0, stderr=0.1)
    large = next_batch_size(400, 8, mu_gap=-5.0, stderr=0.1)
    assert large > small


def test_batch_shrinks_monotonically_as_the_gap_closes():
    """Overshooting the endpoint wastes the most expensive iterations."""
    sizes = [next_batch_size(100, 8, mu_gap=-g, stderr=0.2) for g in (5.0, 0.8, 0.2)]
    assert sizes[0] >= sizes[1] >= sizes[2]
    assert sizes[0] > sizes[2]


def test_batch_respects_its_bounds():
    assert next_batch_size(10_000, 8, -5.0, 0.1, max_batch=50) == 50
    assert next_batch_size(1, 1, -0.001, 1.0, min_batch=3) >= 3


def test_gap_measured_in_sigma_not_absolute_units():
    """A 0.3 kcal/mol gap is decisive at sigma=0.02 and noise at sigma=0.5."""
    precise = next_batch_size(100, 8, mu_gap=-0.3, stderr=0.02)
    noisy = next_batch_size(100, 8, mu_gap=-0.3, stderr=0.5)
    assert precise > noisy


def test_zero_stderr_does_not_divide_by_zero():
    assert next_batch_size(100, 8, mu_gap=-1.0, stderr=0.0) > 0


def test_uptake_uses_the_config_aware_composition_builder():
    """Preparation and uptake must agree on polyatomic counterion inventory."""
    import inspect

    from aemwater import driver

    source = inspect.getsource(driver.run_uptake)
    assert "composition_from_config(config)" in source
    assert "build_composition(" not in source


def test_consecutive_geometric_shortfalls_accumulate_and_full_batch_resets():
    failures = update_failed_batches(0, requested=20, inserted=12)
    assert failures == 1
    failures = update_failed_batches(failures, requested=10, inserted=0)
    assert failures == 2
    assert update_failed_batches(failures, requested=3, inserted=3) == 0


# ----------------------------------------------------------- uptake measures --
def test_lambda_is_waters_per_ionic_group():
    assert hydration_number(120, 8) == pytest.approx(15.0)


def test_lambda_is_nan_without_ionic_groups():
    """An uncharged composition has no lambda, but it is not an error.

    Polyethylene is a legitimate input -- the hydrophobic control that says how
    much of a real AEM's uptake comes from the ionic groups rather than from
    free volume. lambda = 0/0 is undefined for it, so this returns NaN and lets
    the mass uptake and the saturation criterion (neither of which references
    the ionic-group count) carry the result. Raising here discarded a completed
    run over an inapplicable reporting convention.
    """
    assert math.isnan(hydration_number(10, 0))
    assert math.isnan(hydration_number(0, 0))


def test_lambda_agrees_with_the_composition_definition():
    """The two lambda implementations must not disagree on any input.

    `SystemComposition.lambda_from_n_water` returned NaN for an uncharged
    composition while `hydration_number` raised on the same input; the run
    reached the raising one after all the sampling had succeeded.
    """
    from aemwater.chemistry import build_composition

    charged = build_composition(
        "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1", n_chains=2, chain_length=4)
    neutral = build_composition("[*]CC([*])c1ccccc1", n_chains=2, chain_length=4)

    for comp in (charged, neutral):
        theirs = comp.lambda_from_n_water(40)
        ours = hydration_number(40, comp.total_ionic_groups)
        assert (math.isnan(theirs) and math.isnan(ours)) or theirs == ours


def test_neutral_composition_still_has_a_mass_uptake():
    """The measurement survives the undefined lambda."""
    assert math.isnan(hydration_number(120, 0))
    assert water_uptake_percent(120, 6895.8) > 0


def test_mass_uptake_matches_the_definition():
    n, dry = 120, 6895.8
    assert water_uptake_percent(n, dry) == pytest.approx(100 * n * M_WATER / dry)


def test_the_two_conventions_are_consistent():
    """lambda and wt% must describe the same system."""
    n_ionic, dry_mass, n_waters = 8, 6895.8, 120
    lam = hydration_number(n_waters, n_ionic)
    wt = water_uptake_percent(n_waters, dry_mass)
    iec = 1000.0 * n_ionic / dry_mass          # mmol/g
    assert wt == pytest.approx(lam * iec * M_WATER / 10.0, rel=1e-9)


def test_negative_dry_mass_is_rejected():
    with pytest.raises(DriverError):
        water_uptake_percent(10, 0.0)


# ------------------------------------------------------------ derived sizing --
def test_bulk_box_length_sets_the_molecule_count():
    """The finite-size error scales with edge/cutoff, so the user sets the edge."""
    n = bulk_n_waters(WidomSpec(bulk_box_length=25.0))
    volume_cm3 = (25.0e-8) ** 3
    assert n == pytest.approx(0.997 * 6.02214076e23 * volume_cm3 / M_WATER, rel=0.01)


def test_bulk_box_never_smaller_than_the_minimum():
    assert bulk_n_waters(WidomSpec(bulk_box_length=5.0)) >= 64


def test_settling_is_a_fraction_of_the_npt_window():
    assert settle_steps(MDSpec(relax_npt_steps=100_000)) == 10_000
    assert settle_steps(MDSpec(relax_npt_steps=100)) >= 2000


# --------------------------------------------------------------- state readback --
def _data_file(tmp_path, edge=20.0, images=("0 0 0", "0 0 0", "0 0 0")):
    p = tmp_path / "s.data"
    p.write_text(
        "LAMMPS data\n\n3 atoms\n2 atom types\n\n"
        f"0.0 {edge} xlo xhi\n0.0 {edge} ylo yhi\n0.0 {edge} zlo zhi\n\n"
        "Masses\n\n1 15.9994\n2 1.008\n\n"
        "Atoms # full\n\n"
        f"1 1 1 -0.8476 1.0 2.0 3.0 {images[0]}\n"
        f"2 1 2 0.4238 1.5 2.5 3.5 {images[1]}\n"
        f"3 1 2 0.4238 0.5 1.5 2.5 {images[2]}\n"
    )
    return p


def test_state_readback_recovers_geometry_and_box(tmp_path):
    coords, elements, edge = _read_final_state(_data_file(tmp_path, 20.0))
    assert edge == pytest.approx(20.0)
    assert coords.shape == (3, 3)
    assert np.allclose(coords[0], [1.0, 2.0, 3.0])


def test_elements_are_recovered_from_masses(tmp_path):
    """The data file carries no element symbols, only masses."""
    _, elements, _ = _read_final_state(_data_file(tmp_path))
    assert elements == ["O", "H", "H"]


def test_missing_box_is_an_error(tmp_path):
    p = tmp_path / "bad.data"
    p.write_text("LAMMPS data\n\n1 atoms\n")
    with pytest.raises(DriverError, match="box"):
        _read_final_state(p)


def test_fresh_uptake_uses_the_dry_structure(tmp_path):
    dry = tmp_path / "dry" / "dry.data"
    assert _resume_data_file(tmp_path, dry, []) == dry


def test_resume_uses_the_last_checkpointed_relaxed_structure(tmp_path):
    """Previously a resume silently reloaded dry.data and lost every water."""
    dry = tmp_path / "dry" / "dry.data"
    relaxed = tmp_path / "iter_004" / "relaxed.data"
    relaxed.parent.mkdir()
    relaxed.write_text("completed hydrated structure")
    iterations = [_iteration(i, (i + 1) * 20, -8.0) for i in range(5)]

    assert _resume_data_file(tmp_path, dry, iterations) == relaxed


def test_resume_rejects_a_missing_relaxed_structure(tmp_path):
    iterations = [_iteration(3, 80, -8.0)]

    with pytest.raises(DriverError, match=r"iteration 3.*relaxed structure is missing"):
        _resume_data_file(tmp_path, tmp_path / "dry" / "dry.data", iterations)


# ------------------------------------------------------------------- results --
def _iteration(i, n, mu, sat=False):
    return Iteration(index=i, n_waters_before=n - 10, n_requested=10, n_inserted=10,
                     n_waters_after=n, density=1.05, volume=9000.0,
                     lambda_value=n / 8, water_uptake_pct=100 * n * M_WATER / 6895.8,
                     mu_ex=mu, mu_ex_stderr=0.1, mu_gap=mu + 6.5, saturated=sat)


def test_result_reports_both_conventions_and_the_stop_reason():
    its = [_iteration(0, 10, -9.0), _iteration(1, 20, -6.4, True)]
    res = UptakeResult(iterations=its, n_waters=20, lambda_value=2.5,
                       water_uptake_pct=5.2, hydrated_density=1.05, dry_density=1.15,
                       stop_reason="thermodynamic_saturation", converged=True,
                       bulk_mu_ex=-6.5, workdir=__import__("pathlib").Path("."))
    s = res.summary()
    assert s["lambda_waters_per_ionic_group"] == 2.5
    assert s["water_uptake_wt_pct"] == 5.2
    assert s["converged"] is True


def test_running_out_of_iterations_is_not_convergence():
    """A number produced by exhausting the budget is not a measurement."""
    res = UptakeResult(iterations=[_iteration(0, 10, -9.0)], n_waters=10,
                       lambda_value=1.25, water_uptake_pct=2.6,
                       hydrated_density=1.1, dry_density=1.15,
                       stop_reason="max_iterations", converged=False,
                       bulk_mu_ex=-6.5, workdir=__import__("pathlib").Path("."))
    assert res.summary()["converged"] is False


def test_iterations_export_as_a_dataframe():
    res = UptakeResult(iterations=[_iteration(0, 10, -9.0), _iteration(1, 20, -7.0)],
                       n_waters=20, lambda_value=2.5, water_uptake_pct=5.2,
                       hydrated_density=1.05, dry_density=1.15,
                       stop_reason="thermodynamic_saturation", converged=True,
                       bulk_mu_ex=-6.5, workdir=__import__("pathlib").Path("."))
    df = res.to_dataframe()
    assert len(df) == 2 and "lambda_value" in df.columns


# ------------------------------------------------- bulk reference guarding --
def _reference(**overrides):
    from aemwater.bulk import BulkReference, BulkSettings
    from aemwater.widom import WidomEstimate

    settings = dataclasses.replace(
        BulkSettings(water_model="spce", temperature=298.15, pressure=1.0,
                     n_waters=500, cutoff=10.0, kspace_accuracy=1e-4,
                     equil_steps=1000, widom_steps=1000,
                     insertions_per_call=100, seed=1),
        **overrides,
    )
    estimate = WidomEstimate(mu_ex=-6.0, stderr=0.2, temperature=298.15,
                             n_blocks=10, block_values=(1e4,) * 10,
                             mean_boltzmann=1e4, effective_samples=50.0)
    return BulkReference(settings=settings, mu_ex=estimate, density=0.997,
                         volume=15000.0, workdir=pathlib.Path("."))


def test_a_matching_reference_is_accepted():
    from aemwater.driver import _check_reference_matches

    ref = _reference()
    _check_reference_matches(ref, ref.settings)  # must not raise


@pytest.mark.parametrize("field,value", [
    ("water_model", "tip3p"),
    ("temperature", 353.15),
    ("cutoff", 12.0),
    ("insertions_per_call", 500),
])
def test_a_reference_at_different_settings_is_refused(field, value):
    """The bias only cancels between estimates computed the same way."""
    from aemwater.driver import _check_reference_matches

    ref = _reference()
    wanted = dataclasses.replace(ref.settings, **{field: value})
    with pytest.raises(ValueError, match="different settings"):
        _check_reference_matches(ref, wanted)


def test_box_size_and_seed_do_not_invalidate_a_reference():
    """They change the variance, not the bias, so requiring them to match
    would force needless recomputation."""
    from aemwater.driver import _check_reference_matches

    ref = _reference()
    wanted = dataclasses.replace(ref.settings, n_waters=1000, seed=99)
    _check_reference_matches(ref, wanted)  # must not raise


# ------------------------------------------------------- periodic unwrapping --
def test_readback_unwraps_with_image_flags(tmp_path):
    """Coordinates must be unwrapped, or reassembly stretches molecules.

    LAMMPS wraps coordinates into the primary cell, so a water straddling a
    boundary has its H on the opposite face -- 20 A away rather than 1 A. The
    driver rebuilds each iteration from this file, and feeding it wrapped
    coordinates produced 894 bonds over 2.5 A (longest 27 A in a 22 A cell) and
    an opaque abort inside SHAKE.
    """
    from aemwater.driver import _read_final_state

    # A water whose oxygen sits just inside the z = 0 face. LAMMPS wraps one
    # hydrogen to the far face (z ~ 19.6) and records image flag -1 for it, so
    # the unwrapped position is z ~ -0.4 -- next to the oxygen, not 20 A away.
    p = tmp_path / "wrapped.data"
    p.write_text(
        "LAMMPS data\n\n3 atoms\n2 atom types\n\n"
        "0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n0.0 20.0 zlo zhi\n\n"
        "Masses\n\n1 15.9994\n2 1.008\n\n"
        "Atoms # full\n\n"
        "1 1 1 -0.8476 1.0 2.0 0.5 0 0 0\n"
        "2 1 2 0.4238 1.6 2.6 1.0 0 0 0\n"
        "3 1 2 0.4238 1.6 1.4 19.6 0 0 -1\n"
    )
    coords, elements, edge = _read_final_state(p)

    assert edge == pytest.approx(20.0)
    assert coords[2][2] == pytest.approx(19.6 - 20.0)
    # The unwrapped O-H distances are bond-length scale, not box scale.
    for i in (1, 2):
        d = float(np.linalg.norm(coords[i] - coords[0]))
        assert d < 2.5, f"atom {i} is {d:.1f} A from the oxygen"


def test_readback_rejects_a_file_without_image_flags(tmp_path):
    """A missing image-flag column must fail loudly, not silently wrap."""
    from aemwater.driver import DriverError, _read_final_state

    p = tmp_path / "noimg.data"
    p.write_text(
        "LAMMPS data\n\n1 atoms\n1 atom types\n\n"
        "0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n0.0 20.0 zlo zhi\n\n"
        "Masses\n\n1 15.9994\n\n"
        "Atoms # full\n\n1 1 1 -0.8476 1.0 2.0 3.0\n"
    )
    with pytest.raises(DriverError, match="image flags"):
        _read_final_state(p)


def test_readback_sorts_atoms_by_id(tmp_path):
    """LAMMPS writes the Atoms section in internal order, not by ID.

    LAMMPS sorts atoms spatially for cache efficiency, so `write_data` emits
    them in an order that has nothing to do with the ID in column 1. Reading the
    lines sequentially assigns every coordinate to the wrong atom -- silently:
    the atom count is right and the density plausible, but in a real run 509 of
    1056 atoms ended up with a different element from the molecule they were
    assigned to, and reassembly gave 894 bonds over 2.5 A.
    """
    from aemwater.driver import _read_final_state

    p = tmp_path / "shuffled.data"
    # Deliberately out of order: IDs 3, 1, 2.
    p.write_text(
        "LAMMPS data\n\n3 atoms\n2 atom types\n\n"
        "0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n0.0 20.0 zlo zhi\n\n"
        "Masses\n\n1 15.9994\n2 1.008\n\n"
        "Atoms # full\n\n"
        "3 1 2 0.4238 3.0 3.0 3.0 0 0 0\n"
        "1 1 1 -0.8476 1.0 1.0 1.0 0 0 0\n"
        "2 1 2 0.4238 2.0 2.0 2.0 0 0 0\n"
    )
    coords, elements, _ = _read_final_state(p)

    # Row order must follow the ID, so atom 1 (the oxygen) comes first.
    assert coords[0].tolist() == [1.0, 1.0, 1.0]
    assert coords[2].tolist() == [3.0, 3.0, 3.0]
    assert elements == ["O", "H", "H"]


def test_readback_rejects_non_contiguous_ids(tmp_path):
    """IDs must be 1..N, since the inventory is ordered that way."""
    from aemwater.driver import DriverError, _read_final_state

    p = tmp_path / "gappy.data"
    p.write_text(
        "LAMMPS data\n\n2 atoms\n1 atom types\n\n"
        "0.0 20.0 xlo xhi\n0.0 20.0 ylo yhi\n0.0 20.0 zlo zhi\n\n"
        "Masses\n\n1 15.9994\n\n"
        "Atoms # full\n\n"
        "1 1 1 -0.8476 1.0 1.0 1.0 0 0 0\n"
        "7 1 1 -0.8476 2.0 2.0 2.0 0 0 0\n"
    )
    with pytest.raises(DriverError, match="contiguous"):
        _read_final_state(p)
