"""Bulk water reference: box construction, caching, sanity checks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from aemwater.bulk import (
    AVOGADRO,
    LITERATURE_MU_EX,
    M_WATER,
    MIN_OO,
    BulkReference,
    BulkSettings,
    build_bulk_coordinates,
    water_box_edge,
)
from aemwater.forcefield.water import water_model
from aemwater.widom import WidomEstimate


def _settings(**kw):
    base = dict(water_model="spce", temperature=298.15, pressure=1.0, n_waters=216,
                cutoff=10.0, kspace_accuracy=1e-4, equil_steps=1000,
                widom_steps=1000, insertions_per_call=50, seed=1)
    base.update(kw)
    return BulkSettings(**base)


def test_box_edge_reproduces_the_requested_density():
    n, rho = 216, 0.997
    edge = water_box_edge(n, rho)
    volume_cm3 = (edge * 1e-8) ** 3
    assert n * M_WATER / (AVOGADRO * volume_cm3) == pytest.approx(rho, rel=1e-9)


def test_lattice_has_no_close_contacts():
    """Unbounded jitter previously produced 1.5 A O-O pairs that blow up step 1."""
    for n in (64, 216, 512):
        coords, edge = build_bulk_coordinates(n, water_model("spce"), seed=0)
        oxygens = np.mod(coords[0::3], edge)
        d, _ = cKDTree(oxygens, boxsize=edge).query(oxygens, k=2)
        assert d[:, 1].min() >= MIN_OO, f"n={n}: closest O-O {d[:, 1].min():.2f} A"


def test_lattice_nearest_neighbour_is_near_the_liquid_peak():
    coords, edge = build_bulk_coordinates(216, water_model("spce"), seed=0)
    oxygens = np.mod(coords[0::3], edge)
    d, _ = cKDTree(oxygens, boxsize=edge).query(oxygens, k=2)
    assert 2.5 < d[:, 1].mean() < 3.4


def test_all_molecules_are_inside_the_box():
    coords, edge = build_bulk_coordinates(125, water_model("spce"), seed=3)
    assert coords.min() >= 0.0 and coords.max() < edge


def test_jitter_actually_breaks_the_lattice():
    """A perfect lattice would persist as an artificial ice."""
    coords, edge = build_bulk_coordinates(125, water_model("spce"), seed=1)
    oxygens = coords[0::3]
    spacing = edge / 5
    lattice = np.round(oxygens / spacing) * spacing
    assert np.abs(oxygens - lattice).max() > 0.05


def test_water_geometry_is_the_model_geometry():
    model = water_model("spce")
    coords, _ = build_bulk_coordinates(8, model, seed=0)
    for i in range(8):
        o, h1, h2 = coords[3 * i : 3 * i + 3]
        assert np.linalg.norm(h1 - o) == pytest.approx(model.r_OH, abs=1e-6)


def test_settings_key_is_stable_and_discriminating():
    assert _settings().key() == _settings().key()
    assert _settings().key() != _settings(temperature=310.0).key()
    assert _settings().key() != _settings(cutoff=12.0).key()
    assert _settings().key() != _settings(water_model="tip3p").key()


def _reference(mu=-6.5, density=0.998, neff=100.0, model="spce"):
    est = WidomEstimate(mu_ex=mu, stderr=0.1, temperature=298.15, n_blocks=5,
                        block_values=np.full(5, mu), mean_boltzmann=1.0,
                        effective_samples=neff)
    return BulkReference(_settings(water_model=model), est, density, 8000.0,
                         __import__("pathlib").Path("."))


def test_a_reasonable_reference_raises_no_sanity_warnings():
    assert _reference().sanity() == []


def test_mu_ex_far_from_the_published_value_is_flagged():
    issues = _reference(mu=-2.0).sanity()
    assert any("mu_ex" in i for i in issues)


def test_density_far_from_the_published_value_is_flagged():
    issues = _reference(density=0.85).sanity()
    assert any("density" in i for i in issues)


def test_unconverged_widom_average_is_flagged():
    issues = _reference(neff=3.0).sanity()
    assert any("effective samples" in i for i in issues)


def test_unknown_water_model_skips_the_literature_comparison():
    """An unusual model can legitimately differ; it must not be flagged wrongly."""
    assert "custom" not in LITERATURE_MU_EX
    assert _reference(mu=-1.0, model="custom").sanity() == []


def test_cached_reference_is_reused_without_running_lammps(tmp_path, monkeypatch):
    """The reference is identical for every membrane, so it must be cached."""
    from aemwater import bulk

    settings = _settings()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"bulk_{settings.key()}.json").write_text(json.dumps({
        "mu_ex": -6.42, "stderr": 0.08, "n_blocks": 5,
        "block_values": [-6.4] * 5, "mean_boltzmann": 1.0,
        "effective_samples": 250.0, "volume": 6500.0, "density": 0.9971,
    }))

    def explode(*a, **k):
        raise AssertionError("LAMMPS must not run when a cached result exists")

    monkeypatch.setattr(bulk, "_run_bulk_stages", explode)
    ref = bulk.run_bulk_reference(settings, tmp_path / "wd", cache_dir=cache)
    assert ref.mu_ex.mu_ex == pytest.approx(-6.42)
    assert ref.density == pytest.approx(0.9971)


# ------------------------------------------------- Widom molecule template --
def test_widom_template_carries_bond_topology(tmp_path):
    """Without Bonds/Angles, LAMMPS applies no exclusions to the test molecule.

    The inserted water's own O-H and H-H Coulomb terms (about -202 kcal/mol for
    SPC/E) then enter every trial energy and swamp the solvation free energy.
    This produced mu_ex = -194 kcal/mol before the template was fixed.
    """
    from aemwater.forcefield.water import water_model
    from aemwater.lammps.inputs import write_water_molecule_template

    path = write_water_molecule_template(
        tmp_path / "h2o.mol", water_model("spce"), 3, 4, 7, 5
    )
    text = path.read_text()
    assert "2 bonds" in text and "1 angles" in text
    assert "Bonds" in text and "Angles" in text
    # The bond and angle types must be the ones the data file uses.
    assert "1 7 1 2" in text and "2 7 1 3" in text
    assert "1 5 2 1 3" in text


def test_widom_template_self_energy_is_the_scale_of_the_bug():
    """Documents why the missing topology was detectable by inspection."""
    from aemwater.forcefield.water import water_model

    m = water_model("spce")
    coulomb = 332.0637
    r_hh = 2 * m.r_OH * np.sin(np.radians(m.angle_HOH) / 2)
    self_energy = (2 * coulomb * m.charge_O * m.charge_H / m.r_OH
                   + coulomb * m.charge_H**2 / r_hh)
    assert self_energy < -150.0


def test_cache_dir_expands_the_home_tilde(tmp_path, monkeypatch):
    """The default cache_dir is the *string* "~/.cache/aemwater".

    Path() does not expand "~", so an unexpanded cache_dir silently creates a
    literal ./~/ directory relative to the cwd. The cache then stops being
    shared -- every run from a different working directory recomputes an
    expensive bulk reference and believes it was cached. This repo carried a
    committed .gitignore entry for /~/ and two stale Widom-era cache files in
    ./~/.cache/aemwater/ as evidence of exactly that.
    """
    from pathlib import Path

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    resolved = Path("~/.cache/aemwater").expanduser()
    assert not str(resolved).startswith("~"), "tilde survived expansion"
    assert resolved.is_absolute(), f"cache dir must be absolute, got {resolved}"


def test_no_literal_tilde_directory_is_created_by_a_default_cache_dir(tmp_path):
    """Regression guard on the source, not the filesystem.

    A future edit that reintroduces bare Path(cache_dir) would recreate the
    ./~/ directory, and the existing .gitignore entry would hide it again.
    """
    import inspect

    import aemwater.bulk as bulk

    src = inspect.getsource(bulk)
    bare = [
        line.strip() for line in src.splitlines()
        if "Path(cache_dir)" in line and "expanduser" not in line
    ]
    assert not bare, f"cache_dir used without expanduser(): {bare}"
