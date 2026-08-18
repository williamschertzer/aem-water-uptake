"""Void detection and water placement."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from aemwater.forcefield.water import water_model
from aemwater.insertion import (
    InsertionError,
    VDW_RADII,
    atom_radii,
    free_volume_profile,
    insert_waters,
    map_voids,
    water_orientations,
)


def test_atom_radii_fall_back_for_unknown_elements():
    r = atom_radii(["C", "H", "Xx"])
    assert r[0] == pytest.approx(VDW_RADII["C"])
    assert r[2] == pytest.approx(1.70)


def test_radii_scale_multiplicatively():
    assert atom_radii(["C"], 0.5)[0] == pytest.approx(VDW_RADII["C"] * 0.5)


def test_empty_box_is_almost_entirely_free():
    """One atom in a 20 A box: free volume must approach 1."""
    vm = map_voids(np.array([[10.0, 10.0, 10.0]]), ["C"], edge=20.0,
                   grid_spacing=0.8)
    excluded = (4 / 3) * np.pi * (VDW_RADII["C"] + 1.4) ** 3
    expected = 1.0 - excluded / 20.0 ** 3
    assert vm.free_volume_fraction == pytest.approx(expected, abs=0.02)


def test_coordinate_element_mismatch_is_rejected():
    with pytest.raises(InsertionError, match="elements"):
        map_voids(np.zeros((3, 3)), ["C", "H"], edge=10.0)


def test_depths_are_finite_and_sorted_descending():
    """Infinite depths would make the deepest-first ranking arbitrary."""
    rng = np.random.default_rng(0)
    xyz = rng.random((30, 3)) * 25.0
    vm = map_voids(xyz, ["C"] * 30, edge=25.0, grid_spacing=1.0)
    assert np.all(np.isfinite(vm.depths))
    assert np.all(np.diff(vm.depths) <= 1e-9)


def test_sites_respect_the_minimum_separation():
    rng = np.random.default_rng(1)
    xyz = rng.random((40, 3)) * 25.0
    sep = 3.0
    vm = map_voids(xyz, ["C"] * 40, edge=25.0, grid_spacing=0.8,
                   min_site_separation=sep)
    assert len(vm) > 1
    tree = cKDTree(np.mod(vm.positions, vm.edge), boxsize=vm.edge)
    pairs = tree.query_pairs(sep - 1e-6)
    assert not pairs, f"{len(pairs)} site pairs closer than {sep} A"


def test_sites_clear_every_atom_vdw_surface():
    """The acceptance criterion must hold for the returned sites, with PBC."""
    rng = np.random.default_rng(2)
    xyz = rng.random((50, 3)) * 24.0
    els = ["C"] * 25 + ["H"] * 25
    probe = 1.4
    vm = map_voids(xyz, els, edge=24.0, grid_spacing=0.7, probe_radius=probe)
    radii = atom_radii(els)
    for site in vm.positions[:20]:
        d = site - np.mod(xyz, 24.0)
        d -= 24.0 * np.round(d / 24.0)
        clearance = np.linalg.norm(d, axis=1) - radii
        assert clearance.min() >= probe - 1e-6


def test_dense_box_offers_no_sites():
    """A box packed solid with atoms must report geometric saturation."""
    grid = np.stack(np.meshgrid(*[np.arange(0.5, 12, 1.5)] * 3, indexing="ij"), -1)
    xyz = grid.reshape(-1, 3)
    res = insert_waters(xyz, ["C"] * len(xyz), 12.0, 10, water_model("spce"))
    assert res.n_inserted == 0
    assert res.saturated


def test_inserted_waters_have_correct_geometry():
    rng = np.random.default_rng(3)
    xyz = rng.random((20, 3)) * 25.0
    model = water_model("spce")
    res = insert_waters(xyz, ["C"] * 20, 25.0, 5, model, seed=4)
    assert res.n_inserted == 5
    assert res.coordinates.shape == (15, 3)
    for i in range(res.n_inserted):
        o, h1, h2 = res.coordinates[3 * i : 3 * i + 3]
        assert np.linalg.norm(h1 - o) == pytest.approx(model.r_OH, abs=1e-6)
        assert np.linalg.norm(h2 - o) == pytest.approx(model.r_OH, abs=1e-6)
        cos = np.dot(h1 - o, h2 - o) / model.r_OH ** 2
        assert np.degrees(np.arccos(cos)) == pytest.approx(model.angle_HOH, abs=1e-4)


def test_orientations_are_not_all_identical():
    """Identical orientations would bias the inserted water dipoles."""
    ors = water_orientations(20, water_model("spce"), np.random.default_rng(0))
    h_dirs = ors[:, 1, :] / np.linalg.norm(ors[:, 1, :], axis=1, keepdims=True)
    assert np.linalg.norm(h_dirs.mean(axis=0)) < 0.5


def test_insertion_is_deterministic_given_seed():
    rng = np.random.default_rng(5)
    xyz = rng.random((20, 3)) * 25.0
    a = insert_waters(xyz, ["C"] * 20, 25.0, 5, water_model("spce"), seed=11)
    b = insert_waters(xyz, ["C"] * 20, 25.0, 5, water_model("spce"), seed=11)
    assert np.allclose(a.coordinates, b.coordinates)


def test_free_volume_decreases_with_probe_radius():
    rng = np.random.default_rng(6)
    xyz = rng.random((60, 3)) * 24.0
    profile = free_volume_profile(xyz, ["C"] * 60, 24.0, probes=(1.0, 1.4, 1.8),
                                  grid_spacing=1.0)
    values = [profile[r] for r in (1.0, 1.4, 1.8)]
    assert values[0] > values[1] > values[2]


def test_larger_vdw_scale_finds_fewer_sites():
    rng = np.random.default_rng(7)
    xyz = rng.random((60, 3)) * 24.0
    loose = map_voids(xyz, ["C"] * 60, 24.0, grid_spacing=0.9, vdw_scale=0.8)
    tight = map_voids(xyz, ["C"] * 60, 24.0, grid_spacing=0.9, vdw_scale=1.2)
    assert loose.free_volume_fraction > tight.free_volume_fraction
