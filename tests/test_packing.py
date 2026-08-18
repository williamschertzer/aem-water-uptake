"""Cell packing: density accounting, self-avoidance, periodic correctness."""

from __future__ import annotations

import numpy as np
import pytest

from aemwater.chemistry import build_composition
from aemwater.packing import PackingError, pack_cell, target_edge_length

SMILES = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"


def test_target_edge_length_reproduces_density():
    mass = 6895.828  # g/mol
    edge = target_edge_length(mass, 1.1)
    volume_cm3 = (edge ** 3) * 1.0e-24
    assert 6.02214076e23 * volume_cm3 * 1.1 == pytest.approx(mass, rel=1e-9)


def test_target_edge_length_rejects_nonpositive_density():
    with pytest.raises(PackingError):
        target_edge_length(1000.0, 0.0)


def _small_cell(dilation=1.7, seed=1):
    comp = build_composition(SMILES, 2, 3)
    rng = np.random.default_rng(0)
    # Stand-in molecules: compact random blobs, so the test does not pay for a
    # real chain build. Packing only ever treats them as rigid point clouds.
    chains = [rng.normal(scale=2.5, size=(40, 3)) for _ in range(comp.n_chains)]
    ions = [np.zeros((1, 3)) for _ in range(comp.n_counterions)]
    return pack_cell(chains, ions, comp, target_density=1.1,
                     dilation=dilation, seed=seed)


def test_packed_cell_is_self_avoiding():
    cell = _small_cell()
    # Ions are allowed a tighter clearance (0.85x) than chain-chain contacts.
    assert cell.min_intermolecular_distance() > 1.8


def test_dilute_density_matches_dilation():
    cell = _small_cell(dilation=2.0)
    # Linear dilation d divides the density by d^3.
    assert cell.density == pytest.approx(cell.target_density / 8.0, rel=1e-6)


def test_packing_is_deterministic_given_seed():
    a = _small_cell(seed=7)
    b = _small_cell(seed=7)
    assert np.allclose(a.coordinates, b.coordinates)


def test_packing_differs_between_seeds():
    a = _small_cell(seed=1)
    b = _small_cell(seed=2)
    assert not np.allclose(a.coordinates, b.coordinates)


def test_atom_count_and_inventory_are_consistent():
    cell = _small_cell()
    assert cell.n_atoms == sum(cell.molecule_sizes)
    assert len(cell.molecule_kinds) == len(cell.molecule_sizes)
    assert cell.molecule_kinds.count("chain") == 2


def test_minimum_image_distance_sees_across_the_boundary():
    """A contact across the periodic face must be detected, not missed."""
    comp = build_composition(SMILES, 2, 3)
    cell = _small_cell()
    e = cell.edge
    # Place two single atoms just inside opposite faces: 0.4 A apart through the
    # boundary, but e - 0.4 A apart if PBC were ignored.
    cell.coordinates = np.array([[0.1, 1.0, 1.0], [e - 0.3, 1.0, 1.0]])
    cell.molecule_sizes = [1, 1]
    cell.molecule_kinds = ["ion", "ion"]
    assert cell.min_intermolecular_distance() == pytest.approx(0.4, abs=1e-6)


def test_impossible_packing_raises_with_actionable_message():
    """An over-full box must fail loudly, naming the knob that relieves it."""
    comp = build_composition(SMILES, 2, 3)
    rng = np.random.default_rng(0)
    # Bodies far larger than the mass accounting implies: the cell edge is set
    # from the composition mass, so 30 A blobs cannot fit in the ~13 A cell.
    chains = [rng.normal(scale=10.0, size=(300, 3)) for _ in range(6)]
    with pytest.raises(PackingError, match="dilation"):
        pack_cell(chains, [], comp, target_density=1.1, dilation=1.0, seed=0,
                  min_distance=4.0)


def test_dilation_below_one_is_rejected():
    comp = build_composition(SMILES, 1, 2)
    with pytest.raises(PackingError, match="dilation must be"):
        pack_cell([np.zeros((5, 3))], [], comp, target_density=1.0, dilation=0.5)
