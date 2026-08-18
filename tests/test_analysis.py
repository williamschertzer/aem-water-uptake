"""Structural analysis of the absorbed water."""

from __future__ import annotations

import numpy as np
import pytest

from aemwater.analysis import (
    OO_FIRST_SHELL,
    count_hydrogen_bonds,
    hydration_structure,
    is_percolating,
    water_clusters,
)


def _waters(oxygens: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Attach SPC/E-like hydrogens to a set of oxygen positions."""
    offsets = np.array([[0.0, 0, 0], [0.9572, 0, 0], [-0.24, 0.927, 0]])
    coords = (np.repeat(oxygens, 3, axis=0) + np.tile(offsets, (len(oxygens), 1)))
    return coords, ["O", "H", "H"] * len(oxygens)


# ------------------------------------------------------------------ clusters --
def test_isolated_waters_are_separate_clusters():
    ox = np.array([[1.0, 1, 1], [20.0, 20, 20], [40.0, 40, 40]])
    assert len(water_clusters(ox, 60.0)) == 3


def test_neighbouring_waters_form_one_cluster():
    ox = np.array([[10.0, 10, 10], [12.5, 10, 10], [15.0, 10, 10]])
    clusters = water_clusters(ox, 60.0)
    assert len(clusters) == 1 and len(clusters[0]) == 3


def test_clusters_connect_across_the_periodic_boundary():
    """Two waters either side of a face are neighbours, not separate pockets."""
    edge = 20.0
    ox = np.array([[0.5, 10, 10], [edge - 0.5, 10, 10]])
    assert len(water_clusters(ox, edge)) == 1


def test_empty_system_has_no_clusters():
    assert water_clusters(np.empty((0, 3)), 20.0) == []


# --------------------------------------------------------------- percolation --
def test_a_dense_liquid_percolates():
    rng = np.random.default_rng(0)
    assert is_percolating(rng.uniform(0, 20, (250, 3)), 20.0)


def test_sparse_isolated_waters_do_not_percolate():
    rng = np.random.default_rng(0)
    assert not is_percolating(rng.uniform(0, 40, (12, 3)), 40.0)


def test_a_spanning_chain_percolates():
    """A wire through the cell that closes through the boundary."""
    edge = 20.0
    z = np.arange(0, edge, 2.0)
    ox = np.column_stack([np.full_like(z, 10.0), np.full_like(z, 10.0), z])
    assert is_percolating(ox, edge)


def test_a_chain_that_stops_short_does_not_percolate():
    """Touching both faces is not the same as connecting through them."""
    edge = 30.0
    z = np.arange(0, 20.0, 2.0)          # leaves a 10 A gap at the boundary
    ox = np.column_stack([np.full_like(z, 10.0), np.full_like(z, 10.0), z])
    assert not is_percolating(ox, edge)


def test_single_water_does_not_percolate():
    assert not is_percolating(np.array([[5.0, 5, 5]]), 20.0)


# ------------------------------------------------------------ hydrogen bonds --
def test_a_donor_oriented_dimer_has_one_hydrogen_bond():
    coords = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                       [2.8, 0, 0], [3.2, 0.9, 0], [3.5, -0.6, 0]])
    assert count_hydrogen_bonds(coords, ["O", "H", "H"] * 2, 30.0) == 1


def test_distant_waters_form_no_hydrogen_bond():
    coords, elements = _waters(np.array([[0.0, 0, 0], [12.0, 0, 0]]))
    assert count_hydrogen_bonds(coords, elements, 40.0) == 0


def test_close_but_misoriented_waters_form_no_bond():
    """Distance alone roughly doubles the count; the angle criterion matters."""
    coords = np.array([[0.0, 0, 0], [-0.96, 0, 0], [0.24, -0.93, 0],
                       [3.0, 0, 0], [3.96, 0, 0], [2.76, 0.93, 0]])
    assert count_hydrogen_bonds(coords, ["O", "H", "H"] * 2, 30.0) == 0


def test_hydrogen_bonds_respect_the_minimum_image():
    edge = 10.0
    # O1 near x=0 donates through the boundary to O2 near x=edge: 2.7 A apart by
    # minimum image, 7.3 A if the boundary is ignored.
    # O1 donates through the boundary; O2's hydrogens point perpendicular, so
    # exactly one bond exists rather than a mutually-donating pair.
    coords = np.array([[0.30, 5, 5], [-0.66, 5, 5], [0.54, 5.93, 5],
                       [7.60, 5, 5], [7.60, 5.96, 5], [7.60, 4.76, 5.73]])
    assert count_hydrogen_bonds(coords, ["O", "H", "H"] * 2, edge) == 1

    # The same pair in a box big enough that they are simply far apart.
    assert count_hydrogen_bonds(coords, ["O", "H", "H"] * 2, 40.0) == 0


# ---------------------------------------------------------------- structure --
def test_structure_of_a_dry_cell_is_empty():
    s = hydration_structure(np.empty((0, 3)), [], 20.0, 0)
    assert s.n_waters == 0 and not s.percolating


def test_structure_reports_cluster_statistics():
    rng = np.random.default_rng(2)
    ox = rng.uniform(0, 16, (80, 3))
    coords, elements = _waters(ox)
    s = hydration_structure(coords, elements, 16.0, 80)
    assert s.n_waters == 80
    assert 0 < s.largest_cluster_fraction <= 1.0
    assert s.mean_hbonds_per_water >= 0


def test_first_shell_count_uses_the_cation_positions():
    """Waters near the quaternary nitrogen, not all waters in the cell."""
    edge = 30.0
    ox = np.array([[15.0, 15, 15], [17.0, 15, 15], [1.0, 1, 1]])
    water_coords, water_elements = _waters(ox)
    cation = np.array([[15.0, 15, 15]])
    coords = np.vstack([cation, water_coords])
    elements = ["N"] + water_elements
    s = hydration_structure(coords, elements, edge, 3, cation_indices=np.array([0]))
    assert s.waters_per_cation == pytest.approx(2.0)


# ------------------------------------------------------------- report output --
def test_report_table_needs_no_optional_dependency(monkeypatch):
    """The report must not depend on `tabulate`.

    `DataFrame.to_markdown` needs it, and pandas does not require it. Writing
    the report is the last step of a run that takes hours, so a missing
    optional package there throws away all the expensive work. This asserts the
    table is built without it even when the import would fail.
    """
    import builtins

    import pandas as pd

    real_import = builtins.__import__

    def no_tabulate(name, *args, **kwargs):
        if name == "tabulate":
            raise ImportError("tabulate is deliberately unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tabulate)

    from aemwater.analysis import _markdown_table

    df = pd.DataFrame({"index": [0, 1], "lambda_value": [0.625, 1.25],
                       "mu_ex": [1.26, None]})
    table = _markdown_table(df)

    rows = table.splitlines()
    assert rows[0].startswith("| index"), table
    assert set(rows[1]) <= {"|", "-"}, "second row must be the separator"
    assert len(rows) == 4, "header, separator and one row per iteration"
    # A missing mu_ex must read as absent rather than as the string "nan".
    assert "nan" not in table.lower(), table
    assert "--" in rows[3], rows[3]
