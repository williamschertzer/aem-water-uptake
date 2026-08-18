"""Pack chains and counterions into a periodic cell at a target density.

Strategy
--------
The cell is built *dilute* and then compressed by MD, rather than packed directly
at the target density. Packing hundreds of atoms per chain into a dense box
without overlaps is a hard geometric problem; letting LAMMPS compress a dilute
box under a barostat is a solved one, and it produces an equilibrated melt rather
than a jammed configuration.

So the initial edge length is set from the target density scaled by
``dilation``: a 2x linear dilation is 8x the volume, which is loose enough that
random rigid-body placement of whole chains succeeds without atomic overlap.

Placement is rigid-body: each chain keeps the internal conformation the builder
produced (which was validated to be non-collapsed and self-avoiding), and is
moved and randomly rotated as a unit. Only inter-molecular clashes are checked,
because intra-molecular geometry is already correct and re-checking it would
reject good conformers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .chemistry import SystemComposition
from .utils import AVOGADRO, LOG

#: Minimum acceptable inter-molecular heavy-atom separation when packing (A).
#: Below this the first LAMMPS minimisation risks a diverging force; the soft-core
#: push-off stage exists to clean up anything between this and true contact.
MIN_INTERMOLECULAR = 2.2

#: Radius within which a rigid-body placement is retried (A).
PLACEMENT_ATTEMPTS = 400


class PackingError(RuntimeError):
    """Raised when the cell cannot be packed under the given constraints."""


@dataclass
class PackedCell:
    """A packed periodic cell: coordinates plus the molecule inventory."""

    edge: float
    coordinates: np.ndarray
    molecule_sizes: list[int]
    molecule_kinds: list[str]
    target_density: float
    dilation: float
    total_mass: float

    @property
    def n_atoms(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def volume(self) -> float:
        return self.edge ** 3

    @property
    def density(self) -> float:
        """Mass density of the packed (dilute) cell, g/cm^3."""
        return self.total_mass / (AVOGADRO * self.volume * 1.0e-24)

    def min_intermolecular_distance(self) -> float:
        """Closest approach between atoms of different molecules, with PBC."""
        mol_id = np.concatenate(
            [np.full(n, i) for i, n in enumerate(self.molecule_sizes)]
        )
        tree = cKDTree(np.mod(self.coordinates, self.edge), boxsize=self.edge)
        pairs = tree.query_pairs(6.0, output_type="ndarray")
        if len(pairs) == 0:
            return float("inf")
        cross = mol_id[pairs[:, 0]] != mol_id[pairs[:, 1]]
        if not cross.any():
            return float("inf")
        d = self.coordinates[pairs[cross, 0]] - self.coordinates[pairs[cross, 1]]
        d -= self.edge * np.round(d / self.edge)
        return float(np.min(np.linalg.norm(d, axis=1)))

    def summary(self) -> dict[str, object]:
        return {
            "edge_A": round(self.edge, 3),
            "atoms": self.n_atoms,
            "molecules": len(self.molecule_sizes),
            "volume_A3": round(self.volume, 1),
            "packed_density_g_cm3": round(self.density, 4),
            "target_density_g_cm3": self.target_density,
            "dilation": self.dilation,
            "min_intermolecular_A": round(self.min_intermolecular_distance(), 3),
        }


def target_edge_length(total_mass: float, density: float) -> float:
    """Cubic edge (A) giving ``density`` g/cm^3 for ``total_mass`` g/mol of material."""
    if density <= 0:
        raise PackingError(f"density must be positive, got {density}")
    volume_cm3 = total_mass / (AVOGADRO * density)
    return float((volume_cm3 * 1.0e24) ** (1.0 / 3.0))


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix (Shoemake's quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
            math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
            math.sqrt(u1) * math.sin(2 * math.pi * u3),
            math.sqrt(u1) * math.cos(2 * math.pi * u3),
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _place_rigid_body(
    coords: np.ndarray,
    existing: np.ndarray,
    edge: float,
    rng: np.random.Generator,
    min_distance: float,
    attempts: int = PLACEMENT_ATTEMPTS,
) -> np.ndarray:
    """Randomly rotate and translate ``coords`` until it clears ``existing``.

    Distances are minimum-image, so a molecule placed near a face is checked
    against material wrapped around the opposite face -- otherwise the first
    LAMMPS step would discover the overlap instead.
    """
    centred = coords - coords.mean(axis=0)
    tree = cKDTree(np.mod(existing, edge), boxsize=edge) if len(existing) else None

    best: np.ndarray | None = None
    best_gap = -1.0
    for _ in range(attempts):
        placed = centred @ _random_rotation(rng).T + rng.random(3) * edge
        if tree is None:
            return placed
        wrapped = np.mod(placed, edge)
        # Query the nearest existing atom for each candidate atom; the periodic
        # KD-tree handles the minimum image for us.
        dists, _ = tree.query(wrapped, k=1)
        gap = float(dists.min())
        if gap >= min_distance:
            return placed
        if gap > best_gap:
            best_gap, best = gap, placed
    raise PackingError(
        f"could not place a molecule with {min_distance:.1f} A clearance in {attempts} "
        f"attempts (best gap {best_gap:.2f} A). Increase box.dilation or reduce n_chains."
    )


def pack_cell(
    chain_coords: Sequence[np.ndarray],
    ion_coords: Sequence[np.ndarray],
    composition: SystemComposition,
    target_density: float,
    dilation: float = 1.6,
    seed: int = 0,
    min_distance: float = MIN_INTERMOLECULAR,
) -> PackedCell:
    """Pack chains and ions into a dilute cubic cell for later compression.

    Parameters
    ----------
    chain_coords, ion_coords:
        Per-molecule coordinate arrays, already at the correct internal geometry.
    composition:
        Supplies the total dry molar mass, which sets the target edge length.
    target_density:
        The density the *compressed* cell should reach, g/cm^3.
    dilation:
        Linear expansion factor applied to the target edge for packing. 1.6 is
        4.1x the target volume, which packs reliably for chains up to a few
        hundred units without leaving so much void that compression takes long.
    """
    if dilation < 1.0:
        raise PackingError(f"dilation must be >= 1, got {dilation}")
    mass = composition.dry_molar_mass
    edge_target = target_edge_length(mass, target_density)
    edge = edge_target * dilation
    rng = np.random.default_rng(seed)

    # Largest first: the hardest bodies to place go into the emptiest box.
    bodies = [("chain", c) for c in chain_coords] + [("ion", c) for c in ion_coords]
    order = sorted(range(len(bodies)), key=lambda i: -len(bodies[i][1]))

    placed_coords: list[np.ndarray | None] = [None] * len(bodies)
    stack = np.empty((0, 3))
    for count, idx in enumerate(order, start=1):
        kind, coords = bodies[idx]
        # Ions are single atoms or tiny; they tolerate a tighter clearance than
        # the heavy-atom criterion used between polymer chains.
        limit = min_distance if kind == "chain" else min_distance * 0.85
        placed = _place_rigid_body(coords, stack, edge, rng, limit)
        placed_coords[idx] = placed
        stack = np.vstack([stack, placed])
        if count % 25 == 0:
            LOG.debug("packed %d/%d molecules", count, len(bodies))

    coordinates = np.vstack([p for p in placed_coords if p is not None])
    cell = PackedCell(
        edge=edge,
        coordinates=coordinates,
        molecule_sizes=[len(bodies[i][1]) for i in range(len(bodies))],
        molecule_kinds=[bodies[i][0] for i in range(len(bodies))],
        target_density=target_density,
        dilation=dilation,
        total_mass=mass,
    )
    LOG.info(
        "packed %d molecules (%d atoms) into a %.2f A cell: %.4f g/cm3 dilute, "
        "%.4f g/cm3 after compression, closest inter-molecular contact %.2f A",
        len(bodies),
        cell.n_atoms,
        edge,
        cell.density,
        target_density,
        cell.min_intermolecular_distance(),
    )
    return cell


__all__ = [
    "PackedCell",
    "PackingError",
    "pack_cell",
    "target_edge_length",
    "MIN_INTERMOLECULAR",
]
