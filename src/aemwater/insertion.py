"""Find cavities in a configuration and place water molecules in them.

Why geometric insertion at all
------------------------------
Random insertion into a dense polymer fails almost always: at 1.1 g/cm^3 the
accessible free volume is a few percent, so a uniformly sampled point is inside
an atom's excluded volume with overwhelming probability. Grand-canonical MC
solves this correctly but converges slowly in a glassy matrix where the polymer
must relax to accommodate each water.

So insertion is split: geometry proposes, dynamics accepts. This module finds
positions that are *sterically* allowed -- a cavity large enough for a water
oxygen without overlapping any existing atom's van der Waals radius -- and the
subsequent MD relaxation decides whether they are *energetically* reasonable. A
water inserted into a marginal cavity gets squeezed out into a better one during
the NPT stage, which is exactly the behaviour we want.

Method
------
1. Rasterise the cell onto a grid of spacing ``grid_spacing``.
2. Mark every grid point whose distance to the nearest atom exceeds that atom's
   scaled van der Waals radius plus the water probe radius. This is a solvent-
   accessible-volume test, not a simple distance cutoff, because a carbon and a
   hydrogen exclude very different volumes.
3. Keep candidate points separated by at least ``water_water_min`` from each
   other, so a single large cavity receives several waters rather than one.
4. Rank candidates by cavity depth (distance to the nearest atom surface) and
   fill the deepest first: those are the positions most likely to survive
   relaxation, and in an AEM they preferentially sit near the ionic groups.

All distances use the minimum image convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .utils import LOG

#: Van der Waals radii (A), Bondi 1964 with the Rowland-Taylor hydrogen.
VDW_RADII: dict[str, float] = {
    "H": 1.10,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Na": 2.27,
    "K": 2.75,
}
DEFAULT_RADIUS = 1.70

#: Radius of a water oxygen for probe purposes (A). Deliberately smaller than
#: the SPC/E sigma/2 = 1.58: a cavity that tight is acceptable because MD will
#: relax it, and requiring the full radius finds almost no sites in a dry
#: membrane.
WATER_PROBE_RADIUS = 1.40


class InsertionError(RuntimeError):
    """Raised when insertion cannot proceed."""


@dataclass
class VoidMap:
    """Candidate insertion sites in one configuration, deepest cavity first."""

    positions: np.ndarray            # (n, 3) candidate centres
    depths: np.ndarray               # (n,) clearance beyond the vdW surface, A
    edge: float
    grid_spacing: float
    free_volume_fraction: float
    n_grid_points: int

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def summary(self) -> dict[str, object]:
        return {
            "candidate_sites": len(self),
            "free_volume_fraction": round(self.free_volume_fraction, 5),
            "grid_points": self.n_grid_points,
            "grid_spacing_A": self.grid_spacing,
            "max_depth_A": round(float(self.depths.max()), 3) if len(self) else 0.0,
            "median_depth_A": round(float(np.median(self.depths)), 3) if len(self) else 0.0,
        }


def atom_radii(elements: list[str], scale: float = 1.0) -> np.ndarray:
    """Van der Waals radii for ``elements``, scaled by ``scale``."""
    return np.array([VDW_RADII.get(e, DEFAULT_RADIUS) for e in elements]) * scale


def map_voids(
    coordinates: np.ndarray,
    elements: list[str],
    edge: float,
    grid_spacing: float = 0.7,
    probe_radius: float = WATER_PROBE_RADIUS,
    vdw_scale: float = 1.0,
    min_site_separation: float = 2.6,
) -> VoidMap:
    """Locate water-sized cavities in a periodic configuration.

    Parameters
    ----------
    grid_spacing:
        Grid resolution in A. 0.7 A resolves the cavities that matter while
        keeping a 30 A cell to ~80k points, which is a fraction of a second.
    vdw_scale:
        Multiplier on the van der Waals radii. Below 1.0 the matrix is treated as
        softer, finding more (tighter) sites; the MD relaxation is what makes
        that legitimate.
    min_site_separation:
        Minimum spacing between accepted sites (A). Roughly the O-O distance in
        liquid water (2.8 A), slightly reduced -- adjacent waters in a cavity
        should be able to hydrogen bond.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape[0] != len(elements):
        raise InsertionError(
            f"{coordinates.shape[0]} coordinates but {len(elements)} elements"
        )
    radii = atom_radii(elements, vdw_scale)

    n = max(4, int(np.ceil(edge / grid_spacing)))
    axis = (np.arange(n) + 0.5) * (edge / n)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    wrapped = np.mod(coordinates, edge)
    tree = cKDTree(wrapped, boxsize=edge)

    # A point is free if it clears *every* atom's own radius. Query enough
    # neighbours to cover the largest possible exclusion, then test each.
    max_reach = float(radii.max() + probe_radius)
    neighbours = tree.query_ball_point(grid, max_reach, return_sorted=False)
    depths = np.empty(len(grid))
    k_fallback = min(8, len(wrapped))
    for i, idx in enumerate(neighbours):
        if not idx:
            # Deep vacuum: nothing within the largest possible exclusion. The
            # depth still has to be finite, otherwise every such point ties at
            # infinity and the deepest-first ranking becomes arbitrary. The
            # radii span only ~0.75 A, so the nearest few atoms always contain
            # the true minimum-clearance one.
            _, idx = tree.query(grid[i], k=k_fallback)
            idx = np.atleast_1d(idx)
        d = grid[i] - wrapped[idx]
        d -= edge * np.round(d / edge)
        clearance = np.linalg.norm(d, axis=1) - radii[idx]
        depths[i] = clearance.min()

    free = depths >= probe_radius
    free_fraction = float(free.mean())
    candidates = grid[free]
    candidate_depths = depths[free]

    # Deepest first, then greedily thin to enforce site separation. Greedy
    # selection on a depth-sorted list gives the deepest point in each cavity.
    order = np.argsort(-candidate_depths)
    candidates, candidate_depths = candidates[order], candidate_depths[order]
    kept: list[int] = []
    if len(candidates):
        ctree = cKDTree(np.mod(candidates, edge), boxsize=edge)
        blocked = np.zeros(len(candidates), dtype=bool)
        for i in range(len(candidates)):
            if blocked[i]:
                continue
            kept.append(i)
            for j in ctree.query_ball_point(candidates[i], min_site_separation):
                if j > i:
                    blocked[j] = True

    vm = VoidMap(
        positions=candidates[kept] if kept else np.empty((0, 3)),
        depths=candidate_depths[kept] if kept else np.empty(0),
        edge=edge,
        grid_spacing=edge / n,
        free_volume_fraction=free_fraction,
        n_grid_points=len(grid),
    )
    LOG.info(
        "void map: %d sites in %.1f%% free volume (probe %.2f A, %d grid points)",
        len(vm),
        100 * free_fraction,
        probe_radius,
        len(grid),
    )
    return vm


def water_orientations(
    n: int,
    water_model,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` randomly oriented rigid water geometries, oxygen at the origin."""
    import math

    half = math.radians(water_model.angle_HOH) / 2.0
    r = water_model.r_OH
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [r * math.sin(half), 0.0, r * math.cos(half)],
            [-r * math.sin(half), 0.0, r * math.cos(half)],
        ]
    )
    from .packing import _random_rotation

    return np.stack([local @ _random_rotation(rng).T for _ in range(n)])


@dataclass
class InsertionResult:
    """Outcome of one insertion batch."""

    n_requested: int
    n_inserted: int
    coordinates: np.ndarray          # (3 * n_inserted, 3) O,H,H per water
    site_depths: np.ndarray
    void_map: VoidMap

    @property
    def saturated(self) -> bool:
        """True when geometry could not supply the requested number of sites."""
        return self.n_inserted < self.n_requested

    def summary(self) -> dict[str, object]:
        return {
            "requested": self.n_requested,
            "inserted": self.n_inserted,
            "geometrically_saturated": self.saturated,
            **self.void_map.summary(),
        }


def insert_waters(
    coordinates: np.ndarray,
    elements: list[str],
    edge: float,
    n_waters: int,
    water_model,
    grid_spacing: float = 0.7,
    probe_radius: float = WATER_PROBE_RADIUS,
    vdw_scale: float = 1.0,
    water_water_min: float = 2.6,
    seed: int = 0,
) -> InsertionResult:
    """Place up to ``n_waters`` rigid waters into the cavities of a configuration.

    Returns fewer than requested when the geometry is saturated -- that shortfall
    is one of the two saturation signals the driver watches (the other is the
    Widom chemical potential).

    Hydrogens are placed by random rotation about the oxygen and are *not*
    clash-tested: a hydrogen 1 A from the oxygen in a cavity that already clears
    a 1.4 A probe is at worst mildly strained, and the settle stage resolves it.
    Rejecting orientations here would bias the inserted water dipoles.
    """
    vm = map_voids(
        coordinates,
        elements,
        edge,
        grid_spacing=grid_spacing,
        probe_radius=probe_radius,
        vdw_scale=vdw_scale,
        min_site_separation=water_water_min,
    )
    n_place = min(n_waters, len(vm))
    if n_place == 0:
        LOG.info("no insertable cavities found: geometry is saturated")
        return InsertionResult(n_waters, 0, np.empty((0, 3)), np.empty(0), vm)

    rng = np.random.default_rng(seed)
    sites = vm.positions[:n_place]
    depths = vm.depths[:n_place]
    geometries = water_orientations(n_place, water_model, rng)
    placed = (geometries + sites[:, None, :]).reshape(-1, 3)

    LOG.info(
        "inserted %d/%d waters (deepest site %.2f A, shallowest used %.2f A)",
        n_place,
        n_waters,
        float(depths[0]),
        float(depths[-1]),
    )
    return InsertionResult(n_waters, n_place, placed, depths, vm)


def free_volume_profile(
    coordinates: np.ndarray,
    elements: list[str],
    edge: float,
    probes: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6, 1.8),
    grid_spacing: float = 0.8,
) -> dict[float, float]:
    """Accessible free-volume fraction as a function of probe radius.

    A diagnostic, not part of the driver: the curve shows whether a membrane has
    a few large channels or many small isolated cavities, which is what
    determines whether uptake is percolating or not.
    """
    out = {}
    for r in probes:
        vm = map_voids(coordinates, elements, edge, grid_spacing=grid_spacing,
                       probe_radius=r, min_site_separation=grid_spacing)
        out[r] = vm.free_volume_fraction
    return out


__all__ = [
    "VoidMap",
    "InsertionResult",
    "InsertionError",
    "map_voids",
    "insert_waters",
    "free_volume_profile",
    "water_orientations",
    "atom_radii",
    "VDW_RADII",
    "WATER_PROBE_RADIUS",
]
