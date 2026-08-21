"""The ghost water: one SPC/E molecule carrying its own private atom types.

Why unique types
----------------
The alchemical coupling is applied through ``pair_coeff i j eps sigma lambda``,
which addresses *atom types*. If the ghost shared the resident waters' oxygen
type, scaling that type's lambda would decouple every water in the cell at once
and the computed free energy would be that of evaporating the whole liquid.

So the ghost gets duplicated types -- same mass, same epsilon, same sigma, a
distinct type index. Physically identical to a real water, alchemically separate.
The writer keys types on ``name|eps|sigma|mass``, so the duplicate is created by
renaming: ``OW`` -> ``OW_G``.

Why the charges are zero in the data file
-----------------------------------------
Leg 1 grows the LJ core with electrostatics off, so the ghost is written neutral.
Leg 2 scales the charges up, and does so by writing each state's data file with
``q = lambda_Q * q_full`` rather than by using ``fix adapt/fep`` on the charge:
PPPM caches per-atom charge in its grid setup, and a data-file charge is
unambiguous in a way a runtime-modified one is not. It also means every state's
input is a self-contained, re-runnable directory.

Net charge
----------
A neutral ghost keeps the cell's net charge exactly where it was, so PPPM's
neutralising background does not change between legs. On leg 2 the ghost is
neutral *as a molecule* at every lambda_Q (SPC/E sums to zero), so this holds
throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import parmed as pmd

from ..forcefield.builders import water_structure
from ..forcefield.water import WaterModel, water_model
from ..lammps.writer import LammpsSystem, LammpsWriteError
from ..utils import LOG

#: Residue name for the ghost. Distinct from ``WAT`` so that SHAKE grouping,
#: group definitions and the water-type lookup in the writer continue to see
#: only the real waters, and so a ghost can never be counted as uptake.
GHOST_RESIDUE = "GHO"

#: Suffix appended to the water atom type names to make the ghost's own types.
GHOST_SUFFIX = "_G"

#: Sigma written for pairs whose epsilon is zero. The soft styles divide by
#: sigma and refuse sigma = 0; with epsilon = 0 the pair energy is identically
#: zero for any sigma, so this value is inert. See ``ghost_pair_coeff_lines``.
SIGMA_PLACEHOLDER = 1.0


@dataclass(frozen=True)
class GhostTopology:
    """Where the ghost ended up, in the numbering LAMMPS will use.

    Returned by :func:`add_ghost_water` and threaded into every template, so the
    ``pair_coeff`` lines, the ``compute fep`` type ranges and the charge-scaling
    all address the same atoms.
    """

    type_o: int
    type_h: int
    bond_type: int
    angle_type: int
    molecule_id: int
    atom_indices: tuple[int, int, int]
    charge_o: float
    charge_h: float
    #: LJ lives on the oxygen only -- all three-site water models give the
    #: hydrogens zero epsilon -- so the hydrogen LJ parameters are carried
    #: explicitly as zeros rather than left implicit, because the tail
    #: correction sums over all three sites and a silent absence there would
    #: look like a contribution of zero for the wrong reason.
    epsilon_o: float
    sigma_o: float
    epsilon_h: float = 0.0
    sigma_h: float = 0.0

    @property
    def types(self) -> tuple[int, int]:
        return (self.type_o, self.type_h)

    @property
    def type_range(self) -> str:
        """LAMMPS type range for ``compute fep`` / ``fix adapt/fep``.

        Emitted as ``lo*hi`` when the two ghost types are adjacent, which they
        are by construction (both appended at the end of the type table).
        """
        lo, hi = min(self.types), max(self.types)
        return f"{lo}*{hi}" if hi != lo else str(lo)

    def host_type_range(self, n_types: int) -> str:
        """Range covering every non-ghost type, i.e. the host system."""
        lo, hi = min(self.types), max(self.types)
        if hi != n_types or lo != n_types - 1:
            raise LammpsWriteError(
                f"ghost types {self.types} are not the last two of {n_types}; "
                "the host range would not be contiguous"
            )
        return f"1*{n_types - 2}"


def _ghost_structure(model: WaterModel) -> pmd.Structure:
    """A single neutral water with ghost type names, oxygen at the origin.

    Built by taking the workflow's own :func:`water_structure` and renaming its
    types, rather than constructing a molecule from scratch. That guarantees the
    ghost is parameterically identical to a resident water -- same LJ, same
    geometry, same bond and angle constants -- which is the premise of the whole
    calculation. A hand-built duplicate would be one edit away from disagreeing.

    Two changes are made to the copy:

    * type names get :data:`GHOST_SUFFIX`, which is what makes the writer assign
      fresh type indices (it keys on ``name|eps|sigma|mass``);
    * charges are zeroed, because leg 1 runs uncharged and leg 2 writes scaled
      charges per state.
    """
    struct = pmd.structure.copy(water_structure(model))
    for atom in struct.atoms:
        atom.type = f"{atom.type}{GHOST_SUFFIX}"
        if atom.atom_type is not None:
            atom.atom_type.name = f"{atom.atom_type.name}{GHOST_SUFFIX}"
        atom.charge = 0.0
    for residue in struct.residues:
        residue.name = GHOST_RESIDUE
    return struct


def _far_from_everything(coords: np.ndarray, box: np.ndarray, rng: np.random.Generator,
                         n_trials: int = 4000) -> np.ndarray:
    """A position for the ghost oxygen with the largest clearance found.

    The ghost at ``lambda = 0`` is an ideal-gas particle and its starting
    position does not bias the free energy -- it explores the box during the run.
    But starting it inside a polymer backbone means the first few steps of the
    *coupled* states integrate a large force, so the cheapest insurance is to
    start in the roomiest spot available. This is initialisation quality, not
    physics.
    """
    if len(coords) == 0:
        return np.asarray(box[:3], dtype=float) / 2.0
    trials = rng.random((n_trials, 3)) * np.asarray(box[:3], dtype=float)
    # Minimum-image distance from every trial point to every atom.
    best, best_d = trials[0], -1.0
    edge = np.asarray(box[:3], dtype=float)
    for point in trials:
        d = coords - point
        d -= edge * np.round(d / edge)
        dmin = float(np.sqrt((d ** 2).sum(axis=1)).min())
        if dmin > best_d:
            best, best_d = point, dmin
    LOG.debug("ghost placed with %.2f A clearance", best_d)
    return best


def add_ghost_water(
    system: LammpsSystem,
    model: WaterModel | str = "spce",
    seed: int = 1,
) -> tuple[LammpsSystem, GhostTopology]:
    """Append one ghost water to ``system``, returning the new system and its topology.

    The input system is not modified. The ghost's atoms go *last*, so its types
    are the final two in the type table and every existing type index is
    unchanged -- which is what lets a morphology equilibrated without a ghost be
    reused directly.
    """
    resolved = water_model(model) if isinstance(model, str) else model
    host = system.structure
    if host.box is None:
        raise LammpsWriteError("host structure has no box; cannot place a ghost")

    ghost = _ghost_structure(resolved)
    rng = np.random.default_rng(seed)
    box = np.asarray(host.box, dtype=float)
    origin = _far_from_everything(
        np.asarray(host.coordinates, dtype=float), box, rng
    )
    ghost.coordinates = np.asarray(ghost.coordinates, dtype=float) + origin

    combined = host + ghost
    combined.box = host.box

    new_system = LammpsSystem(
        structure=combined,
        water_residue=system.water_residue,
        ion_residues=system.ion_residues,
    )

    n_host_atoms = len(host.atoms)
    ghost_atoms = combined.atoms[n_host_atoms:]
    if len(ghost_atoms) != 3:
        raise LammpsWriteError(
            f"ghost contributed {len(ghost_atoms)} atoms, expected 3"
        )

    key = new_system._atom_type_key
    type_o = new_system.atom_types[key(ghost_atoms[0])]
    type_h = new_system.atom_types[key(ghost_atoms[1])]
    if type_o == type_h:
        raise LammpsWriteError(
            "ghost O and H folded onto one atom type; the type key is not "
            "distinguishing them"
        )

    real_o, real_h = (system.water_atom_types() if system.has_water() else (None, None))
    if real_o is not None and type_o in (real_o, real_h):
        raise LammpsWriteError(
            f"ghost oxygen took type {type_o}, which is already a resident water "
            "type. Scaling it would decouple every water in the cell."
        )

    ghost_bond = next(b for b in combined.bonds
                      if b.atom1.residue.name == GHOST_RESIDUE)
    ghost_angle = next(a for a in combined.angles
                       if a.atom1.residue.name == GHOST_RESIDUE)
    topology = GhostTopology(
        type_o=type_o,
        type_h=type_h,
        bond_type=new_system.bond_types[new_system._bond_key(ghost_bond)],
        angle_type=new_system.angle_types[new_system._angle_key(ghost_angle)],
        molecule_id=len({a.residue.idx for a in combined.atoms}),
        atom_indices=(n_host_atoms + 1, n_host_atoms + 2, n_host_atoms + 3),
        charge_o=float(resolved.charge_O),
        charge_h=float(resolved.charge_H),
        # Read off the assembled structure rather than the model, so these are
        # exactly the numbers the data file will carry.
        epsilon_o=float(ghost_atoms[0].epsilon or 0.0),
        sigma_o=float(ghost_atoms[0].sigma or 0.0),
        epsilon_h=float(ghost_atoms[1].epsilon or 0.0),
        sigma_h=float(ghost_atoms[1].sigma or 0.0),
    )
    LOG.info(
        "ghost water added: types O=%d H=%d, %d atom types total",
        type_o, type_h, len(new_system.atom_types),
    )
    return new_system, topology


def ghost_pair_coeff_lines(
    system: LammpsSystem,
    ghost: GhostTopology,
    lambda_lj: float,
) -> list[str]:
    """Every ``pair_coeff i j`` line, with ghost-host pairs at ``lambda_lj``.

    The soft styles refuse to mix two different lambdas -- ``ERROR: Pair
    lj/cut/soft different lambda values in mix`` -- so every cross term must be
    written out. That error is a safety net, not a nuisance: a forgotten pair
    fails at setup rather than sampling the wrong Hamiltonian.

    Mixing follows GAFF2's Lorentz-Berthelot convention (arithmetic sigma,
    geometric epsilon), matching ``pair_modify mix arithmetic`` in the production
    templates. Computing it here rather than delegating to LAMMPS is what makes
    the per-pair lambda expressible at all.
    """
    coeffs = {tid: (eps, sig, name) for tid, eps, sig, name in system.pair_coeffs()}
    n_types = len(system.atom_types)
    missing = set(range(1, n_types + 1)) - set(coeffs)
    if missing:
        raise LammpsWriteError(f"no pair coefficients for atom types {sorted(missing)}")

    ghost_types = set(ghost.types)
    lines: list[str] = []
    for i in range(1, n_types + 1):
        for j in range(i, n_types + 1):
            eps_i, sig_i, name_i = coeffs[i]
            eps_j, sig_j, name_j = coeffs[j]
            eps = math.sqrt(eps_i * eps_j)
            sig = 0.5 * (sig_i + sig_j)
            if sig <= 0.0:
                # The soft styles reject sigma = 0 ("Incorrect args for pair
                # coefficients"): their denominator contains (r/sigma)^6, so zero
                # is a division by zero rather than a degenerate-but-valid case.
                # The plain lj/cut styles accept it because they precompute
                # sigma^6 and never divide.
                #
                # Only pairs between two LJ-free sites reach here -- water
                # hydrogens, which every three-site model gives zero epsilon --
                # so the interaction is identically zero whatever sigma says.
                # Verified: eps = 0 gives evdwl = 0 exactly for sigma in
                # {1, 2, 3, 5} A, so the placeholder is inert, not merely small.
                sig = SIGMA_PLACEHOLDER
            # Ghost-host pairs carry the coupling. Ghost-ghost is intramolecular
            # only (one molecule, all three atoms excluded by special_bonds), so
            # its lambda is immaterial and is pinned at 1 for clarity.
            i_ghost, j_ghost = i in ghost_types, j in ghost_types
            lam = lambda_lj if (i_ghost != j_ghost) else 1.0
            tag = "ghost-host" if i_ghost != j_ghost else ("ghost" if i_ghost else "host")
            lines.append(
                f"pair_coeff      {i} {j} {eps:.8f} {sig:.8f} {lam:.6f}"
                f"  # {name_i}-{name_j} [{tag}]"
            )
    return lines


def scale_ghost_charges(
    system: LammpsSystem,
    ghost: GhostTopology,
    lambda_q: float,
) -> LammpsSystem:
    """A copy of ``system`` with the ghost's charges scaled by ``lambda_q``.

    Leg 2 is applied by writing each state's charges into its own data file
    rather than by ``fix adapt/fep ... atom charge``. Two reasons, in order of
    importance:

    * PPPM builds its charge grid at setup. A data-file charge is the charge for
      the whole run, with no question about when the k-space solver saw it.
    * Each lambda directory becomes a self-contained, re-runnable input. Handing
      someone a state means handing them a data file, not a data file plus the
      fix that mutates it.

    The ghost stays neutral as a molecule at every ``lambda_q`` (SPC/E sums to
    zero, so scaling all three sites by the same factor preserves that), which
    keeps PPPM's neutralising background out of the alchemical path.
    """
    if not 0.0 <= lambda_q <= 1.0:
        raise ValueError(f"lambda_q must be in [0, 1], got {lambda_q}")
    scaled = pmd.structure.copy(system.structure)
    scaled.box = system.structure.box
    for index in ghost.atom_indices:
        atom = scaled.atoms[index - 1]  # atom_indices are 1-based LAMMPS ids
        if atom.residue.name != GHOST_RESIDUE:
            raise LammpsWriteError(
                f"atom {index} is in residue {atom.residue.name}, not {GHOST_RESIDUE}; "
                "the ghost indices do not match this structure"
            )
        full = ghost.charge_o if atom.atomic_number == 8 else ghost.charge_h
        atom.charge = float(lambda_q) * full
    net = sum(scaled.atoms[i - 1].charge for i in ghost.atom_indices)
    if abs(net) > 1.0e-9:
        raise LammpsWriteError(
            f"scaled ghost carries net charge {net:+.3e} e at lambda_q={lambda_q}; "
            "it must stay neutral so PPPM's background does not change"
        )
    return LammpsSystem(
        structure=scaled,
        water_residue=system.water_residue,
        ion_residues=system.ion_residues,
    )


def tail_correction(
    system: LammpsSystem,
    ghost: GhostTopology,
    cutoff: float,
) -> float:
    """Analytic long-range LJ correction for coupling the ghost, kcal/mol.

    ``compute fep`` refuses ``tail yes`` for soft pair styles ("Compute fep tail
    when pair style does not compute tail corrections"), while the production
    templates run ``pair_modify tail yes``. Dropping the term silently would
    shift mu_ex by a fraction of a kcal/mol -- the same order as the difference
    the saturation criterion resolves -- so it is computed here and added to the
    LJ leg.

    For adding one molecule to a homogeneous fluid, the isotropic correction is

        dU_tail = (8 pi / 3V) * sum_{g in ghost} sum_j N_j eps_gj sigma_gj^3
                                [ (1/3)(sigma_gj/rc)^9 - (sigma_gj/rc)^3 ]

    the standard result for the LJ tail beyond ``rc`` assuming g(r) = 1 there.
    It is applied identically to the bulk and membrane runs, so whatever residual
    error the g(r) = 1 assumption carries cancels in the difference, exactly as
    the cutoff and kspace settings already do.
    """
    struct = system.structure
    volume = float(np.prod(np.asarray(struct.box, dtype=float)[:3]))
    if volume <= 0:
        raise LammpsWriteError(f"non-positive cell volume {volume}")

    ghost_types = set(ghost.types)
    counts: dict[int, int] = {}
    params: dict[int, tuple[float, float]] = {}
    key = system._atom_type_key
    for atom in struct.atoms:
        tid = system.atom_types[key(atom)]
        if tid in ghost_types:
            continue
        counts[tid] = counts.get(tid, 0) + 1
        params.setdefault(tid, (atom.epsilon or 0.0, atom.sigma or 0.0))

    ghost_params = [
        (ghost.epsilon_o, ghost.sigma_o),
        (ghost.epsilon_h, ghost.sigma_h),
        (ghost.epsilon_h, ghost.sigma_h),
    ]

    total = 0.0
    for eps_g, sig_g in ghost_params:
        for tid, n_j in counts.items():
            eps_j, sig_j = params[tid]
            eps = math.sqrt(eps_g * eps_j)
            sig = 0.5 * (sig_g + sig_j)
            if eps == 0.0 or sig == 0.0:
                continue
            x = sig / float(cutoff)
            total += n_j * eps * sig ** 3 * (x ** 9 / 3.0 - x ** 3)
    return 8.0 * math.pi / (3.0 * volume) * total


__all__ = [
    "GHOST_RESIDUE",
    "GHOST_SUFFIX",
    "GhostTopology",
    "add_ghost_water",
    "ghost_pair_coeff_lines",
    "scale_ghost_charges",
    "tail_correction",
]
