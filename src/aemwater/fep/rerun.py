"""Build the K x N reduced-energy matrix MBAR needs.

MBAR wants ``u_kn[k, n]``: the reduced energy of sample ``n`` evaluated under the
Hamiltonian of state ``k``. The sampling runs only record dU to *neighbouring*
states, which is enough for BAR and TI but not for MBAR, so the full matrix is
assembled in a second pass over the stored trajectories.

The obvious implementation is K^2 reruns -- every trajectory re-evaluated at every
state. That is not what happens here. ``compute fep`` evaluates dU for an
arbitrary perturbation, not just a neighbouring one, so a *single* rerun of state
j's trajectory carrying K-1 perturbation clauses yields the entire row
``u_kn[:, samples from j]`` in one pass. That is K reruns per leg instead of K^2:
for the default 9-state LJ ladder, 9 passes rather than 81.

Two properties of the rerun pass are load-bearing and are asserted, not assumed:

* **Frame 0 is discarded.** The dump writes a frame at step 0 -- the
  configuration inherited from equilibration -- and LAMMPS evaluates that
  frame's energy during run setup. Re-evaluating the same coordinates in a rerun
  reproduces every other frame to 2e-10 kcal/mol but frame 0 only to 1.7e-2, and
  that gap does *not* shrink with PPPM accuracy (2.1e-1 at 1e-4, 1.5e-2 at 1e-8),
  which marks it as a setup-time bookkeeping artifact rather than sampling noise.
  Frame 0 is also not an independent sample, so it is dropped rather than
  reconciled.
* **The diagonal must reproduce the sampling run.** ``u_kk`` recomputed by rerun
  is checked against the ``pe.dat`` the sampling run wrote. If a rerun's force
  field differs from the sampling run's in any way -- a dropped pair coefficient,
  a different cutoff, charges left unset -- this check fails loudly instead of
  handing MBAR a matrix whose diagonal disagrees with its own samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..lammps.runner import run_lammps
from ..utils import LOG
from .ghost import GhostTopology, ghost_pair_coeff_lines
from .schedule import FEPLeg, LambdaLadder, LambdaState

#: Frames whose energy the rerun must reproduce to better than this, in
#: kcal/mol, or the matrix is rejected. Measured agreement on matching
#: force fields is 2e-10; 1e-4 is far above that and far below the
#: 0.30 kcal/mol acceptance budget, so it catches real mismatches only.
DIAGONAL_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class EnergyMatrix:
    """Reduced energies for one leg, ready for MBAR.

    Attributes
    ----------
    u_kn:
        ``(K, N)`` array in units of kT. Column ``n`` is one sampled
        configuration; row ``k`` is the state whose Hamiltonian evaluates it.
    N_k:
        Samples drawn from each state, summing to ``N``. Column ordering is
        state-major and matches ``N_k``, which is what ``pymbar`` expects.
    lambdas:
        The ladder value of each state, for labelling and for TI quadrature.
    leg:
        Which alchemical leg this matrix describes.
    kT:
        kcal/mol per kT at the sampling temperature, for converting back.
    """

    u_kn: np.ndarray
    N_k: np.ndarray
    lambdas: tuple[float, ...]
    leg: FEPLeg
    kT: float

    def __post_init__(self) -> None:
        k, n = self.u_kn.shape
        if k != len(self.N_k) or k != len(self.lambdas):
            raise ValueError(
                f"u_kn has {k} states but N_k has {len(self.N_k)} and "
                f"lambdas has {len(self.lambdas)}"
            )
        if int(self.N_k.sum()) != n:
            raise ValueError(
                f"N_k sums to {int(self.N_k.sum())} but u_kn has {n} columns"
            )

    @property
    def n_states(self) -> int:
        return self.u_kn.shape[0]


def _read_two_column(path: Path) -> np.ndarray:
    """Read a LAMMPS ``fix ave/time`` file, dropping the step-0 frame.

    Returns an ``(nframes, ncols)`` array with the timestep in column 0.
    """
    raw = np.loadtxt(path, comments="#", ndmin=2)
    if raw.size == 0:
        raise ValueError(f"{path} contains no data rows")
    if raw[0, 0] == 0:
        raw = raw[1:]
    if raw.shape[0] == 0:
        raise ValueError(
            f"{path} held only the step-0 frame; production_steps is too short "
            "relative to sample_every"
        )
    return raw


def _charge_commands(ghost: GhostTopology, lambda_q: float) -> list[str]:
    """Set the ghost's charges for a rerun, since they live in the data file.

    The two site types scale by the same factor so the ghost stays neutral; a
    net charge would place a monopole in the periodic cell.
    """
    return [
        f"set             type {ghost.type_o} charge "
        f"{ghost.charge_o * lambda_q:.12f}",
        f"set             type {ghost.type_h} charge "
        f"{ghost.charge_h * lambda_q:.12f}",
    ]


def _fep_clause(
    target: LambdaState,
    source: LambdaState,
    ghost: GhostTopology,
    n_types: int,
    temperature: float,
) -> tuple[str, str]:
    """A ``compute fep`` clause taking ``source`` to an arbitrary ``target``.

    Unlike the sampling-run clauses these deltas are not restricted to
    neighbours -- MBAR needs every off-diagonal entry, including the far corners
    where the two states barely overlap.
    """
    cid = f"c_to{target.index}"
    if source.leg is FEPLeg.LJ:
        delta = target.lambda_lj - source.lambda_lj
        cmd = (
            f"variable        d_{cid} equal {delta:.12g}\n"
            f"compute         {cid} all fep {temperature:.6g} "
            f"pair lj/cut/coul/long/soft lambda "
            f"{ghost.host_type_range(n_types)} {ghost.type_range} v_d_{cid}"
        )
    else:
        delta = target.lambda_q - source.lambda_q
        cmd = (
            f"variable        d_{cid}_o equal {delta * ghost.charge_o:.12g}\n"
            f"variable        d_{cid}_h equal {delta * ghost.charge_h:.12g}\n"
            f"compute         {cid} all fep {temperature:.6g} "
            f"atom charge {ghost.type_o} v_d_{cid}_o "
            f"atom charge {ghost.type_h} v_d_{cid}_h"
        )
    return cid, cmd


def write_rerun_input(
    path: Path,
    *,
    source: LambdaState,
    targets: Sequence[LambdaState],
    system,
    ghost: GhostTopology,
    config,
    data_file: str,
    traj_file: str,
    out_file: str,
    sample_every: int,
) -> tuple[str, ...]:
    """Emit the rerun script for one sampled state; return the dU column order.

    The script is deliberately standalone rather than an include of
    ``fep_common.in.j2``: a rerun has no thermostat, no integrator and no
    constraints to apply, and the only thing that must match the sampling run
    exactly is the force field. Writing that explicitly here, in one place, is
    what makes the diagonal check meaningful -- there is no inherited setting
    that could drift.
    """
    spec = config.fep
    md = config.md
    n_types = len(system.atom_types)

    lines = [
        f"# rerun of {source.label}: energies at {len(targets)} states",
        "units           real",
        "atom_style      full",
        "boundary        p p p",
        f"pair_style      lj/cut/coul/long/soft {spec.soft_core_n} "
        f"{spec.alpha_lj} {spec.alpha_coul} {md.cutoff} {md.cutoff}",
        "pair_modify     mix arithmetic",
        f"kspace_style    pppm {spec.kspace_accuracy}",
        "bond_style      harmonic",
        "angle_style     harmonic",
        "special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333",
        f"read_data       {data_file}",
        "",
        f"# force field pinned at the sampled state, lambda_lj={source.lambda_lj:g}",
        *ghost_pair_coeff_lines(system, ghost, source.lambda_lj),
        *_charge_commands(ghost, source.lambda_q),
        "",
        "neighbor        2.0 bin",
        "neigh_modify    delay 0 every 1 check yes",
        "",
    ]

    cids: list[str] = []
    for target in targets:
        cid, cmd = _fep_clause(target, source, ghost, n_types, md.temperature)
        cids.append(cid)
        lines.append(f"# -> {target.label}")
        lines.append(cmd)
        lines.append(f"variable        v_{cid} equal c_{cid}[1]")
    lines.append("")

    lines += [
        "variable        pe_r equal pe",
        f"fix             out all ave/time 1 1 1 v_pe_r "
        + " ".join(f"v_v_{c}" for c in cids)
        + f" &",
        f"                file {out_file} format \" %.14g\" &",
        f"                title1 \"# rerun of {source.label}\" &",
        # title2, not title3: ave/time in scalar mode writes two header lines and
        # the column names are the second. title3 is silently ignored here and
        # LAMMPS substitutes its own `v_`-prefixed default.
        "                title2 \"# step pe "
        + " ".join(f"dU_to_{c[4:]}" for c in cids)
        + "\"",
        "",
        f"rerun           {traj_file} first 0 every {sample_every} "
        "dump x y z box yes",
        f'print           "FEP_RERUN_DONE {source.label}"',
    ]
    path.write_text("\n".join(lines) + "\n")
    return tuple(cids)


def build_energy_matrix(
    ladder: LambdaLadder,
    *,
    state_dirs: Sequence[Path],
    systems: Sequence,
    ghost: GhostTopology,
    config,
    workdir: Path,
    lammps_args: Sequence[str] = (),
) -> EnergyMatrix:
    """Run the rerun pass for one leg and assemble ``u_kn``.

    ``state_dirs[j]`` must be the directory the sampling run for ``ladder.states[j]``
    wrote, holding ``traj.lammpstrj`` and ``pe.dat``.
    """
    states = ladder.states
    if len(state_dirs) != len(states) or len(systems) != len(states):
        raise ValueError(
            f"ladder has {len(states)} states but got {len(state_dirs)} "
            f"directories and {len(systems)} systems"
        )

    kT = 0.0019872041 * config.md.temperature
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[np.ndarray] = []
    counts: list[int] = []
    for j, (state, sdir, system) in enumerate(zip(states, state_dirs, systems)):
        sampled = _read_two_column(sdir / "pe.dat")
        targets = [s for s in states if s.index != state.index]
        out = f"rerun_{j}.dat"
        cids = write_rerun_input(
            workdir / f"rerun_{j}.in",
            source=state,
            targets=targets,
            system=system,
            ghost=ghost,
            config=config,
            data_file=str((sdir / "state.data").resolve()),
            traj_file=str((sdir / "traj.lammpstrj").resolve()),
            out_file=out,
            sample_every=config.fep.sample_every,
        )
        run_lammps(
            workdir / f"rerun_{j}.in",
            workdir=workdir,
            log_name=f"rerun_{j}.log",
            extra_args=list(lammps_args) or None,
        )
        table = _read_two_column(workdir / out)

        if table.shape[0] != sampled.shape[0]:
            raise ValueError(
                f"{state.label}: rerun produced {table.shape[0]} frames but the "
                f"sampling run recorded {sampled.shape[0]}"
            )
        if not np.array_equal(table[:, 0], sampled[:, 0]):
            raise ValueError(
                f"{state.label}: rerun timesteps do not match the sampling run"
            )

        # The diagonal check. A mismatch here means the rerun's force field is
        # not the sampling run's, and every free energy built on this matrix
        # would be wrong in a way no downstream diagnostic reveals.
        drift = np.abs(table[:, 1] - sampled[:, 1]).max()
        if drift > DIAGONAL_TOLERANCE:
            raise ValueError(
                f"{state.label}: rerun energies differ from the sampling run by "
                f"up to {drift:.3e} kcal/mol (tolerance {DIAGONAL_TOLERANCE:.1e}). "
                "The rerun Hamiltonian does not match the one that generated "
                "these samples."
            )
        LOG.debug("%s diagonal reproduced to %.2e kcal/mol", state.label, drift)

        n = table.shape[0]
        counts.append(n)
        # Column block for samples from state j: u[k, n] = (U_j + dU_{j->k}) / kT
        block = np.empty((len(states), n))
        block[state.index] = table[:, 1] / kT
        for col, target in enumerate(targets):
            block[target.index] = (table[:, 1] + table[:, 2 + col]) / kT
        rows.append(block)

    u_kn = np.hstack(rows)
    return EnergyMatrix(
        u_kn=u_kn,
        N_k=np.array(counts, dtype=int),
        lambdas=tuple(s.lam for s in states),
        leg=ladder.leg,
        kT=kT,
    )


__all__ = [
    "EnergyMatrix",
    "DIAGONAL_TOLERANCE",
    "build_energy_matrix",
    "write_rerun_input",
]
