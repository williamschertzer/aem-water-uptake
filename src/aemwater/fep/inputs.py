"""Render fixed-lambda LAMMPS inputs for one alchemical state.

The two legs perturb different things, and ``compute fep`` expresses them with
different syntax:

* **LJ leg** -- one ``pair`` clause perturbing the soft-core ``lambda`` of every
  ghost-host type pair. One clause covers the whole leg.
* **Coulomb leg** -- two ``atom charge`` clauses, because the ghost oxygen and
  hydrogen carry different charges and must scale *proportionally*. A single
  clause with one delta would change the ghost's net charge away from zero and
  put a spurious monopole in a periodic cell, which PPPM would neutralise with a
  uniform background -- a large, silent artefact.

Both forms are validated against explicit finite differences in
``tests/test_fep_inputs.py``; agreement is 2.6e-11 kcal/mol on the LJ leg and
limited by PPPM grid resolution on the charge leg (hence
``FEPSpec.kspace_accuracy``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import FEPSpec
from ..lammps.inputs import render_input
from ..lammps.writer import LammpsSystem
from .ghost import GhostTopology, ghost_pair_coeff_lines
from .schedule import FEPLeg, LambdaState

__all__ = [
    "Perturbation",
    "perturbations_for",
    "render_state_input",
]


@dataclass(frozen=True)
class Perturbation:
    """One ``compute fep`` clause and the variable that reads its dU."""

    name: str
    compute_id: str
    compute_command: str
    comment: str
    #: Which neighbouring state this dU takes us toward, as a ladder index. The
    #: estimators need this to know which pair of states a column refers to; a
    #: column of dU values with no destination is unusable.
    target_index: int
    #: Signed lambda step, for TI and for provenance in the output header.
    delta: float


def perturbations_for(
    state: LambdaState,
    ghost: GhostTopology,
    system: LammpsSystem,
    ladder_lambdas: Sequence[float],
    spec: FEPSpec,
    temperature: float,
) -> tuple[Perturbation, ...]:
    """Perturbation clauses from ``state`` to each of its ladder neighbours.

    Neighbours only: MBAR wants the full matrix, but that comes from the rerun
    pass, which recomputes energies on stored frames rather than from these
    inline differences. What these clauses give is BAR-ready forward and reverse
    dU for adjacent pairs at zero extra sampling cost, plus the finite-difference
    pair for TI. Emitting all K perturbations inline instead would cost an extra
    energy evaluation per frame per state for data the rerun pass produces
    anyway.
    """
    n_types = len(system.atom_types)
    out: list[Perturbation] = []
    i = state.index

    # Forward and reverse neighbours for BAR.
    for label, j in (("fwd", i + 1), ("rev", i - 1)):
        if not 0 <= j < len(ladder_lambdas):
            continue
        delta = ladder_lambdas[j] - state.lam
        out.append(
            _build(state, ghost, n_types, delta, f"dU_{label}", j, temperature)
        )

    # Central finite difference for TI. Separate from the BAR clauses because the
    # ladder spacing is chosen for overlap, not for differentiation: at a 0.1 gap
    # a neighbour difference is a poor derivative, and near lambda = 0 the
    # soft-core dU/dlambda curves sharply.
    if "ti" in spec.estimators:
        for label, sign in (("ti_plus", +1.0), ("ti_minus", -1.0)):
            lam_probe = state.lam + sign * spec.ti_delta
            if not 0.0 <= lam_probe <= 1.0:
                # One-sided at the endpoints; the estimator handles the asymmetry.
                continue
            out.append(
                _build(state, ghost, n_types, sign * spec.ti_delta,
                       f"dU_{label}", i, temperature)
            )
    return tuple(out)


def _build(
    state: LambdaState,
    ghost: GhostTopology,
    n_types: int,
    delta: float,
    name: str,
    target: int,
    temperature: float,
) -> Perturbation:
    """One perturbation clause.

    ``compute fep`` takes the temperature because it also reports
    ``exp(-beta dU)`` in its second column. Only column 1 (the raw dU) is
    consumed downstream -- the estimators do their own Boltzmann weighting, so
    that a temperature typo shows up as an inconsistency rather than silently
    reweighting the result -- but the argument is mandatory, so it is passed
    correctly rather than as a placeholder.
    """
    cid = f"c_{name}"
    if state.leg is FEPLeg.LJ:
        cmd = (
            f"variable        d_{cid} equal {delta:.10g}\n"
            f"compute         {cid} all fep {temperature:.6g} "
            f"pair lj/cut/coul/long/soft lambda "
            f"{ghost.host_type_range(n_types)} {ghost.type_range} v_d_{cid}"
        )
        comment = (
            f"soft-core lambda {state.lam:.4g} -> {state.lam + delta:.4g} "
            f"on ghost-host pairs"
        )
    else:
        # Proportional charge scaling: both sites move by the same FRACTION so the
        # ghost stays neutral at every intermediate lambda.
        dq_o = delta * ghost.charge_o
        dq_h = delta * ghost.charge_h
        cmd = (
            f"variable        d_{cid}_o equal {dq_o:.10g}\n"
            f"variable        d_{cid}_h equal {dq_h:.10g}\n"
            f"compute         {cid} all fep {temperature:.6g} "
            f"atom charge {ghost.type_o} v_d_{cid}_o "
            f"atom charge {ghost.type_h} v_d_{cid}_h"
        )
        comment = (
            f"charge scale {state.lam:.4g} -> {state.lam + delta:.4g} "
            f"(dq_O = {dq_o:+.4f}, dq_H = {dq_h:+.4f}, net {dq_o + 2 * dq_h:+.1e})"
        )
    return Perturbation(
        name=name, compute_id=cid, compute_command=cmd,
        comment=comment, target_index=target, delta=delta,
    )


def render_state_input(
    state: LambdaState,
    *,
    directory: Path,
    system: LammpsSystem,
    ghost: GhostTopology,
    ladder_lambdas: Sequence[float],
    config,
    groups,
    constraints,
    comm_cutoff: float,
    data_file: str,
    seed: int,
    write_state: bool = False,
) -> dict:
    """Write one state's input file and return the paths it will produce.

    The returned mapping is the contract the reader and the estimators rely on:
    which file holds the dU columns, which holds the diagonal energies, which
    holds the trajectory, and what each dU column means. Returning it here rather
    than reconstructing the filenames later keeps one definition of the layout.
    """
    spec = config.fep
    perts = perturbations_for(
        state, ghost, system, ladder_lambdas, spec, config.md.temperature
    )
    directory.mkdir(parents=True, exist_ok=True)

    names = {
        "fep_file": "fep.dat",
        "pe_file": "pe.dat",
        "traj_file": "traj.lammpstrj",
        "out_data": "final.data",
    }
    render_input(
        "fep_state.in.j2",
        directory / "in.fep",
        title=f"{state.leg.value} leg, state {state.index} (lambda = {state.lam:g})",
        leg=state.leg.value,
        lambda_lj=state.lambda_lj,
        lambda_q=state.lambda_q,
        fep=spec,
        md=config.md,
        groups=groups,
        constraints=constraints,
        ghost=ghost,
        ghost_pair_coeffs=ghost_pair_coeff_lines(system, ghost, state.lambda_lj),
        comm_cutoff=comm_cutoff,
        data_file=data_file,
        perturbations=perts,
        seed=seed,
        write_state=write_state,
        **names,
    )
    return {
        "directory": directory,
        "input": directory / "in.fep",
        "perturbations": perts,
        **{k: directory / v for k, v in names.items()},
    }
