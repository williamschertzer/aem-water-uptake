"""End-to-end FEP smoke test in a cell that actually contains polymer.

Every other LAMMPS-backed FEP test in this suite runs on pure water:
``CellContents(chains=[], ions=[])`` with ``n_polymer_molecules=0``. That
covers the estimators and the template plumbing but never once exercises the
case the project exists to measure -- a ghost water coupling to a GAFF2-typed
polyelectrolyte and its counterions, where the ghost-host pair table spans
polymer, ion and water types rather than water alone.

Two things make this test meaningful rather than decorative:

1. The cell is built at a realistic density (~1 g/cm^3). A first version of
   this test used a dilute hand-packed cell and the ghost ended up 8 A from
   every other atom, so dU was 0.04 kT -- a number that would have been
   reproduced exactly if the ghost-host coupling were entirely broken. The
   test would have passed with the feature deleted.
2. It asserts on the *magnitude* of dU and on the presence of polymer types in
   the ghost pair table, not merely that LAMMPS exited zero.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import BTMA_PS, needs_ambertools, needs_lammps

pytestmark = [pytest.mark.slow]

EDGE = 22.0
N_WATER = 320
GHOST_ATOMS = 3


def _grid(n: int, edge: float, start: float = 1.5):
    k = int(math.ceil(n ** (1 / 3)))
    step = (edge - 2 * start) / max(k - 1, 1)
    return [
        [i * step + start, j * step + start, m * step + start]
        for i in range(k) for j in range(k) for m in range(k)
    ][:n]


@pytest.fixture(scope="module")
def typed_chain(tmp_path_factory):
    """One short GAFF2-typed BTMA chain.

    ``charge_method="gas"`` keeps sqm out of the loop: AM1-BCC on this fragment
    is minutes, Gasteiger is a fraction of a second, and this test is about the
    FEP machinery seeing polymer types rather than about charge quality.
    """
    from aemwater.forcefield.gaff2 import GAFF2Backend
    from aemwater.polymer import build_chain

    chain = build_chain(BTMA_PS, 2, terminal_group="H", seed=1)
    typed, _ = GAFF2Backend(charge_method="gas").type_chain(
        chain, tmp_path_factory.mktemp("typing"))
    return typed


@pytest.fixture(scope="module")
def membrane(typed_chain):
    """A dense polymer + counterion + water cell with a ghost water added.

    Molecules are placed on a grid that excludes sites near the chain rather
    than through ``pack_cell`` + compression: this needs to be reproducible and
    fast, and the FEP machinery cannot tell a gridded cell from a packed one.
    Density is the property that matters here, and it is asserted below.
    """
    from aemwater.assembly import (CellContents, assemble, ion_molecules,
                                   water_molecules)
    from aemwater.fep.ghost import add_ghost_water

    ions = ion_molecules(2, "OH-")           # the chain carries net +2
    waters = water_molecules(N_WATER, "spce")
    contents = CellContents(chains=[typed_chain], ions=ions, waters=waters)

    chain_xyz = np.asarray(typed_chain.coordinates)
    chain_xyz = chain_xyz - chain_xyz.mean(0) + EDGE / 2

    need = len(ions) + len(waters)
    pts = np.array(_grid(need + 120, EDGE))
    gap = np.min(np.linalg.norm(pts[:, None, :] - chain_xyz[None, :, :], axis=2),
                 axis=1)
    sites = pts[gap > 3.0]
    assert len(sites) >= need, f"only {len(sites)} free sites for {need} molecules"

    coords = [chain_xyz]
    for i, mol in enumerate(list(ions) + list(waters)):
        xyz = np.asarray(mol.coordinates)
        coords.append(xyz - xyz.mean(0) + sites[i])

    system = assemble(contents, np.vstack(coords), edge=EDGE)

    mass = sum(a.mass for m in contents.molecules for a in m.atoms)
    density = mass / 6.02214076e23 / (EDGE * 1e-8) ** 3
    assert 0.85 < density < 1.15, (
        f"cell is {density:.2f} g/cm3. The whole point of this test is a ghost "
        "in a condensed phase; a dilute cell gives dU ~ 0 and would pass with "
        "the ghost-host coupling removed."
    )

    ghosted, ghost = add_ghost_water(system, "spce", seed=11)
    return ghosted, ghost, contents


def _config():
    from aemwater.config import PolymerSpec, RunConfig

    # 1000/50 is exactly the 20-frame floor fep config validation enforces.
    return RunConfig(
        polymer=PolymerSpec(smiles=BTMA_PS, chain_length=2)
    ).with_overrides(**{
        "fep.equil_steps": 25,
        "fep.production_steps": 1000,
        "fep.sample_every": 50,
    })


def _run_state(state, ladder, membrane, workdir):
    from aemwater.fep.ghost import scale_ghost_charges
    from aemwater.fep.inputs import render_state_input
    from aemwater.lammps.inputs import GroupSpec, comm_cutoff, constraint_spec
    from aemwater.lammps.runner import run_lammps
    from aemwater.lammps.writer import write_data_file

    ghosted, ghost, _ = membrane
    config = _config()
    workdir.mkdir(parents=True, exist_ok=True)

    scaled = scale_ghost_charges(ghosted, ghost, state.lambda_q)
    write_data_file(scaled, workdir / "state.data", "s", include_pair_coeffs=False)
    o_type, h_type = ghosted.water_atom_types()
    render_state_input(
        state, directory=workdir, system=scaled, ghost=ghost, config=config,
        groups=GroupSpec(n_polymer_molecules=1, n_ion_molecules=2,
                         water_type_o=o_type, water_type_h=h_type),
        constraints=constraint_spec(config.md, ghosted.water_bond_type(),
                                    ghosted.water_angle_type()),
        comm_cutoff=comm_cutoff(config.md), seed=3,
        ladder_lambdas=ladder.lambdas, data_file="state.data",
    )
    run_lammps(workdir / "in.fep", workdir=workdir, ranks=1)
    return workdir


@needs_lammps
@needs_ambertools
def test_ghost_is_solvated_by_the_polymer_not_sitting_in_a_void(membrane):
    """The ghost must have a real first shell, or the run measures nothing."""
    ghosted, _, _ = membrane
    xyz = np.asarray(ghosted.structure.coordinates)
    ghost_o, host = xyz[-GHOST_ATOMS], xyz[:-GHOST_ATOMS]

    delta = host - ghost_o
    delta -= EDGE * np.round(delta / EDGE)     # minimum image
    dist = np.linalg.norm(delta, axis=1)

    assert dist.min() < 5.0, f"nearest host atom is {dist.min():.1f} A away"
    assert (dist < 8.0).sum() > 50, (
        f"only {(dist < 8.0).sum()} atoms within 8 A of the ghost"
    )


@needs_lammps
@needs_ambertools
def test_ghost_host_pair_table_covers_polymer_and_ion_types(membrane, tmp_path):
    """Soft-core coefficients must exist for polymer and ion types too.

    On a water-only cell this is trivially satisfied by two water types, which
    is why the existing tests could not catch a ghost-host table that omitted
    polymer. Here the table has to span GAFF2 atom types as well.
    """
    from aemwater.fep.schedule import FEPLeg, LambdaLadder

    ghosted, ghost, _ = membrane
    ladder = LambdaLadder(leg=FEPLeg.LJ, lambdas=(0.0, 0.5, 1.0))
    d = _run_state(ladder.states[1], ladder, membrane, tmp_path / "lj")

    lines = [l for l in (d / "in.fep").read_text().splitlines()
             if l.strip().startswith("pair_coeff") and "[ghost-host]" in l]
    ghost_types = {ghost.type_o, ghost.type_h}
    partners, lam_by_partner = set(), {}
    for line in lines:
        f = line.split()
        i, j, lam = int(f[1]), int(f[2]), float(f[5])
        for t in {i, j} - ghost_types:
            partners.add(t)
            lam_by_partner.setdefault(t, set()).add(lam)

    water_types = set(ghosted.water_atom_types())
    assert partners - water_types, (
        "the ghost-host pair table only covers water types; the polymer and "
        "counterions are invisible to the ghost"
    )
    # Every non-ghost type in the system needs a coefficient, or the ghost
    # passes through part of the membrane with no LJ interaction at all.
    all_types = set(range(1, len(ghosted.atom_types) + 1))
    missing = all_types - partners - ghost_types
    assert not missing, f"no ghost-host LJ coefficients for types {sorted(missing)}"

    # Existence is not enough: each ghost-host pair must carry *this state's*
    # lambda. A polymer pair left at lambda = 1 is emitted, tagged ghost-host
    # and fully coupled at every state, so it never contributes to dU -- the
    # polymer silently drops out of the alchemical transformation while the
    # water part keeps working and the numbers stay plausible. Restricting the
    # coupling to water types passes every other assertion in this file, which
    # is why this one is here.
    lam_expected = ladder.states[1].lambda_lj
    wrong = {t: sorted(v) for t, v in lam_by_partner.items()
             if v != {pytest.approx(lam_expected)}}
    assert not wrong, (
        f"ghost-host pairs not at lambda_lj={lam_expected}: {wrong}. Types held "
        "at 1.0 are fully coupled in every state and cannot contribute to dU."
    )


@needs_lammps
@needs_ambertools
@pytest.mark.parametrize("leg", ["lj", "coul"])
def test_both_legs_produce_a_physically_sized_perturbation(membrane, tmp_path, leg):
    """dU must be many kT, which is what distinguishes this from a dilute cell.

    A ghost in a void gives |dU| ~ 0.04 kT. Creating or charging a water in a
    condensed phase is a several-kT perturbation, so a small |dU| here means
    the ghost is not really coupled to its surroundings -- exactly the silent
    failure a smoke test that only checked LAMMPS's exit code would miss.
    """
    from aemwater.fep.estimators import read_fep_columns
    from aemwater.fep.schedule import FEPLeg, LambdaLadder

    ladder = LambdaLadder(leg=FEPLeg.LJ if leg == "lj" else FEPLeg.COUL,
                          lambdas=(0.0, 0.5, 1.0))
    d = _run_state(ladder.states[1], ladder, membrane, tmp_path / leg)
    cols = read_fep_columns(d / "fep.dat")

    for name in ("dU_fwd", "dU_rev", "dU_ti_plus", "dU_ti_minus"):
        assert name in cols, f"{name} missing from fep.dat"
        assert len(cols[name]) >= 20, f"{name} has {len(cols[name])} frames"
        assert np.all(np.isfinite(cols[name])), f"{name} contains non-finite values"

    scale = max(abs(float(np.mean(cols["dU_fwd"]))),
                abs(float(np.mean(cols["dU_rev"]))))
    assert scale > 0.5, (
        f"{leg} leg: largest |<dU>| is only {scale:.3f} kT. The ghost is not "
        "coupled to the condensed phase around it."
    )

    # The finite-difference derivative must be non-trivial for the same reason,
    # and it is the quantity the TI cross-check integrates.
    config = _config()
    dudl = (float(np.mean(cols["dU_ti_plus"]))
            - float(np.mean(cols["dU_ti_minus"]))) / (2 * config.fep.ti_delta)
    assert abs(dudl) > 0.5, f"{leg} leg: dU/dlambda is {dudl:.3f} kT"


@needs_lammps
@needs_ambertools
def test_ghost_types_are_distinct_from_the_real_water_types(membrane):
    """Unique ghost types are the premise of the whole scheme.

    If the ghost shared O/H types with the bulk waters, scaling its
    interactions would scale every water in the cell.
    """
    ghosted, ghost, _ = membrane
    assert {ghost.type_o, ghost.type_h} & set(ghosted.water_atom_types()) == set()
