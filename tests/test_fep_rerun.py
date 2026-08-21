"""The rerun pass and the K x N matrix it assembles.

The matrix is where a silent error is most expensive: MBAR will happily return a
confident free energy from a matrix whose diagonal disagrees with its own
samples, and nothing downstream reveals it. So the properties tested here are
mostly guards -- that the diagonal is checked, that frame 0 is dropped, that a
mismatched Hamiltonian raises rather than degrades.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import pytest

from aemwater.assembly import CellContents, assemble, water_molecules
from aemwater.config import PolymerSpec, RunConfig
from aemwater.fep.ghost import add_ghost_water, scale_ghost_charges
from aemwater.fep.inputs import render_state_input
from aemwater.fep.rerun import (
    DIAGONAL_TOLERANCE,
    EnergyMatrix,
    build_energy_matrix,
    write_rerun_input,
)
from aemwater.fep.schedule import FEPLeg, LambdaLadder
from aemwater.lammps.inputs import ConstraintSpec, GroupSpec
from aemwater.lammps.writer import write_data_file
from conftest import BTMA_PS, lammps_binary, needs_lammps


def _water_cell(n, edge):
    k = int(math.ceil(n ** (1 / 3)))
    spacing = edge / k
    points = [
        [i * spacing + 1.0, j * spacing + 1.0, m * spacing + 1.0]
        for i in range(k)
        for j in range(k)
        for m in range(k)
    ][:n]
    waters = water_molecules(n, "spce")
    coords = np.vstack(
        [np.asarray(w.coordinates) + np.array(p) for w, p in zip(waters, points)]
    )
    return assemble(CellContents(chains=[], ions=[], waters=waters), coords, edge)


@pytest.fixture(scope="module")
def ghosted():
    return add_ghost_water(_water_cell(195, 18.0), "spce", seed=7)


@pytest.fixture
def cfg():
    return RunConfig(polymer=PolymerSpec(smiles=BTMA_PS)).with_overrides(
        **{
            "fep.equil_steps": 100,
            "fep.production_steps": 2000,
            "fep.sample_every": 100,
        }
    )


@pytest.fixture
def ladder():
    """Three states only: enough to exercise the matrix, fast enough to run."""
    return LambdaLadder(leg=FEPLeg.LJ, lambdas=(0.0, 0.4, 1.0))


def _sample(ladder, ghosted, cfg, tmp_path):
    """Run every state of the ladder; return its directories and systems."""
    system, ghost = ghosted
    groups = GroupSpec(
        n_polymer_molecules=0, n_ion_molecules=0, water_type_o=1, water_type_h=2
    )
    constraints = ConstraintSpec(
        shake_water=True,
        shake_hydrogen=False,
        water_bond_type=ghost.bond_type,
        water_angle_type=ghost.angle_type,
    )
    dirs, systems = [], []
    for state in ladder.states:
        d = tmp_path / f"s{state.index}"
        d.mkdir()
        scaled = scale_ghost_charges(system, ghost, state.lambda_q)
        write_data_file(scaled, d / "state.data", "s", include_pair_coeffs=False)
        render_state_input(
            state,
            directory=d,
            system=scaled,
            ghost=ghost,
            ladder_lambdas=ladder.lambdas,
            config=cfg,
            groups=groups,
            constraints=constraints,
            comm_cutoff=14.0,
            data_file="state.data",
            seed=100 + state.index,
        )
        proc = subprocess.run(
            [lammps_binary(), "-in", "in.fep", "-log", "run.log", "-screen", "none"],
            cwd=d, capture_output=True, text=True,
        )
        assert proc.returncode == 0, (d / "run.log").read_text()[-2000:]
        dirs.append(d)
        systems.append(scaled)
    return dirs, systems


# ------------------------------------------------------------ script contents --


def test_rerun_script_pins_the_sampled_state_and_perturbs_the_rest(
    tmp_path, ghosted, cfg, ladder
):
    system, ghost = ghosted
    states = ladder.states
    source = states[1]
    cids = write_rerun_input(
        tmp_path / "r.in",
        source=source,
        targets=[s for s in states if s.index != source.index],
        system=system,
        ghost=ghost,
        config=cfg,
        data_file="state.data",
        traj_file="traj.lammpstrj",
        out_file="out.dat",
        sample_every=cfg.fep.sample_every,
    )
    text = (tmp_path / "r.in").read_text()
    # One clause per other state -- the whole matrix row in a single pass, not
    # one rerun per (source, target) pair.
    assert len(cids) == len(states) - 1
    assert text.count("compute         c_to") == len(states) - 1
    assert "rerun           traj.lammpstrj" in text


def test_rerun_script_reaches_non_neighbouring_states(
    tmp_path, ghosted, cfg, ladder
):
    """MBAR needs the far corners, which the inline sampling clauses never touch."""
    system, ghost = ghosted
    states = ladder.states
    write_rerun_input(
        tmp_path / "r.in",
        source=states[0],
        targets=[s for s in states if s.index != 0],
        system=system,
        ghost=ghost,
        config=cfg,
        data_file="d",
        traj_file="t",
        out_file="o",
        sample_every=100,
    )
    text = (tmp_path / "r.in").read_text()
    # 0 -> 2 is a delta of 1.0, two rungs away.
    assert "equal 1" in text


def test_rerun_script_uses_full_output_precision(tmp_path, ghosted, cfg, ladder):
    """%g's 6 figures quantise a -1500 kcal/mol PE at 0.01, failing the diagonal."""
    system, ghost = ghosted
    write_rerun_input(
        tmp_path / "r.in",
        source=ladder.states[0],
        targets=list(ladder.states[1:]),
        system=system,
        ghost=ghost,
        config=cfg,
        data_file="d",
        traj_file="t",
        out_file="o",
        sample_every=100,
    )
    assert 'format " %.14g"' in (tmp_path / "r.in").read_text()


def test_rerun_script_sets_ghost_charges(tmp_path, ghosted, cfg):
    """Charges live in the data file, so a rerun at another lambda_q must reset them."""
    system, ghost = ghosted
    coul = LambdaLadder(leg=FEPLeg.COUL, lambdas=(0.0, 0.5, 1.0))
    source = coul.states[1]
    write_rerun_input(
        tmp_path / "r.in",
        source=source,
        targets=[s for s in coul.states if s.index != 1],
        system=system,
        ghost=ghost,
        config=cfg,
        data_file="d",
        traj_file="t",
        out_file="o",
        sample_every=100,
    )
    text = (tmp_path / "r.in").read_text()
    assert f"set             type {ghost.type_o} charge" in text
    assert f"set             type {ghost.type_h} charge" in text
    # Scaled to the source state, not left at full charge.
    assert f"{ghost.charge_o * 0.5:.12f}" in text


def test_charge_leg_rerun_keeps_the_ghost_neutral(tmp_path, ghosted, cfg):
    system, ghost = ghosted
    coul = LambdaLadder(leg=FEPLeg.COUL, lambdas=(0.0, 0.5, 1.0))
    for source in coul.states:
        write_rerun_input(
            tmp_path / "r.in",
            source=source,
            targets=[s for s in coul.states if s.index != source.index],
            system=system,
            ghost=ghost,
            config=cfg,
            data_file="d", traj_file="t", out_file="o", sample_every=100,
        )
        text = (tmp_path / "r.in").read_text()
        q_o = float(
            [l for l in text.splitlines() if f"type {ghost.type_o} charge" in l][0]
            .split()[-1]
        )
        q_h = float(
            [l for l in text.splitlines() if f"type {ghost.type_h} charge" in l][0]
            .split()[-1]
        )
        assert q_o + 2 * q_h == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------- matrix shape --


def test_energy_matrix_rejects_inconsistent_shapes():
    u = np.zeros((3, 10))
    with pytest.raises(ValueError, match="N_k"):
        EnergyMatrix(u_kn=u, N_k=np.array([5, 5]), lambdas=(0.0, 0.5, 1.0),
                     leg=FEPLeg.LJ, kT=0.6)
    with pytest.raises(ValueError, match="columns"):
        EnergyMatrix(u_kn=u, N_k=np.array([4, 4, 4]), lambdas=(0.0, 0.5, 1.0),
                     leg=FEPLeg.LJ, kT=0.6)


def test_build_rejects_mismatched_input_lengths(ghosted, cfg, ladder, tmp_path):
    _, ghost = ghosted
    with pytest.raises(ValueError, match="3 states"):
        build_energy_matrix(
            ladder, state_dirs=[tmp_path], systems=[None], ghost=ghost,
            config=cfg, workdir=tmp_path / "rr",
        )


# -------------------------------------------------------------- LAMMPS-gated --


@needs_lammps
def test_matrix_is_assembled_with_a_reproducible_diagonal(
    tmp_path, ghosted, cfg, ladder
):
    """The end-to-end check: sample every state, then rebuild the matrix.

    ``build_energy_matrix`` raises if any rerun fails to reproduce its sampling
    run's energies, so reaching the assertions at all means the diagonal held to
    1e-4 kcal/mol across every state.
    """
    _, ghost = ghosted
    dirs, systems = _sample(ladder, ghosted, cfg, tmp_path)
    matrix = build_energy_matrix(
        ladder, state_dirs=dirs, systems=systems, ghost=ghost,
        config=cfg, workdir=tmp_path / "rr",
    )
    assert matrix.n_states == 3
    assert matrix.u_kn.shape[1] == int(matrix.N_k.sum())
    assert (matrix.N_k > 0).all()
    # Frame 0 dropped from each state: production/sample_every frames, not +1.
    assert matrix.N_k.tolist() == [
        cfg.fep.production_steps // cfg.fep.sample_every
    ] * 3
    assert np.isfinite(matrix.u_kn).all()
    assert matrix.kT == pytest.approx(0.0019872041 * cfg.md.temperature)


@needs_lammps
def test_only_one_rerun_per_state(tmp_path, ghosted, cfg, ladder):
    """K reruns, not K^2 -- the whole point of the multi-clause design."""
    _, ghost = ghosted
    dirs, systems = _sample(ladder, ghosted, cfg, tmp_path)
    rr = tmp_path / "rr"
    build_energy_matrix(
        ladder, state_dirs=dirs, systems=systems, ghost=ghost,
        config=cfg, workdir=rr,
    )
    assert len(list(rr.glob("rerun_*.in"))) == len(ladder.states)


@needs_lammps
def test_diagonal_check_catches_a_wrong_hamiltonian(
    tmp_path, ghosted, cfg, ladder, monkeypatch
):
    """Corrupt the sampled energies and confirm the guard fires.

    This is the test that gives the diagonal check its value: without it, a
    force-field mismatch between sampling and rerun produces a plausible matrix
    and a wrong free energy.
    """
    _, ghost = ghosted
    dirs, systems = _sample(ladder, ghosted, cfg, tmp_path)
    victim = dirs[1] / "pe.dat"
    rows = victim.read_text().splitlines()
    body = []
    for line in rows:
        if line.startswith("#"):
            body.append(line)
            continue
        step, pe = line.split()
        body.append(f" {step} {float(pe) + 1.0:.14g}")   # 1 kcal/mol offset
    victim.write_text("\n".join(body) + "\n")

    with pytest.raises(ValueError, match="does not match"):
        build_energy_matrix(
            ladder, state_dirs=dirs, systems=systems, ghost=ghost,
            config=cfg, workdir=tmp_path / "rr",
        )


@needs_lammps
def test_matrix_feeds_mbar(tmp_path, ghosted, cfg, ladder):
    """pymbar must accept u_kn/N_k as assembled, with no reshaping."""
    pymbar = pytest.importorskip("pymbar")
    _, ghost = ghosted
    dirs, systems = _sample(ladder, ghosted, cfg, tmp_path)
    matrix = build_energy_matrix(
        ladder, state_dirs=dirs, systems=systems, ghost=ghost,
        config=cfg, workdir=tmp_path / "rr",
    )
    mbar = pymbar.MBAR(matrix.u_kn, matrix.N_k)
    result = mbar.compute_free_energy_differences()
    dg = result["Delta_f"][0, -1] * matrix.kT
    assert np.isfinite(dg)
    # A 3-state ladder over the whole LJ leg is deliberately too coarse for a
    # physically meaningful number, so only finiteness and sign are asserted:
    # growing a repulsive core into liquid water costs free energy.
    assert dg > 0
