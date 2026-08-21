"""Fixed-lambda input generation, and what it must never emit.

Two classes of test here. The pure-Python ones assert properties of the rendered
text: a wrong pair style, a missing cross term, or an unbalanced charge
perturbation are all things that produce a running simulation of the wrong
Hamiltonian. The LAMMPS-gated ones run the generated input and check the
resulting dU against explicit finite differences, which is the only way to know
the ``compute fep`` clauses mean what they say.
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import pytest

from aemwater.assembly import CellContents, assemble, water_molecules
from aemwater.config import PolymerSpec, RunConfig
from aemwater.fep.ghost import (
    add_ghost_water,
    ghost_pair_coeff_lines,
    scale_ghost_charges,
)
from aemwater.fep.inputs import perturbations_for, render_state_input
from aemwater.fep.schedule import FEPLeg, default_ladders
from aemwater.lammps.inputs import ConstraintSpec, GroupSpec
from aemwater.lammps.writer import write_data_file
from conftest import BTMA_PS, lammps_binary, needs_lammps


# ------------------------------------------------------------------ fixtures --


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
def cell():
    """Liquid-density SPC/E water, small enough to run in seconds."""
    return _water_cell(195, 18.0)


@pytest.fixture(scope="module")
def ghosted(cell):
    return add_ghost_water(cell, "spce", seed=7)


@pytest.fixture
def cfg():
    """Short sampling: these tests check mechanics, not convergence.

    The 20-frame floor in ``FEPSpec.validate`` is respected rather than bypassed
    -- a fixture that has to disable a production guard is a fixture testing
    something the workflow cannot actually run.
    """
    return RunConfig(polymer=PolymerSpec(smiles=BTMA_PS)).with_overrides(
        **{
            "fep.equil_steps": 100,
            "fep.production_steps": 2000,
            "fep.sample_every": 100,
        }
    )


@pytest.fixture
def groups():
    return GroupSpec(
        n_polymer_molecules=0, n_ion_molecules=0, water_type_o=1, water_type_h=2
    )


@pytest.fixture
def constraints(ghosted):
    _, ghost = ghosted
    return ConstraintSpec(
        shake_water=True,
        shake_hydrogen=False,
        water_bond_type=ghost.bond_type,
        water_angle_type=ghost.angle_type,
    )


def _render(state, tmp_path, ghosted, cfg, groups, constraints, ladder):
    system, ghost = ghosted
    scaled = scale_ghost_charges(system, ghost, state.lambda_q)
    write_data_file(scaled, tmp_path / "state.data", "t", include_pair_coeffs=False)
    return scaled, ghost, render_state_input(
        state,
        directory=tmp_path,
        system=scaled,
        ghost=ghost,
        ladder_lambdas=ladder,
        config=cfg,
        groups=groups,
        constraints=constraints,
        comm_cutoff=14.0,
        data_file="state.data",
        seed=999,
    )


# ------------------------------------------------------- perturbation clauses --


def test_lj_leg_perturbs_the_pair_lambda(ghosted, cfg):
    system, ghost = ghosted
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    perts = perturbations_for(
        lj.states[3], ghost, system, cfg.fep.lj_lambdas, cfg.fep, 298.0
    )
    for p in perts:
        assert "pair lj/cut/coul/long/soft lambda" in p.compute_command
        assert "atom charge" not in p.compute_command


def test_coul_leg_perturbs_charges_not_the_pair_lambda(ghosted, cfg):
    system, ghost = ghosted
    _, coul = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    perts = perturbations_for(
        coul.states[2], ghost, system, cfg.fep.coul_lambdas, cfg.fep, 298.0
    )
    for p in perts:
        assert "atom charge" in p.compute_command
        assert "lambda" not in p.compute_command


def test_charge_perturbations_keep_the_ghost_neutral(ghosted, cfg):
    """A net charge change would put a monopole in a periodic cell.

    PPPM would then neutralise it against a uniform background -- a large,
    entirely artificial contribution that no amount of sampling reveals.
    """
    system, ghost = ghosted
    _, coul = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    for state in coul.states:
        for p in perturbations_for(
            state, ghost, system, cfg.fep.coul_lambdas, cfg.fep, 298.0
        ):
            dq_o = p.delta * ghost.charge_o
            dq_h = p.delta * ghost.charge_h
            assert dq_o + 2 * dq_h == pytest.approx(0.0, abs=1e-12)


def test_perturbations_target_ladder_neighbours(ghosted, cfg):
    system, ghost = ghosted
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    interior = lj.states[3]
    perts = perturbations_for(
        interior, ghost, system, cfg.fep.lj_lambdas, cfg.fep, 298.0
    )
    bar = {p.name: p for p in perts if p.name in ("dU_fwd", "dU_rev")}
    assert bar["dU_fwd"].target_index == interior.index + 1
    assert bar["dU_rev"].target_index == interior.index - 1
    assert bar["dU_fwd"].delta > 0 > bar["dU_rev"].delta


def test_endpoints_have_one_sided_bar_only(ghosted, cfg):
    """No neighbour below state 0 or above state K-1 to perturb toward."""
    system, ghost = ghosted
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    first = {p.name for p in perturbations_for(
        lj.states[0], ghost, system, cfg.fep.lj_lambdas, cfg.fep, 298.0)}
    last = {p.name for p in perturbations_for(
        lj.states[-1], ghost, system, cfg.fep.lj_lambdas, cfg.fep, 298.0)}
    assert "dU_fwd" in first and "dU_rev" not in first
    assert "dU_rev" in last and "dU_fwd" not in last


def test_ti_clauses_absent_when_ti_not_requested(ghosted, cfg):
    system, ghost = ghosted
    lean = cfg.with_overrides(**{"fep.estimators": ["mbar", "bar"]})
    lj, _ = default_ladders(lean.fep.lj_lambdas, lean.fep.coul_lambdas)
    names = {p.name for p in perturbations_for(
        lj.states[3], ghost, system, lean.fep.lj_lambdas, lean.fep, 298.0)}
    assert not any("ti" in n for n in names)


def test_ti_uses_its_own_step_not_the_ladder_spacing(ghosted, cfg):
    """The ladder is spaced for overlap; a 0.1 secant is a poor derivative."""
    system, ghost = ghosted
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    perts = perturbations_for(
        lj.states[5], ghost, system, cfg.fep.lj_lambdas, cfg.fep, 298.0
    )
    ti = [p for p in perts if "ti" in p.name]
    assert {abs(p.delta) for p in ti} == {cfg.fep.ti_delta}


def test_temperature_is_substituted_not_left_as_a_placeholder(ghosted, cfg):
    system, ghost = ghosted
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    perts = perturbations_for(
        lj.states[2], ghost, system, cfg.fep.lj_lambdas, cfg.fep, 310.0
    )
    for p in perts:
        assert "310" in p.compute_command
        assert "{" not in p.compute_command


# ---------------------------------------------------------- rendered template --


def test_rendered_input_uses_the_soft_pair_style(
    tmp_path, ghosted, cfg, groups, constraints
):
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    assert "pair_style      lj/cut/coul/long/soft" in text
    # The plain style must not appear anywhere -- not even shadowed above an
    # override, which would leave the wrong Hamiltonian in force for the run.
    assert "pair_style      lj/cut/coul/long " not in text


def test_rendered_input_omits_tail_correction(
    tmp_path, ghosted, cfg, groups, constraints
):
    """compute fep refuses `tail yes` for soft styles; the term is added in post."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    # Comments explain *why* it is absent, so only directives are inspected.
    directives = [
        l for l in info["input"].read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")
    ]
    assert not any("tail yes" in l for l in directives)
    assert any(l.startswith("pair_modify") for l in directives)


def test_rendered_input_tightens_kspace(
    tmp_path, ghosted, cfg, groups, constraints
):
    """1e-4 leaves ~0.016 kcal/mol of grid error in each charge-leg dU."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    assert "pppm 1e-06" in text or "pppm 1.0e-06" in text
    assert f"pppm {cfg.md.kspace_accuracy}" not in text


def test_every_pair_is_emitted_explicitly(
    tmp_path, ghosted, cfg, groups, constraints
):
    """Nothing left to mixing rules, which the soft styles refuse across lambdas."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    scaled, ghost, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    n = len(scaled.atom_types)
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            assert f"pair_coeff      {i} {j} " in text, f"missing pair {i} {j}"


def test_ghost_is_excluded_from_the_real_group(
    tmp_path, ghosted, cfg, groups, constraints
):
    """Density and thermostat statistics must describe the physical system."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, ghost, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    assert f"group           ghost type {ghost.type_o} {ghost.type_h}" in text
    assert "group           real subtract all ghost" in text


def test_sampling_is_instantaneous_not_averaged(
    tmp_path, ghosted, cfg, groups, constraints
):
    """BAR and MBAR need the dU distribution; exp(-b<dU>) != <exp(-b dU)>."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    line = [l for l in info["input"].read_text().splitlines() if "fix             fepout" in l][0]
    # fix ID group ave/time Nevery Nrepeat Nfreq -- Nrepeat (index 5) must be 1.
    fields = line.split()
    assert fields[3] == "ave/time", line
    assert fields[5] == "1", f"Nrepeat is {fields[5]}, not 1: {line}"


def test_nvt_not_npt(tmp_path, ghosted, cfg, groups, constraints):
    """A per-state volume would make each lambda a different physical system."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    assert "fix             integrate all nvt" in text
    assert " npt " not in text


def test_state_label_recorded_in_the_output_header(
    tmp_path, ghosted, cfg, groups, constraints
):
    """A dU file with no lambda in it cannot be attributed to a state later."""
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    state = lj.states[3]
    _, _, info = _render(
        state, tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    text = info["input"].read_text()
    assert f"lambda_lj={state.lambda_lj:g}" in text
    assert f"lambda_q={state.lambda_q:g}" in text


# -------------------------------------------------------------- LAMMPS-gated --


@needs_lammps
def test_generated_input_runs_on_the_lj_leg(
    tmp_path, ghosted, cfg, groups, constraints
):
    lj, _ = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        lj.states[3], tmp_path, ghosted, cfg, groups, constraints, cfg.fep.lj_lambdas
    )
    proc = subprocess.run(
        [lammps_binary(), "-in", "in.fep", "-log", "run.log", "-screen", "none"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    log = (tmp_path / "run.log").read_text()
    assert proc.returncode == 0, log[-2000:]
    assert "FEP_STATE_DONE" in log
    assert "Shake determinant" not in log
    assert "Lost atoms" not in log

    data = np.loadtxt(info["fep_file"])
    energies = np.loadtxt(info["pe_file"])
    assert data.shape[0] >= 10
    # dU rows and energy rows must refer to the same frames, or the rerun matrix
    # will be assembled against misaligned samples.
    assert np.array_equal(data[:, 0], energies[:, 0])


@needs_lammps
def test_generated_input_runs_on_the_charge_leg(
    tmp_path, ghosted, cfg, groups, constraints
):
    _, coul = default_ladders(cfg.fep.lj_lambdas, cfg.fep.coul_lambdas)
    _, _, info = _render(
        coul.states[2], tmp_path, ghosted, cfg, groups, constraints,
        cfg.fep.coul_lambdas,
    )
    proc = subprocess.run(
        [lammps_binary(), "-in", "in.fep", "-log", "run.log", "-screen", "none"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    log = (tmp_path / "run.log").read_text()
    assert proc.returncode == 0, log[-2000:]
    assert "FEP_STATE_DONE" in log
    assert np.loadtxt(info["fep_file"]).shape[0] >= 10


@needs_lammps
def test_compute_fep_matches_an_explicit_energy_difference(
    tmp_path, ghosted, cfg, groups, constraints
):
    """The clause means what it says, or this test fails.

    ``compute fep`` reports dU for a perturbation it applies internally. If the
    type ranges or the delta are wrong the number is still plausible -- it is an
    energy difference of *something*. Comparing against U(lambda_b) - U(lambda_a)
    evaluated explicitly on one configuration is the only check that binds it.
    """
    system, ghost = ghosted
    lam_a, lam_b = 0.4, 0.5
    scaled = scale_ghost_charges(system, ghost, 0.0)
    write_data_file(scaled, tmp_path / "s.data", "s", include_pair_coeffs=False)

    header = (
        "units real\natom_style full\nboundary p p p\n"
        f"pair_style lj/cut/coul/long/soft {cfg.fep.soft_core_n} "
        f"{cfg.fep.alpha_lj} {cfg.fep.alpha_coul} {cfg.md.cutoff} {cfg.md.cutoff}\n"
        "pair_modify mix arithmetic\n"
        f"kspace_style pppm {cfg.fep.kspace_accuracy}\n"
        "bond_style harmonic\nangle_style harmonic\n"
        "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333\n"
        "read_data s.data\n"
    )

    def energy_at(lam, tag):
        body = "\n".join(ghost_pair_coeff_lines(scaled, ghost, lam))
        script = header + body + '\nrun 0\nprint "PE $(pe:%.12g)"\n'
        (tmp_path / f"{tag}.in").write_text(script)
        subprocess.run(
            [lammps_binary(), "-in", f"{tag}.in", "-log", f"{tag}.log",
             "-screen", "none"],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        )
        for line in (tmp_path / f"{tag}.log").read_text().splitlines():
            if line.startswith("PE "):
                return float(line.split()[1])
        raise AssertionError("no PE line")

    explicit = energy_at(lam_b, "b") - energy_at(lam_a, "a")

    body = "\n".join(ghost_pair_coeff_lines(scaled, ghost, lam_a))
    script = (
        header + body
        + f"\nvariable dl equal {lam_b - lam_a}\n"
        f"compute df all fep {cfg.md.temperature} pair lj/cut/coul/long/soft "
        f"lambda {ghost.host_type_range(len(scaled.atom_types))} "
        f"{ghost.type_range} v_dl\n"
        'run 0\nprint "DU $(c_df[1]:%.12g)"\n'
    )
    (tmp_path / "f.in").write_text(script)
    subprocess.run(
        [lammps_binary(), "-in", "f.in", "-log", "f.log", "-screen", "none"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    reported = next(
        float(l.split()[1])
        for l in (tmp_path / "f.log").read_text().splitlines()
        if l.startswith("DU ")
    )
    # Measured agreement is ~3e-11 kcal/mol; 1e-6 leaves room for platform
    # differences in the kspace solver without admitting a real discrepancy.
    assert reported == pytest.approx(explicit, abs=1e-6)
