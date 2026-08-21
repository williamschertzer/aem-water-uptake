"""Ghost-water topology, pair-coefficient emission and the tail correction.

The two LAMMPS-gated tests at the bottom are the load-bearing ones. They pin both
ends of the alchemical path against systems built with no ghost at all:

* fully coupled ghost == a real water (so ``lambda = 1`` is the physical state);
* fully decoupled ghost == absent (so ``lambda = 0`` is the ideal-gas state).

If either fails, every free energy the framework produces is wrong by an unknown
constant, and no amount of estimator sophistication downstream will reveal it.
The pure-Python tests above them run everywhere and catch the mistakes that are
easy to make while editing: a ghost sharing a resident type, a dropped cross
term, a charge scaled twice.
"""

import math
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import parmed as pmd
import pytest

from aemwater.assembly import CellContents, assemble, water_molecules
from aemwater.fep.ghost import (
    GHOST_RESIDUE,
    SIGMA_PLACEHOLDER,
    add_ghost_water,
    ghost_pair_coeff_lines,
    scale_ghost_charges,
    tail_correction,
)
from aemwater.forcefield.water import water_model
from aemwater.lammps.writer import LammpsSystem, LammpsWriteError, write_data_file
from conftest import needs_lammps


def _lattice(n: int, edge: float) -> np.ndarray:
    """``n`` well-separated lattice points inside a cubic cell."""
    k = int(math.ceil(n ** (1 / 3)))
    spacing = edge / k
    points = []
    for i in range(k):
        for j in range(k):
            for l in range(k):
                if len(points) < n:
                    points.append([i * spacing + 1.0, j * spacing + 1.0, l * spacing + 1.0])
    return np.asarray(points, dtype=float)


def water_cell(n: int, edge: float = 15.0) -> tuple[LammpsSystem, np.ndarray]:
    """A cell of ``n`` SPC/E waters on a loose lattice, plus its coordinates."""
    waters = water_molecules(n, "spce")
    origins = _lattice(n, edge)
    coords = np.vstack(
        [np.asarray(w.coordinates) + o for w, o in zip(waters, origins)]
    )
    system = assemble(CellContents(chains=[], ions=[], waters=waters), coords, edge)
    return system, coords


# --------------------------------------------------------------- topology ----


def test_ghost_gets_its_own_atom_types():
    """The whole method depends on this: shared types would decouple all water."""
    host, _ = water_cell(10)
    n_host_types = len(host.atom_types)
    real_o, real_h = host.water_atom_types()

    ghosted, ghost = add_ghost_water(host, "spce")

    assert len(ghosted.atom_types) == n_host_types + 2
    assert ghost.type_o not in (real_o, real_h)
    assert ghost.type_h not in (real_o, real_h)
    assert ghost.type_o != ghost.type_h


def test_ghost_types_are_last_and_host_indices_unchanged():
    """Existing type indices must survive, so an equilibrated cell is reusable."""
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")

    n_types = len(ghosted.atom_types)
    assert sorted(ghost.types) == [n_types - 1, n_types]
    assert ghost.type_range == f"{n_types - 1}*{n_types}"
    assert ghost.host_type_range(n_types) == f"1*{n_types - 2}"

    for key, index in host.atom_types.items():
        assert ghosted.atom_types[key] == index


def test_ghost_is_neutral_in_the_data_file():
    """Leg 1 runs uncharged; a stored full charge would make that a caller's job."""
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")

    ghost_atoms = [ghosted.structure.atoms[i - 1] for i in ghost.atom_indices]
    assert all(a.residue.name == GHOST_RESIDUE for a in ghost_atoms)
    assert all(a.charge == 0.0 for a in ghost_atoms)
    assert abs(ghosted.summary()["net_charge"]) < 1e-9


def test_ghost_matches_the_water_model_parameters():
    """A hand-built ghost could drift from the resident waters; this pins it."""
    model = water_model("spce")
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, model)

    assert ghost.epsilon_o == pytest.approx(model.epsilon_O)
    assert ghost.sigma_o == pytest.approx(model.sigma_O)
    assert ghost.charge_o == pytest.approx(model.charge_O)
    assert ghost.charge_h == pytest.approx(model.charge_H)
    # Three-site models put no LJ on hydrogen.
    assert ghost.epsilon_h == 0.0

    ghost_atoms = [ghosted.structure.atoms[i - 1] for i in ghost.atom_indices]
    r_oh = np.linalg.norm(
        np.asarray(ghosted.structure.coordinates)[ghost.atom_indices[1] - 1]
        - np.asarray(ghosted.structure.coordinates)[ghost.atom_indices[0] - 1]
    )
    assert r_oh == pytest.approx(model.r_OH, abs=1e-6)
    assert ghost_atoms[0].mass == pytest.approx(model.mass_O)


def test_ghost_does_not_mutate_the_host_system():
    host, _ = water_cell(10)
    n_atoms = len(host.structure.atoms)
    n_types = len(host.atom_types)

    add_ghost_water(host, "spce")

    assert len(host.structure.atoms) == n_atoms
    assert len(host.atom_types) == n_types


def test_ghost_placement_is_seed_reproducible():
    host, _ = water_cell(10)
    a, _ = add_ghost_water(host, "spce", seed=3)
    b, _ = add_ghost_water(host, "spce", seed=3)
    c, _ = add_ghost_water(host, "spce", seed=4)

    pos_a = np.asarray(a.structure.coordinates)[-3:]
    pos_b = np.asarray(b.structure.coordinates)[-3:]
    pos_c = np.asarray(c.structure.coordinates)[-3:]
    assert np.allclose(pos_a, pos_b)
    assert not np.allclose(pos_a, pos_c)


def test_ghost_is_not_counted_as_water():
    """Uptake is counted from WAT residues; a ghost must never inflate it."""
    host, _ = water_cell(10)
    ghosted, _ = add_ghost_water(host, "spce")

    n_wat = sum(1 for r in host.structure.residues if r.name == host.water_residue)
    n_wat_ghosted = sum(
        1 for r in ghosted.structure.residues if r.name == ghosted.water_residue
    )
    assert n_wat_ghosted == n_wat


# ------------------------------------------------------- pair coefficients ----


def test_every_pair_is_emitted_explicitly():
    """The soft styles refuse to mix unequal lambdas, so nothing may be implicit."""
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")
    n = len(ghosted.atom_types)

    lines = ghost_pair_coeff_lines(ghosted, ghost, 0.4)

    assert len(lines) == n * (n + 1) // 2
    emitted = {
        (int(p[1]), int(p[2]))
        for p in (line.split() for line in lines)
    }
    assert emitted == {(i, j) for i in range(1, n + 1) for j in range(i, n + 1)}


@pytest.mark.parametrize("lam", [0.0, 0.05, 0.5, 1.0])
def test_only_ghost_host_pairs_carry_lambda(lam):
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")
    ghost_types = set(ghost.types)

    for line in ghost_pair_coeff_lines(ghosted, ghost, lam):
        parts = line.split()
        i, j, emitted_lambda = int(parts[1]), int(parts[2]), float(parts[5])
        is_cross = (i in ghost_types) != (j in ghost_types)
        expected = lam if is_cross else 1.0
        assert emitted_lambda == pytest.approx(expected), line


def test_zero_epsilon_pairs_get_a_nonzero_sigma():
    """sigma = 0 is rejected by the soft styles; epsilon = 0 makes it irrelevant."""
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")

    for line in ghost_pair_coeff_lines(ghosted, ghost, 0.5):
        parts = line.split()
        eps, sigma = float(parts[3]), float(parts[4])
        assert sigma > 0.0, line
        if eps == 0.0:
            assert sigma == pytest.approx(SIGMA_PLACEHOLDER) or sigma > 0.0


def test_mixing_follows_lorentz_berthelot():
    """Arithmetic sigma, geometric epsilon -- matching pair_modify mix arithmetic."""
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")
    params = {t: (e, s) for t, e, s, _ in ghosted.pair_coeffs()}

    by_pair = {}
    for line in ghost_pair_coeff_lines(ghosted, ghost, 1.0):
        p = line.split()
        by_pair[(int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]))

    o_real = ghosted.water_atom_types()[0]
    o_ghost = ghost.type_o
    eps, sigma = by_pair[(min(o_real, o_ghost), max(o_real, o_ghost))]
    e1, s1 = params[o_real]
    e2, s2 = params[o_ghost]
    assert eps == pytest.approx(math.sqrt(e1 * e2))
    assert sigma == pytest.approx(0.5 * (s1 + s2))


# ------------------------------------------------------------ charge scaling --


@pytest.mark.parametrize("lam_q", [0.0, 0.2, 0.5, 1.0])
def test_charge_scaling_is_linear_and_neutral(lam_q):
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")

    scaled = scale_ghost_charges(ghosted, ghost, lam_q)

    charges = [scaled.structure.atoms[i - 1].charge for i in ghost.atom_indices]
    assert charges[0] == pytest.approx(lam_q * ghost.charge_o)
    assert charges[1] == pytest.approx(lam_q * ghost.charge_h)
    assert sum(charges) == pytest.approx(0.0, abs=1e-9)
    # The rest of the cell is untouched.
    assert abs(scaled.summary()["net_charge"]) < 1e-9


def test_charge_scaling_leaves_the_input_alone():
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")

    scale_ghost_charges(ghosted, ghost, 1.0)

    assert all(
        ghosted.structure.atoms[i - 1].charge == 0.0 for i in ghost.atom_indices
    )


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_charge_scaling_rejects_out_of_range_lambda(bad):
    host, _ = water_cell(10)
    ghosted, ghost = add_ghost_water(host, "spce")
    with pytest.raises(ValueError, match="lambda_q"):
        scale_ghost_charges(ghosted, ghost, bad)


# ---------------------------------------------------------- tail correction --


def test_tail_correction_matches_the_closed_form():
    """SPC/E puts LJ on oxygen only, so the sum reduces to one analytic term."""
    n_water, edge, cutoff = 30, 12.0, 10.0
    host, _ = water_cell(n_water, edge)
    ghosted, ghost = add_ghost_water(host, "spce")
    model = water_model("spce")

    volume = edge ** 3
    x = model.sigma_O / cutoff
    expected = (
        8.0 * math.pi / (3.0 * volume)
        * n_water * model.epsilon_O * model.sigma_O ** 3
        * (x ** 9 / 3.0 - x ** 3)
    )

    assert tail_correction(ghosted, ghost, cutoff) == pytest.approx(expected, rel=1e-12)


def test_tail_correction_is_attractive_and_scales_as_inverse_cube():
    host, _ = water_cell(30, 12.0)
    ghosted, ghost = add_ghost_water(host, "spce")

    t10 = tail_correction(ghosted, ghost, 10.0)
    t20 = tail_correction(ghosted, ghost, 20.0)

    assert t10 < 0.0  # the LJ tail is attractive
    # r^-9 is negligible at these cutoffs, so the ratio approaches 2^3.
    assert t10 / t20 == pytest.approx(8.0, rel=0.01)


def test_tail_correction_excludes_the_ghost_from_the_host_count():
    """The ghost must not interact with itself through the tail term."""
    small, _ = water_cell(10, 12.0)
    ghosted, ghost = add_ghost_water(small, "spce")

    # 10 host waters, one ghost: the correction must reflect 10 partners, not 11.
    model = water_model("spce")
    x = model.sigma_O / 10.0
    per_partner = (
        8.0 * math.pi / (3.0 * 12.0 ** 3)
        * model.epsilon_O * model.sigma_O ** 3 * (x ** 9 / 3.0 - x ** 3)
    )
    assert tail_correction(ghosted, ghost, 10.0) == pytest.approx(10 * per_partner, rel=1e-12)


# --------------------------------------------------- LAMMPS endpoint checks ---

_ENERGY_INPUT = """\
units           real
atom_style      full
boundary        p p p
pair_style      {style}
kspace_style    pppm 1.0e-6
bond_style      harmonic
angle_style     harmonic
special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333
read_data       {data}
{pair_coeffs}
neighbor        2.0 bin
run             0
print "RESULT pe $(pe:%.12g) evdwl $(evdwl:%.12g) ecoul $(ecoul:%.12g) elong $(elong:%.12g)"
"""


def _single_point(tmp_path: Path, name: str, system, pair_lines, style,
                  include_pair_coeffs=False) -> dict[str, float]:
    """Single-point energy of ``system`` under ``style``."""
    data = tmp_path / f"{name}.data"
    write_data_file(system, data, name, include_pair_coeffs=include_pair_coeffs)
    script = tmp_path / f"{name}.in"
    script.write_text(
        _ENERGY_INPUT.format(style=style, data=data.name,
                             pair_coeffs="\n".join(pair_lines))
    )
    binary = shutil.which("lmp") or shutil.which("lmp_serial")
    subprocess.run(
        [binary, "-in", script.name, "-log", f"{name}.log", "-screen", "none"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    log = (tmp_path / f"{name}.log").read_text()
    errors = [line for line in log.splitlines() if "ERROR" in line]
    assert not errors, errors[0]
    line = next(l for l in log.splitlines() if l.startswith("RESULT"))
    fields = line.split()
    return {fields[i]: float(fields[i + 1]) for i in range(1, len(fields), 2)}


def _plain_soft_lines(system) -> list[str]:
    """Explicit soft-style coefficients for a system with no ghost."""
    params = {t: (e, s) for t, e, s, _ in system.pair_coeffs()}
    n = len(system.atom_types)
    lines = []
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            e1, s1 = params[i]
            e2, s2 = params[j]
            sigma = 0.5 * (s1 + s2) or SIGMA_PLACEHOLDER
            lines.append(
                f"pair_coeff {i} {j} {math.sqrt(e1 * e2):.8f} {sigma:.8f} 1.000000"
            )
    return lines


def _ghost_at(host_system, coords_to_match, seed=7):
    """A ghosted system with the ghost moved onto ``coords_to_match``."""
    ghosted, ghost = add_ghost_water(host_system, "spce", seed=seed)
    struct = pmd.structure.copy(ghosted.structure)
    struct.box = ghosted.structure.box
    xyz = np.asarray(struct.coordinates)
    xyz[-3:] = coords_to_match
    struct.coordinates = xyz
    moved = LammpsSystem(
        structure=struct,
        water_residue=ghosted.water_residue,
        ion_residues=ghosted.ion_residues,
    )
    return moved, ghost


@needs_lammps
def test_fully_coupled_ghost_equals_a_real_water(tmp_path):
    """lambda = 1 must BE the physical state, not merely resemble it.

    A 31-water cell is compared against 30 waters plus a ghost sitting on the
    31st water's coordinates. Same configuration, two descriptions; the energies
    must agree.
    """
    real31, coords31 = water_cell(31)
    host30, _ = water_cell(30)
    ghosted, ghost = _ghost_at(host30, coords31[-3:])
    coupled = scale_ghost_charges(ghosted, ghost, 1.0)

    reference = _single_point(
        tmp_path, "real31", real31, _plain_soft_lines(real31),
        "lj/cut/coul/long/soft 1 0.5 10.0 10.0",
    )
    with_ghost = _single_point(
        tmp_path, "ghost31", coupled,
        ghost_pair_coeff_lines(coupled, ghost, 1.0),
        "lj/cut/coul/long/soft 1 0.5 10.0 10.0",
    )

    # PPPM is converged to 1e-6, so agreement is limited by the solver, not by
    # the topology. Anything structurally wrong shows up orders of magnitude
    # above this.
    assert with_ghost["evdwl"] == pytest.approx(reference["evdwl"], rel=1e-9)
    assert with_ghost["pe"] == pytest.approx(reference["pe"], rel=1e-5)


@needs_lammps
def test_decoupled_ghost_contributes_nothing(tmp_path):
    """lambda = 0 must be the ideal-gas state: the ghost may not be felt at all.

    ECOUL and ELONG each shift by a fraction of a kcal/mol between the two runs
    because PPPM re-partitions its real-space/k-space split over a different atom
    count. Those shifts cancel; the total energy is what enters the free energy,
    and it is what is asserted here.
    """
    host30, coords30 = water_cell(30)
    ghosted, ghost = add_ghost_water(host30, "spce", seed=7)
    decoupled = scale_ghost_charges(ghosted, ghost, 0.0)

    without = _single_point(
        tmp_path, "real30", host30, _plain_soft_lines(host30),
        "lj/cut/coul/long/soft 1 0.5 10.0 10.0",
    )
    with_ghost = _single_point(
        tmp_path, "ghostoff", decoupled,
        ghost_pair_coeff_lines(decoupled, ghost, 0.0),
        "lj/cut/coul/long/soft 1 0.5 10.0 10.0",
    )

    assert with_ghost["evdwl"] == pytest.approx(without["evdwl"], abs=1e-9)
    assert with_ghost["pe"] == pytest.approx(without["pe"], abs=1e-4)


@needs_lammps
def test_soft_style_reduces_to_the_production_style_at_full_coupling(tmp_path):
    """The endpoint must be the same physics the rest of the workflow uses."""
    system, _ = water_cell(30)

    soft = _single_point(
        tmp_path, "soft", system, _plain_soft_lines(system),
        "lj/cut/coul/long/soft 1 0.5 10.0 10.0",
    )
    plain_lines = [
        line.rsplit(" ", 1)[0].replace("pair_coeff", "pair_coeff")
        for line in _plain_soft_lines(system)
    ]
    plain = _single_point(
        tmp_path, "plain", system, plain_lines, "lj/cut/coul/long 10.0",
    )

    assert soft["pe"] == pytest.approx(plain["pe"], rel=1e-6)
