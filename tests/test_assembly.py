"""System assembly and template rendering."""

from __future__ import annotations

import numpy as np
import pytest

from aemwater.assembly import (
    AssemblyError,
    CellContents,
    assemble,
    ion_molecules,
    molecule_id_ranges,
    water_molecules,
)
from aemwater.config import MDSpec
from aemwater.lammps.inputs import (
    ConstraintSpec,
    GroupSpec,
    comm_cutoff,
    minimise_spec,
    render_input,
    soft_push_spec,
    stage_spec,
)


def test_water_molecules_carry_model_charges():
    w = water_molecules(2, "spce")
    assert len(w) == 2
    q = [a.charge for a in w[0].atoms]
    assert sum(q) == pytest.approx(0.0, abs=1e-9)
    assert q[0] == pytest.approx(-0.8476, abs=1e-4)


def test_replicated_molecules_are_independent():
    """Shared atom objects would collapse two molecules onto one molecule ID."""
    w = water_molecules(2, "spce")
    assert w[0].atoms[0] is not w[1].atoms[0]


def test_ion_molecules_accept_a_label_or_a_counterion_object():
    from aemwater.chemistry import build_composition

    comp = build_composition("[*]CC([*])c1ccc(C[N+](C)(C)C)cc1", 1, 2)
    by_label = ion_molecules(1, "Cl-")
    by_object = ion_molecules(1, comp.counterion)
    assert by_label[0].atoms[0].type == by_object[0].atoms[0].type
    assert by_object[0].atoms[0].charge == pytest.approx(-1.0)


def test_molecule_id_ranges_follow_the_canonical_order():
    contents = CellContents(
        chains=water_molecules(2, "spce"),  # stand-ins; only counts matter here
        ions=ion_molecules(3, "Cl-"),
        waters=water_molecules(4, "spce"),
    )
    ranges = molecule_id_ranges(contents)
    assert ranges["polymer"] == (1, 2)
    assert ranges["ions"] == (3, 5)
    assert ranges["water"] == (6, 9)


def test_assemble_rejects_mismatched_coordinates():
    contents = CellContents(chains=[], ions=ion_molecules(2, "Cl-"), waters=[])
    with pytest.raises(AssemblyError, match="expected"):
        assemble(contents, np.zeros((5, 3)), edge=20.0)


def test_assemble_sets_box_and_coordinates():
    contents = CellContents(chains=[], ions=ion_molecules(2, "Cl-"),
                            waters=water_molecules(1, "spce"))
    coords = np.arange(contents.n_atoms * 3, dtype=float).reshape(-1, 3)
    system = assemble(contents, coords, edge=25.0)
    assert system.structure.box[0] == pytest.approx(25.0)
    assert np.allclose(system.structure.coordinates, coords)
    # Cl- and water O/H are three distinct LAMMPS atom types.
    assert len(system.atom_types) == 3


def test_assemble_accepts_structures_that_have_been_pickled():
    """Assembly must survive a pickle round-trip of its inputs.

    The uptake driver reuses GAFF2-typed chains across iterations, and any
    caching or process boundary pickles them. ParmEd's ``AmberParm.__add__``
    calls ``AmberFormat.__copy__``, which increments ``_ncopies`` -- an
    attribute set in ``__init__`` and *not* restored by unpickling -- so adding
    pickled structures together raises ``AttributeError`` (ParmEd 4.3.1). The
    real failure appeared only after an hour of reference sampling, at the first
    insertion iteration.
    """
    import pickle

    contents = CellContents(chains=[], ions=ion_molecules(2, "Cl-"),
                            waters=water_molecules(2, "spce"))
    revived = CellContents(
        chains=[],
        ions=[pickle.loads(pickle.dumps(m)) for m in contents.ions],
        waters=[pickle.loads(pickle.dumps(m)) for m in contents.waters],
    )
    coords = np.arange(revived.n_atoms * 3, dtype=float).reshape(-1, 3)

    system = assemble(revived, coords, edge=25.0)

    assert len(system.structure.atoms) == contents.n_atoms
    assert len(system.atom_types) == 3
    # Charges must survive the downcast, or the electrostatics are silently wrong.
    assert sum(a.charge for a in system.structure.atoms) == pytest.approx(
        sum(a.charge for a in assemble(contents, coords, edge=25.0).structure.atoms)
    )


def _render_for_test(template, tmp_path):
    """Render a template with a complete context, returning the text.

    Shared by the undefined-variable check and the SHAKE-instance check so the
    context is defined once.
    """
    from aemwater.config import PolymerSpec, RunConfig

    cfg = RunConfig(polymer=PolymerSpec(
        smiles="[*]CC([*])c1ccc(C[N+](C)(C)C)cc1", n_chains=2, chain_length=4))
    md = cfg.md
    ctx = dict(
        md=md, insertion=cfg.insertion, widom=cfg.widom, box=cfg.box,
        polymer=cfg.polymer, soft=soft_push_spec(md), minim=minimise_spec(md),
        stages=stage_spec(md), comm_cutoff=comm_cutoff(md),
        constraints=ConstraintSpec(shake_water=True, shake_hydrogen=True,
                                   water_bond_type=8, water_angle_type=14),
        groups=GroupSpec(n_polymer_molecules=2, n_ion_molecules=6,
                         water_type_o=7, water_type_h=8),
        title="test", data_file="system.data", extra_types=None,
        pair_coeff_lines=["pair_coeff 1 1 0.1 3.4"],
        out_data="out.data", out_restart="out.restart",
        dump_file="traj.lammpstrj", density_file="density.dat",
        mu_file="mu.dat", seed=1, velocity_create=True, settle_steps=100,
        npt_equil_steps=100, npt_prod_steps=100, n_averages=5,
        full_energy=True, water_template="h2o.mol",
        n_widom_samples=10, widom_window=1000, widom_steps=1000,
        equil_steps=100, vol_file="vol.dat", out_file="out.dat",
    )
    out = render_input(template, tmp_path / template.replace(".j2", ""), **ctx)
    return out.read_text()


@pytest.mark.parametrize(
    "template",
    ["minimise.in.j2", "equilibrate.in.j2", "insert.in.j2", "widom.in.j2",
     "bulk.in.j2"],
)
def test_templates_render_without_undefined_variables(template, tmp_path):
    """StrictUndefined means a typo in a template is a test failure, not a blank."""
    text = _render_for_test(template, tmp_path)
    assert "pair_modify     mix arithmetic" in text, "arithmetic mixing must be set"
    assert "special_bonds" in text, "Amber 1-4 scaling must be set"
    assert "{{" not in text and "}}" not in text


def test_minimise_template_disables_kspace_for_the_soft_stage(tmp_path):
    """`pair_style soft` has no Coulomb term; PPPM must be off while it is active."""
    from aemwater.config import PolymerSpec, RunConfig

    md = RunConfig(polymer=PolymerSpec(
        smiles="[*]CC([*])c1ccc(C[N+](C)(C)C)cc1", n_chains=2, chain_length=4)).md
    out = render_input(
        "minimise.in.j2", tmp_path / "in.min",
        md=md, soft=soft_push_spec(md), minim=minimise_spec(md),
        stages=stage_spec(md), comm_cutoff=comm_cutoff(md),
        constraints=ConstraintSpec(shake_water=False, shake_hydrogen=False),
        groups=GroupSpec(n_polymer_molecules=1, n_ion_molecules=1),
        title="t", data_file="d.data", extra_types=None,
        pair_coeff_lines=["pair_coeff 1 1 0.1 3.4"],
        out_data="o.data", out_restart="o.restart",
    )
    text = out.read_text()
    soft_at = text.index("pair_style      soft")
    none_at = text.index("kspace_style    none")
    restore_at = text.index("pair_style      lj/cut/coul/long", soft_at)
    assert none_at < soft_at < restore_at


# --------------------------------------------------------------- SHAKE fixes --
def test_constraint_spec_emits_one_shake_command():
    """Water and X-H constraints must share a single ``fix shake``.

    LAMMPS raises "More than one fix shake instance" and aborts if a run defines
    two. The templates previously emitted one fix for water and another for X-H,
    which killed the first insertion iteration of a real run after the bulk
    reference had already been sampled.
    """
    from aemwater.lammps.inputs import ConstraintSpec

    both = ConstraintSpec(shake_water=True, shake_hydrogen=True,
                          water_bond_type=7, water_angle_type=5)
    cmd = both.shake_command
    # One command: the fix *style* keyword appears once (the fix ID also
    # contains "shake", so count the style token, not the substring).
    assert cmd.split().count("shake") == 1, cmd
    # Every constraint keyword rides in that one command.
    assert "b 7" in cmd and "a 5" in cmd and "m 1.008" in cmd, cmd

    water_only = ConstraintSpec(shake_water=True, shake_hydrogen=False,
                               water_bond_type=7, water_angle_type=5)
    assert "m " not in water_only.shake_command

    hydrogen_only = ConstraintSpec(shake_water=False, shake_hydrogen=True)
    assert "b " not in hydrogen_only.shake_command
    assert "m 1.008" in hydrogen_only.shake_command

    # Nothing constrained -> no command at all, not a malformed one.
    assert ConstraintSpec(shake_water=False, shake_hydrogen=False).shake_command == ""


def test_shake_water_without_types_is_an_error():
    """A missing bond/angle type must fail loudly, not emit 'b None'."""
    from aemwater.lammps.inputs import ConstraintSpec
    from aemwater.lammps.writer import LammpsWriteError

    spec = ConstraintSpec(shake_water=True, shake_hydrogen=False)
    with pytest.raises(LammpsWriteError, match="water_bond_type"):
        _ = spec.shake_command


@pytest.mark.parametrize(
    "template",
    ["minimise.in.j2", "equilibrate.in.j2", "insert.in.j2", "widom.in.j2",
     "bulk.in.j2"],
)
def test_rendered_inputs_define_at_most_one_shake_fix(template, tmp_path):
    """No rendered input may contain two ``fix ... shake`` lines."""
    text = _render_for_test(template, tmp_path)
    shake_lines = [ln for ln in text.splitlines()
                   if ln.strip().startswith("fix") and " shake " in ln]
    assert len(shake_lines) <= 1, shake_lines


# ------------------------------------------------- constraints vs Widom sampling --
def test_xh_constraints_are_off_at_a_1fs_timestep():
    """X-H constraints buy nothing at 1 fs and break Widom sampling.

    Measured on the real membrane: adding the X-H constraint to rigid water gave
    98 "Shake determinant < 0.0" warnings under `fix widom`, versus 0 with rigid
    water alone. Enumerating the X-H bond types explicitly reproduced it, so the
    trigger is constraining X-H at all while test particles are inserted, not the
    mass keyword.
    """
    from aemwater.config import MDSpec
    from aemwater.lammps.inputs import constraint_spec

    spec = constraint_spec(MDSpec(timestep=1.0), 8, 14, has_widom=True)
    assert spec.shake_water is True
    assert spec.shake_hydrogen is False
    cmd = spec.shake_command
    assert "b 8" in cmd and "a 14" in cmd
    assert " m " not in cmd, cmd


def test_xh_constraints_are_on_above_1fs():
    """Above 1 fs the constraint earns its cost, so it is requested."""
    from aemwater.config import MDSpec
    from aemwater.lammps.inputs import constraint_spec

    spec = constraint_spec(MDSpec(timestep=2.0), 8, 14)
    assert spec.shake_hydrogen is True
    assert " m 1.008" in spec.shake_command
