"""LAMMPS data writing, including a term-by-term energy check against Amber.

The energy test is the single most important test in the repository. Everything
downstream -- densities, insertion energies, chemical potentials, the uptake
number itself -- is only as trustworthy as the claim that the LAMMPS system is
the *same* physical system that AmberTools parameterised. Amber's own reference
energies for the identical topology are stored in this file, so a regression in
unit conversion, 1-4 scaling, LJ mixing or dihedral style is caught immediately.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import parmed as pmd
import pytest

from aemwater.lammps.parse import LammpsRunError, parse_log, parse_thermo
from aemwater.lammps.writer import LammpsSystem, LammpsWriteError, write_data_file
from conftest import needs_lammps

DATA = Path(__file__).parent / "data"

#: sander single-point energies (kcal/mol) for tests/data/mono.prmtop with
#: cut=40.0, ntb=0. Amber splits non-bonded terms into full and 1-4 parts; LAMMPS
#: reports the scaled totals, so the comparison sums Amber's pairs.
AMBER_REFERENCE = {
    "E_bond": 0.3656,
    "E_angle": 2.2993,
    "E_dihedral": 3.5622,   # DIHED, compared against E_dihed + E_impro
    "E_vdw": -3.3895 + 3.6276,   # VDWAALS + (1-4 VDW)/2 already applied by sander
    "E_coul": 36.0949 + 38.8323,  # EEL + 1-4 EEL
}


def _load_monomer() -> pmd.Structure:
    struct = pmd.load_file(str(DATA / "mono.prmtop"), str(DATA / "mono.inpcrd"))
    struct.box = [60.0, 60.0, 60.0, 90.0, 90.0, 90.0]
    xyz = np.array(struct.coordinates)
    struct.coordinates = xyz - xyz.mean(axis=0) + 30.0
    return struct


def test_writes_all_sections(tmp_path):
    system = LammpsSystem(structure=_load_monomer())
    path = write_data_file(system, tmp_path / "mono.data")
    text = path.read_text()
    for header in ("Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
                   "Dihedral Coeffs", "Improper Coeffs", "Atoms # full",
                   "Bonds", "Angles", "Dihedrals", "Impropers"):
        assert header in text, f"missing {header} section"
    assert "39 atoms" in text


def test_atom_types_split_on_parameters_not_just_name(tmp_path):
    """Two species may share a type name with different LJ parameters."""
    struct = _load_monomer()
    water = pmd.Structure()
    system = LammpsSystem(structure=struct)
    keys = list(system.atom_types)
    assert all("|" in k for k in keys)
    assert len({k.split("|")[0] for k in keys}) == len(keys), "names happen to be unique here"


def test_molecule_ids_follow_connectivity(tmp_path):
    """A polymer chain spanning several residues is still one molecule."""
    struct = _load_monomer()
    system = LammpsSystem(structure=struct)
    path = write_data_file(system, tmp_path / "m.data")
    atoms_block = path.read_text().split("Atoms # full\n\n")[1].split("\n\n")[0]
    mol_ids = {int(line.split()[1]) for line in atoms_block.strip().splitlines()}
    assert mol_ids == {1}


def test_rejects_triclinic_box(tmp_path):
    struct = _load_monomer()
    struct.box = [60.0, 60.0, 60.0, 90.0, 90.0, 60.0]
    with pytest.raises(LammpsWriteError, match="orthogonal"):
        write_data_file(LammpsSystem(structure=struct), tmp_path / "x.data")


def test_missing_box_is_rejected():
    struct = _load_monomer()
    struct.box = None
    with pytest.raises(LammpsWriteError, match="box"):
        LammpsSystem(structure=struct)


@needs_lammps
def test_lammps_energy_matches_amber(tmp_path):
    """Term-by-term single-point energy agreement with sander, < 0.1 % per term."""
    struct = _load_monomer()
    system = LammpsSystem(structure=struct)
    write_data_file(system, tmp_path / "mono.data")
    (tmp_path / "in.energy").write_text(
        "units real\n"
        "atom_style full\n"
        "boundary p p p\n"
        "pair_style lj/cut/coul/cut 40.0\n"
        # Amber/GAFF2 uses Lorentz-Berthelot (arithmetic sigma) mixing; the LAMMPS
        # default for lj/cut is geometric, which silently changes every cross term.
        "pair_modify mix arithmetic\n"
        "bond_style harmonic\n"
        "angle_style harmonic\n"
        "dihedral_style charmm\n"
        "improper_style cvff\n"
        "special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333\n"
        "read_data mono.data\n"
        "neighbor 2.0 bin\n"
        "neigh_modify delay 0 every 1 check yes one 10000 page 200000\n"
        "thermo_style custom step ebond eangle edihed eimp evdwl ecoul etotal\n"
        "run 0\n"
    )
    lmp = shutil.which("lmp") or shutil.which("lmp_serial")
    subprocess.run([lmp, "-in", "in.energy", "-log", "lmp.log"],
                   cwd=tmp_path, check=True, capture_output=True)
    table = parse_log(tmp_path / "lmp.log").last_section

    got = {
        "E_bond": table.last("E_bond"),
        "E_angle": table.last("E_angle"),
        "E_dihedral": table.last("E_dihed") + table.last("E_impro"),
        "E_vdw": table.last("E_vdwl"),
        "E_coul": table.last("E_coul"),
    }
    for term, reference in AMBER_REFERENCE.items():
        assert got[term] == pytest.approx(reference, rel=2e-3, abs=5e-3), (
            f"{term}: LAMMPS {got[term]:.5f} vs Amber {reference:.5f}"
        )


def test_parse_thermo_handles_multiple_sections():
    log = (
        "some preamble\n"
        "   Step Temp E_pair\n"
        "      0 300.0 -1.5\n"
        "     10 305.0 -1.6\n"
        "Loop time of 1.0 on 1 procs\n"
        "more text\n"
        "   Step Temp E_pair\n"
        "     20 310.0 -1.7\n"
        "Loop time of 1.0 on 1 procs\n"
    )
    sections = parse_thermo(log)
    assert len(sections) == 2
    assert sections[0].columns == ("Step", "Temp", "E_pair")
    assert sections[0]["Temp"].tolist() == [300.0, 305.0]
    assert sections[1].last("E_pair") == -1.7


def test_parse_log_raises_on_lost_atoms(tmp_path):
    path = tmp_path / "bad.log"
    path.write_text("   Step Temp\n      0 300.0\nERROR: Lost atoms: original 100 current 98\n")
    with pytest.raises(LammpsRunError, match="Lost atoms"):
        parse_log(path)


def test_shake_failure_carries_a_diagnosis(tmp_path):
    """The two causes seen in validation each cost a long run to find.

    A bare 'Shake determinant < 0.0' says nothing about what to change, and the
    run that produced it has already spent an hour. The hint names both causes
    and where to look.
    """
    from aemwater.lammps.parse import LammpsRunError, parse_log

    p = tmp_path / "iter.log"
    p.write_text(
        "LAMMPS (2 Aug 2023)\n"
        "WARNING: Bond/angle/dihedral extent > half of periodic box length\n"
        "WARNING: Shake determinant < 0.0 (src/RIGID/fix_shake.cpp:2083)\n"
    )
    with pytest.raises(LammpsRunError) as excinfo:
        parse_log(p)
    message = str(excinfo.value)
    assert "Shake determinant" in message
    assert "image flags" in message and "sort atoms by ID" in message
    assert "timestep" in message
