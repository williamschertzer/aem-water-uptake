"""Write a parameterised ParmEd system as a LAMMPS data file.

Style mapping (Amber/GAFF2 -> LAMMPS ``real`` units)
----------------------------------------------------
=====================  ============================================
LAMMPS setting         Reason
=====================  ============================================
``atom_style full``    charges plus molecule IDs are both required
``pair_style lj/cut/coul/long``  GAFF2 is a 12-6 LJ + point-charge model
``bond_style harmonic``  Amber ``K(r-r0)^2``, K already per the LAMMPS convention
``angle_style harmonic``  same convention
``dihedral_style charmm``  Amber propers are ``K[1+cos(n*phi - phase)]`` with
                       phase restricted to 0 or 180 degrees, which is exactly
                       the CHARMM form with integer ``d``; verified against the
                       generated topology rather than assumed
``improper_style cvff``  Amber impropers all have phase 180 and periodicity 2,
                       giving ``K[1 - cos(2*phi)]`` = cvff with ``d = -1``
``special_bonds amber``  1-4 scaling 0.5 (LJ) and 1/1.2 (Coulomb)
=====================  ============================================

The 1-4 scale factors are applied through ``special_bonds`` rather than through
the CHARMM dihedral weight factor, so every dihedral is written with weight 0.
Applying both would double-count the 1-4 interaction; Amber topologies also mark
duplicate multi-term dihedrals so that a per-dihedral weight would miscount them.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import parmed as pmd

from ..utils import LOG

#: Amber 1-4 scaling: LJ divided by 2.0, Coulomb by 1.2.
SPECIAL_BONDS = "lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333333333"


class LammpsWriteError(RuntimeError):
    """Raised when a structure cannot be expressed in the chosen LAMMPS styles."""


@dataclass
class LammpsSystem:
    """A ParmEd system indexed into LAMMPS type numbering.

    Attributes
    ----------
    structure:
        The full periodic system. ``structure.box`` must be set.
    water_residue:
        Residue name used for water, so the input generator can build the SHAKE
        group and the insertion/deletion molecule template.
    ion_residues:
        Residue names of mobile counterions.
    """

    structure: pmd.Structure
    water_residue: str = "WAT"
    ion_residues: tuple[str, ...] = ()
    atom_types: "OrderedDict[str, int]" = field(default_factory=OrderedDict)
    bond_types: "OrderedDict[tuple, int]" = field(default_factory=OrderedDict)
    angle_types: "OrderedDict[tuple, int]" = field(default_factory=OrderedDict)
    dihedral_types: "OrderedDict[tuple, int]" = field(default_factory=OrderedDict)
    improper_types: "OrderedDict[tuple, int]" = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        if self.structure.box is None:
            raise LammpsWriteError("structure.box must be set before writing LAMMPS data")
        self._index_types()

    # ------------------------------------------------------------ indexing ---
    def _index_types(self) -> None:
        for atom in self.structure.atoms:
            key = self._atom_type_key(atom)
            if key not in self.atom_types:
                self.atom_types[key] = len(self.atom_types) + 1
        for bond in self.structure.bonds:
            key = self._bond_key(bond)
            if key not in self.bond_types:
                self.bond_types[key] = len(self.bond_types) + 1
        for angle in self.structure.angles:
            key = self._angle_key(angle)
            if key not in self.angle_types:
                self.angle_types[key] = len(self.angle_types) + 1
        for dih in self.structure.dihedrals:
            key = self._dihedral_key(dih)
            table = self.improper_types if dih.improper else self.dihedral_types
            if key not in table:
                table[key] = len(table) + 1
        for imp in getattr(self.structure, "impropers", []) or []:
            key = ("harm", round(imp.type.psi_k, 6), round(imp.type.psi_eq, 6))
            if key not in self.improper_types:
                self.improper_types[key] = len(self.improper_types) + 1

    @staticmethod
    def _atom_type_key(atom: pmd.Atom) -> str:
        """Distinguish atom types by name *and* LJ parameters.

        Two different molecules can both use a type named ``o`` with different
        parameters (for instance GAFF2 ``o`` and a water oxygen), so folding on
        the name alone would silently overwrite one of them.
        """
        eps = atom.epsilon if atom.epsilon is not None else 0.0
        sig = atom.sigma if atom.sigma is not None else 0.0
        return f"{atom.type}|{eps:.6f}|{sig:.6f}|{atom.mass:.4f}"

    @staticmethod
    def _bond_key(bond: pmd.Bond) -> tuple:
        if bond.type is None:
            raise LammpsWriteError(f"bond {bond} has no parameters")
        return (round(bond.type.k, 6), round(bond.type.req, 6))

    @staticmethod
    def _angle_key(angle: pmd.Angle) -> tuple:
        if angle.type is None:
            raise LammpsWriteError(f"angle {angle} has no parameters")
        return (round(angle.type.k, 6), round(angle.type.theteq, 6))

    @staticmethod
    def _dihedral_key(dih: pmd.Dihedral) -> tuple:
        if dih.type is None:
            raise LammpsWriteError("dihedral has no parameters")
        t = dih.type
        phase = round(float(t.phase), 3) % 360.0
        if phase not in (0.0, 180.0):
            raise LammpsWriteError(
                f"dihedral phase {phase} deg cannot be represented by dihedral_style charmm, "
                "which allows only 0 or 180 degrees. Use a Fourier/opls style instead."
            )
        return (round(float(t.phi_k), 6), int(t.per), int(phase))

    # --------------------------------------------------------------- masses --
    def masses(self) -> list[tuple[int, float, str]]:
        seen: dict[int, tuple[float, str]] = {}
        for atom in self.structure.atoms:
            tid = self.atom_types[self._atom_type_key(atom)]
            seen.setdefault(tid, (atom.mass, atom.type))
        return [(tid, m, name) for tid, (m, name) in sorted(seen.items())]

    def pair_coeffs(self) -> list[tuple[int, float, float, str]]:
        seen: dict[int, tuple[float, float, str]] = {}
        for atom in self.structure.atoms:
            tid = self.atom_types[self._atom_type_key(atom)]
            eps = atom.epsilon if atom.epsilon is not None else 0.0
            sig = atom.sigma if atom.sigma is not None else 0.0
            seen.setdefault(tid, (eps, sig, atom.type))
        return [(tid, e, s, n) for tid, (e, s, n) in sorted(seen.items())]

    # ---------------------------------------------------- water type lookup --
    def water_atom_types(self) -> tuple[int, int]:
        """Numeric LAMMPS atom types of the water oxygen and hydrogen.

        The templates need integers -- SHAKE, group definitions and the Widom
        molecule template all address atoms by type, not by name. Water is
        identified by residue name rather than by element, because a polymer
        hydroxyl oxygen is also an ``O`` and constraining it would be wrong.
        """
        o_type = h_type = None
        for atom in self.structure.atoms:
            if atom.residue.name != self.water_residue:
                continue
            tid = self.atom_types[self._atom_type_key(atom)]
            if atom.atomic_number == 8:
                o_type = tid
            elif atom.atomic_number == 1:
                h_type = tid
            if o_type is not None and h_type is not None:
                break
        if o_type is None or h_type is None:
            raise LammpsWriteError(
                f"no {self.water_residue} residue with both O and H found; the "
                "system contains no water, so water types cannot be resolved."
            )
        return o_type, h_type

    def water_bond_type(self) -> int:
        """LAMMPS bond type of the water O-H bond, for SHAKE."""
        for bond in self.structure.bonds:
            if bond.atom1.residue.name == self.water_residue:
                return self.bond_types[self._bond_key(bond)]
        raise LammpsWriteError(f"no {self.water_residue} bond found")

    def water_angle_type(self) -> int:
        """LAMMPS angle type of the water H-O-H angle, for SHAKE."""
        for angle in self.structure.angles:
            if angle.atom1.residue.name == self.water_residue:
                return self.angle_types[self._angle_key(angle)]
        raise LammpsWriteError(f"no {self.water_residue} angle found")

    def n_ion_molecules(self) -> int:
        """Number of mobile counterion molecules currently in the cell.

        Counted from the structure rather than from the composition: the two
        agree for the dry cell, but only this one stays correct once water has
        been inserted and residues renumbered.
        """
        return sum(1 for res in self.structure.residues
                   if res.name in self.ion_residues)

    def n_polymer_molecules(self) -> int:
        """Number of polymer chains, i.e. residues that are neither water nor ion."""
        skip = set(self.ion_residues) | {self.water_residue}
        return sum(1 for res in self.structure.residues if res.name not in skip)

    def has_water(self) -> bool:
        return any(r.name == self.water_residue for r in self.structure.residues)

    def type_counts(self) -> dict[str, int]:
        return {
            "atom types": len(self.atom_types),
            "bond types": len(self.bond_types),
            "angle types": len(self.angle_types),
            "dihedral types": len(self.dihedral_types),
            "improper types": len(self.improper_types),
        }

    def summary(self) -> dict[str, object]:
        s = self.structure
        box = s.box
        return {
            "atoms": len(s.atoms),
            "molecules": len({a.residue.idx for a in s.atoms}) if s.residues else 0,
            "residues": len(s.residues),
            "net_charge": round(float(sum(a.charge for a in s.atoms)), 6),
            "box_A": [round(float(x), 4) for x in box[:3]],
            "volume_A3": round(float(np.prod(box[:3])), 3),
            **self.type_counts(),
        }


def _molecule_ids(structure: pmd.Structure) -> list[int]:
    """Assign a LAMMPS molecule ID per connected molecule.

    Residue index is not a safe proxy: a polymer chain is one molecule that may
    span many residues, and LAMMPS uses molecule IDs for SHAKE grouping and for
    per-molecule counting during insertion, so they must reflect real bonded
    connectivity.
    """
    n = len(structure.atoms)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for bond in structure.bonds:
        union(bond.atom1.idx, bond.atom2.idx)

    mol_of: dict[int, int] = {}
    ids = []
    for i in range(n):
        root = find(i)
        if root not in mol_of:
            mol_of[root] = len(mol_of) + 1
        ids.append(mol_of[root])
    return ids


def _rounded_charges(structure: pmd.Structure, decimals: int = 6) -> list[float]:
    """Per-atom charges rounded for output, corrected to sum to the exact total.

    Charges are written to finite precision, so a few hundred atoms accumulate a
    rounding residual of order 1e-4 e. PPPM applies a uniform neutralising
    background to any net charge, which silently shifts the electrostatic energy,
    and LAMMPS warns about it. The residual is absorbed by the single atom with
    the largest magnitude charge, where a 1e-4 e correction is negligible
    relative to its own value.
    """
    charges = [float(a.charge) for a in structure.atoms]
    rounded = [round(q, decimals) for q in charges]
    exact_total = round(sum(charges))
    residual = exact_total - sum(rounded)
    if abs(residual) > 0.5 * 10 ** (-decimals):
        idx = max(range(len(rounded)), key=lambda i: abs(rounded[i]))
        rounded[idx] = round(rounded[idx] + residual, decimals)
    return rounded


def write_data_file(system: LammpsSystem, path: Path, title: str = "") -> Path:
    """Write ``system`` as a LAMMPS ``atom_style full`` data file."""
    struct = system.structure
    path = Path(path)
    box = struct.box
    if any(abs(float(a) - 90.0) > 1.0e-6 for a in box[3:]):
        raise LammpsWriteError(
            f"only orthogonal boxes are supported; got angles {tuple(box[3:])}. "
            "A triclinic cell would need the LAMMPS tilt factors xy xz yz."
        )
    coords = np.asarray(struct.coordinates, dtype=float)
    if coords.shape != (len(struct.atoms), 3):
        raise LammpsWriteError(
            f"coordinate array shape {coords.shape} does not match {len(struct.atoms)} atoms"
        )
    mol_ids = _molecule_ids(struct)
    counts = system.type_counts()

    impropers = list(getattr(struct, "impropers", []) or [])
    dihedrals = [d for d in struct.dihedrals if not d.improper]
    amber_impropers = [d for d in struct.dihedrals if d.improper]
    n_impropers = len(impropers) + len(amber_impropers)

    out: list[str] = []
    out.append(title or "LAMMPS data file written by aemwater")
    out.append("")
    out.append(f"{len(struct.atoms)} atoms")
    out.append(f"{len(struct.bonds)} bonds")
    out.append(f"{len(struct.angles)} angles")
    out.append(f"{len(dihedrals)} dihedrals")
    out.append(f"{n_impropers} impropers")
    out.append("")
    out.append(f"{counts['atom types']} atom types")
    if struct.bonds:
        out.append(f"{counts['bond types']} bond types")
    if struct.angles:
        out.append(f"{counts['angle types']} angle types")
    if dihedrals:
        out.append(f"{counts['dihedral types']} dihedral types")
    if n_impropers:
        out.append(f"{counts['improper types']} improper types")
    out.append("")
    out.append(f"0.0 {float(box[0]):.6f} xlo xhi")
    out.append(f"0.0 {float(box[1]):.6f} ylo yhi")
    out.append(f"0.0 {float(box[2]):.6f} zlo zhi")
    out.append("")

    out.append("Masses")
    out.append("")
    for tid, mass, name in system.masses():
        out.append(f"{tid} {mass:.6f}  # {name}")
    out.append("")

    out.append("Pair Coeffs # lj/cut/coul/long")
    out.append("")
    for tid, eps, sigma, name in system.pair_coeffs():
        out.append(f"{tid} {eps:.8f} {sigma:.8f}  # {name}")
    out.append("")

    if struct.bonds:
        out.append("Bond Coeffs # harmonic")
        out.append("")
        for key, tid in sorted(system.bond_types.items(), key=lambda kv: kv[1]):
            out.append(f"{tid} {key[0]:.6f} {key[1]:.6f}")
        out.append("")

    if struct.angles:
        out.append("Angle Coeffs # harmonic")
        out.append("")
        for key, tid in sorted(system.angle_types.items(), key=lambda kv: kv[1]):
            out.append(f"{tid} {key[0]:.6f} {key[1]:.6f}")
        out.append("")

    if dihedrals:
        out.append("Dihedral Coeffs # charmm")
        out.append("")
        for key, tid in sorted(system.dihedral_types.items(), key=lambda kv: kv[1]):
            phi_k, per, phase = key
            # charmm: K n d weight. n = 0 is not allowed, and the 1-4 weight is 0
            # because special_bonds already applies the Amber 1-4 scaling.
            n = max(1, int(per))
            out.append(f"{tid} {phi_k:.6f} {n} {int(phase)} 0.0")
        out.append("")

    if n_impropers:
        out.append("Improper Coeffs # cvff")
        out.append("")
        for key, tid in sorted(system.improper_types.items(), key=lambda kv: kv[1]):
            if key[0] == "harm":  # pragma: no cover - CHARMM-style improper
                raise LammpsWriteError(
                    "harmonic impropers are present; improper_style cvff cannot express them"
                )
            phi_k, per, phase = key
            # cvff: K d n  with E = K[1 + d cos(n phi)]. Amber phase 180 -> d = -1.
            d = -1 if int(phase) == 180 else 1
            out.append(f"{tid} {phi_k:.6f} {d} {max(1, int(per))}")
        out.append("")

    charges = _rounded_charges(struct)
    out.append("Atoms # full")
    out.append("")
    # Image flags are written explicitly, all zero: these coordinates are
    # unwrapped (whole molecules, no atom teleported to the opposite face), so
    # every atom sits in image 0 by construction. Writing them is not cosmetic --
    # the driver reads this file back to build the next iteration and needs the
    # flags to unwrap; without them a molecule across a boundary is rebuilt
    # stretched across the cell and LAMMPS dies in SHAKE.
    for atom, mol_id, xyz in zip(struct.atoms, mol_ids, coords):
        tid = system.atom_types[system._atom_type_key(atom)]
        out.append(
            f"{atom.idx + 1} {mol_id} {tid} {charges[atom.idx]:.6f} "
            f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 0 0 0"
        )
    out.append("")

    if struct.bonds:
        out.append("Bonds")
        out.append("")
        for i, bond in enumerate(struct.bonds, start=1):
            tid = system.bond_types[system._bond_key(bond)]
            out.append(f"{i} {tid} {bond.atom1.idx + 1} {bond.atom2.idx + 1}")
        out.append("")

    if struct.angles:
        out.append("Angles")
        out.append("")
        for i, angle in enumerate(struct.angles, start=1):
            tid = system.angle_types[system._angle_key(angle)]
            out.append(
                f"{i} {tid} {angle.atom1.idx + 1} {angle.atom2.idx + 1} {angle.atom3.idx + 1}"
            )
        out.append("")

    if dihedrals:
        out.append("Dihedrals")
        out.append("")
        for i, dih in enumerate(dihedrals, start=1):
            tid = system.dihedral_types[system._dihedral_key(dih)]
            out.append(
                f"{i} {tid} {dih.atom1.idx + 1} {dih.atom2.idx + 1} "
                f"{dih.atom3.idx + 1} {dih.atom4.idx + 1}"
            )
        out.append("")

    if n_impropers:
        out.append("Impropers")
        out.append("")
        for i, imp in enumerate(amber_impropers, start=1):
            tid = system.improper_types[system._dihedral_key(imp)]
            out.append(
                f"{i} {tid} {imp.atom1.idx + 1} {imp.atom2.idx + 1} "
                f"{imp.atom3.idx + 1} {imp.atom4.idx + 1}"
            )
        out.append("")

    path.write_text("\n".join(out))
    LOG.info(
        "wrote %s: %d atoms, %d types, box %.2f x %.2f x %.2f A",
        path.name,
        len(struct.atoms),
        counts["atom types"],
        float(box[0]),
        float(box[1]),
        float(box[2]),
    )
    return path


__all__ = ["LammpsSystem", "write_data_file", "LammpsWriteError", "SPECIAL_BONDS"]
