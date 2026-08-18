"""Minimal TRIPOS mol2 reader/writer carrying GAFF2 types and charges.

RDKit cannot write a mol2 with externally supplied atom types, and tleap will not
read anything else that carries both types and charges, so this module owns the
one file format that couples the two halves of the toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem

#: RDKit bond order -> TRIPOS bond type.
_BOND_TYPES = {
    Chem.BondType.SINGLE: "1",
    Chem.BondType.DOUBLE: "2",
    Chem.BondType.TRIPLE: "3",
    Chem.BondType.AROMATIC: "ar",
}


@dataclass(frozen=True)
class Mol2Atoms:
    """Per-atom GAFF2 assignment read back from a mol2 file."""

    names: tuple[str, ...]
    types: tuple[str, ...]
    charges: tuple[float, ...]
    elements: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.types)

    @property
    def total_charge(self) -> float:
        return float(sum(self.charges))


def read_mol2(path: Path) -> Mol2Atoms:
    """Read the ATOM block of a mol2 file."""
    text = Path(path).read_text()
    if "@<TRIPOS>ATOM" not in text:
        raise ValueError(f"{path} has no @<TRIPOS>ATOM record")
    block = text.split("@<TRIPOS>ATOM", 1)[1]
    for terminator in ("@<TRIPOS>BOND", "@<TRIPOS>SUBSTRUCTURE"):
        block = block.split(terminator, 1)[0]
    names, types, charges, elements = [], [], [], []
    for line in block.strip().splitlines():
        fields = line.split()
        if len(fields) < 9:
            continue
        names.append(fields[1])
        types.append(fields[5])
        charges.append(float(fields[8]))
        # GAFF types are lower-case and not element symbols, so the element is
        # taken from the atom name, which antechamber writes as <El><serial>.
        elements.append("".join(c for c in fields[1] if c.isalpha()).capitalize())
    return Mol2Atoms(tuple(names), tuple(types), tuple(charges), tuple(elements))


def write_mol2(
    mol: Chem.Mol,
    types: Sequence[str],
    charges: Sequence[float],
    path: Path,
    resname: str = "MOL",
    molname: str | None = None,
) -> Path:
    """Write ``mol`` as a mol2 file with the supplied GAFF2 types and charges."""
    n = mol.GetNumAtoms()
    if len(types) != n or len(charges) != n:
        raise ValueError(f"expected {n} types/charges, got {len(types)}/{len(charges)}")
    if mol.GetNumConformers() == 0:
        raise ValueError("write_mol2 requires 3D coordinates")
    pos = np.array(mol.GetConformer().GetPositions())

    lines = [
        "@<TRIPOS>MOLECULE",
        molname or resname,
        f"{n:5d}{mol.GetNumBonds():6d}{1:6d}{0:6d}{0:6d}",
        "SMALL",
        "USER_CHARGES",
        "",
        "",
        "@<TRIPOS>ATOM",
    ]
    counters: dict[str, int] = {}
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        counters[sym] = counters.get(sym, 0) + 1
        name = f"{sym}{counters[sym]}"
        x, y, z = pos[i]
        lines.append(
            f"{i + 1:7d} {name:<8s}{x:10.4f}{y:10.4f}{z:10.4f} "
            f"{types[i]:<8s}{1:5d} {resname:<8s}{charges[i]:12.6f}"
        )
    lines.append("@<TRIPOS>BOND")
    for k, bond in enumerate(mol.GetBonds(), start=1):
        btype = _BOND_TYPES.get(bond.GetBondType(), "1")
        lines.append(
            f"{k:6d}{bond.GetBeginAtomIdx() + 1:6d}{bond.GetEndAtomIdx() + 1:6d} {btype:<4s}"
        )
    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append(f"{1:7d} {resname:<8s}{1:6d} RESIDUE{0:12d} ****  ****{0:6d} ROOT")
    lines.append("")
    Path(path).write_text("\n".join(lines))
    return Path(path)


__all__ = ["Mol2Atoms", "read_mol2", "write_mol2"]
