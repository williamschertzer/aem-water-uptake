"""Typing-backend protocol: SMILES-derived molecules -> parameterised system.

A backend's single job is to return a :class:`TypedSystem` whose ``structure`` is
a fully parameterised ParmEd ``Structure``. Everything downstream --
:mod:`aemwater.lammps.writer`, packing, insertion, Widom -- consumes only that,
so an alternative backend (OpenFF/SMIRNOFF, a bespoke ionomer force field, or a
hand-built parameter set) can be dropped in without touching the rest of the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import parmed as pmd


class ForceFieldError(RuntimeError):
    """Raised when parameterisation fails or produces an incomplete system."""


@dataclass
class MoleculeSpec:
    """One molecular species to be typed, with the copy count in the cell."""

    name: str
    #: RDKit molecule with explicit hydrogens and 3D coordinates.
    mol: object
    count: int
    #: Residue name used in the topology (<= 3 characters for PDB compatibility).
    residue_name: str = "MOL"
    #: Net formal charge; checked against the typed result.
    formal_charge: int = 0


@dataclass
class TypedSystem:
    """A parameterised molecular system plus provenance."""

    structure: pmd.Structure
    #: Per-species ParmEd structures (one copy each), for reuse and diagnostics.
    templates: dict[str, pmd.Structure] = field(default_factory=dict)
    #: Free-form record of how the parameters were obtained.
    provenance: dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------- checks ----
    @property
    def net_charge(self) -> float:
        return float(sum(a.charge for a in self.structure.atoms))

    def assert_neutral(self, tolerance: float = 1.0e-4) -> None:
        q = self.net_charge
        if abs(q) > tolerance:
            raise ForceFieldError(
                f"system net charge is {q:+.6f} e, which exceeds the tolerance {tolerance}. "
                "A non-neutral cell makes the PPPM Coulomb energy ill-defined; check the "
                "counterion count and the per-residue charge correction."
            )

    def assert_fully_parameterised(self) -> None:
        """Every interaction present must have parameters attached."""
        missing: list[str] = []
        for atom in self.structure.atoms:
            if atom.atom_type is None and (atom.type in (None, "")):
                missing.append(f"atom {atom.idx} ({atom.name}) has no type")
            if atom.epsilon is None or atom.sigma is None:
                missing.append(f"atom {atom.idx} ({atom.name}) has no LJ parameters")
        for bond in self.structure.bonds:
            if bond.type is None:
                missing.append(f"bond {bond.atom1.name}-{bond.atom2.name} has no parameters")
        for angle in self.structure.angles:
            if angle.type is None:
                missing.append(
                    f"angle {angle.atom1.name}-{angle.atom2.name}-{angle.atom3.name} has none"
                )
        for dih in self.structure.dihedrals:
            if dih.type is None:
                missing.append("a dihedral has no parameters")
        if missing:
            preview = "\n  ".join(missing[:12])
            raise ForceFieldError(
                f"{len(missing)} unparameterised interaction(s) found; the first few are:\n  {preview}\n"
                "For GAFF2 this usually means parmchk2 could not supply a missing term -- "
                "inspect the generated .frcmod for ATTN comments."
            )

    def summary(self) -> dict[str, object]:
        s = self.structure
        return {
            "atoms": len(s.atoms),
            "bonds": len(s.bonds),
            "angles": len(s.angles),
            "dihedrals": len(s.dihedrals),
            "impropers": len(getattr(s, "impropers", []) or []),
            "residues": len(s.residues),
            "atom_types": len({a.type for a in s.atoms}),
            "net_charge": round(self.net_charge, 6),
            "total_mass_amu": round(sum(a.mass for a in s.atoms), 4),
        }


@runtime_checkable
class TypingBackend(Protocol):
    """Assigns atom types, charges and bonded parameters to a set of molecules."""

    name: str

    def type_system(
        self,
        molecules: Sequence[MoleculeSpec],
        workdir: Path,
    ) -> TypedSystem:
        """Return a fully parameterised :class:`TypedSystem`."""
        ...


__all__ = ["TypedSystem", "TypingBackend", "MoleculeSpec", "ForceFieldError"]
