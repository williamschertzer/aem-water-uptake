"""Assemble typed molecules plus packed coordinates into a writable system.

This is the seam between the chemistry half of the workflow (which produces
parameterised molecules with no particular position) and the simulation half
(which needs one periodic structure with coordinates). Keeping it separate means
water can be added to an existing cell by exactly the same code path that built
the dry cell in the first place -- important, because the insertion loop calls it
once per iteration.

Ordering is a contract, not an implementation detail: chains first, then
counterions, then water. Molecule IDs follow that order in the data file, and the
LAMMPS group definitions in ``templates/common.in.j2`` rely on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import parmed as pmd

from .forcefield.builders import ion_structure, water_structure
from .forcefield.water import WaterModel, ion_parameters, water_model
from .lammps.writer import LammpsSystem
from .utils import LOG


class AssemblyError(RuntimeError):
    """Raised when molecules and coordinates cannot be combined consistently."""


@dataclass
class CellContents:
    """The molecular inventory of a cell, in canonical order."""

    chains: list[pmd.Structure]
    ions: list[pmd.Structure]
    waters: list[pmd.Structure]

    @property
    def molecules(self) -> list[pmd.Structure]:
        return [*self.chains, *self.ions, *self.waters]

    @property
    def n_atoms(self) -> int:
        return sum(len(m.atoms) for m in self.molecules)

    def counts(self) -> dict[str, int]:
        return {
            "chains": len(self.chains),
            "ions": len(self.ions),
            "waters": len(self.waters),
            "atoms": self.n_atoms,
        }


def replicate(structure: pmd.Structure, n: int) -> list[pmd.Structure]:
    """``n`` independent copies of a typed structure.

    Copies are deep: LAMMPS type indexing walks every atom, and sharing atom
    objects between molecules would make two chains collapse onto one molecule ID.
    """
    return [pmd.structure.copy(structure) for _ in range(n)]


def water_molecules(n: int, model: WaterModel | str) -> list[pmd.Structure]:
    """``n`` copies of the rigid water model."""
    if n <= 0:
        return []
    resolved = water_model(model) if isinstance(model, str) else model
    return replicate(water_structure(resolved), n)


def ion_molecules(n: int, ion) -> list[pmd.Structure]:
    """``n`` copies of a monatomic counterion.

    ``ion`` may be a label (``"Cl-"``), a :class:`Counterion` from the chemistry
    layer, or already-resolved :class:`IonParameters`; the composition object
    carries the middle form, so accepting only a string would force every caller
    to reach into it.
    """
    if n <= 0:
        return []
    params = ion
    if hasattr(ion, "label"):          # Counterion from chemistry.py
        params = ion_parameters(ion.label)
    elif isinstance(ion, str):
        params = ion_parameters(ion)
    return replicate(ion_structure(params), n)


def assemble(
    contents: CellContents,
    coordinates: np.ndarray,
    edge: float,
    water_model_name: str = "SPC/E",
) -> LammpsSystem:
    """Concatenate molecules into one periodic structure carrying ``coordinates``.

    ``coordinates`` must be ordered to match ``contents.molecules`` atom for atom;
    the check below is not defensive padding, it is the one place where a
    mis-ordered coordinate array would otherwise produce a plausible-looking cell
    with scrambled geometry.
    """
    molecules = contents.molecules
    if not molecules:
        raise AssemblyError("no molecules to assemble")
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (contents.n_atoms, 3):
        raise AssemblyError(
            f"coordinate array is {coordinates.shape}, expected "
            f"({contents.n_atoms}, 3) for {len(molecules)} molecules"
        )

    # Accumulate into a plain Structure rather than adding AmberParm objects
    # together. ParmEd's AmberParm.__add__ goes through AmberFormat.__copy__,
    # which increments a counter set in __init__ but not restored by
    # __setstate__: any structure that has been through a pickle -- a cache, a
    # multiprocessing boundary -- raises AttributeError on _ncopies (ParmEd
    # 4.3.1). Downcasting sidesteps that path entirely, and the LAMMPS writer
    # wants a plain Structure anyway. Charges, atom types and all parameter
    # types survive the downcast; the assertion below checks the atom count.
    combined = pmd.structure.Structure()
    for mol in molecules:
        combined += mol.copy(pmd.structure.Structure)
    if len(combined.atoms) != contents.n_atoms:
        raise AssemblyError(
            f"assembly produced {len(combined.atoms)} atoms, expected "
            f"{contents.n_atoms}"
        )
    combined.coordinates = coordinates
    combined.box = [edge, edge, edge, 90.0, 90.0, 90.0]

    ion_names = tuple({r.name for m in contents.ions for r in m.residues})
    water_names = {r.name for m in contents.waters for r in m.residues}
    system = LammpsSystem(
        structure=combined,
        water_residue=next(iter(water_names)) if water_names else "WAT",
        ion_residues=ion_names,
    )
    LOG.info(
        "assembled %s in a %.2f A cell -> %d LAMMPS atom types",
        contents.counts(),
        edge,
        len(system.atom_types),
    )
    return system


def molecule_id_ranges(contents: CellContents) -> dict[str, tuple[int, int]]:
    """Inclusive 1-based molecule-ID ranges for each species group."""
    n_chain, n_ion, n_wat = len(contents.chains), len(contents.ions), len(contents.waters)
    ranges = {"polymer": (1, n_chain)}
    if n_ion:
        ranges["ions"] = (n_chain + 1, n_chain + n_ion)
    if n_wat:
        ranges["water"] = (n_chain + n_ion + 1, n_chain + n_ion + n_wat)
    return ranges


__all__ = [
    "CellContents",
    "AssemblyError",
    "assemble",
    "replicate",
    "water_molecules",
    "ion_molecules",
    "molecule_id_ranges",
]
