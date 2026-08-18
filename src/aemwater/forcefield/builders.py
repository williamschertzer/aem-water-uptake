"""ParmEd structures for water and small ions from tabulated parameters.

These species deliberately bypass antechamber. A water model is a *published,
jointly fitted* set of geometry, charges and Lennard-Jones parameters; replacing
its charges with AM1-BCC values would produce a liquid that is no longer SPC/E or
TIP3P and would invalidate every reference density and chemical potential the
uptake calculation is compared against.

Water is written with real bond and angle terms even though it is constrained at
run time by SHAKE. LAMMPS needs the bonded topology to exist so that intramolecular
non-bonded exclusions are set up correctly; the force constants are the flexible
SPC/E values and are never integrated once SHAKE is applied.
"""

from __future__ import annotations

import math

import numpy as np
import parmed as pmd

from .water import IonParameters, WaterModel

#: Flexible-SPC/E-like force constants, kcal/mol. Used only to define topology;
#: SHAKE (or fix rigid) removes these degrees of freedom at run time.
WATER_BOND_K = 553.0
WATER_ANGLE_K = 100.0


def _atom(name: str, element: str, mass: float, charge: float, sigma: float, epsilon: float,
          atom_type_name: str) -> pmd.Atom:
    atom = pmd.Atom(
        name=name,
        atomic_number=pmd.periodic_table.AtomicNum[element],
        mass=mass,
        charge=charge,
        type=atom_type_name,
    )
    # rmin = sigma * 2^(1/6) / 2; ParmEd stores rmin, LAMMPS wants sigma.
    atype = pmd.AtomType(atom_type_name, None, mass, pmd.periodic_table.AtomicNum[element])
    atype.set_lj_params(eps=epsilon, rmin=sigma * 2 ** (1 / 6) / 2)
    atom.atom_type = atype
    atom.type = atom_type_name
    atom.charge = charge
    return atom


def water_structure(model: WaterModel) -> pmd.Structure:
    """A single rigid water molecule as a parameterised ParmEd structure."""
    model.validate()
    struct = pmd.Structure()
    o = _atom("OW", "O", model.mass_O, model.charge_O, model.sigma_O, model.epsilon_O, f"OW_{model.name}")
    h1 = _atom("HW1", "H", model.mass_H, model.charge_H, 0.0, 0.0, f"HW_{model.name}")
    h2 = _atom("HW2", "H", model.mass_H, model.charge_H, 0.0, 0.0, f"HW_{model.name}")
    for atom in (o, h1, h2):
        struct.add_atom(atom, "WAT", 1)

    bt = pmd.BondType(WATER_BOND_K, model.r_OH)
    struct.bond_types.append(bt)
    bt.list = struct.bond_types
    for h in (h1, h2):
        bond = pmd.Bond(o, h, type=bt)
        struct.bonds.append(bond)

    at = pmd.AngleType(WATER_ANGLE_K, model.angle_HOH)
    struct.angle_types.append(at)
    at.list = struct.angle_types
    struct.angles.append(pmd.Angle(h1, o, h2, type=at))

    # Geometry: oxygen at the origin, molecule in the xz-plane.
    half = math.radians(model.angle_HOH) / 2.0
    r = model.r_OH
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [r * math.sin(half), 0.0, r * math.cos(half)],
            [-r * math.sin(half), 0.0, r * math.cos(half)],
        ]
    )
    struct.coordinates = coords
    return struct


def ion_structure(params: IonParameters, residue_name: str | None = None) -> pmd.Structure:
    """A single mobile ion as a parameterised ParmEd structure."""
    resname = residue_name or params.label.replace("-", "").replace("+", "")[:3].upper()
    struct = pmd.Structure()
    atoms = []
    for i, (element, sigma, epsilon, charge, mass) in enumerate(params.sites):
        name = f"{element}{i + 1}" if len(params.sites) > 1 else element
        atom = _atom(name, element, mass, charge, sigma, epsilon, f"{element}_{resname}")
        struct.add_atom(atom, resname, 1)
        atoms.append(atom)

    if params.bonds:
        for i, j, req in params.bonds:
            bt = pmd.BondType(WATER_BOND_K, req)
            struct.bond_types.append(bt)
            bt.list = struct.bond_types
            struct.bonds.append(pmd.Bond(atoms[i], atoms[j], type=bt))

    coords = [[0.0, 0.0, 0.0]]
    for i, j, req in params.bonds:
        coords.append([0.0, 0.0, req])
    struct.coordinates = np.array(coords[: len(atoms)])
    return struct


__all__ = ["water_structure", "ion_structure", "WATER_BOND_K", "WATER_ANGLE_K"]
