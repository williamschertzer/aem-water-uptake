"""GAFF2 typing via AmberTools (antechamber -> parmchk2 -> tleap).

Why a fragment-transfer scheme
------------------------------
AM1-BCC charges come from a semi-empirical QM calculation (``sqm``), which scales
badly and in practice will not converge for a chain of a few hundred atoms, let
alone the thousands in a membrane cell. GAFF atom types and AM1-BCC charges are
however both *local*: they depend on an atom's bonded environment out to two or
three bonds. So the charges are computed once on a small **capped model
fragment** -- one repeat unit with methyl groups standing in for the neighbouring
units -- and transferred onto every unit of every chain by exact atom
correspondence, using the template indices recorded by :mod:`aemwater.polymer`.

That is the standard construction for polymer force fields, and it has one
bookkeeping consequence that must be handled explicitly: the fragment's charge is
spread over the unit *and* its two capping methyls, so simply copying the unit
atoms' charges leaves a residual. The residual is redistributed (see
:func:`_transfer_charges`) so that each chain carries exactly its integer formal
charge, which is what makes the periodic Ewald sum well defined.

Bonded parameters are not transferred by hand: the fully typed chain is written
as a mol2 file and handed to ``tleap``, which looks up every bond, angle and
dihedral in ``gaff2.dat`` from the atom types, and to ``parmchk2`` for any term
gaff2 lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import parmed as pmd
from rdkit import Chem
from rdkit.Chem import AllChem

from ..chemistry import RepeatUnit
from ..polymer import ROLE_PROP, TEMPLATE_ATOM_PROP, UNIT_INDEX_PROP, Chain
from ..utils import LOG, ExternalToolError, run_command
from .base import ForceFieldError, MoleculeSpec, TypedSystem
from .mol2 import Mol2Atoms, read_mol2, write_mol2

#: Charge-transfer residual above which the run is aborted rather than corrected.
#: A few hundredths of an electron is normal fragment/polymer mismatch; anything
#: larger means the fragment is not a good model of the in-chain environment.
MAX_CHARGE_RESIDUAL = 0.5


@dataclass(frozen=True)
class FragmentTyping:
    """GAFF2 types and AM1-BCC charges for a capped repeat-unit fragment."""

    #: Maps template atom index -> (gaff2 type, charge).
    by_template_atom: dict[int, tuple[str, float]]
    #: (type, charge) for the carbon of a capping methyl and for its hydrogens.
    cap_carbon: tuple[str, float]
    cap_hydrogen: tuple[str, float]
    frcmod: Path
    mol2: Path
    fragment_charge: float
    unit_formal_charge: int


def _embed_for_qm(mol: Chem.Mol, seed: int = 0xC0FFEE) -> Chem.Mol:
    """Produce a clean minimised conformer for the semi-empirical charge step."""
    mol = Chem.AddHs(Chem.Mol(mol), addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed) != 0:
            raise ForceFieldError("could not embed the model fragment for charge derivation")
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:  # pragma: no cover - UFF fallback
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def _capped_fragment(unit: RepeatUnit) -> tuple[Chem.Mol, dict[int, int], tuple[int, int]]:
    """Build the methyl-capped model fragment for one repeat unit.

    Returns ``(fragment, template_to_fragment, (cap_head, cap_tail))`` where the
    map sends repeat-unit template indices to fragment atom indices, and the two
    cap indices are the fragment carbons standing in for the neighbouring units.
    """
    rw = Chem.RWMol(Chem.Mol(unit.mol))
    caps: list[int] = []
    for attach in (unit.head_index, unit.tail_index):
        h_idx = None
        for nbr in rw.GetAtomWithIdx(attach).GetNeighbors():
            if nbr.GetAtomicNum() == 1 and nbr.GetIdx() not in caps:
                h_idx = nbr.GetIdx()
                break
        if h_idx is None:
            raise ForceFieldError(
                f"attachment atom {attach} has no hydrogen to replace with a capping methyl"
            )
        atom = rw.GetAtomWithIdx(h_idx)
        atom.SetAtomicNum(6)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        caps.append(h_idx)

    # Template indices survive this edit because only element identity changed.
    template_to_fragment = {i: i for i in range(rw.GetNumAtoms())}
    frag = rw.GetMol()
    Chem.SanitizeMol(frag)
    n_before = frag.GetNumAtoms()
    frag = Chem.AddHs(frag, addCoords=False, onlyOnAtoms=caps)
    Chem.SanitizeMol(frag)
    if frag.GetNumAtoms() != n_before + 6:
        raise ForceFieldError(
            f"expected 6 methyl hydrogens on the two caps, got {frag.GetNumAtoms() - n_before}"
        )
    return frag, template_to_fragment, (caps[0], caps[1])


def _run_antechamber(
    mol: Chem.Mol,
    workdir: Path,
    stem: str,
    net_charge: int,
    charge_method: str = "bcc",
) -> tuple[Path, Path]:
    """Run antechamber + parmchk2, returning ``(mol2, frcmod)``."""
    workdir.mkdir(parents=True, exist_ok=True)
    sdf = workdir / f"{stem}.mol"
    Chem.MolToMolFile(mol, str(sdf))
    mol2 = workdir / f"{stem}.mol2"
    frcmod = workdir / f"{stem}.frcmod"

    run_command(
        [
            "antechamber",
            "-i", sdf.name, "-fi", "mdl",
            "-o", mol2.name, "-fo", "mol2",
            "-c", charge_method, "-nc", str(net_charge),
            "-at", "gaff2", "-s", "0", "-pf", "y",
        ],
        cwd=workdir,
        hint=(
            "antechamber failed. The usual causes are a fragment sqm cannot converge "
            "(try charge_method='gas' to fall back on Gasteiger charges) and an "
            "inconsistent -nc net charge."
        ),
    )
    if not mol2.exists():
        raise ExternalToolError("antechamber produced no mol2 output", tail="", hint="check sqm.out")

    run_command(
        ["parmchk2", "-i", mol2.name, "-f", "mol2", "-o", frcmod.name, "-s", "gaff2"],
        cwd=workdir,
        hint="parmchk2 failed to generate missing parameters",
    )
    attn = [ln for ln in frcmod.read_text().splitlines() if "ATTN" in ln]
    if attn:
        LOG.warning(
            "parmchk2 flagged %d parameter(s) as needing review in %s; the first is: %s",
            len(attn),
            frcmod.name,
            attn[0].strip(),
        )
    return mol2, frcmod


def type_fragment(unit: RepeatUnit, workdir: Path, charge_method: str = "bcc") -> FragmentTyping:
    """Derive GAFF2 types and AM1-BCC charges for a capped repeat-unit fragment."""
    frag, template_to_fragment, (cap_head, cap_tail) = _capped_fragment(unit)
    frag = _embed_for_qm(frag)
    net = Chem.GetFormalCharge(frag)
    if net != unit.formal_charge:
        raise ForceFieldError(
            f"capped fragment charge {net:+d} does not match the repeat-unit charge "
            f"{unit.formal_charge:+d}"
        )
    mol2, frcmod = _run_antechamber(frag, workdir, "unit", net, charge_method)
    assigned = read_mol2(mol2)
    if len(assigned) != frag.GetNumAtoms():
        raise ForceFieldError(
            f"antechamber returned {len(assigned)} atoms for a {frag.GetNumAtoms()}-atom fragment"
        )

    by_template = {
        t_idx: (assigned.types[f_idx], assigned.charges[f_idx])
        for t_idx, f_idx in template_to_fragment.items()
    }
    cap_h_indices = [
        nbr.GetIdx()
        for nbr in frag.GetAtomWithIdx(cap_head).GetNeighbors()
        if nbr.GetAtomicNum() == 1
    ]
    if not cap_h_indices:
        raise ForceFieldError("capping methyl has no hydrogens after AddHs")
    cap_h_charge = float(np.mean([assigned.charges[i] for i in cap_h_indices]))
    return FragmentTyping(
        by_template_atom=by_template,
        cap_carbon=(assigned.types[cap_head], assigned.charges[cap_head]),
        cap_hydrogen=(assigned.types[cap_h_indices[0]], cap_h_charge),
        frcmod=frcmod,
        mol2=mol2,
        fragment_charge=assigned.total_charge,
        unit_formal_charge=unit.formal_charge,
    )


def _transfer_charges(
    chain: Chain,
    typing: FragmentTyping,
) -> tuple[list[str], list[float]]:
    """Map fragment types/charges onto a full chain and enforce integer charge.

    Chain atoms fall into two classes. Atoms carrying a template index are copies
    of a fragment atom and inherit its type and charge directly. Terminal cap
    atoms have no template index; they inherit the fragment's capping-methyl
    parameters, which is exactly the environment they represent.

    The sum then misses the integer target, because the fragment charge was
    distributed over one unit plus two capping methyls while a real chain shares
    those junctions between neighbouring units. The residual is spread evenly over
    the chain's carbon and hydrogen atoms -- never over the charged group, whose
    local charges carry the electrostatics that drive water uptake and must not be
    diluted.
    """
    types: list[str | None] = [None] * chain.n_atoms
    charges: list[float] = [0.0] * chain.n_atoms

    cap_c_type, cap_c_charge = typing.cap_carbon
    cap_h_type, cap_h_charge = typing.cap_hydrogen

    for atom in chain.mol.GetAtoms():
        i = atom.GetIdx()
        if atom.HasProp(TEMPLATE_ATOM_PROP):
            t_idx = atom.GetIntProp(TEMPLATE_ATOM_PROP)
            try:
                types[i], charges[i] = typing.by_template_atom[t_idx]
            except KeyError as exc:
                raise ForceFieldError(
                    f"chain atom {i} references template atom {t_idx}, which the fragment "
                    "typing does not contain"
                ) from exc
        elif atom.HasProp(ROLE_PROP) and atom.GetProp(ROLE_PROP) == "cap":
            types[i], charges[i] = cap_c_type, cap_c_charge
        elif atom.GetAtomicNum() == 1:
            # A hydrogen added onto a terminal cap methyl.
            heavy = atom.GetNeighbors()[0] if atom.GetDegree() else None
            is_cap_h = (
                heavy is not None
                and heavy.HasProp(ROLE_PROP)
                and heavy.GetProp(ROLE_PROP) == "cap"
            )
            if not is_cap_h:
                raise ForceFieldError(
                    f"hydrogen {i} is neither a template atom nor attached to a terminal cap"
                )
            types[i], charges[i] = cap_h_type, cap_h_charge
        else:
            raise ForceFieldError(
                f"chain atom {i} ({atom.GetSymbol()}) carries no template index and is not a cap"
            )

    if any(t is None for t in types):
        raise ForceFieldError("internal error: some chain atoms were left untyped")

    target = float(chain.formal_charge)
    residual = target - sum(charges)
    if abs(residual) > MAX_CHARGE_RESIDUAL * max(1, chain.chain_length):
        raise ForceFieldError(
            f"charge transfer residual {residual:+.4f} e is too large to redistribute; "
            "the capped fragment is a poor model of the in-chain environment"
        )
    # Neutral-backbone atoms only: skip anything whose transferred charge is large,
    # which is how the ionic head group is identified without extra chemistry.
    adjustable = [
        i
        for i, atom in enumerate(chain.mol.GetAtoms())
        if abs(charges[i]) < 0.35 and atom.GetAtomicNum() in (1, 6)
    ]
    if not adjustable:  # pragma: no cover - degenerate chemistry
        adjustable = list(range(chain.n_atoms))
    per_atom = residual / len(adjustable)
    for i in adjustable:
        charges[i] += per_atom

    final = sum(charges)
    if abs(final - target) > 1.0e-6:
        # Put the floating-point remainder on one atom so the total is exact.
        charges[adjustable[0]] += target - final
    LOG.info(
        "charge transfer: %d atoms typed, residual %+.4f e spread over %d backbone atoms "
        "(%+.2e e each), net charge %+.6f e",
        chain.n_atoms,
        residual,
        len(adjustable),
        per_atom,
        sum(charges),
    )
    return [str(t) for t in types], charges


def _tleap_parameterise(
    mol2_files: Sequence[Path],
    frcmods: Sequence[Path],
    workdir: Path,
    stem: str,
) -> pmd.Structure:
    """Run tleap on typed mol2 files and load the resulting Amber topology."""
    unit_names = [f"m{i}" for i in range(len(mol2_files))]
    # tleap resolves bare filenames against its own search path, not the cwd, and
    # the fragment frcmod lives in a subdirectory, so every path is made absolute.
    lines = ["source leaprc.gaff2"]
    for frcmod in dict.fromkeys(Path(f).resolve() for f in frcmods):
        lines.append(f"loadamberparams {frcmod}")
    for name, mol2 in zip(unit_names, mol2_files):
        lines.append(f"{name} = loadmol2 {Path(mol2).resolve()}")
    combined = unit_names[0] if len(unit_names) == 1 else "sys"
    if len(unit_names) > 1:
        lines.append(f"sys = combine {{ {' '.join(unit_names)} }}")
    lines.append(f"saveamberparm {combined} {stem}.prmtop {stem}.inpcrd")
    lines.append("quit")
    leap_in = workdir / f"{stem}.leap.in"
    leap_in.write_text("\n".join(lines) + "\n")

    log = run_command(
        ["tleap", "-f", leap_in.name],
        cwd=workdir,
        log_path=workdir / f"{stem}.leap.log",
        hint="tleap failed to assign bonded parameters from the GAFF2 atom types",
    )
    fatal = [ln for ln in log.splitlines() if "FATAL" in ln or "Could not find" in ln]
    if fatal:
        raise ForceFieldError(
            "tleap reported missing parameters:\n  " + "\n  ".join(fatal[:8])
        )
    prmtop = workdir / f"{stem}.prmtop"
    inpcrd = workdir / f"{stem}.inpcrd"
    if not prmtop.exists():
        raise ForceFieldError(f"tleap produced no topology; log tail:\n{log[-1500:]}")
    return pmd.load_file(str(prmtop), str(inpcrd))


class GAFF2Backend:
    """Type a membrane system with GAFF2 + AM1-BCC via AmberTools.

    The polymer is handled by fragment transfer (see the module docstring); small
    molecules -- water, monatomic counterions, molecular counterions such as
    bicarbonate -- are typed directly. Water and tabulated ions bypass
    antechamber entirely and use the literature parameters in
    :mod:`aemwater.forcefield.water`, because a semi-empirical charge model would
    otherwise silently replace the published, validated values that make the
    water model what it is.
    """

    name = "gaff2"

    def __init__(self, charge_method: str = "bcc", keep_files: bool = True):
        self.charge_method = charge_method
        self.keep_files = keep_files
        self._fragment_cache: dict[str, FragmentTyping] = {}

    # ------------------------------------------------------------- polymer ---
    def type_chain(
        self,
        chain: Chain,
        workdir: Path,
        typing: FragmentTyping | None = None,
    ) -> tuple[pmd.Structure, FragmentTyping]:
        """Return a parameterised ParmEd structure for one chain."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if typing is None:
            key = Chem.MolToSmiles(chain.repeat_unit.mol)
            if key not in self._fragment_cache:
                LOG.info("deriving GAFF2/AM1-BCC parameters for the repeat-unit fragment")
                self._fragment_cache[key] = type_fragment(
                    chain.repeat_unit, workdir / "fragment", self.charge_method
                )
            typing = self._fragment_cache[key]

        types, charges = _transfer_charges(chain, typing)
        chain_mol2 = write_mol2(
            chain.mol, types, charges, workdir / "chain.mol2", resname="POL", molname="POL"
        )
        struct = _tleap_parameterise([chain_mol2], [typing.frcmod], workdir, "chain")
        if len(struct.atoms) != chain.n_atoms:
            raise ForceFieldError(
                f"tleap returned {len(struct.atoms)} atoms for a {chain.n_atoms}-atom chain"
            )
        struct.coordinates = chain.coordinates()
        return struct, typing

    # -------------------------------------------------------- small molecule -
    def type_small_molecule(
        self,
        mol: Chem.Mol,
        workdir: Path,
        stem: str,
        resname: str = "MOL",
    ) -> pmd.Structure:
        """Type an arbitrary small molecule with antechamber + tleap."""
        workdir = Path(workdir)
        net = Chem.GetFormalCharge(mol)
        mol3d = mol if mol.GetNumConformers() else _embed_for_qm(mol)
        mol2, frcmod = _run_antechamber(mol3d, workdir, stem, net, self.charge_method)
        assigned = read_mol2(mol2)
        retyped = write_mol2(
            mol3d,
            assigned.types,
            assigned.charges,
            workdir / f"{stem}.typed.mol2",
            resname=resname,
            molname=resname,
        )
        return _tleap_parameterise([retyped], [frcmod], workdir, stem)

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "charge_method": self.charge_method,
            "polymer_charges": "AM1-BCC on a methyl-capped repeat unit, transferred by template index",
            "bonded_parameters": "gaff2.dat via tleap, gaps filled by parmchk2",
            "water_and_simple_ions": "literature parameters, not antechamber",
        }


__all__ = [
    "GAFF2Backend",
    "FragmentTyping",
    "type_fragment",
    "MAX_CHARGE_RESIDUAL",
]
