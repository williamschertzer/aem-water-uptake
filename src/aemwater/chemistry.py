"""Repeat-unit interpretation: attachment points, charge and counterion count.

The user supplies a repeat-unit SMILES. Two dialects are accepted:

1. **Explicit attachment points** -- two dummy atoms (``[*]``, ``*``, or
   ``[At]``) mark where the backbone continues, e.g.::

       [*]CC([*])c1ccc(C[N+](C)(C)C)cc1     # benzyltrimethylammonium polystyrene

2. **Vinyl monomer** -- a terminal ``C=C`` is opened automatically, so the
   monomer as drawn for synthesis also works::

       C=Cc1ccc(C[N+](C)(C)C)cc1

The module never builds a chain (that is :mod:`aemwater.polymer`); it only
resolves *what* is to be built and the ion bookkeeping that follows from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .utils import LOG

HYDROGEN_MASS = 1.008

#: Atomic numbers treated as attachment-point placeholders.
DUMMY_ATOMIC_NUMS = (0, 85)  # 0 = '*', 85 = astatine, a common polymer convention


class ChemistryError(ValueError):
    """Raised when a repeat unit cannot be interpreted."""


@dataclass(frozen=True)
class RepeatUnit:
    """A repeat unit resolved to a backbone-connectable fragment.

    ``mol`` has explicit hydrogens and *no* dummy atoms: the attachment points
    are recorded as atom indices with one open valence each, which
    :mod:`aemwater.polymer` consumes to stitch chains together.
    """

    mol: Chem.Mol
    head_index: int
    tail_index: int
    formal_charge: int
    smiles_input: str
    source_dialect: str  # "dummy-atoms" | "vinyl"

    @property
    def n_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def fragment_molar_mass(self) -> float:
        """Mass of the isolated, hydrogen-saturated fragment, g/mol."""
        return Descriptors.MolWt(self.mol)

    @property
    def molar_mass(self) -> float:
        """Mass of the repeat unit *as incorporated in the chain*, g/mol.

        ``mol`` carries one extra hydrogen on each of the head and tail atoms
        (RDKit saturates the open valences left by the attachment points). Those
        two hydrogens are displaced when the unit is bonded to its neighbours,
        so they must be removed here or every derived quantity -- chain mass,
        dry mass, IEC, mass-based uptake -- is systematically too large.
        """
        return Descriptors.MolWt(self.mol) - 2 * HYDROGEN_MASS

    def ionic_group_count(self) -> int:
        """Number of formally charged sites in the repeat unit."""
        return sum(1 for a in self.mol.GetAtoms() if a.GetFormalCharge() != 0)


@dataclass(frozen=True)
class Counterion:
    """A monatomic or small polyatomic mobile ion."""

    label: str
    smiles: str
    charge: int
    mass: float
    residue_name: str
    #: True when the ion is a single atom (monatomic ion LJ parameter sets apply).
    monatomic: bool

    def to_mol(self) -> Chem.Mol:
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:  # pragma: no cover - table is static
            raise ChemistryError(f"internal: bad counterion SMILES {self.smiles!r}")
        return Chem.AddHs(mol)


#: Supported counterions. Parameters live in :mod:`aemwater.forcefield.ions`.
COUNTERIONS: dict[str, Counterion] = {
    "Cl-": Counterion("Cl-", "[Cl-]", -1, 35.453, "CL", True),
    "Br-": Counterion("Br-", "[Br-]", -1, 79.904, "BR", True),
    "OH-": Counterion("OH-", "[OH-]", -1, 17.007, "OHX", False),
    "HCO3-": Counterion("HCO3-", "OC(=O)[O-]", -1, 61.017, "BCX", False),
}


def _find_dummies(mol: Chem.Mol) -> list[int]:
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in DUMMY_ATOMIC_NUMS]


def _parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ChemistryError(
            f"RDKit could not parse the repeat-unit SMILES {smiles!r}. "
            "Check valences, ring closures and bracketed charges "
            "(a quaternary nitrogen must be written [N+])."
        )
    return mol


def _resolve_dummy_dialect(mol: Chem.Mol, dummies: list[int]) -> tuple[Chem.Mol, int, int]:
    """Strip two dummy atoms, returning (mol_with_Hs, head_idx, tail_idx)."""
    if len(dummies) > 2:
        raise ChemistryError(
            f"repeat unit has {len(dummies)} attachment points; exactly 2 are required for a "
            "linear chain. Branched/crosslinked architectures are not supported yet."
        )
    anchors = []
    for d in dummies:
        nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(d).GetNeighbors()]
        if len(nbrs) != 1:
            raise ChemistryError(
                f"attachment point (atom {d}) must be bonded to exactly one heavy atom, found {len(nbrs)}"
            )
        anchors.append(nbrs[0])
    if anchors[0] == anchors[1]:
        LOG.warning(
            "both attachment points are on the same atom (index %d); the backbone will branch "
            "twice from one centre",
            anchors[0],
        )

    rw = Chem.RWMol(mol)
    for d in sorted(dummies, reverse=True):
        rw.RemoveAtom(d)
    stripped = rw.GetMol()
    # RemoveAtom shifts indices above the removed position.
    shifted = []
    for a in anchors:
        shift = sum(1 for d in dummies if d < a)
        shifted.append(a - shift)
    Chem.SanitizeMol(stripped)
    with_h = Chem.AddHs(stripped)
    return with_h, shifted[0], shifted[1]


def _resolve_vinyl_dialect(mol: Chem.Mol) -> tuple[Chem.Mol, int, int]:
    """Open a terminal C=C so a vinyl monomer becomes a repeat unit."""
    patt = Chem.MolFromSmarts("[CX3;H2]=[CX3]")
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        raise ChemistryError(
            "repeat unit contains neither attachment points nor a terminal vinyl group. "
            "Mark the two backbone connection points with dummy atoms, e.g. "
            "'[*]CC([*])c1ccccc1', or supply a vinyl monomer such as 'C=Cc1ccccc1'."
        )
    if len(matches) > 1:
        raise ChemistryError(
            f"found {len(matches)} terminal vinyl groups; the polymerisable bond is ambiguous. "
            "Use explicit [*] attachment points instead."
        )
    ch2, ch = matches[0]
    rw = Chem.RWMol(mol)
    rw.GetBondBetweenAtoms(ch2, ch).SetBondType(Chem.BondType.SINGLE)
    opened = rw.GetMol()
    Chem.SanitizeMol(opened)
    with_h = Chem.AddHs(opened)
    return with_h, ch2, ch


def parse_repeat_unit(smiles: str) -> RepeatUnit:
    """Interpret a repeat-unit SMILES into a :class:`RepeatUnit`.

    Raises
    ------
    ChemistryError
        If the SMILES cannot be parsed or the attachment points are ambiguous.
    """
    raw = _parse_smiles(smiles)
    dummies = _find_dummies(raw)
    if len(dummies) >= 2:
        mol, head, tail = _resolve_dummy_dialect(raw, dummies)
        dialect = "dummy-atoms"
    elif len(dummies) == 1:
        raise ChemistryError(
            "repeat unit has a single attachment point; a linear repeat unit needs two "
            "(head and tail), e.g. '[*]CC([*])c1ccccc1'."
        )
    else:
        mol, head, tail = _resolve_vinyl_dialect(raw)
        dialect = "vinyl"

    charge = Chem.GetFormalCharge(mol)
    unit = RepeatUnit(
        mol=mol,
        head_index=head,
        tail_index=tail,
        formal_charge=charge,
        smiles_input=smiles,
        source_dialect=dialect,
    )
    _warn_on_suspicious_unit(unit)
    return unit


def _warn_on_suspicious_unit(unit: RepeatUnit) -> None:
    if unit.formal_charge == 0:
        LOG.warning(
            "repeat unit is formally neutral: no counterions will be added and the result is a "
            "water-uptake calculation for a NEUTRAL polymer, not an anion exchange membrane. "
            "A quaternary ammonium group must be written with an explicit charge, e.g. [N+](C)(C)C."
        )
    elif unit.formal_charge < 0:
        LOG.warning(
            "repeat unit carries negative formal charge %+d: this is a cation exchange "
            "membrane motif. Counterions will be cations, which are not parameterised here.",
            unit.formal_charge,
        )


@dataclass(frozen=True)
class SystemComposition:
    """Molecular inventory of the dry membrane cell, derived from a config."""

    repeat_unit: RepeatUnit
    n_chains: int
    chain_length: int
    counterion: Counterion
    n_counterions: int
    terminal_group: str
    #: Net charge of the whole cell; must be zero for a PPPM simulation.
    net_charge: int
    #: Formally charged sites per chain (the denominator of lambda).
    ionic_groups_per_chain: int

    # -------------------------------------------------------------- totals --
    @property
    def total_ionic_groups(self) -> int:
        return self.ionic_groups_per_chain * self.n_chains

    @property
    def chain_molar_mass(self) -> float:
        """Chain mass including terminal caps, g/mol."""
        cap = 15.0345 if self.terminal_group.upper() == "CH3" else HYDROGEN_MASS
        return self.repeat_unit.molar_mass * self.chain_length + 2 * cap

    @property
    def dry_molar_mass(self) -> float:
        """Total dry mass of the cell contents, g/mol."""
        return self.chain_molar_mass * self.n_chains + self.counterion.mass * self.n_counterions

    @property
    def ion_exchange_capacity(self) -> float:
        """IEC in mmol of fixed charge per gram of dry membrane (meq/g).

        This is the standard experimental descriptor of an AEM and lets the
        predicted lambda be converted to a mass-based uptake percentage.
        """
        if self.dry_molar_mass <= 0:  # pragma: no cover
            return float("nan")
        return 1000.0 * self.total_ionic_groups / self.dry_molar_mass

    def lambda_from_n_water(self, n_water: int) -> float:
        """Hydration number lambda = n_water / n_ionic_groups."""
        if self.total_ionic_groups == 0:
            return float("nan")
        return n_water / self.total_ionic_groups

    def water_uptake_percent(self, n_water: int) -> float:
        """Mass-based water uptake, % of dry mass."""
        from .utils import WATER_MASS_AMU

        return 100.0 * n_water * WATER_MASS_AMU / self.dry_molar_mass

    def summary(self) -> dict[str, object]:
        return {
            "repeat_unit_smiles": self.repeat_unit.smiles_input,
            "repeat_unit_dialect": self.repeat_unit.source_dialect,
            "repeat_unit_atoms": self.repeat_unit.n_atoms,
            "repeat_unit_mass": round(self.repeat_unit.molar_mass, 4),
            "repeat_unit_fragment_mass": round(self.repeat_unit.fragment_molar_mass, 4),
            "repeat_unit_charge": self.repeat_unit.formal_charge,
            "n_chains": self.n_chains,
            "chain_length": self.chain_length,
            "chain_mass": round(self.chain_molar_mass, 4),
            "ionic_groups_per_chain": self.ionic_groups_per_chain,
            "total_ionic_groups": self.total_ionic_groups,
            "counterion": self.counterion.label,
            "n_counterions": self.n_counterions,
            "dry_mass_g_per_mol": round(self.dry_molar_mass, 4),
            "IEC_meq_per_g": round(self.ion_exchange_capacity, 4),
            "net_charge": self.net_charge,
        }


def build_composition(
    smiles: str,
    n_chains: int,
    chain_length: int,
    counterion: str = "Cl-",
    terminal_group: str = "CH3",
) -> SystemComposition:
    """Resolve the full molecular inventory, enforcing charge neutrality.

    The number of counterions is fixed by neutrality: a chain carrying
    ``chain_length * q`` formal charge requires ``n_chains * chain_length * q``
    anions of charge -1. A non-integer requirement (e.g. a +2 repeat unit with a
    -3 anion) raises rather than silently rounding.
    """
    unit = parse_repeat_unit(smiles)
    if counterion not in COUNTERIONS:
        raise ChemistryError(f"unsupported counterion {counterion!r}; choose from {sorted(COUNTERIONS)}")
    ion = COUNTERIONS[counterion]

    charge_per_chain = unit.formal_charge * chain_length
    total_polymer_charge = charge_per_chain * n_chains

    if total_polymer_charge == 0:
        n_ions = 0
    else:
        if total_polymer_charge % abs(ion.charge) != 0:
            raise ChemistryError(
                f"cannot neutralise total polymer charge {total_polymer_charge:+d} with "
                f"{ion.label} (charge {ion.charge:+d}): the ratio is not an integer. "
                "Adjust chain_length or choose a different counterion."
            )
        n_ions = -total_polymer_charge // ion.charge
        if n_ions < 0:
            raise ChemistryError(
                f"repeat unit charge {unit.formal_charge:+d} would require "
                f"{-n_ions} cations; only anionic counterions are parameterised. "
                "This tool targets ANION exchange membranes (cationic polymer backbone)."
            )

    net = total_polymer_charge + n_ions * ion.charge
    if net != 0:
        raise ChemistryError(f"internal charge-balance error: net charge {net:+d}")

    comp = SystemComposition(
        repeat_unit=unit,
        n_chains=n_chains,
        chain_length=chain_length,
        counterion=ion,
        n_counterions=n_ions,
        terminal_group=terminal_group,
        net_charge=net,
        ionic_groups_per_chain=unit.ionic_group_count() * chain_length,
    )
    LOG.info(
        "composition: %d chains x %d units, %d ionic groups, %d %s, IEC = %.3f meq/g",
        comp.n_chains,
        comp.chain_length,
        comp.total_ionic_groups,
        comp.n_counterions,
        ion.label,
        comp.ion_exchange_capacity,
    )
    return comp


def composition_from_config(cfg) -> SystemComposition:
    """Build a :class:`SystemComposition` from a :class:`~aemwater.config.RunConfig`."""
    p = cfg.polymer
    return build_composition(
        smiles=p.smiles,
        n_chains=p.n_chains,
        chain_length=p.chain_length,
        counterion=p.counterion,
        terminal_group=p.terminal_group,
    )


def describe_smiles(smiles: str) -> dict[str, object]:
    """Human-readable diagnostic used by ``aemwater inspect``."""
    unit = parse_repeat_unit(smiles)
    return {
        "input": smiles,
        "dialect": unit.source_dialect,
        "canonical_repeat_unit": Chem.MolToSmiles(Chem.RemoveHs(unit.mol)),
        "atoms_with_H": unit.n_atoms,
        "heavy_atoms": unit.mol.GetNumHeavyAtoms(),
        "molar_mass": round(unit.molar_mass, 4),
        "formal_charge": unit.formal_charge,
        "ionic_sites": unit.ionic_group_count(),
        "formula": rdMolDescriptors.CalcMolFormula(unit.mol),
        "head_atom": unit.head_index,
        "tail_atom": unit.tail_index,
    }


__all__ = [
    "RepeatUnit",
    "Counterion",
    "COUNTERIONS",
    "SystemComposition",
    "ChemistryError",
    "parse_repeat_unit",
    "build_composition",
    "composition_from_config",
    "describe_smiles",
]
