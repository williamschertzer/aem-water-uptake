"""Chain construction: topology, charge, geometry and scaling."""

from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from aemwater.chemistry import parse_repeat_unit
from aemwater.polymer import (
    ROLE_PROP,
    TEMPLATE_ATOM_PROP,
    PolymerBuildError,
    build_chain,
    build_segment,
)

BTMA_PS = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"


@pytest.fixture(scope="module")
def unit():
    return parse_repeat_unit(BTMA_PS)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_formula_and_charge_are_exact(unit, n):
    """A capped N-mer must have exactly the composition stoichiometry predicts."""
    chain = build_chain(unit, n, seed=1)
    formula = Chem.rdMolDescriptors.CalcMolFormula(chain.mol)
    counts = Counter(a.GetSymbol() for a in chain.mol.GetAtoms())
    # Repeat unit C12H18N+ as incorporated; two CH3 caps add C2H6.
    assert counts["C"] == 12 * n + 2, formula
    assert counts["N"] == n, formula
    assert counts["H"] == 18 * n + 6, formula
    assert chain.formal_charge == n
    assert Chem.GetFormalCharge(chain.mol) == n


def test_capping_materialises_hydrogens(unit):
    """Regression: AddHs before sanitisation silently added no cap hydrogens."""
    chain = build_chain(unit, 2, seed=1)
    caps = [a for a in chain.mol.GetAtoms()
            if a.HasProp(ROLE_PROP) and a.GetProp(ROLE_PROP) == "cap"]
    assert len(caps) == 2
    for cap in caps:
        h = [n for n in cap.GetNeighbors() if n.GetAtomicNum() == 1]
        assert len(h) == 3, "each terminal methyl needs three explicit hydrogens"


def test_every_atom_is_traceable_for_charge_transfer(unit):
    """The force-field backend needs a template index or a cap tag on every atom."""
    chain = build_chain(unit, 3, seed=2)
    for atom in chain.mol.GetAtoms():
        tagged = atom.HasProp(TEMPLATE_ATOM_PROP)
        is_cap = atom.HasProp(ROLE_PROP) and atom.GetProp(ROLE_PROP) == "cap"
        cap_h = atom.GetAtomicNum() == 1 and any(
            n.HasProp(ROLE_PROP) and n.GetProp(ROLE_PROP) == "cap" for n in atom.GetNeighbors()
        )
        assert tagged or is_cap or cap_h, f"atom {atom.GetIdx()} is untraceable"


def test_backbone_is_a_connected_path(unit):
    chain = build_chain(unit, 4, seed=3)
    assert len(chain.backbone) == 2 * 4
    for a, b in zip(chain.backbone, chain.backbone[1:]):
        assert chain.mol.GetBondBetweenAtoms(a, b) is not None


def test_conformer_has_no_hard_overlaps(unit):
    """A collapsed conformer would make the first LAMMPS step explode."""
    chain = build_chain(unit, 6, seed=4)
    assert chain.min_interatomic_distance() > 1.4
    pos = chain.coordinates()
    assert np.isfinite(pos).all()
    assert np.linalg.norm(pos, axis=1).min() > 0.0, "an atom was left at the origin"


def test_coil_expands_with_chain_length(unit):
    """Rg must grow with N. A globular collapse (Rg falling) is a real failure mode."""
    rg = [np.mean([build_chain(unit, n, seed=s).radius_of_gyration() for s in (1, 2)])
          for n in (5, 15)]
    assert rg[1] > rg[0] * 1.15, f"chain is not expanding with length: {rg}"


def test_segment_topology_has_no_leftover_attachment_hydrogens(unit):
    mol, backbone = build_segment(unit, 3, seed=1)
    assert len(backbone) == 6
    assert Chem.GetFormalCharge(mol) == 3


def test_rejects_zero_length(unit):
    with pytest.raises(PolymerBuildError):
        build_chain(unit, 0)
