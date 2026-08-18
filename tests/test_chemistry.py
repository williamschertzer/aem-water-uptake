"""Repeat-unit parsing, charge bookkeeping and counterion balance."""

import math

import pytest

from aemwater.chemistry import (
    COUNTERIONS,
    ChemistryError,
    build_composition,
    describe_smiles,
    parse_repeat_unit,
)

BTMA_PS = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"
VINYL_BTMA = "C=Cc1ccc(C[N+](C)(C)C)cc1"
# Poly(phenylene oxide) with a quaternary ammonium side chain.
QA_PPO = "[*]Oc1cc(C)c([*])c(C[N+](C)(C)C)c1"
NEUTRAL_PS = "[*]CC([*])c1ccccc1"
# Two charges per repeat unit.
DIQA = "[*]CC([*])c1cc(C[N+](C)(C)C)cc(C[N+](C)(C)C)c1"


class TestParsing:
    def test_dummy_and_vinyl_dialects_agree(self):
        a = parse_repeat_unit(BTMA_PS)
        b = parse_repeat_unit(VINYL_BTMA)
        assert a.source_dialect == "dummy-atoms"
        assert b.source_dialect == "vinyl"
        assert a.n_atoms == b.n_atoms
        assert a.formal_charge == b.formal_charge == 1
        assert math.isclose(a.molar_mass, b.molar_mass, rel_tol=1e-9)

    def test_incorporated_mass_excludes_capping_hydrogens(self):
        """Styrene repeat unit is C8H8 = 104.15, not C8H10 = 106.17."""
        unit = parse_repeat_unit(NEUTRAL_PS)
        assert math.isclose(unit.molar_mass, 104.15, abs_tol=0.02)
        assert math.isclose(unit.fragment_molar_mass, 106.17, abs_tol=0.02)

    def test_attachment_atoms_are_distinct_backbone_carbons(self):
        unit = parse_repeat_unit(BTMA_PS)
        assert unit.head_index != unit.tail_index
        for idx in (unit.head_index, unit.tail_index):
            assert unit.mol.GetAtomWithIdx(idx).GetSymbol() == "C"

    def test_single_attachment_point_rejected(self):
        with pytest.raises(ChemistryError, match="single attachment point"):
            parse_repeat_unit("[*]CCc1ccccc1")

    def test_three_attachment_points_rejected(self):
        with pytest.raises(ChemistryError, match="exactly 2"):
            parse_repeat_unit("[*]CC([*])c1ccc([*])cc1")

    def test_no_attachment_and_no_vinyl_rejected(self):
        with pytest.raises(ChemistryError, match="neither attachment points nor"):
            parse_repeat_unit("c1ccccc1")

    def test_unparseable_smiles_rejected(self):
        with pytest.raises(ChemistryError, match="could not parse"):
            parse_repeat_unit("[*]CC([*])c1ccccc")

    def test_ionic_site_count(self):
        assert parse_repeat_unit(BTMA_PS).ionic_group_count() == 1
        assert parse_repeat_unit(DIQA).ionic_group_count() == 2
        assert parse_repeat_unit(NEUTRAL_PS).ionic_group_count() == 0

    def test_describe_smiles_reports_formula(self):
        info = describe_smiles(BTMA_PS)
        assert info["formal_charge"] == 1
        assert "N" in str(info["formula"])
        assert info["heavy_atoms"] == 13


class TestComposition:
    @pytest.mark.parametrize("smiles", [BTMA_PS, VINYL_BTMA, QA_PPO])
    def test_neutrality_for_monocationic_units(self, smiles):
        comp = build_composition(smiles, n_chains=4, chain_length=10)
        assert comp.net_charge == 0
        # one +1 site per unit -> one Cl- per unit
        assert comp.n_counterions == 4 * 10
        assert comp.total_ionic_groups == 4 * 10

    def test_neutrality_for_dicationic_unit(self):
        comp = build_composition(DIQA, n_chains=3, chain_length=6)
        assert comp.net_charge == 0
        assert comp.n_counterions == 2 * 3 * 6
        assert comp.total_ionic_groups == 2 * 3 * 6

    def test_neutral_polymer_gets_no_counterions(self):
        comp = build_composition(NEUTRAL_PS, n_chains=2, chain_length=8)
        assert comp.n_counterions == 0
        assert comp.total_ionic_groups == 0
        assert math.isnan(comp.lambda_from_n_water(10))

    @pytest.mark.parametrize("ion", sorted(COUNTERIONS))
    def test_every_counterion_balances(self, ion):
        comp = build_composition(BTMA_PS, n_chains=2, chain_length=4, counterion=ion)
        assert comp.net_charge == 0
        assert comp.n_counterions == 8

    def test_unsupported_counterion_rejected(self):
        with pytest.raises(ChemistryError, match="unsupported counterion"):
            build_composition(BTMA_PS, 2, 4, counterion="F-")

    def test_cationic_counterion_requirement_rejected(self):
        """A polyanion would need cations, which are not parameterised."""
        with pytest.raises(ChemistryError, match="ANION exchange|cations"):
            build_composition("[*]CC([*])c1ccc(S(=O)(=O)[O-])cc1", 2, 4)

    def test_chain_mass_and_iec_consistency(self):
        comp = build_composition(BTMA_PS, n_chains=4, chain_length=10)
        expected_chain = comp.repeat_unit.molar_mass * 10 + 2 * 15.0345
        assert math.isclose(comp.chain_molar_mass, expected_chain, rel_tol=1e-9)
        expected_dry = expected_chain * 4 + 35.453 * 40
        assert math.isclose(comp.dry_molar_mass, expected_dry, rel_tol=1e-9)
        # IEC = mmol charge / g dry
        assert math.isclose(comp.ion_exchange_capacity, 1000 * 40 / expected_dry, rel_tol=1e-9)
        # BTMA-PS in Cl- form: literature IEC is ~3-4 meq/g for full functionalisation
        assert 2.5 < comp.ion_exchange_capacity < 5.0

    def test_lambda_and_uptake_percent(self):
        comp = build_composition(BTMA_PS, n_chains=2, chain_length=10)
        assert math.isclose(comp.lambda_from_n_water(60), 3.0)
        uptake = comp.water_uptake_percent(60)
        assert math.isclose(uptake, 100 * 60 * 18.01528 / comp.dry_molar_mass, rel_tol=1e-9)
        assert 0 < uptake < 100

    def test_terminal_group_changes_mass(self):
        ch3 = build_composition(BTMA_PS, 1, 5, terminal_group="CH3")
        h = build_composition(BTMA_PS, 1, 5, terminal_group="H")
        assert ch3.chain_molar_mass > h.chain_molar_mass
