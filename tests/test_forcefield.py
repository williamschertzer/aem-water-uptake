"""Force-field assignment: water/ion tables and GAFF2 fragment transfer."""

import math

import numpy as np
import pytest

from aemwater.forcefield.builders import ion_structure, water_structure
from aemwater.forcefield.water import (
    WATER_MODELS,
    ion_parameters,
    water_model,
)
from conftest import needs_ambertools


@pytest.mark.parametrize("name", sorted(WATER_MODELS))
def test_water_models_are_neutral_and_geometrically_correct(name):
    model = water_model(name)
    struct = water_structure(model)
    assert abs(sum(a.charge for a in struct.atoms)) < 1e-9
    coords = struct.coordinates
    r1 = np.linalg.norm(coords[1] - coords[0])
    r2 = np.linalg.norm(coords[2] - coords[0])
    assert r1 == pytest.approx(model.r_OH, abs=1e-6)
    assert r2 == pytest.approx(model.r_OH, abs=1e-6)
    cos = np.dot(coords[1] - coords[0], coords[2] - coords[0]) / (r1 * r2)
    assert math.degrees(math.acos(cos)) == pytest.approx(model.angle_HOH, abs=1e-4)


def test_water_lj_is_on_oxygen_only():
    struct = water_structure(water_model("spce"))
    assert struct.atoms[0].epsilon > 0
    assert struct.atoms[1].epsilon == 0.0
    assert struct.atoms[2].epsilon == 0.0


def test_spce_parameters_match_the_publication():
    """Guards against a silent edit to values taken from Berendsen et al. (1987)."""
    model = water_model("spce")
    assert model.sigma_O == pytest.approx(3.166)
    assert model.epsilon_O == pytest.approx(0.1553)
    assert model.charge_O == pytest.approx(-0.8476)


@pytest.mark.parametrize("label", ["Cl-", "Br-", "OH-"])
def test_ions_carry_integer_charge(label):
    struct = ion_structure(ion_parameters(label))
    assert sum(a.charge for a in struct.atoms) == pytest.approx(-1.0, abs=1e-6)
    assert np.isfinite(struct.coordinates).all()


def test_unparameterised_ion_raises_with_guidance():
    with pytest.raises(KeyError, match="HCO3"):
        ion_parameters("HCO3-")


@needs_ambertools
def test_gaff2_chain_typing_is_complete_and_neutral(tmp_path, btma_ps_smiles):
    """Full antechamber -> transfer -> tleap round trip on a short chain."""
    from aemwater.chemistry import parse_repeat_unit
    from aemwater.forcefield.gaff2 import GAFF2Backend
    from aemwater.polymer import build_chain

    unit = parse_repeat_unit(btma_ps_smiles)
    chain = build_chain(unit, 3, seed=5)
    struct, typing = GAFF2Backend().type_chain(chain, tmp_path)

    assert len(struct.atoms) == chain.n_atoms
    # Charge transfer must recover the integer chain charge exactly.
    assert sum(a.charge for a in struct.atoms) == pytest.approx(float(chain.formal_charge), abs=1e-4)
    # tleap must have found every bonded parameter.
    assert all(b.type is not None for b in struct.bonds)
    assert all(a.type is not None for a in struct.angles)
    assert all(d.type is not None for d in struct.dihedrals)
    assert all(a.epsilon is not None and a.sigma is not None for a in struct.atoms)
    # The quaternary nitrogen must be typed as such, not as a neutral amine.
    assert "n4" in {a.type for a in struct.atoms}
    assert typing.unit_formal_charge == 1
