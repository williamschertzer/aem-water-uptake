"""Rigid three-site water models and monatomic-ion Lennard-Jones parameters.

Values are quoted from the original publications rather than re-derived:

SPC/E
    H. J. C. Berendsen, J. R. Grigera, T. P. Straatsma,
    *J. Phys. Chem.* **91**, 6269 (1987).
    sigma_OO = 3.166 A, eps_OO = 0.1553 kcal/mol, r_OH = 1.0 A,
    angle HOH = 109.47 deg, q_O = -0.8476 e, q_H = +0.4238 e.

TIP3P
    W. L. Jorgensen et al., *J. Chem. Phys.* **79**, 926 (1983).
    sigma_OO = 3.15061 A, eps_OO = 0.1521 kcal/mol, r_OH = 0.9572 A,
    angle HOH = 104.52 deg, q_O = -0.834 e, q_H = +0.417 e.

Halide and hydroxide ion parameters
    Joung & Cheatham, *J. Phys. Chem. B* **112**, 9020 (2008) -- SPC/E-consistent
    monovalent ion set (used for Cl- and Br-).
    Hydroxide: Y. Wu, H. Chen, F. Wang, F. Paesani, G. A. Voth,
    *J. Phys. Chem. B* **112**, 467 (2008) -- rigid OH- LJ/charge set.

Reference bulk properties used for validation targets are also recorded here so
:mod:`aemwater.widom` can report a measured value next to the literature one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WaterModel:
    """A rigid three-site water model."""

    name: str
    #: LJ on oxygen only (hydrogens are LJ-free in all three-site models).
    sigma_O: float
    epsilon_O: float
    charge_O: float
    charge_H: float
    r_OH: float
    angle_HOH: float
    mass_O: float = 15.9994
    mass_H: float = 1.008
    citation: str = ""
    #: Experimental/simulated reference values at 298 K, 1 atm.
    density_ref: float = 0.997          # g/cm^3
    #: Excess chemical potential of the pure liquid, kcal/mol. Simulation values
    #: for SPC/E cluster near -6.3 kcal/mol; used only as a sanity check on the
    #: bulk Widom reference, never as a substitute for it.
    mu_ex_ref: float = float("nan")
    mu_ex_ref_note: str = ""

    @property
    def molar_mass(self) -> float:
        return self.mass_O + 2 * self.mass_H

    @property
    def net_charge(self) -> float:
        return self.charge_O + 2 * self.charge_H

    def validate(self) -> None:
        if abs(self.net_charge) > 1e-6:
            raise ValueError(f"water model {self.name} is not neutral: {self.net_charge:+.6f} e")


SPCE = WaterModel(
    name="spce",
    sigma_O=3.166,
    epsilon_O=0.1553,
    charge_O=-0.8476,
    charge_H=0.4238,
    r_OH=1.0,
    angle_HOH=109.47,
    citation="Berendsen, Grigera & Straatsma, J. Phys. Chem. 91, 6269 (1987)",
    density_ref=0.998,
    mu_ex_ref=-6.3,
    mu_ex_ref_note="SPC/E liquid at 298 K; literature Widom/BAR values span -6.1 to -6.6 kcal/mol",
)

TIP3P = WaterModel(
    name="tip3p",
    sigma_O=3.15061,
    epsilon_O=0.1521,
    charge_O=-0.834,
    charge_H=0.417,
    r_OH=0.9572,
    angle_HOH=104.52,
    citation="Jorgensen et al., J. Chem. Phys. 79, 926 (1983)",
    density_ref=0.986,
    mu_ex_ref=-6.1,
    mu_ex_ref_note="TIP3P liquid at 298 K; literature values near -6.1 kcal/mol",
)

WATER_MODELS: dict[str, WaterModel] = {m.name: m for m in (SPCE, TIP3P)}


def water_model(name: str) -> WaterModel:
    try:
        model = WATER_MODELS[name.lower()]
    except KeyError as exc:
        raise KeyError(f"unknown water model {name!r}; choose from {sorted(WATER_MODELS)}") from exc
    model.validate()
    return model


@dataclass(frozen=True)
class IonParameters:
    """Lennard-Jones and charge parameters for a small mobile ion."""

    label: str
    #: Per-site (element, sigma A, epsilon kcal/mol, charge e, mass amu).
    sites: tuple[tuple[str, float, float, float, float], ...]
    citation: str = ""
    #: Internal geometry for polyatomic ions: (i, j, r_ij) in Angstrom.
    bonds: tuple[tuple[int, int, float], ...] = field(default_factory=tuple)

    @property
    def net_charge(self) -> float:
        return sum(site[3] for site in self.sites)


#: Joung-Cheatham SPC/E-consistent halides; hydroxide from Wu et al.
ION_PARAMETERS: dict[str, IonParameters] = {
    "Cl-": IonParameters(
        label="Cl-",
        sites=(("Cl", 4.83045, 0.0127850, -1.0, 35.453),),
        citation="Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), SPC/E set",
    ),
    "Br-": IonParameters(
        label="Br-",
        sites=(("Br", 4.90890, 0.0269586, -1.0, 79.904),),
        citation="Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), SPC/E set",
    ),
    "OH-": IonParameters(
        label="OH-",
        sites=(
            ("O", 3.166, 0.1553, -1.32, 15.9994),
            ("H", 0.0, 0.0, 0.32, 1.008),
        ),
        bonds=((0, 1, 0.98),),
        citation="Wu, Chen, Wang, Paesani & Voth, J. Phys. Chem. B 112, 467 (2008)",
    ),
}


def ion_parameters(label: str) -> IonParameters:
    """Look up ion parameters, raising a clear error for unparameterised ions.

    ``HCO3-`` is intentionally absent: bicarbonate needs a full GAFF2 treatment
    (it is molecular, not a simple ion), which the GAFF2 backend performs by
    running antechamber on it like any other molecule.
    """
    try:
        params = ION_PARAMETERS[label]
    except KeyError as exc:
        raise KeyError(
            f"no tabulated LJ parameters for {label!r}. Monatomic ions available: "
            f"{sorted(ION_PARAMETERS)}. Molecular ions (e.g. HCO3-) are typed by "
            "antechamber/GAFF2 instead."
        ) from exc
    if abs(params.net_charge - round(params.net_charge)) > 1e-6:
        raise ValueError(f"ion {label} has non-integer charge {params.net_charge}")
    return params


__all__ = [
    "WaterModel",
    "WATER_MODELS",
    "SPCE",
    "TIP3P",
    "water_model",
    "IonParameters",
    "ION_PARAMETERS",
    "ion_parameters",
]
