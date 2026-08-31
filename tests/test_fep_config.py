"""FEP configuration schema, serialisation and validation.

The validation tests matter more than they look. A lambda ladder that does not
reach 1.0, or an MBAR request with no energy matrix to run on, produces a number
rather than an error -- just not the number anyone wanted. Each test below
corresponds to a way of quietly computing the wrong free energy.
"""

import pathlib

import pytest

from aemwater.config import (
    SUPPORTED_MU_EX_METHODS,
    ConfigError,
    FEPSpec,
    PolymerSpec,
    RunConfig,
)
from conftest import BTMA_PS


@pytest.fixture
def cfg():
    return RunConfig(polymer=PolymerSpec(smiles=BTMA_PS))


# ------------------------------------------------------------------ defaults --


def test_fep_is_the_default_estimator(cfg):
    """The point of the exercise: FEP decides saturation unless asked otherwise."""
    assert cfg.mu_ex_method == "fep"
    assert "fep" in SUPPORTED_MU_EX_METHODS


def test_widom_remains_available(cfg):
    """Widom is an independent cross-check, not dead code."""
    switched = cfg.with_overrides(**{"mu_ex_method": "widom"})
    assert switched.mu_ex_method == "widom"
    assert switched.widom.enabled


def test_morphology_count_is_configurable_up_front(cfg):
    """The user asked for this knob explicitly; it must be reachable from YAML."""
    assert cfg.fep.n_morphologies >= 1
    assert cfg.with_overrides(**{"fep.n_morphologies": 8}).fep.n_morphologies == 8


def test_default_ladders_span_the_full_path(cfg):
    for lams in (cfg.fep.lj_lambdas, cfg.fep.coul_lambdas):
        assert lams[0] == 0.0
        assert lams[-1] == 1.0
        assert all(b > a for a, b in zip(lams, lams[1:]))


def test_lj_ladder_is_clustered_near_zero(cfg):
    """Where a soft-core dU/dlambda peaks; an even ladder starves the variance."""
    lams = cfg.fep.lj_lambdas
    lower = [b - a for a, b in zip(lams, lams[1:]) if b <= 0.3]
    upper = [b - a for a, b in zip(lams, lams[1:]) if a >= 0.7]
    assert min(lower) < min(upper)


# ------------------------------------------------------------- serialisation --


def test_yaml_roundtrip_is_exact(cfg, tmp_path):
    path = cfg.dump_yaml(tmp_path / "config.yaml")
    assert RunConfig.from_yaml(path) == cfg


def test_tuple_fields_survive_yaml_as_tuples(cfg, tmp_path):
    """YAML has no tuple type; a list would break frozen-dataclass hashing."""
    reloaded = RunConfig.from_yaml(cfg.dump_yaml(tmp_path / "c.yaml"))
    assert isinstance(reloaded.fep.lj_lambdas, tuple)
    assert isinstance(reloaded.fep.coul_lambdas, tuple)
    assert isinstance(reloaded.fep.estimators, tuple)
    assert hash(reloaded.fep) == hash(cfg.fep)


def test_mu_ex_method_survives_yaml(cfg, tmp_path):
    widom = cfg.with_overrides(**{"mu_ex_method": "widom"})
    reloaded = RunConfig.from_yaml(widom.dump_yaml(tmp_path / "c.yaml"))
    assert reloaded.mu_ex_method == "widom"


def test_fep_section_is_optional_in_yaml(tmp_path):
    """Existing configs predate the section and must still load."""
    path = pathlib.Path(tmp_path / "old.yaml")
    path.write_text(f"polymer:\n  smiles: '{BTMA_PS}'\n")
    loaded = RunConfig.from_yaml(path)
    assert loaded.fep == FEPSpec()
    assert loaded.mu_ex_method == "fep"


def test_unknown_fep_key_is_rejected(tmp_path):
    path = pathlib.Path(tmp_path / "bad.yaml")
    path.write_text(f"polymer:\n  smiles: '{BTMA_PS}'\nfep:\n  n_morpholgies: 3\n")
    with pytest.raises(ConfigError, match="unknown key"):
        RunConfig.from_yaml(path)


# ---------------------------------------------------------------- validation --


@pytest.mark.parametrize(
    "override, match",
    [
        ({"fep.lj_lambdas": [0.0, 0.5]}, "span exactly"),
        ({"fep.coul_lambdas": [0.2, 1.0]}, "span exactly"),
        ({"fep.lj_lambdas": [0.0, 0.5, 0.4, 1.0]}, "strictly increasing"),
        ({"fep.lj_lambdas": [0.0, 0.5, 0.5, 1.0]}, "strictly increasing"),
        ({"fep.lj_lambdas": [0.0]}, "at least 2"),
        ({"fep.n_morphologies": 0}, "n_morphologies"),
        ({"fep.soft_core_n": 0}, "soft_core_n"),
        ({"fep.alpha_lj": 0.0}, "alpha_lj"),
        ({"fep.production_steps": 0}, "production_steps"),
        ({"fep.ti_delta": 0.0}, "ti_delta"),
        ({"fep.ti_delta": 0.6}, "ti_delta"),
        ({"fep.max_stderr": 0.0}, "max_stderr"),
        ({"fep.min_overlap": 1.0}, "min_overlap"),
        ({"fep.estimators": []}, "must not be empty"),
        ({"fep.estimators": ["mbar", "nonsense"]}, "unknown fep.estimators"),
        ({"mu_ex_method": "montecarlo"}, "mu_ex_method must be one of"),
    ],
)
def test_invalid_fep_settings_are_rejected(cfg, override, match):
    with pytest.raises((ConfigError, ValueError), match=match):
        cfg.with_overrides(**override)


def test_mbar_without_the_energy_matrix_is_rejected(cfg):
    """MBAR cannot run on neighbour differences alone; catch it at config time."""
    with pytest.raises((ConfigError, ValueError), match="rerun_matrix"):
        cfg.with_overrides(**{"fep.estimators": ["mbar"], "fep.rerun_matrix": False})


def test_bar_and_ti_do_not_require_the_matrix(cfg):
    """They run off inline compute fep output, so this combination is legal."""
    lean = cfg.with_overrides(
        **{"fep.estimators": ["bar", "ti"], "fep.rerun_matrix": False}
    )
    assert lean.fep.estimators == ("bar", "ti")


def test_too_few_frames_is_rejected(cfg):
    """20 frames per state is the floor for a meaningful MBAR uncertainty."""
    with pytest.raises((ConfigError, ValueError), match="frames per"):
        cfg.with_overrides(**{"fep.sample_every": 100_000})


def test_widom_method_with_widom_disabled_is_rejected(cfg):
    """Otherwise there is no estimator left to decide saturation."""
    with pytest.raises(ConfigError, match="no estimator"):
        cfg.with_overrides(**{"mu_ex_method": "widom", "widom.enabled": False})


def test_fep_default_still_valid_with_widom_disabled(cfg):
    """Turning Widom off is legitimate once FEP is the estimator."""
    assert cfg.with_overrides(**{"widom.enabled": False}).mu_ex_method == "fep"


def test_screening_resolution_is_cheaper_and_still_valid():
    """The screening preset must pass the same validation as production.

    A preset that trips validate() would fail only when the loop first reached
    it, which for an uptake sweep is well into a cluster job.
    """
    prod = FEPSpec()
    scr = prod.at_screening_resolution()
    scr.validate()

    cost = lambda s: (
        (len(s.lj_lambdas) + len(s.coul_lambdas))
        * (s.production_steps + s.equil_steps)
        * s.n_morphologies
    )
    assert cost(scr) / cost(prod) == pytest.approx(0.156, abs=0.01)


def test_screening_ladders_span_the_physical_endpoints():
    """A truncated ladder computes a different free energy without complaining."""
    scr = FEPSpec().at_screening_resolution()
    for lams in (scr.lj_lambdas, scr.coul_lambdas):
        assert lams[0] == 0.0 and lams[-1] == 1.0
        assert all(b > a for a, b in zip(lams, lams[1:]))


def test_screening_relaxes_the_precision_budget():
    """Otherwise every screening point reports unconverged and the loop stalls.

    The screening error bar is ~2x production by construction, so a budget
    inherited unchanged from production would be unreachable.
    """
    scr = FEPSpec(max_stderr=0.30).at_screening_resolution()
    assert scr.max_stderr >= 0.60


def test_screening_never_tightens_a_user_budget():
    """A user who set a looser budget than ours keeps theirs."""
    scr = FEPSpec(max_stderr=1.20).at_screening_resolution()
    assert scr.max_stderr == 1.20


def test_screening_is_idempotent():
    """Applying it twice must not compound into a uselessly short run."""
    once = FEPSpec().at_screening_resolution()
    twice = once.at_screening_resolution()
    assert twice.production_steps == once.production_steps
    assert twice.n_morphologies == once.n_morphologies
    assert twice.lj_lambdas == once.lj_lambdas


def test_screening_preserves_the_alchemical_path():
    """Soft-core parameters and k-space accuracy are not sampling knobs.

    Changing them changes the Hamiltonian, so a screening number would not be
    comparable to the production number it is meant to approximate.
    """
    prod = FEPSpec()
    scr = prod.at_screening_resolution()
    for attr in ("soft_core_n", "alpha_lj", "alpha_coul", "kspace_accuracy"):
        assert getattr(scr, attr) == getattr(prod, attr)
