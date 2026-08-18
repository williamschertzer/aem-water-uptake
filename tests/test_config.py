"""Configuration schema: defaults, validation and round-tripping."""

import pytest
import yaml

from aemwater.config import (
    BoxSpec,
    ConfigError,
    InsertionSpec,
    MDSpec,
    PolymerSpec,
    RunConfig,
    WidomSpec,
    default_config,
)

BTMA_PS = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"


def test_defaults_are_valid():
    cfg = default_config(BTMA_PS)
    assert cfg.polymer.n_chains >= 1
    assert cfg.water_model == "spce"
    cfg.validate()


def test_yaml_round_trip(tmp_path):
    cfg = default_config(BTMA_PS, **{"polymer.n_chains": 6, "md.temperature": 353.0})
    path = cfg.dump_yaml(tmp_path / "c.yaml")
    back = RunConfig.from_yaml(path)
    assert back.polymer.n_chains == 6
    assert back.md.temperature == 353.0
    assert back.to_dict() == cfg.to_dict()


def test_dotted_overrides_do_not_mutate_original():
    cfg = default_config(BTMA_PS)
    new = cfg.with_overrides(**{"insertion.batch_size": 50})
    assert new.insertion.batch_size == 50
    assert cfg.insertion.batch_size != 50


def test_none_overrides_ignored():
    cfg = default_config(BTMA_PS)
    same = cfg.with_overrides(**{"polymer.n_chains": None})
    assert same.polymer.n_chains == cfg.polymer.n_chains


def test_unknown_key_rejected(tmp_path):
    bad = {"polymer": {"smiles": BTMA_PS}, "md": {"temperatrue": 300}}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="unknown key"):
        RunConfig.from_yaml(p)


def test_missing_polymer_section_rejected():
    with pytest.raises(ConfigError, match="must contain a 'polymer' section"):
        RunConfig.from_dict({"md": {"temperature": 300}})


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="unknown top-level"):
        RunConfig.from_dict({"polymer": {"smiles": BTMA_PS}, "watermodel": "spce"})


@pytest.mark.parametrize(
    "spec,kwargs,match",
    [
        (PolymerSpec, {"smiles": "", }, "must not be empty"),
        (PolymerSpec, {"smiles": BTMA_PS, "n_chains": 0}, "n_chains"),
        (PolymerSpec, {"smiles": BTMA_PS, "chain_length": -1}, "chain_length"),
        (PolymerSpec, {"smiles": BTMA_PS, "counterion": "F-"}, "counterion"),
        (PolymerSpec, {"smiles": BTMA_PS, "terminal_group": "Et"}, "terminal_group"),
    ],
)
def test_polymer_validation(spec, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        spec(**kwargs).validate()


def test_box_density_ordering_enforced():
    with pytest.raises(ConfigError, match="below box.target_density"):
        BoxSpec(initial_density=1.5, target_density=1.0).validate()


def test_md_cutoff_floor():
    with pytest.raises(ConfigError, match="cutoff"):
        MDSpec(cutoff=4.0).validate()


def test_md_ranks_positive():
    with pytest.raises(ConfigError, match="mpi_ranks"):
        MDSpec(mpi_ranks=0).validate()


def test_insertion_batch_bounds():
    with pytest.raises(ConfigError, match="min_batch_size"):
        InsertionSpec(batch_size=4, min_batch_size=10).validate()


def test_widom_blocks_floor():
    with pytest.raises(ConfigError, match="n_blocks"):
        WidomSpec(n_blocks=1).validate()


def test_widom_box_floor():
    with pytest.raises(ConfigError, match="bulk_box_length"):
        WidomSpec(bulk_box_length=12.0).validate()


def test_bad_water_model_rejected():
    with pytest.raises(ConfigError, match="water_model"):
        RunConfig(polymer=PolymerSpec(smiles=BTMA_PS), water_model="tip4p")


def test_shipped_examples_load():
    for name in ("examples/qa_polystyrene.yaml", "examples/smoke_test.yaml"):
        cfg = RunConfig.from_yaml(name)
        cfg.validate()
        assert cfg.polymer.smiles
