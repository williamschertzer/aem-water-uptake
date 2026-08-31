"""The README's configuration block must name keys that really exist.

A README that documents a key the config does not have is worse than no
documentation: the loader ignores unknown keys or raises depending on the
section, so a reader following it either silently gets the default or gets an
error naming a key they copied verbatim from the project's own docs. That is
exactly what happened here -- the config example showed only a ``widom:``
block long after ``mu_ex_method`` had defaulted to ``fep``, so following it
configured the backend the project no longer recommends.

Values are deliberately *not* compared. The README shows production-scale
settings (8 ranks, 100k relax steps) against laptop-scale defaults, and that
divergence is intentional.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

README = Path(__file__).resolve().parents[1] / "README.md"


def _documented_config() -> dict:
    block = re.search(r"```yaml\n(.*?)\n```", README.read_text(), re.S)
    assert block, "no ```yaml block in README"
    parsed = yaml.safe_load(block.group(1))
    assert isinstance(parsed, dict), "the README config block is not a mapping"
    return parsed


@pytest.fixture(scope="module")
def defaults():
    from aemwater.config import PolymerSpec, RunConfig

    return RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]")).to_dict()


def test_every_documented_section_exists(defaults):
    documented = _documented_config()
    unknown = sorted(set(documented) - set(defaults))
    assert not unknown, f"README documents non-existent top-level keys: {unknown}"


def test_every_documented_key_exists(defaults):
    documented = _documented_config()
    unknown = []
    for section, body in documented.items():
        if not isinstance(body, dict):
            continue
        for key in body:
            if key not in defaults.get(section, {}):
                unknown.append(f"{section}.{key}")
    assert not unknown, f"README documents non-existent keys: {sorted(unknown)}"


def test_the_documented_mu_ex_method_is_supported(defaults):
    """And the comment must list exactly the supported backends."""
    from aemwater.config import SUPPORTED_MU_EX_METHODS

    documented = _documented_config()
    assert "mu_ex_method" in documented, (
        "the README config block does not mention mu_ex_method, which selects "
        "the measurement backend and is the first thing a reader needs"
    )
    assert documented["mu_ex_method"] in SUPPORTED_MU_EX_METHODS

    text = README.read_text()
    for method in SUPPORTED_MU_EX_METHODS:
        assert re.search(rf"mu_ex\.method: {method}\b|\b{method}\b", text), \
            f"backend {method!r} is supported but never named in the README"


def test_the_documented_default_is_the_real_default(defaults):
    """The one value that must match: documenting the wrong default misleads.

    A reader who omits the key gets the code's default, so if the README shows
    a different one they will believe they are running a backend they are not.
    """
    documented = _documented_config()
    assert documented["mu_ex_method"] == defaults["mu_ex_method"], (
        f"README shows mu_ex_method: {documented['mu_ex_method']} but the "
        f"default is {defaults['mu_ex_method']}"
    )


def test_the_fep_block_is_documented(defaults):
    """FEP is the default backend, so its block cannot be missing.

    Not an exhaustive key check -- ``kspace_accuracy`` and the trajectory flags
    are deliberately left out of a starter example -- but the keys that decide
    cost and correctness have to be there.
    """
    documented = _documented_config()
    assert "fep" in documented, "the default backend has no config example"
    load_bearing = {"n_morphologies", "lj_lambdas", "coul_lambdas",
                    "production_steps", "sample_every", "estimators"}
    missing = sorted(load_bearing - set(documented["fep"]))
    assert not missing, f"fep example omits load-bearing keys: {missing}"
