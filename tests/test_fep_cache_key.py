"""The bulk FEP reference is cached, so its key decides correctness.

The cache is shared across runs and, at the default ``widom.cache_dir``, across
every config on the machine. A key that omits part of the protocol serves a
stale number for a request that would have produced a different one -- and
because the saturation criterion is a *difference* against this reference, a
wrong reservoir value moves the reported uptake without anything looking wrong.
The failure is silent, which is why these are asserted rather than assumed.

The specific hazard motivating the smoke config's local ``cache_dir``: a
2k-step reservoir must never be served to a 500k-step production run.
"""
from __future__ import annotations

import pytest

from aemwater.config import FEPSpec, PolymerSpec, RunConfig
from aemwater.driver import bulk_settings_for
from aemwater.fep.campaign import fep_cache_key


@pytest.fixture
def settings():
    return bulk_settings_for(RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]")))


def test_sampling_length_changes_the_key(settings):
    """The exact collision the smoke config would otherwise risk."""
    smoke = FEPSpec(production_steps=2_000, equil_steps=500, sample_every=100)
    production = FEPSpec()
    assert fep_cache_key(settings, smoke) != fep_cache_key(settings, production)


@pytest.mark.parametrize("field,value", [
    ("lj_lambdas", (0.0, 0.5, 1.0)),
    ("coul_lambdas", (0.0, 0.5, 1.0)),
    ("equil_steps", 12_345),
    ("production_steps", 123_456),
    ("sample_every", 250),
    ("n_morphologies", 7),
    ("soft_core_n", 2),
    ("alpha_lj", 0.75),
    ("alpha_coul", 12.0),
    ("ti_delta", 0.02),
])
def test_every_protocol_field_changes_the_key(settings, field, value):
    """Each of these changes the free energy, so each must change the key.

    Parametrized one field at a time: a single combined config would pass as
    long as *any* field were hashed, which is the bug this guards against.
    """
    from dataclasses import replace

    base = FEPSpec()
    assert getattr(base, field) != value, f"{field} fixture value is not a change"
    assert fep_cache_key(settings, base) != \
        fep_cache_key(settings, replace(base, **{field: value}))


def test_the_key_is_stable_for_an_unchanged_protocol(settings):
    """Otherwise the cache never hits and the reference is recomputed forever."""
    assert fep_cache_key(settings, FEPSpec()) == fep_cache_key(settings, FEPSpec())


def test_estimator_order_does_not_change_the_key(settings):
    """A set of estimators is unordered; re-listing them is not a new protocol."""
    from dataclasses import replace

    base = FEPSpec(estimators=("mbar", "bar", "ti"))
    shuffled = replace(base, estimators=("ti", "mbar", "bar"))
    assert fep_cache_key(settings, base) == fep_cache_key(settings, shuffled)


def test_thermodynamic_settings_change_the_key(settings):
    """Temperature and system size are part of the state point, not the protocol."""
    from dataclasses import replace

    spec = FEPSpec()
    warmer = replace(settings, temperature=settings.temperature + 25.0)
    bigger = replace(settings, n_waters=settings.n_waters * 2)
    assert fep_cache_key(settings, spec) != fep_cache_key(warmer, spec)
    assert fep_cache_key(settings, spec) != fep_cache_key(bigger, spec)


def test_the_screening_preset_gets_its_own_key(settings):
    """The loop runs screening and the final answer runs production.

    They differ in ladder and sampling length, so if they shared a key the
    production campaign would be handed the screening reservoir.
    """
    production = FEPSpec()
    screening = production.at_screening_resolution()
    assert fep_cache_key(settings, screening) != fep_cache_key(settings, production)
