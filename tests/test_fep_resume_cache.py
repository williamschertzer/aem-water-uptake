"""Crash-safety of the resume decision inputs: the bulk reference caches.

The FEP campaign resumes per lambda window, but that machinery only engages
once the campaign is running. Above it sits the bulk reference cache, which is
the single most expensive stage and therefore the one most likely to be killed
by a queue walltime. Two failure modes live there, and both are about the
*next* run rather than the one that died:

* a kill during the cache write leaves truncated JSON, which used to make every
  subsequent run raise ``JSONDecodeError`` from the reader -- a traceback
  pointing far away from the kill that caused it, and unfixable without
  deleting a file by hand;
* ``--force`` did not reach this layer at all, so the documented way to discard
  a stale reference silently kept reading the cache and recomputed only the
  membrane against it.

The second is the dangerous one. It produces a *number*, not an error: a
membrane measured under new settings compared against a reservoir measured
under old ones.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from aemwater.bulk import BulkSettings
from aemwater.config import PolymerSpec, RunConfig
from aemwater.driver import bulk_settings_for, obtain_bulk_reference
from aemwater.utils import read_json_or_none, write_json


def _config(method="fep", cache_dir=None):
    """A config whose bulk cache lives under the test's tmp_path.

    The default is ``~/.cache/aemwater``; a test that left it there would write
    into the developer's real cache and, worse, could *read* a genuine cached
    reference and pass without exercising anything.
    """
    config = RunConfig(
        polymer=PolymerSpec(smiles="[*]CC[*]", n_chains=1, chain_length=1),
        mu_ex_method=method,
    )
    if cache_dir is not None:
        config = replace(config,
                         widom=replace(config.widom, cache_dir=str(cache_dir)))
    return config


# ------------------------------------------------------------ atomic write --
def test_atomic_write_round_trips(tmp_path):
    path = write_json(tmp_path / "c.json", {"mu_ex": -6.5, "n": 3})
    assert json.loads(path.read_text()) == {"mu_ex": -6.5, "n": 3}


def test_atomic_write_leaves_no_temp_file_behind(tmp_path):
    write_json(tmp_path / "c.json", {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_atomic_write_replaces_rather_than_truncating(tmp_path):
    """The old content must survive until the new content is complete.

    Asserted via the temp file's location rather than by racing a kill: the
    rename is only atomic within one filesystem, so a temp file written to the
    system temp dir would silently degrade to a copy on any cluster where the
    workdir is a scratch mount.
    """
    path = tmp_path / "c.json"
    write_json(path, {"v": 1})

    seen = {}
    real_replace = __import__("os").replace

    def spy(src, dst):
        seen["src_parent"] = str(src).rsplit("/", 1)[0]
        seen["old_still_readable"] = json.loads(open(dst).read())
        return real_replace(src, dst)

    import aemwater.utils as utils_mod

    monkey = pytest.MonkeyPatch()
    monkey.setattr(utils_mod.os, "replace", spy)
    try:
        write_json(path, {"v": 2})
    finally:
        monkey.undo()

    assert seen["src_parent"] == str(tmp_path), "temp file must be a sibling"
    assert seen["old_still_readable"] == {"v": 1}
    assert json.loads(path.read_text()) == {"v": 2}


# ------------------------------------------------------------ tolerant read --
def test_truncated_json_reads_as_absent(tmp_path, caplog):
    path = tmp_path / "c.json"
    full = json.dumps({"mu_ex": -6.5, "stderr": 0.1})
    path.write_text(full[: len(full) // 2])
    assert read_json_or_none(path) is None


def test_discarding_a_corrupt_file_is_logged(tmp_path, caplog):
    """Silently recomputing an expensive reference must still be visible."""
    path = tmp_path / "c.json"
    path.write_text("{not json")
    with caplog.at_level("WARNING"):
        read_json_or_none(path, description="bulk reference cache")
    assert "unreadable" in caplog.text
    assert "bulk reference cache" in caplog.text


def test_absent_file_reads_as_absent_without_warning(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert read_json_or_none(tmp_path / "nope.json") is None
    assert caplog.text == ""


def test_valid_json_is_returned(tmp_path):
    path = write_json(tmp_path / "c.json", {"mu_ex": -6.5})
    assert read_json_or_none(path) == {"mu_ex": -6.5}


# ------------------------------------------------------- resume threading ---
@pytest.mark.parametrize("method", ["fep", "widom"])
def test_force_reaches_the_bulk_reference(tmp_path, monkeypatch, method):
    """``resume=False`` must bypass the cache on *both* backends.

    Parametrized rather than tested on FEP alone: the two backends have
    separate cache files and separate readers, and the bug being guarded
    against was one of them not being wired.
    """
    config = _config(method, cache_dir=tmp_path / "cache")
    settings = bulk_settings_for(config)
    cache = tmp_path / "cache"
    cache.mkdir()

    from aemwater.fep.campaign import fep_cache_key

    name = (f"bulkfep_{fep_cache_key(settings, config.fep)}.json"
            if method == "fep" else f"bulk_{settings.key()}.json")
    # A cache that would be *readable* -- so if it is consulted at all, it is
    # returned rather than recomputed, and the test fails by passing silently.
    (cache / name).write_text(json.dumps({
        "mu_ex": -99.0, "stderr": 0.0, "density": 1.0, "volume": 3000.0,
        "n_morphologies": 1, "per_morphology": [], "n_blocks": 1,
        "block_values": [1.0], "mean_boltzmann": 1.0, "effective_samples": 1.0,
    }))

    called = {"ran": False}

    def _refuse(*a, **kw):
        called["ran"] = True
        raise RuntimeError("recomputed")

    # Patched at `aemwater.lammps.runner.run_lammps`, the one chokepoint both
    # backends funnel through -- the FEP path via run_bulk_campaign -> run_leg,
    # the Widom path via _run_bulk_stages. Both import it inside the function,
    # so the module attribute is what the call resolves against. Patching the
    # higher-level entry points instead would cover FEP and silently let the
    # Widom parametrisation launch real LAMMPS.
    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", _refuse)

    # resume=True: the cache is honoured, nothing is recomputed.
    ref = obtain_bulk_reference(config, tmp_path / "bulk", ranks=1, resume=True)
    assert ref.mu_ex.mu_ex == pytest.approx(-99.0)
    assert not called["ran"]

    # resume=False: the cache must be bypassed and the campaign re-entered.
    with pytest.raises(RuntimeError, match="recomputed"):
        obtain_bulk_reference(config, tmp_path / "bulk", ranks=1, resume=False)
    assert called["ran"]


def test_corrupt_cache_recomputes_instead_of_raising(tmp_path, monkeypatch):
    """The regression: a truncated cache used to wedge every later run."""
    config = _config("fep", cache_dir=tmp_path / "cache")
    settings = bulk_settings_for(config)
    cache = tmp_path / "cache"
    cache.mkdir()

    from aemwater.fep.campaign import fep_cache_key

    full = json.dumps({"mu_ex": -6.5, "stderr": 0.1, "density": 1.0})
    (cache / f"bulkfep_{fep_cache_key(settings, config.fep)}.json").write_text(
        full[: len(full) // 2])

    def _refuse(*a, **kw):
        raise RuntimeError("recomputed")

    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", _refuse)

    # Reaching the campaign at all is the assertion: previously this raised
    # JSONDecodeError from the cache reader and never got here.
    with pytest.raises(RuntimeError, match="recomputed"):
        obtain_bulk_reference(config, tmp_path / "bulk", ranks=1)


def test_corrupt_loop_checkpoint_restarts_instead_of_raising(tmp_path, caplog):
    """The uptake checkpoint is rewritten every iteration.

    That makes it the file most likely to be mid-write when a walltime kill
    lands, so it gets the same treatment as the caches: a damaged copy costs
    the hydration trajectory, not the run.

    Driving a whole ``run_uptake`` to reach this line would need a bulk
    reference, a dry membrane and a stubbed engine -- scaffolding that tests
    those preconditions rather than this read. The read itself is asserted
    here; that the driver *uses* it (rather than a bare ``json.loads``) is
    asserted by ``test_checkpoint_readers_tolerate_corruption`` below, which
    is the half that actually rots.
    """
    from aemwater.utils import read_json_or_none

    state = tmp_path / "uptake_state.json"
    full = json.dumps({"iterations": [{"index": 3}], "n_waters": 120})
    state.write_text(full[: len(full) // 2])

    with caplog.at_level("WARNING"):
        assert read_json_or_none(state, description="uptake checkpoint") is None
    assert "uptake checkpoint" in caplog.text


def test_checkpoint_readers_tolerate_corruption():
    """The wiring half: no resume path may read a checkpoint with bare json.

    A tolerant helper proves nothing if the caller does not use it -- which is
    exactly the state this code was in. Scanning the source is crude but it is
    the property that matters, and it covers readers added later.
    """
    import inspect
    import re

    import aemwater.bulk as bulk_mod
    import aemwater.driver as driver_mod

    for mod in (bulk_mod, driver_mod):
        src = inspect.getsource(mod)
        for match in re.finditer(r"json\.loads\((\w+)\.read_text\(\)\)", src):
            name = match.group(1)
            assert not any(k in name for k in ("cache", "state", "checkpoint")), (
                f"{mod.__name__} reads {name} with bare json.loads; a kill "
                f"mid-write would wedge every later run. Use "
                f"utils.read_json_or_none."
            )


def test_checkpoint_writers_do_not_bypass_the_atomic_helper():
    """Guards the pattern, not one call site.

    The duplication is what let this diverge in the first place: two cache
    writers, written at different times, both non-atomic in the same way. A new
    ``write_text(json.dumps(...))`` on a file a resumed run reads would
    reintroduce it silently.
    """
    import inspect

    import aemwater.bulk as bulk_mod
    import aemwater.driver as driver_mod
    import aemwater.fep.resume as resume_mod

    for mod in (bulk_mod, driver_mod, resume_mod):
        src = inspect.getsource(mod)
        assert "write_text(json.dumps" not in src, (
            f"{mod.__name__} writes JSON non-atomically; use utils.write_json "
            f"so a kill mid-write cannot leave a truncated checkpoint"
        )


def test_every_entry_point_between_cli_and_cache_forwards_resume():
    """Guards the wiring, not one call.

    ``--force`` crosses four functions to reach the cache. Each hop is a plain
    keyword argument that a later edit can silently drop, and dropping one
    produces no error -- just a stale reservoir under a fresh membrane.
    """
    import inspect

    from aemwater.bulk import run_bulk_reference, run_bulk_reference_fep
    from aemwater.driver import run_uptake
    from aemwater.uptake_campaign import run_uptake_campaign

    for fn in (obtain_bulk_reference, run_bulk_reference_fep,
               run_bulk_reference, run_uptake, run_uptake_campaign):
        assert "resume" in inspect.signature(fn).parameters, (
            f"{fn.__name__} lost its resume parameter; --force would stop "
            f"propagating here"
        )

    # And the internal calls actually pass it on, rather than defaulting.
    for mod, callee in (("aemwater.driver", "obtain_bulk_reference"),
                        ("aemwater.uptake_campaign", "obtain_bulk_reference")):
        src = inspect.getsource(__import__(mod, fromlist=["x"]))
        idx = src.find(f"{callee}(")
        while idx != -1:
            call = src[idx: src.find(")", idx) + 1]
            if "workdir" in call:  # the real call, not the def
                assert "resume" in call, (
                    f"{mod} calls {callee} without forwarding resume: {call}")
            idx = src.find(f"{callee}(", idx + 1)
