from types import SimpleNamespace

import numpy as np
import pytest

from aemwater import cli, prepare
from aemwater.config import PolymerSpec, RunConfig


def test_restart_skips_build_and_minimization(tmp_path, monkeypatch):
    from aemwater import assembly, driver, polymer
    from aemwater.lammps import inputs, runner

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    dry = tmp_path / "dry"
    dry.mkdir()
    (dry / "min.data").write_text("saved coordinates")
    (dry / "equil.log").write_text("previous failure")
    monkeypatch.setattr(prepare, "_load_typed_chain", lambda p: object())
    monkeypatch.setattr(driver, "_read_final_state",
                        lambda p: (np.zeros((1, 3)), ["C"], 30.0))
    monkeypatch.setattr(assembly, "assemble", lambda *a, **k: object())
    monkeypatch.setattr(inputs, "context_from_config", lambda *a: {})

    def forbidden(*args, **kwargs):
        pytest.fail("restart attempted to rebuild the polymer")

    monkeypatch.setattr(polymer, "build_chain", forbidden)
    rendered = []
    monkeypatch.setattr(inputs, "render_input",
                        lambda template, path, **kwargs: rendered.append((template, kwargs)))

    def stop_before_md(path, **kwargs):
        assert path.name == "in.equilibrate"
        assert rendered[0][0] == "equilibrate.in.j2"
        assert rendered[0][1]["data_file"] == "min.data"
        assert list((dry / "equilibration_attempts").glob("*/equil.log"))[0].read_text() == "previous failure"
        raise RuntimeError("MD intentionally not launched")

    monkeypatch.setattr(runner, "run_lammps", stop_before_md)
    with pytest.raises(RuntimeError, match="intentionally"):
        prepare.prepare_dry_membrane(config, tmp_path, resume_minimized=True)
    assert (dry / "min.data").read_text() == "saved coordinates"


@pytest.mark.parametrize("case", ["force", "changed_polymer", "uptake", "completed", "missing_min"])
def test_cli_rejects_invalid_restart_before_saving_config(tmp_path, monkeypatch, case):
    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    config.dump_yaml(tmp_path / "run_config.yaml")
    requested = config
    if case == "changed_polymer":
        requested = RunConfig(polymer=PolymerSpec(smiles="[*]CCC[*]"))
    if case == "uptake":
        (tmp_path / "uptake_state.json").write_text("{}")
    if case == "completed":
        (tmp_path / "dry").mkdir()
        (tmp_path / "dry/dry.data").write_text("completed dry system")
    monkeypatch.setattr(cli, "_load_config", lambda args: requested)
    monkeypatch.setattr(cli, "_save_run_config",
                        lambda *a: pytest.fail("invalid restart overwrote config"))
    args = SimpleNamespace(workdir=tmp_path, resume_minimized=True, force=case == "force")
    with pytest.raises(SystemExit, match="resume-minimized"):
        cli.cmd_run(args)
