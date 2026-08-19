"""Command line interface and the module-level entry points.

These tests exist because two import errors -- `forcefield.gaff` for what is
actually `forcefield.gaff2`, and `type_molecule` for what is actually
`GAFF2Backend.type_chain` -- survived 179 passing tests and only surfaced when
a real run reached them. Anything imported inside a function body is invisible
to the rest of the suite, so the deferred imports are exercised explicitly.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from aemwater import cli


# --------------------------------------------------------- deferred imports --
@pytest.mark.parametrize("module", [
    "aemwater.prepare", "aemwater.driver", "aemwater.bulk",
    "aemwater.analysis", "aemwater.cli",
])
def test_deferred_imports_resolve(module):
    """Every `from .x import y` inside a function body must actually resolve.

    Walks the source for function-level imports and imports them for real.
    """
    import ast

    mod = importlib.import_module(module)
    tree = ast.parse(inspect.getsource(mod))
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.col_offset == 0:
            continue
        package = "aemwater" if node.level else None
        target = importlib.import_module(
            "." * node.level + (node.module or ""), package=package
        )
        for alias in node.names:
            assert hasattr(target, alias.name), (
                f"{module}: {'.' * node.level}{node.module} has no {alias.name!r}"
            )
            checked += 1
    assert checked > 0, f"no function-level imports found in {module}"


# ------------------------------------------------------------------- parsing --
def test_help_lists_the_three_phases(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "prepare" in out and "bulk" in out and "run" in out


def test_run_without_a_polymer_is_an_error():
    with pytest.raises(SystemExit, match="no polymer specified"):
        cli.main(["run"])


def test_an_invalid_counterion_is_rejected_before_any_simulation():
    from aemwater.config import ConfigError

    with pytest.raises(ConfigError, match="counterion"):
        cli.main(["run", "--smiles", "[*]CC[*]", "--counterion", "Cl"])


def test_command_line_options_override_the_config_file(tmp_path):
    from aemwater.config import PolymerSpec, RunConfig

    path = tmp_path / "c.yaml"
    RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]", n_chains=2)).dump_yaml(path)

    parser_args = cli.argparse.Namespace(
        config=path, smiles=None, n_chains=9, chain_length=None,
        counterion=None, water_model="tip3p", temperature=310.0, ranks=4,
    )
    config = cli._load_config(parser_args)
    assert config.polymer.n_chains == 9
    assert config.water_model == "tip3p"
    assert config.md.temperature == pytest.approx(310.0)
    assert config.md.mpi_ranks == 4


def test_bulk_runs_without_a_polymer():
    """The reservoir does not depend on the membrane chemistry."""
    args = cli.argparse.Namespace(config=None, smiles=None, water_model="spce",
                                  temperature=298.15, ranks=1)
    config = cli._bulk_only_config(args)
    assert config.water_model == "spce"
    assert config.md.temperature == pytest.approx(298.15)


def test_expert_bulk_reference_is_trusted_and_matches_run_settings(tmp_path):
    from aemwater.config import PolymerSpec, RunConfig

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    ref = cli._expert_bulk_reference(config, tmp_path, -6.2, 0.1)
    assert ref.mu_ex.mu_ex == pytest.approx(-6.2)
    assert ref.mu_ex.stderr == pytest.approx(0.1)
    assert ref.mu_ex.converged
    assert ref.settings.water_model == config.water_model
    assert ref.settings.temperature == config.md.temperature


def test_run_help_lists_expert_bulk_override(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    out = capsys.readouterr().out
    assert "--bulk-mu-ex" in out and "--bulk-stderr" in out


# ------------------------------------------------------- attribute auditing --
#: Local variable names whose type is unambiguous across the package, mapped to
#: the class they hold. Attribute access on these is checked statically.
_KNOWN_BINDINGS = {
    "comp": "aemwater.chemistry:SystemComposition",
    "estimate": "aemwater.widom:WidomEstimate",
    "ref": "aemwater.bulk:BulkReference",
    "structure": "aemwater.analysis:HydrationStructure",
    "system": "aemwater.lammps.writer:LammpsSystem",
}


@pytest.mark.parametrize("module_file", [
    "prepare.py", "driver.py", "cli.py", "analysis.py", "bulk.py",
])
def test_no_invented_attributes_on_known_types(module_file):
    """Catch `comp.n_ionic_groups` where the field is `total_ionic_groups`.

    Three such names reached execution: they are invisible to unit tests that
    never construct the real object, and a typo on a dataclass is an
    AttributeError only when that line runs -- which for an expensive pipeline
    means twenty minutes into a job.
    """
    import ast
    import dataclasses
    import importlib
    from pathlib import Path

    import aemwater

    src = Path(aemwater.__file__).parent / module_file
    tree = ast.parse(src.read_text())

    known = {}
    for var, spec in _KNOWN_BINDINGS.items():
        mod_name, cls_name = spec.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        names = {n for n in dir(cls) if not n.startswith("_")}
        if dataclasses.is_dataclass(cls):
            names |= {f.name for f in dataclasses.fields(cls)}
        known[var] = (cls_name, names)

    bad = [
        f"{module_file}:{node.lineno} {node.value.id}.{node.attr} "
        f"(not on {known[node.value.id][0]})"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in known
        and node.attr not in known[node.value.id][1]
    ]
    assert not bad, "attributes that do not exist:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("module_file", [
    "prepare.py", "driver.py", "cli.py", "analysis.py", "bulk.py",
    "lammps/inputs.py",
])
def test_methods_are_called_and_fields_are_not(module_file):
    """Catch `system.ion_residues()` where ion_residues is a tuple field.

    A dataclass field accessed as a method raises `'tuple' object is not
    callable`; a method read as a bare attribute silently yields the bound
    method object, which is worse -- it is truthy, so a validity check passes
    and the wrong thing lands in the template.
    """
    import ast
    import importlib
    import inspect
    from pathlib import Path

    import aemwater

    src = Path(aemwater.__file__).parent / module_file
    tree = ast.parse(src.read_text())

    known = {}
    for var, spec in _KNOWN_BINDINGS.items():
        mod_name, cls_name = spec.split(":")
        known[var] = (cls_name, getattr(importlib.import_module(mod_name), cls_name))

    called: set[tuple[str, str, int]] = {
        (node.func.value.id, node.func.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in known
    }

    bad = []
    for var, attr, lineno in called:
        cls_name, cls = known[var]
        static = inspect.getattr_static(cls, attr, None)
        if static is not None and not callable(static):
            bad.append(f"{module_file}:{lineno} {var}.{attr}() -- "
                       f"{cls_name}.{attr} is a {type(static).__name__}, not a method")

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in known):
            continue
        if (node.value.id, node.attr, node.lineno) in called:
            continue
        cls_name, cls = known[node.value.id]
        static = inspect.getattr_static(cls, node.attr, None)
        if inspect.isfunction(static):
            bad.append(f"{module_file}:{node.lineno} {node.value.id}.{node.attr} -- "
                       f"{cls_name}.{node.attr} is a method; the call is missing ()")

    assert not bad, "method/field confusion:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("module_file", [
    "prepare.py", "driver.py", "cli.py", "bulk.py", "analysis.py",
])
def test_calls_match_the_signatures_they_import(module_file):
    """Catch `pack_cell(min_separation=...)` where the parameter is `min_distance`.

    Resolves each `from .x import f` to the real callable and checks every
    keyword argument at each call site against its signature. A wrong keyword
    is a TypeError at call time, which for the packing step means after the
    charge derivation has already run.
    """
    import ast
    import importlib
    import inspect
    from pathlib import Path

    import aemwater

    src = Path(aemwater.__file__).parent / module_file
    tree = ast.parse(src.read_text())

    imported: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            package = "aemwater" if node.level else None
            try:
                mod = importlib.import_module(
                    "." * node.level + node.module, package=package)
            except ImportError:
                continue
            for alias in node.names:
                obj = getattr(mod, alias.name, None)
                if callable(obj):
                    imported[alias.asname or alias.name] = obj

    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        fn = imported.get(node.func.id)
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        for kw in node.keywords:
            if kw.arg and kw.arg not in params:
                bad.append(f"{module_file}:{node.lineno} {node.func.id}({kw.arg}=...) "
                           f"-- accepts {list(params)}")

        # Positional arity. `context_from_config(config, system)` passed two
        # positionally to a function that took one and swallowed the rest in
        # **extra, so the system silently never reached the template.
        if any(p.kind is p.VAR_POSITIONAL for p in params.values()):
            continue
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue
        n_positional = sum(
            1 for p in params.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        if len(node.args) > n_positional:
            bad.append(f"{module_file}:{node.lineno} {node.func.id}() given "
                       f"{len(node.args)} positional args, accepts {n_positional}")

    assert not bad, "call sites that do not match their signature:\n  " + "\n  ".join(bad)
