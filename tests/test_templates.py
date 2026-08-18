"""Every template variable is supplied by every call site that renders it.

The templates use StrictUndefined, so a missing context key is an error rather
than a silent empty string -- but only when that template is actually rendered,
which for `insert.in.j2` is several minutes into a run. `soft` reached
production missing from the minimise context for exactly this reason.

This walks the AST of the pipeline modules, collects the keyword arguments at
each `render_input(...)` call, and checks them against the variables the named
template declares. Context dicts splatted in as `**common` are resolved by
tracking what the local was assigned.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, meta

import aemwater

TEMPLATE_DIR = Path(aemwater.__file__).parent / "lammps" / "templates"

#: Keys `context_from_config(config, system)` puts in the context. The audit
#: cannot execute it, so the contract is stated here and checked against the
#: real function below -- if they drift, that test fails rather than this one
#: quietly passing on a stale list.
_CONTEXT_FROM_CONFIG_KEYS = frozenset({
    "md", "insertion", "widom", "box", "polymer", "title", "pair_coeff_lines",
    "extra_types", "constraints", "groups", "comm_cutoff", "seed",
})


def _declared_variables() -> dict[str, set[str]]:
    """Template name -> variables it needs, with includes folded in.

    Includes are read from the parsed AST rather than by looking for the
    filename in the source: `insert.in.j2` mentions `bulk.in.j2` in a comment,
    and a substring match would charge it with every variable bulk needs.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    parsed = {path.name: env.parse(path.read_text())
              for path in TEMPLATE_DIR.glob("*.j2")}
    declared = {name: meta.find_undeclared_variables(tree)
                for name, tree in parsed.items()}
    for name, tree in parsed.items():
        for included in meta.find_referenced_templates(tree):
            if included in declared:
                declared[name] |= declared[included]
    return declared


def test_context_from_config_supplies_the_keys_the_audit_assumes():
    """Keeps `_CONTEXT_FROM_CONFIG_KEYS` honest."""
    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import context_from_config

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    keys = set(context_from_config(config))
    assert keys <= _CONTEXT_FROM_CONFIG_KEYS, (
        f"context_from_config gained keys the audit does not know about: "
        f"{sorted(keys - _CONTEXT_FROM_CONFIG_KEYS)}"
    )


@pytest.mark.parametrize("module_file", ["prepare.py", "driver.py", "bulk.py"])
def test_render_calls_supply_every_template_variable(module_file):
    declared = _declared_variables()
    src = Path(aemwater.__file__).parent / module_file
    tree = ast.parse(src.read_text())

    # Locals assigned a dict literal or context_from_config(...), so that
    # `**common` can be resolved to the keys it actually carries.
    dict_locals: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            dict_locals[target.id] = {
                k.value for k in value.keys if isinstance(k, ast.Constant)}
        elif isinstance(value, ast.Call):
            fn = value.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name == "context_from_config":
                dict_locals[target.id] = set(_CONTEXT_FROM_CONFIG_KEYS)
            elif name == "dict":
                dict_locals[target.id] = {
                    kw.arg for kw in value.keywords if kw.arg}

    missing = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "render_input"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)):
            continue
        template = node.args[0].value
        if template not in declared:
            missing.append(f"{module_file}:{node.lineno} no such template {template}")
            continue

        supplied: set[str] = set()
        unresolved = False
        for kw in node.keywords:
            if kw.arg:
                supplied.add(kw.arg)
            elif isinstance(kw.value, ast.Name) and kw.value.id in dict_locals:
                supplied |= dict_locals[kw.value.id]
            else:
                unresolved = True
        if unresolved:
            continue

        absent = declared[template] - supplied
        if absent:
            missing.append(f"{module_file}:{node.lineno} render_input({template!r}) "
                           f"missing {sorted(absent)}")

    assert not missing, "template variables not supplied:\n  " + "\n  ".join(missing)


# ------------------------------------------------------- rendered behaviour --
def _equilibrate_context(**overrides):
    from aemwater.lammps.inputs import ConstraintSpec, GroupSpec, stage_spec

    from aemwater.config import PolymerSpec, RunConfig

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = overrides.pop("md", config.md)
    context = dict(
        md=md, title="t", data_file="a.data", out_data="b.data",
        out_restart="b.rst", density_file="d.dat", dump_file="t.lammpstrj",
        stages=stage_spec(md), n_averages=10, comm_cutoff=12.0,
        extra_types=None, seed=1,
        pair_coeff_lines=["pair_coeff 1 1 0.1 3.0"],
        constraints=ConstraintSpec(shake_water=False, shake_hydrogen=True),
        groups=GroupSpec(n_polymer_molecules=4, n_ion_molecules=8),
    )
    context.update(overrides)
    return context


@pytest.mark.parametrize("dump_every,expected", [(0, False), (500, True)])
def test_dump_is_omitted_when_the_frequency_is_zero(tmp_path, dump_every, expected):
    """`dump_every = 0` means no trajectory, not `dump ... 0 ...`.

    LAMMPS rejects a dump frequency of zero outright, so emitting the directive
    unconditionally made the documented "0 disables trajectory dumps" default
    fail every equilibration.
    """
    import dataclasses

    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import render_input

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = dataclasses.replace(config.md, dump_every=dump_every)

    out = tmp_path / "in.equilibrate"
    render_input("equilibrate.in.j2", out, **_equilibrate_context(md=md))
    emitted = [l for l in out.read_text().splitlines() if l.startswith("dump ")]
    assert bool(emitted) is expected
    assert not any(l.split()[3] == "0" for l in emitted), \
        "a dump directive with frequency 0 is a LAMMPS error"


def test_every_template_is_reachable_as_package_data():
    """Templates must resolve through importlib.resources, not a source path.

    They are data files inside the package, so an installed wheel finds them
    only if pyproject declares them as package-data. A test that reads them off
    the source tree would pass regardless and tell the user nothing about the
    installed artefact.
    """
    from importlib import resources

    packaged = {
        p.name
        for p in resources.files("aemwater.lammps").joinpath("templates").iterdir()
        if p.name.endswith(".j2")
    }
    on_disk = {p.name for p in TEMPLATE_DIR.glob("*.j2")}
    assert packaged == on_disk, (
        f"templates not reachable as package data: {sorted(on_disk - packaged)}"
    )


def test_equilibration_densifies_under_load_before_releasing(tmp_path):
    """The squeeze must run at compression_pressure, and release before output.

    Compressing at the operating pressure does not densify on an affordable
    schedule: the validation system plateaued at 0.70 g/cm3, ~35% below a real
    dry AEM, and reported it as converged. The missing density is void space,
    which is exactly what the insertion loop then fills, so the error inflates
    every uptake number downstream. This pins the ordering.
    """
    import dataclasses

    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import render_input

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = dataclasses.replace(config.md, pressure=1.0, compression_pressure=1000.0)

    out = tmp_path / "in.equilibrate"
    render_input("equilibrate.in.j2", out, **_equilibrate_context(md=md))
    lines = out.read_text().splitlines()

    npt = [(i, l) for i, l in enumerate(lines)
           if l.startswith("fix") and " npt " in l]
    assert len(npt) >= 3, f"expected squeeze, cool and equilibrate NPT fixes: {npt}"

    squeeze, cool, release = npt[0], npt[1], npt[-1]
    assert "1000" in squeeze[1], f"squeeze must run under load: {squeeze[1]}"
    assert "1000" in cool[1], f"cool-down must stay under load: {cool[1]}"
    # The production fix must be at the operating pressure, not the squeeze one.
    tail = release[1].split(" iso ")[1].split()
    assert float(tail[0]) == md.pressure and float(tail[1]) == md.pressure, \
        f"production NPT must run at md.pressure: {release[1]}"

    # Density averaging must start only after the release, or the reported
    # density is contaminated by the compressed transient.
    avg = next(i for i, l in enumerate(lines) if "avg_dens" in l)
    assert avg > release[0], "density averaging must follow the pressure release"
