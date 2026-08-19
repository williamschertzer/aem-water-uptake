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
    """Mirror the context `prepare.py` renders equilibrate.in.j2 with.

    `equil_schedule` decides which of the two schemes the template emits, so it
    is part of the contract: passing `equil_schedule=None` selects the legacy
    single-squeeze path.
    """
    from aemwater.lammps.inputs import (
        ConstraintSpec, GroupSpec, equilibration_schedule, stage_spec,
    )

    from aemwater.config import PolymerSpec, RunConfig

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = overrides.pop("md", config.md)
    equil = overrides.pop("equil", config.equilibration)
    schedule = equilibration_schedule(md, equil) if equil.scheme == "21step" else None
    context = dict(
        md=md, title="t", data_file="a.data", out_data="b.data",
        out_restart="b.rst", density_file="d.dat", dump_file="t.lammpstrj",
        stages=stage_spec(md), n_averages=10, comm_cutoff=12.0,
        extra_types=None, seed=1,
        equil=equil, equil_schedule=schedule,
        equil_total_ps=(sum(s.ps for s in schedule) if schedule else None),
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


def test_widom_templates_preserve_each_insertion_burst():
    """Do not collapse a block before the Python blocking estimator sees it."""
    for name in ("insert.in.j2", "bulk.in.j2", "widom.in.j2"):
        text = (TEMPLATE_DIR / name).read_text()
        assert "{{ widom.every }} 1 {{ widom.every }}" in text


def test_insert_enables_shake_only_after_nve_limit_is_removed(tmp_path):
    """nve/limit and SHAKE must never coexist; LAMMPS can segfault at teardown."""
    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import ConstraintSpec, GroupSpec, render_input

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = config.md
    context = dict(
        md=md, widom=config.widom, title="t", data_file="a.data",
        out_data="b.data", out_restart="b.rst", dump_file="t.lammpstrj",
        density_file="density.dat", mu_file="mu.dat", water_template="h2o.mol",
        pair_coeff_lines=["pair_coeff 1 1 0.1 3.0"], extra_types=None,
        comm_cutoff=12.0, seed=1, velocity_create=True, settle_steps=10,
        soft=type("Soft", (), {"max_displacement": 0.05})(),
        minim=type("Min", (), {"etol": 1e-6, "ftol": 1e-6,
                                "max_iter": 10, "max_eval": 20})(),
        constraints=ConstraintSpec(True, False, 1, 1),
        groups=GroupSpec(n_polymer_molecules=1, n_ion_molecules=1,
                         water_type_o=1, water_type_h=2),
        n_averages=1, n_widom_samples=1, widom_window=100,
        npt_equil_steps=10, npt_prod_steps=10, widom_steps=10,
    )
    out = tmp_path / "in.insert"
    render_input("insert.in.j2", out, **context)
    text = out.read_text()
    assert text.index("unfix           nve_settle") < text.index("minimize")
    assert text.index("minimize") < text.index("fix             shake_all")


def test_21step_schedule_matches_the_published_protocol():
    """The schedule table is the protocol; pin its checkable properties.

    Larsen, Lin & Colina, Macromolecules 44 (2011) 6944: 21 steps, 1560 ps at a
    1 fs timestep, 14 NVT and 7 NPT stages, peak 50 000 atm reached at step 9,
    and the pressure released back to the operating value by step 21.
    """
    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import equilibration_schedule

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    sched = equilibration_schedule(config.md, config.equilibration)

    assert len(sched) == 21
    assert sum(s.ps for s in sched) == pytest.approx(1560.0)
    assert sum(1 for s in sched if s.ensemble == "nvt") == 14
    assert sum(1 for s in sched if s.ensemble == "npt") == 7

    pressures = [s.pressure for s in sched if s.pressure is not None]
    assert max(pressures) == pytest.approx(config.equilibration.max_pressure)
    assert sched[8].pressure == pytest.approx(config.equilibration.max_pressure), \
        "peak compression is step 9"

    # Monotone up to the peak, monotone down after it -- that shape is the
    # compression/decompression cycle, and it is what a single squeeze lacks.
    up = [s.pressure for s in sched[:9] if s.pressure is not None]
    down = [s.pressure for s in sched[9:] if s.pressure is not None]
    assert up == sorted(up), up
    assert down == sorted(down, reverse=True), down

    # The production step must end at the operating pressure exactly: it is the
    # window density is averaged over.
    assert sched[-1].ensemble == "npt"
    assert sched[-1].pressure == pytest.approx(config.md.pressure)

    # Hot excursions are above Tg; quenches are at the operating temperature.
    hot = {s.temperature for s in sched if s.temperature != config.md.temperature}
    assert hot == {config.equilibration.high_temperature}


def test_time_scale_keeps_every_stage(tmp_path):
    """A scaled-down run must exercise all 21 stages, not a shorter scheme.

    A smoke test that silently dropped stages would not test the code path the
    production run takes.
    """
    import dataclasses

    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import equilibration_schedule

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    fast = dataclasses.replace(config.equilibration, time_scale=0.02,
                               final_npt_ps=50.0)
    sched = equilibration_schedule(config.md, fast)

    assert len(sched) == 21
    assert all(s.steps(config.md.timestep) >= 1 for s in sched), \
        "no stage may round down to zero steps"
    assert sum(s.ps for s in sched) < 100.0
    # Pressures and temperatures are unchanged by scaling.
    assert max(s.pressure for s in sched if s.pressure is not None) == \
        pytest.approx(fast.max_pressure)


def test_21step_render_emits_every_stage_and_averages_only_production(tmp_path):
    """All 21 runs appear, and density averaging starts inside step 21 only.

    If `fix ave/time` were emitted before the last stage, the reported density
    would average over the 50 000 atm compression -- the reported value would be
    high and the drift check would be measuring the schedule, not the polymer.
    """
    from aemwater.lammps.inputs import render_input

    out = tmp_path / "in.equilibrate"
    render_input("equilibrate.in.j2", out, **_equilibrate_context())
    text = out.read_text()
    lines = text.splitlines()

    runs = [l for l in lines if l.split()[:1] == ["run"]]
    assert len(runs) == 21, f"expected 21 run commands, got {len(runs)}"
    # Every run must carry an integer step count. `run {{ s.steps }}` silently
    # rendered a bound method here once -- LAMMPS would have died on stage 1.
    for l in runs:
        assert int(l.split()[1]) >= 1, f"run needs an integer step count: {l!r}"

    fixes = [l for l in lines
             if l.split()[:1] == ["fix"] and l.split()[1].startswith("eq")]
    assert len(fixes) == 21
    assert sum(1 for l in fixes if " npt " in l) == 7
    assert sum(1 for l in fixes if " nvt " in l) == 14
    # Every fix is released, or LAMMPS integrates twice.
    assert len([l for l in lines if l.split()[:1] == ["unfix"]
                and l.split()[1].startswith("eq")]) == 21

    # The averaging fix sits after the 20th run and before the 21st.
    avg_index = next(i for i, l in enumerate(lines) if "avg_dens" in l)
    runs_before = sum(1 for l in lines[:avg_index] if l.split()[:1] == ["run"])
    assert runs_before == 20, \
        f"averaging must begin after 20 runs, begins after {runs_before}"

    # Peak pressure must actually appear in a fix line.
    assert any("50000" in l for l in fixes), \
        "the peak compression pressure must reach the rendered deck"
    assert "{{" not in text and "}}" not in text


def test_legacy_equilibration_densifies_under_load_before_releasing(tmp_path):
    """The squeeze must run at compression_pressure, and release before output.

    Compressing at the operating pressure does not densify on an affordable
    schedule: the validation system plateaued at 0.70 g/cm3, ~35% below a real
    dry AEM, and reported it as converged. The missing density is void space,
    which is exactly what the insertion loop then fills, so the error inflates
    every uptake number downstream. This pins the ordering.

    Scoped to ``scheme="legacy"``. The default is now the 21-step scheme, which
    has its own ordering test below; this one keeps the legacy path honest for
    as long as it is selectable.
    """
    import dataclasses

    from aemwater.config import PolymerSpec, RunConfig
    from aemwater.lammps.inputs import render_input

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    md = dataclasses.replace(config.md, pressure=1.0, compression_pressure=1000.0)
    equil = dataclasses.replace(config.equilibration, scheme="legacy")

    out = tmp_path / "in.equilibrate"
    render_input("equilibrate.in.j2", out,
                 **_equilibrate_context(md=md, equil=equil))
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
