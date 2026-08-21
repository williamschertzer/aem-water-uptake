"""That the fep package is actually shipped and its exports are honest.

These are cheap tests for a failure mode the test suite otherwise cannot see:
everything imports fine from the source tree whether or not the package is
declared for distribution, so a packaging omission only surfaces for whoever pip
installs the project.
"""

from __future__ import annotations

import importlib

import pytest

SUBMODULES = ("estimators", "ghost", "inputs", "rerun", "schedule")


def test_fep_is_a_real_package_not_a_namespace_package():
    """`packages.find` uses find_packages, which skips __init__.py-less dirs.

    Without an __init__.py, `aemwater.fep` works in the repo and is silently
    missing from an installed wheel -- the worst kind of defect, invisible to
    every test that runs from a source checkout.
    """
    import aemwater.fep

    assert aemwater.fep.__file__ is not None, (
        "aemwater.fep is an implicit namespace package; add an __init__.py or it "
        "will not be included by [tool.setuptools.packages.find]"
    )


def test_setuptools_discovery_includes_fep():
    setuptools = pytest.importorskip("setuptools")
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    packages = setuptools.find_packages(where=str(src))
    assert "aemwater.fep" in packages, sorted(packages)


def test_declared_exports_all_resolve():
    """An __all__ naming something that does not exist breaks `from x import *`."""
    import aemwater.fep as fep

    missing = [n for n in fep.__all__ if not hasattr(fep, n)]
    assert not missing, f"declared in __all__ but absent: {missing}"


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_exports_all_resolve(name):
    mod = importlib.import_module(f"aemwater.fep.{name}")
    missing = [n for n in getattr(mod, "__all__", ()) if not hasattr(mod, n)]
    assert not missing, f"{name} declares but does not define: {missing}"


@pytest.mark.parametrize(
    "order",
    [
        ("aemwater.config", "aemwater.fep"),
        ("aemwater.fep", "aemwater.config"),
        ("aemwater.fep.inputs", "aemwater.config"),
    ],
)
def test_no_circular_import_in_either_direction(order):
    """config imports fep.schedule; fep.inputs imports config back.

    Eager re-exports in fep/__init__.py close that loop, and the failure is
    order-dependent -- `import aemwater.config` breaks while `import aemwater.fep`
    looks fine. Run in a subprocess so a module already cached by another test
    cannot mask it.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    code = "; ".join(f"import {m}" for m in order)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={"PYTHONPATH": str(root / "src"), "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_importing_config_does_not_pull_in_pymbar():
    """Lazy exports keep the heavy estimator dependency off the config path.

    pymbar imports jax, which is slow and prints a 64-bit-mode banner; nothing
    that merely reads a config should pay for it.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aemwater.config, sys; "
            "sys.exit(1 if 'pymbar' in sys.modules else 0)",
        ],
        cwd=root,
        env={"PYTHONPATH": str(root / "src"), "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, "importing aemwater.config dragged in pymbar"


def test_pymbar_is_a_declared_dependency():
    """The estimators import it at call time, so it must be in the metadata."""
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert "pymbar" in text, "pymbar is imported by fep.estimators but undeclared"
