"""Shared pytest fixtures and capability markers.

The ``src`` layout is prepended to ``sys.path`` so the suite runs against a bare
checkout without requiring ``pip install -e .`` first.
"""

import shutil
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

BTMA_PS = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"


@pytest.fixture(scope="session")
def btma_ps_smiles():
    return BTMA_PS


def _have(*binaries):
    return all(shutil.which(b) for b in binaries)


needs_ambertools = pytest.mark.skipif(
    not _have("antechamber", "parmchk2", "tleap"),
    reason="AmberTools binaries not on PATH",
)
def lammps_binary() -> str:
    """The LAMMPS executable the tests should invoke.

    Resolution order matches ``aemwater.lammps.runner``, so a test and a real run
    exercise the same binary. Homebrew ships ``lmp_serial`` while conda-forge
    ships ``lmp``, and the two can differ by years of release -- which matters
    here, since FEP support and the exact ``compute fep`` argument handling vary
    between them.
    """
    binary = shutil.which("lmp") or shutil.which("lmp_serial")
    if binary is None:  # pragma: no cover -- guarded by needs_lammps
        raise RuntimeError("no LAMMPS binary on PATH")
    return binary


needs_lammps = pytest.mark.skipif(
    not (shutil.which("lmp") or shutil.which("lmp_serial")),
    reason="LAMMPS binary not on PATH",
)
