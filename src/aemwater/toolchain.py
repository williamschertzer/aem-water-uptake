"""Discovery and verification of the external toolchain (AmberTools, LAMMPS).

``aemwater doctor`` calls :func:`check_toolchain` and prints the report, so a
user can tell in one command whether their environment can run the workflow.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .utils import LOG

#: LAMMPS packages required by the workflow and the features that need them.
REQUIRED_LAMMPS_PACKAGES = {
    "MOLECULE": "bonded force field styles for the polymer",
    "KSPACE": "PPPM long-range Coulomb",
    "RIGID": "fix shake/rattle for rigid SPC/E water",
    "MC": "fix widom (excess chemical potential) and fix gcmc",
}

AMBER_BINARIES = ("antechamber", "parmchk2", "tleap")


@dataclass
class ToolReport:
    name: str
    path: str | None
    version: str | None = None
    ok: bool = False
    detail: str = ""


@dataclass
class ToolchainReport:
    tools: list[ToolReport] = field(default_factory=list)
    lammps_packages: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(t.ok for t in self.tools) and not self.missing_packages

    def render(self) -> str:
        width = max(len(t.name) for t in self.tools) if self.tools else 10
        lines = ["toolchain check", "=" * 60]
        for t in self.tools:
            mark = "ok  " if t.ok else "FAIL"
            lines.append(f"[{mark}] {t.name:<{width}}  {t.version or '-':<24} {t.path or 'NOT FOUND'}")
            if t.detail:
                lines.append(f"         {t.detail}")
        if self.lammps_packages:
            lines.append("-" * 60)
            lines.append("LAMMPS packages present: " + ", ".join(sorted(self.lammps_packages)))
        if self.missing_packages:
            lines.append(
                "MISSING LAMMPS packages: "
                + ", ".join(
                    f"{p} ({REQUIRED_LAMMPS_PACKAGES.get(p, '')})" for p in self.missing_packages
                )
            )
        lines.append("=" * 60)
        lines.append("RESULT: " + ("all required tools available" if self.ok else "toolchain incomplete"))
        return "\n".join(lines)


def _python_package_report() -> list[ToolReport]:
    reports = []
    for mod, label in (("rdkit", "rdkit"), ("parmed", "parmed"), ("MDAnalysis", "mdanalysis")):
        try:
            imported = __import__(mod)
            reports.append(
                ToolReport(label, imported.__file__, getattr(imported, "__version__", "?"), True)
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            reports.append(ToolReport(label, None, None, False, f"import failed: {exc}"))
    return reports


def lammps_binary(preferred: str | None = None) -> str | None:
    """Locate a LAMMPS executable, honouring an explicit preference."""
    candidates = [preferred] if preferred else []
    candidates += ["lmp", "lmp_serial", "lmp_mpi", "lammps"]
    for cand in candidates:
        if cand and shutil.which(cand):
            return shutil.which(cand)
    return None


def lammps_info(binary: str) -> tuple[str | None, set[str]]:
    """Return (version string, set of installed packages) for a LAMMPS binary."""
    try:
        out = subprocess.run(
            [binary, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120
        ).stdout
    except Exception as exc:  # pragma: no cover
        LOG.warning("could not query LAMMPS: %s", exc)
        return None, set()
    version = None
    m = re.search(r"Large-scale Atomic/Molecular Massively Parallel Simulator - (.+)", out)
    if m:
        version = m.group(1).strip()
    packages: set[str] = set()
    m = re.search(r"Installed packages:\s*\n(.*?)\n\s*\n", out, re.S)
    block = m.group(1) if m else out
    for token in re.split(r"\s+", block):
        token = token.strip()
        if token and re.fullmatch(r"[A-Z][A-Z0-9\-]*", token):
            packages.add(token)
    return version, packages


def check_toolchain(lammps: str | None = None) -> ToolchainReport:
    """Verify every external dependency needed for an end-to-end run."""
    report = ToolchainReport()
    report.tools.extend(_python_package_report())

    for binary in AMBER_BINARIES:
        path = shutil.which(binary)
        report.tools.append(
            ToolReport(
                binary,
                path,
                "AmberTools" if path else None,
                path is not None,
                "" if path else "install ambertools from conda-forge",
            )
        )

    lmp = lammps_binary(lammps)
    if lmp is None:
        report.tools.append(ToolReport("lammps", None, None, False, "install lammps from conda-forge"))
        report.missing_packages = sorted(REQUIRED_LAMMPS_PACKAGES)
        return report

    version, packages = lammps_info(lmp)
    report.lammps_packages = sorted(packages)
    report.missing_packages = sorted(set(REQUIRED_LAMMPS_PACKAGES) - packages)
    report.tools.append(ToolReport("lammps", lmp, version, True))
    return report


def require_toolchain(lammps: str | None = None) -> ToolchainReport:
    rep = check_toolchain(lammps)
    if not rep.ok:
        raise RuntimeError("Toolchain incomplete.\n" + rep.render())
    return rep


def amber_home() -> Path | None:
    """Best-effort AMBERHOME (conda AmberTools does not always export it)."""
    tleap = shutil.which("tleap")
    if tleap is None:
        return None
    prefix = Path(tleap).resolve().parent.parent
    return prefix if (prefix / "dat" / "leap").is_dir() else None
