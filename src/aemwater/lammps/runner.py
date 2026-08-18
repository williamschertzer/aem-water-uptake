"""Execute LAMMPS, with MPI capability detection and structured failures.

Launching LAMMPS is not simply "call mpirun". Open MPI's launcher opens TCP
sockets on the loopback interface for out-of-band control messages, and in a
sandboxed or containerised environment ``bind()`` is often denied -- which
produces a launcher error that looks nothing like a simulation failure and would
otherwise be reported as "the run failed". The probe below distinguishes the two,
once per process, and falls back to serial execution rather than aborting a
workflow that would run perfectly well on one core.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..utils import LOG
from .parse import LammpsRunError, parse_log


@dataclass(frozen=True)
class LammpsCapability:
    """What this machine can actually do, as opposed to what was requested."""

    binary: str
    mpi_launcher: str | None
    max_ranks: int
    reason: str = ""

    @property
    def parallel(self) -> bool:
        return self.mpi_launcher is not None


@lru_cache(maxsize=8)
def probe_lammps(binary: str = "lmp", launcher: str = "mpirun") -> LammpsCapability:
    """Determine whether LAMMPS runs, and whether it runs in parallel.

    Cached: the probe costs a subprocess launch and the answer cannot change
    within a run.
    """
    resolved = shutil.which(binary)
    if resolved is None:
        raise LammpsRunError(
            f"LAMMPS binary {binary!r} not found on PATH. Install it "
            f"(conda install -c conda-forge lammps) or set md.lammps_binary."
        )
    launcher_path = shutil.which(launcher)
    if launcher_path is None:
        return LammpsCapability(resolved, None, 1, f"{launcher} not on PATH")

    # A trivial two-rank job: if the launcher cannot open its control sockets it
    # fails here, in a tenth of a second, instead of after a long simulation.
    try:
        proc = subprocess.run(
            [launcher_path, "-np", "2", resolved, "-h"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return LammpsCapability(resolved, None, 1, f"{launcher} probe failed: {exc}")
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = next((l for l in detail if "bind()" in l or "error" in l.lower()), "")
        LOG.warning(
            "%s cannot launch here (%s); falling back to serial LAMMPS",
            launcher,
            hint[:120] or f"exit {proc.returncode}",
        )
        return LammpsCapability(resolved, None, 1, hint[:200] or "launcher failed")
    return LammpsCapability(resolved, launcher_path, os.cpu_count() or 1)


@dataclass
class LammpsRun:
    """A completed LAMMPS execution and everything needed to diagnose it."""

    workdir: Path
    input_file: Path
    log_file: Path
    returncode: int
    wall_seconds: float
    log: object = None          # LammpsLog, parsed lazily by run_lammps
    ranks: int = 1

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def output(self, name: str) -> Path:
        """Path to a file the run was expected to write, checked for existence."""
        p = self.workdir / name
        if not p.exists():
            raise LammpsRunError(
                f"{self.input_file.name} completed (exit {self.returncode}) but did "
                f"not write {name}. See {self.log_file}."
            )
        return p


def run_lammps(
    input_file: Path,
    workdir: Path | None = None,
    ranks: int = 1,
    binary: str = "lmp",
    launcher: str = "mpirun",
    log_name: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> LammpsRun:
    """Run one LAMMPS input script and parse its log.

    ``ranks`` is a request, not a guarantee: if MPI cannot launch, the run
    proceeds serially with a warning rather than failing. The number of ranks
    actually used is recorded on the result.
    """
    import time

    input_file = Path(input_file)
    workdir = Path(workdir) if workdir else input_file.parent
    log_file = workdir / (log_name or f"{input_file.stem}.log")

    cap = probe_lammps(binary, launcher)
    use_ranks = 1
    cmd = [cap.binary, "-in", input_file.name, "-log", log_file.name]
    if ranks > 1:
        if cap.parallel:
            use_ranks = min(ranks, cap.max_ranks)
            cmd = [cap.mpi_launcher, "-np", str(use_ranks)] + cmd
        else:
            LOG.warning(
                "%d ranks requested but MPI is unavailable (%s); running serially",
                ranks,
                cap.reason,
            )

    LOG.info("running %s on %d rank(s)", input_file.name, use_ranks)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
    )
    elapsed = time.time() - start

    run = LammpsRun(
        workdir=workdir,
        input_file=input_file,
        log_file=log_file,
        returncode=proc.returncode,
        wall_seconds=elapsed,
        ranks=use_ranks,
    )
    if proc.returncode != 0:
        # LAMMPS writes the actual cause into the log; the exit code alone is
        # useless for diagnosis.
        detail = ""
        if log_file.exists():
            lines = log_file.read_text().splitlines()
            errs = [l for l in lines if l.startswith("ERROR")]
            detail = errs[-1] if errs else "\n".join(lines[-5:])
        raise LammpsRunError(
            f"{input_file.name} failed (exit {proc.returncode}) after {elapsed:.1f}s: "
            f"{detail or (proc.stderr or '').strip()[-300:]}"
        )

    run.log = parse_log(log_file)
    LOG.info(
        "%s finished in %.1f s (%d rank(s))", input_file.name, elapsed, use_ranks
    )
    return run


__all__ = ["LammpsCapability", "LammpsRun", "probe_lammps", "run_lammps"]
