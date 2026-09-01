"""Small shared helpers: logging, units, block statistics, JSON I/O."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------- constants --
# LAMMPS "real" units: energy kcal/mol, length Angstrom, time fs, pressure atm.
KCAL_PER_MOL_PER_K = 0.0019872041  # Boltzmann constant, kcal/mol/K
AVOGADRO = 6.02214076e23
#: g/cm^3 per (amu / A^3)
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660

WATER_MASS_AMU = 18.01528


def kT(temperature_K: float) -> float:
    """Thermal energy in kcal/mol."""
    return KCAL_PER_MOL_PER_K * temperature_K


# ------------------------------------------------------------------ logging --
def get_logger(name: str = "aemwater", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


LOG = get_logger()


# ----------------------------------------------------------------- file I/O --
def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    try:  # numpy scalars / arrays without importing numpy at module scope
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    return obj


def write_json(path: os.PathLike | str, obj: Any) -> Path:
    """Atomically write ``obj`` as JSON (dataclasses supported)."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, p)
    return p


def read_json(path: os.PathLike | str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def read_json_or_none(path: os.PathLike | str, *,
                      description: str = "checkpoint") -> Any:
    """Return parsed JSON, or ``None`` if the file is absent or unusable.

    For files that are *caches and checkpoints* -- things whose contents can
    always be recomputed. Corruption is deliberately treated as absence: the
    alternative is that one truncated file makes every subsequent run fail
    until someone deletes it by hand, with a traceback pointing at this reader
    rather than at the kill that truncated it. The cost of ignoring the file is
    CPU time; the cost of raising is a stuck pipeline.

    Use ``read_json`` for inputs the caller cannot regenerate, where a
    corrupt file should stop the run rather than be silently replaced.

    Logged at WARNING because recomputing an expensive reference should not be
    invisible just because it was the safe thing to do.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning(
            "%s %s is unreadable (%s); ignoring it and recomputing. "
            "Delete it to silence this.", description, p, exc,
        )
        return None


# -------------------------------------------------------------- statistics ---
def block_average(values: Sequence[float], n_blocks: int = 5) -> tuple[float, float]:
    """Return (mean, standard error of the mean) from ``n_blocks`` blocks.

    Falls back to the plain standard error when fewer samples than blocks are
    available. Correlated MD/MC samples make the naive SEM optimistic; block
    averaging with a handful of blocks is the standard cheap correction.
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return vals[0], float("nan")
    mean = sum(vals) / n
    nb = max(2, min(n_blocks, n))
    size = n // nb
    if size < 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        return mean, math.sqrt(var / n)
    block_means = [sum(vals[i * size : (i + 1) * size]) / size for i in range(nb)]
    bmean = sum(block_means) / nb
    bvar = sum((b - bmean) ** 2 for b in block_means) / (nb - 1)
    return mean, math.sqrt(bvar / nb)


def propagate_sum(errors: Iterable[float]) -> float:
    """Quadrature sum of independent uncertainties (NaNs ignored)."""
    acc = 0.0
    seen = False
    for e in errors:
        if e is None:
            continue
        e = float(e)
        if math.isfinite(e):
            acc += e * e
            seen = True
    return math.sqrt(acc) if seen else float("nan")


# ----------------------------------------------------------- subprocess ------
class ExternalToolError(RuntimeError):
    """An external binary failed; carries the captured output tail."""

    def __init__(self, tool: str, returncode: int, output: str, hint: str = ""):
        self.tool = tool
        self.returncode = returncode
        self.output = output
        tail = "\n".join(output.strip().splitlines()[-25:])
        msg = f"{tool} exited with code {returncode}.\n--- last output ---\n{tail}"
        if hint:
            msg += f"\n--- hint ---\n{hint}"
        super().__init__(msg)


def which_or_raise(binary: str, hint: str = "") -> str:
    path = shutil.which(binary)
    if path is None:
        raise FileNotFoundError(
            f"Required executable '{binary}' was not found on PATH. "
            + (hint or "Install it via environment.yml (conda-forge).")
        )
    return path


def run_command(
    argv: Sequence[str],
    cwd: os.PathLike | str | None = None,
    log_path: os.PathLike | str | None = None,
    env: dict[str, str] | None = None,
    hint: str = "",
    check: bool = True,
) -> str:
    """Run ``argv``, capturing merged stdout/stderr; optionally tee to a file."""
    LOG.debug("run: %s (cwd=%s)", " ".join(map(str, argv)), cwd)
    proc = subprocess.run(
        [str(a) for a in argv],
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(proc.stdout)
    if check and proc.returncode != 0:
        raise ExternalToolError(argv[0], proc.returncode, proc.stdout, hint)
    return proc.stdout
