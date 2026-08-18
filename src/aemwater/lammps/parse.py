"""Parse LAMMPS logs into thermo tables and detect run failures.

The insertion driver reads two things out of every LAMMPS run: the thermo table
(to get densities, energies and the Widom fix output) and any error the run
produced. A silent LAMMPS failure -- a lost atom, a bad SHAKE constraint, a
non-converged minimisation -- must never be mistaken for a successful step, so
error detection here is deliberately aggressive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Lines that mean the run did not do what was asked, even at exit status 0.
_FAILURE_PATTERNS = (
    re.compile(r"^ERROR", re.MULTILINE),
    re.compile(r"Lost atoms", re.IGNORECASE),
    re.compile(r"Bond atom .* missing", re.IGNORECASE),
    re.compile(r"Non-numeric (pressure|atom coords|position)", re.IGNORECASE),
    re.compile(r"Out of range atoms", re.IGNORECASE),
    re.compile(r"Shake determinant", re.IGNORECASE),
)

#: What to try when a given failure is detected. These are the causes actually hit
#: while validating this package, each of which cost a long run to diagnose; the
#: message is where that time is repaid.
_FAILURE_HINTS = (
    (re.compile(r"Shake determinant", re.IGNORECASE),
     "SHAKE could not resolve a constrained cluster. Two causes seen here: (1) "
     "molecules reassembled across a periodic boundary, giving bonds the length "
     "of the box -- check for 'Bond/angle/dihedral extent > half of periodic box' "
     "just above, and see driver._read_final_state, which must unwrap with the "
     "image flags AND sort atoms by ID; (2) X-H constraints active during Widom "
     "insertion, which resolves clusters against transient test particles -- "
     "keep md.timestep at 1.0 fs so they are not requested."),
    (re.compile(r"More than one fix shake", re.IGNORECASE),
     "LAMMPS permits one fix shake per run. Merge the keywords into a single "
     "command via ConstraintSpec.shake_command rather than emitting two fixes."),
    (re.compile(r"Lost atoms", re.IGNORECASE),
     "Atoms left the box, usually after a bad insertion or too aggressive a "
     "push-off. Reduce insertion.batch_fraction or md.soft_push max displacement."),
)

_WARNING_PATTERN = re.compile(r"^WARNING: (.*)$", re.MULTILINE)

#: LAMMPS echoes the input script into the log verbatim, comments included, so a
#: comment explaining *why* a stage guards against lost atoms would otherwise be
#: detected as the failure itself. Echoed comment lines are stripped before the
#: failure scan; nothing LAMMPS emits about a real failure starts with '#'.
_COMMENT_LINE = re.compile(r"^\s*#.*$", re.MULTILINE)


def _diagnostic_text(text: str) -> str:
    """Log text with echoed input comments removed, for failure scanning."""
    return _COMMENT_LINE.sub("", text)


class LammpsRunError(RuntimeError):
    """Raised when a LAMMPS log shows the run did not complete correctly."""


@dataclass
class ThermoTable:
    """One thermo section of a LAMMPS log."""

    columns: tuple[str, ...]
    data: np.ndarray

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.data[:, self.columns.index(name)]
        except ValueError as exc:
            raise KeyError(
                f"thermo column {name!r} not in {list(self.columns)}"
            ) from exc

    def last(self, name: str) -> float:
        col = self[name]
        if col.size == 0:
            raise LammpsRunError(f"thermo column {name!r} is empty; the run produced no output")
        return float(col[-1])

    def mean(self, name: str, last_fraction: float = 0.5) -> float:
        """Mean over the final ``last_fraction`` of the section."""
        col = self[name]
        if col.size == 0:
            raise LammpsRunError(f"thermo column {name!r} is empty")
        start = int(len(col) * (1.0 - last_fraction))
        return float(np.mean(col[start:]))

    def to_dict(self) -> dict[str, list[float]]:
        return {c: self.data[:, i].tolist() for i, c in enumerate(self.columns)}


@dataclass
class LammpsLog:
    """A parsed LAMMPS log: every thermo section plus diagnostics."""

    sections: list[ThermoTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def last_section(self) -> ThermoTable:
        if not self.sections:
            raise LammpsRunError(
                "the LAMMPS log contains no thermo output; the run failed before the first step"
            )
        return self.sections[-1]

    def concat(self) -> ThermoTable:
        """All sections sharing the last section's columns, concatenated."""
        cols = self.last_section.columns
        blocks = [s.data for s in self.sections if s.columns == cols]
        return ThermoTable(cols, np.vstack(blocks) if blocks else np.empty((0, len(cols))))


def _is_float_row(fields: list[str]) -> bool:
    if not fields:
        return False
    try:
        [float(f) for f in fields]
    except ValueError:
        return False
    return True


def parse_thermo(text: str) -> list[ThermoTable]:
    """Extract every thermo table from a LAMMPS log.

    A section starts at a header line beginning with ``Step`` (LAMMPS always puts
    ``Step`` first) and continues while rows parse as floats. ``Loop time`` and
    ``SHAKE stats`` lines end a section; interleaved warnings do not, because
    LAMMPS emits them in the middle of a run.
    """
    tables: list[ThermoTable] = []
    columns: tuple[str, ...] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal columns, rows
        if columns is not None and rows:
            tables.append(ThermoTable(columns, np.array(rows, dtype=float)))
        columns, rows = None, []

    for line in text.splitlines():
        stripped = line.strip()
        fields = stripped.split()
        if fields and fields[0] == "Step" and not _is_float_row(fields):
            flush()
            columns = tuple(fields)
            continue
        if columns is None:
            continue
        if _is_float_row(fields) and len(fields) == len(columns):
            rows.append([float(f) for f in fields])
            continue
        if stripped.startswith(("Loop time", "ERROR", "Performance:")):
            flush()
    flush()
    return tables


def parse_log(path: Path | str, check: bool = True) -> LammpsLog:
    """Parse a LAMMPS log file, raising on any sign of a failed run."""
    text = Path(path).read_text(errors="replace")
    log = LammpsLog(
        sections=parse_thermo(text),
        warnings=_WARNING_PATTERN.findall(text),
        text=text,
    )
    if check:
        scannable = _diagnostic_text(text)
        for pattern in _FAILURE_PATTERNS:
            match = pattern.search(scannable)
            if match:
                line = scannable[match.start() : scannable.find("\n", match.start())]
                message = (
                    f"LAMMPS reported a failure in {Path(path).name}: {line.strip()}"
                )
                for hint_pattern, hint in _FAILURE_HINTS:
                    if hint_pattern.search(line):
                        message += f"\n\n{hint}"
                        break
                raise LammpsRunError(message)
    return log


__all__ = ["parse_log", "parse_thermo", "LammpsLog", "ThermoTable", "LammpsRunError"]
