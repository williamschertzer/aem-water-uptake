"""Post-processing: uptake curves, structure, and the report.

Everything here reads the trajectory the driver wrote and produces figures and
tables. Kept separate from the driver so a finished run can be re-analysed
without re-simulating, and so a failed run can still be inspected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import LOG

#: Water oxygen-oxygen first-shell cutoff (A), for hydrogen-bond and cluster
#: analysis. The first minimum of the bulk water O-O RDF.
OO_FIRST_SHELL = 3.5

#: Nitrogen-oxygen cutoff (A) for counting waters in a cation's first shell.
NO_FIRST_SHELL = 4.0


@dataclass
class HydrationStructure:
    """Structural description of the water at saturation."""

    n_waters: int
    mean_cluster_size: float
    largest_cluster_fraction: float
    percolating: bool
    waters_per_cation: float
    mean_hbonds_per_water: float

    def summary(self) -> dict[str, object]:
        return {
            "mean_cluster_size": round(self.mean_cluster_size, 2),
            "largest_cluster_fraction": round(self.largest_cluster_fraction, 3),
            "percolating_water_network": self.percolating,
            "waters_in_first_cation_shell": round(self.waters_per_cation, 2),
            "hbonds_per_water": round(self.mean_hbonds_per_water, 2),
        }


def water_clusters(oxygens: np.ndarray, edge: float,
                   cutoff: float = OO_FIRST_SHELL) -> list[np.ndarray]:
    """Connected components of the water network under PBC.

    Whether the absorbed water forms isolated pockets or a connected network is
    the structural question that matters for an AEM: hydroxide conduction needs
    a percolating path, so two membranes with the same uptake can behave very
    differently.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if len(oxygens) == 0:
        return []
    tree = cKDTree(np.mod(oxygens, edge), boxsize=edge)
    pairs = np.array(list(tree.query_pairs(cutoff))) if len(oxygens) > 1 else np.empty((0, 2), int)
    n = len(oxygens)
    if len(pairs) == 0:
        return [np.array([i]) for i in range(n)]
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    return [np.where(labels == c)[0] for c in range(n_comp)]


def is_percolating(oxygens: np.ndarray, edge: float,
                   cutoff: float = OO_FIRST_SHELL) -> bool:
    """Does a water cluster span the cell in any direction?

    Tested by checking whether the largest cluster connects to its own periodic
    image: a cluster that touches both faces of the box but not through the
    boundary is not a conduction path.
    """
    clusters = water_clusters(oxygens, edge, cutoff)
    if not clusters:
        return False
    from scipy.spatial import cKDTree

    largest = max(clusters, key=len)
    if len(largest) < 2:
        return False
    pts = oxygens[largest]
    for axis in range(3):
        lo = pts[np.mod(pts[:, axis], edge) < cutoff]
        hi = pts[np.mod(pts[:, axis], edge) > edge - cutoff]
        if len(lo) == 0 or len(hi) == 0:
            continue
        shifted = hi.copy()
        shifted[:, axis] -= edge
        if cKDTree(lo).query(shifted, k=1)[0].min() < cutoff:
            return True
    return False


def count_hydrogen_bonds(coords: np.ndarray, elements: list[str], edge: float,
                         donor_acceptor_cutoff: float = 3.5,
                         angle_cutoff: float = 30.0) -> int:
    """Geometric hydrogen-bond count: O...O within cutoff and O-H...O near linear.

    The Luzar-Chandler criterion. A distance-only count roughly doubles the
    answer because it accepts pairs whose hydrogens point away from each other.
    """
    from scipy.spatial import cKDTree

    idx_o = np.array([i for i, e in enumerate(elements) if e == "O"])
    if len(idx_o) < 2:
        return 0
    ox = np.mod(coords[idx_o], edge)
    pairs = cKDTree(ox, boxsize=edge).query_pairs(donor_acceptor_cutoff)

    # Each water's hydrogens are the two atoms following its oxygen.
    n_bonds = 0
    cos_cut = np.cos(np.radians(angle_cutoff))
    for i, j in pairs:
        oi, oj = idx_o[i], idx_o[j]
        for donor, acceptor in ((oi, oj), (oj, oi)):
            hs = [donor + 1, donor + 2]
            for h in hs:
                if h >= len(elements) or elements[h] != "H":
                    continue
                oh = coords[h] - coords[donor]
                oo = coords[acceptor] - coords[donor]
                oo -= edge * np.round(oo / edge)     # minimum image
                norm = np.linalg.norm(oh) * np.linalg.norm(oo)
                if norm > 0 and float(oh @ oo) / norm > cos_cut:
                    n_bonds += 1
                    break
    return n_bonds


def hydration_structure(coords: np.ndarray, elements: list[str], edge: float,
                        n_waters: int, cation_indices: np.ndarray | None = None
                        ) -> HydrationStructure:
    """Cluster, percolation and first-shell analysis of the absorbed water."""
    if n_waters == 0:
        return HydrationStructure(0, 0.0, 0.0, False, 0.0, 0.0)

    # Water is the last n_waters * 3 atoms: assembly appends it after polymer
    # and ions, so the ordering is fixed by construction.
    water = coords[-3 * n_waters:]
    oxygens = water[0::3]
    clusters = water_clusters(oxygens, edge)
    sizes = np.array([len(c) for c in clusters])
    largest = sizes.max() / n_waters if n_waters else 0.0

    per_cation = 0.0
    if cation_indices is not None and len(cation_indices):
        from scipy.spatial import cKDTree

        tree = cKDTree(np.mod(oxygens, edge), boxsize=edge)
        counts = [len(tree.query_ball_point(np.mod(coords[i], edge), NO_FIRST_SHELL))
                  for i in cation_indices]
        per_cation = float(np.mean(counts))

    n_hb = count_hydrogen_bonds(coords[-3 * n_waters:],
                                elements[-3 * n_waters:], edge)
    return HydrationStructure(
        n_waters=n_waters,
        mean_cluster_size=float(sizes.mean()),
        largest_cluster_fraction=float(largest),
        percolating=is_percolating(oxygens, edge),
        waters_per_cation=per_cation,
        mean_hbonds_per_water=n_hb / n_waters,
    )


def plot_uptake(result, path: Path | str, bulk_mu_ex: float | None = None):
    """Four-panel summary of the loading run.

    (a) hydration number against iteration, the loading curve
    (b) chemical potential approaching the bulk reference -- the stop criterion
    (c) density and volume, showing the swelling
    (d) free volume consumed as water fills the cavities
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.to_dataframe()
    if df.empty:
        raise ValueError("no iterations to plot")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    c_main, c_ref, c_alt = "#1f4e79", "#b04a3a", "#4a7c59"

    ax = axes[0, 0]
    ax.plot(df["index"], df["lambda_value"], "o-", color=c_main, lw=1.6, ms=4)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\lambda$ (H$_2$O per ionic group)")
    ax.set_title("(a) water loading", loc="left", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(df["index"], df["water_uptake_pct"], alpha=0)
    ax2.set_ylabel("water uptake (wt %)")

    ax = axes[0, 1]
    ok = df["mu_ex"].notna()
    ax.errorbar(df.loc[ok, "lambda_value"], df.loc[ok, "mu_ex"],
                yerr=df.loc[ok, "mu_ex_stderr"], fmt="o-", color=c_main,
                lw=1.6, ms=4, capsize=3)
    if bulk_mu_ex is not None:
        ax.axhline(bulk_mu_ex, color=c_ref, ls="--", lw=1.4)
        ax.annotate("bulk water", xy=(0.98, bulk_mu_ex), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", color=c_ref, fontsize=8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\mu^{\mathrm{ex}}$ (kcal mol$^{-1}$)")
    ax.set_title("(b) saturation criterion", loc="left", fontsize=10)

    ax = axes[1, 0]
    ax.plot(df["lambda_value"], df["density"], "o-", color=c_main, lw=1.6, ms=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"density (g cm$^{-3}$)")
    ax.set_title("(c) swelling", loc="left", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(df["lambda_value"], df["volume"], "s--", color=c_alt, lw=1.2, ms=3)
    ax2.set_ylabel(r"volume ($\mathrm{\AA}^3$)", color=c_alt)
    ax2.tick_params(axis="y", colors=c_alt)

    ax = axes[1, 1]
    ax.plot(df["lambda_value"], 100 * df["free_volume_fraction"], "o-",
            color=c_main, lw=1.6, ms=4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("accessible free volume (%)")
    ax.set_title("(d) cavity filling", loc="left", fontsize=10)

    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    axes[0, 0].spines["right"].set_visible(True)
    axes[1, 0].spines["right"].set_visible(True)

    fig.suptitle(
        rf"$\lambda_\mathrm{{max}}$ = {result.lambda_value:.1f}, "
        rf"{result.water_uptake_pct:.1f} wt % ({result.stop_reason.replace('_', ' ')})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _markdown_table(df) -> str:
    """Render a DataFrame as a markdown table.

    `DataFrame.to_markdown` needs `tabulate`, an optional pandas dependency.
    Requiring it would make the report -- the last step of a run that takes
    hours -- fail on a missing package after all the expensive work is done, so
    the eight columns are formatted here instead.
    """
    header = [str(c) for c in df.columns]
    def cell(v) -> str:
        # A missing mu_ex arrives as None from the driver and as NaN once pandas
        # has been through it; both mean "not measured this iteration".
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "--"
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    rows = [[cell(v) for v in row]
            for row in df.itertuples(index=False, name=None)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    return "\n".join([line(header),
                      "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])


def write_report(result, structure, path: Path | str) -> Path:
    """Markdown summary of the run: the number, how it was reached, caveats."""
    df = result.to_dataframe()
    lines = [
        "# Water uptake",
        "",
        f"**lambda = {result.lambda_value:.1f} H2O per ionic group** "
        f"({result.water_uptake_pct:.1f} wt %)",
        "",
        f"- stop reason: `{result.stop_reason}`",
        f"- converged: **{result.converged}**",
        f"- iterations: {len(df)}",
        f"- dry density: {result.dry_density:.4f} g/cm3",
        f"- hydrated density: {result.hydrated_density:.4f} g/cm3",
        f"- bulk reference mu_ex: {result.bulk_mu_ex:.3f} kcal/mol",
        "",
    ]
    if not result.converged:
        lines += [
            "> The loop stopped before the chemical potential of water in the",
            "> membrane reached the bulk value. The uptake above is a **lower",
            "> bound**, not a saturation measurement.",
            "",
        ]
    if structure is not None:
        lines += ["## Hydration structure", ""]
        lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in structure.summary().items()]
        lines += [""]
        if not structure.percolating:
            lines += [
                "> The water does not form a cell-spanning cluster. At this",
                "> hydration the membrane has isolated pockets rather than a",
                "> connected conduction path.",
                "",
            ]
    lines += ["## Trajectory", "", _markdown_table(df), ""]
    Path(path).write_text("\n".join(lines))
    return Path(path)


__all__ = [
    "HydrationStructure",
    "hydration_structure",
    "water_clusters",
    "is_percolating",
    "count_hydrogen_bonds",
    "plot_uptake",
    "write_report",
    "OO_FIRST_SHELL",
    "NO_FIRST_SHELL",
]
