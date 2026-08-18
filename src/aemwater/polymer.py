"""Linear chain construction from a repeat unit, with 3D coordinates.

Two costs have to be controlled here. Stitching topology is cheap. Generating a
*non-collapsed* 3D conformer is not: a single ETKDG embedding of a 40-mer
quaternary-ammonium chain is both slow and biased toward globular, self-clashing
geometries because the embedding minimises an internal distance objective with
no notion of the melt it will live in.

The builder therefore grows the chain segment-wise: it embeds short blocks
(``segment_length`` repeat units) independently, then docks each block onto the
growing chain by rigid-body alignment along the backbone bond vector, applying a
random torsion at the junction. Cost is linear in chain length, and the result
is a self-avoiding random coil whose radius of gyration scales as a real chain
rather than a compacted ball. A short MMFF relaxation with backbone position
restraints cleans up the junction geometry without collapsing the coil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Geometry import Point3D

from .chemistry import RepeatUnit, SystemComposition, parse_repeat_unit
from .utils import LOG, ensure_dir

#: Property name marking backbone atoms of each repeat unit on the built chain.
BACKBONE_PROP = "aemwater_backbone"
UNIT_INDEX_PROP = "aemwater_unit"
#: Index of the atom within the repeat-unit template it was copied from. This is
#: what lets the force-field backend transfer per-atom charges and atom types
#: from a single small capped fragment onto every unit of a long chain, without
#: relying on substructure matching of the assembled polymer.
TEMPLATE_ATOM_PROP = "aemwater_template_atom"
ROLE_PROP = "aemwater_role"  # "head" | "tail" | "cap"
#: Atoms forming a segment-to-segment junction; only these are relaxed.
JUNCTION_PROP = "aemwater_junction"

#: Backbone C-C bond length used when welding segments, Angstrom.
CC_BOND = 1.53
#: Below this closest-contact distance a built chain is reported as strained.
MIN_ACCEPTABLE_CONTACT = 2.0


class PolymerBuildError(RuntimeError):
    """Raised when a chain cannot be constructed or embedded."""


@dataclass
class Chain:
    """A built polymer chain with 3D coordinates."""

    mol: Chem.Mol
    chain_length: int
    repeat_unit: RepeatUnit
    #: Indices of the backbone atoms, in order along the chain.
    backbone: list[int]

    @property
    def n_atoms(self) -> int:
        return self.mol.GetNumAtoms()

    @property
    def formal_charge(self) -> int:
        return Chem.GetFormalCharge(self.mol)

    def coordinates(self) -> np.ndarray:
        conf = self.mol.GetConformer()
        return np.array(conf.GetPositions(), dtype=float)

    def radius_of_gyration(self, heavy_only: bool = True) -> float:
        """Mass-weighted radius of gyration, Angstrom."""
        pos = self.coordinates()
        masses = np.array([a.GetMass() for a in self.mol.GetAtoms()])
        if heavy_only:
            keep = np.array([a.GetAtomicNum() > 1 for a in self.mol.GetAtoms()])
            pos, masses = pos[keep], masses[keep]
        com = (pos * masses[:, None]).sum(0) / masses.sum()
        d2 = ((pos - com) ** 2).sum(1)
        return float(math.sqrt((masses * d2).sum() / masses.sum()))

    def end_to_end_distance(self) -> float:
        if len(self.backbone) < 2:
            return 0.0
        pos = self.coordinates()
        return float(np.linalg.norm(pos[self.backbone[-1]] - pos[self.backbone[0]]))

    def min_interatomic_distance(self, bond_separation: int = 3) -> float:
        """Smallest distance between atoms separated by more than 3 bonds.

        A healthy conformer keeps this above ~2 A; a collapsed embedding shows
        values near 1 A, which would blow up the first LAMMPS step. Evaluated
        with a KD-tree so the check stays cheap for long chains.
        """
        from scipy.spatial import cKDTree

        pos = self.coordinates()
        if len(pos) < 2:
            return float("nan")
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=4.0, output_type="ndarray")
        if pairs.size == 0:
            return float("nan")
        dmat = Chem.GetDistanceMatrix(self.mol)
        topo = dmat[pairs[:, 0], pairs[:, 1]]
        keep = pairs[topo > bond_separation]
        if keep.size == 0:
            return float("nan")
        d = np.linalg.norm(pos[keep[:, 0]] - pos[keep[:, 1]], axis=1)
        return float(d.min())

    def to_pdb(self, path: str | Path, resname: str = "POL") -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        mol = Chem.Mol(self.mol)
        for atom in mol.GetAtoms():
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(resname.ljust(3)[:3])
            info.SetResidueNumber(1)
            info.SetName(f"{atom.GetSymbol()}{atom.GetIdx() % 10000}".ljust(4)[:4])
            info.SetIsHeteroAtom(False)
            info.SetOccupancy(1.0)
            info.SetTempFactor(0.0)
            atom.SetMonomerInfo(info)
        Chem.MolToPDBFile(mol, str(p), flavor=4)
        return p

    def to_mol(self, path: str | Path) -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        Chem.MolToMolFile(self.mol, str(p))
        return p


# --------------------------------------------------------------- geometry ----
def _embed(mol: Chem.Mol, seed: int, max_attempts: int = 5) -> Chem.Mol:
    """ETKDGv3 embed + MMFF (or UFF) relaxation of a small fragment."""
    mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.maxIterations = 400
    for attempt in range(max_attempts):
        params.randomSeed = seed + 1000 * attempt
        if AllChem.EmbedMolecule(mol, params) == 0:
            break
    else:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise PolymerBuildError(
                "RDKit could not generate 3D coordinates for the repeat-unit fragment. "
                "Check that the SMILES is chemically reasonable (valences, ring strain)."
            )
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception as exc:  # pragma: no cover - force field coverage varies
        LOG.debug("fragment relaxation skipped: %s", exc)
    return mol


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
            [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
            [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c],
        ]
    )


def _align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``src`` onto unit vector ``dst``."""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    v = np.cross(src, dst)
    c = float(np.dot(src, dst))
    if np.linalg.norm(v) < 1e-8:
        if c > 0:
            return np.eye(3)
        # anti-parallel: rotate pi about any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(src[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, perp)
        return _rotation_matrix(axis, math.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


# ------------------------------------------------------------ construction ---
def _unit_template(unit: RepeatUnit) -> tuple[Chem.Mol, int, int, int, int]:
    """Return (fragment, head, tail, head_H, tail_H) for one repeat unit.

    ``head_H``/``tail_H`` are the hydrogens RDKit added to saturate the open
    valences; they are removed when the unit is bonded to a neighbour.
    """
    mol = Chem.Mol(unit.mol)

    def _pick_h(idx: int, exclude: set[int]) -> int:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1 and nbr.GetIdx() not in exclude:
                return nbr.GetIdx()
        raise PolymerBuildError(
            f"attachment atom {idx} has no free hydrogen to displace; the repeat unit "
            "appears to be fully substituted at a backbone position."
        )

    head_h = _pick_h(unit.head_index, set())
    tail_h = _pick_h(unit.tail_index, {head_h})
    return mol, unit.head_index, unit.tail_index, head_h, tail_h


def _annotate(mol: Chem.Mol, unit_idx: int, head: int, tail: int) -> None:
    for atom in mol.GetAtoms():
        atom.SetIntProp(UNIT_INDEX_PROP, unit_idx)
        # ``mol`` is a fresh copy of the repeat-unit template, so the current
        # index *is* the template index. Recorded now because later stitching,
        # hydrogen deletion and capping all renumber the atoms.
        atom.SetIntProp(TEMPLATE_ATOM_PROP, atom.GetIdx())
    mol.GetAtomWithIdx(head).SetProp(ROLE_PROP, "head")
    mol.GetAtomWithIdx(tail).SetProp(ROLE_PROP, "tail")
    mol.GetAtomWithIdx(head).SetBoolProp(BACKBONE_PROP, True)
    mol.GetAtomWithIdx(tail).SetBoolProp(BACKBONE_PROP, True)


def build_segment(unit: RepeatUnit, n_units: int, seed: int) -> tuple[Chem.Mol, list[int]]:
    """Topologically stitch ``n_units`` repeat units; returns (mol, backbone).

    No 3D coordinates are produced here.
    """
    if n_units < 1:
        raise PolymerBuildError("n_units must be >= 1")
    template, head, tail, head_h, tail_h = _unit_template(unit)

    combined = Chem.RWMol(Chem.Mol(template))
    _annotate(combined, 0, head, tail)
    offsets = [0]
    for i in range(1, n_units):
        frag = Chem.Mol(template)
        _annotate(frag, i, head, tail)
        offset = combined.GetNumAtoms()
        combined.InsertMol(frag)
        offsets.append(offset)

    # Bond unit i's tail to unit i+1's head, deleting the displaced hydrogens.
    to_delete: list[int] = []
    for i in range(n_units - 1):
        a = offsets[i] + tail
        b = offsets[i + 1] + head
        combined.AddBond(a, b, Chem.BondType.SINGLE)
        to_delete.extend([offsets[i] + tail_h, offsets[i + 1] + head_h])

    backbone_pairs = [(offsets[i] + head, offsets[i] + tail) for i in range(n_units)]
    keep_map = _delete_atoms_tracking(combined, to_delete, backbone_pairs)
    mol = combined.GetMol()
    Chem.SanitizeMol(mol)
    backbone = [idx for pair in keep_map for idx in pair]
    return mol, backbone


def _delete_atoms_tracking(
    rw: Chem.RWMol, to_delete: Sequence[int], tracked: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Delete atoms and return ``tracked`` index pairs remapped to new indices."""
    dels = sorted(set(to_delete), reverse=True)
    for idx in dels:
        rw.RemoveAtom(idx)
    dels_sorted = sorted(set(to_delete))

    def remap(i: int) -> int:
        return i - sum(1 for d in dels_sorted if d < i)

    return [(remap(a), remap(b)) for a, b in tracked]


def _cap_termini(mol: Chem.Mol, backbone: list[int], terminal_group: str) -> Chem.Mol:
    """Cap the free chain ends. ``H`` needs nothing (RDKit already saturated)."""
    if terminal_group.upper() == "H":
        return mol
    rw = Chem.RWMol(mol)
    for end in (backbone[0], backbone[-1]):
        # Replace one hydrogen on the terminal backbone atom with a methyl carbon.
        h_idx = None
        for nbr in rw.GetAtomWithIdx(end).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                h_idx = nbr.GetIdx()
                break
        if h_idx is None:
            LOG.warning("terminal atom %d has no hydrogen to replace; leaving uncapped", end)
            continue
        cap_atom = rw.GetAtomWithIdx(h_idx)
        cap_atom.SetAtomicNum(6)
        cap_atom.SetNoImplicit(False)
        cap_atom.SetNumExplicitHs(0)
        cap_atom.SetProp(ROLE_PROP, "cap")
    out = rw.GetMol()
    # Sanitize FIRST so the new carbons are assigned their three implicit
    # hydrogens; AddHs only materialises hydrogens the valence model already
    # knows about, so calling it before sanitisation silently adds nothing and
    # leaves the chain short by 6 atoms per chain.
    Chem.SanitizeMol(out)
    has_conf = out.GetNumConformers() > 0
    out = Chem.AddHs(out, addCoords=has_conf, onlyOnAtoms=[
        a.GetIdx() for a in out.GetAtoms()
        if a.HasProp(ROLE_PROP) and a.GetProp(ROLE_PROP) == "cap"
    ])
    Chem.SanitizeMol(out)
    return out


def _backbone_order(mol: Chem.Mol, backbone_candidates: Sequence[int]) -> list[int]:
    """Order backbone atoms by walking the chain from one end."""
    cand = set(backbone_candidates)
    if not cand:
        return []
    sub = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in cand]
           for i in cand}
    ends = [i for i, nbrs in sub.items() if len(nbrs) == 1]
    start = min(ends) if ends else min(cand)
    order = [start]
    prev = None
    while True:
        nxt = [n for n in sub[order[-1]] if n != prev and n not in order]
        if not nxt:
            break
        prev = order[-1]
        order.append(nxt[0])
    return order


def _grow_coil(
    unit: RepeatUnit,
    chain_length: int,
    seed: int,
    segment_length: int,
    torsion_jitter: float,
) -> Chem.Mol:
    """Build a chain by docking independently embedded segments end to end."""
    rng = np.random.default_rng(seed)
    n_segments = max(1, math.ceil(chain_length / segment_length))
    sizes = [segment_length] * (n_segments - 1)
    sizes.append(chain_length - segment_length * (n_segments - 1))
    sizes = [s for s in sizes if s > 0]

    chain: Chem.Mol | None = None
    chain_backbone: list[int] = []

    for k, size in enumerate(sizes):
        seg_mol, seg_backbone = build_segment(unit, size, seed=int(rng.integers(1, 2**31 - 1)))
        seg_mol = _embed(seg_mol, seed=int(rng.integers(1, 2**31 - 1)))
        seg_backbone = _backbone_order(seg_mol, seg_backbone)
        if chain is None:
            chain, chain_backbone = seg_mol, seg_backbone
            continue
        chain, chain_backbone = _dock_with_retries(
            chain, chain_backbone, unit, size, rng, torsion_jitter, seg_mol, seg_backbone
        )
    assert chain is not None
    return chain


def _trial_placement(
    cpos: np.ndarray,
    spos: np.ndarray,
    c_tail: int,
    c_prev: int,
    s_head: int,
    s_next: int,
    rng: np.random.Generator,
    torsion_jitter: float,
) -> np.ndarray:
    """One random rigid-body placement of a segment onto the chain end."""
    axis = cpos[c_tail] - cpos[c_prev]
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)

    # Deflect about a random perpendicular axis so successive segments trace a
    # coil rather than a rod. The nominal deflection is the tetrahedral backbone
    # angle, but it is jittered: with a single fixed deflection the reachable
    # placements form one narrow cone, and when that cone is blocked by the
    # chain already built there is no escape, so the walk folds back on itself.
    perp = np.cross(axis, rng.normal(size=3))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0.0, 0.0, 1.0])
    deflection = math.radians(180.0 - rng.uniform(100.0, 175.0))
    direction = _rotation_matrix(perp, deflection) @ axis

    seg_axis = spos[s_next] - spos[s_head]
    if np.linalg.norm(seg_axis) < 1e-6:
        seg_axis = np.array([1.0, 0.0, 0.0])
    rot = _align_vectors(seg_axis, direction)
    twist = _rotation_matrix(direction, rng.uniform(0, 2 * math.pi) * torsion_jitter)
    rot = twist @ rot
    return (spos - spos[s_head]) @ rot.T + cpos[c_tail] + direction * CC_BOND


def _clash_score(
    cpos: np.ndarray,
    new_spos: np.ndarray,
    c_tail: int,
    junction_exclude: float = 2.0,
) -> float:
    """Smallest chain-to-segment distance, ignoring the bonded junction itself.

    The newly formed bond necessarily brings two atoms to ~1.5 A, and their
    hydrogens closer still, so atoms within ``junction_exclude`` bonds' worth of
    space around the weld are excluded; everything else must stay apart.
    """
    from scipy.spatial import cKDTree

    keep_chain = np.linalg.norm(cpos - cpos[c_tail], axis=1) > junction_exclude
    keep_seg = np.linalg.norm(new_spos - cpos[c_tail], axis=1) > junction_exclude
    if not keep_chain.any() or not keep_seg.any():
        return float("inf")
    tree = cKDTree(cpos[keep_chain])
    d, _ = tree.query(new_spos[keep_seg], k=1)
    return float(d.min())


def _dock_segment(
    chain: Chem.Mol,
    chain_backbone: list[int],
    seg: Chem.Mol,
    seg_backbone: list[int],
    rng: np.random.Generator,
    torsion_jitter: float,
    n_trials: int = 40,
    accept_distance: float = 2.4,
) -> tuple[Chem.Mol, list[int]]:
    """Rigid-body place ``seg`` after ``chain``, avoiding overlap, then bond it.

    Placement is a self-avoiding trial loop: random orientations are generated
    until one clears every existing atom by ``accept_distance``, otherwise the
    least-clashing of ``n_trials`` is used. Without this the random torsion at a
    junction regularly folds the incoming segment back onto the chain it was
    just attached to, producing sub-Angstrom contacts that no amount of
    subsequent local relaxation can undo.
    """
    cpos = np.array(chain.GetConformer().GetPositions())
    spos = np.array(seg.GetConformer().GetPositions())

    c_tail = chain_backbone[-1]
    c_prev = chain_backbone[-2] if len(chain_backbone) > 1 else chain_backbone[-1]
    s_head = seg_backbone[0]
    s_next = seg_backbone[1] if len(seg_backbone) > 1 else seg_backbone[0]

    # Two-criterion trial search. Clearance alone is not enough: a placement that
    # curls the new segment into a pocket beside the existing chain clears every
    # atom yet makes the coil more compact, and repeating that at every junction
    # drives the walk toward globular (nu ~ 1/3) rather than self-avoiding
    # (nu ~ 0.6) scaling. Excluded volume in a real chain also pushes new
    # material *outward*, so among placements that clear the chain the one making
    # the most radial progress away from the current centre of mass is taken.
    chain_com = cpos.mean(axis=0)
    feasible: list[tuple[float, np.ndarray]] = []
    best_fallback: tuple[float, np.ndarray] | None = None
    for _ in range(n_trials):
        trial = _trial_placement(cpos, spos, c_tail, c_prev, s_head, s_next, rng, torsion_jitter)
        clearance = _clash_score(cpos, trial, c_tail)
        if best_fallback is None or clearance > best_fallback[0]:
            best_fallback = (clearance, trial)
        if clearance >= accept_distance:
            outward = float(np.linalg.norm(trial.mean(axis=0) - chain_com))
            feasible.append((outward, trial))
            if len(feasible) >= max(4, n_trials // 8):
                break
    if feasible:
        spos_new = max(feasible, key=lambda pair: pair[0])[1]
        best_score = accept_distance
    else:
        assert best_fallback is not None
        best_score, spos_new = best_fallback
        LOG.debug(
            "segment docking best clearance %.2f A after %d trials; the chain is crowded "
            "and LAMMPS soft push-off will have to finish the job",
            best_score,
            n_trials,
        )

    combined = Chem.RWMol(chain)
    offset = combined.GetNumAtoms()
    combined.InsertMol(seg)
    combined.AddBond(c_tail, offset + s_head, Chem.BondType.SINGLE)
    combined.GetAtomWithIdx(c_tail).SetBoolProp(JUNCTION_PROP, True)
    combined.GetAtomWithIdx(offset + s_head).SetBoolProp(JUNCTION_PROP, True)

    # Remove one hydrogen from each newly bonded atom to keep valences correct.
    to_delete: list[int] = []
    for atom_idx in (c_tail, offset + s_head):
        for nbr in combined.GetAtomWithIdx(atom_idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1 and nbr.GetIdx() not in to_delete:
                to_delete.append(nbr.GetIdx())
                break

    tracked = [(b, b) for b in chain_backbone] + [(offset + b, offset + b) for b in seg_backbone]
    new_pairs = _delete_atoms_tracking(combined, to_delete, tracked)
    mol = combined.GetMol()

    conf = Chem.Conformer(mol.GetNumAtoms())
    all_pos = np.vstack([cpos, spos_new])
    keep = [i for i in range(all_pos.shape[0]) if i not in set(to_delete)]
    for new_i, old_i in enumerate(keep):
        x, y, z = all_pos[old_i]
        conf.SetAtomPosition(new_i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    return mol, [pair[0] for pair in new_pairs]


def _dock_with_retries(
    chain: Chem.Mol,
    chain_backbone: list[int],
    unit: RepeatUnit,
    size: int,
    rng: np.random.Generator,
    torsion_jitter: float,
    seg_mol: Chem.Mol,
    seg_backbone: list[int],
    n_rounds: int = 3,
) -> tuple[Chem.Mol, list[int]]:
    """Dock a segment, re-embedding a fresh conformer if placement stays crowded.

    Rotating one rigid conformer cannot help when the conformer's own shape is
    what collides; generating a new segment conformer gives the placement search
    a genuinely different object to fit into the available space.
    """
    best: tuple[float, Chem.Mol, list[int]] | None = None
    for round_idx in range(n_rounds):
        if round_idx > 0:
            seg_mol, seg_backbone = build_segment(unit, size, seed=int(rng.integers(1, 2**31 - 1)))
            seg_mol = _embed(seg_mol, seed=int(rng.integers(1, 2**31 - 1)))
            seg_backbone = _backbone_order(seg_mol, seg_backbone)
        candidate, cand_backbone = _dock_segment(
            chain, chain_backbone, seg_mol, seg_backbone, rng, torsion_jitter
        )
        clearance = _closest_nonbonded_contact(candidate)
        if best is None or clearance > best[0]:
            best = (clearance, candidate, cand_backbone)
        if clearance >= MIN_ACCEPTABLE_CONTACT:
            break
    assert best is not None
    return best[1], best[2]


def _closest_nonbonded_contact(mol: Chem.Mol, bond_separation: int = 3) -> float:
    """Smallest distance between atoms more than ``bond_separation`` bonds apart."""
    from scipy.spatial import cKDTree

    pos = np.array(mol.GetConformer().GetPositions())
    if len(pos) < 2:
        return float("inf")
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=4.0, output_type="ndarray")
    if pairs.size == 0:
        return float("inf")
    dmat = Chem.GetDistanceMatrix(mol)
    keep = pairs[dmat[pairs[:, 0], pairs[:, 1]] > bond_separation]
    if keep.size == 0:
        return float("inf")
    return float(np.linalg.norm(pos[keep[:, 0]] - pos[keep[:, 1]], axis=1).min())


def _junction_neighbourhood(mol: Chem.Mol, radius: int = 3) -> list[int]:
    """Atom indices within ``radius`` bonds of a segment junction or cap."""
    seeds = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.HasProp(JUNCTION_PROP)
        or (a.HasProp(ROLE_PROP) and a.GetProp(ROLE_PROP) == "cap")
    ]
    if not seeds:
        return []
    dmat = Chem.GetDistanceMatrix(mol)
    mobile = set()
    for s in seeds:
        mobile.update(int(i) for i in np.where(dmat[s] <= radius)[0])
    return sorted(mobile)


def _relax_junctions(mol: Chem.Mol, max_iters: int = 400) -> Chem.Mol:
    """Relax *only* the junction neighbourhoods, holding the coil fixed.

    An unrestrained minimisation is the wrong tool here. In vacuum, MMFF/UFF
    sees no solvent and the intramolecular vdW attraction is unopposed, so a
    long chain collapses into a compact globule: measured Rg and end-to-end
    distance then *decrease* with chain length instead of growing, and the
    packing step inherits an unphysically dense starting structure.

    Segment docking already produced sensible coil dimensions; the only bad
    geometry is at the junctions where two independently embedded blocks were
    welded together. So every atom more than a few bonds from a junction (or a
    terminal cap) is held fixed and only the welds are allowed to move.
    """
    mobile = _junction_neighbourhood(mol)
    if not mobile:
        return mol
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        if ff is None:
            return mol
        mobile_set = set(mobile)
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in mobile_set:
                ff.AddFixedPoint(atom.GetIdx())
        ff.Minimize(maxIts=max_iters)
    except Exception as exc:  # pragma: no cover
        LOG.debug("junction relaxation skipped: %s", exc)
    return mol


def _clashing_atoms(mol: Chem.Mol, threshold: float, bond_separation: int = 3) -> list[int]:
    """Atoms involved in a non-bonded contact closer than ``threshold``."""
    from scipy.spatial import cKDTree

    pos = np.array(mol.GetConformer().GetPositions())
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=threshold, output_type="ndarray")
    if pairs.size == 0:
        return []
    dmat = Chem.GetDistanceMatrix(mol)
    keep = pairs[dmat[pairs[:, 0], pairs[:, 1]] > bond_separation]
    return sorted({int(i) for i in keep.ravel()})


def _relieve_clashes(
    mol: Chem.Mol,
    threshold: float = 2.2,
    radius: int = 2,
    max_passes: int = 4,
    max_iters: int = 300,
) -> Chem.Mol:
    """Relax the neighbourhoods of residual non-bonded overlaps.

    Segment docking is a self-avoiding walk over rigid blocks, so it cannot
    always find a placement that clears every atom of a long, already-crowded
    chain. The leftover contacts are local, so they are removed locally: only
    atoms within ``radius`` bonds of a clashing atom are allowed to move, which
    relieves the overlap without letting the vacuum force field pull the whole
    coil into a globule (see :func:`_relax_junctions`).

    Contacts that survive all passes are left for the LAMMPS soft-core push-off
    stage, which is designed to remove exactly this kind of residual overlap.
    """
    dmat = None
    for _ in range(max_passes):
        clashing = _clashing_atoms(mol, threshold)
        if not clashing:
            break
        if dmat is None:
            dmat = Chem.GetDistanceMatrix(mol)
        mobile = set()
        for idx in clashing:
            mobile.update(int(i) for i in np.where(dmat[idx] <= radius)[0])
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                props = AllChem.MMFFGetMoleculeProperties(mol)
                ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is None:
                break
            for atom in mol.GetAtoms():
                if atom.GetIdx() not in mobile:
                    ff.AddFixedPoint(atom.GetIdx())
            ff.Minimize(maxIts=max_iters)
        except Exception as exc:  # pragma: no cover
            LOG.debug("clash relief skipped: %s", exc)
            break
    return mol


def build_chain(
    unit: RepeatUnit | str,
    chain_length: int,
    seed: int = 0,
    terminal_group: str = "CH3",
    segment_length: int = 4,
    torsion_jitter: float = 1.0,
    relax: bool = True,
) -> Chain:
    """Build one polymer chain with 3D coordinates.

    Parameters
    ----------
    unit:
        A :class:`~aemwater.chemistry.RepeatUnit` or a repeat-unit SMILES.
    chain_length:
        Number of repeat units.
    segment_length:
        Repeat units per independently embedded block. Small values keep the
        embedding cheap and the coil open; large values give better local
        geometry. 4 is a good compromise for vinyl backbones.
    """
    if isinstance(unit, str):
        unit = parse_repeat_unit(unit)
    if chain_length < 1:
        raise PolymerBuildError("chain_length must be >= 1")

    if chain_length <= segment_length:
        mol, backbone = build_segment(unit, chain_length, seed=seed)
        mol = _embed(mol, seed=seed)
        backbone = _backbone_order(mol, backbone)
    else:
        mol = _grow_coil(unit, chain_length, seed, segment_length, torsion_jitter)
        backbone = _backbone_order(mol, [
            a.GetIdx() for a in mol.GetAtoms() if a.HasProp(BACKBONE_PROP)
        ])

    mol = _cap_termini(mol, backbone, terminal_group)
    if mol.GetNumConformers() == 0:  # pragma: no cover
        raise PolymerBuildError("capping lost the 3D conformer")
    # Safety net: any hydrogen that AddHs could not position sits at the origin.
    if terminal_group.upper() != "H":
        mol = _place_missing_hydrogens(mol)
    if relax:
        mol = _relax_junctions(mol)
        mol = _relieve_clashes(mol)

    backbone = _backbone_order(mol, [a.GetIdx() for a in mol.GetAtoms() if a.HasProp(BACKBONE_PROP)])
    chain = Chain(mol=mol, chain_length=chain_length, repeat_unit=unit, backbone=backbone)
    LOG.info(
        "built chain: %d units, %d atoms, charge %+d, Rg = %.2f A, Ree = %.2f A",
        chain_length,
        chain.n_atoms,
        chain.formal_charge,
        chain.radius_of_gyration(),
        chain.end_to_end_distance(),
    )
    return chain


def _place_missing_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Give any zero-coordinate hydrogen a position using local geometry."""
    if mol.GetNumConformers() == 0:
        return mol
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions())
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        i = atom.GetIdx()
        if np.linalg.norm(pos[i]) > 1e-6:
            continue
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if not nbrs:
            continue
        heavy = pos[nbrs[0]]
        second = [n.GetIdx() for n in mol.GetAtomWithIdx(nbrs[0]).GetNeighbors() if n.GetIdx() != i]
        if second:
            out = heavy - pos[second].mean(axis=0)
            if np.linalg.norm(out) < 1e-6:
                out = np.array([1.0, 0.0, 0.0])
        else:
            out = np.array([1.0, 0.0, 0.0])
        new = heavy + 1.09 * out / np.linalg.norm(out)
        conf.SetAtomPosition(i, Point3D(*map(float, new)))
        pos[i] = new
    return mol


def build_chains(comp: SystemComposition, seed: int = 0, **kwargs) -> list[Chain]:
    """Build every chain in a composition, each with a different conformer."""
    chains = []
    for i in range(comp.n_chains):
        chains.append(
            build_chain(
                comp.repeat_unit,
                comp.chain_length,
                seed=seed + 7919 * i,
                terminal_group=comp.terminal_group,
                **kwargs,
            )
        )
    return chains


__all__ = [
    "Chain",
    "PolymerBuildError",
    "build_chain",
    "build_chains",
    "build_segment",
    "BACKBONE_PROP",
    "UNIT_INDEX_PROP",
    "ROLE_PROP",
    "TEMPLATE_ATOM_PROP",
]
