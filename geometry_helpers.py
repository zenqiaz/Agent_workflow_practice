from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt, acos, atan2, degrees, pi
from typing import List, Tuple, Optional, Dict, Literal
import json

Vec3 = Tuple[float, float, float]

COV_RAD = {  # Å, rough
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
    # metals
    "Cu": 1.32, "Fe": 1.32, "Ru": 1.46, "Ni": 1.24, "Pt": 1.36,
    "Zn": 1.22, "Co": 1.26, "Mn": 1.29, "Cr": 1.27, "Pd": 1.39,
    "Rh": 1.42, "Ir": 1.41,
}

VDW_RAD: Dict[str, float] = {  # Å, for clash detection
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Cu": 1.40, "Fe": 1.40, "Ni": 1.63, "Zn": 1.39,
    "Co": 1.40, "Ru": 1.46, "Pt": 1.72, "Pd": 1.63,
}

# Reference bond lengths (Å) by sorted element pair.
# Used with short_thresh (default 0.85×) and long_thresh (default 1.20×).
# Metal–ligand pairs use wider thresholds (0.80× / 1.25×) because of variability.
BOND_REF: Dict[Tuple[str, str], float] = {
    ("C",  "C"):  1.54,
    ("C",  "N"):  1.47,
    ("C",  "O"):  1.43,
    ("C",  "H"):  1.09,
    ("C",  "S"):  1.82,
    ("C",  "Cl"): 1.77,
    ("C",  "Br"): 1.94,
    ("C",  "F"):  1.35,
    ("H",  "N"):  1.01,
    ("H",  "O"):  0.96,
    ("H",  "S"):  1.34,
    ("N",  "N"):  1.45,
    ("N",  "O"):  1.40,
    ("S",  "S"):  2.05,
    # metal–ligand (sorted so metal comes first alphabetically only if needed)
    ("Cu", "N"):  2.05,
    ("Cu", "O"):  2.00,
    ("Cl", "Cu"): 2.28,
    ("Cu", "S"):  2.31,
    ("Fe", "N"):  2.00,
    ("Fe", "O"):  2.05,
    ("Cl", "Fe"): 2.32,
    ("C",  "Fe"): 1.85,
    ("N",  "Ru"): 2.10,
    ("C",  "Ru"): 1.86,
    ("N",  "Ni"): 1.95,
    ("Ni", "O"):  2.05,
    ("N",  "Pt"): 2.02,
    ("Cl", "Pt"): 2.33,
    ("N",  "Zn"): 2.05,
    ("O",  "Zn"): 2.00,
    ("Co", "N"):  1.97,
    ("Co", "O"):  1.90,
}

METALS = frozenset({
    "Cu", "Fe", "Ru", "Ni", "Pt", "Zn", "Co", "Mn", "Cr", "Pd", "Rh", "Ir",
    "Ti", "V", "Mo", "W", "Re", "Os", "Au", "Ag",
})

ATOMIC_NUM: Dict[str, int] = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,  "C": 6,  "N": 7,
    "O": 8,  "F": 9,   "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S": 16,  "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47,
    "Ir": 77, "Pt": 78, "Au": 79,
}

BOND_LEN = {  # Å typical X–H
    "O": 0.97, "N": 1.01, "S": 1.34, "Cl": 1.27, "F": 0.92
}

NONMETALS = {"H","C","N","O","F","P","S","Cl","Br","I"}

def _dist(a: Vec3, b: Vec3) -> float:
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def _vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def _vmul(a: Vec3, s: float) -> Vec3:
    return (a[0]*s, a[1]*s, a[2]*s)

def _norm(a: Vec3) -> float:
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

def _unit(a: Vec3) -> Vec3:
    n = _norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0]/n, a[1]/n, a[2]/n)

# ── Additional vector math ────────────────────────────────────────────────────

def _dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def _angle_deg(a: Vec3, center: Vec3, b: Vec3) -> float:
    """Angle a–center–b in degrees."""
    u = _unit(_vsub(a, center))
    v = _unit(_vsub(b, center))
    return degrees(acos(max(-1.0, min(1.0, _dot(u, v)))))

def _dihedral_deg(p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3) -> float:
    """Signed dihedral p1–p2–p3–p4 in degrees (−180 to 180)."""
    b1 = _vsub(p2, p1)
    b2 = _vsub(p3, p2)
    b3 = _vsub(p4, p3)
    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    m1 = _cross(n1, _unit(b2))
    return degrees(atan2(_dot(m1, n2), _dot(n1, n2)))

# ── Inspection dataclasses ────────────────────────────────────────────────────

@dataclass
class BondIssue:
    i: int
    j: int
    element_i: str
    element_j: str
    distance: float
    reference: float
    flag: str  # "too_short" | "too_long"

@dataclass
class ClashIssue:
    i: int
    j: int
    element_i: str
    element_j: str
    distance: float
    threshold: float

@dataclass
class AngleIssue:
    i: int
    j: int   # central atom
    k: int
    element_j: str
    angle_deg: float
    lo: float
    hi: float

@dataclass
class TorsionIssue:
    i: int
    j: int
    k: int
    l: int
    dihedral_deg: float
    flag: str  # "eclipsed"

@dataclass
class InspectionReport:
    n_atoms: int
    n_bonds: int
    bond_issues: List[BondIssue]       = field(default_factory=list)
    clash_issues: List[ClashIssue]     = field(default_factory=list)
    angle_issues: List[AngleIssue]     = field(default_factory=list)
    torsion_issues: List[TorsionIssue] = field(default_factory=list)
    charge_spin_issues: List[str]      = field(default_factory=list)
    # each hazard: (code, message, relevant_atom_indices)
    qm_hazards: List[Tuple[str, str, List[int]]] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not (self.bond_issues or self.clash_issues or self.angle_issues
                    or self.torsion_issues or self.charge_spin_issues or self.qm_hazards)

    def summary(self) -> str:
        parts = []
        if self.bond_issues:       parts.append(f"{len(self.bond_issues)} bond issue(s)")
        if self.clash_issues:      parts.append(f"{len(self.clash_issues)} clash(es)")
        if self.angle_issues:      parts.append(f"{len(self.angle_issues)} angle issue(s)")
        if self.torsion_issues:    parts.append(f"{len(self.torsion_issues)} torsion issue(s)")
        if self.charge_spin_issues:parts.append(f"{len(self.charge_spin_issues)} charge/spin issue(s)")
        if self.qm_hazards:        parts.append(f"{len(self.qm_hazards)} QM hazard(s)")
        return "clean" if not parts else "; ".join(parts)

# ── Bond list helpers ─────────────────────────────────────────────────────────

def bonds_to_adj(n: int, bond_list: List[Tuple[int, int]]) -> List[List[int]]:
    """Convert (i,j) bond pair list to adjacency list."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bond_list:
        adj[i].append(j)
        adj[j].append(i)
    return adj

def infer_bond_list(
    atoms: List[str],
    coords: List[Vec3],
    organic_scale: float = 1.20,
    metal_scale: float = 1.30,
) -> List[Tuple[int, int]]:
    """Infer bonds from coordinates using covalent radii thresholds.

    Returns sorted list of (i, j) pairs with i < j.
    Uses a higher threshold for bonds involving metal atoms.
    This is Section 0 of the geometry inspection pipeline; skip for SDF/mol2/SMILES input
    by passing an explicit bond_list to inspect_geometry instead.
    """
    n = len(atoms)
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        ri = COV_RAD.get(atoms[i], 0.77)
        is_metal_i = atoms[i] in METALS
        for j in range(i + 1, n):
            rj = COV_RAD.get(atoms[j], 0.77)
            scale = metal_scale if (is_metal_i or atoms[j] in METALS) else organic_scale
            if _dist(coords[i], coords[j]) <= scale * (ri + rj):
                bonds.append((i, j))
    return bonds

# ── Section A — bond length checks ───────────────────────────────────────────

def _bond_ref_key(el1: str, el2: str) -> Tuple[str, str]:
    return (el1, el2) if el1 <= el2 else (el2, el1)

def inspect_bonds(
    atoms: List[str],
    coords: List[Vec3],
    bond_list: List[Tuple[int, int]],
    short_thresh: float = 0.85,
    long_thresh: float = 1.20,
    metal_short_thresh: float = 0.80,
    metal_long_thresh: float = 1.25,
) -> List[BondIssue]:
    """Check each bond against expected reference length."""
    issues: List[BondIssue] = []
    for i, j in bond_list:
        key = _bond_ref_key(atoms[i], atoms[j])
        ref = BOND_REF.get(key)
        if ref is None:
            continue
        d = _dist(coords[i], coords[j])
        is_metal = atoms[i] in METALS or atoms[j] in METALS
        lo = metal_short_thresh if is_metal else short_thresh
        hi = metal_long_thresh  if is_metal else long_thresh
        if d < lo * ref:
            issues.append(BondIssue(i, j, atoms[i], atoms[j], d, ref, "too_short"))
        elif d > hi * ref:
            issues.append(BondIssue(i, j, atoms[i], atoms[j], d, ref, "too_long"))
    return issues

# ── Section B — non-bonded clashes ───────────────────────────────────────────

def inspect_clashes(
    atoms: List[str],
    coords: List[Vec3],
    bond_list: List[Tuple[int, int]],
    vdw_scale: float = 0.70,
) -> List[ClashIssue]:
    """Flag non-bonded pairs closer than vdw_scale × (sum of vdW radii).

    Excludes 1–2 (bonded) and 1–3 (share one bond) pairs to avoid false positives
    in normal valence geometry.
    """
    n = len(atoms)
    bonded_set = set(bond_list) | {(j, i) for i, j in bond_list}
    adj = bonds_to_adj(n, bond_list)

    one_three: set = set()
    for k in range(n):
        nbrs = adj[k]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                pair = (min(nbrs[a], nbrs[b]), max(nbrs[a], nbrs[b]))
                one_three.add(pair)

    issues: List[ClashIssue] = []
    for i in range(n):
        ri = VDW_RAD.get(atoms[i], 1.70)
        for j in range(i + 1, n):
            if (i, j) in bonded_set or (i, j) in one_three:
                continue
            rj = VDW_RAD.get(atoms[j], 1.70)
            threshold = vdw_scale * (ri + rj)
            d = _dist(coords[i], coords[j])
            if d < threshold:
                issues.append(ClashIssue(i, j, atoms[i], atoms[j], d, threshold))
    return issues

# ── Section C — bond angles ───────────────────────────────────────────────────

def _angle_range(el: str, degree: int) -> Tuple[float, float]:
    """Expected (lo, hi) angle range for atom el with given bond degree.

    O and S with degree 2 are always sp3-like (bent) due to lone pairs — they
    are never linear regardless of degree. Only C with degree 2 is treated as
    sp (acetylene, nitrile, CO2 central carbon).
    """
    if el in METALS:
        return (60.0, 180.0)   # checked separately per coordination geometry
    if el == "C":
        if degree >= 4: return (85.0, 135.0)   # sp3
        if degree == 3: return (100.0, 140.0)  # sp2
        if degree == 2: return (155.0, 180.0)  # sp / linear (CO2, nitrile, acetylene)
    if el == "N":
        if degree >= 3: return (85.0, 135.0)   # sp3 amine / sp2 amide — use wide range
        if degree == 2: return (100.0, 140.0)  # sp2 imine; don't assume linear
    if el in ("O", "S"):
        return (85.0, 135.0)   # always sp3-like (water 104.5°, ether ~111°)
    if el == "P":
        return (85.0, 135.0)
    return (80.0, 145.0)  # fallback

def inspect_angles(
    atoms: List[str],
    coords: List[Vec3],
    bond_list: List[Tuple[int, int]],
) -> List[AngleIssue]:
    """Check all bond angles against expected ranges inferred from connectivity."""
    adj = bonds_to_adj(len(atoms), bond_list)
    issues: List[AngleIssue] = []
    for j, el_j in enumerate(atoms):
        nbrs = adj[j]
        if len(nbrs) < 2:
            continue
        lo, hi = _angle_range(el_j, len(nbrs))
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                ang = _angle_deg(coords[i], coords[j], coords[k])
                if ang < lo or ang > hi:
                    issues.append(AngleIssue(i, j, k, el_j, ang, lo, hi))
    return issues

# ── Section D — torsion / conformation ───────────────────────────────────────

def inspect_torsions(
    atoms: List[str],
    coords: List[Vec3],
    bond_list: List[Tuple[int, int]],
    eclipsed_thresh: float = 10.0,
) -> List[TorsionIssue]:
    """Flag eclipsed sp3–sp3 bonds (dihedral near 0°).

    For each rotatable heavy-atom bond, picks the first heavy (non-H) neighbor
    on each side; falls back to H if no heavy neighbor exists.
    Reports at most one issue per bond.
    """
    adj = bonds_to_adj(len(atoms), bond_list)
    issues: List[TorsionIssue] = []
    for j, k in bond_list:
        if atoms[j] in METALS or atoms[k] in METALS:
            continue
        # prefer heavy-atom neighbors for the dihedral reference
        def _pick_nbr(center: int, exclude: int) -> Optional[int]:
            heavy = [nb for nb in adj[center] if nb != exclude and atoms[nb] != "H"]
            if heavy:
                return heavy[0]
            fallback = [nb for nb in adj[center] if nb != exclude]
            return fallback[0] if fallback else None

        i = _pick_nbr(j, k)
        l = _pick_nbr(k, j)
        if i is None or l is None:
            continue
        dih = abs(_dihedral_deg(coords[i], coords[j], coords[k], coords[l]))
        if dih < eclipsed_thresh:
            issues.append(TorsionIssue(i, j, k, l, dih, "eclipsed"))
    return issues

# ── Section E — charge and spin consistency ───────────────────────────────────

def inspect_charge_spin(
    atoms: List[str],
    charge: int,
    multiplicity: int,
) -> List[str]:
    """Check that total electron count is consistent with charge and multiplicity."""
    issues: List[str] = []
    total_e = sum(ATOMIC_NUM.get(el, 0) for el in atoms) - charge
    if total_e <= 0:
        issues.append(
            f"Total electron count is {total_e} with charge={charge} — charge may be wrong"
        )
        return issues
    # Parity rule: odd electron count → even multiplicity (2,4,6,...);
    #              even electron count → odd multiplicity (1,3,5,...).
    odd = total_e % 2 == 1
    if odd and multiplicity % 2 != 0:
        # odd electrons need even multiplicity (doublet=2, quartet=4, ...)
        issues.append(
            f"Odd electron count ({total_e}) requires even multiplicity (2, 4, …), "
            f"but multiplicity={multiplicity} (odd) is given — molecule must be open-shell (radical)"
        )
    elif not odd and multiplicity % 2 == 0:
        # even electrons need odd multiplicity (singlet=1, triplet=3, ...)
        issues.append(
            f"Even electron count ({total_e}) requires odd multiplicity (1, 3, …), "
            f"but multiplicity={multiplicity} (even) is given"
        )
    return issues

# ── Section F — QM-specific hazards ──────────────────────────────────────────

_EXPECTED_CN: Dict[str, Tuple[int, int]] = {
    "Cu": (4, 6), "Fe": (4, 6), "Ni": (4, 6), "Pt": (4, 4),
    "Zn": (4, 6), "Co": (4, 6), "Ru": (4, 6), "Pd": (4, 4),
}

def inspect_qm_hazards(
    atoms: List[str],
    coords: List[Vec3],
    bond_list: List[Tuple[int, int]],
) -> List[Tuple[str, str, List[int]]]:
    """Return list of (code, message, atom_indices) for QM-specific hazards."""
    n = len(atoms)
    adj = bonds_to_adj(n, bond_list)
    hazards: List[Tuple[str, str, List[int]]] = []

    # F1: near-linear bond angle (> 175°) — ORCA internal coord issue
    for j in range(n):
        nbrs = adj[j]
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                i, k = nbrs[a], nbrs[b]
                ang = _angle_deg(coords[i], coords[j], coords[k])
                if ang > 175.0:
                    hazards.append((
                        "F1",
                        f"Near-linear angle {ang:.1f}° at atom {j} ({atoms[j]}) "
                        f"between atoms {i} and {k} — add %geom AngleConstraint or use Cartesian coords",
                        [i, j, k],
                    ))

    # F2: duplicate coordinates
    for i in range(n):
        for j in range(i + 1, n):
            if _dist(coords[i], coords[j]) < 0.01:
                hazards.append((
                    "F2",
                    f"Duplicate coordinates: atoms {i} ({atoms[i]}) and {j} ({atoms[j]}) "
                    f"are {_dist(coords[i], coords[j]):.4f} Å apart",
                    [i, j],
                ))

    # F4: likely missing H (organic atom with fewer bonds than expected valence)
    _expected_bonds = {"C": 4, "N": 3, "O": 2, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
    for i, el in enumerate(atoms):
        if el not in _expected_bonds:
            continue
        expected = _expected_bonds[el]
        actual = len(adj[i])
        if actual < expected - 1:
            hazards.append((
                "F4",
                f"Atom {i} ({el}) has {actual} bond(s), expected ~{expected} — possible missing H",
                [i],
            ))

    # F5: metal coordination number out of expected range
    for i, el in enumerate(atoms):
        if el not in _EXPECTED_CN:
            continue
        lo, hi = _EXPECTED_CN[el]
        cn = len(adj[i])
        if cn < lo or cn > hi:
            hazards.append((
                "F5",
                f"Metal atom {i} ({el}) has coordination number {cn}, "
                f"expected {lo}–{hi}",
                [i],
            ))

    # F6: disconnected molecular graph
    if n > 0:
        visited: set = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(nb for nb in adj[node] if nb not in visited)
        if len(visited) < n:
            unconnected = [i for i in range(n) if i not in visited]
            hazards.append((
                "F6",
                f"Disconnected graph: {len(unconnected)} atom(s) not connected to the main fragment",
                unconnected,
            ))

    return hazards

# ── Top-level inspector ───────────────────────────────────────────────────────

def inspect_geometry(
    atoms: List[str],
    coords: List[Vec3],
    charge: int = 0,
    multiplicity: int = 1,
    bond_list: Optional[List[Tuple[int, int]]] = None,
) -> InspectionReport:
    """Run all geometry inspection checks.

    Parameters
    ----------
    atoms, coords : parsed from XYZ, SDF, mol2, or exyz
    charge, multiplicity : for Section E (charge/spin consistency)
    bond_list : explicit (i,j) bond pairs with i<j.
        - Pass None to infer bonds from coordinates (XYZ/exyz input — Section 0 runs).
        - Pass a list to skip inference (SDF/mol2/SMILES input — Section 0 skipped).
          For SMILES, parse the SMILES externally (e.g. with RDKit) to get the bond list,
          then pass it here alongside the XYZ coordinates.
    """
    if bond_list is None:
        bond_list = infer_bond_list(atoms, coords)

    return InspectionReport(
        n_atoms=len(atoms),
        n_bonds=len(bond_list),
        bond_issues=inspect_bonds(atoms, coords, bond_list),
        clash_issues=inspect_clashes(atoms, coords, bond_list),
        angle_issues=inspect_angles(atoms, coords, bond_list),
        torsion_issues=inspect_torsions(atoms, coords, bond_list),
        charge_spin_issues=inspect_charge_spin(atoms, charge, multiplicity),
        qm_hazards=inspect_qm_hazards(atoms, coords, bond_list),
    )


def parse_xyz_flexible(xyz: str) -> Tuple[List[str], List[Vec3]]:
    """Accept XYZ with or without header. Returns atoms + coords."""
    atoms, coords, _ = parse_xyz_with_tags(xyz)
    return atoms, coords


def parse_xyz_with_tags(xyz: str) -> Tuple[List[str], List[Vec3], List[Optional[str]]]:
    """Accept XYZ with or without header. Returns (atoms, coords, tags).

    Tags are read from the 5th token on each atom line; a leading '#' is stripped.
    Atoms with no 5th token have tag=None.

    Tagged XYZ examples (both forms accepted):
        O   1.2  0.0  0.0  O_target
        O  -0.6  1.04 0.0  # O_hydroxyl
    """
    if xyz is None or not isinstance(xyz, str) or not xyz.strip():
        raise ValueError("xyz is empty/None")

    lines = [ln.strip() for ln in xyz.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("xyz has no lines")

    # Skip natoms + comment header if present
    try:
        nat = int(lines[0])
        if len(lines) >= nat + 2:
            lines = lines[2:2 + nat]
    except Exception:
        pass

    atoms: List[str] = []
    coords: List[Vec3] = []
    tags: List[Optional[str]] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        tag: Optional[str] = None
        if len(parts) >= 5:
            # Join everything after x y z, strip leading '#' (with or without space)
            rest = ' '.join(parts[4:]).lstrip('#').strip()
            tag = rest.split()[0] if rest else None
        atoms.append(sym)
        coords.append((x, y, z))
        tags.append(tag)
    if not atoms:
        raise ValueError("failed to parse any atoms")
    return atoms, coords, tags

def format_xyz(atoms: List[str], coords: List[Vec3], comment: str = "") -> str:
    if len(atoms) != len(coords):
        raise ValueError("atoms/coords length mismatch")
    out = [str(len(atoms)), comment]
    for s, (x,y,z) in zip(atoms, coords):
        out.append(f"{s:<2} {x: .10f} {y: .10f} {z: .10f}")
    return "\n".join(out) + "\n"

def format_xyz_no_header(atoms: List[str], coords: List[Vec3]) -> str:
    """Return atom lines only (no natoms/comment header) for GeometryRegistry compatibility."""
    if len(atoms) != len(coords):
        raise ValueError("atoms/coords length mismatch")
    lines = []
    for s, (x,y,z) in zip(atoms, coords):
        lines.append(f"{s} {x: .10f} {y: .10f} {z: .10f}")
    return "\n".join(lines)

def infer_bonds(atoms: List[str], coords: List[Vec3], scale: float = 1.25) -> List[List[int]]:
    """Very simple bond inference by covalent radii sum * scale."""
    n = len(atoms)
    adj = [[] for _ in range(n)]
    for i in range(n):
        ri = COV_RAD.get(atoms[i], 0.77)
        for j in range(i+1, n):
            rj = COV_RAD.get(atoms[j], 0.77)
            cutoff = scale * (ri + rj)
            if _dist(coords[i], coords[j]) <= cutoff:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def find_heavy_neighbor_for_H(
    atoms: List[str],
    coords: List[Vec3],
    h_index: int,
    max_distance: float = 1.3,
) -> Optional[int]:
    h_coord = coords[h_index]
    best_j = None
    best_d = 1e9
    for j, (sym_j, coord_j) in enumerate(zip(atoms, coords)):
        if j == h_index or sym_j == "H":
            continue
        d = _dist(h_coord, coord_j)
        if d < best_d:
            best_d = d
            best_j = j
    if best_j is not None and best_d <= max_distance:
        return best_j
    return None

def find_acidic_H_by_distance(atoms: List[str], coords: List[Vec3]) -> List[int]:
    """Return indices of H that are nearest to O/N/S and within a cutoff."""
    acidic = []
    for i, sym in enumerate(atoms):
        if sym != "H":
            continue
        j = find_heavy_neighbor_for_H(atoms, coords, i)
        if j is None:
            continue
        if atoms[j] in ("O", "N", "S"):
            acidic.append(i)
    return acidic

def pick_deprotonation_site(atoms: List[str], coords: List[Vec3]) -> Tuple[int, int, str]:
    """
    Returns (h_index, heavy_index, reason).
    Preference: O–H > S–H > N–H (simple heuristic).
    """
    cand = []
    for h in find_acidic_H_by_distance(atoms, coords):
        heavy = find_heavy_neighbor_for_H(atoms, coords, h)
        if heavy is None:
            continue
        score = {"O": 3, "S": 2, "N": 1}.get(atoms[heavy], 0)
        cand.append((score, h, heavy))
    if not cand:
        # fallback: any H bound to a non-metal heavy atom
        for i,s in enumerate(atoms):
            if s != "H":
                continue
            heavy = find_heavy_neighbor_for_H(atoms, coords, i, max_distance=1.35)
            if heavy is not None and atoms[heavy] in NONMETALS and atoms[heavy] != "H":
                cand.append((0, i, heavy))
    if not cand:
        raise ValueError("No removable H found by geometry. Need user site or population fallback.")
    cand.sort(reverse=True)
    score, h, heavy = cand[0]
    return h, heavy, f"picked H index {h} attached to {atoms[heavy]} (score={score})"

def pick_protonation_site(atoms: List[str], coords: List[Vec3]) -> Tuple[int, str]:
    """
    Returns (target_atom_index, reason). Simple: choose hetero atom O > N > S with 'available' valence.
    Uses bond inference to estimate coordination.
    """
    adj = infer_bonds(atoms, coords)
    cand = []
    for i, sym in enumerate(atoms):
        if sym not in ("O", "N", "S"):
            continue
        # crude coordination count
        deg = len(adj[i])
        # prefer less-coordinated atoms (more likely to have lone pair protonation)
        # and prefer O/N over S
        base = {"O": 3, "N": 2, "S": 1}[sym]
        score = base * 10 - deg
        cand.append((score, i))
    if not cand:
        raise ValueError("No obvious protonation site (O/N/S) found by geometry. Need user site or population fallback.")
    cand.sort(reverse=True)
    score, i = cand[0]
    return i, f"picked {atoms[i]} atom index {i} (deg~{len(infer_bonds(atoms, coords)[i])}, score={score})"


_SEMANTIC_SELECTORS = {
    "oxygen_terminal", "oxygen_terminal_not_hydroxyl", "o_terminal", "o_nonhydroxyl",
    "oxygen_hydroxyl", "o_hydroxyl", "oh",
    "nitrogen", "n_atom", "n",
    "alpha_carbon", "alpha_to_carbonyl", "alpha_h", "alpha_ch", "carbonyl_alpha", "alpha_c",
}


def pick_protonation_site_by_selector(
    atoms: List[str],
    coords: List[Vec3],
    tags: List[Optional[str]],
    site_selector: str,
    variant: int = 0,
) -> Tuple[int, str]:
    """Resolve a site_selector to an atom index.

    Three selector kinds (tried in order):

    1. Direct — line number:
         "line:N"   →  1-based atom line N  (e.g. "line:2" = second atom, index 1)

    2. Direct — atom tag:
         Any string that does NOT match a semantic name is looked up in the tags
         extracted from the 5th XYZ column (case-insensitive, leading '@' stripped).
         Tagged XYZ:  O  1.2  0.0  0.0  O_target
                      O -0.6  1.04 0.0  # O_hydroxyl

    3. Semantic — bond-inference:
         "oxygen_terminal" / "o_terminal" / "oxygen_terminal_not_hydroxyl" / "o_nonhydroxyl"
             → O atoms NOT bonded to any H
         "oxygen_hydroxyl" / "o_hydroxyl" / "oh"
             → O atoms bonded to at least one H
         "nitrogen" / "n_atom" / "n"
             → N atoms

    variant: 0-based index when multiple atoms match (sorted by atom index).
    Returns (atom_index, reason_string).
    """
    # --- Kind 1: line:N ---
    if site_selector.lower().startswith("line:"):
        try:
            n = int(site_selector.split(":", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid line selector '{site_selector}'. Use 'line:N' with N as a 1-based integer."
            )
        idx = n - 1
        if idx < 0 or idx >= len(atoms):
            raise ValueError(
                f"line:{n} is out of range (geometry has {len(atoms)} atoms, valid: 1–{len(atoms)})."
            )
        return idx, f"line:{n} → {atoms[idx]} atom index {idx}"

    sel = site_selector.lower().strip()

    # --- Kind 2: atom tag (any unrecognised string) ---
    if sel not in _SEMANTIC_SELECTORS:
        tag_query = site_selector.lstrip('@').strip()
        # "*" matches any non-empty tag (first tagged atom, or variant-th)
        if tag_query == "*":
            matches = [i for i, t in enumerate(tags) if t is not None]
        else:
            matches = [
                i for i, t in enumerate(tags)
                if t is not None and t.lower() == tag_query.lower()
            ]
        if matches:
            matches.sort()
            if variant >= len(matches):
                raise ValueError(
                    f"variant={variant} out of range: only {len(matches)} atom(s) "
                    f"tagged '{tag_query}'."
                )
            idx = matches[variant]
            return idx, f"atom tag '{tag_query}' variant={variant} → {atoms[idx]} index {idx}"
        raise ValueError(
            f"Unknown site_selector '{site_selector}'. "
            "Direct: 'line:N' (1-based), '*' (first tagged atom), or a named atom tag "
            "in the XYZ 5th column. "
            "Semantic: 'oxygen_terminal', 'oxygen_hydroxyl', 'nitrogen', 'alpha_carbon'."
        )

    # --- Kind 3: semantic ---
    adj = infer_bonds(atoms, coords)
    candidates: List[int] = []

    if sel in ("oxygen_terminal", "oxygen_terminal_not_hydroxyl", "o_terminal", "o_nonhydroxyl"):
        for i, sym in enumerate(atoms):
            if sym == "O" and not any(atoms[j] == "H" for j in adj[i]):
                candidates.append(i)

    elif sel in ("oxygen_hydroxyl", "o_hydroxyl", "oh"):
        for i, sym in enumerate(atoms):
            if sym == "O" and any(atoms[j] == "H" for j in adj[i]):
                candidates.append(i)

    elif sel in ("nitrogen", "n_atom", "n"):
        for i, sym in enumerate(atoms):
            if sym == "N":
                candidates.append(i)

    elif sel in ("alpha_carbon", "alpha_to_carbonyl", "alpha_h", "alpha_ch",
                 "carbonyl_alpha", "alpha_c"):
        # Carbonyl-C: carbon bonded to at least one terminal O (no H on that O)
        carbonyl_c_set = set()
        for i, sym in enumerate(atoms):
            if sym == "C":
                for j in adj[i]:
                    if atoms[j] == "O" and not any(atoms[k] == "H" for k in adj[j]):
                        carbonyl_c_set.add(i)
                        break
        # Alpha-H: H bonded to a C that is adjacent to at least one carbonyl-C
        for i, sym in enumerate(atoms):
            if sym == "H":
                for j in adj[i]:
                    if atoms[j] == "C" and any(k in carbonyl_c_set for k in adj[j]):
                        candidates.append(i)
                        break

    if not candidates:
        raise ValueError(
            f"No atoms found matching site_selector='{site_selector}'."
        )
    candidates.sort()
    if variant >= len(candidates):
        raise ValueError(
            f"variant={variant} out of range: only {len(candidates)} candidate(s) "
            f"for site_selector='{site_selector}'."
        )
    idx = candidates[variant]
    return idx, (
        f"site_selector='{site_selector}' variant={variant}: "
        f"{atoms[idx]} index {idx} ({len(candidates)} candidate(s))"
    )

def place_H_on_atom(atoms: List[str], coords: List[Vec3], atom_index: int) -> Vec3:
    """Place H along direction opposite to neighbor vectors (rough lone-pair direction)."""
    sym = atoms[atom_index]
    bl = BOND_LEN.get(sym, 1.00)
    adj = infer_bonds(atoms, coords)
    center = coords[atom_index]
    if not adj[atom_index]:
        direction = (0.0, 0.0, 1.0)
    else:
        v = (0.0, 0.0, 0.0)
        for j in adj[atom_index]:
            u = _unit(_vsub(coords[j], center))  # atom -> neighbor
            v = _vadd(v, u)
        direction = _unit(_vmul(v, -1.0))  # away from neighbors
    return _vadd(center, _vmul(direction, bl))

def remove_atom(atoms: List[str], coords: List[Vec3], idx: int) -> Tuple[List[str], List[Vec3]]:
    return atoms[:idx] + atoms[idx+1:], coords[:idx] + coords[idx+1:]

def structure_proton_edit(
    xyz: str,
    mode: Literal["add", "remove"],
    charge: int,
    multiplicity: int,
    # optional explicit site
    site_selector: Optional[str] = None,
    variant: int = 0,
    h_index: Optional[int] = None,
    target_atom_index: Optional[int] = None,
    geometry_name: Optional[str] = None,
    strategy: Literal["auto", "distance"] = "auto",
) -> Dict:
    atoms, coords, tags = parse_xyz_with_tags(xyz)

    if mode == "remove":
        if h_index is not None:
            heavy_index = find_heavy_neighbor_for_H(atoms, coords, h_index) or -1
            reason = f"user-selected H index {h_index} (heavy={heavy_index})"
        elif site_selector and site_selector.lower() not in ("auto", ""):
            resolved_idx, sel_reason = pick_protonation_site_by_selector(
                atoms, coords, tags, site_selector, variant
            )
            if atoms[resolved_idx] == "H":
                h_index = resolved_idx
                heavy_index = find_heavy_neighbor_for_H(atoms, coords, h_index) or -1
                reason = f"{sel_reason} (H atom, removed directly)"
            else:
                # Heavy atom: find the bonded H to remove
                adj = infer_bonds(atoms, coords)
                h_candidates = sorted(j for j in adj[resolved_idx] if atoms[j] == "H")
                if not h_candidates:
                    raise ValueError(
                        f"site_selector='{site_selector}' resolved to {atoms[resolved_idx]} "
                        f"index {resolved_idx} but it has no bonded H to remove."
                    )
                h_index = h_candidates[0]
                heavy_index = resolved_idx
                reason = f"{sel_reason} → remove H index {h_index} on {atoms[resolved_idx]}"
        else:
            h_index, heavy_index, reason = pick_deprotonation_site(atoms, coords)
        if atoms[h_index] != "H":
            raise ValueError(f"h_index {h_index} is not H (got {atoms[h_index]})")
        atoms2, coords2 = remove_atom(atoms, coords, h_index)
        return {
            "status": "ok",
            "mode": "remove",
            "chosen_h_index": h_index,
            "heavy_neighbor_index": heavy_index,
            "reason": reason,
            "old_charge": charge,
            "new_charge": charge - 1,
            "old_multiplicity": multiplicity,
            "new_multiplicity": multiplicity,
            "geometry_xyz": format_xyz_no_header(atoms2, coords2),
            "provenance": {"geometry": "proton_edit_remove"},
        }

    if mode == "add":
        if target_atom_index is not None:
            reason = f"user-selected atom index {target_atom_index}"
        elif site_selector and site_selector.lower() not in ("auto", ""):
            target_atom_index, reason = pick_protonation_site_by_selector(
                atoms, coords, tags, site_selector, variant
            )
        else:
            target_atom_index, reason = pick_protonation_site(atoms, coords)
        if atoms[target_atom_index] == "H":
            raise ValueError("target_atom_index cannot be H")
        h_pos = place_H_on_atom(atoms, coords, target_atom_index)
        atoms2 = atoms + ["H"]
        coords2 = coords + [h_pos]
        return {
            "status": "ok",
            "mode": "add",
            "geometry_xyz": format_xyz_no_header(atoms2, coords2),
            "provenance": {"geometry": "proton_edit_add"},
            "old_charge": charge,
            "new_charge": charge + 1,
            "old_multiplicity": multiplicity,
            "new_multiplicity": multiplicity,
            "site_selector": site_selector,
            "reason": reason,
        }

    raise ValueError(f"Unknown mode: {mode}")
