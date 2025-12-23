from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple, Optional, Dict, Literal
import json

Vec3 = Tuple[float, float, float]

COV_RAD = {  # Å, rough
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
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

def parse_xyz_flexible(xyz: str) -> Tuple[List[str], List[Vec3]]:
    """Accept XYZ with or without header. Returns atoms + coords."""
    if xyz is None or not isinstance(xyz, str) or not xyz.strip():
        raise ValueError("xyz is empty/None")

    lines = [ln.strip() for ln in xyz.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("xyz has no lines")

    # Try XYZ header
    start = 0
    try:
        nat = int(lines[0])
        if len(lines) >= nat + 2:
            start = 2
            lines = lines[start:start+nat]
    except Exception:
        pass

    atoms: List[str] = []
    coords: List[Vec3] = []
    for ln in lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(sym)
        coords.append((x, y, z))
    if not atoms:
        raise ValueError("failed to parse any atoms")
    return atoms, coords

def format_xyz(atoms: List[str], coords: List[Vec3], comment: str = "") -> str:
    if len(atoms) != len(coords):
        raise ValueError("atoms/coords length mismatch")
    out = [str(len(atoms)), comment]
    for s, (x,y,z) in zip(atoms, coords):
        out.append(f"{s:<2} {x: .10f} {y: .10f} {z: .10f}")
    return "\n".join(out) + "\n"

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
    h_index: Optional[int] = None,
    target_atom_index: Optional[int] = None,
    geometry_name: Optional[str] = None,
    strategy: Literal["auto", "distance"] = "auto",
) -> Dict:
    atoms, coords = parse_xyz_flexible(xyz)

    if mode == "remove":
        if h_index is None:
            h_index, heavy_index, reason = pick_deprotonation_site(atoms, coords)
        else:
            heavy_index = find_heavy_neighbor_for_H(atoms, coords, h_index) or -1
            reason = f"user-selected H index {h_index} (heavy={heavy_index})"
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
            "xyz": format_xyz(atoms2, coords2, comment="deprotonated"),
        }

    if mode == "add":
        if target_atom_index is None:
            target_atom_index, reason = pick_protonation_site(atoms, coords)
        else:
            reason = f"user-selected atom index {target_atom_index}"
        if atoms[target_atom_index] == "H":
            raise ValueError("target_atom_index cannot be H")
        h_pos = place_H_on_atom(atoms, coords, target_atom_index)
        atoms2 = atoms + ["H"]
        coords2 = coords + [h_pos]
        # inside structure_proton_edit result:
        return {
            "status": "ok",
            "geometry_xyz": xyz_to_no_header(full_xyz_string),
            "provenance": {"geometry": "proton_edit"},
            "edit": {
                "mode": mode,
                "reason": reason,
                "chosen_site": {...},
                "old_charge": charge,
                "new_charge": new_charge,
                "old_multiplicity": multiplicity,
                "new_multiplicity": multiplicity,
            }
        }

    raise ValueError(f"Unknown mode: {mode}")
