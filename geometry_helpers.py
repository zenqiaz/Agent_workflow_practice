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
            "Semantic: 'oxygen_terminal', 'oxygen_hydroxyl', 'nitrogen'."
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
