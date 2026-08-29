# client_helpers.py
from __future__ import annotations
import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, List
from dataclasses import dataclass, field
import hashlib
import uuid
import requests, time
import json
from datetime import datetime, timezone
# --- State access helpers -----------------------------------------------------
def _state_get(state: Any, key: str, default: Any = None) -> Any:
    """Get a value from AgentState supporting both dict-style (TypedDict) and attribute-style (dataclass)."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _state_set(state: Any, key: str, value: Any) -> None:
    """Set a value on AgentState supporting both dict-style (TypedDict) and attribute-style (dataclass)."""
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def _state_get_dict(state: Any, key: str) -> Dict[str, Any]:
    v = _state_get(state, key, {})
    return v if isinstance(v, dict) else {}

NEEDS_GEOM_SINGLE = {'run_solvator_cluster_thermo', 'run_opt_job', 'run_nbo_job', 'run_sp_energy', 'run_freq_job', 'run_spectrum_job', 'run_tddft_job', 'run_scan_job', 'run_ts_opt_job', 'run_solvator_cluster', 'structure_add_remove_proton', 'run_casscf_job'}

TOOLS_RETURNING_STRUCTURE = {
    "name_to_geometry_xyz",
    "build_coordination_complex",
    "build_dimer_xyz",
    "set_geometry_xyz",
    "run_opt_job",
    "structure_add_remove_proton",
    "run_solvator_cluster_thermo",  # if it returns cluster geometry
    "run_solvator_cluster",         # if it returns cluster geometry
}



# --- Geometry model (new) ----------------------------------------------------
METHOD_KEYWORDS = {
    "b3lyp": "B3LYP",
    "pbe0": "PBE0",
    "pbeh-3c": "PBEh-3c",
    "tpssh": "TPSSh",
    "bp86": "BP86",
    "pw6b95-d3": "PW6B95-D3",
}

BASIS_KEYWORDS = {
    "def2-svp": "def2-SVP",
    "def2-tzvp": "def2-TZVP",
    "def2-tzvpp": "def2-TZVPP",
    "def2-qzvp": "def2-QZVP",
}

JOBTYPE_KEYWORDS = {
    "opt": "opt",
    "optimize": "opt",
    "optimization": "opt",
    "geometry optimization": "opt",
}

NEEDS_GEOM = {"run_opt_job", "run_nbo_job", "run_sp_energy"}

def extract_orca_hints(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()
    hints: Dict[str, Any] = {}

    for key, val in METHOD_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["method"] = val
            break

    for key, val in BASIS_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["basis"] = val
            break

    for key in JOBTYPE_KEYWORDS:
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["job_type"] = "opt"
            break

    return hints

def load_xyz_as_geometry(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [ln.rstrip("\n\r") for ln in f.readlines()]

    if not raw_lines:
        raise ValueError(f"XYZ file {path!r} is empty")

    natoms = None
    first_tokens = raw_lines[0].split()
    if first_tokens:
        try:
            natoms = int(first_tokens[0])
        except ValueError:
            natoms = None

    if natoms is not None and len(raw_lines) >= natoms + 2:
        coord_lines = raw_lines[2 : 2 + natoms]
    else:
        coord_lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith(("#", "!", "//"))]

    if not coord_lines:
        raise ValueError(f"No coordinate lines found in {path!r}")

    return "\n".join(coord_lines)
'''
def _ensure_geom_registry(state: Dict[str, Any]) -> GeometryRegistry:
    """Ensure GeometryRegistry backing keys exist on dict state.

    New architecture:
      - no GeometryRegistry object is stored in state
      - truth lives in state["geometries"], ["geom_meta"], ["name_to_geom"], ["current_geom"]

    If a legacy state["geom_registry"] object exists, best-effort migrate its contents
    into the flat keys, then drop it.
    """
    if not isinstance(state, dict):
        raise TypeError(f"_ensure_geom_registry expects dict state, got {type(state).__name__}")

    # 0) Best-effort migrate legacy object-in-state -> flat keys
    legacy = state.get("geom_registry")
    if legacy is not None and legacy is not state:
        # try to detect a legacy registry-like object
        geoms = getattr(legacy, "geometries", None)
        meta = getattr(legacy, "geom_meta", None)
        name_map = getattr(legacy, "name_to_geom", None)
        cur = getattr(legacy, "current_geom", None)

        if isinstance(geoms, dict) or isinstance(meta, dict) or isinstance(name_map, dict):
            state.setdefault("geometries", {})
            state.setdefault("geom_meta", {})
            state.setdefault("name_to_geom", {})
            state.setdefault("current_geom", None)

            if isinstance(geoms, dict):
                # legacy may already be headerless; we keep as-is
                state["geometries"].update(geoms)
            if isinstance(meta, dict):
                state["geom_meta"].update(meta)
            if isinstance(name_map, dict):
                state["name_to_geom"].update(name_map)
            if isinstance(cur, str) and cur:
                state["current_geom"] = cur

            # stop keeping the object around
            try:
                del state["geom_registry"]
            except Exception:
                state["geom_registry"] = None

    # 1) Ensure backing keys exist (GeometryRegistry wrapper will also ensure)
    reg = GeometryRegistry(state)
    return reg
'''
def summarize_geometries_prompt(state: Dict[str, Any]) -> str:
    reg = get_geometry_registry(state)

    if not reg.geometries:
        return "No geometries are loaded."

    cur = reg.current_geom
    parts = ["Loaded geometries (geom_id, name, q, mult):"]

    # Build reverse alias map: geom_id -> [names...]
    rev_names: Dict[str, list] = {}
    for nm, gid in (reg.name_to_geom or {}).items():
        if isinstance(nm, str) and nm and isinstance(gid, str) and gid:
            rev_names.setdefault(gid, []).append(nm)

    for gid in sorted(reg.geometries.keys()):
        meta = (reg.geom_meta.get(gid) or {}) if isinstance(reg.geom_meta, dict) else {}
        nm = meta.get("name")
        if not (isinstance(nm, str) and nm.strip()):
            aliases = rev_names.get(gid) or []
            nm = aliases[0] if aliases else "-"
        chg = meta.get("charge", None)
        mult = meta.get("multiplicity", None)
        mark = "  (current)" if (cur and gid == cur) else ""
        xyz = reg.geometries.get(gid, "")
        natoms_str = ""
        try:
            if xyz:
                coord_lines = [l for l in xyz.strip().splitlines() if l.strip() and l.split()[0].isalpha()]
                n_atoms = len(coord_lines)
                # Group by element, preserving index order: e.g. "C[0-5] H[6-11]"
                from collections import OrderedDict as _OD
                groups: _OD = _OD()
                for idx, l in enumerate(coord_lines):
                    el = l.split()[0]
                    groups.setdefault(el, []).append(idx)
                def _fmt_range(idxs):
                    if len(idxs) == 1:
                        return f"{idxs[0]}"
                    return f"{idxs[0]}-{idxs[-1]}"
                atom_str = " ".join(f"{el}[{_fmt_range(idxs)}]" for el, idxs in groups.items())
                natoms_str = f"  n_atoms={n_atoms}  atoms: {atom_str}"
        except Exception:
            pass
        parts.append(f"- {gid}: {nm}  q={chg}, mult={mult}{natoms_str}{mark}")

    parts.append(
        f"Defaults (fallback only): charge={_state_get(state,'default_charge',0)}, "
        f"multiplicity={_state_get(state,'default_multiplicity',1)}."
    )
    parts.append("Tool nodes should reference geometries via node_spec.input_id/output_id (geom_ids).")
    return "\n".join(parts)


def geom_key_from_path(path: str) -> str:
    return os.path.basename(path)

import json, time, urllib.parse, requests

OPSIN = "https://opsin.ch.cam.ac.uk/opsin/"

def opsin_resolve(name: str, timeout=15):
    url = OPSIN + urllib.parse.quote(name) + ".json"
    r = requests.get(url, timeout=timeout)
    data = r.json()  # OPSIN always returns JSON, even on failure 
    if data.get("status") == "SUCCESS":
        return {
            "source": "opsin",
            "name": name,
            "smiles": data.get("smiles"),
            "inchi": data.get("stdinchi") or data.get("inchi"),
            "inchikey": data.get("stdinchikey"),
        }
    return None

def pubchem_name_to_cid(name: str, timeout=15, retries=3):
    # canonical PUG-REST prolog described in cookbook
    import time as _time
    prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{prolog}/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                if attempt < retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                return None
            data = r.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            return int(cids[0]) if cids else None
        except requests.RequestException:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
            else:
                return None
    return None

def pubchem_cid_to_props(cid: int, timeout=15):
    prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    props = "property/IsomericSMILES,InChI,InChIKey,IUPACName/JSON"
    url = f"{prolog}/compound/cid/{cid}/{props}"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json()
    rows = data.get("PropertyTable", {}).get("Properties", [])
    return rows[0] if rows else None

def resolve_name_to_identifiers(name: str):
    name = name.strip()
    if not name:
        return {"status":"error","error":"empty name"}

    hit = opsin_resolve(name)
    if hit:
        return {"status":"ok", **hit}

    time.sleep(0.25)  # be polite to PubChem 
    cid = pubchem_name_to_cid(name)
    if cid is None:
        return {"status":"not_found","name":name}

    time.sleep(0.25)
    props = pubchem_cid_to_props(cid) or {}
    return {
        "status": "ok",
        "source": "pubchem",
        "name": name,
        "cid": cid,
        "smiles": props.get("IsomericSMILES"),
        "inchi": props.get("InChI"),
        "inchikey": props.get("InChIKey"),
        "iupac": props.get("IUPACName"),
    }


def fetch_pubchem_3d_sdf(cid: int, timeout=30) -> str | None:
    prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    # Try a 3D record first (uses record_type=3d) 
    url1 = f"{prolog}/compound/cid/{cid}/record/SDF?record_type=3d"
    r = requests.get(url1, timeout=timeout)
    if r.status_code == 200 and "M  END" in r.text:
        return r.text

    time.sleep(0.25)

    # Fallback: diverse conformers list + fetch first conformer in SDF 
    url_list = f"{prolog}/compound/cid/{cid}/conformers/TXT"
    r2 = requests.get(url_list, timeout=timeout)
    if r2.status_code != 200:
        return None
    conf_ids = [ln.strip() for ln in r2.text.splitlines() if ln.strip()]
    if not conf_ids:
        return None

    time.sleep(0.25)
    url_conf = f"{prolog}/conformers/{conf_ids[0]}/SDF"
    r3 = requests.get(url_conf, timeout=timeout)
    if r3.status_code == 200 and "M  END" in r3.text:
        return r3.text
    return None

from rdkit import Chem

def sdf_to_xyz_no_header(sdf_text: str) -> str:
    mol = Chem.MolFromMolBlock(sdf_text, removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("RDKit could not parse SDF")
    conf = mol.GetConformer()
    lines = []
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines)



PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def _pubchem_name_to_sdf3d(name: str, timeout=30) -> str | None:
    # Direct 3D SDF by name (no CID needed) 
    url = f"{PUBCHEM}/compound/name/{urllib.parse.quote(name)}/SDF?record_type=3d"
    r = requests.get(url, timeout=timeout)
    if r.status_code == 200 and "M  END" in r.text:
        return r.text
    return None

_ELEMENT_RE = __import__("re").compile(
    r"^([A-Z][a-z]?)(?:(\d+)?([+-]))?$"
)

# Common element names and ion names → (symbol, charge)
_ION_NAME_MAP: dict = {
    # Halide anions (multiple name variants)
    "chloride": ("Cl", -1), "chloride ion": ("Cl", -1), "chloride anion": ("Cl", -1),
    "fluoride": ("F", -1), "fluoride ion": ("F", -1), "fluoride anion": ("F", -1),
    "bromide": ("Br", -1), "bromide ion": ("Br", -1), "bromide anion": ("Br", -1),
    "iodide": ("I", -1), "iodide ion": ("I", -1), "iodide anion": ("I", -1),
    # Oxide-family anions
    "oxide": ("O", -2), "sulfide": ("S", -2),
    # Common cation names
    "ferrous": ("Fe", +2), "ferric": ("Fe", +3),
    "ferrous ion": ("Fe", +2), "ferric ion": ("Fe", +3),
    "iron(ii)": ("Fe", +2), "iron(iii)": ("Fe", +3),
    "cobalt(ii)": ("Co", +2), "cobalt(iii)": ("Co", +3),
    "nickel(ii)": ("Ni", +2), "copper(ii)": ("Cu", +2), "zinc(ii)": ("Zn", +2),
    "manganese(ii)": ("Mn", +2), "manganese(iii)": ("Mn", +3),
    # Neutral element names (common ones used in QC)
    "iron atom": ("Fe", 0), "iron": ("Fe", 0),
    "cobalt atom": ("Co", 0), "nickel atom": ("Ni", 0),
    "copper atom": ("Cu", 0), "zinc atom": ("Zn", 0),
    "chlorine atom": ("Cl", 0), "bromine atom": ("Br", 0),
    "sodium ion": ("Na", +1), "potassium ion": ("K", +1), "lithium ion": ("Li", +1),
    "magnesium ion": ("Mg", +2), "calcium ion": ("Ca", +2),
}

# Atomic numbers for elements commonly used as bare ions in QC calculations
_ATOMIC_NUMBERS: dict = {
    "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
    "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
    "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
    "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
    "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"I":53,"Cs":55,"Ba":56,
    "La":57,"Ce":58,"Gd":64,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Pb":82,
}


def _try_bare_ion_geometry(name: str):
    """Return single-atom xyz dict for bare elements/ions like 'Fe', 'Fe3+', 'Cl-', or None.

    Multiplicity is set to the minimum allowed by electron count parity:
    odd electrons → doublet (2), even electrons → singlet (1).
    """
    m = _ELEMENT_RE.match(name.strip())
    if m is None:
        return None
    elem, mag, sign = m.group(1), m.group(2), m.group(3)
    charge = 0
    if sign:
        charge = int(mag or 1) * (1 if sign == "+" else -1)
    Z = _ATOMIC_NUMBERS.get(elem)
    if Z is None:
        return None  # unknown element — fall through to PubChem
    n_electrons = Z - charge
    if n_electrons < 0:
        return None  # unphysical
    # Minimum spin multiplicity consistent with electron count parity
    mult = (n_electrons % 2) + 1
    xyz = f"{elem}  0.000  0.000  0.000"
    return {"status": "ok", "name": name, "geometry_xyz": xyz,
            "charge": charge, "multiplicity": mult,
            "provenance": {"geometry": "single_atom"}}


def _bare_ion_from_elem_charge(elem: str, charge: int) -> Optional[dict]:
    """Build single-atom geometry dict from element symbol + charge, or None if unknown."""
    Z = _ATOMIC_NUMBERS.get(elem)
    if Z is None:
        return None
    n_electrons = Z - charge
    if n_electrons < 0:
        return None
    mult = (n_electrons % 2) + 1
    xyz = f"{elem}  0.000  0.000  0.000"
    return {"status": "ok", "name": f"{elem}{charge:+d}" if charge else elem,
            "geometry_xyz": xyz, "charge": charge, "multiplicity": mult,
            "provenance": {"geometry": "single_atom"}}


def _smiles_to_xyz(smiles: str) -> str | None:
    """Convert a SMILES string to XYZ geometry (no header) via RDKit ETKDG + MMFF."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if res == -1:
            # Fallback to random coords
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res == -1:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        lines = []
        for i in range(mol.GetNumAtoms()):
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            pos = conf.GetAtomPosition(i)
            lines.append(f"{sym}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}")
        return "\n".join(lines)
    except Exception:
        return None


def _looks_like_smiles(s: str) -> bool:
    """Heuristic: SMILES strings contain = or # or [ but not spaces (except salts)."""
    import re as _re
    if " " in s:
        return False  # compound names have spaces; SMILES generally don't
    # Must contain at least one SMILES-specific character
    return bool(_re.search(r'[=#\[\]/\\@]', s))


def name_to_geometry_xyz(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        return {"status": "error", "error": "empty name"}
    default_charge = 0
    default_multiplicity = 1
    # Infer charge from ion suffix in name: "NO2+" → +1, "OH-" → -1, "Fe3+" → +3
    import re as _re
    _ion_suffix = _re.search(r'([+-])(\d*)$', name.strip())
    if _ion_suffix:
        sign = 1 if _ion_suffix.group(1) == '+' else -1
        mag  = int(_ion_suffix.group(2)) if _ion_suffix.group(2) else 1
        default_charge = sign * mag
    # 0a-pre) SMILES input: detect by SMILES-specific characters, no spaces
    if _looks_like_smiles(name):
        xyz = _smiles_to_xyz(name)
        if xyz:
            # Count formal charges from SMILES brackets to set default_charge
            import re as _re2
            charges = _re2.findall(r'\[([^\]]+)\]', name)
            inferred_charge = default_charge  # already parsed from suffix
            if inferred_charge == 0:
                # sum [O-], [N+], [Fe3+] etc. from SMILES
                for part in charges:
                    m = _re2.search(r'([+-])(\d*)$', part)
                    if m:
                        s = 1 if m.group(1) == '+' else -1
                        mag = int(m.group(2)) if m.group(2) else 1
                        inferred_charge += s * mag
            mult = 1
            if inferred_charge != 0:
                # simple guess: closed-shell default
                mult = 1
            return {
                "status": "ok",
                "name": name,
                "geometry_xyz": xyz,
                "charge": inferred_charge,
                "multiplicity": mult,
                "provenance": {"geometry": "rdkit_smiles_embed"},
            }
    # 0a) bare element symbol / simple ion notation: Fe, Fe3+, Cl-, Na+, etc.
    bare = _try_bare_ion_geometry(name)
    if bare is not None:
        return bare
    # 0b) common ion/element names: "chloride", "iron atom", "ferric ion", etc.
    name_lower = name.lower()
    ion = _ION_NAME_MAP.get(name_lower)
    if ion is not None:
        result = _bare_ion_from_elem_charge(ion[0], ion[1])
        if result is not None:
            result["name"] = name
            return result
    # 0c) "<Symbol> atom" pattern: "Fe atom", "Cl atom", etc.
    if name_lower.endswith(" atom"):
        elem_sym = name[:-5].strip()
        bare = _try_bare_ion_geometry(elem_sym)
        if bare is not None:
            return bare
    # 1) try PubChem 3D directly by name first
    sdf = _pubchem_name_to_sdf3d(name)
    if sdf:
        try:
            xyz = sdf_to_xyz_no_header(sdf)
            return {"status": "ok", "name": name, "geometry_xyz": xyz, "charge": default_charge, "multiplicity": default_multiplicity, "provenance": {"geometry": "pubchem_name_3d_sdf"}}
        except Exception as e:
            return {"status": "error", "name": name, "error": f"sdf_to_xyz failed: {e}"}

    # 2) fallback: identifiers (OPSIN / PubChem)
    ids = resolve_name_to_identifiers(name)
    cid = ids.get("cid") if isinstance(ids, dict) else None
    if cid:
        sdf = fetch_pubchem_3d_sdf(cid)
        if sdf:
            try:
                xyz = sdf_to_xyz_no_header(sdf)
                return {
                    "status": "ok",
                    "name": name,
                    "cid": cid,
                    "geometry_xyz": xyz,
                    "charge": default_charge, 
                    "multiplicity": default_multiplicity,
                    "identifiers": ids,
                    "provenance": {"geometry": "pubchem_cid_3d_sdf", "identifiers": ids.get("source")},
                }
            except Exception as e:
                return {"status": "error", "name": name, "cid": cid, "error": f"sdf_to_xyz failed: {e}", "identifiers": ids}

    return {"status": "not_found", "name": name, "identifiers": ids, "error": "No PubChem 3D SDF found"}


# build_coordination_complex is a server-side MCP tool (molSimplify on pod).
# Kept in TOOLS_RETURNING_STRUCTURE so the REPL handles its geometry_xyz output.


def print_tool_output(text: str, limit: int = 8000):
    shown = text
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            # Your server puts the human-readable ORCA summary here
            if isinstance(payload.get("text"), str) and payload["text"].strip():
                shown = payload["text"]
            # On errors, you may store the tail here
            elif isinstance(payload.get("error_tail"), str) and payload["error_tail"].strip():
                shown = payload["error_tail"]
    except Exception:
        pass

    print("\n[Tool raw output]\n")
    print(shown[:limit])
    if len(shown) > limit:
        print("\n...[truncated]...\n")


import json
from typing import Any, Dict, Optional

def _xyz_no_header_to_species_coords(xyz: str):
    species, coords = [], []
    for ln in xyz.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        species.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return species, coords


def build_dimer_xyz(
    state: dict,
    geom_a_id: str,
    geom_b_id: str,
    distance_ang: float = 3.5,
    ref_atom_b: int = 0,
    axis: str = "z",
    output_id: str = None,
) -> dict:
    """Assemble two molecules into a dimer starting geometry for an interaction scan.

    Centers molecule A at its centroid. Translates molecule B along `axis` so that
    atom `ref_atom_b` of B is placed at `distance_ang` from the centroid of A.

    Args:
        geom_a_id:     Geometry registry key for molecule A (host / larger molecule).
        geom_b_id:     Geometry registry key for molecule B (guest).
        distance_ang:  Initial distance in Å between centroid of A and ref_atom_b of B.
        ref_atom_b:    0-based atom index in B to use as the approach reference (default 0).
        axis:          Approach axis: 'x', 'y', or 'z' (default 'z').
        output_id:     Key under which to store the assembled geometry in state.
        state:         Session state dict (injected by executor).

    Returns:
        dict with status, geometry_xyz, n_atoms_a, n_atoms_b, offset_b,
        scan_atom_b (index of ref_atom_b in the combined geometry),
        charge_a, charge_b.
    """
    import numpy as _np

    if not isinstance(state, dict):
        return {"status": "error", "error": "state dict is required"}

    reg = get_geometry_registry(state)
    xyz_a = reg.get_xyz(geom_a_id)
    xyz_b = reg.get_xyz(geom_b_id)
    if not xyz_a:
        return {"status": "error", "error": f"Geometry not found: {geom_a_id!r}"}
    if not xyz_b:
        return {"status": "error", "error": f"Geometry not found: {geom_b_id!r}"}

    sp_a, co_a = _xyz_no_header_to_species_coords(xyz_a)
    sp_b, co_b = _xyz_no_header_to_species_coords(xyz_b)
    if not sp_a or not sp_b:
        return {"status": "error", "error": "Could not parse one or both geometries"}

    n_a = len(sp_a)
    n_b = len(sp_b)
    ref_atom_b   = int(ref_atom_b)
    distance_ang = float(distance_ang)
    if ref_atom_b >= n_b:
        return {"status": "error",
                "error": f"ref_atom_b={ref_atom_b} out of range (B has {n_b} atoms)"}

    co_a = _np.array(co_a, dtype=float)
    co_b = _np.array(co_b, dtype=float)

    # Center A at its centroid
    centroid_a = co_a.mean(axis=0)
    co_a -= centroid_a

    # Center B at its centroid, then shift so ref_atom_b lands at distance_ang along axis
    centroid_b = co_b.mean(axis=0)
    co_b -= centroid_b                       # center B
    ref_pos_b = co_b[ref_atom_b].copy()     # position of ref atom after centering

    ax_vec = {"x": _np.array([1., 0., 0.]),
              "y": _np.array([0., 1., 0.]),
              "z": _np.array([0., 0., 1.])}.get(axis.lower())
    if ax_vec is None:
        return {"status": "error", "error": f"axis must be 'x', 'y', or 'z', got {axis!r}"}

    # Translate B so ref_atom_b ends up at distance_ang along the approach axis from origin
    shift = ax_vec * distance_ang - ref_pos_b
    co_b += shift

    # Build combined XYZ string
    lines = []
    for s, c in zip(sp_a, co_a):
        lines.append(f"{s:4s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}")
    for s, c in zip(sp_b, co_b):
        lines.append(f"{s:4s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}")
    xyz_combined = "\n".join(lines)

    # Determine combined charge/multiplicity
    meta_a = _state_get_dict(state, "geom_meta").get(geom_a_id, {})
    meta_b = _state_get_dict(state, "geom_meta").get(geom_b_id, {})
    charge_a = int(meta_a.get("charge", 0))
    charge_b = int(meta_b.get("charge", 0))
    total_charge = charge_a + charge_b
    # Multiplicity: assume singlet/doublet based on electron count parity
    n_elec = sum(
        _ATOMIC_NUMBERS.get(s, 0) for s in (sp_a + sp_b)
    ) - total_charge
    total_mult = 1 if (n_elec % 2 == 0) else 2

    # Store in state
    geom_out = output_id or f"{geom_a_id}_{geom_b_id}_dimer"
    _store_new_geometry(
        state,
        xyz=xyz_combined,
        charge=total_charge,
        multiplicity=total_mult,
        geom_id=geom_out,
        overwrite=True,
    )

    scan_atom_b = n_a + ref_atom_b   # 0-based index in combined geometry

    return {
        "status": "ok",
        "geometry_xyz": xyz_combined,
        "n_atoms_a": n_a,
        "n_atoms_b": n_b,
        "offset_b": n_a,
        "scan_atom_b": scan_atom_b,
        "ref_atom_b_local": ref_atom_b,
        "charge": total_charge,
        "multiplicity": total_mult,
        "charge_a": charge_a,
        "charge_b": charge_b,
        "output_id": geom_out,
        "description": (
            f"Atoms 0–{n_a - 1}: {geom_a_id} | "
            f"Atoms {n_a}–{n_a + n_b - 1}: {geom_b_id} | "
            f"Scan: B atom {scan_atom_b} ({sp_b[ref_atom_b]}) "
            f"vs centroid of A along {axis.upper()}-axis"
        ),
    }

def set_geometry_xyz(
    state: dict,
    geom_id: str,
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
) -> dict:
    """Store a literal XYZ coordinate block directly into the geometry registry.

    Use this when the planner computes a custom dimer geometry (tilted approach,
    off-centre target atom, EAS starting geometry, etc.) and needs to inject it
    into state for downstream tool nodes.

    Args:
        geom_id:      Key to store the geometry under in state.
        geometry_xyz: Raw coordinate block — NO natoms/comment header.
                      One 'El  x  y  z' line per atom.
        charge:       Total charge of the molecule (default 0).
        multiplicity: Spin multiplicity (default 1).
        state:        Session state dict (injected by executor).

    Returns:
        dict with status, geom_id, n_atoms.
    """
    if not isinstance(state, dict):
        return {"status": "error", "error": "state dict is required"}

    lines = [l for l in geometry_xyz.strip().splitlines() if l.strip()]
    coord_lines = [l for l in lines if l.split() and l.split()[0].isalpha()]
    n_atoms = len(coord_lines)
    if n_atoms == 0:
        return {"status": "error", "error": "no valid coordinate lines found in geometry_xyz"}

    _store_new_geometry(
        state,
        xyz="\n".join(coord_lines),
        charge=int(charge),
        multiplicity=int(multiplicity),
        geom_id=geom_id,
        overwrite=True,
    )
    return {
        "status": "ok",
        "geom_id": geom_id,
        "n_atoms": n_atoms,
        "geometry_xyz": "\n".join(coord_lines),
    }


def build_approach_scan_geometries(
    state: dict,
    mol_a_id: str = None,
    mol_b_id: str = None,
    n_steps: int = None,
    step_ang: float = None,
    output_prefix: str = None,
    # common planner aliases
    host_geom_id: str = None,
    guest_geom_id: str = None,
    fixed_geom_id: str = None,
    moving_geom_id: str = None,
    geom_a_id: str = None,
    geom_b_id: str = None,
    n_points: int = None,
    step_size_ang: float = None,
    label_prefix: str = None,
    geom_prefix: str = None,
    prefix: str = None,
    ref_atom_a: int = None,   # 0-based atom index in mol_a to aim toward (default: centroid)
    target_atom: int = None,  # alias for ref_atom_a
    **kwargs,  # absorb any other planner-invented aliases
) -> dict:
    """Generate a series of dimer geometries by rigidly translating mol_b toward mol_a.

    Reads mol_a (fixed) and mol_b (moving) from the geometry registry.
    Computes the approach vector as centroid_b → centroid_a (normalized).
    At step i, mol_b is displaced by step_ang * i Å along that vector and
    combined with mol_a into a dimer geometry stored as {output_prefix}_{i:02d}.

    Use this for rigid-body approach scans (no ORCA constraints needed).
    Follow with parallel run_sp_energy nodes on each generated geometry ID.

    Args:
        mol_a_id:       Geometry key for the fixed molecule (host).
        mol_b_id:       Geometry key for the moving molecule (guest).
                        Its current position in state sets the starting geometry.
        n_steps:        Number of geometries to generate (step 0 = starting position).
        step_ang:       Step size in Angstrom. mol_b moves this far per step.
        output_prefix:  Prefix for generated geometry IDs.
                        Default: "{mol_a_id}_{mol_b_id}_approach"
        state:          Session state dict (injected by executor).

    Returns:
        dict with status, geom_ids (list), n_steps, step_ang,
        start_distance_ang, end_distance_ang, charge, multiplicity.
    """
    import numpy as _np

    if not isinstance(state, dict):
        return {"status": "error", "error": "state dict is required"}

    # Resolve aliases
    mol_a_id     = mol_a_id or host_geom_id or fixed_geom_id or geom_a_id
    mol_b_id     = mol_b_id or guest_geom_id or moving_geom_id or geom_b_id
    n_steps      = n_steps or n_points
    step_ang     = step_ang or step_size_ang
    output_prefix = output_prefix or label_prefix or geom_prefix or prefix

    if not mol_a_id:
        return {"status": "error", "error": "mol_a_id (or host_geom_id) is required"}
    if not mol_b_id:
        return {"status": "error", "error": "mol_b_id (or guest_geom_id) is required"}
    if n_steps is None:
        return {"status": "error", "error": "n_steps is required"}
    if step_ang is None:
        return {"status": "error", "error": "step_ang is required"}

    n_steps  = int(n_steps)
    step_ang = float(step_ang)
    if n_steps < 2:
        return {"status": "error", "error": "n_steps must be at least 2"}
    if step_ang <= 0:
        return {"status": "error", "error": "step_ang must be positive"}

    reg   = get_geometry_registry(state)
    xyz_a = reg.get_xyz(mol_a_id)
    xyz_b = reg.get_xyz(mol_b_id)
    if not xyz_a:
        return {"status": "error", "error": f"Geometry not found: {mol_a_id!r}"}
    if not xyz_b:
        return {"status": "error", "error": f"Geometry not found: {mol_b_id!r}"}

    sp_a, co_a = _xyz_no_header_to_species_coords(xyz_a)
    sp_b, co_b = _xyz_no_header_to_species_coords(xyz_b)
    if not sp_a or not sp_b:
        return {"status": "error", "error": "Could not parse one or both geometries"}

    co_a = _np.array(co_a, dtype=float)
    co_b = _np.array(co_b, dtype=float)

    centroid_a = co_a.mean(axis=0)
    centroid_b = co_b.mean(axis=0)

    # Target point on mol_a: specific atom if ref_atom_a given, else centroid
    _ref_a = ref_atom_a if ref_atom_a is not None else target_atom
    if _ref_a is not None:
        _ref_a = int(_ref_a)
        if _ref_a >= len(co_a):
            return {"status": "error",
                    "error": f"ref_atom_a={_ref_a} out of range (mol_a has {len(co_a)} atoms)"}
        target_a = co_a[_ref_a]
    else:
        target_a = centroid_a

    # Approach vector: from mol_b centroid toward target point on mol_a
    raw_vec = target_a - centroid_b
    dist0   = float(_np.linalg.norm(raw_vec))
    if dist0 < 1e-6:
        return {"status": "error", "error": "mol_a target and mol_b centroid are coincident — cannot determine approach direction"}
    approach_vec = raw_vec / dist0   # unit vector

    # Combined charge/multiplicity
    meta_a = _state_get_dict(state, "geom_meta").get(mol_a_id, {})
    meta_b = _state_get_dict(state, "geom_meta").get(mol_b_id, {})
    charge_a    = int(meta_a.get("charge", 0))
    charge_b    = int(meta_b.get("charge", 0))
    total_charge = charge_a + charge_b
    n_elec = sum(_ATOMIC_NUMBERS.get(s, 0) for s in (sp_a + sp_b)) - total_charge
    total_mult = 1 if (n_elec % 2 == 0) else 2

    prefix = (output_prefix or f"{mol_a_id}_{mol_b_id}_approach").rstrip("_- ")
    geom_ids = []

    def _fmt_xyz(species, coords):
        return "\n".join(
            f"{s:4s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}"
            for s, c in zip(species, coords)
        )

    for i in range(n_steps):
        displacement = approach_vec * step_ang * i
        co_b_shifted = co_b + displacement
        xyz_combined = _fmt_xyz(sp_a, co_a) + "\n" + _fmt_xyz(sp_b, co_b_shifted)
        geom_id = f"{prefix}_{i:02d}"
        _store_new_geometry(
            state,
            xyz=xyz_combined,
            charge=total_charge,
            multiplicity=total_mult,
            geom_id=geom_id,
            overwrite=True,
        )
        geom_ids.append(geom_id)

    end_dist = dist0 - step_ang * (n_steps - 1)

    # Return new geometries and geom_meta explicitly so the LangGraph executor
    # propagates them to downstream nodes via the merge reducer.
    new_geometries = {gid: state["geometries"][gid] for gid in geom_ids if gid in state.get("geometries", {})}
    new_geom_meta  = {gid: state["geom_meta"][gid]  for gid in geom_ids if gid in state.get("geom_meta",  {})}

    return {
        "status":             "ok",
        "geom_ids":           geom_ids,
        "n_steps":            n_steps,
        "step_ang":           step_ang,
        "start_distance_ang": round(dist0, 4),
        "end_distance_ang":   round(end_dist, 4),
        "charge":             total_charge,
        "multiplicity":       total_mult,
        "output_prefix":      prefix,
        # Propagate new geometries into LangGraph state
        "geometries":         new_geometries,
        "geom_meta":          new_geom_meta,
        "description": (
            f"{n_steps} dimer geometries: {mol_b_id} approaches {mol_a_id} "
            f"from {dist0:.2f} Ang to {end_dist:.2f} Ang in {step_ang} Ang steps"
        ),
    }


def get_trivial_properties(state, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns: formula, mol_weight, point_group (if geometry exists).
    Prefers: explicit name -> identifiers[name]; else current geometry metadata.
    """
    # Resolve which record to use
    meta: Dict[str, Any] = {}
    if name and name in state.identifiers:
        meta = dict(state.identifiers[name])
        meta.setdefault("name", name)
    elif state.current_geom and state.current_geom in state.geom_meta:
        meta = dict(state.geom_meta[state.current_geom])
        meta.setdefault("geometry_key", state.current_geom)
    else:
        meta = {"geometry_key": state.current_geom}

    smiles = meta.get("smiles")

    # 1) Formula + MW (RDKit from SMILES if available)
    formula = None
    mw = None
    if smiles:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                formula = rdMolDescriptors.CalcMolFormula(mol)
                mw = float(Descriptors.MolWt(mol))
        except Exception as e:
            meta["rdkit_error"] = str(e)

    # 2) Point group from current geometry (pymatgen)
    point_group = None
    geom_key = state.current_geom
    if geom_key and geom_key in state.geometries:
        xyz = state.geometries[geom_key]
        try:
            from pymatgen.core import Molecule
            from pymatgen.symmetry.analyzer import PointGroupAnalyzer

            species, coords = _xyz_no_header_to_species_coords(xyz)
            if species and coords:
                mol = Molecule(species, coords)
                pga = PointGroupAnalyzer(mol)
                point_group = pga.get_pointgroup()
        except Exception as e:
            meta["pymatgen_error"] = str(e)

    return {
        "status": "ok",
        "name": meta.get("name"),
        "geometry_key": geom_key,
        "smiles": smiles,
        "formula": formula,
        "mol_weight": mw,
        "point_group": point_group,
        "meta": meta,
    }

import requests
import urllib.parse
from typing import Any, Dict, Optional

PUBCHEM_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# PUG-REST "property" endpoint supports pulling multiple properties at once, including SMILES/InChI/InChIKey. 
PUBCHEM_PROPERTIES = [
    "MolecularFormula",
    "MolecularWeight",
    "Charge",
    "Complexity",
    "IsomericSMILES",
    "InChI",
    "InChIKey",
]

def pubchem_get_basic_properties(name: str, timeout: int = 20) -> Dict[str, Any]:
    """
    Query PubChem for basic computed properties by *compound name*.

    Returns:
      {"status":"ok", "name":..., "cid":..., "formula":..., "mw":..., "charge":..., "complexity":..., "smiles":..., "inchi":..., "inchikey":...}
    or {"status":"not_found", ...}
    or {"status":"error", "error":...}
    """
    q = (name or "").strip()
    if not q:
        return {"status": "error", "error": "empty name"}

    prop_str = ",".join(PUBCHEM_PROPERTIES)
    url = f"{PUBCHEM_PUG}/compound/name/{urllib.parse.quote(q)}/property/{prop_str}/JSON"

    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        return {"status": "error", "name": q, "error": f"request failed: {e}"}

    if r.status_code == 404:
        return {"status": "not_found", "name": q, "error": "PubChem returned 404"}
    if r.status_code != 200:
        return {"status": "error", "name": q, "error": f"HTTP {r.status_code}: {r.text[:300]}"}

    try:
        data = r.json()
        props = data["PropertyTable"]["Properties"][0]
    except Exception as e:
        return {"status": "error", "name": q, "error": f"unexpected JSON shape: {e}", "raw": r.text[:500]}

    # Normalize keys to your preferred names
    out: Dict[str, Any] = {
        "status": "ok",
        "name": q,
        "cid": props.get("CID"),
        "formula": props.get("MolecularFormula"),
        "mw": props.get("MolecularWeight"),
        "charge": props.get("Charge"),         # net charge 
        "complexity": props.get("Complexity"), # Bertz complexity 
        "smiles": props.get("IsomericSMILES"),
        "inchi": props.get("InChI"),
        "inchikey": props.get("InChIKey"),
        "source": "pubchem_pug_rest",
        "request_url": url,
    }
    return out


def structure_add_remove_proton(
    geometry_xyz: str,
    mode: str,
    charge: int = 0,
    multiplicity: int = 1,
    site_selector: Optional[str] = None,
    variant: int = 0,
    h_index: Optional[int] = None,
    target_atom_index: Optional[int] = None,
    geometry_name: Optional[str] = None,
    strategy: str = "auto",
) -> dict:
    """Local implementation of structure_add_remove_proton — no MCP/SSH round-trip."""
    from geometry_helpers import structure_proton_edit
    return structure_proton_edit(
        xyz=geometry_xyz,
        mode=mode,
        charge=charge,
        multiplicity=multiplicity,
        site_selector=site_selector,
        variant=variant,
        h_index=h_index,
        target_atom_index=target_atom_index,
        geometry_name=geometry_name,
        strategy=strategy,
    )


# Proton free-energy references, in Hartree. Which one is correct depends
# entirely on whether the calculation is calibrated against reference acids.
#
#   THERMAL: the proton's gas-phase thermal correction only (H - TS for a
#   monatomic ideal gas). Valid ONLY in the calibrated path, where the term
#   cancels exactly between target and references and its value is therefore
#   irrelevant. Using it uncalibrated omits the proton's aqueous solvation
#   free energy (~-266 kcal/mol) and inflates the result by ~195 pKa units.
#
#   AQUEOUS: the full aqueous proton free energy, the physically meaningful
#   choice for an uncalibrated thermodynamic cycle. Built from published
#   components rather than written as a single opaque constant:
#       dG_solv(H+) = -265.9 kcal/mol   (Tissandier et al. 1998)
#       G_gas(H+)   =   -6.28 kcal/mol  (H - TS, 298.15 K, 1 atm)
#       1 atm -> 1 M standard-state correction = +1.89 kcal/mol
_EH_PER_KCAL = 1.0 / 627.509474
G_H_PLUS_THERMAL_EH = -0.01372
G_H_PLUS_AQUEOUS_EH = (-265.9 - 6.28 + 1.89) * _EH_PER_KCAL   # ~ -0.43074 Eh


_EAS_DEFAULT_HETEROATOMS = ("N", "O", "S")


def rank_eas_sites(
    mulliken_charges: Any,
    include_heteroatoms: bool = True,
    heteroatoms: Optional[List[str]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Rank EAS sites by partial charge, in code rather than in a language model.

    The EAS skill previously instructed the planner to emit a kind:"llm" node
    whose entire task was: keep the carbons (and any heteroatoms), sort ascending
    by charge, return the ranking. That is filter-and-sort over a table the
    upstream tool already produced -- the chemistry sits in the element
    predicate, which is fixed, not in the sorting. Doing it in the model cost
    roughly 1,500 tokens per molecule and inherited the reproducibility problem
    documented for llm arithmetic nodes, in exchange for no judgement the code
    cannot make.

    Most negative charge first: the most electron-rich site is the most
    activated toward electrophilic attack.

    Accepts the tool's list of {atom_index, symbol, charge} dicts, or a JSON
    string of the same, since planners pass artifacts through in both forms.
    """
    if isinstance(mulliken_charges, str):
        try:
            mulliken_charges = json.loads(mulliken_charges)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"mulliken_charges was a string but not valid JSON: {exc}") from exc

    if isinstance(mulliken_charges, dict):
        for key in ("mulliken_charges", "charges", "atoms"):
            if isinstance(mulliken_charges.get(key), list):
                mulliken_charges = mulliken_charges[key]
                break

    if not isinstance(mulliken_charges, list):
        raise ValueError(
            "mulliken_charges must be a list of {atom_index, symbol, charge} "
            f"entries; got {type(mulliken_charges).__name__}")

    wanted = {"C"}
    if include_heteroatoms:
        wanted |= {str(s).strip().title() for s in (heteroatoms or _EAS_DEFAULT_HETEROATOMS)}

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for entry in mulliken_charges:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        symbol = str(entry.get("symbol") or entry.get("element") or "").strip().title()
        charge = entry.get("charge", entry.get("mulliken_charge"))
        idx = entry.get("atom_index", entry.get("index"))
        if symbol not in wanted:
            continue
        try:
            charge = float(charge)
        except (TypeError, ValueError):
            skipped += 1          # a malformed row is dropped, never coerced to 0.0,
            continue              # which would rank it as the most activated site
        rows.append({"atom_idx": idx, "element": symbol,
                     "mulliken_charge": charge})

    rows.sort(key=lambda r: r["mulliken_charge"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    if top_n is not None:
        rows = rows[:int(top_n)]

    warnings: List[str] = []
    if skipped:
        warnings.append(f"{skipped} entr{'y' if skipped == 1 else 'ies'} skipped "
                        "(not a dict, or charge not numeric)")
    if not rows:
        warnings.append("no atoms matched the element filter "
                        f"({', '.join(sorted(wanted))})")

    return {
        # Client-side tools report status the same way remote ones do; without
        # it the executor records the node as 'unknown' and a node spec's
        # expect.status_in cannot be satisfied.
        "status": "ok",
        "site_ranking": rows,
        "most_activated": rows[0] if rows else None,
        "n_sites": len(rows),
        "elements_considered": sorted(wanted),
        "warnings": warnings,
    }


def compute_spin_gap(
    E_hs_eh: float,
    E_ls_eh: float,
) -> dict:
    """Deterministic CASSCF spin-state gap (no LLM arithmetic).

    The CASSCF skill previously instructed the planner to emit a kind:"llm"
    node whose entire task was subtracting two already-computed energies and
    converting Hartree to kcal/mol -- the same class of arithmetic-in-a-language-
    model defect documented for the pKa and EAS skills. There is no judgement
    call here: which two energies to subtract and which conversion factor to
    use are both fixed once the two CASSCF jobs are specified.

    A positive result means the high-spin state lies above the low-spin state.

    Args:
        E_hs_eh: high-spin state CASSCF energy (Hartree).
        E_ls_eh: low-spin state CASSCF energy (Hartree).

    Returns:
        {"status": "ok", "delta_E_kcal": float, "delta_E_eh": float,
         "ground_state": "high-spin" | "low-spin" | "degenerate"}
        or {"status": "error", "error": str} on invalid input.
    """
    EH_TO_KCAL = 627.509474
    try:
        e_hs = float(E_hs_eh)
        e_ls = float(E_ls_eh)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid energy input: {exc}"}

    delta_eh = e_hs - e_ls
    delta_kcal = delta_eh * EH_TO_KCAL
    if abs(delta_kcal) < 1e-6:
        ground_state = "degenerate"
    elif delta_kcal > 0:
        ground_state = "low-spin"
    else:
        ground_state = "high-spin"

    return {
        "status": "ok",
        "delta_E_kcal": delta_kcal,
        "delta_E_eh": delta_eh,
        "ground_state": ground_state,
    }


def compute_pka_calibrated(
    G_HA_eh: float,
    G_A_minus_eh: float,
    references: List[Dict[str, float]],
    G_H_plus_ref_eh: Optional[float] = None,
    temperature_K: float = 298.15,
) -> dict:
    """Deterministic reference-acid-calibrated pKa (no LLM arithmetic).

    Generalizes the standard isodesmic-correction formula used throughout the
    pKa skill to N >= 1 reference acids: for each reference, back out the
    systematic gas-phase error (epsilon_i = pKa_calc,i - pKa_exp,i) from its
    own computed Gibbs energies and known experimental pKa, then apply the
    mean correction across all references to the target compound.

    N == 0 (empty references) skips calibration entirely and returns the raw
    formula pKa — this is the plain, uncalibrated pKa case (e.g. a simple
    carboxylic-acid O-H pKa with no reference-acid correction). N == 1
    reproduces the single-reference isodesmic scheme (e.g. ethanal for alpha-CH
    pKa). N > 1 reproduces the multi-reference-acid averaging scheme (e.g. the
    El Agente pKa-of-carboxylic-acids benchmark). One function covers all three
    cases so a skill only needs to document a single tool/node pattern.

    Args:
        G_HA_eh: Gibbs free energy of the target neutral acid (Hartree).
        G_A_minus_eh: Gibbs free energy of the target conjugate base (Hartree).
        references: list of {"G_HA_eh": ..., "G_A_minus_eh": ..., "pka_exp": ...}
            dicts, one per reference acid. Empty list = no calibration.
        G_H_plus_ref_eh: proton free energy (Hartree). Defaults to whichever
            constant is correct for the call: G_H_PLUS_AQUEOUS_EH when no
            references are supplied (the term is then physically meaningful and
            must include the proton's solvation free energy), and
            G_H_PLUS_THERMAL_EH when they are (the term cancels exactly, so its
            value cannot affect the result). Pass a value explicitly to override.
        temperature_K: temperature for the RT ln10 conversion (default 298.15 K).

    Returns:
        {"status": "ok", "pka": float, "pka_raw": float, "epsilon_avg": float,
         "per_reference_epsilon": [float, ...], "calibrated": bool,
         "G_H_plus_ref_eh": float, "warnings": [str, ...]}
        or {"status": "error", "error": str} on invalid input.
    """
    R = 8.314462618          # J / (mol K)
    LN10 = 2.302585093
    EH_TO_J_PER_MOL = 2625499.638

    calibrated = bool(references)
    warnings: List[str] = []
    if G_H_plus_ref_eh is None:
        G_H_plus_ref_eh = (G_H_PLUS_THERMAL_EH if calibrated
                           else G_H_PLUS_AQUEOUS_EH)
    elif not calibrated and abs(G_H_plus_ref_eh - G_H_PLUS_THERMAL_EH) < 1e-9:
        # The single most damaging misuse of this function: the gas-phase
        # thermal correction used as if it were the aqueous proton free energy,
        # with no references to cancel it. Silently correcting the caller would
        # hide a bad plan, so compute what was asked and flag it loudly.
        warnings.append(
            "Uncalibrated pKa requested with the gas-phase thermal proton "
            "reference (-0.01372 Eh). This omits the proton's aqueous solvation "
            "free energy and overestimates pKa by roughly 195 units. Supply "
            "reference acids, or use G_H_PLUS_AQUEOUS_EH."
        )

    def _raw_pka(g_ha: float, g_a: float) -> float:
        dg_eh = g_a + G_H_plus_ref_eh - g_ha
        dg_j = dg_eh * EH_TO_J_PER_MOL
        return dg_j / (R * temperature_K * LN10)

    try:
        epsilons = []
        for ref in (references or []):
            pka_ref_calc = _raw_pka(float(ref["G_HA_eh"]), float(ref["G_A_minus_eh"]))
            epsilons.append(pka_ref_calc - float(ref["pka_exp"]))
        epsilon_avg = (sum(epsilons) / len(epsilons)) if epsilons else 0.0
        pka_raw = _raw_pka(float(G_HA_eh), float(G_A_minus_eh))
        pka = pka_raw - epsilon_avg
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid reference/energy input: {exc}"}

    if not calibrated and not (-20.0 < pka < 60.0):
        warnings.append(
            f"Uncalibrated pKa {pka:.2f} is outside any physically plausible "
            f"range. Check that the Gibbs energies include solvation (for an "
            f"implicit model, put it in the method string, e.g. "
            f"method=\"B3LYP CPCM(Water)\") and that the proton reference is "
            f"the aqueous value."
        )

    return {
        "status": "ok",
        "pka": pka,
        "pka_raw": pka_raw,
        "epsilon_avg": epsilon_avg,
        "per_reference_epsilon": epsilons,
        "calibrated": calibrated,
        "G_H_plus_ref_eh": G_H_plus_ref_eh,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Pre-planning compound identification helpers
# ---------------------------------------------------------------------------

def extract_compound_names_llm(user_text: str, openai_client: Any) -> List[str]:
    """Use a fast LLM call to extract chemical compound names from free-form text.

    Returns a (possibly empty) list of name strings.
    """
    prompt = (
        "Extract chemical compound or molecule names from the following text. "
        "Return ONLY a JSON array of strings (the names), nothing else. "
        "If no compound names are present, return []. "
        f"Text: {user_text!r}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-5.2"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        names = json.loads(raw)
        if isinstance(names, list):
            return [str(n) for n in names if isinstance(n, str) and n.strip()]
    except Exception:
        pass
    return []


def fetch_compound_card(name: str) -> Dict[str, Any]:
    """Fetch compound info from PubChem and OPSIN, returning a unified card dict.

    Keys: name, formula, smiles, mw, charge, cid, inchi, inchikey, source.
    source is 'pubchem', 'opsin', or 'not_found'.
    """
    card: Dict[str, Any] = {
        "name": name,
        "formula": None,
        "smiles": None,
        "mw": None,
        "charge": 0,
        "cid": None,
        "inchi": None,
        "inchikey": None,
        "source": "not_found",
    }

    # Try PubChem first (richer data)
    pc = pubchem_get_basic_properties(name)
    if pc.get("status") == "ok":
        card.update({
            "formula": pc.get("formula"),
            "smiles": pc.get("smiles"),
            "mw": pc.get("mw"),
            "charge": pc.get("charge") or 0,
            "cid": pc.get("cid"),
            "inchi": pc.get("inchi"),
            "inchikey": pc.get("inchikey"),
            "source": "pubchem",
        })

    # Always try OPSIN to fill in missing SMILES (PubChem sometimes omits it)
    if not card.get("smiles"):
        op = opsin_resolve(name)
        if op:
            card.update({
                "smiles": op.get("smiles") or card.get("smiles"),
                "inchi": card.get("inchi") or op.get("inchi"),
                "inchikey": card.get("inchikey") or op.get("inchikey"),
            })
            if card["source"] == "not_found":
                card["source"] = "opsin"

    if card["source"] != "not_found":
        return card

    # OPSIN-only path (PubChem failed entirely)
    op = opsin_resolve(name)
    if op:
        card.update({
            "smiles": op.get("smiles"),
            "inchi": op.get("inchi"),
            "inchikey": op.get("inchikey"),
            "source": "opsin",
        })

    return card


def _xyz_no_header_to_block(xyz_no_header: str, charge: int = 0) -> str:
    """Add natoms + comment header so RDKit's MolFromXYZBlock can parse it."""
    lines = [ln for ln in xyz_no_header.splitlines() if ln.strip()]
    return f"{len(lines)}\ncharge={charge}\n" + "\n".join(lines) + "\n"


def render_xyz_image_rdkit(xyz_no_header: str, label: str, charge: int = 0) -> Optional[str]:
    """Render a 2D structure PNG directly from XYZ atom lines (no SMILES needed).

    Uses RDKit MolFromXYZBlock + DetermineBonds to infer connectivity, then
    Compute2DCoords for a clean 2D layout.  Opens the PNG in the OS viewer.
    Returns the saved path, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem, rdDetermineBonds
    except ImportError:
        return None

    try:
        xyz_block = _xyz_no_header_to_block(xyz_no_header, charge)
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=charge)

        # Project to 2D for a clean drawing
        mol_2d = Chem.RWMol(Chem.RemoveAllHs(mol))  # hide H for larger molecules
        # For small molecules keep explicit H so bonds are visible
        if mol.GetNumAtoms() <= 6:
            mol_2d = Chem.RWMol(mol)
        AllChem.Compute2DCoords(mol_2d)

        import tempfile, platform, subprocess as sp
        tmp_dir = Path(tempfile.gettempdir()) / "qcagent_structures"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", label)[:40]
        png_path = tmp_dir / f"{safe_label}.png"

        img = Draw.MolToImage(mol_2d, size=(500, 400))
        img.save(str(png_path))

        system = platform.system()
        if system == "Windows":
            os.startfile(str(png_path))
        elif system == "Darwin":
            sp.Popen(["open", str(png_path)])
        else:
            sp.Popen(["xdg-open", str(png_path)])

        return str(png_path)
    except Exception:
        return None


def fetch_compound_card_from_xyz(
    xyz_no_header: str, geom_id: str, charge: int = 0
) -> Dict[str, Any]:
    """Derive a compound card from loaded XYZ via bond detection + optional PubChem lookup.

    Tries DetermineBonds → canonical SMILES → PubChem lookup by SMILES.
    Falls back to a minimal card with just the geom_id as name.
    """
    card: Dict[str, Any] = {
        "name": geom_id,
        "formula": None,
        "smiles": None,
        "mw": None,
        "charge": charge,
        "cid": None,
        "inchi": None,
        "inchikey": None,
        "source": "xyz",
    }

    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds, Descriptors
        xyz_block = _xyz_no_header_to_block(xyz_no_header, charge)
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return card
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        smiles = Chem.MolToSmiles(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mw = round(Descriptors.ExactMolWt(mol), 4)
        card.update({"smiles": smiles, "formula": formula, "mw": mw})
    except Exception:
        return card

    # Try PubChem lookup by SMILES to get CID / InChI
    try:
        import urllib.parse
        prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        prop_str = "MolecularFormula,MolecularWeight,Charge,IsomericSMILES,InChI,InChIKey"
        url = f"{prolog}/compound/smiles/{urllib.parse.quote(smiles)}/property/{prop_str}/JSON"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            props = r.json()["PropertyTable"]["Properties"][0]
            card.update({
                "formula": props.get("MolecularFormula") or card["formula"],
                "mw": props.get("MolecularWeight") or card["mw"],
                "cid": props.get("CID"),
                "inchi": props.get("InChI"),
                "inchikey": props.get("InChIKey"),
                "source": "xyz+pubchem",
            })
    except Exception:
        pass

    return card


def render_compound_image_rdkit(smiles: str, name: str) -> Optional[str]:
    """Render a 2D structure PNG from SMILES using RDKit, then open it in the OS viewer.

    Returns the saved PNG path, or None if rendering failed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, AllChem
    except ImportError:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add explicit H atoms so small molecules (e.g. water, HF) show bonds
        mol = Chem.AddHs(mol)
        AllChem.Compute2DCoords(mol)

        import tempfile, platform, subprocess as sp
        tmp_dir = Path(tempfile.gettempdir()) / "qcagent_structures"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", name)[:40]
        png_path = tmp_dir / f"{safe_name}.png"

        img = Draw.MolToImage(mol, size=(500, 400))
        img.save(str(png_path))

        system = platform.system()
        if system == "Windows":
            os.startfile(str(png_path))
        elif system == "Darwin":
            sp.Popen(["open", str(png_path)])
        else:
            sp.Popen(["xdg-open", str(png_path)])

        return str(png_path)
    except Exception:
        return None


def display_and_confirm_compound(card: Dict[str, Any], openai_client: Any, _depth: int = 0) -> Optional[Dict[str, Any]]:
    """Show a compound card to the user and ask for confirmation.

    Returns the confirmed card (possibly for a renamed compound), or None if skipped.
    Allows up to 3 rename attempts.
    """
    SEP = "─" * 56
    name = card.get("name", "?")
    print(f"\n{SEP}")
    print(f"  Compound : {name}")
    if card.get("formula"):
        print(f"  Formula  : {card['formula']}")
    if card.get("smiles"):
        print(f"  SMILES   : {card['smiles']}")
    if card.get("mw") is not None:
        print(f"  MW       : {card['mw']} g/mol")
    chg = card.get("charge", 0)
    print(f"  Charge   : {chg}")
    if card.get("cid"):
        print(f"  CID      : {card['cid']}")
    if card.get("source") == "not_found":
        print("  [Not found in PubChem or OPSIN]")

    if card.get("smiles"):
        img_path = render_compound_image_rdkit(card["smiles"], name)
        if img_path:
            print(f"  Structure: {img_path} [opened]")
        else:
            print("  Structure: (RDKit rendering unavailable)")
    print(SEP)

    while True:
        raw = input("  Correct compound? [y / n / rename to <name>]: ").strip()
        if not raw or raw.lower() in ("y", "yes"):
            return card
        if raw.lower() in ("n", "no"):
            # Offer manual SMILES entry before skipping
            manual = input("  Enter SMILES manually (or blank to skip): ").strip()
            if manual:
                card = dict(card)
                card["smiles"] = manual
                card["source"] = "manual"
                return card
            return None
        if raw.lower().startswith("rename to "):
            new_name = raw[len("rename to "):].strip()
            if new_name and _depth < 3:
                new_card = fetch_compound_card(new_name)
                return display_and_confirm_compound(new_card, openai_client, _depth + 1)
            print("  Max rename attempts reached or empty name.")
        else:
            print("  Please type y, n, or 'rename to <name>'.")


def compounds_to_planner_context(compounds: List[Dict[str, Any]], mode: str = "full") -> str:
    """Format a list of confirmed compound cards as a planner system-message string.

    mode controls what fields are included in CONFIRMED_COMPOUNDS:
      "full"   (default): name, formula, SMILES, MW, CID, charge
      "smiles": name, SMILES, charge only
      "name":   name, charge only
      "xyz":    returns "" (planner reasons from geometry state only)
    """
    if not compounds or mode == "xyz":
        return ""
    lines = ["CONFIRMED COMPOUNDS (verified by user before planning):"]
    for c in compounds:
        if mode == "name":
            detail = f"charge={c.get('charge', 0)}"
        elif mode == "smiles":
            parts = [f"SMILES={c['smiles']}" if c.get("smiles") else None,
                     f"charge={c.get('charge', 0)}"]
            detail = ", ".join(p for p in parts if p)
        else:  # "full"
            parts = [f"formula={c['formula']}" if c.get("formula") else None,
                     f"SMILES={c['smiles']}" if c.get("smiles") else None,
                     f"charge={c.get('charge', 0)}",
                     f"MW={c['mw']} g/mol" if c.get("mw") else None,
                     f"CID={c['cid']}" if c.get("cid") else None]
            detail = ", ".join(p for p in parts if p)
        lines.append(f"  - {c['name']}: {detail}")
    lines.append("Use these confirmed identities when assigning geom_ids and charge/multiplicity.")
    return "\n".join(lines)


ALLOWED_STATE_KEYS = {
    "default_charge",
    "default_multiplicity",
    "default_method",
    "default_basis",
    "default_solvent",
    "current_geom",
    "current_name",   # optional: user's "active molecule name"
}

ALLOWED_STATE_KEYS = {
    "default_charge",
    "default_multiplicity",
    "current_geom",
}



def xyz_to_no_header(xyz: str) -> str:
    """Accept XYZ with or without header; return atom-lines-only string."""
    if not isinstance(xyz, str):
        raise TypeError(f"xyz must be str, got {type(xyz).__name__}")

    lines = [ln.strip() for ln in xyz.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Try to detect XYZ header: first line nat (int), then one comment line.
    try:
        nat = int(lines[0])
        if len(lines) >= nat + 2:
            body = lines[2 : 2 + nat]
            # Basic sanity: each line should have >= 4 columns
            if sum(1 for ln in body if len(ln.split()) >= 4) >= max(1, nat // 2):
                return "\n".join(body).rstrip() + "\n"
    except Exception:
        pass

    # Already headerless or malformed header
    return "\n".join(lines).rstrip() + "\n"


def coerce_geometry_xyz(xyz: Any) -> str:
    """Best-effort: coerce tool outputs to headerless xyz."""
    if xyz is None:
        return ""
    if not isinstance(xyz, str):
        xyz = str(xyz)
    return xyz_to_no_header(xyz)


# -------------------------
# GeometryRegistry
# -------------------------
def inspect_geom_registry_state(state: Dict[str, Any], *, preview_lines: int = 6) -> str:
    """Return a debug string for GeometryRegistry-backed state.

    Prints:
      - current_geom
      - keys present in state
      - geometries keys + xyz preview
      - geom_meta (charge/multiplicity/name) per geom_id
      - name_to_geom mapping
    """
    if not isinstance(state, dict):
        return f"[inspect_geom_registry_state] state is not a dict: {type(state).__name__}"

    reg = get_geometry_registry(state)  # thin wrapper over state keys

    lines: list[str] = []
    lines.append("=== GeometryRegistry / state inspection ===")
    lines.append(f"state keys: {sorted(list(state.keys()))}")

    lines.append(f"state['current_geom']: {state.get('current_geom', None)!r}")
    lines.append(f"reg.current_geom: {getattr(reg, 'current_geom', None)!r}")

    geoms = state.get("geometries")
    meta = state.get("geom_meta")
    name_map = state.get("name_to_geom")

    lines.append(f"type(state['geometries']): {type(geoms).__name__}")
    lines.append(f"type(state['geom_meta']): {type(meta).__name__}")
    lines.append(f"type(state['name_to_geom']): {type(name_map).__name__}")

    if not isinstance(geoms, dict) or not geoms:
        lines.append("No geometries in state['geometries'].")
    else:
        lines.append(f"Geometries ({len(geoms)}): {sorted(list(geoms.keys()))}")
        for gid in sorted(list(geoms.keys())):
            xyz = geoms.get(gid)
            xyz_str = xyz if isinstance(xyz, str) else ""
            xyz_lines = xyz_str.strip().splitlines()

            m = meta.get(gid) if isinstance(meta, dict) else None
            if not isinstance(m, dict):
                m = {}

            nm = m.get("name", None)
            chg = m.get("charge", None)
            mult = m.get("multiplicity", None)

            mark = " (current)" if state.get("current_geom") == gid else ""
            lines.append(f"- {gid}{mark}: name={nm!r} charge={chg!r} mult={mult!r} xyz_lines={len(xyz_lines)}")

            if xyz_lines:
                head = xyz_lines[:preview_lines]
                lines.append("  xyz preview:")
                for ln in head:
                    lines.append(f"    {ln}")
                if len(xyz_lines) > preview_lines:
                    lines.append(f"    ... ({len(xyz_lines) - preview_lines} more lines)")

    if isinstance(name_map, dict) and name_map:
        lines.append("name_to_geom:")
        for k in sorted(name_map.keys()):
            lines.append(f"  - {k!r} -> {name_map.get(k)!r}")
    else:
        lines.append("name_to_geom: (empty)")

    # Also show what the registry wrapper thinks it has (in case keys differ)
    try:
        reg_ids = list(getattr(reg, "geometries", {}).keys())
        lines.append(f"reg.geometries keys: {sorted(reg_ids)}")
    except Exception as e:
        lines.append(f"reg.geometries keys: <error reading> {e!r}")

    try:
        lines.append(f"reg.resolve_id(current_geom): {reg.resolve_id(state.get('current_geom'))!r}")
    except Exception as e:
        lines.append(f"reg.resolve_id: <error> {e!r}")

    lines.append("=== end ===")
    return "\n".join(lines)

class GeometryRegistry:
    """A thin wrapper that stores geometries inside a dict state."""

    def __init__(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            raise TypeError(f"GeometryRegistry expects dict state, got {type(state).__name__}")
        self.state = state
        self._ensure_state()

    def _ensure_state(self) -> None:
        self.state.setdefault("geometries", {})
        self.state.setdefault("geom_meta", {})
        self.state.setdefault("name_to_geom", {})
        self.state.setdefault("current_geom", None)

    @property
    def geometries(self) -> Dict[str, str]:
        g = self.state.get("geometries")
        if not isinstance(g, dict):
            g = {}
            self.state["geometries"] = g
        return g

    @property
    def geom_meta(self) -> Dict[str, Dict[str, Any]]:
        m = self.state.get("geom_meta")
        if not isinstance(m, dict):
            m = {}
            self.state["geom_meta"] = m
        return m

    @property
    def name_to_geom(self) -> Dict[str, str]:
        n = self.state.get("name_to_geom")
        if not isinstance(n, dict):
            n = {}
            self.state["name_to_geom"] = n
        return n

    @property
    def current_geom(self) -> Optional[str]:
        cg = self.state.get("current_geom")
        return cg if isinstance(cg, str) and cg else None

    def set_current(self, geom_id: str) -> None:
        if geom_id not in self.geometries:
            raise KeyError(f"Unknown geom_id: {geom_id!r}")
        self.state["current_geom"] = geom_id

    def _new_id(self, prefix: str = "geom") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"{prefix}_{ts}"
        gid = base
        i = 2
        while gid in self.geometries:
            gid = f"{base}_{i}"
            i += 1
        return gid

    def register(
        self,
        *,
        xyz: str,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        geom_id: Optional[str] = None,
        name: Optional[str] = None,
        set_current: bool = True,
        overwrite: bool = False,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        xyz_nh = coerce_geometry_xyz(xyz)
        if not xyz_nh.strip():
            raise ValueError("Cannot register empty xyz")

        gid = geom_id or self._new_id()
        if (not overwrite) and gid in self.geometries:
            raise KeyError(f"geom_id already exists: {gid!r}")

        self.geometries[gid] = xyz_nh
        meta = dict(self.geom_meta.get(gid) or {})
        if charge is not None:
            meta["charge"] = int(charge)
        if multiplicity is not None:
            meta["multiplicity"] = int(multiplicity)
        if meta_extra:
            meta.update(meta_extra)
        self.geom_meta[gid] = meta

        if name:
            self.name_to_geom[str(name)] = gid

        if set_current:
            self.state["current_geom"] = gid
        return gid

    def update(
        self,
        geom_id: str,
        *,
        xyz: Optional[str] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        name: Optional[str] = None,
        set_current: bool = True,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        if geom_id not in self.geometries:
            raise KeyError(f"Unknown geom_id: {geom_id!r}")

        if xyz is not None:
            xyz_nh = coerce_geometry_xyz(xyz)
            if not xyz_nh.strip():
                raise ValueError("Cannot set empty xyz")
            self.geometries[geom_id] = xyz_nh

        meta = dict(self.geom_meta.get(geom_id) or {})
        if charge is not None:
            meta["charge"] = int(charge)
        if multiplicity is not None:
            meta["multiplicity"] = int(multiplicity)
        if meta_extra:
            meta.update(meta_extra)
        self.geom_meta[geom_id] = meta

        if name:
            self.name_to_geom[str(name)] = geom_id

        if set_current:
            self.state["current_geom"] = geom_id
        return geom_id

    def resolve_id(self, geom_id_or_name: Optional[str]) -> Optional[str]:
        if geom_id_or_name is None:
            return self.current_geom
        if geom_id_or_name in self.geometries:
            return geom_id_or_name
        gid = self.name_to_geom.get(str(geom_id_or_name))
        if gid in self.geometries:
            return gid
        return None

    def get_xyz(self, geom_id_or_name: Optional[str] = None) -> Optional[str]:
        gid = self.resolve_id(geom_id_or_name)
        if not gid:
            return None
        return self.geometries.get(gid)

    def get_charge_mult(self, geom_id_or_name: Optional[str] = None) -> Tuple[Optional[int], Optional[int]]:
        gid = self.resolve_id(geom_id_or_name)
        if not gid:
            return None, None
        meta = self.geom_meta.get(gid) or {}
        chg = meta.get("charge")
        mult = meta.get("multiplicity")
        try:
            chg = int(chg) if chg is not None else None
        except Exception:
            chg = None
        try:
            mult = int(mult) if mult is not None else None
        except Exception:
            mult = None
        return chg, mult


def get_geometry_registry(state: Dict[str, Any]) -> GeometryRegistry:
    return GeometryRegistry(state)


# -------------------------
# Tool-payload application
# -------------------------


def _store_new_geometry(
    state: Dict[str, Any],
    *,
    xyz: str,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
    geom_id: Optional[str] = None,
    name: Optional[str] = None,
    set_current: bool = True,
    overwrite: bool = False,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> str:
    reg = get_geometry_registry(state)
    return reg.register(
        xyz=xyz,
        charge=charge,
        multiplicity=multiplicity,
        geom_id=geom_id,
        name=name,
        set_current=set_current,
        overwrite=overwrite,
        meta_extra=meta_extra,
    )


def _normalize_payload_to_contract(
    payload: Any,
    *,
    geom_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize tool outputs into {new_geometry, product:{...}}.

    Policy
    ------
    - The geometry id used for state updates MUST come from the node (graph),
      not from the tool payload. Therefore, this function *never* trusts or
      forwards any geom_id-like field in the payload.
    - If `payload["product"]` is a string (e.g. "energy"), interpret it as
      a key that should be copied into product.meta_extra.
    """
    if payload is None:
        return {"product": {}}

    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                payload = json.loads(s)
            except Exception:
                payload = {"raw": payload}
        else:
            payload = {"raw": payload}

    if not isinstance(payload, dict):
        return {"product": {"raw": payload}}


    # Legacy mapping
    product: Dict[str, Any] = {}

    # xyz keys
    for k in ("xyz", "geometry_xyz", "final_geometry_xyz", "cluster_xyz"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            product["xyz"] = v
            break
    # Heuristic: first *_xyz string field
    if "xyz" not in product:
        for k, v in payload.items():
            if isinstance(k, str) and k.lower().endswith("_xyz") and isinstance(v, str) and v.strip():
                product["xyz"] = v
                break

    # meta keys
    for k in ("charge", "new_charge"):
        if k in payload and payload.get(k) is not None:
            product["charge"] = payload.get(k)
            break

    for k in ("multiplicity", "new_multiplicity"):
        if k in payload and payload.get(k) is not None:
            product["multiplicity"] = payload.get(k)
            break

    extra: Dict[str, Any] = {}

    for k in ("product", "produces"):
        if k not in payload or payload.get(k) is None:
            continue

        p = payload.get(k)

        # Case 1: "product": "energy"
        if isinstance(p, str) and p.strip():
            key = p.strip()
            if key in payload:
                extra[key] = payload.get(key)

        # Case 2: "produces": ["energy", "dipole"]
        elif isinstance(p, (list, tuple)):
            for item in p:
                if isinstance(item, str) and item.strip():
                    key = item.strip()
                    if key in payload:
                        extra[key] = payload.get(key)

    if extra:
        product.setdefault("meta_extra", {}).update(extra)

    # geom_id / name are injected from the node layer; keep them out of product.
    # (Arguments are accepted for API symmetry but intentionally unused here.)
    _ = (geom_id, name)

    return {"product": product}


def _maybe_store_geometry_from_payload(
    state: Dict[str, Any],
    payload: Any,
    *,
    input_geom_id: Optional[str] = None,
    output_geom_id: Optional[str] = None,
    output_name: Optional[str] = None,
    new_geometry: Optional[bool]
) -> Dict[str, Any]:
    """Apply tool output payload to the GeometryRegistry stored in `state`.

    Rules
    -----
    - Geometry ids are supplied by the graph/node layer (input_geom_id/output_geom_id).
      This function does NOT read geom_id from payload.
    - If payload.new_geometry is True: write to output_geom_id (or register a fresh id).
    - Else: update output_geom_id if given, otherwise update input_geom_id, otherwise
      fall back to current geometry.
    - Meta-only payloads (no xyz but meta_extra present) update the chosen geometry's meta.

    Returns a small dict for logging/testing:
      {"applied": bool, "geom_id": str|None, "new_geometry": bool}
    """
    reg = get_geometry_registry(state)
    contract = _normalize_payload_to_contract(payload, geom_id=output_geom_id, name=output_name)
    #print("contract:",contract)
    #new_geometry = bool(contract.get("new_geometry"))
    product = contract.get("product") or {}
    if not isinstance(product, dict):
        product = {}
    
    new_geometry = False
    if not input_geom_id or output_geom_id != input_geom_id:
        new_geometry = True
    xyz = product.get("xyz")
    charge = product.get("charge")
    multiplicity = product.get("multiplicity")
    meta_extra = product.get("meta_extra")
    if meta_extra is None:
        meta_extra = product.get("meta")

    meta_extra_dict = meta_extra if isinstance(meta_extra, dict) else None

    # Inherit charge/multiplicity from input geometry when payload doesn't provide them.
    if (charge is None or multiplicity is None) and input_geom_id:
        try:
            in_chg, in_mult = reg.get_charge_mult(str(input_geom_id))
            if charge is None and in_chg is not None:
                charge = in_chg
            if multiplicity is None and in_mult is not None:
                multiplicity = in_mult
        except Exception:
            pass

    # Decide write target.
    if new_geometry:
        # Prefer writing into output_geom_id if given. If it already exists, update it.
        target = reg.resolve_id(output_geom_id) if output_geom_id else None
        if target and isinstance(xyz, str) and xyz.strip():
            reg.update(
                target,
                xyz=xyz,
                charge=int(charge) if charge is not None else None,
                multiplicity=int(multiplicity) if multiplicity is not None else None,
                name=str(output_name) if output_name else None,
                set_current=True,
                meta_extra=meta_extra_dict,
            )
            return {"applied": True, "geom_id": target, "new_geometry": True, "note": "updated existing output_geom_id"}

        # Otherwise, register a new geometry id (or the provided output_geom_id).
        if not (isinstance(xyz, str) and xyz.strip()):
            return {"applied": False, "geom_id": None, "new_geometry": True, "note": "new_geometry without xyz"}

        gid = reg.register(
            xyz=xyz,
            charge=int(charge) if charge is not None else None,
            multiplicity=int(multiplicity) if multiplicity is not None else None,
            geom_id=str(output_geom_id) if output_geom_id else None,
            name=str(output_name) if output_name else None,
            set_current=True,
            overwrite=True if output_geom_id else False,
            meta_extra=meta_extra_dict,
        )
        return {"applied": True, "geom_id": gid, "new_geometry": True}

    # Non-new_geometry: update output_geom_id if given, else input_geom_id, else current.
    target = (
        reg.resolve_id(output_geom_id)
        if output_geom_id
        else (reg.resolve_id(input_geom_id) if input_geom_id else None)
    )
    if not target:
        target = reg.current_geom

    if not target:
        # No target; can only proceed if xyz exists (register defensively).
        if not (isinstance(xyz, str) and xyz.strip()):
            return {"applied": False, "geom_id": None, "new_geometry": False, "note": "no target for meta-only payload"}
        gid = reg.register(
            xyz=xyz,
            charge=int(charge) if charge is not None else None,
            multiplicity=int(multiplicity) if multiplicity is not None else None,
            geom_id=str(output_geom_id) if output_geom_id else None,
            name=str(output_name) if output_name else None,
            set_current=True,
            overwrite=False,
            meta_extra=meta_extra_dict,
        )
        return {"applied": True, "geom_id": gid, "new_geometry": True, "note": "no target; registered new"}

    # Apply xyz update if present; otherwise meta-only update.
    if isinstance(xyz, str) and xyz.strip():
        reg.update(
            target,
            xyz=xyz,
            charge=int(charge) if charge is not None else None,
            multiplicity=int(multiplicity) if multiplicity is not None else None,
            name=str(output_name) if output_name else None,
            set_current=True,
            meta_extra=meta_extra_dict,
        )
        return {"applied": True, "geom_id": target, "new_geometry": False}

    if meta_extra_dict or charge is not None or multiplicity is not None or output_name:
        reg.update(
            target,
            xyz=None,
            charge=int(charge) if charge is not None else None,
            multiplicity=int(multiplicity) if multiplicity is not None else None,
            name=str(output_name) if output_name else None,
            set_current=True,
            meta_extra=meta_extra_dict,
        )
        return {"applied": True, "geom_id": target, "new_geometry": False, "note": "meta-only update"}

    return {"applied": False, "geom_id": target, "new_geometry": False, "note": "no xyz/meta to apply"}

def state_update(state: Dict[str, Any], set: dict) -> dict:
    """Update global defaults / current geometry (dict-state only)."""
    if not isinstance(state, dict):
        return {"status": "error", "error": f"state must be a dict, got {type(state).__name__}"}

    reg = get_geometry_registry(state)

    for k, v in (set or {}).items():
        if k not in ALLOWED_STATE_KEYS:
            return {"status": "error", "error": f"Unknown state key: {k}", "allowed": sorted(ALLOWED_STATE_KEYS)}

        if k in {"default_charge", "default_multiplicity"}:
            try:
                v = int(v)
            except Exception:
                return {"status": "error", "error": f"{k} must be an integer", "value": v}
            state[k] = v
            continue

        if k == "current_geom":
            if v is None or (isinstance(v, str) and not v.strip()):
                state["current_geom"] = None
                return {"status": "ok"}

            if not isinstance(v, str):
                return {"status": "error", "error": f"current_geom must be a string geom_id/name or null, got {type(v).__name__}"}

            gid = reg.resolve_id(v)
            if not gid:
                return {"status": "error", "error": f"Unknown geometry reference for current_geom: {v!r}"}
            reg.set_current(gid)
            state["current_geom"] = gid
            continue

        state[k] = v

    return {"status": "ok"}



# Harness-level argument names a plan may legitimately use that never reach the
# tool itself: state_get_tool_args() below pops them and resolves them against
# the GeometryRegistry. Keep in sync with the pop chains in that function --
# check_tool_arg_schemas() treats these as valid plan arguments even though they
# appear in no tool's schema. Scoped per tool group, exactly as the pops are:
# allowing the solvator names globally would let a bogus `solvent=...` on
# run_opt_job (which has no solvation parameter at all) pass validation.
_GEOM_REF_ARGS = {"input_geom_id", "geom_id", "input_id", "geom_in"}
_SOLVATOR_REF_ARGS = {
    "solute_geom_id", "solute_id", "solute",
    "solvent_geom_id", "solvent_id", "solvent",
}


def _executor_consumed_args_for(tool_name: str) -> Set[str]:
    consumed: Set[str] = set()
    if tool_name in NEEDS_GEOM_SINGLE:
        consumed |= _GEOM_REF_ARGS
    if tool_name == "run_solvator_job":
        consumed |= _SOLVATOR_REF_ARGS
    return consumed

# Arguments state_get_tool_args() supplies itself from the registry, so a plan is
# not required to spell them out even when the tool schema marks them required.
_EXECUTOR_INJECTED_ARGS = {
    "geometry_xyz", "xyz", "charge", "multiplicity",
    "solute_xyz", "solvent_xyz",
}


def state_get_tool_args(
    state: Dict[str, Any],
    tool_name: str,
    overrides: Optional[dict] = None,
) -> dict:
    """Build tool args from dict-state + GeometryRegistry (no dataclass/legacy paths).

    Note: the geometry-reference names popped below are mirrored in
    _GEOM_REF_ARGS / _SOLVATOR_REF_ARGS above, which plan-time validation
    (check_tool_arg_schemas) relies on to avoid flagging them as unknown.
    """
    if not isinstance(state, dict):
        return {"status": "error", "tool_name": tool_name, "error": f"state must be a dict, got {type(state).__name__}"}

    reg = get_geometry_registry(state)
    overrides = dict(overrides or {})
    args: dict = {}

    # Single-geometry tools: accept only geom id/name ref (prefer input_geom_id)
    if tool_name in NEEDS_GEOM_SINGLE:
        ref = (
            overrides.pop("input_geom_id", None)
            or overrides.pop("geom_id", None)
            or overrides.pop("input_id", None)
            or overrides.pop("geom_in", None)
        )
        gid = reg.resolve_id(ref) if ref else reg.current_geom
        #print("ref,gid:", repr(ref), repr(gid), inspect_geom_registry_state(state))
        if not gid:
            return {"status": "error", "tool_name": tool_name, "error": "No geometry available (no current geometry set)."}
        xyz = reg.get_xyz(gid)
        if not isinstance(xyz, str) or not xyz.strip():
            return {"status": "error", "tool_name": tool_name, "error": f"Geometry {gid!r} has empty xyz."}

        args["geometry_xyz"] = xyz.strip()
        chg, mult = reg.get_charge_mult(gid)
        if "charge" not in overrides and chg is not None:
            args["charge"] = int(chg)
        if "multiplicity" not in overrides and mult is not None:
            args["multiplicity"] = int(mult)

    # Special-case: solvator-like tools requiring two geometries
    if tool_name in {"run_solvator_job"}:
        solute_ref = (
            overrides.pop("solute_geom_id", None)
            or overrides.pop("solute_id", None)
            or overrides.pop("solute", None)
        )
        solvent_ref = (
            overrides.pop("solvent_geom_id", None)
            or overrides.pop("solvent_id", None)
            or overrides.pop("solvent", None)
        )

        solute_gid = reg.resolve_id(solute_ref) if solute_ref else None
        solvent_gid = reg.resolve_id(solvent_ref) if solvent_ref else None

        if not solute_gid:
            return {"status": "error", "tool_name": tool_name, "error": f"No solute geometry for {solute_ref!r}."}
        if not solvent_gid:
            return {"status": "error", "tool_name": tool_name, "error": f"No solvent geometry for {solvent_ref!r}."}

        solute_xyz = reg.get_xyz(solute_gid)
        solvent_xyz = reg.get_xyz(solvent_gid)
        if not (isinstance(solute_xyz, str) and solute_xyz.strip()):
            return {"status": "error", "tool_name": tool_name, "error": f"Solute geometry {solute_gid!r} has empty xyz."}
        if not (isinstance(solvent_xyz, str) and solvent_xyz.strip()):
            return {"status": "error", "tool_name": tool_name, "error": f"Solvent geometry {solvent_gid!r} has empty xyz."}

        args["solute_xyz"] = solute_xyz.strip()
        args["solvent_xyz"] = solvent_xyz.strip()

    args.update(overrides)
    return args



def _geom_preview(xyz_no_header: str, max_lines: int = 6) -> str:
    lines = [ln for ln in (xyz_no_header or "").splitlines() if ln.strip()]
    if not lines:
        return "  (empty)"
    head = lines[:max_lines]
    more = "" if len(lines) <= max_lines else f"\n  ... ({len(lines) - max_lines} more lines)"
    return "  " + "\n  ".join(head) + more


def print_session_state(
    state: Dict[str, Any],
    preview_lines: int = 6,
    show_full_current: bool = True,
    max_full_lines: Optional[int] = None,
) -> None:
    """Pretty-print dict-state contents, including GeometryRegistry."""
    if not isinstance(state, dict):
        print(f"state must be dict, got {type(state).__name__}")
        return

    reg = get_geometry_registry(state)
    print("current_geom:", state.get("current_geom", None))

    if not reg.geometries:
        print("No geometries loaded.")
    else:
        # reverse name map: geom_id -> aliases
        rev_names: Dict[str, list] = {}
        for nm, gid in (reg.name_to_geom or {}).items():
            if isinstance(nm, str) and nm and isinstance(gid, str) and gid:
                rev_names.setdefault(gid, []).append(nm)

        print("\nGeometries:")
        for gid in sorted(reg.geometries.keys()):
            meta = (reg.geom_meta.get(gid) or {}) if isinstance(reg.geom_meta, dict) else {}
            nm = meta.get("name")
            if not (isinstance(nm, str) and nm.strip()):
                aliases = rev_names.get(gid) or []
                nm = aliases[0] if aliases else "-"
            aliases = ", ".join(sorted(rev_names.get(gid, []))) if rev_names.get(gid) else "-"
            mark = " (current)" if gid == reg.current_geom else ""
            chg = meta.get("charge", None)
            mult = meta.get("multiplicity", None)
            print(f"- {gid}{mark}: {nm} | q={chg}, mult={mult} | aliases: {aliases}")
            xyz = reg.get_xyz(gid) or ""
            print(_geom_preview(xyz.strip(), max_lines=preview_lines))

    identifiers = state.get("identifiers")
    if isinstance(identifiers, dict) and identifiers:
        print("\nIdentifiers:")
        for k, v in identifiers.items():
            print(f"  - {k}: {v}")

    cached_props = state.get("cached_props")
    if isinstance(cached_props, dict) and cached_props:
        print("\nCached props keys:", list(cached_props.keys())[:20])

    tu = state.get("token_usage")
    if isinstance(tu, dict) and tu.get("total", 0) > 0:
        parts = [f"{k}={v}" for k, v in tu.items() if k != "total" and v]
        print(f"\nToken usage (session): {' | '.join(parts)} | total={tu.get('total', 0)}")

    if show_full_current and reg.current_geom and reg.current_geom in reg.geometries:
        xyz = (reg.get_xyz(reg.current_geom) or "").strip().splitlines()
        print("\nCurrent geometry full XYZ:")
        if max_full_lines is None or len(xyz) <= max_full_lines:
            print("\n".join(xyz))
        else:
            print("\n".join(xyz[:max_full_lines]))
            print(f"... (truncated, {len(xyz) - max_full_lines} more lines)")


def normalize_structure_payload(payload: dict, tool_name: str) -> dict | None:
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return []
    # --- geometry string: accept xyz / geometry_xyz / final_geometry_xyz ---
    xyz = (
        payload.get("geometry_xyz")
        or payload.get("final_geometry_xyz")
        or payload.get("xyz")
    )
    if not isinstance(xyz, str) or not xyz.strip():
        return []

    # --- infer name if missing: use payload name/label, else XYZ comment line ---
    name = payload.get("name") or payload.get("label")
    if not name:
        # If xyz has header: line2 is comment
        lines = [ln.strip() for ln in xyz.splitlines() if ln.strip()]
        if len(lines) >= 2:
            try:
                int(lines[0])  # natoms?
                name = lines[1]
            except Exception:
                pass
    name = (name or tool_name).strip()

    # --- charge/multiplicity: prefer NEW_* fields, fallback to plain ones ---
    charge = payload.get("new_charge", payload.get("charge"))
    mult   = payload.get("new_multiplicity", payload.get("multiplicity"))

    return [{
        "name": name,
        "geometry_xyz": xyz_to_no_header(xyz).strip(),   # always store no-header
        "charge": int(charge) if charge is not None else None,
        "multiplicity": int(mult) if mult is not None else None,
    }]



def plan_json_to_mermaid(plan: Dict[str, Any], direction: str = "TB") -> str:
    """
    Convert a workflow plan JSON (with nodes[], needs[], kind/tool/checkpoint/llm)
    into a Mermaid flowchart.

    Features:
    - Solid arrows for normal dependencies (needs -> node)
    - Dashed arrows for checkpoint escalation/branch and tool on_error branching
    - Simple styling by node kind

    Returns:
        Mermaid text (flowchart).
    """
    nodes: List[Dict[str, Any]] = plan.get("nodes", []) or []
    node_by_id = {n.get("id"): n for n in nodes if n.get("id")}
    if not node_by_id:
        raise ValueError("plan has no nodes with 'id' fields")

    # --- helpers ---
    def safe_id(node_id: str) -> str:
        # Mermaid node identifiers must be simple; map to a safe token
        return "N_" + re.sub(r"[^A-Za-z0-9_]", "_", node_id)

    def esc_label(s: str) -> str:
        # Mermaid labels go inside quotes; keep it simple
        s = str(s)
        s = s.replace('"', "'")
        s = s.replace("\n", " ")
        return s

    # deterministic ordering
    orig_ids = [n["id"] for n in nodes if "id" in n]
    id_map = {oid: safe_id(oid) for oid in orig_ids}

    # Mermaid lines
    lines: List[str] = [f"flowchart {direction}"]

    # Collect class assignments
    class_lines: List[str] = []
    kind_to_class = {
        "tool": "toolNode",
        "checkpoint": "checkNode",
        "llm": "llmNode",
    }

    # --- declare nodes ---
    for oid in orig_ids:
        n = node_by_id[oid]
        kind = n.get("kind", "tool")
        tool = n.get("tool", "")
        # Node label
        if kind == "tool":
            label = f'{oid}\\n[{tool}]' if tool else f"{oid}\\n[tool]"
            shape = f'{id_map[oid]}["{esc_label(label)}"]'
        elif kind == "checkpoint":
            label = f"{oid}\\n[checkpoint]"
            shape = f'{id_map[oid]}{{"{esc_label(label)}"}}'
        elif kind == "llm":
            pat = n.get("pattern", "")
            label = f"{oid}\\n[LLM:{pat}]" if pat else f"{oid}\\n[LLM]"
            shape = f'{id_map[oid]}(["{esc_label(label)}"])'
        else:
            label = f"{oid}\\n[{kind}]"
            shape = f'{id_map[oid]}["{esc_label(label)}"]'

        lines.append(shape)

        cls = kind_to_class.get(kind)
        if cls:
            class_lines.append(f"class {id_map[oid]} {cls};")

    # --- dependency edges (needs) ---
    for oid in orig_ids:
        n = node_by_id[oid]
        needs = n.get("needs") or []
        for dep in needs:
            if dep in id_map:
                lines.append(f"{id_map[dep]} --> {id_map[oid]}")

    # --- checkpoint on_fail edges ---
    # Use dashed arrows for branch/escalate
    for oid in orig_ids:
        n = node_by_id[oid]
        if n.get("kind") != "checkpoint":
            continue
        on_fail = n.get("on_fail") or {}
        action = on_fail.get("action")
        if action == "branch":
            bt = on_fail.get("branch_to")
            if bt in id_map:
                lines.append(f'{id_map[oid]} -. "on_fail: branch" .-> {id_map[bt]}')
        elif action == "escalate_llm":
            ln = on_fail.get("llm_node")
            pat = on_fail.get("pattern", "")
            if ln in id_map:
                lbl = f"on_fail: escalate {pat}".strip()
                lines.append(f'{id_map[oid]} -. "{esc_label(lbl)}" .-> {id_map[ln]}')

    # --- tool on_error edges (only for branch/stop/escalate via checkpoint pattern) ---
    # We’ll visualize branch targets if present.
    for oid in orig_ids:
        n = node_by_id[oid]
        if n.get("kind") != "tool":
            continue
        on_err = n.get("on_error") or []
        for rule in on_err:
            action = rule.get("action")
            when = rule.get("when", {}) or {}
            code_in = when.get("code_in") or []
            code_tag = ",".join(code_in) if code_in else "error"
            if action == "branch":
                bt = rule.get("branch_to")
                if bt in id_map:
                    lines.append(
                        f'{id_map[oid]} -. "on_error({esc_label(code_tag)}): branch" .-> {id_map[bt]}'
                    )
            elif action == "stop":
                # Represent stop as a terminal node
                stop_id = f"{id_map[oid]}_STOP_{abs(hash(oid + str(rule.get('stop_reason','')))) % 10_000}"
                reason = rule.get("stop_reason", "stop")
                lines.append(f'{stop_id}(["STOP\\n{esc_label(reason)}"])')
                class_lines.append(f"class {stop_id} stopNode;")
                lines.append(f'{id_map[oid]} -. "on_error({esc_label(code_tag)}): stop" .-> {stop_id}')

    # --- add summarizer/final_report (if present) ---
    # These are not in nodes[] usually, so we add them as terminal steps for visualization.
    if plan.get("summarizer", {}).get("enabled"):
        sid = "N_SUMMARIZER"
        lines.append(f'{sid}(["Summarizer LLM"])')
        class_lines.append(f"class {sid} llmNode;")

        # Try to connect from final_report.collect_from or summarizer.expected_inputs
        fr = plan.get("final_report", {}) or {}
        collect_from = fr.get("collect_from") or []
        if collect_from:
            for nid in collect_from:
                if nid in id_map:
                    lines.append(f"{id_map[nid]} --> {sid}")
        else:
            # fall back: connect from last node in list
            lines.append(f"{id_map[orig_ids[-1]]} --> {sid}")

    # --- styles ---
    lines.append("")
    lines.append("%% Styles")
    lines.append("classDef toolNode fill:#eef,stroke:#55f,stroke-width:1px;")
    lines.append("classDef checkNode fill:#ffe,stroke:#aa0,stroke-width:1px;")
    lines.append("classDef llmNode fill:#efe,stroke:#5a5,stroke-width:1px;")
    lines.append("classDef stopNode fill:#fee,stroke:#f55,stroke-width:1px;")
    lines.extend(class_lines)

    return "\n".join(lines)


def download_mermaid_ink_png(encoded: str, out_path: str) -> None:
    # encoded is the “encoded string” shown in Mermaid Live Editor URL
    url = f"https://mermaid.ink/img/{encoded}"
    urlretrieve(url, out_path)

def parse_json_only(content: str) -> dict:
    """Extract and parse the first JSON object from model output.

    Handles:
    - Plain JSON
    - JSON wrapped in ```json ... ``` fences
    - <think>...</think> reasoning blocks (qwen / deepseek models)
    - Preamble / postamble text around the JSON object
    """
    import re as _re

    content = content.strip()

    # Strip <think>...</think> reasoning blocks produced by some local models
    content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

    # Strip ``` code fences (```json ... ``` or ``` ... ```)
    fence = _re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", content, _re.DOTALL)
    if fence:
        content = fence.group(1).strip()

    # Fast path: content is already valid JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Slow path: find the first top-level { ... } block using brace counting
    start = content.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output:\n{content[:300]}")

    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(content[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(content[start: i + 1])

    raise ValueError(f"Unmatched braces in model output:\n{content[:300]}")



def get_latest_plan_pkg(state):
    wf = _state_get(state, "workflow", None)
    if wf is None:
        return None, None

    # workflow may be a dataclass/object or a dict
    name = wf.get("latest_plan_name") if isinstance(wf, dict) else getattr(wf, "latest_plan_name", None)
    if not name:
        return None, None

    plans = wf.get("plans", {}) if isinstance(wf, dict) else getattr(wf, "plans", {})
    if not isinstance(plans, dict):
        return None, None

    plan_pkg = plans.get(name)
    return name, plan_pkg

def ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_latest_plan_pkg(state):
    if isinstance(state, dict):
        name = state.get("latest_plan_name")
        return (state.get("plans") or {}).get(name) if name else None
    name = getattr(state, "latest_plan_name", None)
    return getattr(state, "plans", {}).get(name) if name else None


def _get_latest_or_approved_plan_pkg(state):
    if isinstance(state, dict):
        name = state.get("approved_plan_name") or state.get("latest_plan_name")
        return (state.get("plans") or {}).get(name) if name else None
    name = getattr(state, "approved_plan_name", None) or getattr(state, "latest_plan_name", None)
    return getattr(state, "plans", {}).get(name) if name else None


def save_plan_mermaid_files(state, out_dir="plans", fmt="png", direction="TB", background="white", width=1800):
    """
    Save Mermaid source and optionally render an image for the latest plan.
    fmt: "mmd" | "png" | "svg" | "pdf"
    """
    plan_name, plan_pkg = get_latest_plan_pkg(state)
    if not plan_pkg:
        raise RuntimeError("No latest plan in state. Run 'plan' first.")

    plan = plan_pkg["plan"]
    out_dir = ensure_dir(out_dir)

    # 1) Save plan JSON too (handy)
    json_path = out_dir / f"{plan_name}.plan.json"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 2) Mermaid text
    mermaid = plan_json_to_mermaid(plan, direction=direction)
    mmd_path = out_dir / f"{plan_name}.mmd"
    mmd_path.write_text(mermaid.strip() + "\n", encoding="utf-8")

    # 3) Render image if requested
    saved = {"plan_json": str(json_path), "mmd": str(mmd_path)}
    if fmt != "mmd":
        img_path = out_dir / f"{plan_name}.{fmt}"
        render_mermaid_with_mmdc(mermaid, str(img_path), background=background, width=width)
        saved[fmt] = str(img_path)

    # 4) Store paths back into plan_pkg for convenience
    plan_pkg.setdefault("ui", {}).setdefault("saved_files", {})
    plan_pkg["ui"]["saved_files"].update(saved)

    return saved


def render_mermaid_with_mmdc(mermaid_text: str, out_image: str, *, background: str = "white", width: int = 1600) -> None:
    """
    Render Mermaid using mermaid-cli (mmdc). Requires `mmdc` on PATH.
    Saves:
      - diagram.mmd (next to out_image)
      - out_image (.png/.svg/.pdf supported by mmdc)
    """
    out_path = Path(out_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_path = out_path.with_suffix(".mmd")
    in_path.write_text(mermaid_text.strip() + "\n", encoding="utf-8")

    cmd = ["mmdc", "-i", str(in_path), "-o", str(out_path), "-b", background, "--width", str(width)]
    subprocess.run(cmd, check=True)

def handle_planimg_command(state, line: str):
    parts = line.strip().split()
    fmt = parts[1].lower() if len(parts) > 1 else "png"
    if fmt not in ("mmd", "png", "svg", "pdf"):
        print("Usage: planimg [mmd|png|svg|pdf]")
        return

    try:
        saved = save_plan_mermaid_files(state, out_dir="plans", fmt=fmt, direction="TB")
        print("Saved:")
        for k, v in saved.items():
            print(f"  {k}: {v}")
    except FileNotFoundError:
        print("Could not find 'mmdc' (mermaid-cli). Install with:")
        print("  npm install -g @mermaid-js/mermaid-cli")
    except Exception as e:
        print("Failed:", e)


def summarize_workflow_state(state, max_chars: int = 3000) -> str:
    """
    Summarize the current workflow context from:
      state.workflow: WorkflowState
        - latest_plan_name: Optional[str]
        - approved_plan_name: Optional[str]
        - plans: Dict[str, Dict[str, Any]]  # plan_pkg
          where plan_pkg typically contains: {"plan": {...}, "review": {...}, "ui": {...}}

    Returns a compact JSON string (truncated to max_chars).
    """
    wf = getattr(state, "workflow", None)
    if wf is None:
        return "No workflow state attached to SessionState."

    latest_name: Optional[str] = getattr(wf, "latest_plan_name", None)
    if not latest_name:
        return "No workflow plan in state."

    plans = getattr(wf, "plans", {}) or {}
    plan_pkg = plans.get(latest_name)
    if not plan_pkg:
        return f"Latest plan '{latest_name}' not found in state.workflow.plans."

    plan = plan_pkg.get("plan", {}) or {}
    review = plan_pkg.get("review", {}) or {}
    ui = plan_pkg.get("ui", {}) or {}

    summary_obj = {
        "latest_plan_name": latest_name,
        "latest_plan_title": plan.get("title"),
        "approved_plan_name": getattr(wf, "approved_plan_name", None),
        "review_summary": review.get("summary", []),
        "key_defaults": plan.get("defaults", {}),
        "saved_files": ui.get("saved_files", {}),
        "plans_count": len(plans),
        "last_run_id": getattr(wf, "last_run_id", None),
        "runs_count": len(getattr(wf, "runs", []) or []),
    }

    s = json.dumps(summary_obj, ensure_ascii=False, indent=2)
    return s[:max_chars] + ("\n...<truncated>..." if len(s) > max_chars else "")



def save_plan_json_file_from_pkg(plan_pkg: dict, out_path: str) -> str:
    """Save plan_pkg['plan'] to out_path; also records it in plan_pkg['ui']['saved_files']."""
    plan = plan_pkg.get("plan") or {}
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan_pkg.setdefault("ui", {}).setdefault("saved_files", {})
    plan_pkg["ui"]["saved_files"]["plan_json"] = str(out_p)
    return str(out_p)


def save_plan_json_file(state, out_path: str) -> str:
    """Save the latest (or approved) plan_pkg['plan'] to a JSON file.

    Returns the written path as string and also records it in plan_pkg['ui']['saved_files']['plan_json'].
    """
    plan_pkg = _get_latest_or_approved_plan_pkg(state)
    if not plan_pkg:
        raise ValueError("No plan found in state. Run 'plan' first.")
    return save_plan_json_file_from_pkg(plan_pkg, out_path)


def handle_plansave_command(state, line: str):
    """REPL command: plansave [<path>] | plansave <plan_name|approved|latest> <path>

    Examples:
      - plansave
      - plansave plans/myplan.json
      - plansave approved plans/approved_plan.json
      - plansave MyPlanName plans/MyPlanName.json
    """
    parts = line.strip().split()
    plans = (state.get("plans") or {}) if isinstance(state, dict) else getattr(state, "plans", {})  # legacy support

    which: str | None = None
    out_path: str | None = None

    if len(parts) == 1:
        plan_pkg = _get_latest_or_approved_plan_pkg(state)
        if not plan_pkg:
            print("No plan found. Use 'plan' first.")
            return
        plan = plan_pkg.get("plan") or {}
        plan_name = plan.get("name") or plan.get("title") or "plan"
        out_path = str(Path("plans") / f"{plan_name}.json")
    elif len(parts) == 2:
        out_path = parts[1]
        plan_pkg = _get_latest_or_approved_plan_pkg(state)
        if not plan_pkg:
            print("No plan found. Use 'plan' first.")
            return
    else:
        which = parts[1]
        out_path = parts[2]
        if which in ("approved", "latest"):
            plan_pkg = _get_latest_or_approved_plan_pkg(state) if which == "approved" else _get_latest_plan_pkg(state)
        else:
            plan_pkg = plans.get(which)
        if not plan_pkg:
            print(f"Plan not found for selector: {which!r}. Available: {list(plans.keys())}")
            return

    try:
        written = save_plan_json_file_from_pkg(plan_pkg, out_path)
        print(f"Saved plan JSON: {written}")
    except Exception as e:
        print("Failed:", e)


def build_plan_review(plan: dict) -> dict:
    """Deterministic, non-LLM review summary of a plan JSON."""
    nodes = plan.get("nodes", []) or []
    tool_nodes = [n for n in nodes if n.get("kind") == "tool"]
    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]
    check_nodes = [n for n in nodes if n.get("kind") == "checkpoint"]

    tools_used = []
    for n in tool_nodes:
        t = n.get("tool")
        if t and t not in tools_used:
            tools_used.append(t)

    defaults = plan.get("defaults", {}) or {}
    final_report = plan.get("final_report", {}) or {}
    artifacts_to_save = plan.get("artifacts_to_save", []) or []

    summary_lines = []
    title = plan.get("title") or plan.get("name") or "Untitled plan"
    goal = plan.get("goal") or ""

    summary_lines.append(f"Title: {title}")
    if goal:
        summary_lines.append(f"Goal: {goal}")

    # key defaults (keep short)
    key_defaults = {}
    for k in ("solvent", "temperature_K", "standard_state"):
        if k in defaults:
            key_defaults[k] = defaults.get(k)
    if key_defaults:
        summary_lines.append(f"Defaults: {key_defaults}")

    summary_lines.append(
        f"Nodes: total={len(nodes)}, tool={len(tool_nodes)}, checkpoint={len(check_nodes)}, llm={len(llm_nodes)}"
    )
    if tools_used:
        summary_lines.append(f"Tools used: {tools_used}")

    if artifacts_to_save:
        summary_lines.append(f"Artifacts to save: {artifacts_to_save}")

    fr_fields = final_report.get("fields") or []
    fr_from = final_report.get("collect_from") or []
    if fr_fields or fr_from:
        summary_lines.append(f"Final report: fields={fr_fields}, collect_from={fr_from}")

    return {
        "summary": summary_lines,         # consumed by summarize_workflow_state(...)
        "tools_used": tools_used,
        "counts": {
            "nodes_total": len(nodes),
            "nodes_tool": len(tool_nodes),
            "nodes_checkpoint": len(check_nodes),
            "nodes_llm": len(llm_nodes),
        },
        "final_report": {
            "fields": fr_fields,
            "collect_from": fr_from,
            "format": final_report.get("format", "json"),
        },
    }


def check_struct(plan: dict, valid_tools: set) -> Dict[str, bool]:
    """Minimal structural validity every plan must pass, independent of task type.

    Shared between the test harness (test_success_rate.py, which passes its own
    _VALID_TOOLS) and production agent.py (which passes a set built from its own
    CLIENT_SIDE_TOOL_FUNCS + the live MCP session's tool list) — this is the
    validation gate a plan-retry loop checks against in both places.
    """
    nodes     = plan.get("nodes") or []
    artifacts = plan.get("artifacts_to_save") or []
    # Auto-artifact mode: artifacts_to_save may be empty; final_report.fields holds outputs.
    if not artifacts:
        fr = plan.get("final_report") or {}
        artifacts = fr.get("fields") or []
    tool_nodes_valid = all(
        n.get("tool") in valid_tools
        for n in nodes if n.get("kind") == "tool"
    )
    return {
        "is_dict":             isinstance(plan, dict),
        "has_name":            bool(plan.get("name")),
        "has_nodes":           len(nodes) > 0,
        "has_artifacts":       len(artifacts) > 0,
        "nodes_have_id":       all(n.get("id") for n in nodes),
        "nodes_have_kind":     all(n.get("kind") for n in nodes),
        "tools_valid":         tool_nodes_valid,
        "dep_graph_valid":     _check_dep_graph(plan),
    }


def _check_dep_graph(plan: dict) -> bool:
    """Return True if every tool node's input_id was produced by a prior node or is a declared geom_id."""
    nodes = plan.get("nodes") or []
    geom_ids = set(plan.get("geom_ids") or [])
    output_ids: set = set()
    for n in nodes:
        if n.get("output_id"):
            output_ids.add(n["output_id"])
    for n in nodes:
        if n.get("kind") != "tool":
            continue
        input_id = n.get("input_id") or (n.get("args") or {}).get("input_geom_id")
        if input_id and input_id not in output_ids and input_id not in geom_ids:
            return False
    return True


def _schema_from_callable(fn) -> Optional[dict]:
    """Derive a JSON-Schema-shaped dict from a Python function's own signature.

    Client-side (skill-owned) tools declare their contract in their type hints;
    this reads it rather than requiring a separately maintained schema, matching
    the client_tools ownership convention in skills.py.
    """
    import inspect as _inspect
    try:
        sig = _inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    try:
        import typing as _typing
        hints = _typing.get_type_hints(fn)
    except Exception:
        hints = {}

    _PY_TO_JSON = {str: "string", int: "integer", float: "number",
                   bool: "boolean", list: "array", dict: "object"}
    properties: Dict[str, dict] = {}
    required: List[str] = []
    accepts_any = False
    for p in sig.parameters.values():
        if p.kind is _inspect.Parameter.VAR_KEYWORD:
            accepts_any = True
            continue
        if p.kind is _inspect.Parameter.VAR_POSITIONAL or p.name == "state":
            continue
        properties[p.name] = {"type": _PY_TO_JSON.get(hints.get(p.name), None)}
        if p.default is _inspect.Parameter.empty:
            required.append(p.name)
    return {"properties": properties, "required": required, "accepts_any": accepts_any}


def build_tool_schemas(mcp_tools=None, client_side_funcs: Optional[dict] = None) -> Dict[str, dict]:
    """Collect argument schemas for every tool a plan may reference.

    Both sources are already self-describing, so nothing here is hand-authored:
      - MCP tools expose a real JSON Schema as `.inputSchema`, generated
        server-side by FastMCP from the tool functions' type hints. Pass the
        `.tools` list from `await session.list_tools()`.
      - Client-side tools (including skill-owned ones from
        collect_skill_client_tools()) are introspected from their signatures.

    Returns {tool_name: {"properties": {...}, "required": [...], "accepts_any": bool}}.
    """
    schemas: Dict[str, dict] = {}
    for t in (mcp_tools or []):
        schema = getattr(t, "inputSchema", None)
        if isinstance(schema, dict):
            schemas[getattr(t, "name", "")] = {
                "properties": schema.get("properties") or {},
                "required": list(schema.get("required") or []),
                "accepts_any": False,
            }
    for name, fn in (client_side_funcs or {}).items():
        derived = _schema_from_callable(fn)
        if derived is not None:
            schemas[name] = derived
    schemas.pop("", None)
    return schemas


_ANY_REF_RE = re.compile(r'\$\(([^)]+)\)')
# The executor expands exactly these two prefixes (_expand_settings_refs and
# _expand_artifact_refs). Anything else is passed through verbatim.
_EXPANDABLE_REF_PREFIXES = ("artifacts.", "settings.")


def _unexpandable_refs(value: Any) -> List[str]:
    """References the executor will NOT expand, and so will pass on literally.

    A reference the executor does not recognise -- for example
    "$(load_methanol.geometry_xyz)", scoped by node id rather than by
    artifacts/settings -- is not an error at dispatch time. It is silently handed
    to the tool as the literal string "$(load_methanol.geometry_xyz)", which the
    tool then treats as data: in one observed case it was written into an .xyz
    file and rejected by ORCA as a malformed coordinate line, thirteen
    milliseconds in, with no chemistry performed.

    This is a form check, not a value-type check. It asks only whether a
    reference *can* resolve, never what the resolved value should look like, so
    it does not carry the false-positive risk that keeps type enforcement out of
    scope here: a reference with an unexpandable prefix is wrong unconditionally.
    """
    out: List[str] = []
    if isinstance(value, str):
        for m in _ANY_REF_RE.finditer(value):
            ref = m.group(1)
            if not ref.startswith(_EXPANDABLE_REF_PREFIXES):
                out.append(ref)
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_unexpandable_refs(v))
    elif isinstance(value, list):
        for v in value:
            out.extend(_unexpandable_refs(v))
    return out


def check_tool_arg_schemas(plan: dict,
                           tool_schemas: Dict[str, dict]) -> Tuple[Dict[str, bool], str]:
    """Validate every tool node's args against the tool's real declared schema.

    Closes the gap check_struct leaves: check_struct validates plan *shape*
    (node ids, kinds, registered tool names, geometry dependency graph) but never
    compares an argument against what the tool actually accepts. Unsupported
    argument names are the single most common recorded planner mistake (see
    bug_reports/), and today they surface only as a runtime TypeError after
    execution has already begun.

    Scope is deliberately narrow -- argument NAMES and REQUIRED arguments only,
    no value-type or enum enforcement -- to keep false positives near zero.

    Template references are resolved at execution time, not plan time
    (_expand_settings_refs/_expand_artifact_refs run inside
    execute_tool_from_node_spec), so a plan legitimately carries literal
    "$(settings.ncores)"/"$(artifacts.KEY)" strings here. Because this check
    only inspects argument NAMES, such values need no special handling: they
    satisfy presence and are never type-inspected.

    Returns (checks, detail) where detail names the offending
    node.tool.argument triples -- a bare "tool_args_known: False" is not
    actionable feedback for the planner on its own.
    """
    unknown_args: List[str] = []
    missing_required: List[str] = []
    bad_refs: List[str] = []

    for node in (plan.get("nodes") or []):
        if node.get("kind") != "tool":
            continue
        for arg_name, value in (node.get("args") or {}).items():
            for ref in _unexpandable_refs(value):
                bad_refs.append(f"{node.get('id')}.{arg_name} -> $({ref})")
        tool_name = node.get("tool")
        schema = tool_schemas.get(tool_name)
        if not schema:
            continue   # unregistered tool names are check_struct's tools_valid job
        args = node.get("args") or {}
        if not isinstance(args, dict):
            continue
        properties = schema.get("properties") or {}
        consumed = _executor_consumed_args_for(tool_name)
        if properties and not schema.get("accepts_any"):
            for arg_name in args:
                # Harness-level geometry references (input_geom_id, solute_id, …)
                # are resolved and popped by state_get_tool_args before dispatch,
                # so they are valid in a plan despite being in no tool schema.
                if arg_name in consumed:
                    continue
                if arg_name not in properties:
                    unknown_args.append(f"{node.get('id')}.{tool_name}.{arg_name}")
        for req in (schema.get("required") or []):
            # Geometry-bearing args are injected by the executor from the
            # registry, so a plan is not required to spell them out.
            if req in _EXECUTOR_INJECTED_ARGS:
                continue
            if req not in args:
                missing_required.append(f"{node.get('id')}.{tool_name}.{req}")

    detail_parts: List[str] = []
    if unknown_args:
        detail_parts.append(
            "These arguments are not accepted by the tool they were passed to "
            "(node.tool.argument): " + ", ".join(unknown_args)
            + ". Remove them or use the tool's documented argument names."
        )
    if missing_required:
        detail_parts.append(
            "These required arguments are missing (node.tool.argument): "
            + ", ".join(missing_required) + "."
        )
    if bad_refs:
        detail_parts.append(
            "These references cannot be resolved by the executor and would be "
            "passed to the tool as literal text (node.argument -> reference): "
            + ", ".join(bad_refs)
            + ". Only $(artifacts.KEY) and $(settings.KEY) are expanded. To use "
            "an upstream node's output, name the artifact key it produces, as "
            "$(artifacts.KEY)."
        )

    return (
        {
            "tool_args_known": not unknown_args,
            "tool_args_required_present": not missing_required,
            "tool_arg_refs_resolvable": not bad_refs,
        },
        " ".join(detail_parts),
    )


_C_PLACEHOLDER_RE = re.compile(r"\{C[^}]*\}")


def _has_c_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_C_PLACEHOLDER_RE.search(value))
    if isinstance(value, dict):
        return any(_has_c_placeholder(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_c_placeholder(v) for v in value)
    return False


def _references_every_compound(node_tmpl: dict, compounds: List[dict]) -> bool:
    """True if this post_template node's needs/needs_artifacts reference
    something for every compound -- the signal that it is a genuine
    once-per-request aggregation step rather than a per-compound node that
    merely lost its {C} placeholder."""
    needs_fields = list(node_tmpl.get("needs") or []) + list(node_tmpl.get("needs_artifacts") or [])
    if not needs_fields or not compounds:
        return False
    slugs = [
        c.get("id") or re.sub(r"[^a-z0-9]+", "_", str(c.get("name", "")).lower()).strip("_")
        for c in compounds
    ]
    joined = " ".join(str(n) for n in needs_fields)
    return all(slug and slug in joined for slug in slugs)


def find_duplicate_id_fix(error_text: str, plan_raw: dict) -> Optional[dict]:
    """Deterministically repair the one root-caused template-expansion failure:
    a post_template node with a fixed (non-{C}) id whose applies_to is missing
    or "all", which expand_template() duplicates once per compound.

    Returns a patched deep copy, or None if the safety heuristic cannot
    confidently confirm "once" is the right fix -- a duplicate id could equally
    mean a per-compound node lost its {C} placeholder, where forcing "once"
    would silently drop N-1 compounds' work.
    """
    m = re.search(r"Duplicate node IDs after template expansion: \{(.+)\}", error_text or "")
    if not m:
        return None
    dup_ids = {s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()}
    if not dup_ids:
        return None

    post_template = ((plan_raw.get("template") or {}).get("post_template")) or []
    compounds = plan_raw.get("compounds") or []
    patched = copy.deepcopy(plan_raw)
    fixed_any = False

    for i, node_tmpl in enumerate(post_template):
        if node_tmpl.get("id") not in dup_ids:
            continue
        if _has_c_placeholder(node_tmpl.get("id")):
            continue
        if node_tmpl.get("applies_to") in ("once", "targets_only", "references_only"):
            continue
        if not _references_every_compound(node_tmpl, compounds):
            continue
        patched["template"]["post_template"][i]["applies_to"] = "once"
        fixed_any = True

    return patched if fixed_any else None


def try_offline_template_patch(checks: Dict[str, bool], error_text: str,
                               plan_raw: dict) -> Optional[dict]:
    """Repair a template-expansion failure in code, with no extra planner call.

    Returns a fully expanded, re-validated plan, or None if the failure is not
    the known-and-root-caused pattern (caller falls through to a real LLM
    retry). Saves a full planning round-trip on the one failure mode that is
    deterministically fixable.
    """
    if checks.get("template_expansion_ok") is not False:
        return None
    patched_raw = find_duplicate_id_fix(error_text, plan_raw)
    if patched_raw is None:
        return None
    try:
        from build_graph_from_plan import expand_template
        return expand_template(patched_raw)
    except Exception:
        return None


def format_plan_validation_feedback(checks: Dict[str, bool], error: str = "") -> str:
    """Turn a failed structural-check result into an LLM-readable retry message.

    Used by the plan-retry loop (test harness and production agent.py) to tell
    the planner exactly what was wrong with its previous plan before asking it
    to try again, instead of just discarding the failure.
    """
    failed = [k for k, v in checks.items() if not v]
    lines = [
        "Your previous plan failed structural validation on these checks: "
        + ", ".join(failed) + ".",
    ]
    if error:
        lines.append(f"Error detail: {error}")
    lines.append(
        "Return a corrected plan as a single JSON object, fixing these issues. "
        "Follow the same schema as before."
    )
    return "\n".join(lines)


def print_plan_review(plan_pkg: dict) -> None:
    """Human-facing plan summary right after planning."""
    review = (plan_pkg or {}).get("review", {}) or {}
    lines = review.get("summary") or []
    if not lines:
        print("\n[Plan]\n(no review summary)\n")
        return
    print("\n[Plan summary]\n")
    for ln in lines:
        print(f"- {ln}")
    print("")



def _mcp_content_to_text(mcp_res: Any) -> str:
    parts: List[str] = []
    content = getattr(mcp_res, "content", None)
    if not content:
        return ""
    for item in content:
        txt = getattr(item, "text", None)
        if isinstance(txt, str):
            parts.append(txt)
        else:
            parts.append(repr(item))
    return "".join(parts)

def _normalize_tool_name(tool_name: str) -> str:
    if not isinstance(tool_name, str):
        return tool_name
    t = tool_name.strip()
    for prefix in ("functions.", "function."):
        if t.startswith(prefix):
            return t[len(prefix):]
    return t

def _expand_settings_refs(obj: Any, settings: dict) -> Any:
    """Expand '$(settings.KEY)' template strings using plan settings values.

    Full-match replaces the entire value (preserving bool/int type).
    Partial-match replaces within a string (result is always str).
    Works recursively for dicts and lists.
    """
    import re as _re
    if isinstance(obj, str):
        m = _re.fullmatch(r'\$\(settings\.([^)]+)\)', obj)
        if m:
            return settings.get(m.group(1), obj)
        return _re.sub(
            r'\$\(settings\.([^)]+)\)',
            lambda match: str(settings.get(match.group(1), match.group(0))),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_settings_refs(v, settings) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_settings_refs(v, settings) for v in obj]
    return obj


def _expand_artifact_refs(obj: Any, artifacts: dict) -> Any:
    """Expand '$(artifacts.KEY)' template strings using state['artifacts'] values.

    Mirrors _expand_settings_refs exactly, against artifacts instead of plan
    settings. Lets a kind:"tool" node consume a value computed by an earlier
    node (e.g. a Gibbs free energy) as an argument, the same way calc_expr/llm
    nodes already do via needs_artifacts. Recurses through dicts/lists so
    nested structures (e.g. a "references": [{...}, ...] list) are covered.
    """
    import re as _re
    if isinstance(obj, str):
        m = _re.fullmatch(r'\$\(artifacts\.([^)]+)\)', obj)
        if m:
            return artifacts.get(m.group(1), obj)
        return _re.sub(
            r'\$\(artifacts\.([^)]+)\)',
            lambda match: str(artifacts.get(match.group(1), match.group(0))),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_artifact_refs(v, artifacts) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_artifact_refs(v, artifacts) for v in obj]
    return obj


async def execute_tool_from_node_spec(*, state: Dict[str, Any], node_spec: Dict[str, Any]) -> Dict[str, Any]:
    tool_raw = node_spec.get("tool")
    tool_name = _normalize_tool_name(tool_raw)

    # --- geometry routing (node-injected; NEVER from tool payload) ---
    input_geom_id = node_spec.get("input_id") or node_spec.get("input_geom_id") or node_spec.get("geom_in")
    output_geom_id = node_spec.get("output_id") or node_spec.get("output_geom_id") or node_spec.get("geom_out")
    output_name = node_spec.get("output_name") or node_spec.get("geom_out_name") or None

    # --- overrides / args ---
    overrides = node_spec.get("args")
    if overrides is None:
        overrides = node_spec.get("overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        return {"status": "error", "tool": tool_name, "error": "node_spec.args must be a dict"}

    # Expand $(settings.KEY) references using plan settings.
    # _plan_settings is injected by build_graph_from_plan.make_node; fall back to state["plan"].
    plan_settings = (
        node_spec.get("_plan_settings")
        or ((state.get("plan") or {}).get("settings") if isinstance(state, dict) else None)
        or {}
    )
    if plan_settings:
        overrides = _expand_settings_refs(overrides, plan_settings)

    # Expand $(artifacts.KEY) references using previously-computed artifacts.
    artifacts_state = state.get("artifacts") if isinstance(state, dict) else None
    if artifacts_state:
        overrides = _expand_artifact_refs(overrides, artifacts_state)
    new_geometry = False
    if not input_geom_id or output_geom_id != input_geom_id:
        new_geometry = True
    # Inject geometry_xyz/charge/multiplicity from GeometryRegistry using input_geom_id when needed.
    # Do it *before* state_get_tool_args so its mapping/validation can run normally.
    try:
        if tool_name in NEEDS_GEOM_SINGLE and input_geom_id:
            reg = get_geometry_registry(state)
            xyz = reg.get_xyz(str(input_geom_id))
            chg, mult = reg.get_charge_mult(str(input_geom_id))
            if xyz and ("geometry_xyz" not in overrides) and ("xyz" not in overrides):
                overrides["geometry_xyz"] = xyz
            if chg is not None and ("charge" not in overrides):
                overrides["charge"] = chg
            if mult is not None and ("multiplicity" not in overrides):
                overrides["multiplicity"] = mult

    except Exception as e:
        return {"status": "error", "tool": tool_name, "error": f"Failed to inject geometry from registry: {e}", "args": overrides}

    # Build final args (may perform additional tool-specific normalization/validation)
    args = state_get_tool_args(state, tool_name, overrides)
    if isinstance(args, dict) and args.get("status") == "error" and args.get("tool_name") == tool_name:
        return {"status": "error", "tool": tool_name, "error": args.get("error"), "args": overrides}

    # MCP tools: session must be on state
    session = state.get("session") if isinstance(state, dict) else getattr(state, "session", None)
    if session is None:
        return {"status": "error", "tool": tool_name, "error": "Missing state['session']", "args": args}

    client_side_tools = state.get("client_side_tools")
    if not isinstance(client_side_tools, dict):
        client_side_tools = {}

    # ---------------- client-side tool ----------------
    if tool_name in client_side_tools:
        fn = client_side_tools[tool_name]
        import inspect as _inspect
        try:
            sig = _inspect.signature(fn)
            params = list(sig.parameters.values())
            has_var_keyword = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in params
            )
            accepted_names = {
                p.name for p in params
                if p.kind in (_inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              _inspect.Parameter.KEYWORD_ONLY)
            }
        except (ValueError, TypeError):
            sig, params, accepted_names = None, [], set()
            has_var_keyword = True

        # Keep only arguments this function actually declares. Replaces a former
        # hardcoded {"wall_timeout_seconds","ncores","job_label"} blocklist —
        # signature-derived, so it covers every MCP-only or misnamed argument,
        # not just three known-bad ones.
        if has_var_keyword:
            _client_args = args
            _dropped: List[str] = []
        else:
            _client_args = {k: v for k, v in args.items() if k in accepted_names}
            _dropped = sorted(set(args) - accepted_names)
        if _dropped:
            # Surface, don't swallow: a dropped argument means the planner asked
            # for something this tool cannot do (check_tool_arg_schemas catches
            # this at plan time; this is the runtime backstop).
            print(f"  [tool_args] {tool_name}: ignoring unsupported argument(s) "
                  f"{_dropped} (not in signature)")

        # Only pass `state` positionally to functions that actually declare it
        # first. Client-side tools are heterogeneous: build_dimer_xyz/
        # set_geometry_xyz/build_approach_scan_geometries take state first, while
        # compute_pka_calibrated/structure_add_remove_proton/name_to_geometry_xyz
        # do not. A blind `except TypeError: fn(state, **args)` fallback used to
        # bind `state` to whatever the first parameter happened to be — producing
        # a spurious "got multiple values for argument 'metal'" that masked the
        # real error, or, worse, silently succeeding with `state` bound to a real
        # parameter (e.g. pubchem_get_basic_properties(name=...)).
        _first_is_state = bool(params) and params[0].name == "state" \
            and "state" not in _client_args
        if _first_is_state:
            out = fn(state, **_client_args)
        else:
            out = fn(**_client_args)

        if isinstance(out, dict):
            payload = out
        elif isinstance(out, str):
            try:
                payload = json.loads(out)
                if not isinstance(payload, dict):
                    payload = {"status": "ok", "tool": tool_name, "value": payload}
            except Exception:
                payload = {"status": "ok", "tool": tool_name, "text": out}
        else:
            payload = {"status": "ok", "tool": tool_name, "value": out}

        # Update GeometryRegistry from payload using node-injected ids (input_id/output_id)
        try:
            if payload.get("status") == "ok":
                _maybe_store_geometry_from_payload(
                    state,
                    payload,
                    input_geom_id=str(input_geom_id) if input_geom_id else None,
                    output_geom_id=str(output_geom_id) if output_geom_id else None,
                    output_name=str(output_name) if output_name else None,
                    new_geometry = new_geometry
                )
        except Exception as e:
            # do not fail the tool run; attach warning
            payload.setdefault("warnings", []).append(f"GeometryRegistry update failed: {e}")

        payload.setdefault("tool", tool_name)
        payload.setdefault("args", args)

        return {
    "last_tool_result": payload,
    "last_status": payload.get("status"),
    "geometries": state.get("geometries"),
    "geom_meta": state.get("geom_meta"),
    "name_to_geom": state.get("name_to_geom"),
    "current_geom": state.get("current_geom"),
        }

    # ---------------- MCP tool ----------------
    #print("args:", args)
    mcp_res = await session.call_tool(tool_name, args)
    text = _mcp_content_to_text(mcp_res)

    try:
        payload = json.loads(text)
        if not isinstance(payload, dict):
            payload = {"status": "ok", "tool": tool_name, "value": payload}
    except Exception:
        payload = {"status": "ok", "tool": tool_name, "text": text}

    # FastMCP wraps tool exceptions as plain text with "Error executing tool ..."
    # Promote these to status=error so downstream geometry/artifact logic is skipped.
    if payload.get("status") == "ok" and payload.get("text"):
        _t = str(payload["text"])
        if _t.startswith("Error executing tool") or _t.startswith("Error:"):
            payload["status"] = "error"
            payload["error"] = _t
            del payload["text"]

    # Update GeometryRegistry from payload using node-injected ids (input_id/output_id)
    try:
        if payload.get("status") == "ok":
            _maybe_store_geometry_from_payload(
                    state,
                    payload,
                    input_geom_id=str(input_geom_id) if input_geom_id else None,
                    output_geom_id=str(output_geom_id) if output_geom_id else None,
                    output_name=str(output_name) if output_name else None,
                    new_geometry = new_geometry
                )
    except Exception as e:
        payload.setdefault("warnings", []).append(f"GeometryRegistry update failed: {e}")

    payload.setdefault("tool", tool_name)
    payload.setdefault("args", args)

    return {
    "last_tool_result": payload,
    "status": payload.get("status"),
    "geometries": state.get("geometries"),
    "geom_meta": state.get("geom_meta"),
    "name_to_geom": state.get("name_to_geom"),
    "current_geom": state.get("current_geom"),
        }


async def run_tool_node(state, node_spec) -> dict:
    """LangGraph ToolNodeRunner: deterministic tool execution (no LLM)."""
    payload = await execute_tool_from_node_spec(state=state, node_spec=node_spec)
    return payload if isinstance(payload, dict) else {"status": "ok", "tool": node_spec.get("tool"), "value": payload}


def persist_node_outputs(state: Dict[str, Any], node_id: str, node_spec: Dict[str, Any], node_run: Dict[str, Any]) -> None:
    state.setdefault("node_results", {})[node_id] = node_run

    state.setdefault("artifacts", {})
    # Convention A: tool returns {"artifacts": {...}}
    for p in node_run.get("tool_payloads", []):
        if isinstance(p, dict) and isinstance(p.get("artifacts"), dict):
            state["artifacts"].update(p["artifacts"])

    # Convention B: node spec declares what it produces and where to pick it from
    # e.g. node_spec["produces"] = {"opt_geom": "tool_payloads[-1].geometry_xyz"}
    produces = node_spec.get("produces")
    if isinstance(produces, dict):
        # implement your own resolver if needed; simplest is: if key exists in last payload, store it
        last_payload = None
        for p in reversed(node_run.get("tool_payloads", [])):
            if isinstance(p, dict):
                last_payload = p
                break
        if isinstance(last_payload, dict):
            for art_name, result_key in produces.items():
                if isinstance(result_key, str) and result_key in last_payload:
                    state["artifacts"][art_name] = last_payload[result_key]


def make_run_llm_node(
    *,
    client,
    session,
    SYSTEM_PROMPT,
    TextContent,
    CLIENT_SIDE_TOOL_FUNCS,
    NEEDS_GEOM_SINGLE,
    TOOLS_RETURNING_STRUCTURE,
    tools_for_this_call,
    tool_names_for_model,
    summarize_geometries_prompt,
    max_tool_rounds=6,
):
    async def run_llm_node(state, node_spec) -> dict:
        node_run = await run_node_via_existing_executor(
            state=state,
            node_spec=node_spec,
            client=client,
            session=session,
            SYSTEM_PROMPT=SYSTEM_PROMPT,
            TextContent=TextContent,
            CLIENT_SIDE_TOOL_FUNCS=CLIENT_SIDE_TOOL_FUNCS,
            NEEDS_GEOM_SINGLE=NEEDS_GEOM_SINGLE,
            TOOLS_RETURNING_STRUCTURE=TOOLS_RETURNING_STRUCTURE,
            tools_for_this_call=tools_for_this_call,
            tool_names_for_model=tool_names_for_model,
            summarize_geometries_prompt=summarize_geometries_prompt,
            max_tool_rounds=max_tool_rounds,
        )
        if not isinstance(node_run, dict):
            return {"status": "error", "error": "executor returned non-dict"}
        return {"status": "ok", **node_run}
    return run_llm_node

def _latest_node_id_from_run_log(run_log: Any) -> Optional[str]:
    if not isinstance(run_log, list) or not run_log:
        return None
    last = run_log[-1]
    if isinstance(last, dict):
        node = last.get("node")
        if isinstance(node, str) and node:
            return node
    return None


def _extract_geom_snapshot_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort: build a minimal geometry-registry-like dict for formatting."""
    # Case A: result is a full executor state.
    if isinstance(result.get("geometries"), dict):
        return {
            "geometries": result.get("geometries") or {},
            "geom_meta": result.get("geom_meta") or {},
            "name_to_geom": result.get("name_to_geom") or {},
            "current_geom": result.get("current_geom"),
            "default_charge": result.get("default_charge", 0),
            "default_multiplicity": result.get("default_multiplicity", 1),
        }

    # Case B: result is the orchestrator payload with embedded runtime_report.
    rr = result.get("runtime_report")
    if not isinstance(rr, dict):
        return {}

    tail = rr.get("node_results_tail")
    if not isinstance(tail, dict) or not tail:
        return {}

    # Prefer the node that ran last.
    last_node = _latest_node_id_from_run_log(rr.get("run_log"))
    snap = tail.get(last_node) if (last_node and isinstance(tail.get(last_node), dict)) else None
    if snap is None:
        # Fall back to any available snapshot.
        for _k in reversed(list(tail.keys())):
            v = tail.get(_k)
            if isinstance(v, dict):
                snap = v
                break
    if not isinstance(snap, dict):
        return {}

    return {
        "geometries": snap.get("geometries") or {},
        "geom_meta": snap.get("geom_meta") or {},
        "name_to_geom": snap.get("name_to_geom") or {},
        "current_geom": snap.get("current_geom"),
        "default_charge": 0,
        "default_multiplicity": 1,
    }


def _extract_plan_brief_from_result(result: Dict[str, Any], max_nodes: int = 30) -> Dict[str, Any]:
    rr = result.get("runtime_report")
    plan = None
    if isinstance(rr, dict) and isinstance(rr.get("plan"), dict):
        plan = rr.get("plan")
    elif isinstance(result.get("plan"), dict):
        plan = result.get("plan")

    if not isinstance(plan, dict):
        return {}

    nodes = plan.get("nodes")
    nodes_brief = []
    if isinstance(nodes, list):
        for n in nodes[:max_nodes]:
            if not isinstance(n, dict):
                continue
            nodes_brief.append(
                {
                    "id": n.get("id"),
                    "kind": n.get("kind"),
                    "tool": n.get("tool"),
                    "task": n.get("task"),
                    "input_id": n.get("input_id"),
                    "output_id": n.get("output_id"),
                }
            )

    out = {
        "name": plan.get("name") or plan.get("title"),
        "version": plan.get("version"),
        "geom_ids": plan.get("geom_ids"),
        "artifacts_to_save": plan.get("artifacts_to_save"),
        "nodes": nodes_brief,
    }
    if isinstance(nodes, list) and len(nodes) > max_nodes:
        out["nodes_truncated"] = len(nodes) - max_nodes
    return out


def _extract_artifacts_from_result(result: Dict[str, Any], artifact_keys_hint: Optional[List[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # If this is a full executor state, artifacts live under result["artifacts"].
    if isinstance(result.get("artifacts"), dict):
        out.update(result.get("artifacts") or {})

    # Orchestrator payloads often lift key artifacts to the top-level.
    keys: List[str] = []
    if artifact_keys_hint:
        keys.extend([k for k in artifact_keys_hint if isinstance(k, str)])

    rr = result.get("runtime_report")
    if isinstance(rr, dict):
        summ = rr.get("summary")
        if isinstance(summ, dict) and isinstance(summ.get("artifact_keys"), list):
            keys.extend([k for k in summ.get("artifact_keys") if isinstance(k, str)])
        plan = rr.get("plan")
        if isinstance(plan, dict) and isinstance(plan.get("artifacts_to_save"), list):
            keys.extend([k for k in plan.get("artifacts_to_save") if isinstance(k, str)])

    # Deduplicate while preserving order.
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    for k in keys:
        if k in out:
            continue
        if k in result:
            out[k] = result.get(k)

    return out


def result_dict_to_prompt(
    result: Dict[str, Any],
    *,
    user_text: Optional[str] = None,
    max_chars: int = 8000,
    include_run_log_tail: int = 20,
) -> str:
    """Convert a LangGraph/orchestrator result dict to a compact LLM-ready prompt."""
    if not isinstance(result, dict):
        return f"<non-dict result: {type(result).__name__}>"

    rr = result.get("runtime_report") if isinstance(result.get("runtime_report"), dict) else {}
    status = None
    if isinstance(rr, dict):
        status = rr.get("final_status")
        if not status and isinstance(rr.get("final_report"), dict):
            status = rr.get("final_report", {}).get("status")
    if not status:
        status = result.get("last_status") or result.get("status")

    run_id = rr.get("run_id") if isinstance(rr, dict) else result.get("run_id")
    started = rr.get("run_started_utc") if isinstance(rr, dict) else result.get("run_started_utc")
    finished = rr.get("run_finished_utc") if isinstance(rr, dict) else result.get("run_finished_utc")

    plan_brief = _extract_plan_brief_from_result(result)
    artifacts = _extract_artifacts_from_result(result)

    geom_state = _extract_geom_snapshot_from_result(result)
    geom_summary = summarize_geometries_prompt(geom_state) if geom_state else "No geometries found in result."

    # run_log tail
    run_log_tail = []
    if isinstance(rr, dict) and isinstance(rr.get("run_log"), list):
        run_log_tail = rr.get("run_log")[-include_run_log_tail:]
    elif isinstance(result.get("run_log"), list):
        run_log_tail = result.get("run_log")[-include_run_log_tail:]

    # Paths (if present)
    report_json = result.get("runtime_report_json_path")
    report_md = result.get("runtime_report_md_path")

    blocks: List[str] = []
    if user_text:
        blocks.append("User request:\n" + str(user_text).strip())
    blocks.append(f"Workflow status: {status}")
    if run_id:
        blocks.append(f"Run id: {run_id}")
    if started:
        blocks.append(f"Run started (UTC): {started}")
    if finished:
        blocks.append(f"Run finished (UTC): {finished}")
    if report_json or report_md:
        blocks.append(
            "Saved reports:\n"
            + (f"- JSON: {report_json}\n" if report_json else "")
            + (f"- MD: {report_md}" if report_md else "")
        )

    if plan_brief:
        blocks.append("Plan (brief):\n```json\n" + json.dumps(plan_brief, ensure_ascii=False, indent=2) + "\n```")

    if artifacts:
        blocks.append("Artifacts (key outputs):\n```json\n" + json.dumps(artifacts, ensure_ascii=False, indent=2) + "\n```")

    blocks.append("Geometries:\n" + geom_summary)

    if run_log_tail:
        blocks.append("Run log (tail):\n```json\n" + json.dumps(run_log_tail, ensure_ascii=False, indent=2) + "\n```")

    prompt = "\n\n".join(blocks).strip() + "\n"
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n...<truncated>...\n"
    return prompt


def print_result_prompt(
    result: Dict[str, Any],
    *,
    user_text: Optional[str] = None,
    max_chars: int = 8000,
    include_run_log_tail: int = 20,
) -> str:
    """Build prompt with result_dict_to_prompt and print it. Returns the prompt string."""
    prompt = result_dict_to_prompt(
        result,
        user_text=user_text,
        max_chars=max_chars,
        include_run_log_tail=include_run_log_tail,
    )
    print(prompt)
    return prompt


# ─── Spectrum visualization ────────────────────────────────────────────────────

def render_spectrum_image(
    spectrum_data: list,
    spectrum_type: str,
    label: str,
) -> Optional[str]:
    """Render an IR or UV-Vis spectrum PNG and open it in the OS viewer.

    spectrum_type: "ir"    → x=freq_cm1, y=intensity_km_mol, Lorentzian (FWHM 30 cm⁻¹)
    spectrum_type: "uvvis" → x=wavelength_nm, y=oscillator_strength, Gaussian (σ=10 nm)

    Returns the saved PNG path, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[spectrum] matplotlib / numpy not available — skipping visualization")
        return None

    if not spectrum_data:
        return None

    import tempfile
    import platform
    import subprocess as sp

    tmp_dir = Path(tempfile.gettempdir()) / "qcagent_spectra"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", label)[:50]
    png_path = tmp_dir / f"{safe_label}.png"

    fig, ax = plt.subplots(figsize=(10, 4))

    if spectrum_type == "ir":
        freqs = np.array([p["freq_cm1"] for p in spectrum_data])
        ints  = np.array([p["intensity_km_mol"] for p in spectrum_data])

        x_min = max(0.0, float(freqs.min()) - 150.0)
        x_max = float(freqs.max()) + 150.0
        x = np.linspace(x_min, x_max, 5000)

        gamma = 15.0  # half-width at half-maximum (cm⁻¹), FWHM = 30
        y = np.zeros_like(x)
        for f, I in zip(freqs, ints):
            y += I * (gamma ** 2) / ((x - f) ** 2 + gamma ** 2)

        ax.plot(x, y, color="steelblue", linewidth=1.4, label="Lorentzian (FWHM 30 cm⁻¹)")
        markerline, stemlines, baseline = ax.stem(freqs, ints, linefmt="grey",
                                                   markerfmt=" ", basefmt=" ")
        stemlines.set_linewidth(0.8)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Intensity (km mol⁻¹)", fontsize=11)
        ax.set_xlim(x_max, x_min)   # inverted: high freq on left, conventional
        ax.set_title(f"IR Spectrum — {label}", fontsize=12)

    elif spectrum_type == "uvvis":
        wls  = np.array([p["wavelength_nm"] for p in spectrum_data])
        oscs = np.array([p["oscillator_strength"] for p in spectrum_data])

        x_min = max(100.0, float(wls.min()) - 80.0)
        x_max = float(wls.max()) + 80.0
        x = np.linspace(x_min, x_max, 5000)

        sigma = 10.0  # nm
        y = np.zeros_like(x)
        for wl, f in zip(wls, oscs):
            y += f * np.exp(-0.5 * ((x - wl) / sigma) ** 2)

        ax.plot(x, y, color="darkorange", linewidth=1.4, label=f"Gaussian (σ={sigma} nm)")
        markerline, stemlines, baseline = ax.stem(wls, oscs, linefmt="grey",
                                                   markerfmt=" ", basefmt=" ")
        stemlines.set_linewidth(0.8)
        ax.set_xlabel("Wavelength (nm)", fontsize=11)
        ax.set_ylabel("Oscillator strength", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_title(f"UV-Vis Spectrum — {label}", fontsize=12)

    else:
        plt.close(fig)
        return None

    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    system = platform.system()
    if system == "Windows":
        os.startfile(str(png_path))
    elif system == "Darwin":
        sp.Popen(["open", str(png_path)])
    else:
        sp.Popen(["xdg-open", str(png_path)])

    return str(png_path)


def auto_display_spectra(artifacts: dict) -> None:
    """Scan artifacts for IR / UV-Vis / PES data and auto-display each one.

    Recognised key patterns:
      ir_spectrum_<label>    or  ir_spectrum    → IR render
      excited_states_<label> or  excited_states → UV-Vis render
      scan_results_<label>   or  scan_results   → PES plot
    """
    if not isinstance(artifacts, dict):
        return
    for key, value in artifacts.items():
        if not isinstance(value, list) or not value:
            continue
        k = key.lower()
        if k.startswith("ir_spectrum_") or k == "ir_spectrum":
            mol_label = key[len("ir_spectrum_"):] if k.startswith("ir_spectrum_") else "molecule"
            path = render_spectrum_image(value, "ir", f"IR_{mol_label}")
            if path:
                print(f"[spectrum] IR plot → {path}  [opened]")
        elif k.startswith("excited_states_") or k == "excited_states":
            mol_label = key[len("excited_states_"):] if k.startswith("excited_states_") else "molecule"
            path = render_spectrum_image(value, "uvvis", f"UVVis_{mol_label}")
            if path:
                print(f"[spectrum] UV-Vis plot → {path}  [opened]")
        elif k.startswith("scan_results_") or k == "scan_results":
            scan_label = key[len("scan_results_"):] if k.startswith("scan_results_") else "scan"
            path = render_pes_plot(value, f"PES_{scan_label}")
            if path:
                print(f"[spectrum] PES plot → {path}  [opened]")


def render_pes_plot(scan_results: list, label: str) -> Optional[str]:
    """Render a potential energy surface (PES) plot from scan results.

    scan_results: [{"step": int, "value": float, "energy_eh": float}, ...]
    x-axis: coordinate value (Å or degrees)
    y-axis: relative energy in kcal/mol (minimum set to 0)

    Returns the saved PNG path, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[spectrum] matplotlib / numpy not available — skipping PES plot")
        return None

    if not scan_results:
        return None

    import tempfile
    import platform
    import subprocess as sp

    tmp_dir = Path(tempfile.gettempdir()) / "qcagent_spectra"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", label)[:50]
    png_path = tmp_dir / f"{safe_label}.png"

    values    = np.array([p["value"]     for p in scan_results], dtype=float)
    energies  = np.array([p["energy_eh"] for p in scan_results], dtype=float)
    kcal_conv = 627.509474          # Eh → kcal/mol
    rel_kcal  = (energies - energies.min()) * kcal_conv
    min_idx   = int(np.argmin(energies))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(values, rel_kcal, color="steelblue", linewidth=1.6, marker="o",
            markersize=5, markerfacecolor="white", markeredgecolor="steelblue",
            markeredgewidth=1.5)
    ax.axvline(values[min_idx], color="tomato", linewidth=1.0, linestyle="--",
               label=f"min @ {values[min_idx]:.4f}")

    # Axis labels — guess units from value range
    value_range = float(values.max() - values.min())
    x_label = "Coordinate value (Å)" if value_range < 10 else "Coordinate value (degrees)"
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Relative energy (kcal mol⁻¹)", fontsize=11)
    ax.set_title(f"Potential Energy Surface — {label}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    system = platform.system()
    if system == "Windows":
        os.startfile(str(png_path))
    elif system == "Darwin":
        sp.Popen(["open", str(png_path)])
    else:
        sp.Popen(["xdg-open", str(png_path)])

    return str(png_path)
