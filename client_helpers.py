# client_helpers.py
from __future__ import annotations
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

NEEDS_GEOM_SINGLE = {'run_solvator_cluster_thermo', 'run_opt_job', 'run_nbo_job', 'run_sp_energy', 'run_solvator_cluster', 'structure_add_remove_proton'}

TOOLS_RETURNING_STRUCTURE = {
    "name_to_geometry_xyz",
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
        parts.append(f"- {gid}: {nm}  q={chg}, mult={mult}{mark}")

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

def pubchem_name_to_cid(name: str, timeout=15):
    # canonical PUG-REST prolog described in cookbook 
    prolog = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    url = f"{prolog}/compound/name/{urllib.parse.quote(name)}/cids/JSON"
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        return None
    data = r.json()
    cids = data.get("IdentifierList", {}).get("CID", [])
    return int(cids[0]) if cids else None

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

def name_to_geometry_xyz(name: str) -> dict:
    name = (name or "").strip()
    if not name:
        return {"status": "error", "error": "empty name"}
    default_charge = 0
    default_multiplicity = 1
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

ALLOWED_STATE_KEYS = {
    "default_charge",
    "default_multiplicity",
    "default_method",
    "default_basis",
    "default_solvent",
    "current_geom",
    "current_name",   # optional: user’s “active molecule name”
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



def state_get_tool_args(
    state: Dict[str, Any],
    tool_name: str,
    overrides: Optional[dict] = None,
) -> dict:
    """Build tool args from dict-state + GeometryRegistry (no dataclass/legacy paths)."""
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
    content = content.strip()
    # If model wraps JSON in ```json fences, strip them
    if content.startswith("```"):
        content = content.strip("`")
        # crude: remove a leading 'json' line if present
        content = content.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    return json.loads(content)



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
        try:
            out = fn(**args)
        except TypeError:
            out = fn(state, **args)

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
'''
async def run_plan_deterministically(
    plan: Dict[str, Any],
    *,
    state: Dict[str, Any],
    run_tool_node,   # async callable: (state, node_spec) -> dict
    run_llm_node=None,  # optional
) -> Dict[str, Any]:
    """
    Runs the plan and returns the final_report dict (also stored in state["result"]).
    """
    graph = build_graph_from_plan(
        plan=plan,
        run_tool_node=run_tool_node,
        run_llm_node=run_llm_node,
        # if your builder still requires call_tool/get_tool_args/call_llm_task, pass dummies
    )
    out_state = await graph.ainvoke(state)
    return out_state.get("result") or out_state.get("final_report") or {}

'''

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
