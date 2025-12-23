# client_helpers.py
from __future__ import annotations
import os
import re
from typing import Any, Dict, Optional


NEEDS_GEOM_SINGLE = {'run_solvator_cluster_thermo', 'run_opt_job', 'run_nbo_job', 'run_sp_energy', 'run_solvator_cluster', 'structure_add_remove_proton'}

TOOLS_RETURNING_STRUCTURE = {
    "run_opt_job",
    "structure_add_remove_proton",
    "run_solvator_cluster_thermo",  # if it returns cluster geometry
    "run_solvator_cluster",         # if it returns cluster geometry
}

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

def summarize_geometries_prompt(state) -> str:
    keys = list(state.geometries.keys())
    if not keys:
        return "No geometries are loaded."
    cur = state.current_geom
    lines = [f"Loaded geometries: {', '.join(keys)}."]
    if cur:
        lines.append(f"Current geometry: {cur!r}.")
    lines.append(f"Defaults: charge={state.default_charge}, multiplicity={state.default_multiplicity}.")
    lines.append("When calling tools, prefer passing geometry_name. You may omit geometry_xyz; the host will inject it.")
    
    return "\n".join(lines)

def resolve_geometry_xyz(state, args: dict) -> Optional[str]:
    gname = args.pop("geometry_name", None)
    if isinstance(gname, str) and gname in state.geometries:
        return state.geometries[gname]

    gxyz = args.get("geometry_xyz", "")
    if isinstance(gxyz, str) and gxyz.strip():
        return gxyz

    if state.current_geom and state.current_geom in state.geometries:
        return state.geometries[state.current_geom]

    return None

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


import requests, time

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

    # 1) try PubChem 3D directly by name first
    sdf = _pubchem_name_to_sdf3d(name)
    if sdf:
        try:
            xyz = sdf_to_xyz_no_header(sdf)
            return {"status": "ok", "name": name, "geometry_xyz": xyz, "provenance": {"geometry": "pubchem_name_3d_sdf"}}
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

def state_update(state: "SessionState", set: dict) -> dict:
    """
    Uniformly update session defaults / current geometry.
    Example: state_update(set={"default_charge": 3, "default_multiplicity": 6})
    """
    #unset = unset or [] # we do not need the unset function to prevent the agent from doing something strange

    # apply set
    for k, v in (set or {}).items():
        if k not in ALLOWED_STATE_KEYS:
            return {
                "status": "error",
                "error": f"Unknown state key: {k}",
                "allowed": sorted(ALLOWED_STATE_KEYS),
            }

        if k in {"default_charge", "default_multiplicity"}:
            try:
                v = int(v)
            except Exception:
                return {
                    "status": "error",
                    "error": f"{k} must be an integer",
                    "value": v,
                }

        setattr(state, k, v)

    '''# apply unset
    for k in unset:
        if k not in ALLOWED_STATE_KEYS:
            return {
                "status": "error",
                "error": f"Unknown state key to unset: {k}",
                "allowed": sorted(ALLOWED_STATE_KEYS),
            }
        setattr(state, k, None)
    '''
    return {
        "status": "ok",
        "state": {
            "default_charge": state.default_charge,
            "default_multiplicity": state.default_multiplicity,
            "current_geom": state.current_geom,
        },
    }


def state_get_tool_args(
    state: "SessionState",
    tool_name: str,
    overrides: Optional[dict] = None,
) -> dict:
    """
    Build a dict of arguments for a given MCP tool using current state.
    - injects default charge/multiplicity
    - injects geometry_xyz for single-geometry tools
    - for run_solvator_job, injects solute_xyz/solvent_xyz from names
    """
    overrides = dict(overrides or {})
    args: dict = {}

    # defaults for QC tools
    if state.default_charge is not None:
        args["charge"] = int(state.default_charge)
    if state.default_multiplicity is not None:
        args["multiplicity"] = int(state.default_multiplicity)

    # SINGLE-GEOMETRY TOOLS
    if tool_name in NEEDS_GEOM_SINGLE:
        # caller can explicitly specify a geometry name
        geom_name = overrides.pop("geometry_name", None)
        xyz = None
        if geom_name:
            xyz = get_geometry_xyz_by_name(state, geom_name)
        else:
            # fall back to current_geom
            key = state.current_geom
            if key:
                xyz = state.geometries.get(key)

        if not xyz:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": "No geometry available (geometry_name missing and current_geom is empty or unknown).",
            }

        args["geometry_xyz"] = xyz

    # SOLVATOR / MULTI-GEOMETRY TOOL (example)
    elif tool_name == "run_solvator_job":
        solute_name = overrides.pop("solute_name", None)
        solvent_name = overrides.pop("solvent_name", None)

        solute_xyz = get_geometry_xyz_by_name(state, solute_name) if solute_name else None
        solvent_xyz = get_geometry_xyz_by_name(state, solvent_name) if solvent_name else None

        if not solute_xyz:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"No solute geometry for name {solute_name!r}. Load or resolve it first.",
            }
        if not solvent_xyz:
            return {
                "status": "error",
                "tool_name": tool_name,
                "error": f"No solvent geometry for name {solvent_name!r}. Load or resolve it first.",
            }

        args["solute_xyz"] = solute_xyz
        args["solvent_xyz"] = solvent_xyz

    # any remaining overrides (method, basis, etc.) overwrite defaults
    args.update(overrides)

    return {
        "status": "ok",
        "tool_name": tool_name,
        "args": args,
    }



from typing import Optional

def get_geometry_xyz_by_name(state: "SessionState", name: str) -> Optional[str]:
    """Return XYZ (no header) for a given molecule name, or None if not found."""
    if not name:
        return None
    name = name.strip()

    # 1) direct mapping
    key = state.name_to_geom.get(name)
    if key and key in state.geometries:
        return state.geometries[key]

    # 2) fallback: search geom_meta by recorded name
    for geom_key, meta in state.geom_meta.items():
        if meta.get("name") == name and geom_key in state.geometries:
            # cache for next time
            state.name_to_geom[name] = geom_key
            return state.geometries[geom_key]

    return None

from math import sqrt
from typing import List, Tuple

def parse_xyz(xyz: str) -> Tuple[List[str], List[Tuple[float, float, float]]]:
    atoms = []
    coords = []
    for line in xyz.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(sym)
        coords.append((x, y, z))
    return atoms, coords

def distance(c1, c2):
    return sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def find_heavy_neighbor_for_H(
    atoms: List[str],
    coords: List[Tuple[float, float, float]],
    h_index: int,
    max_distance: float = 1.3,  # loose O–H / N–H range
) -> int | None:
    h_coord = coords[h_index]
    best_j = None
    best_d = 1e9
    for j, (sym_j, coord_j) in enumerate(zip(atoms, coords)):
        if j == h_index or sym_j == "H":
            continue
        d = distance(h_coord, coord_j)
        if d < best_d:
            best_d = d
            best_j = j
    if best_j is not None and best_d <= max_distance:
        return best_j
    return None

def find_acidic_H_by_distance(
    atoms: List[str],
    coords: List[Tuple[float, float, float]],
) -> List[int]:
    """Return indices of H that are nearest to O/N/S and within a cutoff."""
    acidic = []
    for i, sym in enumerate(atoms):
        if sym != "H":
            continue
        j = find_heavy_neighbor_for_H(atoms, coords, i)
        if j is None:
            continue
        heavy = atoms[j]
        if heavy in ("O", "N", "S"):
            acidic.append(i)
    return acidic

def builtin_hydronium_xyz() -> str:
    # e.g. simple C3v-ish hydronium geometry
    return """3
H3O+
O   0.000000   0.000000   0.000000
H   0.000000   0.942000   0.333000
H   0.816000  -0.471000   0.333000
H  -0.816000  -0.471000   0.333000
"""

def builtin_water_xyz() -> str:
    return """3
H2O
O   0.000000   0.000000   0.000000
H   0.758000   0.000000   0.504000
H  -0.758000   0.000000   0.504000
"""


def _maybe_store_geometry_from_payload(state, payload: dict, tool_name: str):
    if not isinstance(payload, dict):
        return
    if payload.get("status") != "ok":
        return

    geom = payload.get("geometry_xyz")
    if not isinstance(geom, str) or not geom.strip():
        return

    key = payload.get("name") or payload.get("label") or tool_name
    key = (key or tool_name).strip()

    base = key
    i = 1
    while key in state.geometries:
        i += 1
        key = f"{base}_{i}"

    state.geometries[key] = geom.strip()
    state.current_geom = key

    # Optionally update defaults if present
    if "new_charge" in payload and payload["new_charge"] is not None:
        state.default_charge = int(payload["new_charge"])
    if "new_multiplicity" in payload and payload["new_multiplicity"] is not None:
        state.default_multiplicity = int(payload["new_multiplicity"])

def xyz_to_no_header(xyz: str) -> str:
    lines = [ln.strip() for ln in (xyz or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    try:
        nat = int(lines[0])
        if len(lines) >= nat + 2:
            return "\n".join(lines[2:2+nat])
    except Exception:
        pass
    return "\n".join(lines)



'''def _store_new_geometry(state, geom_xyz: str, key_hint: str, carry_meta_from: Optional[str] = None):
    geom_xyz = xyz_to_no_header(geom_xyz).strip()
    if not geom_xyz:
        return None

    key = key_hint
    base = key
    i = 1
    while key in state.geometries:
        i += 1
        key = f"{base}_{i}"

    state.geometries[key] = geom_xyz
    state.current_geom = key

    # carry metadata
    if carry_meta_from and carry_meta_from in state.geom_meta:
        state.geom_meta[key] = dict(state.geom_meta[carry_meta_from])
    return key'''
    
def _store_new_geometry(
    state,
    geom_xyz: str,
    key_hint: str,
    carry_meta_from: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
):
    geom_xyz = xyz_to_no_header(geom_xyz).strip()
    if not geom_xyz:
        return None

    key = key_hint
    base = key
    i = 1
    while key in state.geometries:
        i += 1
        key = f"{base}_{i}"

    state.geometries[key] = geom_xyz
    state.current_geom = key

    # Start meta from previous only if we don't have explicit new values
    meta = {}
    if carry_meta_from and carry_meta_from in state.geom_meta:
        meta.update(state.geom_meta[carry_meta_from])

    # Override with explicit electronic state if provided
    if charge is not None:
        meta["charge"] = int(charge)
    if multiplicity is not None:
        meta["multiplicity"] = int(multiplicity)

    if meta:
        state.geom_meta[key] = meta

    return key



def _geom_preview(xyz_no_header: str, max_lines: int = 6) -> str:
    lines = [ln for ln in (xyz_no_header or "").splitlines() if ln.strip()]
    if not lines:
        return "  (empty)"
    head = lines[:max_lines]
    more = "" if len(lines) <= max_lines else f"\n  ... ({len(lines) - max_lines} more lines)"
    return "  " + "\n  ".join(head) + more

def print_session_state(
    state,
    preview_lines: int = 6,
    show_full_current: bool = True,
    max_full_lines: Optional[int] = None,
) -> None:
    """
    Pretty-print SessionState contents: current geom, geometry keys + xyz preview, identifiers, meta.
    Assumes state.geometries stores XYZ atom lines (no header).
    """
    print("current_geom:", state.current_geom)

    geoms = getattr(state, "geometries", {}) or {}
    if not geoms:
        print("geometries: []")
    else:
        print("geometries:")
        for k in geoms.keys():
            geom = geoms.get(k, "") or ""
            n = len([ln for ln in geom.splitlines() if ln.strip()])
            mark = " *" if k == getattr(state, "current_geom", None) else ""
            print(f"- {k}{mark} (natoms={n})")
            print(_geom_preview(geom, max_lines=preview_lines))

    ids = getattr(state, "identifiers", {}) or {}
    print("identifiers:", list(ids.keys()))

    gm = getattr(state, "geom_meta", {}) or {}
    cur = getattr(state, "current_geom", None)
    print("geom_meta:", gm.get(cur, {}) if cur else {})

    if show_full_current and cur and cur in geoms:
        print("\nCURRENT_GEOMETRY_XYZ (no header):")
        full = (geoms[cur] or "").splitlines()
        if max_full_lines is None or len(full) <= max_full_lines:
            print("\n".join(full))
        else:
            print("\n".join(full[:max_full_lines]))
            print(f"... (truncated, {len(full) - max_full_lines} more lines)")

def coerce_geometry_xyz(state, args: dict) -> str:
    gx = (args.get("geometry_xyz") or "").strip()
    if gx:
        if gx in state.geometries:
            return state.geometries[gx]
        # if it already looks like XYZ, keep it; else ignore it
    # fallback: current geometry content
    if state.current_geom and state.current_geom in state.geometries:
        return state.geometries[state.current_geom]
    return ""
