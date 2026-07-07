"""
canonical_ir.py

Shared schema and normalization tables for the MOSAIC-QC dataset.
Both nomad_parse_inp.py (ORCA) and nomad_parse_gjf.py (Gaussian) import
from here to guarantee their output records have identical field sets.

Canonical record schema
-----------------------
Every parser must produce a dict with exactly these keys.
Fields that the source code cannot supply are set to None.

    entry_id, source_code, orca_version (or gaussian_version),
    formula, n_atoms, elements, charge, multiplicity,
    has_tm, is_open_shell, system_type,
    task_type, method_family,
    functional, basis,
    ri_approx,          # always None for Gaussian
    aux_basis,          # always None for Gaussian
    grid_level,         # 1–4 canonical (see below)
    final_grid_level,   # raw FinalGrid integer; None for Gaussian
    dispersion, abc_correction,
    scf_convergence,
    relativistic,
    solvent_model, solvent,
    nroots, casscf_nel, casscf_norb, cbs_scheme,
    specialist_cell,
    parse_warnings, unknown_tokens,
    n_sites_index       # from NOMAD index; cross-check

Grid canonical scale (grid_level 1–4):
    1 = coarse   (ORCA Grid1-3 / DEFGRID1 / Gaussian FineGrid default)
    2 = default  (ORCA Grid4   / DEFGRID2 / Gaussian FineGrid explicit)
    3 = fine     (ORCA Grid5   / DEFGRID3 / Gaussian UltraFine)
    4 = ultrafine(ORCA Grid6-7 / DEFGRID4 / Gaussian SuperFineGrid)
"""

from __future__ import annotations
import math
from collections import Counter

# ---------------------------------------------------------------------------
# Element sets (shared by both parsers)
# ---------------------------------------------------------------------------

ALL_ELEMENTS: frozenset[str] = frozenset({
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
    "Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
})

TM: frozenset[str] = frozenset({
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
})
OPEN_SHELL_TM: frozenset[str] = frozenset({
    "Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
    "Mo","Tc","Ru","Rh",
    "Re","Os","Ir","Pt",
})
HEAVY: frozenset[str] = frozenset({
    "Br","Kr","Rb","Sr","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","Tl","Pb","Bi","Po","At","Rn",
})

# ---------------------------------------------------------------------------
# Functional name normalisation
# Gaussian uses non-standard names for some functionals.
# Map everything to the ORCA canonical uppercase form.
# ---------------------------------------------------------------------------

FUNC_NORMALIZE: dict[str, str] = {
    # Gaussian-specific aliases
    "PBE1PBE":    "PBE0",      # Gaussian's internal name for PBE0
    "PBEPBE":     "PBE",
    "SVWN":       "LDA",
    "SVWN5":      "LDA",
    "SVWN3":      "LDA",
    "WB97XD":     "WB97X-D",
    "WB97X-D":    "WB97X-D",
    "WB97X":      "WB97X",
    "WB97":       "WB97",
    "WB97X-D3":   "WB97X-D3",
    "WB97M-V":    "WB97M-V",
    "M062X":      "M06-2X",
    "M06-2X":     "M06-2X",
    "M06HF":      "M06-HF",
    "M06L":       "M06-L",
    "M11L":       "M11-L",
    "MN15L":      "MN15-L",
    "MN12L":      "MN12-L",
    "CAMB3LYP":   "CAM-B3LYP",
    "CAM-B3LYP":  "CAM-B3LYP",
    "LC-WPBE":    "LC-WPBE",
    "LCWPBE":     "LC-WPBE",
    "B97D":       "B97-D",
    "B97-D":      "B97-D",
    "B97D3":      "B97-D3",
    "B2PLYP":     "B2PLYP",
    "DSDPBEP86":  "DSDPBEP86",
    # GGA functionals missing from original list (common in CCCBDB and older literature)
    "BLYP":       "BLYP",
    "UBLYP":      "BLYP",
    "BP86":       "BP86",
    "UBP86":      "BP86",
    "BPW91":      "BPW91",
    "UBPW91":     "BPW91",
    "PW91":       "PW91",
    "PW91PW91":   "PW91",
    "PBEH":       "PBE0",
    "UPBEPBE":    "PBE",       # unrestricted PBE in Gaussian
    "HCTH":       "HCTH",
    "HCTH93":     "HCTH",
    "HCTH147":    "HCTH",
    "HCTH407":    "HCTH",
    "OLYP":       "OLYP",
    # Hybrid functionals missing
    "B3P86":      "B3P86",
    "UB3P86":     "B3P86",
    "B3PW91":     "B3PW91",
    "UB3LYP":     "B3LYP",
    "UB3PW91":    "B3PW91",
    "O3LYP":      "O3LYP",
    "X3LYP":      "X3LYP",
    "BMK":        "BMK",
    "HSE06":      "HSE06",
    "HSE":        "HSE06",
    "TPSSh":      "TPSSH",
    "B2PLYPD3":   "B2PLYP",   # B2PLYP with D3 dispersion
    "UB97D":      "B97-D",    # unrestricted B97-D
    "UB97D3":     "B97-D3",
    "UWB97XD":    "WB97X-D",
    "UWB97X-D":   "WB97X-D",
    # meta-GGA
    "VSXC":       "VSXC",
    "PKZB":       "PKZB",
    "BB95":       "BB95",
    "BB1K":       "BB1K",
    "TPSSTPSS":   "TPSS",     # Gaussian's form of TPSS (exchange+correlation both TPSS)
    # CCCBDB-common functionals
    "BEPBE":      "BEPBE",    # Becke88 exchange + PBE correlation (GGA)
    "B1B95":      "B1B95",    # Becke 1-parameter + B95 (hybrid meta-GGA)
    "BB95":       "BB95",     # Becke88 + B95 correlation (meta-GGA)
    "BB1K":       "BB1K",
    "MPW1B95":    "MPW1B95",
    "MPW1PW91":   "MPW1PW91",
    "MPW3PBE":    "MPW3PBE",
    "MPWB1K":     "MPWB1K",
    "MPWLYP":     "MPWLYP",
    "MPWPBE":     "MPWPBE",
    "G96LYP":     "G96LYP",
    "G96P86":     "G96P86",
    "G96PW91":    "G96PW91",
    "PBELYP":     "PBELYP",
    "PBEP86":     "PBEP86",
    "HSEH1PBE":   "HSE06",    # Gaussian's internal name for HSE06
    "BE1PBE":     "BE1PBE",  # Becke 1-param + PBE correlation
    "BRX":        "BRX",      # Becke Roussel exchange
    "PKZB":       "PKZB",
    "VSXC":       "VSXC",
    # Keep these (already canonical)
    "B3LYP": "B3LYP", "PBE0": "PBE0", "PBE": "PBE",
    "TPSS": "TPSS", "TPSSH": "TPSSH",
    "M06": "M06", "M062": "M06-2X",
    "HF": "HF", "RHF": "RHF", "UHF": "UHF", "ROHF": "ROHF",
    "MP2": "MP2", "CCSD": "CCSD",
}

# ---------------------------------------------------------------------------
# Basis set normalisation
# Gaussian sometimes omits hyphens; normalise to canonical form.
# ---------------------------------------------------------------------------

def normalize_basis(raw: str) -> str:
    b = raw.strip()
    # def2TZVP → def2-TZVP
    import re
    b = re.sub(r"(?i)def2([A-Z])", r"def2-\1", b)
    # 6-31G* → keep; 6-31Gd → 6-31G(d) is same thing, keep raw
    return b

# ---------------------------------------------------------------------------
# Classification (shared logic)
# ---------------------------------------------------------------------------

def hill_formula(elements: list[str]) -> str:
    cnt = Counter(elements)
    parts: list[str] = []
    for sym in ("C", "H"):
        if sym in cnt:
            parts.append(f"{sym}{cnt[sym]}" if cnt[sym] > 1 else sym)
    for sym in sorted(cnt):
        if sym not in ("C", "H"):
            parts.append(f"{sym}{cnt[sym]}" if cnt[sym] > 1 else sym)
    return "".join(parts)


def classify_system(elements: list[str], multiplicity: int) -> dict:
    eset = set(elements)
    has_heavy   = bool(eset & HEAVY)
    has_tm      = bool(eset & TM)
    has_os_tm   = bool(eset & OPEN_SHELL_TM)
    is_open_shell = multiplicity > 1

    if has_heavy:
        system_type = "heavy_elem"
    elif has_tm and (is_open_shell or has_os_tm):
        system_type = "TM_open"
    elif has_tm:
        system_type = "TM_closed"
    else:
        system_type = "organic"

    return {
        "has_tm":        has_tm,
        "is_open_shell": is_open_shell,
        "system_type":   system_type,
    }


def classify_cell(system_type: str, task_type: str, method_family: str) -> str:
    # RAG-only corpus tags (excluded from LoRA SFT, used for RAG retrieval only)
    if method_family == "CASSCF":
        return "rag_casscf"
    if task_type in ("SCAN", "TS_OPT", "IRC", "NBO", "NMR"):
        return "rag_protocol"

    # 3 LoRA cells — CCSD/MP2 records are merged into organic/metal cells so
    # each specialist also learns high-level method parameter patterns alongside DFT.
    # metal_TDDFT collapsed into metal_general (only 6 records total).
    if system_type in ("TM_open", "TM_closed", "heavy_elem"):
        return "metal_general"
    return "organic_TDDFT" if task_type == "TDDFT" else "organic_general"


# ---------------------------------------------------------------------------
# Canonical record template (documents all required fields)
# ---------------------------------------------------------------------------

CANONICAL_FIELDS: tuple[str, ...] = (
    "entry_id", "source_code", "program_version",
    "formula", "n_atoms", "elements", "charge", "multiplicity",
    "has_tm", "is_open_shell", "system_type",
    "task_type", "method_family",
    "functional", "basis",
    "ri_approx", "aux_basis",
    "grid_level", "final_grid_level",
    "dispersion", "abc_correction",
    "scf_convergence",
    "relativistic",
    "solvent_model", "solvent",
    "nroots", "casscf_nel", "casscf_norb", "cbs_scheme",
    "specialist_cell",
    "parse_warnings", "unknown_tokens",
    "n_sites_index",
)


def empty_record() -> dict:
    """Return a record with all canonical fields set to their zero value."""
    rec = {f: None for f in CANONICAL_FIELDS}
    rec["abc_correction"] = False
    rec["parse_warnings"] = []
    rec["unknown_tokens"] = []
    return rec
