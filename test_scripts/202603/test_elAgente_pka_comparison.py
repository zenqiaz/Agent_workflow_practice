"""
test_elAgente_pka_comparison.py

Direct comparison with El Agente paper (Section B.6.2, pp. 117–121):
  "El Agente: An Autonomous Agent for Quantum Chemistry"

Task: predict pKa of chlorofluoroacetic acid (CHClFCOOH)

Approach A (default, --approach A):
  B3LYP/6-31G*, CPCM(water), calibrated from 3 reference acids
  Method encoded as method="B3LYP CPCM(Water)" so _build_calc emits:
    ! B3LYP CPCM(Water) 6-31G* OPT  /  ! B3LYP CPCM(Water) 6-31G* FREQ
  -- same as El Agente; direct apples-to-apples comparison

Approach B (--approach B):
  PBE0/def2-TZVP, CPCM(water), calibrated from 3 reference acids
  Method encoded as method="PBE0 CPCM(Water)" so _build_calc emits:
    ! PBE0 CPCM(Water) def2-TZVP OPT  /  ! PBE0 CPCM(Water) def2-TZVP FREQ
  -- our standard higher-level method, same calibration protocol

NOTE on why calibration is required for both approaches:
  The Tissandier G(H+_ref) = -0.01372 Eh is ONLY the thermal correction
  for the proton, not the full aqueous proton free energy (~-265 kcal/mol).
  Without calibration the raw pKa formula gives ~196-235 (completely wrong).
  Calibration from reference acids cancels the ~263 kcal/mol systematic error.

Reference values:
  El Agente predicted pKa:         -2.40  (B3LYP/6-31G*, empirical calibration)
  Experimental pKa (CHClFCOOH):     2.72

Modes:
  plan  (default): generate and check plan only, no ORCA needed
  full:            execute ORCA jobs + post-process pKa

Usage:
  python test_elAgente_pka_comparison.py                       # plan check, Approach A
  python test_elAgente_pka_comparison.py --mode full           # run ORCA, Approach A
  python test_elAgente_pka_comparison.py --approach B          # plan check, Approach B
  python test_elAgente_pka_comparison.py --mode full --approach B
  python test_elAgente_pka_comparison.py --env-file .env.vm    # use fallback VM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


def _preload_env_file() -> str:
    for i, arg in enumerate(sys.argv):
        if arg in ("--env-file", "--env") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--env-file=") or arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return os.environ.get("ENV_FILE", ".env")


load_dotenv(_preload_env_file())

from build_graph_from_plan import expand_template
from client_helpers import (
    summarize_geometries_prompt,
    summarize_workflow_state,
    parse_json_only,
    name_to_geometry_xyz,
    pubchem_get_basic_properties,
    structure_add_remove_proton,
    build_dimer_xyz,
    set_geometry_xyz,
    build_approach_scan_geometries,
    run_tool_node,
    state_get_tool_args,
    result_dict_to_prompt,
)
from prompts import SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT
from skills import run_planning_skills

LLM_MODEL     = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# ─── Chemistry constants ───────────────────────────────────────────────────────

G_H_PLUS_REF_EH = -0.01372      # Tissandier et al. proton free energy (Eh)
EH_TO_KCAL      = 627.509475    # Hartree → kcal/mol
RT_LN10         = 1.3644        # RT x ln10 at 298.15 K (kcal/mol)

# El Agente published result
EL_AGENTE_PKA = -2.40

# Literature experimental pKa for CHClF–COOH
EXP_PKA_CLFA = 2.72

# Reference acids and their experimental pKa values (same set as El Agente B.6.2)
REFERENCE_ACIDS: Dict[str, float] = {
    "acetic":       4.76,
    "fluoroacetic": 2.586,
    "chloroacetic": 2.86,
}

# ─── Agent messages ────────────────────────────────────────────────────────────

# Approach A: replicate El Agente's method (B3LYP/6-31G*, CPCM)
# Pass CPCM(Water) as part of the method string so _build_calc emits
# "! B3LYP CPCM(Water) 6-31G* OPT/FREQ" — valid ORCA 6 syntax.
# IMPORTANT message design: list all 8 molecules explicitly as individual
# parallel node groups so the planner does NOT use a template loop.
# A template loop with a calibration node inside causes duplicate node IDs.
MESSAGE_A = (
    "Calculate the pKa of chlorofluoroacetic acid (CHClFCOOH) using B3LYP/6-31G* "
    "with CPCM(Water) implicit solvation. "
    "To calibrate the systematic error, also compute Gibbs free energies for "
    "three reference carboxylic acids (and their conjugate bases) at the same level.\n\n"
    "Step 1 — load 4 neutral acids by name (charge=0, mult=1):\n"
    "  ha_acetic (acetic acid), ha_fluoroacetic (fluoroacetic acid), "
    "ha_chloroacetic (chloroacetic acid), ha_clfa (chlorofluoroacetic acid).\n\n"
    "Step 2 — derive conjugate base geometries using structure_add_remove_proton "
    "(action='remove', auto=True) on each neutral acid. Use output_id:\n"
    "  a_acetic (from ha_acetic), a_fluoroacetate (from ha_fluoroacetic), "
    "a_chloroacetate (from ha_chloroacetic), a_clfa (from ha_clfa).\n"
    "Each output has charge=-1, mult=1.\n\n"
    "Step 3 — run geometry optimisation + frequency job for all 8 species in parallel. "
    "Use method=\"B3LYP CPCM(Water)\", basis=\"6-31G*\", use_ri=False for ALL jobs. "
    "CRITICAL: do NOT use a 'template' or 'compounds' block — write every node explicitly "
    "as an individual entry in the nodes list. "
    "Use the geometry from the proton-removal step for each anion node.\n\n"
    "Store Gibbs free energies as: "
    "G_ha_acetic_eh, G_a_acetic_eh, G_ha_fluoroacetic_eh, G_a_fluoroacetate_eh, "
    "G_ha_chloroacetic_eh, G_a_chloroacetate_eh, G_ha_clfa_eh, G_a_clfa_eh.\n\n"
    "After all 8 freq jobs complete, add ONE LLM node (id: calibrate_pka) that:\n"
    "  (a) Uses G_H_plus_ref_eh from settings and the 8 G values above.\n"
    "  (b) Computes raw pKa for each reference acid: "
    "pKa = (G_A + G_H_plus_ref - G_HA) * 627.509 / 1.3644.\n"
    "  (c) Derives correction delta_i = pKa_exp_i - pKa_raw_i for each reference acid "
    "(acetic=4.76, fluoroacetic=2.586, chloroacetic=2.86).\n"
    "  (d) Applies delta_i to pKa_raw(CHClFCOOH) and reports their average as "
    "pka_clfa_calibrated.\n"
    "Settings: G_H_plus_ref_eh = -0.01372"
)

# Approach B: our standard method (PBE0/def2-TZVP, CPCM, same calibration protocol)
# Identical structure to MESSAGE_A — 8 species explicitly listed, no template loop,
# one LLM calibration node — but uses PBE0/def2-TZVP instead of B3LYP/6-31G*.
# CPCM must be embedded in the method string because _build_calc constructs
# "! {method} {basis} {task_kw}..." — a separate solvation parameter is ignored.
MESSAGE_B = (
    "Calculate the pKa of chlorofluoroacetic acid (CHClFCOOH) using PBE0/def2-TZVP "
    "with CPCM(Water) implicit solvation. "
    "To calibrate the systematic error, also compute Gibbs free energies for "
    "three reference carboxylic acids (and their conjugate bases) at the same level.\n\n"
    "Step 1 — load 4 neutral acids by name (charge=0, mult=1):\n"
    "  ha_acetic (acetic acid), ha_fluoroacetic (fluoroacetic acid), "
    "ha_chloroacetic (chloroacetic acid), ha_clfa (chlorofluoroacetic acid).\n\n"
    "Step 2 — derive conjugate base geometries using structure_add_remove_proton "
    "(action='remove', auto=True) on each neutral acid. Use output_id:\n"
    "  a_acetic (from ha_acetic), a_fluoroacetate (from ha_fluoroacetic), "
    "a_chloroacetate (from ha_chloroacetic), a_clfa (from ha_clfa).\n"
    "Each output has charge=-1, mult=1.\n\n"
    "Step 3 — run geometry optimisation + frequency job for all 8 species in parallel. "
    "Use method=\"PBE0 CPCM(Water)\", basis=\"def2-TZVP\", use_ri=False for ALL jobs. "
    "Do NOT use a template loop — list each as an individual node pair. "
    "Use the geometry from the proton-removal step for each anion node.\n\n"
    "Store Gibbs free energies as: "
    "G_ha_acetic_eh, G_a_acetic_eh, G_ha_fluoroacetic_eh, G_a_fluoroacetate_eh, "
    "G_ha_chloroacetic_eh, G_a_chloroacetate_eh, G_ha_clfa_eh, G_a_clfa_eh.\n\n"
    "After all 8 freq jobs complete, add ONE LLM node (id: calibrate_pka) that:\n"
    "  (a) Uses G_H_plus_ref_eh from settings and the 8 G values above.\n"
    "  (b) Computes raw pKa for each reference acid: "
    "pKa = (G_A + G_H_plus_ref - G_HA) * 627.509 / 1.3644.\n"
    "  (c) Derives correction delta_i = pKa_exp_i - pKa_raw_i for each reference acid "
    "(acetic=4.76, fluoroacetic=2.586, chloroacetic=2.86).\n"
    "  (d) Applies delta_i to pKa_raw(CHClFCOOH) and reports their average as "
    "pka_clfa_calibrated.\n"
    "Settings: G_H_plus_ref_eh = -0.01372"
)



# ─── Compound context (pre-confirmed, injected as CONFIRMED_COMPOUNDS) ─────────

COMPOUND_CONTEXT_A = """\
acetic acid: formula=C2H4O2, SMILES=CC(=O)O, charge=0, mult=1
acetate: formula=C2H3O2, SMILES=CC(=O)[O-], charge=-1, mult=1
fluoroacetic acid: formula=C2H3FO2, SMILES=OC(=O)CF, charge=0, mult=1
fluoroacetate: formula=C2H2FO2, SMILES=[O-]C(=O)CF, charge=-1, mult=1
chloroacetic acid: formula=C2H3ClO2, SMILES=OC(=O)CCl, charge=0, mult=1
chloroacetate: formula=C2H2ClO2, SMILES=[O-]C(=O)CCl, charge=-1, mult=1
chlorofluoroacetic acid: formula=C2H2ClFO2, SMILES=OC(=O)C(Cl)F, charge=0, mult=1
chlorofluoroacetate: formula=C2HClFO2, SMILES=[O-]C(=O)C(Cl)F, charge=-1, mult=1\
"""

COMPOUND_CONTEXT_B = """\
acetic acid: formula=C2H4O2, SMILES=CC(=O)O, charge=0, mult=1
acetate: formula=C2H3O2, SMILES=CC(=O)[O-], charge=-1, mult=1
fluoroacetic acid: formula=C2H3FO2, SMILES=OC(=O)CF, charge=0, mult=1
fluoroacetate: formula=C2H2FO2, SMILES=[O-]C(=O)CF, charge=-1, mult=1
chloroacetic acid: formula=C2H3ClO2, SMILES=OC(=O)CCl, charge=0, mult=1
chloroacetate: formula=C2H2ClO2, SMILES=[O-]C(=O)CCl, charge=-1, mult=1
chlorofluoroacetic acid: formula=C2H2ClFO2, SMILES=OC(=O)C(Cl)F, charge=0, mult=1
chlorofluoroacetate: formula=C2HClFO2, SMILES=[O-]C(=O)C(Cl)F, charge=-1, mult=1\
"""

EMPTY_STATE: dict = {
    "geometries": {}, "geom_meta": {}, "name_to_geom": {},
    "identifiers": {}, "cached_props": {}, "plans": {}, "runs": [],
    "run_log": [], "artifacts": {}, "node_results": {},
    "default_charge": 0, "default_multiplicity": 1,
}

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz":           name_to_geometry_xyz,
    "pubchem_get_basic_properties":   pubchem_get_basic_properties,
    "structure_add_remove_proton":    structure_add_remove_proton,
    "build_dimer_xyz":                build_dimer_xyz,
    "set_geometry_xyz":               set_geometry_xyz,
    "build_approach_scan_geometries": build_approach_scan_geometries,
}

# ─── Plan checkers ─────────────────────────────────────────────────────────────

def _all_product_keys(plan: dict) -> set:
    keys: set = set()
    for n in plan.get("nodes", []):
        keys.update((n.get("product") or {}).keys())
    fr = plan.get("final_report") or {}
    keys.update(fr.get("fields") or [])
    return keys


def check_clfa_pka_plan_A(plan: dict) -> Dict[str, bool]:
    """Approach A structural checks: 8 freq jobs + calibration LLM node."""
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    settings   = plan.get("settings") or {}
    all_keys   = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]

    n_freq = sum(1 for t in tools_used if t == "run_freq_job")
    n_opt  = sum(1 for t in tools_used if t == "run_opt_job")

    # Artifact naming checks
    has_g_ha_clfa = any("ha" in k.lower() and "clfa" in k.lower() and
                        k.lower().endswith("_eh") for k in all_keys)
    has_g_a_clfa  = any(
        ("a_clfa" in k.lower() or "anion" in k.lower() or "conjugate" in k.lower())
        and k.lower().endswith("_eh") for k in all_keys
    )
    has_g_reference_acids = sum(
        1 for acid in ("acetic", "fluoroacetic", "chloroacetic")
        if any(acid in k.lower() and k.lower().endswith("_eh") for k in all_keys)
    )
    has_pka_out = any("pka" in k.lower() and "clfa" in k.lower() for k in all_keys)

    return {
        "uses_freq":              "run_freq_job" in tools_set,
        "freq_count_ge_8":        n_freq >= 8,
        "opt_count_ge_8":         n_opt >= 8,
        "has_h_plus_ref":         "G_H_plus_ref_eh" in settings,
        "has_g_ha_clfa":          has_g_ha_clfa,
        "has_g_a_clfa":           has_g_a_clfa,
        "ref_acid_artifacts_ge3": has_g_reference_acids >= 3,
        "has_pka_clfa_out":       has_pka_out,
        "has_llm_calibration":    len(llm_nodes) >= 1,
    }


def check_clfa_pka_plan_B(plan: dict) -> Dict[str, bool]:
    """Approach B structural checks: same as A but PBE0/def2-TZVP.
    8 freq jobs (4 pairs) + LLM calibration node from 3 reference acids.
    """
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    settings   = plan.get("settings") or {}
    all_keys   = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]

    n_freq = sum(1 for t in tools_used if t == "run_freq_job")
    n_opt  = sum(1 for t in tools_used if t == "run_opt_job")

    has_g_ha_clfa = any("ha" in k.lower() and "clfa" in k.lower() and
                        k.lower().endswith("_eh") for k in all_keys)
    has_g_a_clfa  = any(
        ("a_clfa" in k.lower() or "anion" in k.lower() or "conjugate" in k.lower())
        and k.lower().endswith("_eh") for k in all_keys
    )
    has_g_reference_acids = sum(
        1 for acid in ("acetic", "fluoroacetic", "chloroacetic")
        if any(acid in k.lower() and k.lower().endswith("_eh") for k in all_keys)
    )
    has_pka_out = any("pka" in k.lower() and "clfa" in k.lower() for k in all_keys)

    return {
        "uses_freq":              "run_freq_job" in tools_set,
        "freq_count_ge_8":        n_freq >= 8,
        "opt_count_ge_8":         n_opt >= 8,
        "has_h_plus_ref":         "G_H_plus_ref_eh" in settings,
        "has_g_ha_clfa":          has_g_ha_clfa,
        "has_g_a_clfa":           has_g_a_clfa,
        "ref_acid_artifacts_ge3": has_g_reference_acids >= 3,
        "has_pka_clfa_out":       has_pka_out,
        "has_llm_calibration":    len(llm_nodes) >= 1,
    }


# ─── pKa post-processing ───────────────────────────────────────────────────────

def _find_g_pair(
    artifacts: dict, acid_keywords: List[str]
) -> Tuple[Optional[float], Optional[float]]:
    """Find (G_HA, G_A) in artifacts for an acid matching all keywords.

    Neutral: key matches G_ha_* (prefix ha_) or contains 'acid'/'neutral'.
    Anion:   key matches G_a_*  (prefix a_, NOT ha_) or contains 'anion'/'minus'.

    Uses regex prefix matching to avoid false positives like 'G_ha_clfa_eh'
    being mistaken for anion because it contains the substring 'a_'.
    Returns (None, None) if not found.
    """
    import re as _re

    g_keys = {k: v for k, v in artifacts.items()
               if isinstance(v, (int, float)) and k.lower().endswith("_eh")}

    def _matches_all(k: str) -> bool:
        kl = k.lower()
        for kw in acid_keywords:
            if kw in kl:
                continue
            # also accept -ate form: acetic->acetate, fluoroacetic->fluoroacetate, etc.
            if kw.endswith("ic") and kw[:-2] + "ate" in kl:
                continue
            return False
        return True

    def _is_neutral(k: str) -> bool:
        kl = k.lower()
        # G_ha_* prefix is the canonical neutral pattern
        if _re.match(r"g_ha_", kl):
            return True
        return any(s in kl for s in ("acid", "neutral", "_ha_", "_ha"))

    def _is_anion(k: str) -> bool:
        kl = k.lower()
        # G_a_* but NOT G_ha_* — must not start with ha after "g_"
        if _re.match(r"g_a_", kl) and not _re.match(r"g_ha_", kl):
            return True
        return any(s in kl for s in ("anion", "minus", "_neg", "conj"))

    candidates = [k for k in g_keys if _matches_all(k)]

    # Prefer explicit prefix matches; fall back to suffix/keyword matches
    neutral_keys = [k for k in candidates if _is_neutral(k) and not _is_anion(k)]
    anion_keys   = [k for k in candidates if _is_anion(k) and not _is_neutral(k)]

    # Second pass: "ate" suffix for named anions (e.g. G_acetate_eh, G_fluoroacetate_eh)
    if not anion_keys:
        anion_keys = [k for k in candidates
                      if any(kw + "ate" in k.lower() or kw + "_anion" in k.lower()
                             for kw in acid_keywords)]

    g_ha = g_keys[neutral_keys[0]] if neutral_keys else None
    g_a  = g_keys[anion_keys[0]]   if anion_keys   else None
    return g_ha, g_a


def compute_pka_raw(g_ha_eh: float, g_a_eh: float) -> float:
    """pKa from DFT Gibbs energies using Tissandier G(H+) reference."""
    delta_g_eh   = g_a_eh + G_H_PLUS_REF_EH - g_ha_eh
    delta_g_kcal = delta_g_eh * EH_TO_KCAL
    return delta_g_kcal / RT_LN10


def analyze_pka_calibration(artifacts: dict, approach: str) -> dict:
    """Independent post-processing: extract G values, compute calibrated pKa.

    Returns a dict with per-acid results and final predictions.
    Works even if the agent's own LLM node made an error.
    """
    result: dict = {
        "approach":              approach,
        "g_h_plus_ref_eh":       G_H_PLUS_REF_EH,
        "reference_acids":       {},
        "g_ha_clfa_eh":          None,
        "g_a_clfa_eh":           None,
        "pka_clfa_raw":          None,
        "corrections":           {},
        "pka_clfa_per_ref":      {},
        "pka_clfa_calibrated":   None,
        "pka_clfa_agent_report": None,
        "artifacts_all":         {k: v for k, v in artifacts.items()
                                  if isinstance(v, (int, float))},
    }

    # Locate target molecule G values
    clfa_kws = ["clfa"]
    g_ha_clfa, g_a_clfa = _find_g_pair(artifacts, clfa_kws)
    # Fallback: try "chlorofluoro"
    if g_ha_clfa is None or g_a_clfa is None:
        g_ha_clfa, g_a_clfa = _find_g_pair(artifacts, ["chlorofluoro"])

    result["g_ha_clfa_eh"] = g_ha_clfa
    result["g_a_clfa_eh"]  = g_a_clfa
    if g_ha_clfa is not None and g_a_clfa is not None:
        result["pka_clfa_raw"] = compute_pka_raw(g_ha_clfa, g_a_clfa)

    # Grab agent-reported pKa (may differ from our independent calculation)
    for k, v in artifacts.items():
        if "pka" in k.lower() and "clfa" in k.lower() and isinstance(v, (int, float)):
            result["pka_clfa_agent_report"] = v
            break
    if result["pka_clfa_agent_report"] is None:
        # single-compound plan may just use "pka"
        if isinstance(artifacts.get("pka"), (int, float)):
            result["pka_clfa_agent_report"] = artifacts["pka"]

    if approach in ("A", "B"):
        # Calibration from reference acids
        corrections: Dict[str, float] = {}
        for acid, pka_exp in REFERENCE_ACIDS.items():
            kws = [acid] if acid != "acetic" else ["acetic"]
            g_ha, g_a = _find_g_pair(artifacts, kws)
            if g_ha is None or g_a is None:
                result["reference_acids"][acid] = {
                    "pka_exp": pka_exp, "g_ha": None, "g_a": None,
                    "pka_calc": None, "correction": None,
                }
                continue
            pka_calc = compute_pka_raw(g_ha, g_a)
            correction = pka_exp - pka_calc
            corrections[acid] = correction
            result["reference_acids"][acid] = {
                "pka_exp": pka_exp,
                "g_ha":    round(g_ha, 8),
                "g_a":     round(g_a,  8),
                "pka_calc": round(pka_calc, 3),
                "correction": round(correction, 3),
            }
        result["corrections"] = {k: round(v, 3) for k, v in corrections.items()}

        # Apply corrections to target
        if result["pka_clfa_raw"] is not None and corrections:
            per_ref = {}
            for acid, corr in corrections.items():
                per_ref[acid] = round(result["pka_clfa_raw"] + corr, 3)
            result["pka_clfa_per_ref"] = per_ref
            result["pka_clfa_calibrated"] = round(
                sum(per_ref.values()) / len(per_ref), 3
            )

    return result


def format_comparison_table(analysis: dict, approach: str) -> str:
    """Return a human-readable comparison summary."""
    lines = [
        "",
        "=" * 65,
        f"  pKa of Chlorofluoroacetic Acid — Approach {approach} Results",
        "=" * 65,
    ]

    method_label = "B3LYP/6-31G*/CPCM" if approach == "A" else "PBE0/def2-TZVP/CPCM"
    lines.append(f"  Reference acid calibration ({method_label}):")
    lines.append(f"  {'Acid':<20}  {'G_HA (Eh)':<14}  {'G_A (Eh)':<14}  "
                 f"{'pKa_raw':>8}  {'pKa_exp':>8}  {'delta':>7}")
    lines.append("  " + "-" * 80)
    for acid, info in analysis.get("reference_acids", {}).items():
        if info["g_ha"] is None:
            lines.append(f"  {acid:<20}  {'N/A':>14}  {'N/A':>14}  "
                          f"{'N/A':>8}  {info['pka_exp']:>8.3f}  {'N/A':>7}")
        else:
            lines.append(f"  {acid:<20}  {info['g_ha']:>14.8f}  {info['g_a']:>14.8f}  "
                          f"{info['pka_calc']:>8.3f}  {info['pka_exp']:>8.3f}  "
                          f"{info['correction']:>+7.3f}")

    lines.append("")
    lines.append("  Per-reference predicted pKa for CHClFCOOH:")
    for acid, pka in analysis.get("pka_clfa_per_ref", {}).items():
        lines.append(f"    via {acid:<20} {pka:>8.3f}")
    cal = analysis.get("pka_clfa_calibrated")
    if cal is not None:
        lines.append(f"    {'Average (calibrated)':<25} {cal:>8.3f}  <- our prediction")

    lines.append("")
    lines.append("  --- Comparison --------------------------------------------------")
    pred = analysis.get("pka_clfa_calibrated")
    agent_rep = analysis.get("pka_clfa_agent_report")
    lines.append(f"  El Agente result (B3LYP/6-31G*, cal.):  {EL_AGENTE_PKA:>8.3f}")
    if agent_rep is not None:
        lines.append(f"  Our agent reported pKa:                {agent_rep:>8.3f}")
    if pred is not None:
        lines.append(f"  Our independent calculation (Appr. {approach}): {pred:>8.3f}")
    lines.append(f"  Experimental pKa:                        {EXP_PKA_CLFA:>8.3f}")
    if pred is not None:
        our_err    = abs(pred - EXP_PKA_CLFA)
        agent_err  = abs(EL_AGENTE_PKA - EXP_PKA_CLFA)
        better = "YES" if our_err < agent_err else "NO"
        lines.append(f"  |error| ours:       {our_err:.3f} pKa units")
        lines.append(f"  |error| El Agente:  {agent_err:.3f} pKa units")
        lines.append(f"  Our prediction closer to experiment: {better}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ─── Planner helpers (same pattern as test_success_rate.py) ────────────────────

def _build_messages(message: str, state: dict, skill_contexts: list,
                    compound_context: str) -> list:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ctx in skill_contexts:
        msgs.append({"role": "system", "content": "SKILL:\n" + ctx})
    msgs.append({"role": "system", "content": "STATE:\n" + summarize_geometries_prompt(state)})
    msgs.append({"role": "system", "content": "WORKFLOW_STATE:\n" + summarize_workflow_state(state)})
    if compound_context:
        msgs.append({"role": "system", "content": "CONFIRMED_COMPOUNDS:\n" + compound_context})
    msgs.append({"role": "user", "content": message})
    return msgs


def _call_planner(client: OpenAI, msgs: list) -> Tuple[dict, dict]:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        tool_choice="none",
        tools=[],
    )
    usage = getattr(resp, "usage", None)
    planner_total = getattr(usage, "total_tokens", 0) or 0
    token_usage = {
        "context_tokens": getattr(usage, "prompt_tokens",     0) or 0,
        "plan_tokens":    getattr(usage, "completion_tokens", 0) or 0,
        "planner":        planner_total,
        "calculator":     0,
        "reporter":       0,
        "total":          planner_total,
    }
    raw  = (resp.choices[0].message.content or "").strip()
    plan = parse_json_only(raw)
    if not isinstance(plan, dict):
        raise ValueError(f"Planner returned non-dict JSON:\n{raw[:300]}")
    return plan, token_usage


# ─── Structural plan check ─────────────────────────────────────────────────────

VALID_TOOLS = {
    "run_sp_energy", "run_opt_job", "run_freq_job", "run_nbo_job",
    "run_solvator_cluster", "run_solvator_cluster_thermo",
    "run_tddft_job", "run_scan_job", "run_ts_opt_job", "run_casscf_job",
    "run_spectrum_job", "structure_add_remove_proton",
    "name_to_geometry_xyz", "pubchem_get_basic_properties",
    "build_dimer_xyz", "set_geometry_xyz", "build_approach_scan_geometries",
    "build_coordination_complex",
}


def check_struct(plan: dict) -> Dict[str, bool]:
    nodes     = plan.get("nodes") or []
    artifacts = plan.get("artifacts_to_save") or []
    if not artifacts:
        fr = plan.get("final_report") or {}
        artifacts = fr.get("fields") or []
    tool_nodes_valid = all(
        n.get("tool") in VALID_TOOLS
        for n in nodes if n.get("kind") == "tool"
    )
    return {
        "is_dict":         isinstance(plan, dict),
        "has_name":        bool(plan.get("name")),
        "has_nodes":       len(nodes) > 0,
        "has_artifacts":   len(artifacts) > 0,
        "nodes_have_id":   all(n.get("id") for n in nodes),
        "nodes_have_kind": all(n.get("kind") for n in nodes),
        "tools_valid":     tool_nodes_valid,
    }


# ─── Single run (plan only) ────────────────────────────────────────────────────

def run_plan_check(
    client: OpenAI,
    message: str,
    approach: str,
    compound_context: str,
) -> dict:
    state = dict(EMPTY_STATE)
    t0 = time.monotonic()
    error = ""
    plan: dict = {}
    all_checks: Dict[str, bool] = {}
    token_usage: Dict[str, int] = {"planner": 0, "total": 0}

    try:
        skill_ctxs = run_planning_skills(message, state)
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan, token_usage = _call_planner(client, msgs)
        plan = expand_template(plan)

        all_checks.update(check_struct(plan))
        checker = check_clfa_pka_plan_A if approach == "A" else check_clfa_pka_plan_B
        all_checks.update(checker(plan))
    except Exception as exc:
        error = str(exc)
        all_checks["exception"] = False

    return {
        "passed":      all(all_checks.values()) and not error,
        "checks":      all_checks,
        "error":       error,
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "plan":        plan,
        "token_usage": token_usage,
    }


# ─── Single run (full execution) ──────────────────────────────────────────────

async def run_full(
    session,
    client: OpenAI,
    message: str,
    approach: str,
    compound_context: str,
) -> dict:
    from build_graph_from_plan import build_graph_from_plan, build_state

    state = dict(EMPTY_STATE)
    t0 = time.monotonic()
    error = ""
    plan: dict = {}
    all_checks: Dict[str, bool] = {}
    token_usage: Dict[str, int] = {"planner": 0, "calculator": 0, "reporter": 0, "total": 0}
    agent_report = ""
    analysis: dict = {}

    try:
        # Step 1: plan
        skill_ctxs = run_planning_skills(message, state)
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan, token_usage = _call_planner(client, msgs)
        plan = expand_template(plan)

        all_checks.update(check_struct(plan))
        checker = check_clfa_pka_plan_A if approach == "A" else check_clfa_pka_plan_B
        all_checks.update(checker(plan))

        if not all(all_checks.values()):
            raise ValueError("Plan failed structural checks; skipping execution")

        # Step 2: execute
        graph      = build_graph_from_plan(
            plan,
            get_tool_args=state_get_tool_args,
            run_tool_node=run_tool_node,
            openai_client=client,
        )
        init_state = build_state(
            plan, session,
            seed={"client_side_tools": CLIENT_SIDE_TOOL_FUNCS},
            client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
        )
        t1_plan = time.monotonic()
        result = await graph.ainvoke(init_state)
        t2_orca = time.monotonic()

        all_checks["execution_ok"] = result.get("last_status") == "ok"

        # Merge calculator tokens
        result_tu = result.get("token_usage") or {}
        calc_tokens = result_tu.get("calculator", 0) or 0
        token_usage["calculator"] = calc_tokens
        token_usage["total"]      = token_usage.get("planner", 0) + calc_tokens

        # Per-node timing from LangGraph run_log
        run_log = result.get("run_log") or []
        node_timings = [
            {
                "node":           e.get("node"),
                "kind":           e.get("kind"),
                "tool":           e.get("tool"),
                "status":         e.get("status"),
                "duration_ms":    e.get("duration_ms"),
                "started_utc":    e.get("started_utc"),
                "ended_utc":      e.get("ended_utc"),
                "retry_attempts": e.get("retry_attempts"),  # None if no retry
            }
            for e in run_log
        ]

        # Step 3: independent post-processing
        artifacts = result.get("artifacts", {})
        analysis  = analyze_pka_calibration(artifacts, approach)

        pred = analysis.get("pka_clfa_calibrated")  # calibrated for both A and B
        if pred is not None:
            all_checks["pka_present"]    = True
            all_checks["pka_reasonable"] = -10 < pred < 20
            all_checks["closer_than_elAgente"] = abs(pred - EXP_PKA_CLFA) < abs(EL_AGENTE_PKA - EXP_PKA_CLFA)

        # Step 4: reporter
        user_payload = (
            f"Original user request:\n{message}\n\n"
            f"Run result:\n{result_dict_to_prompt(result=result, user_text=message)}"
        )
        report_resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_payload},
            ],
        )
        agent_report = report_resp.choices[0].message.content or ""
        reporter_tokens = (report_resp.usage.total_tokens or 0) if report_resp.usage else 0
        token_usage["reporter"] = reporter_tokens
        token_usage["total"]   += reporter_tokens

    except Exception as exc:
        error = str(exc)
        all_checks["exception"] = False

    t_end = time.monotonic()
    duration_total_ms = int((t_end - t0) * 1000)
    # planner wall time = from t0 to graph start (t1_plan set only on success path)
    planner_ms = int((locals().get("t1_plan", t0) - t0) * 1000)
    orca_ms    = int((locals().get("t2_orca", locals().get("t1_plan", t0))
                      - locals().get("t1_plan", t0)) * 1000)
    reporter_ms = duration_total_ms - planner_ms - orca_ms

    return {
        "passed":       all(all_checks.values()) and not error,
        "checks":       all_checks,
        "error":        error,
        "duration_ms":  duration_total_ms,
        "timing": {
            "planner_ms":  planner_ms,
            "orca_ms":     orca_ms,
            "reporter_ms": reporter_ms,
            "total_ms":    duration_total_ms,
        },
        "node_timings": locals().get("node_timings", []),
        "plan":         plan,
        "token_usage":  token_usage,
        "agent_report": agent_report,
        "analysis":     analysis,
    }


# ─── Logging ───────────────────────────────────────────────────────────────────

def save_log(run: dict, approach: str, mode: str, log_dir: str = "test_logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    now   = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fname = f"elAgente_pka_approach{approach}_{mode}_{now}.json"
    path  = os.path.join(log_dir, fname)

    log = {
        "type":                "elAgente_pka_comparison",
        "timestamp_utc":       now,
        "approach":            approach,
        "mode":                mode,
        "el_agente_pka":       EL_AGENTE_PKA,
        "exp_pka_clfa":        EXP_PKA_CLFA,
        "passed":              run["passed"],
        "checks":              run["checks"],
        "error":               run["error"],
        "duration_ms":         run["duration_ms"],
        "timing":              run.get("timing", {}),
        "node_timings":        run.get("node_timings", []),
        "token_usage":         run.get("token_usage", {}),
        "plan":                run.get("plan", {}),
        "analysis":            run.get("analysis", {}),
        "agent_report":        run.get("agent_report", ""),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=True)
    return path


# ─── Report printer ────────────────────────────────────────────────────────────

def print_plan_report(run: dict, approach: str) -> None:
    print(f"\n{'='*65}")
    print(f"  El Agente Comparison — Plan Check (Approach {approach})")
    print(f"{'='*65}")
    tag = "PASS" if run["passed"] else "FAIL"
    print(f"  Result: {tag}  ({run['duration_ms']} ms)")
    if run["error"]:
        print(f"  Error:  {run['error']}")
    print(f"\n  {'Check':<35}  {'Result':>6}")
    print(f"  {'-'*35}  {'-'*6}")
    for k, v in run["checks"].items():
        flag = "PASS" if v else "FAIL"
        mark = "" if v else "  << FAILED"
        print(f"  {k:<35}  {flag}{mark}")
    if run.get("plan"):
        print(f"\n  Plan name: {run['plan'].get('name')}")
        nodes = run["plan"].get("nodes", [])
        tool_nodes = [n for n in nodes if n.get("kind") == "tool"]
        llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]
        calc_nodes = [n for n in nodes if n.get("kind") == "calc_expr"]
        print(f"  Nodes: {len(nodes)} total  "
              f"({len(tool_nodes)} tool, {len(llm_nodes)} llm, {len(calc_nodes)} calc_expr)")


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="El Agente pKa comparison experiment")
    p.add_argument("--approach", choices=["A", "B"], default="A",
                   help="A: B3LYP/6-31G*/CPCM, calibrated (same as El Agente)  "
                        "B: PBE0/def2-TZVP/CPCM, calibrated (higher level)")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of plan-mode runs (for success-rate measurement, default 1)")
    p.add_argument("--mode", choices=["plan", "full"], default="plan",
                   help="plan: check plan structure only  "
                        "full: execute ORCA jobs and compute pKa")
    p.add_argument("--env-file", "--env", default=None,
                   help="Dotenv file (default: .env)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full plan JSON")
    return p.parse_args()


async def _run_full_async(args: argparse.Namespace) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from agent import _build_mcp_server_params

    approach         = args.approach
    message          = MESSAGE_A if approach == "A" else MESSAGE_B
    compound_context = COMPOUND_CONTEXT_A if approach == "A" else COMPOUND_CONTEXT_B

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    print(f"\nConnecting to MCP server...")
    server_params = _build_mcp_server_params()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP tools available: {len(tools.tools)}")

            print(f"\nRunning Approach {approach} (full mode)...")
            print(f"Message:\n  {message[:120]}...\n")

            run = await run_full(session, client, message, approach, compound_context)

    # Save log first — before any print that could crash on Windows GBK stdout
    log_path = save_log(run, approach, "full")

    if args.verbose and run.get("plan"):
        print("\nPlan JSON:")
        print(json.dumps(run["plan"], indent=2, ensure_ascii=True))

    # Print plan check results
    print_plan_report(run, approach)

    # Print pKa comparison table
    if run.get("analysis"):
        print(format_comparison_table(run["analysis"], approach))

    if run.get("agent_report"):
        print("\n  Agent report:")
        safe_report = run["agent_report"].encode("ascii", errors="replace").decode("ascii")
        for line in safe_report.splitlines():
            print(f"  {line}")

    print(f"\n  Log saved: {log_path}")


def _run_plan_sync(args: argparse.Namespace) -> None:
    approach         = args.approach
    n_runs           = args.runs
    message          = MESSAGE_A if approach == "A" else MESSAGE_B
    compound_context = COMPOUND_CONTEXT_A if approach == "A" else COMPOUND_CONTEXT_B

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    print(f"\nRunning Approach {approach} (plan-only check, {n_runs} run(s))...")
    print(f"Model: {LLM_MODEL}")
    print(f"Message:\n  {message[:150]}...\n")

    results: List[dict] = []
    for i in range(n_runs):
        if n_runs > 1:
            print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
        run = run_plan_check(client, message, approach, compound_context)
        if n_runs > 1:
            print("PASS" if run["passed"] else "FAIL", f"({run['duration_ms']} ms)")
        results.append(run)

    # Print detail for last (or only) run
    if args.verbose and results[-1].get("plan"):
        print("Plan JSON:")
        print(json.dumps(results[-1]["plan"], indent=2, ensure_ascii=True))

    print_plan_report(results[-1], approach)

    # Aggregate success rate when multiple runs
    if n_runs > 1:
        n_pass = sum(1 for r in results if r["passed"])
        print(f"\n  Overall: {n_pass}/{n_runs} PASSED  ({n_pass/n_runs*100:.1f}%)")

        # Per-check breakdown
        all_keys: List[str] = []
        seen: set = set()
        for r in results:
            for k in r["checks"]:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        print(f"\n  {'Check':<35}  {'Pass':>4}/{n_runs:<4}")
        for k in all_keys:
            cp = sum(1 for r in results if r["checks"].get(k, False))
            print(f"  {k:<35}  {cp:>4}/{n_runs:<4}")

    # Save log for last run (or all if multiple)
    log_path = save_log(results[-1], approach, "plan")
    print(f"\n  Log saved: {log_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "full":
        asyncio.run(_run_full_async(args))
    else:
        _run_plan_sync(args)


if __name__ == "__main__":
    main()
