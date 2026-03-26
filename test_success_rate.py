"""
test_success_rate.py — Success rate tester for the QC planning/execution agent.

Runs the agent N times with a fixed user message and reports what fraction of
runs produce a plan (and optionally an execution result) that passes all
structural/chemical checks.

Modes
-----
  plan  (default, fast)  Call the planner LLM only — no ORCA connection needed.
  full  (slow)           Generate plan via LLM, then execute via LangGraph+ORCA.
                         Requires SSH connection to the VM.

Checkers (auto-detected from message keywords, or set CHECKER below)
---------------------------------------------------------------------
  pka        run_freq_job, G_H_plus_ref_eh in settings, G_*_eh artifacts, LLM node
  sp         run_sp_energy node, energy artifact, no accidental freq
  nbo        run_opt_job → run_nbo_job chain, nbo_section artifact
  solvation  run_solvator_cluster_thermo, nsolv argument present
  ts         run_scan_job → run_ts_opt_job → freq, no LLM node
  struct     minimal structural validity only (valid JSON, nodes, artifacts)

Usage
-----
  # Edit MESSAGE / N_RUNS / MODE / CHECKER at the top, then:
  python test_success_rate.py

  # Or pass CLI flags:
  python test_success_rate.py --message "pKa of HF" --runs 5
  python test_success_rate.py --runs 20 --no-skills
  python test_success_rate.py --mode full --runs 3
  python test_success_rate.py --checker pka --runs 10
  python test_success_rate.py --compound-context "acetic acid: formula=C2H4O2, SMILES=CC(O)=O, charge=0"
  python test_success_rate.py --compound-mode xyz --preload "acetic acid:ha_neutral" --preload "acetate:a_anion"
  python test_success_rate.py --compound-mode xyz --preload "molecule.xyz:mol"
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


def _preload_env_file() -> str:
    """Scan sys.argv for --env-file before argparse runs, so load_dotenv picks
    up the right file before module-level LLM_MODEL / _LLM_BASE_URL are set."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--env-file", "--env") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--env-file=") or arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return os.environ.get("ENV_FILE", ".env")


load_dotenv(_preload_env_file())

from client_helpers import (
    summarize_geometries_prompt,
    summarize_workflow_state,
    parse_json_only,
    name_to_geometry_xyz,
    load_xyz_as_geometry,
    geom_key_from_path,
    pubchem_get_basic_properties,
    structure_add_remove_proton,
    build_dimer_xyz,
    set_geometry_xyz,
    build_approach_scan_geometries,
    run_tool_node,
    state_get_tool_args,
    auto_display_spectra,
    result_dict_to_prompt,
)
from prompts import SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT
from skills import run_planning_skills, SKILL_REGISTRY

LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# ─── TOP-LEVEL CONFIG (edit here or override with CLI flags) ───────────────────
MESSAGE     = ("calculate the alpha-CH pKa of ethyl acetylacetate, barbituric acid, "
               "acetone, and ethyl chloroacetate. Use ethanal (acetaldehyde, experimental "
               "pKa=17.0) as an isodesmic calibration reference to correct the systematic "
               "gas-phase DFT error.")
N_RUNS      = 10
MODE        = "plan"   # "plan" | "full"
WITH_SKILLS = True
CHECKER     = "auto"   # "auto" | "pka" | "sp" | "nbo" | "solvation" | "struct"

# Optional: pre-confirmed compound context injected into every planning call.
# Leave "" to skip.  Used to match normal REPL flow without interactive prompts.
COMPOUND_CONTEXT = ""
# ──────────────────────────────────────────────────────────────────────────────

# Per-checker default messages — used when --checker is explicit but --message is not.
# Falls back to MESSAGE (the global default) for any checker not listed here.
_CHECKER_DEFAULT_MESSAGES: dict = {
    "pka": MESSAGE,   # alias for the global default
    "eas": ("rank EAS reactivity of naphthalene, pyrrole, imidazole, pyrazole, "
            "thiophene, thiazole using Mulliken charges from DFT"),
    "ts":  ("find the transition state for H2O2 cis conformation interconversion "
            "by scanning the H-O-O-H dihedral D=[2,0,1,3] from 0 to 180 degrees "
            "(9 points), then optimizing the TS and computing frequencies"),
}


EMPTY_STATE: dict = {
    "geometries": {}, "geom_meta": {}, "name_to_geom": {},
    "identifiers": {}, "cached_props": {}, "plans": {}, "runs": [],
    "run_log": [], "artifacts": {}, "node_results": {},
    "default_charge": 0, "default_multiplicity": 1,
}

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    "structure_add_remove_proton": structure_add_remove_proton,
    "build_dimer_xyz": build_dimer_xyz,
    "set_geometry_xyz": set_geometry_xyz,
    "build_approach_scan_geometries": build_approach_scan_geometries,
}


# ─── Geometry pre-loading ──────────────────────────────────────────────────────

def _preload_geometries(specs: List[str], state: dict) -> None:
    """Resolve preload specs into state['geometries'] / state['geom_meta'].

    Each spec is either:
      "name:geom_id"       — call name_to_geometry_xyz(name), store as geom_id
      "path/to/file.xyz:geom_id" — load XYZ file, store as geom_id
      "path/to/file.xyz"   — load XYZ file, use filename stem as geom_id

    On error the spec is skipped with a warning; it never raises.
    """
    for spec in specs:
        if ":" in spec:
            src, geom_id = spec.rsplit(":", 1)
            geom_id = geom_id.strip()
            src     = src.strip()
        else:
            src     = spec.strip()
            geom_id = geom_key_from_path(src) if os.path.exists(src) else src

        if os.path.exists(src):
            try:
                xyz = load_xyz_as_geometry(src)
                state.setdefault("geometries", {})[geom_id] = xyz
                state.setdefault("geom_meta", {}).setdefault(geom_id, {})
                print(f"  [preload] {geom_id!r} ← file {src!r}")
            except Exception as exc:
                print(f"  [preload] WARNING: could not load {src!r}: {exc}")
        else:
            result = name_to_geometry_xyz(src)
            if result.get("status") == "ok":
                xyz = result["geometry_xyz"]
                state.setdefault("geometries", {})[geom_id] = xyz
                meta = state.setdefault("geom_meta", {}).setdefault(geom_id, {})
                meta.setdefault("charge", result.get("charge", 0))
                meta.setdefault("multiplicity", result.get("multiplicity", 1))
                print(f"  [preload] {geom_id!r} ← name {src!r} (charge={meta['charge']})")
            else:
                print(f"  [preload] WARNING: name_to_geometry_xyz({src!r}) failed: {result.get('error')}")


# ─── Checker functions ─────────────────────────────────────────────────────────

_VALID_TOOLS = {
    # MCP tools
    "run_sp_energy", "run_opt_job", "run_freq_job", "run_nbo_job",
    "run_solvator_cluster", "run_solvator_cluster_thermo",
    "run_tddft_job", "run_scan_job", "run_ts_opt_job", "run_casscf_job",
    "run_spectrum_job", "structure_add_remove_proton", "inspect_job",
    # Client-side tools
    "name_to_geometry_xyz", "load_xyz_as_geometry",
    "pubchem_get_basic_properties", "state_update", "state_get_tool_args",
    "build_dimer_xyz", "set_geometry_xyz",
    "build_approach_scan_geometries",
    # server-side MCP tools (also valid in plans):
    "build_coordination_complex",
}


def check_struct(plan: dict) -> Dict[str, bool]:
    """Minimal structural validity — every plan must pass these."""
    nodes     = plan.get("nodes") or []
    artifacts = plan.get("artifacts_to_save") or []
    tool_nodes_valid = all(
        n.get("tool") in _VALID_TOOLS
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
    }


def check_pka(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    settings   = plan.get("settings") or {}
    artifacts  = plan.get("artifacts_to_save") or []
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]
    needs_arts = [a for n in llm_nodes for a in (n.get("needs_artifacts") or [])]

    return {
        "uses_freq_not_sp":    "run_freq_job" in tools_used and "run_sp_energy" not in tools_used,
        "has_h_plus_ref":      "G_H_plus_ref_eh" in settings,
        "artifacts_G_eh":      any(a.startswith("G_") and a.endswith("_eh") for a in artifacts),
        "pka_in_artifacts":    any("pka" in a.lower() for a in artifacts),
        "llm_node_exists":     len(llm_nodes) >= 1,
        "freq_arts_in_needs":  any("G_" in a and "_eh" in a for a in needs_arts),
    }


def _all_product_keys(plan: dict) -> set:
    """Collect all artifact keys from every node's product dict."""
    keys = set()
    for n in plan.get("nodes", []):
        keys.update((n.get("product") or {}).keys())
    return keys


def check_sp(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    artifacts  = plan.get("artifacts_to_save") or []
    products   = _all_product_keys(plan)
    all_keys   = set(artifacts) | products

    # Energy artifact: E_*, *_eh (non-Gibbs), or product keys from SP nodes
    def _is_energy(k: str) -> bool:
        k = k.lower()
        return (k.startswith("e_") or k.endswith("_eh") or "energy" in k) \
               and not k.startswith("g_")

    has_energy_art = any(_is_energy(k) for k in all_keys)
    return {
        "has_sp_node":      "run_sp_energy" in tools_used,
        "no_freq_job":      "run_freq_job" not in tools_used,
        "has_energy_art":   has_energy_art,
    }


def check_protonation(plan: dict) -> Dict[str, bool]:
    """Checks for a multi-site protonation comparison plan."""
    nodes        = plan.get("nodes", [])
    tools_used   = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set    = set(tools_used)
    artifacts    = plan.get("artifacts_to_save") or []
    all_keys     = set(artifacts) | _all_product_keys(plan)

    # Plan should add protons and run multiple SP energies
    proton_tools = {"structure_add_remove_proton", "structure_add_proton",
                    "structure_remove_proton"}
    n_sp = sum(1 for t in tools_used if t == "run_sp_energy")

    # Comparison artifact: something that picks a winner or reports delta E
    has_comparison = any(
        any(kw in k.lower() for kw in
            ("best", "lowest", "delta", "comparison", "site", "result", "analysis"))
        for k in artifacts
    )
    def _is_energy(k: str) -> bool:
        k = k.lower()
        return (k.startswith("e_") or k.endswith("_eh") or "energy" in k) \
               and not k.startswith("g_")

    return {
        "has_sp_node":          "run_sp_energy" in tools_set,
        "has_proton_tool":      bool(proton_tools & tools_set),
        "multiple_sp_nodes":    n_sp >= 2,
        "has_energy_artifacts": any(_is_energy(k) for k in all_keys),
        "has_comparison":       has_comparison or bool(
                                    [n for n in nodes if n.get("kind") == "llm"]
                                ),
    }


def check_nbo(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)

    # Check that run_nbo_job appears AFTER run_opt_job in the node list
    opt_idx = next((i for i, t in enumerate(tools_used) if t == "run_opt_job"),  -1)
    nbo_idx = next((i for i, t in enumerate(tools_used) if t == "run_nbo_job"),  -1)

    products_flat = {}
    for n in nodes:
        products_flat.update(n.get("product") or {})

    return {
        "has_opt_node":   "run_opt_job"  in tools_set,
        "has_nbo_node":   "run_nbo_job"  in tools_set,
        "nbo_after_opt":  opt_idx >= 0 and nbo_idx > opt_idx,
        "nbo_artifact":   any("nbo" in k.lower() for k in products_flat),
    }


def check_solvation(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = {n.get("tool") for n in nodes if n.get("kind") == "tool"}

    # Check any node's args contain nsolv
    has_nsolv = any(
        "nsolv" in (n.get("args") or {})
        for n in nodes
    )
    return {
        "has_solvator":    "run_solvator_cluster_thermo" in tools_used
                           or "run_solvator_cluster" in tools_used,
        "nsolv_arg_set":   has_nsolv,
        "uses_freq":       "run_freq_job" in tools_used
                           or "run_solvator_cluster_thermo" in tools_used,
    }


def check_spectrum(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    artifacts  = plan.get("artifacts_to_save") or []
    all_keys   = set(artifacts) | _all_product_keys(plan)

    opt_idx  = next((i for i, t in enumerate(tools_used) if t == "run_opt_job"),    -1)
    spec_idx = next((i for i, t in enumerate(tools_used) if t == "run_spectrum_job"), -1)

    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]
    return {
        "has_spectrum_node":  "run_spectrum_job" in tools_set,
        "no_plain_freq":      "run_freq_job" not in tools_set,
        "has_opt_node":       "run_opt_job" in tools_set,
        "spectrum_after_opt": opt_idx >= 0 and spec_idx > opt_idx,
        "ir_spectrum_art":    any("ir_spectrum" in k.lower() for k in all_keys),
        "no_llm_node":        len(llm_nodes) == 0,
    }


def check_tddft(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    all_keys   = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]
    opt_idx    = next((i for i, t in enumerate(tools_used) if t == "run_opt_job"),    -1)
    tddft_idx  = next((i for i, t in enumerate(tools_used) if t == "run_tddft_job"), -1)
    return {
        "has_tddft_node":     "run_tddft_job" in tools_set,
        "has_opt_node":       "run_opt_job" in tools_set,
        "tddft_after_opt":    opt_idx >= 0 and tddft_idx > opt_idx,
        "excited_states_art": any("excited_states" in k.lower() for k in all_keys),
        "no_llm_node":        len(llm_nodes) == 0,
    }


def check_scan(plan: dict) -> Dict[str, bool]:
    nodes     = plan.get("nodes", [])
    tools_set = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    all_keys  = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]
    return {
        "has_scan_node":     "run_scan_job" in tools_set,
        "scan_results_art":  any("scan_results" in k.lower() for k in all_keys),
        "no_llm_node":       len(llm_nodes) == 0,
    }


def check_ts(plan: dict) -> Dict[str, bool]:
    """TS search plan: scan → ts_opt → freq (via run_spectrum_job or run_freq_job)."""
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]

    scan_idx   = next((i for i, t in enumerate(tools_used) if t == "run_scan_job"),     -1)
    tsopt_idx  = next((i for i, t in enumerate(tools_used) if t == "run_ts_opt_job"),   -1)
    freq_idx   = next((i for i, t in enumerate(tools_used)
                       if t in ("run_freq_job", "run_spectrum_job")), -1)

    return {
        "has_scan_node":    "run_scan_job" in tools_set,
        "has_ts_opt_node":  "run_ts_opt_job" in tools_set,
        "has_freq_node":    "run_freq_job" in tools_set or "run_spectrum_job" in tools_set,
        "ts_opt_after_scan": scan_idx >= 0 and tsopt_idx > scan_idx,
        "freq_after_ts_opt": tsopt_idx >= 0 and freq_idx > tsopt_idx,
        "no_llm_node":      len(llm_nodes) == 0,
    }


def check_casscf(plan: dict) -> Dict[str, bool]:
    """CASSCF plan: run_casscf_job with nel/norb args."""
    nodes     = plan.get("nodes", [])
    tools_set = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    all_keys  = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    casscf_nodes = [n for n in nodes if n.get("tool") == "run_casscf_job"]
    has_active_space = all(
        n.get("args", {}).get("nel") is not None and n.get("args", {}).get("norb") is not None
        for n in casscf_nodes
    ) if casscf_nodes else False
    return {
        "has_casscf_node":   "run_casscf_job" in tools_set,
        "has_active_space":  has_active_space,
        "has_energy_artifact": any("energy" in k.lower() for k in all_keys),
    }


def check_interaction_scan(plan: dict) -> Dict[str, bool]:
    nodes     = plan.get("nodes", [])
    tools_set = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    all_keys  = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]
    return {
        "has_dimer_build":        "build_dimer_xyz" in tools_set,
        "has_sp_monomer":         "run_sp_energy" in tools_set,
        "has_scan_node":          "run_scan_job" in tools_set,
        "has_delta_e_llm":        len(llm_nodes) == 1,  # exactly one LLM node for ΔE_int
        "interaction_curve_art":  any("interaction_curve" in k.lower() or "binding_energy" in k.lower()
                                      for k in all_keys),
        "scan_results_art":       any("scan_results" in k.lower() for k in all_keys),
    }


def check_eas(plan: dict) -> Dict[str, bool]:
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]
    all_keys   = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)

    # SP for charge analysis: Mulliken (no extra properties) OR NBO/NPA
    sp_nodes    = [n for n in nodes if n.get("tool") == "run_sp_energy"]
    has_sp_node = len(sp_nodes) >= 1
    has_nbo_job = "run_nbo_job" in tools_set

    # Charge artifact produced (Mulliken, NPA, NBO, Fukui all accepted)
    has_charge_art = any(
        any(kw in k.lower() for kw in ("npa", "nbo", "charge", "fukui", "mulliken"))
        for k in all_keys
    )
    # Ranking artifact from LLM node
    has_ranking_art = any(
        any(kw in k.lower() for kw in ("ranking", "site", "eas", "fukui"))
        for k in all_keys
    )

    return {
        "has_opt_node":     "run_opt_job" in tools_set,
        "has_charge_calc":  has_sp_node or has_nbo_job,
        "has_ranking_llm":  len(llm_nodes) >= 1,
        "charge_artifact":  has_charge_art,
        "ranking_artifact": has_ranking_art,
    }


def check_coordination_sp(plan: dict) -> Dict[str, bool]:
    """SP on coordination compounds: build_coordination_complex → opt → SP, all three properties.

    When used with the four standard test compounds (Fe(CN)6^3-, Pt(en)2^2+, CuEDTA^2-,
    Ni(dmgH)2), also checks charge and spin/multiplicity in each build node.
    """
    nodes      = plan.get("nodes", [])
    all_keys   = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    all_keys_l = {k.lower() for k in all_keys}

    build_nodes = [n for n in nodes if n.get("tool") == "build_coordination_complex"]
    opt_nodes   = [n for n in nodes if n.get("tool") == "run_opt_job"]
    sp_nodes    = [n for n in nodes if n.get("tool") == "run_sp_energy"]

    # Every build node must be followed by an opt node
    build_output_ids    = {n.get("output_id") for n in build_nodes if n.get("output_id")}
    opt_input_ids       = {n.get("input_id")  for n in opt_nodes  if n.get("input_id")}
    all_built_are_opted = build_output_ids.issubset(opt_input_ids) if build_output_ids else False

    # First-occurrence ordering
    build_idx = next((i for i, n in enumerate(nodes) if n.get("tool") == "build_coordination_complex"), -1)
    opt_idx   = next((i for i, n in enumerate(nodes) if n.get("tool") == "run_opt_job"), -1)
    sp_idx    = next((i for i, n in enumerate(nodes) if n.get("tool") == "run_sp_energy"), -1)

    # Hard-coded charge/spin for known test compounds.
    # Matched by keywords in node id or output_id (case-insensitive).
    # spin here = multiplicity (2S+1) as used in build_coordination_complex args.
    _EXPECTED = [
        # (id_keywords,              charge, mult)
        (("fe", "cn"),               -3,     2),   # Fe(CN)6^3-, Fe3+ d5 low-spin S=1/2
        (("pt", "en"),               +2,     1),   # Pt(en)2^2+, Pt2+ d8 square planar S=0
        (("cu", "edta"),             -2,     2),   # CuEDTA^2-, Cu2+ d9 S=1/2
        (("ni", "dmg"),               0,     1),   # Ni(dmgH)2, Ni2+ d8 square planar S=0
        (("pt", "nh3"),               0,     1),   # cis-Pt(NH3)2Cl2, Pt2+ d8 S=0, neutral
        (("pt", "cl"),                0,     1),   # cis-Pt(NH3)2Cl2 (alt label), same
        (("cisplatin",),              0,     1),   # cisplatin by name
        (("cu", "en"),               +2,     2),   # Cu(en)2^2+, Cu2+ d9 S=1/2
    ]

    def _node_label(n: dict) -> str:
        return ((n.get("id") or "") + " " + (n.get("output_id") or "")).lower()

    charge_ok: Dict[str, bool] = {}
    spin_ok:   Dict[str, bool] = {}
    for keywords, exp_charge, exp_mult in _EXPECTED:
        matched = [n for n in build_nodes
                   if all(kw in _node_label(n) for kw in keywords)]
        if not matched:
            continue  # compound not in plan — skip (generic test still valid)
        n    = matched[0]
        args      = n.get("args") or {}
        key       = "_".join(keywords)
        actual_spin = args.get("spin") if args.get("spin") is not None else args.get("multiplicity")
        # None means the planner relied on the default (multiplicity=1); treat as 1
        if actual_spin is None:
            actual_spin = 1
        charge_ok[f"charge_{key}"] = args.get("charge") == exp_charge
        spin_ok[f"spin_{key}"]     = actual_spin == exp_mult

    result = {
        "has_build_node":      len(build_nodes) >= 1,
        "has_opt_node":        len(opt_nodes) >= 1,
        "has_sp_node":         len(sp_nodes) >= 1,
        "build_before_opt":    build_idx >= 0 and opt_idx > build_idx,
        "opt_before_sp":       opt_idx >= 0 and sp_idx > opt_idx,
        "all_built_are_opted": all_built_are_opted,
        "has_energy_art":      any("energy" in k and "_eh" in k for k in all_keys_l),
        "has_homo_lumo_art":   any("homo" in k or "lumo" in k or "gap" in k for k in all_keys_l),
        "has_dipole_art":      any("dipole" in k for k in all_keys_l),
    }
    result.update(charge_ok)
    result.update(spin_ok)
    return result


_CHECKER_MAP = {
    "pka":              check_pka,
    "sp":               check_sp,
    "nbo":              check_nbo,
    "solvation":        check_solvation,
    "protonation":      check_protonation,
    "spectrum":         check_spectrum,
    "ts":               check_ts,
    "casscf":           check_casscf,
    "tddft":            check_tddft,
    "scan":             check_scan,
    "interaction_scan":  check_interaction_scan,
    "eas":               check_eas,
    "coordination_sp":   check_coordination_sp,
    "struct":            lambda _: {},   # only struct checks apply
}

_AUTO_KEYWORDS = {
    "pka":          ["pka", "acidity", "deprotonation", "acid dissociation", "ka "],
    "protonation":  ["protonation", "protonization", "protonation site", "protonation sites"],
    "sp":           ["single-point", "single point", "sp energy", "run_sp"],
    "nbo":          ["nbo", "natural bond", "wiberg"],
    "solvation":    ["solvat", "aqueous", "microsolvat", "nsolv", "in water"],
    "spectrum":     ["ir spectrum", "infrared", "raman", "vibrational spectrum",
                     "absorption band", "ir band", "spectroscop"],
    "tddft":        ["tddft", "td-dft", "excited state", "excitation energy",
                     "uv-vis", "uv/vis", "absorption spectrum", "electronic transition",
                     "oscillator strength", "vertical excitation"],
    "scan":         ["scan", "pes scan", "potential energy surface", "reaction coordinate",
                     "bond scan", "angle scan", "dihedral scan", "surface scan",
                     "energy profile", "dissociation curve", "torsion scan"],
    "ts":           ["transition state", "ts search", "ts opt", "saddle point",
                     "activation barrier", "activation energy", "optts",
                     "ts structure", "find ts", "locate ts", "reaction barrier"],
    "casscf":           ["casscf", "cas(", "active space", "multireference",
                         "multi-reference", "sa-casscf", "state-averaged",
                         "complete active space", "mcscf"],
    "interaction_scan": ["interaction scan", "interaction energy", "binding energy scan",
                         "approach curve", "dissociation curve", "intermolecular scan",
                         "pi stacking", "π stacking", "cation pi", "cation-pi",
                         "dimer scan", "complex scan"],
    "coordination_sp":  ["coordination compound", "build_coordination_complex",
                         "metal complex", "coordination complex",
                         "hexacyanide", "ethylenediamine", "edta", "dimethylglyoxim"],
    "eas":              ["eas", "electrophilic aromatic", "aromatic substitution",
                         "site selectivity", "site reactivity", "ortho para",
                         "o/p director", "meta director", "fukui f",
                         "npa charge", "natural charge", "aromatic reactivity",
                         "most reactive site", "preferred site"],
}


def detect_checker(message: str) -> str:
    lower = message.lower()
    for name, kws in _AUTO_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return name
    return "struct"


# ─── Execution result checkers (full mode) ────────────────────────────────────

def check_pka_result(artifacts: dict) -> Dict[str, bool]:
    # Multi-compound plans use per-compound keys like pka_acetone, pka_barbituric_acid, etc.
    # Fall back to plain "pka" for single-compound plans.
    pka_vals = {k: v for k, v in artifacts.items() if k == "pka" or k.startswith("pka_")}
    g_vals = {k: v for k, v in artifacts.items()
              if k.startswith("G_") and k.endswith("_eh")}
    # At least one pKa must be in a reasonable calibrated range (-20 to 60)
    reasonable = any(
        isinstance(v, (int, float)) and -20 < v < 60
        for v in pka_vals.values()
    )
    return {
        "pka_present":    len(pka_vals) > 0,
        "pka_reasonable": reasonable,
        "G_eh_present":   len(g_vals) >= 2,
        "G_negative":     all(isinstance(v, (int, float)) and v < 0
                              for v in g_vals.values()),
    }


def check_sp_result(artifacts: dict) -> Dict[str, bool]:
    energy_keys = [k for k in artifacts if k.startswith("E_") or "energy" in k.lower()]
    vals = [artifacts[k] for k in energy_keys]
    return {
        "energy_present":  len(energy_keys) > 0,
        "energy_negative": all(isinstance(v, (int, float)) and v < 0 for v in vals),
    }


def check_spectrum_result(artifacts: dict) -> Dict[str, bool]:
    ir_keys = [k for k in artifacts if "ir_spectrum" in k.lower()]
    ir_list: list = []
    for k in ir_keys:
        v = artifacts.get(k)
        if isinstance(v, list) and v:
            ir_list = v
            break
    positive_freqs = [
        e for e in ir_list
        if isinstance(e, dict) and isinstance(e.get("freq_cm1"), (int, float))
           and e["freq_cm1"] > 0
    ]
    return {
        "ir_spectrum_present":     len(ir_list) > 0,
        "has_positive_freqs":      len(positive_freqs) > 0,
        "has_intensity_field":     all("intensity_km_mol" in e for e in ir_list[:3]),
    }


def check_tddft_result(artifacts: dict) -> Dict[str, bool]:
    es_keys = [k for k in artifacts if "excited_states" in k.lower()]
    es_list: list = []
    for k in es_keys:
        v = artifacts.get(k)
        if isinstance(v, list) and v:
            es_list = v
            break
    sample = es_list[:3]
    nonempty = len(es_list) > 0
    return {
        "excited_states_present":   nonempty,
        "has_energy_ev":            nonempty and all("energy_ev" in e for e in sample),
        "has_wavelength_nm":        nonempty and all("wavelength_nm" in e for e in sample),
        "has_oscillator_strength":  nonempty and all("oscillator_strength" in e for e in sample),
    }


def check_scan_result(artifacts: dict) -> Dict[str, bool]:
    sr_keys = [k for k in artifacts if "scan_results" in k.lower()]
    sr_list: list = []
    for k in sr_keys:
        v = artifacts.get(k)
        if isinstance(v, list) and v:
            sr_list = v
            break
    sample   = sr_list[:3]
    nonempty = len(sr_list) > 0
    return {
        "scan_results_present": nonempty,
        "has_step":             nonempty and all("step"      in p for p in sample),
        "has_value":            nonempty and all("value"     in p for p in sample),
        "has_energy_eh":        nonempty and all("energy_eh" in p for p in sample),
    }


def check_interaction_scan_result(artifacts: dict) -> Dict[str, bool]:
    curve = artifacts.get("interaction_curve") or []
    # also accept any key containing "interaction_curve"
    if not curve:
        for k, v in artifacts.items():
            if "interaction_curve" in k.lower() and isinstance(v, list):
                curve = v
                break
    sample   = curve[:3]
    nonempty = len(curve) > 0
    return {
        "interaction_curve_present":  nonempty,
        "has_distance":               nonempty and all("distance_ang" in p or "value" in p for p in sample),
        "has_delta_e":                nonempty and all("delta_e_int_kcal" in p for p in sample),
        "binding_energy_present":     "binding_energy_kcal" in artifacts or any(
            "binding_energy" in k for k in artifacts),
    }


_RESULT_CHECKER_MAP = {
    "pka":              check_pka_result,
    "sp":               check_sp_result,
    "spectrum":         check_spectrum_result,
    "tddft":            check_tddft_result,
    "scan":             check_scan_result,
    "interaction_scan": check_interaction_scan_result,
}


# ─── Planner helpers ───────────────────────────────────────────────────────────

def _build_messages(message: str, state: dict, skill_contexts: list,
                    compound_context: str = "") -> list:
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
    """Call the planner LLM. Returns (plan, token_usage_dict)."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        tool_choice="none",
        tools=[],
    )
    usage = getattr(resp, "usage", None)
    planner_total = getattr(usage, "total_tokens",      0) or 0
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


# ─── Single run ────────────────────────────────────────────────────────────────

RunResult = Dict[str, Any]   # keys: passed, checks, error, duration_ms, plan


def run_plan_once(
    client: OpenAI,
    message: str,
    checker_name: str,
    with_skills: bool,
    compound_context: str,
    preload_specs: Optional[List[str]] = None,
) -> RunResult:
    """One planning run. Returns pass/fail + per-check breakdown."""
    state = dict(EMPTY_STATE)
    if preload_specs:
        _preload_geometries(preload_specs, state)
    t0    = time.monotonic()
    error = ""
    plan  = {}
    all_checks: Dict[str, bool] = {}

    token_usage: Dict[str, int] = {"planner": 0, "calculator": 0, "total": 0}
    try:
        skill_ctxs = run_planning_skills(message, state) if with_skills else []
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan, token_usage = _call_planner(client, msgs)

        # Structural checks always run
        all_checks.update(check_struct(plan))

        # Task-specific checks
        checker_fn = _CHECKER_MAP.get(checker_name)
        if checker_fn:
            checks = checker_fn(plan)
            # When geometries are preloaded: opt-before-TDDFT not required;
            # LLM comparison node allowed (reporter handles comparison, but planner may add one too)
            if preload_specs and checker_name == "tddft":
                checks.pop("has_opt_node", None)
                checks.pop("tddft_after_opt", None)
                checks.pop("no_llm_node", None)
            all_checks.update(checks)

    except Exception as exc:
        error = str(exc)
        all_checks["exception"] = False

    duration_ms = int((time.monotonic() - t0) * 1000)
    passed      = all(all_checks.values()) and not error

    return {
        "passed":      passed,
        "checks":      all_checks,
        "error":       error,
        "duration_ms": duration_ms,
        "plan":        plan,
        "token_usage": token_usage,
    }


async def run_full_once(
    session,
    client: OpenAI,
    message: str,
    checker_name: str,
    with_skills: bool,
    compound_context: str,
    preload_specs: Optional[List[str]] = None,
) -> RunResult:
    """One full run: plan → execute → check artifacts."""
    from build_graph_from_plan import build_graph_from_plan, build_state

    state = dict(EMPTY_STATE)
    if preload_specs:
        _preload_geometries(preload_specs, state)
    t0    = time.monotonic()
    error = ""
    plan  = {}
    all_checks: Dict[str, bool] = {}

    token_usage: Dict[str, int] = {"planner": 0, "calculator": 0, "reporter": 0, "total": 0}
    agent_report = ""
    try:
        # Step 1: generate plan
        skill_ctxs = run_planning_skills(message, state) if with_skills else []
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan, token_usage = _call_planner(client, msgs)

        all_checks.update(check_struct(plan))
        checker_fn = _CHECKER_MAP.get(checker_name)
        if checker_fn:
            checks = checker_fn(plan)
            # When geometries are preloaded: opt-before-TDDFT not required;
            # LLM comparison node allowed (reporter handles comparison, but planner may add one too)
            if preload_specs and checker_name == "tddft":
                checks.pop("has_opt_node", None)
                checks.pop("tddft_after_opt", None)
                checks.pop("no_llm_node", None)
            all_checks.update(checks)

        if not all(all_checks.values()):
            raise ValueError("Plan failed structural checks; skipping execution")

        # Step 2: execute
        graph       = build_graph_from_plan(
            plan,
            get_tool_args=state_get_tool_args,
            run_tool_node=run_tool_node,
            openai_client=client,
        )
        init_state  = build_state(
            plan, session,
            seed={
                "client_side_tools": CLIENT_SIDE_TOOL_FUNCS,
                "geometries":  dict(state.get("geometries") or {}),
                "geom_meta":   dict(state.get("geom_meta") or {}),
                "name_to_geom": dict(state.get("name_to_geom") or {}),
            },
            client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
        )
        result      = await graph.ainvoke(init_state)

        all_checks["execution_ok"] = result.get("last_status") == "ok"

        # Merge calculator token usage from LangGraph result
        result_tu = result.get("token_usage") or {}
        calc_tokens = result_tu.get("calculator", 0) or 0
        token_usage["calculator"] = calc_tokens
        token_usage["total"]      = token_usage.get("planner", 0) + calc_tokens

        # Step 3: artifact checks + auto-display spectra/PES
        artifacts = result.get("artifacts", {})
        result_checker = _RESULT_CHECKER_MAP.get(checker_name)
        if result_checker:
            all_checks.update(result_checker(artifacts))
        auto_display_spectra(artifacts)

        # Step 4: generate agent report (same as REPL does after execution)
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
        token_usage["total"]    = token_usage.get("total", 0) + reporter_tokens

    except Exception as exc:
        error = str(exc)
        all_checks["exception"] = False

    duration_ms = int((time.monotonic() - t0) * 1000)
    passed      = all(all_checks.values()) and not error

    return {
        "passed":      passed,
        "checks":      all_checks,
        "error":       error,
        "duration_ms": duration_ms,
        "plan":        plan,
        "token_usage": token_usage,
        "agent_report": agent_report,
    }


# ─── Logging ───────────────────────────────────────────────────────────────────

def save_test_log(
    message: str,
    mode: str,
    checker_name: str,
    with_skills: bool,
    results: List[RunResult],
    log_dir: str = "test_logs",
    compound_mode: Optional[str] = None,
) -> str:
    """Save test suite results to test_logs/<timestamp>_<checker>_<mode>.json.

    Returns the path of the saved file.
    """
    os.makedirs(log_dir, exist_ok=True)

    now    = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    fname  = f"test_{now}_{checker_name}_{mode}.json"
    path   = os.path.join(log_dir, fname)

    n      = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    rate   = round(n_pass / n * 100, 1) if n else 0.0

    # Aggregate token usage across all runs
    total_tu: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("token_usage") or {}).items():
            total_tu[k] = total_tu.get(k, 0) + (v or 0)

    log = {
        "type":          "test_run",
        "timestamp_utc": now,
        "message":       message,
        "mode":          mode,
        "checker":       checker_name,
        "with_skills":   with_skills,
        "compound_mode": compound_mode,
        "n_runs":        n,
        "n_pass":        n_pass,
        "rate":          rate,
        "token_usage":         total_tu,
        "sample_agent_report": next(
            (r.get("agent_report", "") for r in results
             if r.get("agent_report") and r.get("passed")), ""
        ),
        "runs": [
            {
                "run":         i + 1,
                "passed":      r["passed"],
                "duration_ms": r["duration_ms"],
                "checks":      r["checks"],
                "error":       r["error"],
                "plan":        r["plan"],
                "token_usage": r.get("token_usage", {}),
            }
            for i, r in enumerate(results)
        ],
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)

    return path


# ─── Reporting ─────────────────────────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(width * n / total) if total else 0
    return "#" * filled + "." * (width - filled)


def print_report(
    message: str,
    mode: str,
    checker_name: str,
    with_skills: bool,
    results: List[RunResult],
) -> None:
    n        = len(results)
    n_pass   = sum(1 for r in results if r["passed"])
    rate     = n_pass / n * 100 if n else 0

    # Collect all check keys across all runs
    all_check_keys: list = []
    seen: set = set()
    for r in results:
        for k in r["checks"]:
            if k not in seen:
                all_check_keys.append(k)
                seen.add(k)

    # Per-check pass counts
    check_pass: Dict[str, int] = {k: 0 for k in all_check_keys}
    for r in results:
        for k in all_check_keys:
            if r["checks"].get(k, False):
                check_pass[k] += 1

    print("\n" + "=" * 70)
    print(f"  Success Rate Test")
    print(f"  Message : {message!r}")
    print(f"  Mode    : {mode}  |  Runs: {n}  |  Skills: {'ON' if with_skills else 'OFF'}")
    print(f"  Checker : {checker_name}")
    print("=" * 70)

    # Per-run status
    for i, r in enumerate(results, 1):
        tag  = "PASS" if r["passed"] else "FAIL"
        ms   = r["duration_ms"]
        fail_keys = [k for k, v in r["checks"].items() if not v]
        detail = ""
        if not r["passed"]:
            if r["error"]:
                detail = f"  error: {r['error'][:80]}"
            else:
                detail = f"  failed: {', '.join(fail_keys)}"
        print(f"  Run {i:>3}/{n}  {tag}  ({ms:>5} ms){detail}")

    # Summary bar
    print()
    print(f"  {_bar(n_pass, n)} {n_pass}/{n} passed  ({rate:.1f}%)")

    # Per-check breakdown
    print()
    print(f"  {'Check':<30}  {'Pass':>4}/{n:<4}  {'Rate':>6}  Bar")
    print(f"  {'-'*30}  {'-'*9}  {'-'*6}  {'-'*20}")
    for k in all_check_keys:
        cp   = check_pass[k]
        pct  = cp / n * 100 if n else 0
        bar  = _bar(cp, n, width=15)
        flag = "  << weakest" if cp == min(check_pass.values()) and cp < n else ""
        print(f"  {k:<30}  {cp:>4}/{n:<4}  {pct:>5.1f}%  {bar}{flag}")

    # Failures detail
    failures = [(i + 1, r) for i, r in enumerate(results) if not r["passed"]]
    if failures:
        print()
        print("  Failed runs:")
        for run_num, r in failures:
            fail_keys = [k for k, v in r["checks"].items() if not v]
            err = f"  [{r['error'][:60]}]" if r["error"] else ""
            print(f"    Run {run_num:>3}: {', '.join(fail_keys)}{err}")

    print()
    if rate == 100.0:
        print("  All runs PASSED.")
    elif rate >= 80.0:
        print(f"  {rate:.1f}% success  (above 80% threshold).")
    else:
        weakest = min(check_pass, key=lambda k: check_pass[k])
        print(f"  {rate:.1f}% success  — weakest check: {weakest!r} ({check_pass[weakest]}/{n})")

    # Token usage summary
    total_tu: Dict[str, int] = {}
    for r in results:
        for k, v in (r.get("token_usage") or {}).items():
            total_tu[k] = total_tu.get(k, 0) + (v or 0)
    if total_tu.get("total", 0) > 0:
        avg_ctx  = total_tu.get("context_tokens", 0) // n if n else 0
        avg_plan = total_tu.get("plan_tokens",    0) // n if n else 0
        avg_tot  = total_tu["total"] // n if n else 0
        calc     = total_tu.get("calculator", 0)
        reporter = total_tu.get("reporter",   0)
        extras   = ""
        if calc:     extras += f" | calculator={calc}"
        if reporter: extras += f" | reporter={reporter}"
        print(f"  Tokens (avg/run): context={avg_ctx}  plan={avg_plan}  planner_total={avg_tot}{extras}")


# ─── Random compound sampler ───────────────────────────────────────────────────

def build_random_sp_message(csv_path: str, n: int, seed: Optional[int] = 0) -> str:
    """Sample N compounds from a CSV and build an SP message for test_success_rate."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        compounds = list(csv.DictReader(fh))
    rng = random.Random(seed)
    sample = rng.sample(compounds, min(n, len(compounds)))
    names = [c["compound_name"] for c in sample]
    compound_list = "\n".join(f"  - {n}" for n in names)
    return (
        f"Calculate single-point energy (B3LYP/def2-SVP) for each of the following "
        f"{len(names)} compounds. Run them in parallel where possible. "
        f"For each compound report: SP energy (energy_eh), HOMO-LUMO gap (homo_lumo_gap_ev), "
        f"and dipole moment (dipole_moment_debye). All three are returned by run_sp_energy "
        f"at no extra cost.\n\nCompounds:\n{compound_list}"
    )


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QC agent success rate tester")
    p.add_argument("--message",          default=None, help="User message to test")
    p.add_argument("--runs",             type=int, default=None, help="Number of runs")
    p.add_argument("--mode",             choices=["plan", "full"], default=None)
    p.add_argument("--checker",          choices=["auto", "pka", "sp", "nbo",
                                                   "solvation", "protonation", "spectrum",
                                                   "tddft", "scan", "ts", "casscf",
                                                   "interaction_scan", "eas",
                                                   "coordination_sp", "struct"], default=None)
    p.add_argument("--no-skills",        action="store_true", help="Disable skill injection")
    p.add_argument("--compound-context", default=None,
                   help="Pre-confirmed compound context string injected into every run")
    p.add_argument("--compound-mode", choices=["full", "name", "smiles", "xyz"],
                   default=None,
                   help=(
                       "Compound information mode for contrast experiments. "
                       "'xyz': force empty compound context (planner sees geometry only). "
                       "'name'/'smiles'/'full': use --compound-context as-is; "
                       "caller is responsible for providing the appropriately filtered context."
                   ))
    p.add_argument("--preload", metavar="SPEC", action="append", default=None,
                   help=(
                       "Pre-load a geometry into state before planning. "
                       "Format: 'name:geom_id' (fetch by name) or "
                       "'path/to/file.xyz:geom_id' (load file). "
                       "The geom_id is the key used in the plan. "
                       "Can be repeated: --preload ethanol:mol1 --preload water:solvent"
                   ))
    p.add_argument("--csv", default=None,
                   help="Compound CSV file (e.g. organic_compounds_100.csv). "
                        "When set with --n-compounds, auto-builds the SP message "
                        "and sets checker=sp.")
    p.add_argument("--n-compounds", type=int, default=None,
                   help="Number of compounds to randomly sample from --csv.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for compound sampling (default: 0).")
    p.add_argument("--env-file", "--env", default=None,
                   help="Dotenv file to load (default: .env). "
                        "Use .env.local for lab server / qwen2.5:32b. "
                        "Equivalent to: set ENV_FILE=.env.local")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full plan JSON after each run. "
                        "Use --verbose --runs 1 to inspect a single plan.")
    return p.parse_args()


async def _run_full_suite(args):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from nbo_agent_planning import _build_mcp_server_params

    server_params = _build_mcp_server_params()

    if getattr(args, "csv", None) and getattr(args, "n_compounds", None):
        message      = build_random_sp_message(args.csv, args.n_compounds,
                                               seed=getattr(args, "seed", 0))
        checker_name = "sp"
    else:
        checker_name = (args.checker or CHECKER)
        message      = args.message or _CHECKER_DEFAULT_MESSAGES.get(checker_name, MESSAGE)
    n_runs           = args.runs    or N_RUNS
    with_skills      = not args.no_skills if args.no_skills else WITH_SKILLS
    compound_context = args.compound_context if args.compound_context is not None else COMPOUND_CONTEXT
    if getattr(args, "compound_mode", None) == "xyz":
        compound_context = ""  # planner reasons from geometry state only
    if checker_name == "auto":
        checker_name = detect_checker(message)
    preload_specs    = getattr(args, "preload", None) or []

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    if preload_specs:
        print(f"  Preload specs: {preload_specs}")

    print(f"\nConnecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP tools: {[t.name for t in tools.tools]}")

            verbose = getattr(args, "verbose", False)
            print(f"\nRunning {n_runs} full runs of: {message!r}")
            results: List[RunResult] = []
            for i in range(n_runs):
                print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
                r = await run_full_once(
                    session, client, message, checker_name, with_skills, compound_context,
                    preload_specs=preload_specs,
                )
                tag = "PASS" if r["passed"] else "FAIL"
                print(f"{tag}  ({r['duration_ms']} ms)")
                if verbose and r["plan"]:
                    print(json.dumps(r["plan"], indent=2, ensure_ascii=False))
                results.append(r)

    print_report(message, "full", checker_name, with_skills, results)
    log_path = save_test_log(message, "full", checker_name, with_skills, results,
                             compound_mode=getattr(args, "compound_mode", None))
    print(f"  Log saved: {log_path}")


def _run_plan_suite(args):
    if getattr(args, "csv", None) and getattr(args, "n_compounds", None):
        message      = build_random_sp_message(args.csv, args.n_compounds,
                                               seed=getattr(args, "seed", 0))
        checker_name = "sp"
    else:
        checker_name = (args.checker or CHECKER)
        message      = args.message or _CHECKER_DEFAULT_MESSAGES.get(checker_name, MESSAGE)
    n_runs           = args.runs    or N_RUNS
    with_skills      = not args.no_skills if args.no_skills else WITH_SKILLS
    compound_context = args.compound_context if args.compound_context is not None else COMPOUND_CONTEXT
    if getattr(args, "compound_mode", None) == "xyz":
        compound_context = ""  # planner reasons from geometry state only
    if checker_name == "auto":
        checker_name = detect_checker(message)
    preload_specs    = getattr(args, "preload", None) or []

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    print(f"\nRunning {n_runs} planning runs of: {message!r}")
    print(f"Checker: {checker_name}  |  Skills: {'ON' if with_skills else 'OFF'}\n")

    # Print which skills will fire
    dummy_state = dict(EMPTY_STATE)
    if with_skills:
        matched = [s.name for s in sorted(SKILL_REGISTRY, key=lambda s: s.priority)
                   if s.matches(message, dummy_state)]
        print(f"Skills matched: {matched}\n")

    if preload_specs:
        print(f"  Preload specs: {preload_specs}")

    verbose = getattr(args, "verbose", False)

    results: List[RunResult] = []
    for i in range(n_runs):
        print(f"  Run {i+1:>3}/{n_runs}...", end=" ", flush=True)
        r = run_plan_once(client, message, checker_name, with_skills, compound_context,
                          preload_specs=preload_specs)
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"{tag}  ({r['duration_ms']:>5} ms)")
        if verbose and r["plan"]:
            print(json.dumps(r["plan"], indent=2, ensure_ascii=False))
        results.append(r)

    print_report(message, "plan", checker_name, with_skills, results)
    log_path = save_test_log(message, "plan", checker_name, with_skills, results,
                             compound_mode=getattr(args, "compound_mode", None))
    print(f"  Log saved: {log_path}")
    return results


def main():
    args = parse_args()
    mode = args.mode or MODE

    if mode == "full":
        asyncio.run(_run_full_suite(args))
    else:
        _run_plan_suite(args)


if __name__ == "__main__":
    main()
