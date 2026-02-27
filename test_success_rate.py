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
    pubchem_get_basic_properties,
    structure_add_remove_proton,
    run_tool_node,
    state_get_tool_args,
    auto_display_spectra,
)
from prompts import SYSTEM_PROMPT
from skills import run_planning_skills, SKILL_REGISTRY

LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# ─── TOP-LEVEL CONFIG (edit here or override with CLI flags) ───────────────────
MESSAGE     = "calculate the pKa of acetic acid"
N_RUNS      = 10
MODE        = "plan"   # "plan" | "full"
WITH_SKILLS = True
CHECKER     = "auto"   # "auto" | "pka" | "sp" | "nbo" | "solvation" | "struct"

# Optional: pre-confirmed compound context injected into every planning call.
# Leave "" to skip.  Used to match normal REPL flow without interactive prompts.
COMPOUND_CONTEXT = ""
# ──────────────────────────────────────────────────────────────────────────────


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
}


# ─── Checker functions ─────────────────────────────────────────────────────────

def check_struct(plan: dict) -> Dict[str, bool]:
    """Minimal structural validity — every plan must pass these."""
    nodes     = plan.get("nodes") or []
    artifacts = plan.get("artifacts_to_save") or []
    return {
        "is_dict":             isinstance(plan, dict),
        "has_name":            bool(plan.get("name")),
        "has_nodes":           len(nodes) > 0,
        "has_artifacts":       len(artifacts) > 0,
        "nodes_have_id":       all(n.get("id") for n in nodes),
        "nodes_have_kind":     all(n.get("kind") for n in nodes),
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
        "pka_in_artifacts":    "pka" in artifacts,
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


_CHECKER_MAP = {
    "pka":          check_pka,
    "sp":           check_sp,
    "nbo":          check_nbo,
    "solvation":    check_solvation,
    "protonation":  check_protonation,
    "spectrum":     check_spectrum,
    "ts":           check_ts,
    "tddft":        check_tddft,
    "scan":         check_scan,
    "struct":       lambda _: {},   # only struct checks apply
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
}


def detect_checker(message: str) -> str:
    lower = message.lower()
    for name, kws in _AUTO_KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return name
    return "struct"


# ─── Execution result checkers (full mode) ────────────────────────────────────

def check_pka_result(artifacts: dict) -> Dict[str, bool]:
    pka = artifacts.get("pka")
    g_vals = {k: v for k, v in artifacts.items()
              if k.startswith("G_") and k.endswith("_eh")}
    return {
        "pka_present":    pka is not None,
        "pka_reasonable": isinstance(pka, (int, float)) and -5 < pka < 60,
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


_RESULT_CHECKER_MAP = {
    "pka":      check_pka_result,
    "sp":       check_sp_result,
    "spectrum": check_spectrum_result,
    "tddft":    check_tddft_result,
    "scan":     check_scan_result,
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


def _call_planner(client: OpenAI, msgs: list) -> dict:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        tool_choice="none",
        tools=[],
    )
    raw  = (resp.choices[0].message.content or "").strip()
    plan = parse_json_only(raw)
    if not isinstance(plan, dict):
        raise ValueError(f"Planner returned non-dict JSON:\n{raw[:300]}")
    return plan


# ─── Single run ────────────────────────────────────────────────────────────────

RunResult = Dict[str, Any]   # keys: passed, checks, error, duration_ms, plan


def run_plan_once(
    client: OpenAI,
    message: str,
    checker_name: str,
    with_skills: bool,
    compound_context: str,
) -> RunResult:
    """One planning run. Returns pass/fail + per-check breakdown."""
    state = dict(EMPTY_STATE)
    t0    = time.monotonic()
    error = ""
    plan  = {}
    all_checks: Dict[str, bool] = {}

    try:
        skill_ctxs = run_planning_skills(message, state) if with_skills else []
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan       = _call_planner(client, msgs)

        # Structural checks always run
        all_checks.update(check_struct(plan))

        # Task-specific checks
        checker_fn = _CHECKER_MAP.get(checker_name)
        if checker_fn:
            all_checks.update(checker_fn(plan))

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
    }


async def run_full_once(
    session,
    client: OpenAI,
    message: str,
    checker_name: str,
    with_skills: bool,
    compound_context: str,
) -> RunResult:
    """One full run: plan → execute → check artifacts."""
    from build_graph_from_plan import build_graph_from_plan, build_state

    state = dict(EMPTY_STATE)
    t0    = time.monotonic()
    error = ""
    plan  = {}
    all_checks: Dict[str, bool] = {}

    try:
        # Step 1: generate plan
        skill_ctxs = run_planning_skills(message, state) if with_skills else []
        msgs       = _build_messages(message, state, skill_ctxs, compound_context)
        plan       = _call_planner(client, msgs)

        all_checks.update(check_struct(plan))
        checker_fn = _CHECKER_MAP.get(checker_name)
        if checker_fn:
            all_checks.update(checker_fn(plan))

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
            seed={"client_side_tools": CLIENT_SIDE_TOOL_FUNCS},
            client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
        )
        result      = await graph.ainvoke(init_state)

        all_checks["execution_ok"] = result.get("last_status") == "ok"

        # Step 3: artifact checks + auto-display spectra/PES
        artifacts = result.get("artifacts", {})
        result_checker = _RESULT_CHECKER_MAP.get(checker_name)
        if result_checker:
            all_checks.update(result_checker(artifacts))
        auto_display_spectra(artifacts)

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
    }


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


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QC agent success rate tester")
    p.add_argument("--message",          default=None, help="User message to test")
    p.add_argument("--runs",             type=int, default=None, help="Number of runs")
    p.add_argument("--mode",             choices=["plan", "full"], default=None)
    p.add_argument("--checker",          choices=["auto", "pka", "sp", "nbo",
                                                   "solvation", "protonation", "spectrum",
                                                   "tddft", "scan", "ts", "struct"], default=None)
    p.add_argument("--no-skills",        action="store_true", help="Disable skill injection")
    p.add_argument("--compound-context", default=None,
                   help="Pre-confirmed compound context string injected into every run")
    p.add_argument("--env-file", "--env", default=None,
                   help="Dotenv file to load (default: .env). "
                        "Use .env.local for lab server / qwen2.5:32b. "
                        "Equivalent to: set ENV_FILE=.env.local")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print full plan JSON after each run. "
                        "Use --verbose --runs 1 to inspect a single plan.")
    return p.parse_args()


async def _run_full_suite(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _ssh_bin  = os.getenv("MCP_SSH_BIN",  "ssh")
    _ssh_key  = os.getenv("MCP_SSH_KEY",  "C:/Users/zrqrc/.ssh/droplet1")
    _ssh_host = os.getenv("MCP_SSH_HOST", "root@188.166.232.163")
    _cmd      = os.getenv(
        "MCP_SERVER_CMD",
        "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && "
        "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH "
        "python server_with_product.py",
    )
    server_params = StdioServerParameters(
        command=_ssh_bin,
        args=["-i", _ssh_key, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _ssh_host, _cmd],
        env=dict(os.environ),  # MCP's default env filter strips vars SSH needs
    )

    message          = args.message or MESSAGE
    n_runs           = args.runs    or N_RUNS
    with_skills      = not args.no_skills if args.no_skills else WITH_SKILLS
    compound_context = args.compound_context if args.compound_context is not None else COMPOUND_CONTEXT
    checker_name     = (args.checker or CHECKER)
    if checker_name == "auto":
        checker_name = detect_checker(message)

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

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
                    session, client, message, checker_name, with_skills, compound_context
                )
                tag = "PASS" if r["passed"] else "FAIL"
                print(f"{tag}  ({r['duration_ms']} ms)")
                if verbose and r["plan"]:
                    print(json.dumps(r["plan"], indent=2, ensure_ascii=False))
                results.append(r)

    print_report(message, "full", checker_name, with_skills, results)


def _run_plan_suite(args):
    message          = args.message or MESSAGE
    n_runs           = args.runs    or N_RUNS
    with_skills      = not args.no_skills if args.no_skills else WITH_SKILLS
    compound_context = args.compound_context if args.compound_context is not None else COMPOUND_CONTEXT
    checker_name     = (args.checker or CHECKER)
    if checker_name == "auto":
        checker_name = detect_checker(message)

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

    verbose = getattr(args, "verbose", False)

    results: List[RunResult] = []
    for i in range(n_runs):
        print(f"  Run {i+1:>3}/{n_runs}...", end=" ", flush=True)
        r = run_plan_once(client, message, checker_name, with_skills, compound_context)
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"{tag}  ({r['duration_ms']:>5} ms)")
        if verbose and r["plan"]:
            print(json.dumps(r["plan"], indent=2, ensure_ascii=False))
        results.append(r)

    print_report(message, "plan", checker_name, with_skills, results)
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
