"""
test_context_scaling.py — Planner robustness under increasing compound count.

Reads organic_compounds_100.csv, randomly samples N compounds for each step in
SAMPLE_SIZES, asks the planner to make a single-point energy plan for all of
them, and checks whether each compound got a proper SP node in the plan.

Goal: find the N at which planner performance begins to degrade.

Usage
-----
  python test_context_scaling.py
  python test_context_scaling.py --csv other_list.csv
  python test_context_scaling.py --sizes 5,10,20,40,60,80,100
  python test_context_scaling.py --runs 3 --seed 42
  python test_context_scaling.py --env-file .env.local
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


def _preload_env(argv=sys.argv):
    for i, arg in enumerate(argv):
        if arg in ("--env-file", "--env") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(("--env-file=", "--env=")):
            return arg.split("=", 1)[1]
    return os.environ.get("ENV_FILE", ".env")

load_dotenv(_preload_env())

from openai import OpenAI
from test_success_rate import (
    _call_planner,
    _build_messages,
    check_struct,
    EMPTY_STATE,
)
from build_graph_from_plan import expand_template
from skills import run_planning_skills
from prompts import SYSTEM_PROMPT

LLM_MODEL     = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

# ─── defaults ─────────────────────────────────────────────────────────────────
CSV_FILE     = "organic_compounds_100.csv"
SAMPLE_SIZES = list(range(5, 105, 5))   # 5, 10, 15, …, 100
RUNS_PER_N   = 3                         # independent runs per sample size
RANDOM_SEED  = 0                         # None → truly random


# ─── CSV loader ───────────────────────────────────────────────────────────────

def load_compounds(csv_path: str) -> List[Dict[str, str]]:
    """Return list of dicts with keys from CSV header."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─── message builders ─────────────────────────────────────────────────────────

def _build_sp_message(compounds: List[Dict]) -> str:
    names = [c["compound_name"] for c in compounds]
    compound_list = "\n".join(f"  - {n}" for n in names)
    return (
        f"Calculate single-point energy (B3LYP/def2-SVP) for each of the following "
        f"{len(names)} compounds. Run them in parallel where possible. "
        f"For each compound report: SP energy (energy_eh), HOMO-LUMO gap (homo_lumo_gap_ev), "
        f"and dipole moment (dipole_moment_debye). All three are returned by run_sp_energy at no extra cost.\n\n"
        f"Compounds:\n{compound_list}"
    )


def _build_eas_message(compounds: List[Dict]) -> str:
    names = [c["compound_name"] for c in compounds]
    compound_list = "\n".join(f"  - {n}" for n in names)
    return (
        f"Compare EAS reactivity of the following {len(names)} aromatic compounds. "
        f"For each compound: optimize geometry, then run SP with NBO charge analysis, "
        f"then rank the aromatic carbon sites by NPA charge (most negative = most activated). "
        f"Use standard IUPAC ring numbering and group symmetry-equivalent sites. "
        f"Run per-compound chains in parallel. "
        f"Report a ranked site list per compound.\n\n"
        f"Compounds:\n{compound_list}"
    )


# ─── task registry ────────────────────────────────────────────────────────────

# Each entry: (message_builder, checker_fn, label)
# checker_fn signature: (plan, compounds) -> Dict[str, bool]

# ─── plan checkers ────────────────────────────────────────────────────────────

def check_sp_scaling(plan: dict, compounds: List[Dict]) -> Dict[str, bool]:
    """
    Checks beyond basic struct:
    - n_sp_correct      : number of run_sp_energy nodes == number of compounds
    - names_all_found   : every compound name appears in plan text
    - no_freq           : no run_freq_job nodes (wrong tool for SP task)
    - has_energy_art    : energy_eh artifacts produced
    - has_homo_lumo_art : homo_lumo_gap_ev artifacts produced
    - has_dipole_art    : dipole_moment_debye artifacts produced
    - no_llm_node       : no unnecessary LLM calculator node
    """
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    n_sp       = sum(1 for t in tools_used if t == "run_sp_energy")
    n_expected = len(compounds)

    all_keys   = set(plan.get("artifacts_to_save") or [])
    for n in nodes:
        all_keys.update((n.get("product") or {}).keys())
    # Auto-artifact mode: planner declares outputs in final_report.fields instead of product dicts
    fr = plan.get("final_report") or {}
    all_keys.update(fr.get("fields") or [])

    def _is_energy(k: str) -> bool:
        k = k.lower()
        return (k.startswith("e_") or k.endswith("_eh") or "energy" in k) and not k.startswith("g_")

    # Check each compound name appears somewhere in the plan text
    plan_text  = json.dumps(plan).lower()
    names_found = [
        c["compound_name"]
        for c in compounds
        if c["compound_name"].lower().replace(" ", "_") in plan_text
        or c["compound_name"].lower() in plan_text
    ]

    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]

    all_keys_lower = {k.lower() for k in all_keys}

    return {
        "n_sp_correct":      n_sp == n_expected,
        "n_sp_partial":      n_sp >= max(1, n_expected // 2),
        "names_all_found":   len(names_found) == n_expected,
        "names_most_found":  len(names_found) >= max(1, int(n_expected * 0.8)),
        "no_freq":           "run_freq_job" not in tools_used,
        "has_energy_art":    any(_is_energy(k) for k in all_keys),
        "has_homo_lumo_art": any("homo" in k or "lumo" in k or "gap" in k for k in all_keys_lower),
        "has_dipole_art":    any("dipole" in k for k in all_keys_lower),
        "no_llm_node":       len(llm_nodes) == 0,
    }


def check_eas_scaling(plan: dict, compounds: List[Dict]) -> Dict[str, bool]:
    """
    EAS-specific checks:
    - n_sp_correct    : one run_sp_energy with nbo properties per compound
    - n_opt_correct   : one run_opt_job per compound (required before NBO)
    - has_ranking_llm : at least one LLM node per compound for site ranking
    - no_bad_tools    : no hallucinated tools outside the valid set
    - names_all_found : every compound name present in plan
    - has_npa_artifact: NPA/NBO charge artifacts produced
    - has_ranking_art : site ranking artifacts produced
    """
    from test_success_rate import _VALID_TOOLS

    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    tools_set  = set(tools_used)
    n_expected = len(compounds)

    n_sp  = sum(1 for t in tools_used if t == "run_sp_energy")
    n_opt = sum(1 for t in tools_used if t == "run_opt_job")

    # SP nodes that request NBO
    sp_nodes  = [n for n in nodes if n.get("tool") == "run_sp_energy"]
    n_nbo_sp  = sum(
        1 for n in sp_nodes
        if "nbo" in (n.get("args") or {}).get("properties", [])
        or "run_nbo_job" in tools_set   # fallback: standalone NBO job
    )
    n_nbo_job = sum(1 for t in tools_used if t == "run_nbo_job")
    n_charge_calc = max(n_nbo_sp, n_nbo_job)

    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]

    all_keys  = set(plan.get("artifacts_to_save") or [])
    for n in nodes:
        all_keys.update((n.get("product") or {}).keys())

    plan_text   = json.dumps(plan).lower()
    names_found = [
        c["compound_name"] for c in compounds
        if c["compound_name"].lower().replace(" ", "_") in plan_text
        or c["compound_name"].lower() in plan_text
    ]

    bad_tools = [t for t in tools_set if t and t not in _VALID_TOOLS]

    return {
        "n_opt_correct":    n_opt == n_expected,
        "n_opt_partial":    n_opt >= max(1, n_expected // 2),
        "n_charge_correct": n_charge_calc == n_expected,
        "n_charge_partial": n_charge_calc >= max(1, n_expected // 2),
        "has_ranking_llm":  len(llm_nodes) >= n_expected,
        "no_bad_tools":     len(bad_tools) == 0,
        "names_all_found":  len(names_found) == n_expected,
        "names_most_found": len(names_found) >= max(1, int(n_expected * 0.8)),
        "has_npa_artifact": any(
            any(kw in k.lower() for kw in ("npa", "nbo", "charge")) for k in all_keys
        ),
        "has_ranking_art":  any(
            any(kw in k.lower() for kw in ("ranking", "site", "eas")) for k in all_keys
        ),
    }


TASK_REGISTRY = {
    #  task_name : (message_builder,   checker_fn)
    "sp":  (_build_sp_message,  check_sp_scaling),
    "eas": (_build_eas_message,  check_eas_scaling),
}

NO_TEMPLATE_SUFFIX = (
    "\n\nIMPORTANT: Do NOT use template mode. "
    "Write all nodes explicitly in the flat 'nodes' list. "
    "Do not use a 'template' key in the plan."
)

EXPLICIT_ARTIFACT_SUFFIX = (
    "\n\nIMPORTANT: Use template mode (per_compound nodes with {C} placeholders in node fields "
    "such as id, input_id, output_id, product keys, and needs entries), but override the default "
    "artifact rule: list every compound's artifact names EXPLICITLY in artifacts_to_save and "
    "geom_ids — do NOT use {C} patterns there. "
    "For example: \"artifacts_to_save\": [\"energy_acetone_eh\", \"energy_ethane_eh\", ...] "
    "with one entry per compound."
)


# ─── single run ───────────────────────────────────────────────────────────────

def run_once(
    client: OpenAI,
    compounds: List[Dict],
    task: str = "sp",
    with_skills: bool = True,
    no_template: bool = False,
    explicit_artifact: bool = False,
) -> Dict:
    """Plan once; return result dict."""
    build_msg, check_fn = TASK_REGISTRY[task]
    state   = dict(EMPTY_STATE)
    message = build_msg(compounds)
    if no_template:
        message += NO_TEMPLATE_SUFFIX
    elif explicit_artifact:
        message += EXPLICIT_ARTIFACT_SUFFIX
    t0      = time.monotonic()
    error   = ""
    plan_raw: Dict = {}   # raw planner output (pre-expansion)
    plan:     Dict = {}   # expanded flat plan used for checks
    checks  = {}
    tu: Dict[str, int] = {}

    try:
        skill_ctxs = run_planning_skills(message, state) if with_skills else []
        msgs       = _build_messages(message, state, skill_ctxs, compound_context="")
        plan_raw, tu = _call_planner(client, msgs)
        plan       = expand_template(plan_raw)   # expand template → flat nodes before checking
        checks.update(check_struct(plan))
        checks.update(check_fn(plan, compounds))
    except Exception as exc:
        error           = str(exc)
        checks["exception"] = False

    duration_ms = int((time.monotonic() - t0) * 1000)
    passed      = all(checks.values()) and not error

    return {
        "n_compounds":         len(compounds),
        "passed":              passed,
        "checks":              checks,
        "error":               error,
        "duration_ms":         duration_ms,
        "token_usage":         tu,
        "plan_nodes":          len(plan.get("nodes", [])) if plan else 0,
        "used_template":       bool(plan_raw.get("template")) if plan_raw else False,
        "n_compounds_in_plan": len(plan_raw.get("compounds", [])) if plan_raw else 0,
        "no_template_forced":  no_template,
        "explicit_artifact":   explicit_artifact,
        "plan_json":           plan_raw,   # raw planner output; useful as sample plan
    }


# ─── scaling sweep ────────────────────────────────────────────────────────────

def run_sweep(
    client: OpenAI,
    all_compounds: List[Dict],
    sample_sizes: List[int],
    runs_per_n: int,
    rng: random.Random,
    task: str = "sp",
    with_skills: bool = True,
    no_template: bool = False,
    explicit_artifact: bool = False,
) -> List[Dict]:
    """Run all (N, run) combinations; return list of result records."""
    records = []
    for n in sample_sizes:
        if n > len(all_compounds):
            print(f"  N={n}: skipped (only {len(all_compounds)} compounds available)")
            continue
        for run_i in range(runs_per_n):
            sample = rng.sample(all_compounds, n)
            print(f"  N={n:>3}  run {run_i+1}/{runs_per_n} ...", end=" ", flush=True)
            r = run_once(client, sample, task=task, with_skills=with_skills,
                         no_template=no_template, explicit_artifact=explicit_artifact)
            tag = "PASS" if r["passed"] else "FAIL"
            ctx     = r["token_usage"].get("context_tokens", 0)
            plan_tk = r["token_usage"].get("plan_tokens", 0)
            n_nodes = r["plan_nodes"]
            if r.get("no_template_forced"):
                mode_tag = " [flat]"
            elif r.get("explicit_artifact"):
                mode_tag = " [explicit-art]"
            elif r.get("used_template"):
                mode_tag = " [full-scheme]"
            else:
                mode_tag = ""
            print(f"{tag}  ({r['duration_ms']} ms)  "
                  f"ctx={ctx} plan={plan_tk} nodes={n_nodes}{mode_tag}")
            records.append({**r, "run": run_i + 1, "sample_names": [c["compound_name"] for c in sample]})
    return records


# ─── report ───────────────────────────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 15) -> str:
    filled = round(width * n / total) if total else 0
    return "#" * filled + "." * (width - filled)


def print_scaling_report(records: List[Dict], runs_per_n: int) -> None:
    # Group by n_compounds
    by_n: Dict[int, List[Dict]] = {}
    for r in records:
        by_n.setdefault(r["n_compounds"], []).append(r)

    print("\n" + "=" * 110)
    print("  Context Scaling Report")
    print("=" * 110)
    print(f"  {'N':>4}  {'Pass':>6}  {'Rate':>6}  {'Bar':<15}  "
          f"{'avg ctx':>7}  {'avg plan':>8}  {'avg nodes':>9}  {'avg time':>8}  Degraded checks")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*15}  "
          f"{'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*30}")

    prev_rate = 100.0
    first_degraded_n = None

    for n in sorted(by_n):
        runs   = by_n[n]
        n_pass = sum(1 for r in runs if r["passed"])
        rate   = n_pass / len(runs) * 100

        avg_ctx   = sum(r["token_usage"].get("context_tokens", 0) for r in runs) // len(runs)
        avg_plan  = sum(r["token_usage"].get("plan_tokens",    0) for r in runs) // len(runs)
        avg_nodes = sum(r["plan_nodes"] for r in runs) / len(runs)
        avg_time_s = sum(r["duration_ms"] for r in runs) / len(runs) / 1000

        # Find most commonly failing check
        fail_counts: Dict[str, int] = {}
        for r in runs:
            for k, v in r["checks"].items():
                if not v:
                    fail_counts[k] = fail_counts.get(k, 0) + 1
        top_fails = sorted(fail_counts, key=lambda k: -fail_counts[k])[:2]
        fail_str  = ", ".join(f"{k}({fail_counts[k]})" for k in top_fails) if top_fails else ""

        degraded = rate < 100.0
        if degraded and first_degraded_n is None:
            first_degraded_n = n
        marker = " <<" if degraded and prev_rate == 100.0 else ""

        print(f"  {n:>4}  {n_pass:>3}/{len(runs):<2}  {rate:>5.0f}%  "
              f"{_bar(n_pass, len(runs)):<15}  "
              f"{avg_ctx:>7}  {avg_plan:>8}  {avg_nodes:>9.1f}  {avg_time_s:>7.1f}s  {fail_str}{marker}")

        prev_rate = rate

    print()
    if first_degraded_n:
        print(f"  First degradation at N={first_degraded_n} compounds.")
    else:
        print(f"  No degradation observed across all tested sizes.")


# ─── log saver ────────────────────────────────────────────────────────────────

def save_log(records: List[Dict], sample_sizes: List[int], runs_per_n: int,
             csv_file: str, task: str = "sp", log_dir: str = "test_logs",
             wall_elapsed_s: float = 0.0, no_template: bool = False,
             explicit_artifact: bool = False) -> str:
    os.makedirs(log_dir, exist_ok=True)
    now   = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if no_template:
        mode      = "no_template"
        mode_tag  = "_flat"
    elif explicit_artifact:
        mode      = "explicit_artifact"
        mode_tag  = "_expart"
    else:
        mode      = "full_scheme"
        mode_tag  = "_fullscheme"
    fname = f"scaling_{now}_{task}{mode_tag}_n{min(sample_sizes)}-{max(sample_sizes)}.json"
    path  = os.path.join(log_dir, fname)

    by_n: Dict[int, List] = {}
    for r in records:
        by_n.setdefault(r["n_compounds"], []).append(r)

    summary = []
    for n in sorted(by_n):
        runs   = by_n[n]
        n_pass = sum(1 for r in runs if r["passed"])
        summary.append({
            "n_compounds":        n,
            "n_runs":             len(runs),
            "n_pass":             n_pass,
            "pass_rate":          round(n_pass / len(runs) * 100, 1),
            "avg_context_tokens": sum(r["token_usage"].get("context_tokens", 0) for r in runs) // len(runs),
            "avg_plan_tokens":    sum(r["token_usage"].get("plan_tokens",    0) for r in runs) // len(runs),
            "avg_plan_nodes":     round(sum(r["plan_nodes"] for r in runs) / len(runs), 1),
            "avg_time_s":         round(sum(r["duration_ms"] for r in runs) / len(runs) / 1000, 1),
        })

    log = {
        "type":              "context_scaling_test",
        "timestamp_utc":     now,
        "csv_file":          csv_file,
        "task":              task,
        "mode":              mode,
        "sample_sizes":      sample_sizes,
        "runs_per_n":        runs_per_n,
        "model":             LLM_MODEL,
        "wall_elapsed_s":    round(wall_elapsed_s, 1),
        "summary":           summary,
        "runs":              records,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=True)
    return path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Planner context-scaling robustness test")
    p.add_argument("--csv",   default=CSV_FILE,
                   help=f"Compound CSV file (default: {CSV_FILE})")
    p.add_argument("--sizes", default=None,
                   help="Comma-separated sample sizes, e.g. '5,10,20,40'. "
                        "Default: 5,10,15,...,100")
    p.add_argument("--runs",  type=int, default=RUNS_PER_N,
                   help=f"Runs per sample size (default: {RUNS_PER_N})")
    p.add_argument("--seed",  type=int, default=RANDOM_SEED,
                   help="Random seed (default: 0); use -1 for truly random")
    p.add_argument("--task", choices=list(TASK_REGISTRY), default="sp",
                   help="Task type to test: 'sp' (single-point) or 'eas' (EAS reactivity). "
                        "Default: sp")
    p.add_argument("--no-skills", action="store_true",
                   help="Disable skill injection (baseline comparison)")
    p.add_argument("--no-template", action="store_true",
                   help="Override planner to use flat nodes list instead of template mode "
                        "(flat mode)")
    p.add_argument("--explicit-artifact", action="store_true",
                   help="Use template mode for plan structure but enumerate all artifact names "
                        "explicitly in artifacts_to_save (explicit-artifact mode)")
    p.add_argument("--env-file", "--env", default=None)
    return p.parse_args()


def main():
    args  = parse_args()
    seed  = None if args.seed == -1 else args.seed
    rng   = random.Random(seed)

    if args.sizes:
        sample_sizes = [int(x) for x in args.sizes.split(",")]
    else:
        sample_sizes = list(range(5, 105, 5))

    with_skills       = not args.no_skills
    no_template       = args.no_template
    explicit_artifact = args.explicit_artifact
    task              = args.task

    if no_template and explicit_artifact:
        raise SystemExit("--no-template and --explicit-artifact are mutually exclusive.")

    if no_template:
        mode_label = "no-template (flat)"
    elif explicit_artifact:
        mode_label = "explicit-artifact"
    else:
        mode_label = "full-scheme"

    all_compounds = load_compounds(args.csv)
    print(f"Loaded {len(all_compounds)} compounds from {args.csv}")
    print(f"Task: {task}  |  Sample sizes: {sample_sizes}")
    print(f"Runs per N:   {args.runs}  |  Seed: {seed}  |  Skills: {'ON' if with_skills else 'OFF'}  |  Mode: {mode_label}")
    print(f"Model: {LLM_MODEL}\n")

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    wall_t0 = time.monotonic()
    records = run_sweep(client, all_compounds, sample_sizes, args.runs, rng,
                        task=task, with_skills=with_skills, no_template=no_template,
                        explicit_artifact=explicit_artifact)
    wall_elapsed_s = time.monotonic() - wall_t0

    print_scaling_report(records, args.runs)

    log_path = save_log(records, sample_sizes, args.runs, args.csv, task=task,
                        wall_elapsed_s=wall_elapsed_s, no_template=no_template,
                        explicit_artifact=explicit_artifact)
    total_m, total_s = divmod(int(wall_elapsed_s), 60)
    print(f"  Total wall time: {total_m}m {total_s:02d}s")
    print(f"  Log saved: {log_path}")


if __name__ == "__main__":
    main()
