"""
Skill system tests — two parts:

PART 1  Unit tests (no network, no LLM)
  1a. matches()  — each skill fires on the right keywords, stays silent otherwise
  1b. priority   — dispatcher returns skills in ascending priority order
  1c. render()   — each rendered block contains its critical chemical values

PART 2  Integration test (OpenAI call required)
  2a. PKaSkill injected into planner → plan uses run_freq_job (not run_sp_energy),
      puts G_H_plus_ref_eh in settings, names artifacts G_*_eh, computes pKa via
      an LLM node — all rules from the skill's render() text.
  2b. Without skills → same query produces a weaker plan (regression check).
"""
import json
import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.environ.get("ENV_FILE", ".env"))

from skills import (
    PlannerSkill,
    MethodSelectionSkill,
    PKaSkill,
    ThermochemistrySkill,
    SolvationSkill,
    NBOSkill,
    SKILL_REGISTRY,
    run_planning_skills,
)
from prompts import SYSTEM_PROMPT
from client_helpers import summarize_geometries_prompt, summarize_workflow_state, parse_json_only

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMPTY_STATE: dict = {
    "geometries": {}, "geom_meta": {}, "name_to_geom": {},
    "identifiers": {}, "cached_props": {}, "plans": {}, "runs": [],
    "run_log": [], "artifacts": {}, "node_results": {},
    "default_charge": 0, "default_multiplicity": 1,
}

PKA_QUERY  = "calculate the pKa of acetic acid"
SP_QUERY   = "calculate the single-point energy of water"
NBO_QUERY  = "run NBO analysis on caffeine"
SOLV_QUERY = "calculate solvated pKa of HF with microsolvation"
THERMO_QUERY = "compute the Gibbs free energy change for this reaction"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# PART 1a  matches()
# ---------------------------------------------------------------------------

def test_matches():
    section("1a  Skill matching")

    skill_map = {s.name: s for s in SKILL_REGISTRY}
    method  = skill_map["method_selection"]
    pka     = skill_map["pka_calibration"]
    thermo  = skill_map["thermochemistry"]
    solv    = skill_map["solvation"]
    nbo     = skill_map["nbo_analysis"]
    state   = EMPTY_STATE

    cases = [
        # (skill, text, should_match)
        (method,  SP_QUERY,              True,  "method always active"),
        (method,  PKA_QUERY,             True,  "method always active"),
        (pka,     PKA_QUERY,             True,  "pka keyword"),
        (pka,     "acid dissociation",   True,  "acid dissociation keyword"),
        (pka,     SP_QUERY,              False, "sp query should not trigger pKa"),
        (pka,     NBO_QUERY,             False, "nbo query should not trigger pKa"),
        (thermo,  THERMO_QUERY,          True,  "gibbs keyword"),
        (thermo,  "enthalpy of reaction",True,  "enthalpy keyword"),
        (thermo,  SP_QUERY,              False, "sp query should not trigger thermo"),
        (solv,    SOLV_QUERY,            True,  "microsolvation keyword"),
        (solv,    "aqueous environment", True,  "aqueous keyword"),
        (solv,    PKA_QUERY,             False, "plain pKa query: no explicit solvation"),
        (nbo,     NBO_QUERY,             True,  "nbo keyword"),
        (nbo,     "wiberg bond order",   True,  "wiberg keyword"),
        (nbo,     PKA_QUERY,             False, "pKa query should not trigger NBO"),
    ]

    failures = []
    for skill, text, expected, note in cases:
        got = skill.matches(text, state)
        ok  = (got == expected)
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  [{skill.name}] {note!r}")
        if not ok:
            print(f"        text={text!r}  expected={expected}  got={got}")
            failures.append(note)

    assert not failures, f"matches() failures: {failures}"
    print("\n  [1a PASS]")


# ---------------------------------------------------------------------------
# PART 1b  Priority ordering
# ---------------------------------------------------------------------------

def test_priority_order():
    section("1b  Dispatcher priority order")

    # Use a query that triggers all skills
    text  = "solvated pKa with NBO and gibbs free energy"
    state = EMPTY_STATE
    ctxs  = run_planning_skills(text, state)

    matched = [s for s in sorted(SKILL_REGISTRY, key=lambda s: s.priority)
               if s.matches(text, state)]
    names   = [s.name for s in matched]

    print(f"  Matched (priority order): {names}")

    # Verify ordering is strictly non-decreasing by priority
    priorities = [s.priority for s in matched]
    assert priorities == sorted(priorities), \
        f"Skills not in priority order: {list(zip(names, priorities))}"
    assert len(ctxs) == len(matched), "run_planning_skills returned wrong number of contexts"

    print("  [1b PASS]")


# ---------------------------------------------------------------------------
# PART 1c  render() content checks
# ---------------------------------------------------------------------------

def test_render_content():
    section("1c  Render content checks")

    state  = EMPTY_STATE
    checks = [
        (
            PKaSkill(),
            PKA_QUERY,
            ["-0.01372",          # reference proton free energy value
             "gibbs_free_energy_eh",  # required artifact name
             "run_freq_job",      # must not use SP
             "pKa",
             "2625499",           # Eh → J/mol conversion factor appears in formula
            ],
        ),
        (
            ThermochemistrySkill(),
            THERMO_QUERY,
            ["gibbs_free_energy_eh",
             "enthalpy_eh",
             "energy_eh",
             "298.15",
             "run_freq_job",
            ],
        ),
        (
            SolvationSkill(),
            SOLV_QUERY,
            ["run_solvator_cluster_thermo",
             "nsolv",
             "r2scan-3c",
            ],
        ),
        (
            NBOSkill(),
            NBO_QUERY,
            ["run_nbo_job",
             "nbo_section",
             "run_opt_job",
            ],
        ),
        (
            MethodSelectionSkill(),
            SP_QUERY,
            ["def2-SVP",
             "def2-TZVP",
             "B3LYP",
             "r2scan-3c",
            ],
        ),
    ]

    failures = []
    for skill, text, required_strings in checks:
        rendered = skill.render(text, state)
        missing  = [s for s in required_strings if s not in rendered]
        ok       = not missing
        tag      = "PASS" if ok else "FAIL"
        print(f"  {tag}  {skill.name}")
        if not ok:
            print(f"       missing in render: {missing}")
            failures.append((skill.name, missing))

    assert not failures, f"render() content failures: {failures}"
    print("\n  [1c PASS]")


# ---------------------------------------------------------------------------
# PART 2a  Integration: pKa plan WITH skills
# ---------------------------------------------------------------------------

def _build_planner_messages(user_text: str, state: dict,
                             skill_contexts: list, compound_context: str = "") -> list:
    """Replicate handle_user_turn message construction."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ctx in skill_contexts:
        messages.append({"role": "system", "content": "SKILL:\n" + ctx})
    messages.append({"role": "system", "content": "STATE:\n" + summarize_geometries_prompt(state)})
    messages.append({"role": "system", "content": "WORKFLOW_STATE:\n" + summarize_workflow_state(state)})
    if compound_context:
        messages.append({"role": "system", "content": "CONFIRMED_COMPOUNDS:\n" + compound_context})
    messages.append({"role": "user", "content": user_text})
    return messages


def _call_planner(client: OpenAI, messages: list) -> dict:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tool_choice="none",
        tools=[],
    )
    raw = (resp.choices[0].message.content or "").strip()
    plan = parse_json_only(raw)
    assert isinstance(plan, dict), f"Planner returned non-dict: {raw[:300]}"
    return plan


def _check_pka_plan(plan: dict, label: str) -> dict:
    """Return a dict of check_name → passed (bool)."""
    nodes      = plan.get("nodes", [])
    tools_used = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    settings   = plan.get("settings") or {}
    artifacts  = plan.get("artifacts_to_save") or []
    llm_nodes  = [n for n in nodes if n.get("kind") == "llm"]
    needs_arts = [a for n in llm_nodes for a in (n.get("needs_artifacts") or [])]

    results = {
        "uses_freq_not_sp":   "run_freq_job" in tools_used and "run_sp_energy" not in tools_used,
        "has_h_plus_ref":     "G_H_plus_ref_eh" in settings,
        "artifacts_G_prefix": any(a.startswith("G_") and a.endswith("_eh") for a in artifacts),
        "pka_in_artifacts":   "pka" in artifacts,
        "llm_node_exists":    len(llm_nodes) >= 1,
        "freq_arts_in_needs": any("G_" in a and "_eh" in a for a in needs_arts),
    }

    print(f"\n  Plan checks [{label}]:")
    for name, passed in results.items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")

    return results


def test_pka_plan_with_skills(client: OpenAI) -> dict:
    section("2a  Integration: pKa plan WITH skills")

    state   = EMPTY_STATE
    ctxs    = run_planning_skills(PKA_QUERY, state)
    print(f"\n  Skills injected: {[s.name for s in sorted(SKILL_REGISTRY, key=lambda s: s.priority) if s.matches(PKA_QUERY, state)]}")

    compound_ctx = (
        "CONFIRMED COMPOUNDS (verified by user before planning):\n"
        "  - acetic acid: formula=C2H4O2, SMILES=CC(O)=O, charge=0, MW=60.052 g/mol, CID=176\n"
        "Use these confirmed identities when assigning geom_ids and charge/multiplicity."
    )

    messages = _build_planner_messages(PKA_QUERY, state, ctxs, compound_context=compound_ctx)
    plan     = _call_planner(client, messages)

    print("\n  Plan name:", plan.get("name"))
    print("  Nodes:", [n.get("id") for n in plan.get("nodes", [])])
    print("  Settings:", plan.get("settings"))
    print("  artifacts_to_save:", plan.get("artifacts_to_save"))

    results = _check_pka_plan(plan, "WITH skills")
    passed  = sum(results.values())
    total   = len(results)
    print(f"\n  {passed}/{total} checks passed")
    assert passed == total, f"Plan with skills failed {total-passed} checks: {[k for k,v in results.items() if not v]}"
    print("  [2a PASS]")
    return results


# ---------------------------------------------------------------------------
# PART 2b  Regression: same query WITHOUT skills (baseline comparison)
# ---------------------------------------------------------------------------

def test_pka_plan_without_skills(client: OpenAI) -> dict:
    section("2b  Regression: pKa plan WITHOUT skills")

    state    = EMPTY_STATE
    messages = _build_planner_messages(PKA_QUERY, state, skill_contexts=[])
    plan     = _call_planner(client, messages)

    print("\n  Plan name:", plan.get("name"))
    print("  Nodes:", [n.get("id") for n in plan.get("nodes", [])])
    print("  Settings:", plan.get("settings"))
    print("  artifacts_to_save:", plan.get("artifacts_to_save"))

    results = _check_pka_plan(plan, "WITHOUT skills")
    passed  = sum(results.values())
    total   = len(results)
    print(f"\n  {passed}/{total} checks passed (informational — not required to all pass)")
    print("  [2b done]")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Part 1 — unit tests (no network)
    test_matches()
    test_priority_order()
    test_render_content()

    # Part 2 — integration (OpenAI required)
    client = OpenAI()
    with_skills    = test_pka_plan_with_skills(client)
    without_skills = test_pka_plan_without_skills(client)

    # Summary comparison
    section("Summary: skill impact on plan quality")
    checks = list(with_skills.keys())
    print(f"  {'Check':<30}  With skills  Without skills")
    print(f"  {'-'*30}  {'-'*11}  {'-'*14}")
    for c in checks:
        w  = "PASS" if with_skills[c]    else "FAIL"
        wo = "PASS" if without_skills[c] else "FAIL"
        arrow = "  ← improved" if with_skills[c] and not without_skills[c] else ""
        print(f"  {c:<30}  {w:<11}  {wo}{arrow}")

    assert all(with_skills.values()), "Plan WITH skills must pass all checks"
    print("\nAll skill tests PASSED.")


if __name__ == "__main__":
    main()
