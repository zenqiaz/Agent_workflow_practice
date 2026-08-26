"""
test_skill_schema_consistency.py -- assert the skill library agrees with the
tool schemas it describes.

Why this exists. Skill modules are prose: they restate tool names and argument
names that already exist authoritatively in the tool schemas (MCP inputSchema
for remote tools, signature introspection for skill-owned ones). Nothing kept
the two copies in agreement, so the prose drifted, silently and repeatedly:

  - the spectrum skill documented run_tddft_job(n_states=...) for a tool whose
    parameter is nroots. The planner obeyed the prose, and the argument-schema
    validator rejected the resulting plan on every single run -- a defect that
    had been latent in the library, costing a full re-plan whenever that
    workflow was requested.
  - the solvation skill documented only run_solvator_cluster_thermo, never the
    plain run_solvator_cluster, so the planner inferred the plain tool's
    arguments from the thermo variant and passed ncores, which it rejects.

Both were found by running the system, not by reading it. A schema is machine
readable, so this class of drift does not need to be found by running anything.
This test converts it into a build-time failure.

Scope is deliberately the same as the runtime argument check: names only. It
asks whether a tool or argument the prose names actually exists -- never whether
the prose gives good advice.

Usage:
  python test_skill_schema_consistency.py      # exits 1 on any inconsistency
"""

from __future__ import annotations

import re
import sys
from typing import List, Tuple

from skills import SKILL_REGISTRY
from test_success_rate import load_tool_schemas

# Tokens shaped like tool names. Kept broad on purpose: a name that looks like a
# tool and is not one is exactly what we want to hear about.
TOOL_TOKEN = re.compile(
    r"\b(run_[a-z0-9_]+|compute_[a-z0-9_]+|rank_[a-z0-9_]+|build_[a-z0-9_]+|"
    r"name_to_[a-z0-9_]+|structure_[a-z0-9_]+|inspect_[a-z0-9_]+|"
    r"pubchem_[a-z0-9_]+|set_geometry_[a-z0-9_]+)\b")

ARG_TOKEN = re.compile(r"\b([a-z_][a-z0-9_]{2,})\s*=")

# Harness-level argument names the executor injects or consumes; they are valid
# in a plan but appear in no tool schema.
HARNESS_ARGS = {
    "input_id", "output_id", "input_geom_id", "solute_id", "geom_a_id",
    "geom_b_id", "mol_a_id", "mol_b_id", "geometry_name", "applies_to",
    "needs_artifacts", "artifacts_to_save", "product", "task", "kind", "tool",
    "id", "needs", "expect", "on_error",
}


def _is_node_id_or_task(text: str, token: str, pos: int) -> bool:
    """Skill prose also uses tool-shaped words as node ids and llm task names.

    'build_dimer -> build_dimer_xyz(...)' names a node then the tool it calls;
    'task: compute_spin_gap' names an llm node's job. Neither is a tool
    reference, and flagging them would make this test cry wolf until it was
    ignored -- which is worse than not having it.
    """
    after = text[pos + len(token): pos + len(token) + 12]
    # A node id is followed by the tool it dispatches to, joined by a dash or
    # arrow: 'build_dimer — build_dimer_xyz(...)'. The skill text uses an em
    # dash here, not an arrow.
    if re.match(r"\s*(->|=>|[‒-―→⇒])", after):
        return True
    # Template node ids carry the per-compound placeholder: 'compute_pka_{C}'.
    if after.startswith("{"):
        return True
    before = text[max(0, pos - 24): pos]
    if re.search(r"(task|id)\s*:\s*[\"']?\s*$", before):
        return True
    # Prose naming an llm node: 'the kind:"llm" compute_delta_e node ...'
    if re.search(r"kind\s*:\s*[\"']?llm[\"']?\s*$", before):
        return True
    return False


def main() -> int:
    schemas = load_tool_schemas()
    known_tools = set(schemas)

    unknown_tools: List[Tuple[str, str]] = []
    unknown_args: List[Tuple[str, str, str]] = []

    for skill in SKILL_REGISTRY:
        name = getattr(skill, "name", skill.__class__.__name__)
        try:
            text = skill.render("", {})
        except Exception as exc:                      # a skill that cannot render
            unknown_tools.append((name, f"<render failed: {exc}>"))
            continue

        mentioned = set()
        for m in TOOL_TOKEN.finditer(text):
            token = m.group(1)
            if _is_node_id_or_task(text, token, m.start()):
                continue
            if token in known_tools:
                mentioned.add(token)
            else:
                unknown_tools.append((name, token))

        # Arguments are checked against the union of the parameters of the tools
        # this skill actually mentions: prose rarely says which call an argument
        # belongs to, and the union keeps false positives at zero while still
        # catching a name no mentioned tool accepts at all.
        allowed = set(HARNESS_ARGS)
        for t in mentioned:
            allowed |= set((schemas[t].get("properties") or {}).keys())
        if not mentioned:
            continue
        for m in ARG_TOKEN.finditer(text):
            arg = m.group(1)
            if arg in allowed:
                continue
            near = text[max(0, m.start() - 140): m.start()]
            owner = next((t for t in mentioned if t in near), None)
            if owner:
                unknown_args.append((name, owner, arg))

    print(f"skills: {len(SKILL_REGISTRY)}   tools in registry: {len(known_tools)}")
    print()

    ok = True
    if unknown_tools:
        ok = False
        print(f"FAIL  tool names in skill prose that do not exist ({len(unknown_tools)}):")
        for s, t in unknown_tools:
            print(f"        {s:<22} {t}")
    else:
        print("PASS  every tool named in skill prose exists in the schema registry")

    if unknown_args:
        ok = False
        print(f"FAIL  arguments attributed to a tool that rejects them "
              f"({len(unknown_args)}):")
        for s, t, a in unknown_args:
            print(f"        {s:<22} {t}({a}=...)")
    else:
        print("PASS  every argument named in skill prose is accepted by a mentioned tool")

    if not check_status_contract():
        ok = False

    if not ok:
        print("\nThe skill library and the tool schemas disagree. The schema is the "
              "source of truth: correct the skill text, not the schema.")
    return 0 if ok else 1


def check_status_contract() -> bool:
    """Skill-owned tools must satisfy the executor's status contract.

    The executor records a node as 'unknown' when a tool result carries no
    "status", and no expect.status_in can then be satisfied. collect_skill_client_tools
    guarantees the field; this asserts the guarantee is still in place, because
    the failure it prevents is silent in exactly the wrong way -- the tool
    computes the right answer and the node fails anyway, which is how three
    benchmark repeats were lost when rank_eas_sites was first added.
    """
    from skills import PlannerSkill, collect_skill_client_tools

    def _no_status(value: int = 1) -> dict:
        return {"value": value}          # a well-behaved tool that forgets status

    def _own_status(value: int = 1) -> dict:
        return {"status": "error", "value": value}   # must not be overwritten

    class _Probe(PlannerSkill):
        name = "_status_contract_probe"
        priority = 999
        client_tools = {"_probe_no_status": _no_status,
                        "_probe_own_status": _own_status}

        def matches(self, user_text, state):
            return False

        def render(self, user_text, state):
            return ""

    tools = collect_skill_client_tools([_Probe()])
    injected = tools["_probe_no_status"]()
    preserved = tools["_probe_own_status"]()

    problems = []
    if injected.get("status") != "ok":
        problems.append(f"status not injected: got {injected!r}")
    if injected.get("value") != 1:
        problems.append(f"payload altered: got {injected!r}")
    if preserved.get("status") != "error":
        problems.append(f"existing status overwritten: got {preserved!r}")

    # Every real skill-owned tool must also survive introspection, since
    # build_tool_schemas derives its schema from the signature.
    import inspect
    for name, fn in collect_skill_client_tools().items():
        try:
            inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            problems.append(f"{name}: signature not introspectable ({exc})")

    if problems:
        print(f"FAIL  client-tool status contract ({len(problems)}):")
        for p in problems:
            print(f"        {p}")
        return False
    print("PASS  skill-owned tools satisfy the executor's status contract")
    return True


if __name__ == "__main__":
    sys.exit(main())
