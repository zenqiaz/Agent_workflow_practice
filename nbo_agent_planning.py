# nbo_agent_repl.py
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict, TypedDict, List
import traceback
from dataclasses import asdict
from pprint import pprint

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI
from urllib.request import urlretrieve

from client_helpers import (
    NEEDS_GEOM_SINGLE,
    TOOLS_RETURNING_STRUCTURE,
    load_xyz_as_geometry,
    summarize_geometries_prompt,
    geom_key_from_path,
    name_to_geometry_xyz,
    print_tool_output,
    get_trivial_properties,
    pubchem_get_basic_properties,
    ALLOWED_STATE_KEYS,
    state_update,
    state_get_tool_args,
    _maybe_store_geometry_from_payload,
    xyz_to_no_header,
    print_session_state,
    handle_planimg_command,
    handle_plansave_command,
    summarize_workflow_state,
    parse_json_only,
    build_plan_review,
    print_plan_review,
    #run_node_via_existing_executor,
    #run_plan_deterministically,
    run_tool_node,
    make_run_llm_node,
    result_dict_to_prompt,
)
from build_graph_from_plan import build_graph_from_plan, build_state



load_dotenv()


SYSTEM_PROMPT = """
You are QC-PLANNER, a workflow planning agent for molecular quantum-chemistry tasks.

Mission
- Produce a complete, deterministic WORKFLOW PLAN in JSON for the user’s request.
- Assume all required QC tools exist (geometry loading, ORCA OPI jobs, solvator/microsolvation, parsing, analysis, database I/O).
- Assume facts, constants and other resourses can be retrieved via agent skill
- Do NOT execute tools. Do NOT fabricate numerical results. Do NOT ask the user to run commands.
- Your output MUST be a single valid JSON object and nothing else. User input must be copied into the plan.

Operating model (control room vs assembly line)
- The workflow should run mostly without LLM intervention once determined.
- Use custodian-like planned error handling (retry/patch/branch) wherever possible.
- Only include LLM “supervisor” steps for a SMALL, task-specific whitelist of potential patterns (typically 1–3). Do NOT design a general “fix everything” supervisor.

Node types
- There are 3 kinds of node in the workflow: tool, calculation and LLM, each node should show their type in its kind.
- tools: running tools. kind: "tool"
- calculation: calculating with values yielded form tools or other sources. kind: "calc"
- LLM: generate report, or for other purposes necessary like severe error handling. kind: "LLM"

TOOL NODES
- Always omit charge and multiplicity in tool input, do not guess the values.
- Tools are invoked by name with JSON args and return JSON with:
  - status: "ok" | "warning" | "error"
  - code: optional machine code (e.g., "SCF_NOT_CONVERGED", "IMAG_FREQ", "GEOM_INVALID", "RESOURCE_LIMIT", "PARSER_FAILED", ...)
  - messages: optional list of strings
  - artifacts_required: optional dict of artifact references (paths/ids/uris)
   - artifact keys MUST be stable, human-readable identifiers that match plan node.product keys (and ideally appear in artifacts_to_save). 
   - except for name_to_geometry_xyz, all tool nodes MUST contain a geometry artifact.
  
CALC NODES (deterministic math)

Purpose
- A calc node computes one or more numeric/text values deterministically (no tool call).
- Calc nodes MUST NOT invent new artifact keys in free text. All outputs must be declared in `product`.

Required fields (calc node)
- kind: "calc"  (alias: "expr" or "calc_expr" allowed if supported)
- id: unique string
- needs: [ "<artifact_key>", ... ]   // REQUIRED: artifact keys needed before this node runs
- expr: string                      // REQUIRED: expression to evaluate
- inputs: { SYMBOL: "artifacts.<key>" | "node_results.<node_id>.<path>" | "settings.<key>" }  // REQUIRED
- constants: { SYMBOL: number }     // OPTIONAL, small numeric constants only
- product: { "<artifact_key>": "<result_path>" }  // REQUIRED

Evaluation rules
- Build an evaluation environment ENV:
  - For each (SYMBOL, PATH) in `inputs`, resolve PATH from the current state/result and set ENV[SYMBOL] = float(value) (or keep as string if explicitly needed).
  - For each (SYMBOL, NUMBER) in `constants`, set ENV[SYMBOL] = float(NUMBER).
- Evaluate `expr` using ENV (safe arithmetic only: + - * / ** parentheses, and selected functions if supported).
- The calc node returns a JSON result with at least:
  - status: "ok"
  - value: the computed scalar (default output)
  - values: optional dict if multiple named outputs are produced

Output mapping
- If `product` maps to "value", stash the computed scalar into state.artifacts[<artifact_key>].
- If `product` maps to "values.<name>", stash that named output.
- Calc nodes SHOULD write outputs into artifact keys that already exist in plan.artifacts_to_save.

Constraints
- needs MUST be a list (JSON array) of artifact keys (strings).
- Calc nodes MUST be deterministic: no randomness, no tool use, no hidden constants.
- If a constant is conceptually global (e.g., R, T, ln10), prefer to place it in plan.artifacts_to_save as a pre-seeded artifact constant, then reference it via inputs rather than hardcoding in expr.

- The deterministic executor will:
  - call tools
  - run math calculations
  - persist artifacts
  - resolve references between nodes
  - apply on_error policies
  - only call LLM supervisor steps that YOU explicitly include and whitelist

Workflow JSON output contract (MUST follow, if the domain is not needed in this plan, let it be empty rather than delete it. Not necessarily to contain all kinds of nodes in the plan.)
Output a single JSON object with these top-level keys:
{
  "user_text": "Please first optimize the geometry of this molecule and calculate frequency and SP energy."
  "name": "Example: DFT opt + freq + SP (GeometryRegistry-based)",
  "version": "1.3",
  "geom_ids": ["the_molecule"],
  "artifacts_to_save": [
    "freq_summary",
    "gibbs_free_energy_hartree",
    "sp_energy_hartree",
    "final_report_md",
    "deltaG_pka_J_mol",
    "pka"
  ],
  "settings": {
    "gas_constant_R_J_molK": 8.314462618,
    "temperature_K": 298.15,
    "ln10": 2.302585092994046
  },
  "nodes": [
    {
      "id": "load_geom",
      "kind": "tool",
      "tool": "name_to_geometry_xyz",
      "output_id": "the_molecule",
      "args": { "name": "water" },
      "expect": { "status_in": ["ok"], "artifacts_required": [], "properties_required": [] },
      "product": {}
    },
    {
      "id": "opt",
      "kind": "tool",
      "tool": "run_opt_job",
      "input_id": "the_molecule",
      "output_id": "the_molecule",
      "args": {
        "input_geom_id": "the_molecule",
        "method": "B3LYP",
        "basis": "def2-SVP",
      },
      "expect": { "status_in": ["ok"], "artifacts_required": [], "properties_required": [] },
      "product": {}
    },
    {
      "id": "freq",
      "kind": "tool",
      "tool": "orca_freq",
      "input_id": "the_molecule",
      "output_id": "the_molecule",
      "args": {
        "input_geom_id": "the_molecule",
        "method": "B3LYP",
        "basis": "def2-SVP",
      },
      "expect": { "status_in": ["ok"], "artifacts_required": [], "properties_required": [] },
      "product": {
        "freq_summary": "properties.freq_summary",
        "gibbs_free_energy_hartree": "properties.gibbs_free_energy_hartree"
      }
    },
    {
      "id": "sp",
      "kind": "tool",
      "tool": "run_sp_energy",
      "input_id": "the_molecule",
      "output_id": "the_molecule",
      "args": {
        "input_geom_id": "the_molecule",
        "method": "B3LYP",
        "basis": "def2-SVP",
      },
      "expect": { "status_in": ["ok"], "artifacts_required": [], "properties_required": [] },
      "product": { "sp_energy_hartree": "properties.total_energy_hartree" }
    },
    {
      "id": "report",
      "kind": "llm_task",
      "task": "write_final_report",
      "inputs": {
        "geom_id": "the_molecule",
        "freq_summary": "$(artifacts.freq_summary)",
        "gibbs_free_energy_hartree": "$(artifacts.gibbs_free_energy_hartree)",
        "sp_energy_hartree": "$(artifacts.sp_energy_hartree)"
      },
      "product": { "final_report_md": "artifacts.final_report_md" }
    },
    {
      "id": "calc_pka",
      "kind": "calc",
      "needs": ["deltaG_pka_J_mol"],
      "expr": "deltaG_pka_J_mol / (gas_constant_R_J_molK * temperature_K * ln10)",
      "inputs": {
        "deltaG_pka_J_mol": "artifacts.deltaG_pka_J_mol",
        "gas_constant_R_J_molK": "settings.gas_constant_R_J_molK",
        "temperature_K": "settings.temperature_K",
        "ln10": "settings.ln10"
      },
      "product": { "pka": "value" }
    }
  ],
  "final_report": { "format": "markdown", "fields": ["final_report_md"] }
}

Artifacts contract
- artifacts_to_save is the authoritative list of artifact KEYS that must be available in state["artifacts"] at the end.
- Every key in artifacts_to_save MUST be produced by at least one node via node.product and/or tool-result artifacts/properties.
- For tool nodes: if you declare product {"K": ...} and K is an artifact, the tool MUST return result.artifacts.K (preferred) or an equivalent path you map in product.
- Node args may reference artifacts with $(artifacts.K) or $(node_id.artifacts.K) (use whichever is more natural).

Geometry rules
- First, check state to see if the user has added geometries manually.
- If the user tells the directory of the xyz file, choose load_xyz_as_geometry to retrieve the geometry.
- Else, choose name_to_geometry_xyz to retrieve the geometry.

Reference and templating rules
- Use string references like "$(node_id.properties.G_aq_hartree)" or "$(node_id.artifacts.final_xyz)" inside args and summarizer.expected_inputs.
- Every node must have unique "id".
- Every node must list "needs" (empty array allowed for first nodes).
- "expect" MUST be present for every tool node to enable deterministic validation.
- For every tool node that is critical to the goal, include at least one on_error rule for common failures.

Error handling policy (custodian-like)
- Prefer deterministic fixes (patch_and_retry) for predictable errors:
  - SCF_NOT_CONVERGED (increase maxiter, add damping/level shift, change guess, etc.)
  - GEOM_INVALID (pre-opt with cheaper method, constrain, rebuild, etc.)
  - RESOURCE_LIMIT (downshift method/basis, reduce parallelism, etc.)
- Cap retries (max_attempts). If exhausted, stop OR escalate to a whitelisted supervisor pattern.
- Do NOT escalate to LLM by default. Only escalate for patterns listed in supervision.allowed_patterns.

Supervisor (LLM) policy
- Only include kind="llm" nodes for 1–3 patterns that are actually plausible and consequential for this specific task.
- Each supervisor node must be narrowly scoped to its "pattern" and must have a strict output schema.
- Supervisor output MUST be machine-actionable and minimal. Allowed supervisor actions:
  - patch_and_retry (target an existing node with a JSON patch of args)
  - branch (choose among predeclared branch_to options)
  - stop (with a clear reason)
- Supervisor MUST NOT invent new tools, new scientific results, or long prose.

Summarization policy (LLM post-processing)
- Tools do QC computations; the summarizer does lightweight math + interpretation (e.g., ΔG → pKa) and reporting.
- The plan MUST specify what scalar data are expected from tools (e.g., G_aq for HA and A−) and exactly how the summarizer should compute derived quantities.
- Summarizer must not compute if required inputs are missing or flagged; instead it should explain what is missing and how the workflow should be adjusted.

Domain requirements (QC)
- If user requests ΔG, pKa, or equilibrium constants, the workflow must include thermochemistry (frequency) OR explicitly rely on a validated protocol tool that provides those quantities.
- For pKa planning, ensure the plan yields free energies for HA and A− in the appropriate environment (gas/solvent/microsolvation) as required by the chosen validated protocol.
- Record any external reference constants (e.g., aqueous proton free energy reference) as explicit assumptions and summarizer.constants, not implicit text.

Output constraints
- Output MUST be valid JSON (no markdown, no comments, no trailing commas).
- Output MUST contain no extra text outside the JSON.


""".strip()

REPORTER_SYSTEM_PROMPT = """You are a workflow reporter.
Write a concise, factual markdown report of the run result.
- Highlight key numeric results (energies, ΔG, frequencies) with units.
- Mention failures/errors clearly and suggest next debugging step.
- Do not invent values not present in the input.
"""

OPENAI_TOOLS = json.loads(Path("openai_tools_geom.json").read_text(encoding="utf-8"))
CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    
    "state_update": lambda **kw: state_update(state, **kw),          # not so useful: applicable only when data readily to be fill into the state
    "state_get_tool_args": lambda **kw: state_get_tool_args(state, **kw),
    #"pubchem_get_record_fields": pubchem_get_record_fields,
}


class AgentState(TypedDict, total=False):
    # -------- session / planning data --------
    files: Dict[str, str]
    geometries: Dict[str, str]
    current_geom: Optional[str]
    name_to_geom: Dict[str, str]
    identifiers: Dict[str, Dict[str, Any]]
    geom_meta: Dict[str, Dict[str, Any]]
    cached_props: Dict[str, Dict[str, Any]]

    default_charge: int
    default_multiplicity: int

    latest_plan_name: Optional[str]
    approved_plan_name: Optional[str]
    plans: Dict[str, Dict[str, Any]]      # plan_name -> plan_pkg
    last_run_id: Optional[str]
    runs: List[Dict[str, Any]]
    last_user_text: str

    # -------- execution / graph bookkeeping not used--------
    run_log: List[Dict[str, Any]]
    last_status: str
    last_tool_result: Dict[str, Any]

    artifacts: Dict[str, Any]
    node_results: Dict[str, Any]

    final_report: Dict[str, Any]
    result: Any

def store_plan(state: AgentState, plan_pkg: dict) -> None:
    plan = plan_pkg.get("plan") or {}
    name = plan.get("name") or plan.get("title") or f"plan_{len(state.get('plans', {}))+1}"
    plan.setdefault("name", name)
    state.setdefault("plans", {})[name] = plan_pkg
    state["latest_plan_name"] = name

def approve_latest_plan(state: AgentState) -> None:
    state["approved_plan_name"] = state.get("latest_plan_name")

def tools_for_server(tool_names: list[str]) -> list[dict]:
    return [OPENAI_TOOLS[name] for name in tool_names if name in OPENAI_TOOLS]



async def handle_user_turn(session, client, state, user_text: str, tools_for_this_call: list):
    # Merge: MCP tools + client-side tools, but only if schema exists in OPENAI_TOOLS


    '''user_prompt = f"""
{summarize_geometries_prompt(state)}

User request:
{user_text}
""".strip()'''
    #state_msg_index = 1
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": "STATE:\n" + summarize_geometries_prompt(state)},
    {"role": "system", "content": "WORKFLOW_STATE:\n" + summarize_workflow_state(state)},
    {"role": "user", "content": user_text},
    ]
    state["last_user_text"] = user_text

    MAX_TOOL_ROUNDS = 4
    for _round in range(MAX_TOOL_ROUNDS):
        #print("TOOL NAMES FOR MODEL:", tool_names_for_model)
        #print("TOOLS SENT:", [t["function"]["name"] for t in tools_for_this_call])
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_for_this_call,
            tool_choice="none",
        )
        msg = resp.choices[0].message
        messages.append(msg)
        raw = (msg.content or "").strip()
        print(raw)
        plan = parse_json_only(raw)
        review = build_plan_review(plan)
        plan_pkg = {"plan": plan, "review": review, "ui": {"saved_files": {}}}
        store_plan(state, plan_pkg)
        print_plan_review(plan_pkg)
        return plan_pkg


    #print("\n[Final]\nEarly exit: too many tool rounds. Please rephrase or provide missing info.")


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "server_with_product.py"],
        env={"PATH": "/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:" + os.environ.get("PATH", "")},
    )

    client = OpenAI()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            state: AgentState = {
    "files": {},
    "geometries": {},
    "name_to_geom": {},
    "identifiers": {},
    "geom_meta": {},
    "cached_props": {},
    "plans": {},
    "runs": [],
    "run_log": [],
    "artifacts": {},
    "node_results": {},
    "default_charge": 0,
    "default_multiplicity": 1,
            }

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print("MCP tools available:", tool_names)
            tool_names_for_model = [n for n in tool_names if n in OPENAI_TOOLS]
            for n in CLIENT_SIDE_TOOL_FUNCS:
                if n in OPENAI_TOOLS and n not in tool_names_for_model:
                    tool_names_for_model.append(n)

            tools_for_this_call = [OPENAI_TOOLS[n] for n in tool_names_for_model]

            

            run_llm_node = make_run_llm_node(
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
            )
            while True:
                line = input("> ").strip()
                if not line:
                    continue
                if line.lower() in {"quit", "exit"}:
                    break

                if line.lower().startswith("load "):
                    path = line.split(maxsplit=1)[1]
                    geometry_xyz = load_xyz_as_geometry(path)
                    key = geom_key_from_path(path)
                    state.setdefault("files", {})[key] = path
                    state.setdefault("geometries", {})[key] = geometry_xyz
                    state["current_geom"] = key
                    print(f"Loaded geometry from {path!r} as {key!r}")
                    continue

                if line.strip().lower() == "state":
                    print_session_state(state, preview_lines=6, show_full_current=True, max_full_lines=None)
                    continue

                if line.strip().lower().startswith("planimg"):
                    handle_planimg_command(state, line)
                    continue

                if line.strip().lower().startswith("plansave"):
                    handle_plansave_command(state, line)
                    continue
                
                
                if line.strip().lower() == "run":
                    # 1) pick plan (prefer approved)
                    plan_name = state.get("approved_plan_name") or state.get("latest_plan_name")
                    plan_pkg = (state.get("plans") or {}).get(plan_name)

                    if not plan_pkg:
                        print("No plan found. Use 'plan' first.")
                        continue

                    # 2) show review
                    print_plan_review(plan_pkg)
                    yn = input("Execute this plan? (y/N): ").strip().lower()
                    if yn not in ("y", "yes"):
                        print("Cancelled.")
                        continue


                    # 3) execute
                    graph = build_graph_from_plan(plan_pkg["plan"], get_tool_args=state_get_tool_args,
                             run_tool_node=run_tool_node, run_llm_node=run_llm_node)

                    init_state = build_state(plan_pkg["plan"], session, seed={
                        "client_side_tools": CLIENT_SIDE_TOOL_FUNCS,  # optional but recommended
                    },client_side_tools=CLIENT_SIDE_TOOL_FUNCS)
                    result = await graph.ainvoke(init_state)   # or graph.invoke(...)
                    print("Done:", result.get("status", "unknown"))
                    pprint(result.get("artifacts"))
                    user_payload = (
                        f"Original user request:\n{state.get('last_user_text','(not provided)')}\n\n"
                        f"Run result:\n{result_dict_to_prompt(result=result, user_text=state.get('last_user_text'))}"
                    )
                    report_messages = [
                        {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ]
                    
                    resp = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=report_messages,
                    )
                    msg = resp.choices[0].message
                    print(msg.content)
                    #print(json.dumps(result, ensure_ascii=False, indent=2))
                    continue
                
                await handle_user_turn(session, client, state, line, tools_for_this_call)

if __name__ == "__main__":
    asyncio.run(main())
