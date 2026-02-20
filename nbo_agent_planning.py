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
    result_dict_to_prompt,
    extract_compound_names_llm,
    fetch_compound_card,
    fetch_compound_card_from_xyz,
    display_and_confirm_compound,
    compounds_to_planner_context,
    render_xyz_image_rdkit,
)
from build_graph_from_plan import build_graph_from_plan, build_state
from prompts import SYSTEM_PROMPT, CALCULATOR_SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT



load_dotenv(os.environ.get("ENV_FILE", ".env"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

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



async def handle_user_turn(session, client, state, user_text: str, tools_for_this_call: list,
                           compound_context: str = ""):
    # Merge: MCP tools + client-side tools, but only if schema exists in OPENAI_TOOLS

    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": "STATE:\n" + summarize_geometries_prompt(state)},
    {"role": "system", "content": "WORKFLOW_STATE:\n" + summarize_workflow_state(state)},
    ]
    if compound_context:
        messages.append({"role": "system", "content": "CONFIRMED_COMPOUNDS:\n" + compound_context})
    messages.append({"role": "user", "content": user_text})
    state["last_user_text"] = user_text

    MAX_TOOL_ROUNDS = 4
    for _round in range(MAX_TOOL_ROUNDS):
        #print("TOOL NAMES FOR MODEL:", tool_names_for_model)
        #print("TOOLS SENT:", [t["function"]["name"] for t in tools_for_this_call])
        resp = client.chat.completions.create(
            model=LLM_MODEL,
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


def _cache_confirmed_card(state: AgentState, name: str, card: dict) -> None:
    """Store a confirmed compound card in state caches."""
    state.setdefault("cached_props", {})[name] = card
    if card.get("inchi") or card.get("smiles"):
        state.setdefault("identifiers", {})[name] = {
            "smiles": card.get("smiles"),
            "inchi": card.get("inchi"),
            "inchikey": card.get("inchikey"),
            "formula": card.get("formula"),
            "charge": card.get("charge", 0),
        }


async def identify_and_confirm_compounds(
    user_text: str, client, state: AgentState
) -> list:
    """Pre-planning phase: identify all compounds (by name and/or loaded geometries).

    Two sources:
      1. Compound names extracted from user_text via LLM → PubChem/OPSIN lookup.
      2. Geometries already in state that have no identity cached yet → XYZ-derived
         SMILES + PubChem lookup, displayed for user confirmation.

    Returns list of confirmed compound cards cached in state['cached_props'] /
    state['identifiers'].
    """
    print("\n[Identifying compounds...]")
    confirmed = []

    # --- Source 1: names mentioned in the planning request ---
    names = extract_compound_names_llm(user_text, client)
    for name in names:
        card = fetch_compound_card(name)
        result = display_and_confirm_compound(card, client)
        if result is None:
            continue
        confirmed.append(result)
        _cache_confirmed_card(state, name, result)

    # --- Source 2: geometries loaded in state with no cached identity ---
    geometries = state.get("geometries") or {}
    identifiers = state.get("identifiers") or {}
    cached_props = state.get("cached_props") or {}
    geom_meta = state.get("geom_meta") or {}

    unidentified = [
        gid for gid in geometries
        if gid not in identifiers and gid not in cached_props
    ]

    if unidentified:
        print(f"\n[{len(unidentified)} loaded geometry/geometries not yet identified]")

    for geom_id in unidentified:
        xyz = geometries[geom_id]
        charge = (geom_meta.get(geom_id) or {}).get("charge", 0) or 0

        print(f"\n── Loaded geometry: {geom_id} ──")
        # Derive card from XYZ (bond detection + PubChem by SMILES)
        card = fetch_compound_card_from_xyz(xyz, geom_id, charge=charge)
        if card.get("smiles"):
            print(f"  Detected SMILES : {card['smiles']}")
        if card.get("formula"):
            print(f"  Detected formula: {card['formula']}")

        result = display_and_confirm_compound(card, client)
        if result is None:
            continue
        confirmed.append(result)
        _cache_confirmed_card(state, geom_id, result)

    if not confirmed:
        print("[No compounds identified — proceeding to planning with geometry state only]")

    return confirmed


async def main():
    _mcp_ssh_key = os.getenv("MCP_SSH_KEY", "C:/Users/zrqrc/.ssh/droplet1")
    _mcp_ssh_host = os.getenv("MCP_SSH_HOST", "root@188.166.232.163")
    _mcp_server_cmd = os.getenv(
        "MCP_SERVER_CMD",
        "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && "
        "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH python server_with_product.py",
    )
    server_params = StdioServerParameters(
        command="ssh",
        args=["-i", _mcp_ssh_key, _mcp_ssh_host, _mcp_server_cmd],
    )

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)
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
                    charge = (state.get("geom_meta") or {}).get(key, {}).get("charge", 0) or 0
                    img_path = render_xyz_image_rdkit(geometry_xyz, key, charge=charge)
                    if img_path:
                        print(f"Structure: {img_path} [opened]")
                    else:
                        print("Structure: (could not render — check RDKit / XYZ format)")
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

                if line.strip().lower().startswith("planload"):
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: planload <path-to-plan.json>")
                        continue
                    plan_path = Path(parts[1].strip())
                    if not plan_path.exists():
                        print(f"File not found: {plan_path}")
                        continue
                    try:
                        plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"Failed to load plan: {e}")
                        continue
                    review = build_plan_review(plan)
                    plan_pkg = {"plan": plan, "review": review, "ui": {"saved_files": {}}}
                    store_plan(state, plan_pkg)
                    print(f"Loaded plan from {plan_path}")
                    print_plan_review(plan_pkg)
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
                             run_tool_node=run_tool_node, openai_client=client)

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
                        model=LLM_MODEL,
                        messages=report_messages,
                    )
                    msg = resp.choices[0].message
                    print(msg.content)
                    #print(json.dumps(result, ensure_ascii=False, indent=2))
                    continue
                
                compounds = await identify_and_confirm_compounds(line, client, state)
                compound_context = compounds_to_planner_context(compounds)
                await handle_user_turn(session, client, state, line, tools_for_this_call,
                                       compound_context=compound_context)

if __name__ == "__main__":
    asyncio.run(main())
