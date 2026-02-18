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
)
from build_graph_from_plan import build_graph_from_plan, build_state
from prompts import SYSTEM_PROMPT, CALCULATOR_SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT



load_dotenv()



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
        command="ssh",
        args=[
            "-i", "C:/Users/zrqrc/.ssh/droplet1",
            "root@188.166.232.163",
            "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH python server_with_product.py",
        ],
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
