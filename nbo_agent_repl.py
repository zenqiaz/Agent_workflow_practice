# nbo_agent_repl.py
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict
import traceback

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI

from client_helpers import (
    NEEDS_GEOM_SINGLE,
    TOOLS_RETURNING_STRUCTURE,
    load_xyz_as_geometry,
    summarize_geometries_prompt,
    #resolve_geometry_xyz,
    geom_key_from_path,
    name_to_geometry_xyz,
    print_tool_output,
    get_trivial_properties,
    pubchem_get_basic_properties,
    ALLOWED_STATE_KEYS,
    state_update,
    state_get_tool_args,
    get_geometry_xyz_by_name,
    _maybe_store_geometry_from_payload,
    xyz_to_no_header,
    _store_new_geometry,
    print_session_state,
    coerce_geometry_xyz,
)

load_dotenv()

SYSTEM_PROMPT = """
You are an assistant that runs NBO (Natural Bond Orbital) calculations via an MCP tool run_nbo_job, or solvation frequency calculation with run_solvator_cluster_thermo.

- If the user tell you not to call any tools, then use reasoning only.
- Otherwise, use the tool to run some calculations for a given question.
- If the user does not specify charge/multiplicity, you may assume charge = 0 and multiplicity = 1 unless there is a clear reason not to.
- If no structure is explicitly provided by the user, the agent must obtain it via tools (PCP, file lookup, etc.) or ask the user for it.
- Prefer B3LYP/def2-SVP, TightSCF, RIJCOSX+def2/J for typical ground-state jobs.
- If the tool result shows an error or timeout, do NOT keep retrying; instead, explain the problem.
- After a successful job, summarize the key NBO/NPA results clearly.
- If a tool call fails (non-ok status or error text), do NOT say you will retry unless you actually make another tool call in this same turn. Prefer: explain the failure and show the likely fix.
- If a tool requires a geometry, retrieve the value with corresponding name as the key, or use the content of current_geom.  

STATE & DEFAULTS
- The session has a state containing:
  - current_geom (key of the active geometry)
  - geometries[current_geom] = XYZ coordinates (no natoms/comment header)
  - default_charge (int) and default_multiplicity (int)
  - identifiers[name] (cid/smiles/inchi/inchikey) and cached_props (formula/mw/etc.)
TOOL SEQUENCING POLICY
- Avoid redundant state_update calls. Do not call state_update unless a user explicitly asks to change defaults, or a tool output requires updating defaults (charge/multiplicity).
- When a tool returns new_charge/new_multiplicity, update the in-memory SessionState directly (no tool call needed).
- For multi-step workflows (load → edit structure → compute):
   1,run the structure/lookup tool (e.g., name_to_geometry_xyz)
   2,run the structure edit tool (structure_add_remove_proton)
   3,run the compute tool (run_solvator_cluster_thermo)
- Never call state_update twice with the same values in a single turn.

ARGUMENT RULES
- Do NOT guess charge/multiplicity if state already has defaults.
- If the user explicitly the name of structure, try to retrieve the structure from Sessionstate before use current_geom or call client tools.
- If the user explicitly provides charge or multiplicity, use these parameters directly and bypass the default ones.
- If the user does not provide them, use state defaults (default_charge/default_multiplicity).
- Only call ORCA tools (run_opt_job/run_nbo_job/run_sp_energy) after you have a valid geometry in state
  (current_geom exists and geometries[current_geom] is available).

GEOMETRY RULES
- If the user names a molecule but no geometry is loaded, first call name_to_geometry_xyz(name=...),
  which should load geometry into state (or return not_found). If you do so, do not call state_update then.
- If an optimization produces final_geometry_xyz, treat it as the new active geometry (current_geom="opt")
  and preserve metadata linkages in geom_meta.

LIMITS / EARLY EXIT
- Avoid endless iterations: at most one ORCA job per round; stop and ask the user if repeated failures happen.

""".strip()

OPENAI_TOOLS = json.loads(Path("openai_tools.json").read_text(encoding="utf-8"))
CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    
    "state_update": lambda **kw: state_update(state, **kw),          # not so useful: applicable only when data readily to be fill into the state
    "state_get_tool_args": lambda **kw: state_get_tool_args(state, **kw),
    #"pubchem_get_record_fields": pubchem_get_record_fields,
}
#CLIENT_SIDE_TOOL_FUNCS = {
#}

@dataclass
class SessionState:
    files: Dict[str, str] = field(default_factory=dict)        # key -> path
    geometries: Dict[str, str] = field(default_factory=dict)   # key -> xyz(no header)
    current_geom: Optional[str] = None
    name_to_geom: Dict[str, str] = field(default_factory=dict)
    identifiers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # name -> {smiles,inchi,inchikey,cid,...}
    geom_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)    # geom_key -> {name,cid,smiles,...}

    cached_props: dict[str, dict] = field(default_factory=dict)  # key (name or cid) -> {formula,mw,dipole,homo,lumo,...}

    last_tool_name: Optional[str] = None
    last_tool_raw: Optional[str] = None
    default_charge: int = 0
    default_multiplicity: int = 1   # PubChem won't give multiplicity; keep 1 unless user says otherwise

def tools_for_server(tool_names: list[str]) -> list[dict]:
    return [OPENAI_TOOLS[name] for name in tool_names if name in OPENAI_TOOLS]

async def handle_user_turn(session, client, state, user_text: str, server_tool_names: list[str]):
    # Merge: MCP tools + client-side tools, but only if schema exists in OPENAI_TOOLS
    tool_names_for_model = [n for n in server_tool_names if n in OPENAI_TOOLS]
    for n in CLIENT_SIDE_TOOL_FUNCS:
        if n in OPENAI_TOOLS and n not in tool_names_for_model:
            tool_names_for_model.append(n)

    tools_for_this_call = [OPENAI_TOOLS[n] for n in tool_names_for_model]

    user_prompt = f"""
{summarize_geometries_prompt(state)}

User request:
{user_text}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    MAX_TOOL_ROUNDS = 4
    for _round in range(MAX_TOOL_ROUNDS):
        #print("TOOL NAMES FOR MODEL:", tool_names_for_model)
        #print("TOOLS SENT:", [t["function"]["name"] for t in tools_for_this_call])
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_for_this_call,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # Done: model answered without tools
        if not msg.tool_calls:
            print("\n[Final]\n")
            print(msg.content)
            return

        tool_messages = []
        for tc in msg.tool_calls:
            print("tool:", tc.function.name)
            print("raw arguments type:", type(tc.function.arguments))
            print("raw arguments:", tc.function.arguments)
            tool_name = tc.function.name
            raw_args = tc.function.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)

            # ---- client-side tool ----
            
            if tool_name in CLIENT_SIDE_TOOL_FUNCS:
                print("\n[Client tool call]", tool_name, args)
                try:
                    if tool_name == "get_trivial_properties":#not trivial, using pymatgen to find the symmetry with the structure
                        out = get_trivial_properties(state, name=args.get("name"))
                    else:
                        out = CLIENT_SIDE_TOOL_FUNCS[tool_name](**args)
                except Exception as e:
                    out = {"status": "error", "tool": tool_name, "error": str(e)}
                text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
                
                # If it produced geometry, store to state and set current
                try:
                    payload = json.loads(text)

                    # existing behavior
                    if tool_name == "run_opt_job" and payload.get("status") == "ok":
                        final_xyz = (payload.get("final_geometry_xyz") or "").strip()
                        if final_xyz:
                            prev = state.current_geom
                            state.geometries["opt"] = final_xyz
                            if prev and prev in state.geom_meta:
                                state.geom_meta["opt"] = dict(state.geom_meta[prev])
                            state.current_geom = "opt"

                    # NEW: generic “structure-like” geometry outputs from MCP tools
                    if payload.get("status") == "ok":
                        geom_out = (payload.get("xyz") or payload.get("geometry_xyz") or "").strip()
                        if geom_out:
                            prev = state.current_geom
                            key_hint = payload.get("name") or payload.get("label") or tool_name
                            new_key = _store_new_geometry(
                                state,
                                geom_out,
                                key_hint=key_hint,
                                carry_meta_from=prev,
                                charge=payload.get("new_charge"),
                                multiplicity=payload.get("new_multiplicity"),
                            )
                            print(f"[STATE] Stored MCP geometry as {new_key!r} and set current.")
                except Exception:
                    pass

                tool_messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": text}
                )
                continue

            # ---- MCP tool ----
            if tool_name in NEEDS_GEOM_SINGLE:
                '''gxyz = resolve_geometry_xyz(state, args)
                if not gxyz:
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "name": tool_name,
                         "content": json.dumps({"status":"error","error":"No geometry loaded. Use `load <file.xyz>` or resolve a name first."})}
                    )
                    continue
                args["geometry_xyz"] = gxyz'''
                args["geometry_xyz"] = coerce_geometry_xyz(state, args)

            print("\n[MCP tool call]", tool_name)
            mcp_res = await session.call_tool(tool_name, args)

            text = ""
            for item in mcp_res.content:
                text += item.text if isinstance(item, TextContent) else repr(item)
            print_tool_output(text, limit=20000)

            tool_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": text}
            )

            # Keep your existing opt -> state update
            try:
                payload = json.loads(text)
                if tool_name in TOOLS_RETURNING_STRUCTURE and payload.get("status") == "ok":
                    geom_out = (payload.get("geometry_xyz") or payload.get("final_geometry_xyz") or payload.get("xyz") or "").strip()
                    print("geom_out:", geom_out)
                    if geom_out:
                        prev = state.current_geom
                        key_hint = payload.get("name") or payload.get("label") or tool_name
                        new_key = _store_new_geometry(state, geom_out, key_hint=key_hint, carry_meta_from=prev)
                        if new_key:
                            # update defaults if tool provides them
                            if payload.get("new_charge") is not None:
                                state.default_charge = int(payload["new_charge"])
                            if payload.get("new_multiplicity") is not None:
                                state.default_multiplicity = int(payload["new_multiplicity"])
                            print(f"[STATE] Stored MCP geometry as {new_key!r} and set current.")
            except Exception as e:
                print("[STATE] postprocess failed:", repr(e))
                traceback.print_exc()

        messages.extend(tool_messages)

    print("\n[Final]\nEarly exit: too many tool rounds. Please rephrase or provide missing info.")


async def main():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "orca_nbo_server_opi_with_solvator_thermo_debug.py"],
        env={"PATH": "/Users/zhangruiqi/Library/orca_6_1_1:" + os.environ.get("PATH", "")},
    )

    client = OpenAI()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            state = SessionState()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print("MCP tools available:", tool_names)

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
                    state.files[key] = path
                    state.geometries[key] = geometry_xyz
                    state.current_geom = key
                    print(f"Loaded geometry from {path!r} as {key!r}")
                    continue

                if line.strip().lower() == "state":
                    print_session_state(state, preview_lines=6, show_full_current=True, max_full_lines=None)
                    continue

                await handle_user_turn(session, client, state, line, tool_names)

if __name__ == "__main__":
    asyncio.run(main())
