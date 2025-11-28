import asyncio
import json
import os
import sys
from typing import Optional, Any, Dict
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env
import re

# Very small keyword dictionaries; expand as needed.
METHOD_KEYWORDS = {
    "b3lyp": "B3LYP",
    "pbe0": "PBE0",
    "pbeh-3c": "PBEh-3c",
    "tpssh": "TPSSh",
    "bp86": "BP86",
    "pw6b95-d3": "PW6B95-D3",
}

BASIS_KEYWORDS = {
    "def2-svp": "def2-SVP",
    "def2-tzvp": "def2-TZVP",
    "def2-tzvpp": "def2-TZVPP",
    "def2-qzvp": "def2-QZVP",
}

JOBTYPE_KEYWORDS = {
    "opt": "opt",          # we tunnel this to job_type="opt"
    "optimize": "opt",
    "optimization": "opt",
    "geometry optimization": "opt",
}


def extract_orca_hints(user_text: str) -> Dict[str, Any]:
    """
    Look for obvious ORCA-related keywords in the user's text:
    - method (B3LYP, PBE0, ...)
    - basis (def2-SVP, ...)
    - job_type (opt vs sp)
    Returns a dict of hints, e.g. {"method": "B3LYP", "basis": "def2-TZVP"}.
    """
    text = user_text.lower()
    hints: Dict[str, Any] = {}

    # Methods
    for key, val in METHOD_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["method"] = val
            break  # prefer first match; you can change this if needed

    # Basis sets
    for key, val in BASIS_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["basis"] = val
            break

    # Job type: if we see "opt" or similar, treat as optimization
    for key, val in JOBTYPE_KEYWORDS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            hints["job_type"] = "opt"
            break

    return hints

@dataclass
class SessionState:
    files: Dict[str, str] = field(default_factory=dict)  # name -> local path
    # named geometries (e.g. "initial", "opt", "ts1", etc.)
    geometries: Dict[str, str] = field(default_factory=dict)
    current_geom_name: Optional[str] = None

    # cache some scalar properties
    last_energy: Optional[float] = None
    last_jobs: list[dict[str, Any]] = field(default_factory=list)

def load_xyz_as_geometry(path: str) -> str:
    """Load an XYZ file and return only the coordinate lines (no natoms/comment)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    raw_lines = [ln.rstrip("\n\r") for ln in raw_lines]
    if not raw_lines:
        raise ValueError(f"XYZ file {path!r} is empty")

    # Try natoms on first token, even if there is a trailing comment
    first_tokens = raw_lines[0].split()
    natoms = None
    if first_tokens:
        try:
            natoms = int(first_tokens[0])
        except ValueError:
            natoms = None

    if natoms is not None and len(raw_lines) >= natoms + 2:
        coord_lines = raw_lines[2 : 2 + natoms]
    else:
        coord_lines = [
            ln
            for ln in raw_lines
            if ln.strip() and not ln.lstrip().startswith(("#", "!", "//"))
        ]

    if not coord_lines:
        raise ValueError(f"No coordinate lines found in {path!r}")

    return "\n".join(coord_lines)


SYSTEM_PROMPT = """
You are an assistant that runs NBO (Natural Bond Orbital) calculations via an MCP tool called run_nbo_job.

- Use the tool to run at most one or two calculations for a given question.
- If the user does not specify charge/multiplicity, you may assume charge = 0 and multiplicity = 1 unless there is a clear reason not to.
- Prefer B3LYP/def2-SVP, TightSCF, RIJCOSX+def2/J for typical ground-state jobs.
- If the tool result shows an error or timeout, do NOT keep retrying; instead, explain the problem.
- After a successful job, summarize the key NBO/NPA results clearly.
- If a tool call fails (non-ok status or error text), do NOT say you will retry unless you actually make another tool call in this same turn. Prefer: explain the failure and show the likely fix.
""".strip()

# JSON schema for the OpenAI tool corresponding to the MCP tool
OPENAI_TOOLS = {
    "run_opt_job": {
        "type": "function",
        "function": {
            "name": "run_opt_job",
            "description": "Optimize geometry with ORCA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_name": {
                        "type": "string",
                        "description": "Name/key of a loaded geometry in session state (preferred).",
                    },
                    "geometry_xyz": {
                        "type": "string",
                        "description": "XYZ coords without natoms/comment header (optional; host can inject).",
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "method": {"type": "string", "default": "B3LYP"},
                    "basis": {"type": "string", "default": "def2-SVP"},
                    "use_ri": {"type": "boolean", "default": True},
                    "scf_max_iter": {"type": "integer", "default": 150},
                    "opt_max_iter": {"type": "integer", "default": 100},
                    "wall_timeout_seconds": {"type": "integer", "default": 1800},
                },
                "required": [],  # <- IMPORTANT: allow omitting geometry_xyz
            },
        },
    },

    "run_nbo_job": {
        "type": "function",
        "function": {
            "name": "run_nbo_job",
            "description": "Run an ORCA NBO/NPA calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_name": {
                        "type": "string",
                        "description": "Name/key of a loaded geometry in session state (preferred).",
                    },
                    "geometry_xyz": {
                        "type": "string",
                        "description": "XYZ coords without natoms/comment header (optional; host can inject).",
                    },
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "method": {"type": "string", "default": "B3LYP"},
                    "basis": {"type": "string", "default": "def2-SVP"},
                    "job_type": {"type": "string", "enum": ["sp", "opt"], "default": "sp"},
                    "use_ri": {"type": "boolean", "default": True},
                    "scf_max_iter": {"type": "integer", "default": 150},
                    "opt_max_iter": {"type": "integer", "default": 50},
                    "wall_timeout_seconds": {"type": "integer", "default": 600},
                },
                "required": [],
            },
        },
    },

    "run_sp_energy": {
        "type": "function",
        "function": {
            "name": "run_sp_energy",
            "description": "Run a single-point energy calculation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "geometry_name": {"type": "string", "description": "Loaded geometry key (preferred)."},
                    "geometry_xyz": {"type": "string", "description": "Optional; host can inject."},
                    "charge": {"type": "integer", "default": 0},
                    "multiplicity": {"type": "integer", "default": 1},
                    "method": {"type": "string", "default": "B3LYP"},
                    "basis": {"type": "string", "default": "def2-SVP"},
                    "use_ri": {"type": "boolean", "default": True},
                    "scf_max_iter": {"type": "integer", "default": 150},
                    "wall_timeout_seconds": {"type": "integer", "default": 600},
                },
                "required": [],
            },
        },
    },
}

def tools_for_server(tool_names: list[str]) -> list[dict]:
    """Return OpenAI tool schemas that are both defined in client and available on MCP server."""
    out = []
    for name in tool_names:
        if name in OPENAI_TOOLS:
            out.append(OPENAI_TOOLS[name])
    return out

NEEDS_GEOM = {"run_opt_job", "run_nbo_job", "run_sp_energy"}

def summarize_geometries(state) -> str:
    keys = list(state.geometries.keys())
    if not keys:
        return "No geometries are loaded."
    cur = state.current_geom
    lines = [f"Loaded geometries: {', '.join(keys)}."]
    if cur:
        lines.append(f"Current geometry: {cur!r}.")
    lines.append("When calling tools, prefer passing geometry_name. You may omit geometry_xyz; the host will inject it.")
    return "\n".join(lines)

def resolve_geometry_xyz(state, args: dict) -> str | None:
    # 1) explicit geometry_name
    gname = args.pop("geometry_name", None)
    if isinstance(gname, str) and gname in state.geometries:
        return state.geometries[gname]

    # 2) already provided geometry_xyz
    gxyz = args.get("geometry_xyz", "")
    if isinstance(gxyz, str) and gxyz.strip():
        return gxyz

    # 3) fallback to current geometry
    if state.current_geom and state.current_geom in state.geometries:
        return state.geometries[state.current_geom]

    return None

async def handle_user_turn(session, client, state, user_text: str, server_tool_names: list[str]):
    tools_for_this_call = [OPENAI_TOOLS[name] for name in server_tool_names if name in OPENAI_TOOLS]

    user_prompt = f"""
{summarize_geometries(state)}

User request:
{user_text}
""".strip()

    first = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools_for_this_call,
        tool_choice="auto",
    )

    msg = first.choices[0].message
    if not msg.tool_calls:
        print("\n[Model response]\n")
        print(msg.content)
        return

    # Execute tool calls (supports multiple)
    tool_messages = []
    for tc in msg.tool_calls:
        tool_name = tc.function.name
        raw_args = tc.function.arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)

        if tool_name in NEEDS_GEOM:
            gxyz = resolve_geometry_xyz(state, args)
            if not gxyz:
                print("\n[Agent note] No geometry available. Please `load <file.xyz>` first.\n")
                return
            args["geometry_xyz"] = gxyz

        # Call MCP
        print("\n[Tool call]", tool_name)
        for k, v in args.items():
            if k == "geometry_xyz":
                print("  geometry_xyz = <injected>")
            else:
                print(f"  {k} = {v!r}")
        mcp_res = await session.call_tool(tool_name, args)

        text = ""
        for item in mcp_res.content:
            if isinstance(item, TextContent):
                text += item.text
            else:
                text += repr(item)
        print("\n[Tool raw output]\n")
        print(text[:4000])  # don’t spam; adjust size if you want
        if len(text) > 4000:
            print("\n...[truncated]...\n")
        tool_messages.append(
            {"role": "tool", "tool_call_id": tc.id, "name": tool_name, "content": text}
        )

        # Optional: if your opt tool returns final xyz in JSON, store it here.
        # (Only if you’re returning JSON from tools.)
        try:
            payload = json.loads(text)
            if tool_name == "run_opt_job" and payload.get("status") == "ok":
                final_xyz = (payload.get("final_geometry_xyz") or "").strip()
                if final_xyz:
                    state.geometries["opt"] = final_xyz
                    state.current_geom = "opt"
        except Exception:
            pass

    second = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            msg,
            *tool_messages,
        ],
    )

    print("\n[Final]\n")
    print(second.choices[0].message.content)

async def main():
    # Optional: initial geometry from CLI argument
    geometry_xyz: str | None = None
    if len(sys.argv) >= 2:
        initial_xyz = sys.argv[1]
        if os.path.exists(initial_xyz):
            geometry_xyz = load_xyz_as_geometry(initial_xyz)
            print(f"Loaded initial geometry from {initial_xyz!r}")
        else:
            print(f"Warning: initial XYZ file {initial_xyz!r} not found.")

    # --- Start MCP server (same config that worked in your simple client) ---
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "orca_nbo_server.py"],
        env={
            "PATH": "/Applications/ORCA:" + os.environ.get("PATH", ""),
        },
    )

    client = OpenAI()
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            state = SessionState()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print("MCP tools available:", tool_names)
            if "run_nbo_job" not in tool_names:
                print("ERROR: run_nbo_job not exposed by MCP server.")
                return

            print(
                "\nInteractive NBO agent ready.\n"
                "Commands:\n"
                "  load <file.xyz>   - load or change the current geometry\n"
                "  quit / exit       - stop\n"
                "  state / status    - show current state"
                "  geoms / geometries- show current geometry files"
                "  show xx           - show file content"
                "Anything else       - natural language question (may trigger NBO run)\n"
            )

            # Interactive loop
                        # Interactive loop
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if not line:
                    continue

                # Explicit quit
                if line.lower() in {"quit", "exit"}:
                    print("Bye.")
                    break

                # Explicit load command still works
                if line.lower().startswith("load "):
                    path = line.split(maxsplit=1)[1]
                    if not os.path.exists(path):
                        print(f"File not found: {path}")
                        continue
                    try:
                        geometry_xyz = load_xyz_as_geometry(path)
                        state.files["test_1.xyz"] = path
                        state.geometries["test_1.xyz"] = geometry_xyz
                        state.current_geom = "test_1.xyz"
                        print(f"Loaded geometry from {path!r}")
                    except Exception as e:
                        print(f"Failed to load {path!r}: {e}")
                    continue
                if line.lower() in {"state", "status"}:
                    print("\n[STATE]")
                    print("  current_geom:", state.current_geom)
                    print("  geometries:", list(state.geometries.keys()))
                    print("  files:", getattr(state, "files", {}))
                    print("  tasks:", [(t.id, t.kind, t.subtype, t.status) for t in getattr(state, "tasks", [])])
                    print()
                    continue

                if line.lower() in {"geoms", "geometries"}:
                    print("\n[GEOMETRIES]")
                    for k in state.geometries.keys():
                        mark = " (current)" if k == state.current_geom else ""
                        print(f"  - {k}{mark}")
                    print()
                    continue
                if line.lower().startswith("show "):
                    name = line.split(maxsplit=1)[1].strip()
                    if name == "current":
                        name = state.current_geom
                    if not name or name not in state.geometries:
                        print(f"Unknown geometry: {name!r}. Available: {list(state.geometries.keys())}")
                        continue
                    print(f"\n[GEOMETRY {name}]\n{state.geometries[name]}\n")
                    continue
                # NEW: auto-detect .xyz path in a normal sentence
                # e.g. "run NBO on ./geom/fe.xyz with charge 2"
                tokens = line.split()
                xyz_path = None
                for tok in tokens:
                    # crude but effective: token ending in .xyz that exists
                    if tok.endswith(".xyz") and os.path.exists(tok):
                        xyz_path = tok
                if xyz_path is not None:
                    try:
                        geometry_xyz = load_xyz_as_geometry(xyz_path)
                        state.files["test_1.xyz"] = path
                        state.geometries["test_1.xyz"] = geometry_xyz
                        state.current_geom = "test_1.xyz"
                        print(f"Loaded geometry from {xyz_path!r}")
                    except Exception as e:
                        print(f"Failed to load {xyz_path!r}: {e}")
                        # keep going, but don't try to run NBO
                        continue

                    # Remove the path token from the user text before sending to the LLM
                    tokens = [t for t in tokens if t != xyz_path]
                    line = " ".join(tokens).strip()
                    if not line:
                        # user only gave a path, no question → just update geometry and wait
                        continue

                # Normal question turn (may trigger a tool call)
                await handle_user_turn(session, client, state, line, tool_names)



if __name__ == "__main__":
    asyncio.run(main())
