"""
Test SCF_NOT_CONVERGED patch_and_retry on a perturbed test.xyz structure.

Strategy:
- Load test.xyz geometry
- Build a plan with run_opt_job and scf_max_iter=5 (guaranteed to fail)
- on_error: patch scf_max_iter=300 and retry once
- Execute and verify the [retry] log line appears + job eventually succeeds
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── env ──────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
ENV_FILE = os.getenv("ENV_FILE", ".env.lab")
load_dotenv(ENV_FILE, override=True)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from client_helpers import (
    run_tool_node, state_get_tool_args,
    name_to_geometry_xyz, pubchem_get_basic_properties, structure_add_remove_proton,
)
from build_graph_from_plan import build_graph_from_plan, build_state

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz":          name_to_geometry_xyz,
    "pubchem_get_basic_properties":  pubchem_get_basic_properties,
    "structure_add_remove_proton":   structure_add_remove_proton,
}


# ── parse geometry from test.xyz ─────────────────────────────────────────────
def _read_geom(path: str) -> str:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].split()[0])
    return "\n".join(lines[2 : 2 + natoms])


GEOM_ID  = "test_mol"
GEOM_XYZ = _read_geom("test.xyz")

# ── plan ──────────────────────────────────────────────────────────────────────
# scf_max_iter=5 is deliberately too low so that SCF fails on the first attempt.
# The on_error rule patches it to 300 for the retry.
PLAN = {
    "user_text": "Test SCF retry on perturbed structure",
    "name":      "scf_retry_test",
    "version":   "1.3",
    "geom_ids":  [GEOM_ID],
    "artifacts_to_save": ["opt_energy_eh"],
    "settings":  {},
    "nodes": [
        {
            "id":        "opt_test",
            "kind":      "tool",
            "tool":      "run_opt_job",
            "input_id":  GEOM_ID,
            "output_id": GEOM_ID,
            "needs":     [],
            "args": {
                "input_geom_id": GEOM_ID,
                "method": "B3LYP",
                "basis":  "def2-SVP",
                "use_ri": True,
                "scf_max_iter": 5,
            },
            "expect": {"status_in": ["ok", "warning"], "artifacts_required": [], "properties_required": []},
            "product": {"opt_energy_eh": "energy_eh"},
            "on_error": [
                {
                    "if":          {"code_in": ["SCF_NOT_CONVERGED"]},
                    "action":      "patch_and_retry",
                    "patch":       {"scf_max_iter": 300},
                    "max_attempts": 1,
                }
            ],
        }
    ],
    "final_report": {"format": "markdown", "fields": ["opt_energy_eh"]},
}


# ── MCP connection setup ──────────────────────────────────────────────────────
_SSH_BIN  = os.getenv("MCP_SSH_BIN",  "ssh")
_SSH_KEY  = os.getenv("MCP_SSH_KEY",  "C:/Users/zrqrc/.ssh/droplet1")
_SSH_HOST = os.getenv("MCP_SSH_HOST", "root@188.166.232.163")
_CMD      = os.getenv(
    "MCP_SERVER_CMD",
    "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && "
    "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH "
    "python server_with_product.py",
)


async def main():
    server_params = StdioServerParameters(
        command=_SSH_BIN,
        args=["-i", _SSH_KEY, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _SSH_HOST, _CMD],
        env=dict(os.environ),
    )

    print(f"Connecting to MCP server at {_SSH_HOST}...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"MCP tools available: {[t.name for t in tools.tools]}\n")

            # Build graph
            graph = build_graph_from_plan(
                PLAN,
                get_tool_args=state_get_tool_args,
                run_tool_node=run_tool_node,
            )

            # Seed state with pre-loaded geometry
            init_state = build_state(
                PLAN, session,
                seed={
                    "client_side_tools": CLIENT_SIDE_TOOL_FUNCS,
                    "geometries":  {GEOM_ID: GEOM_XYZ},
                    "geom_meta":   {GEOM_ID: {"source": "test.xyz"}},
                    "name_to_geom": {},
                    "default_charge":       0,
                    "default_multiplicity": 1,
                },
                client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
            )

            print("Running plan (expect SCF failure on attempt 1, retry on attempt 2)...\n")
            result = await graph.ainvoke(init_state)

            # ── Report ────────────────────────────────────────────────────────
            run_log    = result.get("run_log", [])
            artifacts  = result.get("artifacts", {})
            last_status = result.get("last_status", "?")

            print("\n=== RUN LOG ===")
            for entry in run_log:
                retries = entry.get("retry_attempts", 0)
                print(f"  node={entry.get('node') or entry.get('node_id')!r:20s}  "
                      f"status={entry.get('status')!r:10s}  "
                      f"retries={retries}")

            print("\n=== ARTIFACTS ===")
            for k, v in artifacts.items():
                print(f"  {k} = {v}")

            print(f"\n=== FINAL STATUS: {last_status} ===\n")

            # ── Pass/Fail check ───────────────────────────────────────────────
            opt_entry = next((e for e in run_log if e.get("node_id") == "opt_test"), None)
            retried   = (opt_entry or {}).get("retry_attempts", 0) >= 1
            succeeded = last_status in ("ok", "warning") or artifacts.get("opt_energy_eh") is not None

            print("RETRY FIRED:  ", "YES [PASS]" if retried   else "NO  [FAIL]  (SCF may not have failed with only 5 iters, or code not detected)")
            print("JOB FINISHED: ", "YES [PASS]" if succeeded else "NO  [FAIL]")

            if retried and succeeded:
                print("\n[PASS] patch_and_retry for SCF_NOT_CONVERGED is WORKING correctly.")
            elif not retried:
                print("\n[WARN] Retry did not fire -- possible reasons:")
                print("   1. ORCA terminated abnormally rather than returning SCF_NOT_CONVERGED code")
                print("   2. Error code was not detected — check run_log details below")
                print("\n--- full run_log ---")
                print(json.dumps(run_log, indent=2))
            else:
                print("\n✗ Retry fired but job did not finish successfully.")


if __name__ == "__main__":
    asyncio.run(main())
