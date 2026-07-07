"""
test_auto_artifact.py — Smoke test for auto-artifact mode in build_graph_from_plan.

Runs a single run_sp_energy node on water with NO 'product' dict.
Verifies that artifacts are auto-populated as "{node_id}.{field}".

Usage:
    python test_auto_artifact.py
    ENV_FILE=.env python test_auto_artifact.py
"""
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv(os.environ.get("ENV_FILE", ".env"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from build_graph_from_plan import build_graph_from_plan, build_state
from client_helpers import (
    run_tool_node,
    state_get_tool_args,
    name_to_geometry_xyz,
    pubchem_get_basic_properties,
    structure_add_remove_proton,
    build_dimer_xyz,
    set_geometry_xyz,
    build_approach_scan_geometries,
)

# Minimal plan: SP on water, no product dict — auto-artifact mode
PLAN = {
    "user_text": "SP energy of water (smoke test)",
    "name":      "smoke_auto_artifact",
    "version":   "1.3",
    "geom_ids":  ["water"],
    "artifacts_to_save": [],
    "nodes": [
        {
            "id":        "geom_water",
            "kind":      "tool",
            "tool":      "name_to_geometry_xyz",
            "needs":     [],
            "args":      {"name": "water"},
            "output_id": "water",
            "expect":    {"status_in": ["ok"]},
        },
        {
            "id":       "sp_water",
            "kind":     "tool",
            "tool":     "run_sp_energy",
            "needs":    ["geom_water"],
            "input_id": "water",
            "args":     {"method": "B3LYP", "basis": "def2-SVP"},
            "expect":   {"status_in": ["ok"]},
            # NO "product" key — auto-artifact mode
        },
    ],
    "final_report": {
        "format": "markdown",
        "fields": ["sp_water.energy_eh", "sp_water.homo_lumo_gap_ev", "sp_water.dipole_moment_debye"],
    },
}


def _build_server_params() -> StdioServerParameters:
    _mode = os.getenv("MCP_MODE", "ssh").strip().lower()
    if _mode == "local":
        cmd = os.getenv("MCP_SERVER_CMD", "python server_with_product.py").split()
        return StdioServerParameters(command=cmd[0], args=cmd[1:], env=dict(os.environ))
    _ssh_bin  = os.getenv("MCP_SSH_BIN",  "ssh")
    _ssh_key  = os.getenv("MCP_SSH_KEY",  "C:/Users/zrqrc/.ssh/droplet1")
    _ssh_host = os.getenv("MCP_SSH_HOST", "root@188.166.232.163")
    _ssh_cmd  = os.getenv(
        "MCP_SERVER_CMD",
        "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && "
        "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH python server_with_product.py",
    )
    return StdioServerParameters(
        command=_ssh_bin,
        args=["-i", _ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", _ssh_host, _ssh_cmd],
        env=dict(os.environ),
    )


async def main():
    server_params = _build_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # state dict for geometry tracking (state_get_tool_args reads from this)
            _geom_state: dict = {
                "geometries":   {},
                "geom_meta":    {},
                "name_to_geom": {},
                "defaults":     {},
            }

            client_side_tools = {
                "name_to_geometry_xyz":           name_to_geometry_xyz,
                "pubchem_get_basic_properties":    pubchem_get_basic_properties,
                "structure_add_remove_proton":     structure_add_remove_proton,
                "build_dimer_xyz":                 build_dimer_xyz,
                "set_geometry_xyz":                set_geometry_xyz,
                "build_approach_scan_geometries":  build_approach_scan_geometries,
                "state_get_tool_args": lambda **kw: state_get_tool_args(_geom_state, **kw),
            }

            graph = build_graph_from_plan(PLAN, run_tool_node=run_tool_node)
            # Pass the MCP ClientSession as 'session' — build_state stores it as state["session"]
            init_state = build_state(PLAN, session, seed={
                "client_side_tools": client_side_tools,
            }, client_side_tools=client_side_tools)

            print("Running plan (water B3LYP/def2-SVP SP)...")
            result = await graph.ainvoke(init_state)

            artifacts = result.get("artifacts") or {}
            print("\n=== Artifacts ===")
            for k, v in sorted(artifacts.items()):
                if not k.startswith("runtime_report") and not k.startswith("bug_report"):
                    print(f"  {k}: {v}")

            expected = [
                "sp_water.energy_eh",
                "sp_water.homo_lumo_gap_ev",
                "sp_water.dipole_moment_debye",
            ]
            missing = [k for k in expected if artifacts.get(k) is None]

            print("\n=== Checks ===")
            for k in expected:
                val = artifacts.get(k)
                status = "PASS" if val is not None else "FAIL"
                print(f"  [{status}] {k} = {val}")

            if missing:
                print(f"\nFAIL — missing auto-artifacts: {missing}")
                sys.exit(1)
            else:
                print("\nPASS — all auto-artifacts present")


if __name__ == "__main__":
    asyncio.run(main())
