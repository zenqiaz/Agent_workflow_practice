"""
Full pKa test: MCP tool nodes (load geom, remove proton, solvator+freq) + LLM calc node.
Bypasses the planner to use a known-good plan JSON.
"""
import asyncio
import json
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from build_graph_from_plan import build_graph_from_plan, build_state
from client_helpers import (
    run_tool_node,
    state_get_tool_args,
    name_to_geometry_xyz,
    pubchem_get_basic_properties,
    state_update,
    state_get_tool_args as _sga,
)

load_dotenv()

# Replicate the client-side tools registry from nbo_agent_planning.py
# (state is not used here since build_state provides one)
CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
}

PLAN = {
    "user_text": "Calculate the pKa of hydrogen fluoride",
    "name": "pKa of HF via microsolvation thermochemistry",
    "version": "1.3",
    "geom_ids": ["hf_neutral", "f_anion"],
    "artifacts_to_save": ["G_HA_eh", "G_A_minus_eh", "pka"],
    "settings": {
        "gas_constant_R_J_molK": 8.314462618,
        "temperature_K": 298.15,
        "ln10": 2.302585092994046,
        "Eh_to_J_mol": 2625499.638,
        "G_H_plus_ref_eh": -0.01372
    },
    "nodes": [
        {
            "id": "load_hf",
            "kind": "tool",
            "tool": "name_to_geometry_xyz",
            "output_id": "hf_neutral",
            "needs": [],
            "args": {"name": "hydrogen fluoride"},
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {}
        },
        {
            "id": "remove_proton",
            "kind": "tool",
            "tool": "structure_add_remove_proton",
            "input_id": "hf_neutral",
            "output_id": "f_anion",
            "needs": ["load_hf"],
            "args": {
                "mode": "remove",
                "strategy": "auto"
            },
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {}
        },
        {
            "id": "thermo_hf",
            "kind": "tool",
            "tool": "run_solvator_cluster_thermo",
            "input_id": "hf_neutral",
            "output_id": "hf_neutral",
            "needs": ["load_hf"],
            "args": {
                "input_geom_id": "hf_neutral",
                "method": "r2scan-3c",
                "basis": "",
                "nsolv": 3,
                "ncores": 2,
                "thermo_timeout_seconds": 3600
            },
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {"G_HA_eh": "G_eh"}
        },
        {
            "id": "thermo_f_anion",
            "kind": "tool",
            "tool": "run_solvator_cluster_thermo",
            "input_id": "f_anion",
            "output_id": "f_anion",
            "needs": ["remove_proton"],
            "args": {
                "input_geom_id": "f_anion",
                "method": "r2scan-3c",
                "basis": "",
                "nsolv": 3,
                "charge": -1,
                "multiplicity": 1,
                "ncores": 2,
                "thermo_timeout_seconds": 3600
            },
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {"G_A_minus_eh": "G_eh"}
        },
        {
            "id": "compute_pka",
            "kind": "llm",
            "task": "compute_pka",
            "needs": ["thermo_hf", "thermo_f_anion"],
            "prompt": (
                "Compute the pKa of HF from the Gibbs free energies.\n"
                "Formula: deltaG_eh = G_A_minus_eh + G_H_plus_ref_eh - G_HA_eh (all in Hartree).\n"
                "G_H_plus_ref_eh is provided in plan settings.\n"
                "Then convert: deltaG_J_mol = deltaG_eh * Eh_to_J_mol.\n"
                "Then: pKa = deltaG_J_mol / (R * T * ln10).\n"
                "Use constants from plan settings.\n"
                "Return JSON with: status, deltaG_eh, deltaG_J_mol, pka."
            ),
            "needs_artifacts": ["G_HA_eh", "G_A_minus_eh"],
            "product": {"pka": "pka"}
        }
    ],
    "final_report": {"format": "markdown", "fields": ["G_HA_eh", "G_A_minus_eh", "pka"]}
}


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

            tools = await session.list_tools()
            print("MCP tools:", [t.name for t in tools.tools])

            graph = build_graph_from_plan(
                PLAN,
                get_tool_args=state_get_tool_args,
                run_tool_node=run_tool_node,
                openai_client=client,
            )

            init_state = build_state(PLAN, session, seed={
                "client_side_tools": CLIENT_SIDE_TOOL_FUNCS,
            }, client_side_tools=CLIENT_SIDE_TOOL_FUNCS)

            print("\n=== Starting pKa calculation for HF ===")
            print("Plan nodes:", [n["id"] for n in PLAN["nodes"]])
            print()

            result = await graph.ainvoke(init_state)

            print("\n=== Final Status ===")
            print("Status:", result.get("last_status"))

            print("\n=== Artifacts ===")
            artifacts = result.get("artifacts", {})
            for k in ["G_HA_eh", "G_A_minus_eh", "pka"]:
                print(f"  {k}: {artifacts.get(k)}")

            print("\n=== Run Log ===")
            for entry in (result.get("run_log") or []):
                node = entry.get("node", "?")
                kind = entry.get("kind", "?")
                status = entry.get("status", "?")
                dur = entry.get("duration_ms", "?")
                print(f"  {node:25s}  {kind:6s}  {status:8s}  {dur}ms")

            print("\n=== LLM Node Result ===")
            pprint((result.get("node_results") or {}).get("compute_pka"))


if __name__ == "__main__":
    asyncio.run(main())
