"""
pKa test using SP energies as stand-in for G (avoids solvator/freq issues).
Tests the full pipeline: tool nodes (load, remove proton, 2x SP) + LLM calc node.
"""
import asyncio
import json
import os
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
)

load_dotenv()

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
}

PLAN = {
    "user_text": "Calculate pKa of HF using SP energies (test)",
    "name": "pKa of HF (SP proxy test)",
    "version": "1.3",
    "geom_ids": ["hf_neutral", "f_anion"],
    "artifacts_to_save": ["E_HA_eh", "E_A_minus_eh", "pka"],
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
            "args": {"mode": "remove", "strategy": "auto"},
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {}
        },
        {
            "id": "sp_hf",
            "kind": "tool",
            "tool": "run_sp_energy",
            "input_id": "hf_neutral",
            "output_id": "hf_neutral",
            "needs": ["load_hf"],
            "args": {"input_geom_id": "hf_neutral", "method": "B3LYP", "basis": "def2-SVP"},
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {"E_HA_eh": "energy"}
        },
        {
            "id": "sp_f_anion",
            "kind": "tool",
            "tool": "run_sp_energy",
            "input_id": "f_anion",
            "output_id": "f_anion",
            "needs": ["remove_proton"],
            "args": {
                "input_geom_id": "f_anion",
                "method": "B3LYP",
                "basis": "def2-SVP"
            },
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {"E_A_minus_eh": "energy"}
        },
        {
            "id": "compute_pka",
            "kind": "llm",
            "task": "compute_pka",
            "needs": ["sp_hf", "sp_f_anion"],
            "prompt": (
                "Compute an approximate pKa of HF from the electronic energies of HF and F-.\n"
                "These are SP energies used as proxies for Gibbs free energies (test only).\n"
                "Formula: deltaG_eh = E_A_minus_eh + G_H_plus_ref_eh - E_HA_eh (all in Hartree).\n"
                "G_H_plus_ref_eh is provided in plan settings.\n"
                "Then convert: deltaG_J_mol = deltaG_eh * Eh_to_J_mol.\n"
                "Then: pKa = deltaG_J_mol / (R * T * ln10).\n"
                "Use constants from plan settings.\n"
                "Return JSON with: status, deltaG_eh, deltaG_J_mol, pka."
            ),
            "needs_artifacts": ["E_HA_eh", "E_A_minus_eh"],
            "product": {"pka": "pka"}
        }
    ],
    "final_report": {"format": "markdown", "fields": ["E_HA_eh", "E_A_minus_eh", "pka"]}
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

            print("\n=== Starting pKa test (SP proxy) for HF ===")
            print("Nodes:", [n["id"] for n in PLAN["nodes"]])
            print()

            result = await graph.ainvoke(init_state)

            print("\n=== Final Status ===")
            print("Status:", result.get("last_status"))

            print("\n=== Artifacts ===")
            artifacts = result.get("artifacts", {})
            for k in ["E_HA_eh", "E_A_minus_eh", "pka"]:
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
