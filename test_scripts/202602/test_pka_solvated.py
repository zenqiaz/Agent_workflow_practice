"""
pKa test with explicit solvation: solvator clusters + freq for Gibbs free energies.
Pipeline: load HF -> remove proton -> solvate(HF) -> solvate(F-) -> freq(cluster_HF) -> freq(cluster_F-) -> LLM pKa
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
    "user_text": "Calculate pKa of HF in water using explicit microsolvation",
    "name": "pKa of HF (solvated freq)",
    "version": "1.3",
    "geom_ids": ["hf_neutral", "f_anion", "hf_cluster", "f_cluster"],
    "artifacts_to_save": ["G_HA_eh", "G_A_minus_eh", "pka"],
    "settings": {
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
            "expect": {"status_in": ["ok"]},
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
            "expect": {"status_in": ["ok"]},
            "product": {}
        },
        {
            "id": "solvate_hf",
            "kind": "tool",
            "tool": "run_solvator_cluster",
            "input_id": "hf_neutral",
            "output_id": "hf_cluster",
            "needs": ["load_hf"],
            "args": {"input_geom_id": "hf_neutral", "nsolv": 3},
            "expect": {"status_in": ["ok"]},
            "product": {}
        },
        {
            "id": "solvate_f",
            "kind": "tool",
            "tool": "run_solvator_cluster",
            "input_id": "f_anion",
            "output_id": "f_cluster",
            "needs": ["remove_proton"],
            "args": {"input_geom_id": "f_anion", "nsolv": 3},
            "expect": {"status_in": ["ok"]},
            "product": {}
        },
        {
            "id": "freq_hf_cluster",
            "kind": "tool",
            "tool": "run_freq_job",
            "input_id": "hf_cluster",
            "output_id": "hf_cluster",
            "needs": ["solvate_hf"],
            "args": {"input_geom_id": "hf_cluster", "method": "B3LYP", "basis": "def2-SVP"},
            "expect": {"status_in": ["ok"]},
            "product": {"G_HA_eh": "gibbs_free_energy_eh"}
        },
        {
            "id": "freq_f_cluster",
            "kind": "tool",
            "tool": "run_freq_job",
            "input_id": "f_cluster",
            "output_id": "f_cluster",
            "needs": ["solvate_f"],
            "args": {"input_geom_id": "f_cluster", "method": "B3LYP", "basis": "def2-SVP"},
            "expect": {"status_in": ["ok"]},
            "product": {"G_A_minus_eh": "gibbs_free_energy_eh"}
        },
        {
            "id": "compute_pka",
            "kind": "llm",
            "task": "compute_pka",
            "needs": ["freq_hf_cluster", "freq_f_cluster"],
            "prompt": (
                "Compute pKa of HF in water from Gibbs free energies of solvated clusters.\n"
                "Formula: deltaG_eh = G_A_minus_eh + G_H_plus_ref_eh - G_HA_eh (all in Hartree).\n"
                "G_H_plus_ref_eh is provided in plan settings.\n"
                "Then convert: deltaG_J_mol = deltaG_eh * Eh_to_J_mol.\n"
                "Then: pKa = deltaG_J_mol / (R * T * ln10).\n"
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

            print("\n=== Starting pKa test (solvated freq) for HF ===")
            print("Nodes:", [n["id"] for n in PLAN["nodes"]])
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

            print("\n=== Node Results ===")
            for nid in ["solvate_hf", "solvate_f", "freq_hf_cluster", "freq_f_cluster", "compute_pka"]:
                nr = (result.get("node_results") or {}).get(nid)
                if nr:
                    # Print compact summary
                    if isinstance(nr, dict):
                        status = nr.get("status", "?")
                        err = nr.get("error", "")
                        print(f"\n  [{nid}] status={status}")
                        if err:
                            print(f"    error: {err}")
                        for k in ["gibbs_free_energy_eh", "pka", "deltaG_eh", "deltaG_J_mol",
                                   "cluster_geometry_xyz", "values"]:
                            if k in nr:
                                v = nr[k]
                                if k == "cluster_geometry_xyz" and isinstance(v, str):
                                    natoms = len([l for l in v.splitlines() if l.strip()])
                                    print(f"    {k}: ({natoms} atoms)")
                                else:
                                    print(f"    {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
