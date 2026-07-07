"""
Test: run_casscf_job on [Fe(CN)6]^{3-} high-spin (S=5/2, mult=6)
Active space: CAS(5,5) — 5 d-electrons in 5 d-orbitals of Fe(III)

Uses the same geometry from test_fe_cn6.py.
"""

import asyncio, json, os, time
from dotenv import load_dotenv
load_dotenv(os.getenv("ENV_FILE", ".env.lab"), override=True)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from client_helpers import (
    build_coordination_complex, run_tool_node, state_get_tool_args,
    name_to_geometry_xyz, pubchem_get_basic_properties, structure_add_remove_proton,
)
from build_graph_from_plan import build_graph_from_plan, build_state

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz":         name_to_geometry_xyz,
    "build_coordination_complex":   build_coordination_complex,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    "structure_add_remove_proton":  structure_add_remove_proton,
}

METAL   = "Fe"
LIGANDS = ["cyanide"] * 6
CHARGE  = -3

print("Building [Fe(CN)6]3- geometry (high spin, mult=6)...")
hs = build_coordination_complex(METAL, LIGANDS, "octahedral", charge=CHARGE, multiplicity=6, bond_length=1.92)
if hs["status"] != "ok":
    raise RuntimeError(f"Geometry build failed: {hs['error']}")
print(f"  {hs['n_atoms']} atoms built")

HS_XYZ = hs["geometry_xyz"]

PLAN = {
    "user_text": "CASSCF SP on [Fe(CN)6]3- high spin CAS(5,5)",
    "name":      "fe_cn6_casscf",
    "version":   "1.3",
    "geom_ids":  ["fe_cn6_hs"],
    "artifacts_to_save": ["e_casscf_eh"],
    "settings":  {},
    "nodes": [
        {
            "id":       "casscf_hs",
            "kind":     "tool",
            "tool":     "run_casscf_job",
            "input_id": "fe_cn6_hs",
            "needs":    [],
            "args": {
                "input_geom_id":        "fe_cn6_hs",
                "nel":                  5,
                "norb":                 5,
                "nroots":               1,
                "basis":                "def2-SVP",
                "scf_max_iter":         300,
                "wall_timeout_seconds": 1800,
            },
            "product": {"e_casscf_eh": "energy_eh"},
            "on_error": [
                {"if": {"code_in": ["SCF_NOT_CONVERGED"]},
                 "action": "patch_and_retry", "patch": {"scf_max_iter": 500}, "max_attempts": 1}
            ],
        },
    ],
    "final_report": {"format": "markdown", "fields": ["e_casscf_eh"]},
}

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

    print("\nConnecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected.\n")

            graph = build_graph_from_plan(
                PLAN,
                get_tool_args=state_get_tool_args,
                run_tool_node=run_tool_node,
            )
            init_state = build_state(
                PLAN, session,
                seed={
                    "client_side_tools":    CLIENT_SIDE_TOOL_FUNCS,
                    "geometries":           {"fe_cn6_hs": HS_XYZ},
                    "geom_meta":            {
                        "fe_cn6_hs": {"label": "Fe(CN)6 3- high spin", "charge": CHARGE, "multiplicity": 6},
                    },
                    "name_to_geom":         {},
                    "default_charge":       CHARGE,
                    "default_multiplicity": 6,
                },
                client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
            )

            print("Running CASSCF(5,5) SP on [Fe(CN)6]3- high spin ...")
            t0     = time.monotonic()
            result = await graph.ainvoke(init_state)
            dt     = time.monotonic() - t0

            arts  = result.get("artifacts", {})
            e     = arts.get("e_casscf_eh")

            print("\n" + "="*55)
            print("RESULTS  [Fe(CN)6]^3-  CASSCF(5,5)/def2-SVP  HS")
            print("="*55)
            if e is not None:
                print(f"  E(CASSCF, HS, mult=6) = {e:>14.6f} Eh")
            else:
                print("  E(CASSCF) = N/A")
            print(f"\n  Total wall time: {dt:.0f}s")

            print("\nRun log:")
            for entry in result.get("run_log", []):
                node = entry.get("node") or entry.get("node_id", "?")
                print(f"  {node:15s}  status={entry.get('status'):9s}  "
                      f"retries={entry.get('retry_attempts', 0)}  "
                      f"t={entry.get('duration_ms', 0)//1000}s")

if __name__ == "__main__":
    asyncio.run(main())
