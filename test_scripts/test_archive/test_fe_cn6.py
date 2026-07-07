"""
Test: [Fe(CN)6]^{3-} high spin (S=5/2, mult=6) vs low spin (S=1/2, mult=2)

Workflow:
  1. build_coordination_complex → Fe(CN)6 high-spin geometry
  2. build_coordination_complex → Fe(CN)6 low-spin geometry
  3. run_sp_energy on both (B3LYP/def2-SVP)
  4. Report ΔE = E(HS) − E(LS)

Fe(III) is d5. CN⁻ is a strong-field ligand → low spin expected to be lower in energy.
Complex charge: Fe³⁺ + 6 CN⁻ = 3+ + 6×(1-) = −3.
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

# ── build both geometries locally ─────────────────────────────────────────────
METAL   = "Fe"
LIGANDS = ["cyanide"] * 6   # CN⁻ × 6
CHARGE  = -3                # Fe³⁺ + 6 CN⁻

print("Building geometries locally...")
hs = build_coordination_complex(METAL, LIGANDS, "octahedral", charge=CHARGE, multiplicity=6, bond_length=1.92)
ls = build_coordination_complex(METAL, LIGANDS, "octahedral", charge=CHARGE, multiplicity=2, bond_length=1.92)

for label, r in [("High spin (mult=6)", hs), ("Low spin  (mult=2)", ls)]:
    if r["status"] != "ok":
        raise RuntimeError(f"{label}: {r['error']}")
    print(f"  {label}: {r['n_atoms']} atoms")
    # Show Fe + first CN donor atoms
    for line in r["geometry_xyz"].splitlines()[:3]:
        print(f"    {line}")
    print("    ...")

HS_XYZ  = hs["geometry_xyz"]
LS_XYZ  = ls["geometry_xyz"]

# ── plan: two SP nodes ────────────────────────────────────────────────────────
PLAN = {
    "user_text": "SP energy of [Fe(CN)6]3- high spin and low spin",
    "name":      "fe_cn6_spin_states",
    "version":   "1.3",
    "geom_ids":  ["fe_cn6_hs", "fe_cn6_ls"],
    "artifacts_to_save": ["e_hs_eh", "e_ls_eh"],
    "settings":  {},
    "nodes": [
        {
            "id":       "sp_hs",
            "kind":     "tool",
            "tool":     "run_sp_energy",
            "input_id": "fe_cn6_hs",
            "needs":    [],
            "args": {
                "input_geom_id":        "fe_cn6_hs",
                "method":               "B3LYP",
                "basis":                "def2-SVP",
                "use_ri":               True,
                "scf_max_iter":         300,
                "wall_timeout_seconds": 1800,
            },
            "product": {"e_hs_eh": "energy_eh"},
            "on_error": [
                {"if": {"code_in": ["SCF_NOT_CONVERGED"]},
                 "action": "patch_and_retry", "patch": {"scf_max_iter": 500}, "max_attempts": 1}
            ],
        },
        {
            "id":       "sp_ls",
            "kind":     "tool",
            "tool":     "run_sp_energy",
            "input_id": "fe_cn6_ls",
            "needs":    [],
            "args": {
                "input_geom_id":        "fe_cn6_ls",
                "method":               "B3LYP",
                "basis":                "def2-SVP",
                "use_ri":               True,
                "scf_max_iter":         300,
                "wall_timeout_seconds": 1800,
            },
            "product": {"e_ls_eh": "energy_eh"},
            "on_error": [
                {"if": {"code_in": ["SCF_NOT_CONVERGED"]},
                 "action": "patch_and_retry", "patch": {"scf_max_iter": 500}, "max_attempts": 1}
            ],
        },
    ],
    "final_report": {"format": "markdown", "fields": ["e_hs_eh", "e_ls_eh"]},
}

# ── MCP ───────────────────────────────────────────────────────────────────────
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
                    "geometries":           {"fe_cn6_hs": HS_XYZ, "fe_cn6_ls": LS_XYZ},
                    "geom_meta":            {
                        "fe_cn6_hs": {"label": "Fe(CN)6 3- high spin",  "charge": CHARGE, "multiplicity": 6},
                        "fe_cn6_ls": {"label": "Fe(CN)6 3- low spin",   "charge": CHARGE, "multiplicity": 2},
                    },
                    "name_to_geom":         {},
                    "default_charge":       CHARGE,
                    "default_multiplicity": 1,
                },
                client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
            )

            print("Running SP calculations (HS and LS) ...")
            t0     = time.monotonic()
            result = await graph.ainvoke(init_state)
            dt     = time.monotonic() - t0

            arts  = result.get("artifacts", {})
            e_hs  = arts.get("e_hs_eh")
            e_ls  = arts.get("e_ls_eh")
            EH_TO_KCAL = 627.509

            print("\n" + "="*55)
            print("RESULTS  [Fe(CN)6]^3-  B3LYP/def2-SVP SP")
            print("="*55)
            print(f"  E (high spin, mult=6) = {e_hs:>14.6f} Eh" if e_hs else "  E (high spin) = N/A")
            print(f"  E (low  spin, mult=2) = {e_ls:>14.6f} Eh" if e_ls else "  E (low  spin) = N/A")
            if e_hs is not None and e_ls is not None:
                dE_eh   = e_hs - e_ls
                dE_kcal = dE_eh * EH_TO_KCAL
                print(f"\n  dE = E(HS) - E(LS) = {dE_eh:+.6f} Eh  =  {dE_kcal:+.1f} kcal/mol")
                if dE_kcal > 0:
                    print("  => Low spin is lower in energy (expected for CN- strong-field ligand)")
                else:
                    print("  => High spin is lower in energy")
            print(f"\n  Total wall time: {dt:.0f}s")

            # Run log
            print("\nRun log:")
            for e in result.get("run_log", []):
                node = e.get("node") or e.get("node_id", "?")
                print(f"  {node:10s}  status={e.get('status'):9s}  "
                      f"retries={e.get('retry_attempts',0)}  "
                      f"t={e.get('duration_ms',0)//1000}s")

if __name__ == "__main__":
    asyncio.run(main())
