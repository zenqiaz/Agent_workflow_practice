"""
CCSD perturbation scan on ethanol (test_9.xyz).

Two atoms are displaced:
  atom 0 (C)  y -= delta   (compresses C0-O1 bond)
  atom 2 (C)  y += delta   (stretches O1-C2 bond)

For each delta we run run_sp_energy with:
  method="CCSD", basis="cc-pVDZ"
  scf_max_iter=150  (normal)
  on_error: SCF_NOT_CONVERGED -> scf_max_iter=500, max_attempts=1

CCSD is single-reference but more convergence-sensitive than B3LYP,
so it should trigger SCF/amplitude failures at smaller perturbations.
"""

import asyncio
import math
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.getenv("ENV_FILE", ".env.lab"), override=True)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from client_helpers import (
    run_tool_node, state_get_tool_args,
    name_to_geometry_xyz, pubchem_get_basic_properties, structure_add_remove_proton,
)
from build_graph_from_plan import build_graph_from_plan, build_state

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz":         name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    "structure_add_remove_proton":  structure_add_remove_proton,
}

# ── geometry ──────────────────────────────────────────────────────────────────
def _read_geom_lines(path: str):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].split()[0])
    return lines[2 : 2 + natoms]

ORIG_LINES = _read_geom_lines("test_9.xyz")

def apply_perturbation(delta: float) -> str:
    result = []
    for i, line in enumerate(ORIG_LINES):
        parts = line.split()
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        if i == 0:
            y -= delta      # compress C0-O1
        elif i == 2:
            y += delta      # stretch O1-C2
        result.append(f"{elem:<4s}{x:>16.8f}{y:>16.8f}{z:>16.8f}")
    return "\n".join(result)

DELTAS = [0.0, 0.2, 0.4, 0.6, 0.8]

# Reference distances (analytical)
C0 = (-0.01480, 1.39240, 0.00570)
O1 = (-0.00470,-0.01360, 0.01460)
C2 = ( 0.63790,-0.55330,-1.11360)

def c0o1_dist(delta):
    return math.sqrt((C0[0]-O1[0])**2 + (C0[1]-delta-O1[1])**2 + (C0[2]-O1[2])**2)

def o1c2_dist(delta):
    return math.sqrt((O1[0]-C2[0])**2 + (O1[1]-(C2[1]+delta))**2 + (O1[2]-C2[2])**2)

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

def make_plan(geom_id: str) -> dict:
    return {
        "user_text": "CCSD SP on perturbed ethanol",
        "name":      f"ccsd_{geom_id}",
        "version":   "1.3",
        "geom_ids":  [geom_id],
        "artifacts_to_save": ["ccsd_energy_eh"],
        "settings":  {},
        "nodes": [
            {
                "id":       "sp",
                "kind":     "tool",
                "tool":     "run_sp_energy",
                "input_id": geom_id,
                "needs":    [],
                "args": {
                    "input_geom_id":        geom_id,
                    "method":               "CCSD",
                    "basis":                "cc-pVDZ",
                    "use_ri":               False,   # no RI for canonical CCSD
                    "scf_max_iter":         150,
                    "wall_timeout_seconds": 1800,
                },
                "expect": {
                    "status_in": ["ok", "warning"],
                    "artifacts_required": [], "properties_required": [],
                },
                "product": {"ccsd_energy_eh": "energy_eh"},
                "on_error": [
                    {
                        "if":          {"code_in": ["SCF_NOT_CONVERGED"]},
                        "action":      "patch_and_retry",
                        "patch":       {"scf_max_iter": 500},
                        "max_attempts": 1,
                    }
                ],
            }
        ],
        "final_report": {"format": "markdown", "fields": ["ccsd_energy_eh"]},
    }


async def run_one(session, delta: float) -> dict:
    geom_id  = f"etoh_d{int(delta*10):02d}"
    geom_xyz = apply_perturbation(delta)
    plan     = make_plan(geom_id)

    graph = build_graph_from_plan(
        plan,
        get_tool_args=state_get_tool_args,
        run_tool_node=run_tool_node,
    )
    init_state = build_state(
        plan, session,
        seed={
            "client_side_tools":    CLIENT_SIDE_TOOL_FUNCS,
            "geometries":           {geom_id: geom_xyz},
            "geom_meta":            {geom_id: {"source": "test_9.xyz", "delta": delta}},
            "name_to_geom":         {},
            "default_charge":       0,
            "default_multiplicity": 1,
        },
        client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
    )

    t0      = time.monotonic()
    result  = await graph.ainvoke(init_state)
    dt      = time.monotonic() - t0

    run_log     = result.get("run_log", [])
    artifacts   = result.get("artifacts", {})
    last_status = result.get("last_status", "?")
    sp_entry    = next((e for e in run_log if (e.get("node") or e.get("node_id")) == "sp"), None)
    retries     = (sp_entry or {}).get("retry_attempts", 0)
    energy      = artifacts.get("ccsd_energy_eh")

    # Also grab the raw tool result to see CCSD-specific output
    node_results = result.get("node_results", {})
    sp_raw       = (node_results.get("sp") or {}).get("last_tool_result", {})
    error_code   = sp_raw.get("code") if isinstance(sp_raw, dict) else None

    return {
        "delta":      delta,
        "c0o1":       round(c0o1_dist(delta), 3),
        "o1c2":       round(o1c2_dist(delta), 3),
        "status":     last_status,
        "retried":    retries >= 1,
        "error_code": error_code,
        "energy_eh":  energy,
        "duration_s": round(dt),
    }


async def main():
    server_params = StdioServerParameters(
        command=_SSH_BIN,
        args=["-i", _SSH_KEY, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _SSH_HOST, _CMD],
        env=dict(os.environ),
    )

    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected. Running CCSD/cc-pVDZ SP on perturbed ethanol.\n")

            results      = []
            baseline_e   = None

            for delta in DELTAS:
                d01 = c0o1_dist(delta)
                d12 = o1c2_dist(delta)
                print(f"--- delta={delta:.1f} A  C0-O1={d01:.3f} A  O1-C2={d12:.3f} A ---")
                r = await run_one(session, delta)
                results.append(r)

                if delta == 0.0:
                    baseline_e = r["energy_eh"]

                retry_tag = "[RETRY]" if r["retried"] else "      "
                e_str     = f"{r['energy_eh']:.6f}" if r["energy_eh"] is not None else "    N/A   "
                dE_str    = ""
                if r["energy_eh"] is not None and baseline_e is not None:
                    dE_str = f"  dE={r['energy_eh']-baseline_e:+.4f} Eh"
                code_str  = f"  code={r['error_code']!r}" if r["error_code"] else ""
                print(f"  status={r['status']:9s} {retry_tag}  E={e_str} Eh{dE_str}{code_str}  ({r['duration_s']}s)\n")

                # Stop early once we've seen a retry that converged (sweet spot found)
                if r["retried"] and r["status"] in ("ok", "warning"):
                    print("  >> Sweet spot found — stopping scan.")
                    break

            # ── Summary ───────────────────────────────────────────────────────
            print("\n" + "="*74)
            print("SUMMARY  (CCSD/cc-pVDZ SP on ethanol)")
            print("="*74)
            print(f"{'delta':>6}  {'C0-O1':>6}  {'O1-C2':>6}  {'status':>9}  {'retry':>5}  {'energy_eh':>14}  {'dE (Eh)':>10}  {'t':>4}")
            print("-"*74)
            for r in results:
                e_str  = f"{r['energy_eh']:.6f}" if r["energy_eh"] is not None else "         N/A"
                dE_str = f"{r['energy_eh']-baseline_e:+.6f}" if (r["energy_eh"] is not None and baseline_e is not None) else "         N/A"
                print(f"{r['delta']:>6.1f}  {r['c0o1']:>6.3f}  {r['o1c2']:>6.3f}  {r['status']:>9}  {'YES' if r['retried'] else 'no':>5}  {e_str:>14}  {dE_str:>10}  {r['duration_s']:>3}s")
            print("="*74)

            sweet = [r for r in results if r["retried"] and r["status"] in ("ok", "warning")]
            if sweet:
                b = sweet[0]
                print(f"\nSweet spot: delta={b['delta']} A  (C0-O1={b['c0o1']} A)")
                print("  Retry fired AND CCSD converged.")
            else:
                any_retry = [r for r in results if r["retried"]]
                if not any_retry:
                    print("\nNo retry triggered. CCSD/cc-pVDZ converges natively on all tested geometries.")
                else:
                    print("\nRetry fired but CCSD did not finish successfully.")


if __name__ == "__main__":
    asyncio.run(main())
