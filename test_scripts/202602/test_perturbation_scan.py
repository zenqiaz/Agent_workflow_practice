"""
Gradual perturbation scan to find the SCF retry threshold.

Two atoms in test_true.xyz are displaced:
  atom 0 (C)  y -= delta
  atom 3 (C)  y += delta

For each delta we run run_opt_job with:
  scf_max_iter=150  (normal production settings)
  on_error: SCF_NOT_CONVERGED -> patch scf_max_iter=500, max_attempts=1

Results are compared with the unperturbed (delta=0) baseline.
The "sweet spot" is the smallest delta that triggers the retry AND converges.
"""

import asyncio
import json
import os
import time
from pathlib import Path

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
    "name_to_geometry_xyz":         name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    "structure_add_remove_proton":  structure_add_remove_proton,
}

# ── parse original geometry ───────────────────────────────────────────────────
def _read_geom_lines(path: str):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    natoms = int(lines[0].split()[0])
    return lines[2 : 2 + natoms]

ORIG_LINES = _read_geom_lines("test_true.xyz")

def apply_perturbation(delta: float) -> str:
    """Shift atom 0 y by -delta and atom 3 y by +delta."""
    result = []
    for i, line in enumerate(ORIG_LINES):
        parts = line.split()
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        if i == 0:
            y -= delta
        elif i == 3:
            y += delta
        result.append(f"{elem:<4s}{x:>16.8f}{y:>16.8f}{z:>16.8f}")
    return "\n".join(result)

# ── perturbation levels to test ───────────────────────────────────────────────
DELTAS = [0.0, 0.2, 0.4, 0.6, 0.8]

# ── MCP setup ─────────────────────────────────────────────────────────────────
_SSH_BIN  = os.getenv("MCP_SSH_BIN",  "ssh")
_SSH_KEY  = os.getenv("MCP_SSH_KEY",  "C:/Users/zrqrc/.ssh/droplet1")
_SSH_HOST = os.getenv("MCP_SSH_HOST", "root@188.166.232.163")
_CMD      = os.getenv(
    "MCP_SERVER_CMD",
    "source ~/venvs/QCagent/bin/activate && cd /root/nbo_agent && "
    "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH "
    "python server_with_product.py",
)

def make_plan(geom_id: str, delta: float) -> dict:
    return {
        "user_text": f"Optimize perturbed structure delta={delta}",
        "name":      f"perturb_scan_d{int(delta*10):02d}",
        "version":   "1.3",
        "geom_ids":  [geom_id],
        "artifacts_to_save": ["opt_energy_eh"],
        "settings":  {},
        "nodes": [
            {
                "id":        "opt",
                "kind":      "tool",
                "tool":      "run_opt_job",
                "input_id":  geom_id,
                "output_id": geom_id,
                "needs":     [],
                "args": {
                    "input_geom_id": geom_id,
                    "method":        "B3LYP",
                    "basis":         "def2-SVP",
                    "use_ri":        True,
                    "scf_max_iter":  150,
                    "wall_timeout_seconds": 3600,
                },
                "expect": {
                    "status_in": ["ok", "warning"],
                    "artifacts_required": [], "properties_required": [],
                },
                "product": {"opt_energy_eh": "energy_eh"},
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
        "final_report": {"format": "markdown", "fields": ["opt_energy_eh"]},
    }


async def run_one(session, delta: float) -> dict:
    """Run a single perturbed optimization and return a result summary dict."""
    geom_id  = f"mol_d{int(delta*10):02d}"
    geom_xyz = apply_perturbation(delta)
    plan     = make_plan(geom_id, delta)

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
            "geom_meta":            {geom_id: {"source": "test_true.xyz", "delta": delta}},
            "name_to_geom":         {},
            "default_charge":       0,
            "default_multiplicity": 1,
        },
        client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
    )

    t0     = time.monotonic()
    result = await graph.ainvoke(init_state)
    dt     = time.monotonic() - t0

    run_log    = result.get("run_log", [])
    artifacts  = result.get("artifacts", {})
    last_status = result.get("last_status", "?")
    opt_entry  = next((e for e in run_log if (e.get("node") or e.get("node_id")) == "opt"), None)
    retries    = (opt_entry or {}).get("retry_attempts", 0)
    energy     = artifacts.get("opt_energy_eh")

    return {
        "delta":      delta,
        "status":     last_status,
        "retried":    retries >= 1,
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
            print("Connected.\n")

            results = []
            baseline_energy = None

            for delta in DELTAS:
                # C0-C1 distance at this delta (analytical)
                import math
                c0y_orig, c1y_orig = 1.45590, -0.06000
                c0x, c0z = 0.02020, 0.22170
                c1x, c1z = -0.12230, 0.06320
                c0y_new = c0y_orig - delta
                d01 = math.sqrt((c0x-c1x)**2 + (c0y_new-c1y_orig)**2 + (c0z-c1z)**2)

                print(f"--- delta={delta:.1f} A  C0-C1={d01:.3f} A ---")
                r = await run_one(session, delta)
                results.append({**r, "c0c1_dist": round(d01, 3)})

                if delta == 0.0:
                    baseline_energy = r["energy_eh"]

                retry_tag = "[RETRY]" if r["retried"] else "      "
                energy_str = f"{r['energy_eh']:.6f}" if r["energy_eh"] is not None else "   N/A  "
                ddE = ""
                if r["energy_eh"] is not None and baseline_energy is not None:
                    ddE = f"  dE={r['energy_eh']-baseline_energy:+.4f} Eh"
                print(f"  status={r['status']:9s} {retry_tag}  E={energy_str} Eh{ddE}  ({r['duration_s']}s)\n")

            # ── Summary table ─────────────────────────────────────────────────
            print("\n" + "="*70)
            print("SUMMARY TABLE")
            print("="*70)
            print(f"{'delta':>6}  {'C0-C1':>6}  {'status':>9}  {'retry':>5}  {'energy_eh':>14}  {'dE vs 0.0':>12}  {'time':>5}")
            print("-"*70)
            for r in results:
                e_str = f"{r['energy_eh']:.6f}" if r["energy_eh"] is not None else "      N/A"
                de_str = ""
                if r["energy_eh"] is not None and baseline_energy is not None:
                    de_str = f"{r['energy_eh']-baseline_energy:+.6f}"
                else:
                    de_str = "       N/A"
                print(f"{r['delta']:>6.1f}  {r['c0c1_dist']:>6.3f}  {r['status']:>9}  {'YES' if r['retried'] else 'no':>5}  {e_str:>14}  {de_str:>12}  {r['duration_s']:>4}s")
            print("="*70)

            sweet = [r for r in results if r["retried"] and r["status"] in ("ok", "warning")]
            if sweet:
                best = sweet[0]
                print(f"\nSWEET SPOT: delta={best['delta']} A  (C0-C1={best['c0c1_dist']} A)")
                print("  Retry fired AND optimization converged.")
            else:
                no_retry = [r for r in results if not r["retried"]]
                did_retry = [r for r in results if r["retried"]]
                if not did_retry:
                    print("\nNo retry triggered at any delta tested (SCF converged natively on all).")
                    print("Consider testing larger deltas or a more convergence-sensitive method.")
                else:
                    print("\nRetry fired but optimization did not converge within wall time.")


if __name__ == "__main__":
    asyncio.run(main())
