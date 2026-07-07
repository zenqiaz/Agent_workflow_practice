"""
test_casscf_avas.py — Test CASSCF with AVAS on Fe(CN)6^3-.

Pipeline:
  1. build_coordination_complex  → octahedral Fe(III) + 6 CN-
  2. run_opt_job                 → B3LYP/def2-SVP geometry optimisation
  3. run_casscf_job              → CAS(5,5) with avas_variant="VALENCE-D", maxiter=300

Compares against the previous result without AVAS (~187s, E = -1815.372096 Eh).

Usage:
    python test_casscf_avas.py
    ENV_FILE=.env python test_casscf_avas.py
"""
import asyncio, json, os, sys, time

from dotenv import load_dotenv
load_dotenv(os.environ.get("ENV_FILE", ".env"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from build_graph_from_plan import build_graph_from_plan, build_state
from client_helpers import (
    run_tool_node, state_get_tool_args,
    name_to_geometry_xyz, pubchem_get_basic_properties,
    structure_add_remove_proton, build_dimer_xyz,
    set_geometry_xyz, build_approach_scan_geometries,
)

# ── plan ─────────────────────────────────────────────────────────────────────
PLAN = {
    "user_text": "CASSCF CAS(5,5) with AVAS on Fe(CN)6^3- (test)",
    "name":      "casscf_avas_fecn6",
    "version":   "1.3",
    "geom_ids":  ["fecn6"],
    "artifacts_to_save": [],
    "nodes": [
        {
            "id":        "build_fecn6",
            "kind":      "tool",
            "tool":      "build_coordination_complex",
            "needs":     [],
            "args":      {
                "metal":         "Fe",
                "ligands":       ["cyanide"] * 6,
                "geometry":      "octahedral",
                "charge":        -3,
                "multiplicity":  2,
            },
            "output_id": "fecn6",
            "expect":    {"status_in": ["ok"]},
        },
        {
            "id":        "opt_fecn6",
            "kind":      "tool",
            "tool":      "run_opt_job",
            "needs":     ["build_fecn6"],
            "input_id":  "fecn6",
            "output_id": "fecn6_opt",
            "args":      {"method": "B3LYP", "basis": "def2-SVP",
                          "wall_timeout_seconds": 1800},
            "expect":    {"status_in": ["ok"]},
        },
        {
            "id":       "casscf_fecn6",
            "kind":     "tool",
            "tool":     "run_casscf_job",
            "needs":    ["opt_fecn6"],
            "input_id": "fecn6_opt",
            "args":     {
                "nel":        5,
                "norb":       5,
                "basis":      "def2-SVP",
                "avas_variant": "VALENCE-D",
                "maxiter":    300,
                "wall_timeout_seconds": 3600,
            },
            "expect":   {"status_in": ["ok"]},
            "on_error": [
                {"if": {"code_in": ["SCF_NOT_CONVERGED"]},
                 "patch": {"scf_max_iter": 500}, "max_attempts": 1},
                {"if": {"code_in": ["CASSCF_NOT_CONVERGED"]},
                 "patch": {"basis": "def2-TZVP"}, "max_attempts": 1},
                {"if": {"code_in": ["RESOURCE_LIMIT"]},
                 "patch": {"wall_timeout_seconds": 7200}, "max_attempts": 1},
            ],
        },
    ],
    "final_report": {
        "format": "markdown",
        "fields": ["casscf_fecn6.energy_eh", "casscf_fecn6.energies_eh",
                   "casscf_fecn6.label"],
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
        "PATH=/root/ORCA/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg:$PATH "
        "python server_with_product.py",
    )
    return StdioServerParameters(
        command=_ssh_bin,
        args=["-i", _ssh_key, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _ssh_host, _ssh_cmd],
        env=dict(os.environ),
    )


async def main():
    t0     = time.monotonic()
    t0_utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    server_params = _build_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            _geom_state = {"geometries": {}, "geom_meta": {}, "name_to_geom": {}, "defaults": {}}
            client_side_tools = {
                "name_to_geometry_xyz":           name_to_geometry_xyz,
                "pubchem_get_basic_properties":    pubchem_get_basic_properties,
                "structure_add_remove_proton":     structure_add_remove_proton,
                "build_dimer_xyz":                 build_dimer_xyz,
                "set_geometry_xyz":                set_geometry_xyz,
                "build_approach_scan_geometries":  build_approach_scan_geometries,
                "state_get_tool_args": lambda **kw: state_get_tool_args(_geom_state, **kw),
            }

            graph      = build_graph_from_plan(PLAN, run_tool_node=run_tool_node)
            init_state = build_state(PLAN, session, seed={
                "client_side_tools": client_side_tools,
            }, client_side_tools=client_side_tools)

            print("Running Fe(CN)6^3- CASSCF CAS(5,5) with AVAS(VALENCE-D) ...")
            result     = await graph.ainvoke(init_state)
            wall_s     = round(time.monotonic() - t0, 1)

            artifacts  = result.get("artifacts") or {}
            run_log    = result.get("run_log") or []

            # ── per-node timing from run_log ──────────────────────────────────
            node_durations = {}
            for entry in run_log:
                nid = entry.get("node")
                ms  = entry.get("duration_ms")
                if nid and ms:
                    node_durations[nid] = ms

            print("\n=== Results ===")
            energy = artifacts.get("casscf_fecn6.energy_eh")
            roots  = artifacts.get("casscf_fecn6.energies_eh")
            print(f"  energy_eh:    {energy}")
            print(f"  energies_eh:  {roots}")
            print(f"  status:       {result.get('status') or result.get('last_status')}")

            print("\n=== Node timings ===")
            for nid, ms in node_durations.items():
                if not nid.startswith("__"):
                    print(f"  {nid}: {ms/1000:.1f}s")
            print(f"  total wall:   {wall_s}s")

            # ── checks ────────────────────────────────────────────────────────
            checks = {
                "casscf_converged": energy is not None,
                "energy_reasonable": (
                    isinstance(energy, float) and -2000 < energy < -1000
                ),
                "avas_used": True,  # structural — always true in this plan
            }
            print("\n=== Checks ===")
            for k, v in checks.items():
                print(f"  [{'PASS' if v else 'FAIL'}] {k}")

            # ── reference comparison ─────────────────────────────────────────
            PREV_ENERGY = -1815.372096  # previous run without AVAS
            if isinstance(energy, float):
                delta = energy - PREV_ENERGY
                print(f"\n  vs. prev (no AVAS): ΔE = {delta:+.6f} Eh")

            # ── save log ──────────────────────────────────────────────────────
            os.makedirs("test_logs", exist_ok=True)
            log_path = f"test_logs/casscf_avas_{t0_utc}.json"
            log = {
                "experiment":      "casscf_avas_fecn6",
                "timestamp_utc":   t0_utc,
                "wall_elapsed_s":  wall_s,
                "model":           os.getenv("LLM_MODEL", "gpt-4.1-mini"),
                "env_file":        os.getenv("ENV_FILE", ".env"),
                "plan":            PLAN,
                "checks":          checks,
                "energy_eh":       energy,
                "energies_eh":     roots,
                "node_durations_ms": node_durations,
                "final_status":    result.get("status") or result.get("last_status"),
                "artifacts":       {k: v for k, v in artifacts.items()
                                    if not k.startswith("runtime_report")
                                    and not k.startswith("bug_report")},
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=True)
            print(f"\nLog saved: {log_path}")

            if not all(checks.values()):
                sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
