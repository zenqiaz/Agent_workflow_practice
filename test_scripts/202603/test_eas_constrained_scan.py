"""
Test: benzene + NO2+ EAS constrained-opt approach scan (Workflow C).

Preloads benzene (q=0) and no2plus positioned with N above C0 at 1.8 Å.
Plans with build_approach_scan_geometries + parallel run_opt_job (C0-N fixed).
Monomer energies from run_sp_energy on monomers at their input geometries.

Atom indices in merged dimer: benzene 0-11, NO2+ O=12, O=13, N=14.
Constraint: fix bond between atom 0 (C0 in benzene) and atom 14 (N in NO2+).

Usage:
  python test_eas_constrained_scan.py           # plan mode only
  TEST_MODE=full python test_eas_constrained_scan.py  # plan + execute
"""
import asyncio, json, os, sys, textwrap, time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.environ.get("ENV_FILE", ".env.lab"))
sys.path.insert(0, str(Path(__file__).parent))

from client_helpers import (
    set_geometry_xyz, run_tool_node, state_get_tool_args,
    auto_display_spectra, build_approach_scan_geometries,
)
from build_graph_from_plan import build_graph_from_plan, build_state
from skills import run_planning_skills
import test_success_rate as tsr

MODE = os.environ.get("TEST_MODE", "plan")

# ── Geometries ────────────────────────────────────────────────────────────────
# Benzene: ring in XY plane, centroid at origin, C0 at (0, 1.396, 0)
BENZENE_XYZ = textwrap.dedent("""\
    C    0.000000    1.396160    0.000000
    C    1.208916    0.698080    0.000000
    C    1.208916   -0.698080    0.000000
    C    0.000000   -1.396160    0.000000
    C   -1.208916   -0.698080    0.000000
    C   -1.208916    0.698080    0.000000
    H    0.000000    2.479500    0.000000
    H    2.147500    1.239750    0.000000
    H    2.147500   -1.239750    0.000000
    H    0.000000   -2.479500    0.000000
    H   -2.147500   -1.239750    0.000000
    H   -2.147500    1.239750    0.000000""").strip()

# NO2+: N directly above C0 (0, 1.396, 0) at z=1.8 Å; O-N-O along X
NO2PLUS_XYZ = textwrap.dedent("""\
    O   -1.154000    1.396160    1.800000
    O    1.154000    1.396160    1.800000
    N    0.000000    1.396160    1.800000""").strip()

MESSAGE = (
    "Compute the EAS bond-forming energy curve for benzene + NO2+ using "
    "constrained geometry optimization (Workflow C). "
    "benzene (q=0, 12 atoms) is the fixed host and no2plus (q=+1, 3 atoms: O=0,O=1,N=2) "
    "is positioned with N directly above C0 (atom 0 of benzene) at 1.8 Ang. "
    "Step 1: Use build_approach_scan_geometries with mol_a_id=benzene, mol_b_id=no2plus, "
    "n_steps=9, step_ang=0.05, ref_atom_a=0, output_prefix=approach "
    "to generate approach_00 through approach_08 (C0-N from 1.80 down to 1.40 Ang). "
    "In each merged dimer geometry, benzene occupies atoms 0-11 and NO2+ occupies atoms 12-14 "
    "(O=12, O=13, N=14). "
    "Step 2: Run constrained geometry optimization on each of the 9 dimer geometries in parallel "
    "using run_opt_job with constraints=[{\"type\":\"B\",\"atoms\":[0,14]}] "
    "(fixes the C0-N bond at its current distance, relaxes all other degrees of freedom). "
    "Step 3: Run monomer SPs for benzene and no2plus in parallel (unconstrained run_sp_energy). "
    "Step 4: Compute delta_E_int = E(opt_dimer) - E(benzene) - E(no2plus) in kcal/mol "
    "at each step using ONE llm node for the full curve. "
    "Use B3LYP/def2-SVP."
)


def check_plan(plan: dict) -> dict:
    nodes = plan.get("nodes", [])
    tools = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    results = {
        "has_approach_scan":   "build_approach_scan_geometries" in tools,
        "has_constrained_opt": tools.count("run_opt_job") >= 3,   # ≥3 approach opts
        "has_sp_monomers":     tools.count("run_sp_energy") >= 2,
        "has_delta_e_node":    any(n.get("kind") == "llm" for n in nodes),
        "no_orca_scan":        "run_scan_job" not in tools,
    }
    results["PASS"] = all(results.values())
    return results


async def main():
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from nbo_agent_planning import _build_mcp_server_params

    server_params = _build_mcp_server_params()

    client = OpenAI()

    print(f"\n{'='*60}")
    print(f"EAS constrained-opt scan test  [mode={MODE}]")
    print(f"{'='*60}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP connected.")

            state = {"session": session, "client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
                     "geometries": {}, "geom_meta": {}, "name_to_geom": {},
                     "artifacts": {}, "run_log": []}
            set_geometry_xyz(state, "benzene", BENZENE_XYZ, charge=0, multiplicity=1)
            set_geometry_xyz(state, "no2plus", NO2PLUS_XYZ, charge=1, multiplicity=1)
            print("Preloaded: benzene (12 atoms, q=0), no2plus (3 atoms, q=+1, N above C0 at z=1.8 Ang)\n")

            # Plan
            skill_ctxs = run_planning_skills(MESSAGE, state)
            msgs = tsr._build_messages(MESSAGE, state, skill_ctxs, compound_context="")
            t0 = time.monotonic()
            plan, usage = tsr._call_planner(client, msgs)
            print(f"Planner tokens: {usage}")

            print("\n── Plan nodes ────────────────────────────────────────────")
            for n in plan.get("nodes", []):
                kind = n.get("kind", "tool")
                tool = n.get("tool", n.get("id", "?"))
                args = n.get("args", {})
                cstr = args.get("constraints", "")
                extra = f"  constraints={cstr}" if cstr else ""
                print(f"  [{kind}] {n.get('id','?'):30s}  {tool}{extra}")

            checks = check_plan(plan)
            print("\n── Plan checks ───────────────────────────────────────────")
            for k, v in checks.items():
                print(f"  {'OK' if v else 'NG'} {k}")

            if not checks["PASS"]:
                print("\nPLAN CHECK FAILED")
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                return

            if MODE == "plan":
                print("\nPlan OK — run with TEST_MODE=full to execute.")
                return

            # Execute
            print("\n── Executing ─────────────────────────────────────────────")
            graph = build_graph_from_plan(
                plan, get_tool_args=state_get_tool_args, run_tool_node=run_tool_node,
                openai_client=client,
            )
            init_state = build_state(
                plan, session,
                seed={
                    "client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
                    "geometries":  dict(state["geometries"]),
                    "geom_meta":   dict(state["geom_meta"]),
                    "name_to_geom": dict(state["name_to_geom"]),
                },
                client_side_tools=tsr.CLIENT_SIDE_TOOL_FUNCS,
            )
            result = await graph.ainvoke(init_state)
            elapsed = time.monotonic() - t0

            artifacts = result.get("artifacts", {})
            print(f"\nDone in {elapsed:.0f}s")
            print("\n── Artifacts ─────────────────────────────────────────────")
            for k, v in artifacts.items():
                display = json.dumps(v)[:120] if not isinstance(v, str) else v[:120]
                print(f"  {k}: {display}")

            curve = artifacts.get("interaction_curve")
            if curve:
                print("\n── Interaction energy curve ──────────────────────────────")
                print(f"  {'step':>4}  {'dist(Å)':>8}  {'ΔE_int(kcal/mol)':>18}")
                for pt in curve:
                    print(f"  {pt.get('step','?'):>4}  "
                          f"{float(pt.get('distance_ang', 0)):>8.2f}  "
                          f"{float(pt.get('delta_e_int_kcal', 0)):>18.2f}")
                be = artifacts.get("binding_energy_kcal")
                de = artifacts.get("d_eq_ang")
                if be is not None:
                    print(f"\n  Min ΔE_int: {be:.2f} kcal/mol  at {de:.2f} Å")

            auto_display_spectra(artifacts)


if __name__ == "__main__":
    asyncio.run(main())
