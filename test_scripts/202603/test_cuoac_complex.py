"""
Test: build Cu(OAc)4(H2O) mononuclear unit via build_coordination_complex (molSimplify),
then run_opt_job on the result.

Note: Cu2(OAc)4(H2O)2 is a dinuclear paddlewheel; molSimplify builds one mononuclear Cu
unit [Cu(OAc)4(H2O)]^2- with square-pyramidal geometry as a starting point for optimization.

Usage:
  python test_cuoac_complex.py           # plan mode
  TEST_MODE=full python test_cuoac_complex.py  # plan + execute
"""
import asyncio, json, os, sys, time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.environ.get("ENV_FILE", ".env.lab"))
sys.path.insert(0, str(Path(__file__).parent))

from client_helpers import run_tool_node, state_get_tool_args, auto_display_spectra
from build_graph_from_plan import build_graph_from_plan, build_state
from skills import run_planning_skills
import test_success_rate as tsr

MODE = os.environ.get("TEST_MODE", "plan")

MESSAGE = (
    "Build a geometry for one Cu unit of the Cu2(OAc)4(H2O)2 paddlewheel complex. "
    "Model it as mononuclear [Cu(OAc)4(H2O)]^2- with square pyramidal geometry: "
    "4 acetate ligands (monodentate O-donor, equatorial) and 1 water (axial). "
    "Use build_coordination_complex with metal=cu, ligands=[acetate,acetate,acetate,acetate,water], "
    "coordination_number=5, geometry=sqp (square pyramidal), "
    "oxidation_state=II, spin=2, charge=-2. "
    "Then run_opt_job on the result with B3LYP/def2-SVP. "
    "Report the final energy_eh and opt_converged status."
)


def check_plan(plan: dict) -> dict:
    nodes = plan.get("nodes", [])
    tools = [n.get("tool") for n in nodes if n.get("kind") == "tool"]
    results = {
        "has_build_complex": "build_coordination_complex" in tools,
        "has_opt":           "run_opt_job" in tools,
        "no_llm":            not any(n.get("kind") == "llm" for n in nodes),
    }
    results["PASS"] = results["has_build_complex"] and results["has_opt"]
    return results


async def main():
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from nbo_agent_planning import _build_mcp_server_params

    server_params = _build_mcp_server_params()

    client = OpenAI()

    print(f"\n{'='*60}")
    print(f"Cu(OAc)4(H2O) complex test  [mode={MODE}]")
    print(f"{'='*60}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP connected.")

            state = {"session": session, "client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
                     "geometries": {}, "geom_meta": {}, "name_to_geom": {},
                     "artifacts": {}, "run_log": []}

            skill_ctxs = run_planning_skills(MESSAGE, state)
            msgs = tsr._build_messages(MESSAGE, state, skill_ctxs, compound_context="")
            t0 = time.monotonic()
            plan, usage = tsr._call_planner(client, msgs)
            print(f"Planner tokens: {usage}")

            print("\n── Plan nodes ────────────────────────────────────────────")
            for n in plan.get("nodes", []):
                args = n.get("args", {})
                extra = ""
                if "ligands" in args:
                    extra = f"  ligands={args['ligands']}"
                if "constraints" in args:
                    extra += f"  constraints={args['constraints']}"
                print(f"  [{n.get('kind','tool')}] {n.get('id','?'):30s}  "
                      f"{n.get('tool', n.get('id','?'))}{extra}")

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

            print("\n── Executing ─────────────────────────────────────────────")
            graph = build_graph_from_plan(
                plan, get_tool_args=state_get_tool_args, run_tool_node=run_tool_node,
                openai_client=client,
            )
            init_state = build_state(
                plan, session,
                seed={"client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
                      "geometries": {}, "geom_meta": {}, "name_to_geom": {}},
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


if __name__ == "__main__":
    asyncio.run(main())
