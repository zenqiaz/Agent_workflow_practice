"""
Test build_coordination_complex + run_opt_job for two simple cases:
  1. [Fe(CN)6]^3-  oct, Fe(III) LS, spin=2, charge=-3
  2. [Cr(en)3]^3+  oct, Cr(III),   spin=4, charge=+3

Usage:
  python test_coord_complexes.py              # plan mode (both)
  TEST_MODE=full python test_coord_complexes.py  # plan + execute (both)
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

CASES = [
    {
        "label": "Fe(CN)6^3-",
        "message": (
            "Build [Fe(CN)6]^3- (hexacyanoferrate(III)), octahedral geometry. "
            "Use build_coordination_complex with metal=fe, ligands=[cn,cn,cn,cn,cn,cn], "
            "geometry=oct, oxidation_state=III, charge=-3. "
            "Omit spin — let the ANN predict (CN is strong-field, expect LS doublet). "
            "Then run_opt_job with B3LYP/def2-SVP, use_ri=True. "
            "Report energy_eh and opt_converged."
        ),
    },
    {
        "label": "Pt(en)2^2+",
        "message": (
            "Build [Pt(en)2]^2+ (bis(ethylenediamine)platinum(II)), square planar geometry. "
            "Use build_coordination_complex with metal=pt, ligands=[en,en], "
            "geometry=sqp, oxidation_state=II, charge=+2. "
            "Omit spin — Pt(II) d8 square planar is always diamagnetic (spin=1, auto). "
            "Then run_opt_job with B3LYP/def2-SVP, use_ri=True. "
            "Report energy_eh and opt_converged."
        ),
    },
]


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


async def run_case(case: dict, session, client: OpenAI):
    label   = case["label"]
    message = case["message"]

    print(f"\n{'='*60}")
    print(f"  {label}  [mode={MODE}]")
    print(f"{'='*60}\n")

    state = {
        "session": session,
        "client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
        "geometries": {}, "geom_meta": {}, "name_to_geom": {},
        "artifacts": {}, "run_log": [],
    }

    skill_ctxs = run_planning_skills(message, state)
    msgs = tsr._build_messages(message, state, skill_ctxs, compound_context="")
    t0 = time.monotonic()
    plan, usage = tsr._call_planner(client, msgs)
    print(f"Planner tokens: {usage}")

    print("\n── Plan nodes ────────────────────────────────────────────")
    for n in plan.get("nodes", []):
        args  = n.get("args", {})
        extra = f"  ligands={args['ligands']}" if "ligands" in args else ""
        print(f"  [{n.get('kind','tool')}] {n.get('id','?'):30s}  "
              f"{n.get('tool', n.get('id','?'))}{extra}")

    checks = check_plan(plan)
    print("\n── Plan checks ───────────────────────────────────────────")
    for k, v in checks.items():
        print(f"  {'OK' if v else 'NG'} {k}")

    if not checks["PASS"]:
        print("\nPLAN CHECK FAILED")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return {"label": label, "status": "plan_fail"}

    if MODE == "plan":
        print("\nPlan OK — run with TEST_MODE=full to execute.")
        return {"label": label, "status": "plan_ok"}

    print("\n── Executing ─────────────────────────────────────────────")
    graph = build_graph_from_plan(
        plan,
        get_tool_args=state_get_tool_args,
        run_tool_node=run_tool_node,
        openai_client=client,
    )
    init_state = build_state(
        plan, session,
        seed={"client_side_tools": tsr.CLIENT_SIDE_TOOL_FUNCS,
              "geometries": {}, "geom_meta": {}, "name_to_geom": {}},
        client_side_tools=tsr.CLIENT_SIDE_TOOL_FUNCS,
    )
    result  = await graph.ainvoke(init_state)
    elapsed = time.monotonic() - t0

    artifacts = result.get("artifacts", {})
    print(f"\nDone in {elapsed:.0f}s")
    print("── Artifacts ─────────────────────────────────────────────")
    for k, v in artifacts.items():
        display = json.dumps(v)[:120] if not isinstance(v, str) else v[:120]
        print(f"  {k}: {display}")

    return {"label": label, "elapsed": elapsed, "artifacts": artifacts}


async def main():
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from nbo_agent_planning import _build_mcp_server_params

    server_params = _build_mcp_server_params()

    client = OpenAI()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP connected.")

            results = []
            for case in CASES:
                r = await run_case(case, session, client)
                results.append(r)

            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            for r in results:
                status = r.get("status", "")
                E      = r.get("artifacts", {}).get("energy_eh") or r.get("artifacts", {}).get("E_opt_eh")
                conv   = r.get("artifacts", {}).get("opt_converged")
                elapsed = r.get("elapsed", 0)
                print(f"  {r['label']:20s}  status={status or 'executed'}  "
                      f"E={E}  converged={conv}  t={elapsed:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
