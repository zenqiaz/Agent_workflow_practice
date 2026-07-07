"""Quick test: LLM calculation node with pre-populated artifacts."""
import asyncio
import json
from openai import OpenAI
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from build_graph_from_plan import build_graph_from_plan, build_state

# A minimal plan with only an LLM calc node.
# Artifacts are pre-seeded (no tool nodes needed).
PLAN = {
    "name": "Test LLM pKa calculation from G values",
    "version": "1.3",
    "geom_ids": [],
    "artifacts_to_save": ["pka", "deltaG_J_mol"],
    "settings": {
        "gas_constant_R_J_molK": 8.314462618,
        "temperature_K": 298.15,
        "ln10": 2.302585092994046,
        "Eh_to_J_mol": 2625499.638,
        "G_H_plus_ref_eh": -0.01372,
    },
    "nodes": [
        {
            "id": "compute_pka",
            "kind": "llm",
            "task": "compute_pka",
            "needs": [],
            "prompt": (
                "Compute the pKa of HA from the Gibbs free energies of HA and A-.\n"
                "Formula: deltaG = G_A_minus_eh + G_H_plus_ref_eh - G_HA_eh (all in Hartree).\n"
                "Then convert: deltaG_J_mol = deltaG * Eh_to_J_mol.\n"
                "Then: pKa = deltaG_J_mol / (R * T * ln10).\n"
                "G_H_plus_ref_eh is provided in plan settings.\n"
                "Return JSON with: status, deltaG_eh, deltaG_J_mol, pka."
            ),
            "needs_artifacts": ["G_HA_eh", "G_A_minus_eh"],
            "product": {"pka": "pka", "deltaG_J_mol": "deltaG_J_mol"},
        }
    ],
    "final_report": {"format": "markdown", "fields": ["pka", "deltaG_J_mol"]},
}

# Pre-seed artifacts with example Gibbs free energies (Hartree)
# These are made-up but realistic values for acetic acid / acetate
SEED_ARTIFACTS = {
    "G_HA_eh": -229.12345,      # G(acetic acid) in Eh
    "G_A_minus_eh": -228.58901,  # G(acetate anion) in Eh
}


async def main():
    client = OpenAI()

    graph = build_graph_from_plan(
        PLAN,
        openai_client=client,
    )

    # Build initial state — no MCP session needed for LLM-only plans
    init_state = build_state(PLAN, session=None, seed={
        "artifacts": dict(SEED_ARTIFACTS),
    })

    print("=== Initial artifacts ===")
    pprint(init_state.get("artifacts"))
    print()

    result = await graph.ainvoke(init_state)

    print("=== Final status ===")
    print("Status:", result.get("last_status"))
    print()

    print("=== Final artifacts ===")
    pprint(result.get("artifacts"))
    print()

    print("=== Run log ===")
    for entry in (result.get("run_log") or []):
        print(f"  {entry.get('node'):20s}  {entry.get('kind'):6s}  {entry.get('status'):6s}  {entry.get('duration_ms', '?')}ms")

    print()
    print("=== LLM node result ===")
    node_res = (result.get("node_results") or {}).get("compute_pka")
    pprint(node_res)


if __name__ == "__main__":
    asyncio.run(main())
