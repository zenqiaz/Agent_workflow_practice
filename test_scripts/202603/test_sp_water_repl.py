"""
Test: SP energy of water using the new REPL identification flow.

Exercises:
  1. extract_compound_names_llm  — extract "water" from a planning request
  2. fetch_compound_card          — PubChem/OPSIN lookup for water
  3. display_and_confirm_compound — display card + image (auto-confirms via mocked input)
  4. compounds_to_planner_context — format confirmed card as planner context string
  5. Full graph execution         — name_to_geometry_xyz → run_sp_energy for water
"""
import asyncio
import json
import os
from pprint import pprint
from unittest.mock import patch

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from build_graph_from_plan import build_graph_from_plan, build_state
from client_helpers import (
    extract_compound_names_llm,
    fetch_compound_card,
    display_and_confirm_compound,
    compounds_to_planner_context,
    run_tool_node,
    state_get_tool_args,
    name_to_geometry_xyz,
    pubchem_get_basic_properties,
)

load_dotenv(os.environ.get("ENV_FILE", ".env"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

USER_TEXT = "calculate sp energy of water"

CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
}

PLAN = {
    "user_text": USER_TEXT,
    "name": "SP energy of water",
    "version": "1.3",
    "geom_ids": ["water"],
    "artifacts_to_save": ["E_water_eh"],
    "settings": {},
    "nodes": [
        {
            "id": "load_water",
            "kind": "tool",
            "tool": "name_to_geometry_xyz",
            "output_id": "water",
            "needs": [],
            "args": {"name": "water"},
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {},
        },
        {
            "id": "sp_water",
            "kind": "tool",
            "tool": "run_sp_energy",
            "input_id": "water",
            "output_id": "water",
            "needs": ["load_water"],
            "args": {"input_geom_id": "water", "method": "B3LYP", "basis": "def2-SVP"},
            "expect": {"status_in": ["ok"], "artifacts_required": [], "properties_required": []},
            "product": {"E_water_eh": "energy"},
        },
    ],
    "final_report": {"format": "markdown", "fields": ["E_water_eh"]},
}


def run_identification_phase(client) -> str:
    """Run the new compound identification flow with auto-confirm (mocked input).

    Returns the compound_context string that would be injected into the planner.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Compound identification (new REPL flow)")
    print("=" * 60)
    print(f"User text: {USER_TEXT!r}")

    # Step 1: extract names from user text
    names = extract_compound_names_llm(USER_TEXT, client)
    print(f"\nextract_compound_names_llm → {names}")
    assert names, "Expected at least one compound name to be extracted"

    # Step 2: fetch card for each name; display with auto-confirmed input ("y")
    confirmed = []
    with patch("builtins.input", return_value="y"):
        for name in names:
            card = fetch_compound_card(name)
            print(f"\nfetch_compound_card({name!r}) →")
            print(f"  source  : {card.get('source')}")
            print(f"  formula : {card.get('formula')}")
            print(f"  smiles  : {card.get('smiles')}")
            print(f"  charge  : {card.get('charge')}")
            print(f"  MW      : {card.get('mw')}")
            result = display_and_confirm_compound(card, client)
            if result:
                confirmed.append(result)

    assert confirmed, "Expected at least one compound to be confirmed"

    # Step 3: format context for planner
    compound_context = compounds_to_planner_context(confirmed)
    print("\ncompounds_to_planner_context →")
    print(compound_context)

    # Sanity checks
    water_card = confirmed[0]
    assert water_card.get("formula") in ("H2O", None), \
        f"Unexpected formula: {water_card.get('formula')!r}"
    assert water_card.get("charge", 0) == 0, \
        f"Unexpected charge for water: {water_card.get('charge')}"
    assert "water" in compound_context.lower() or "H2O" in compound_context, \
        "Compound context should mention water or H2O"

    print("\n[Identification phase PASSED]")
    return compound_context


async def run_sp_graph(session, client) -> dict:
    """Build and execute the SP energy graph for water."""
    print("\n" + "=" * 60)
    print("PHASE 2: Graph execution (SP energy of water)")
    print("=" * 60)
    print("Nodes:", [n["id"] for n in PLAN["nodes"]])

    graph = build_graph_from_plan(
        PLAN,
        get_tool_args=state_get_tool_args,
        run_tool_node=run_tool_node,
        openai_client=client,
    )
    init_state = build_state(
        PLAN,
        session,
        seed={"client_side_tools": CLIENT_SIDE_TOOL_FUNCS},
        client_side_tools=CLIENT_SIDE_TOOL_FUNCS,
    )
    result = await graph.ainvoke(init_state)
    return result


def print_results(result: dict, compound_context: str) -> None:
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFinal status : {result.get('last_status')}")

    print("\nArtifacts:")
    artifacts = result.get("artifacts", {})
    e = artifacts.get("E_water_eh")
    print(f"  E_water_eh : {e}")

    print("\nRun log:")
    for entry in (result.get("run_log") or []):
        node = entry.get("node", "?")
        kind = entry.get("kind", "?")
        status = entry.get("status", "?")
        dur = entry.get("duration_ms", "?")
        print(f"  {node:20s}  {kind:6s}  {status:8s}  {dur} ms")

    print("\nCompound context that was available to planner:")
    print(compound_context or "  (none)")

    # Final assertion
    assert result.get("last_status") == "ok", \
        f"Graph did not finish with status 'ok': {result.get('last_status')}"
    assert e is not None, "E_water_eh artifact is missing"
    assert isinstance(e, (int, float)) and e < 0, \
        f"SP energy should be a negative number in Hartree, got {e}"

    print("\n[SP energy test PASSED]")
    print(f"  B3LYP/def2-SVP energy of water = {e:.8f} Eh")


async def main():
    from nbo_agent_planning import _build_mcp_server_params

    server_params = _build_mcp_server_params()

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)

    # Phase 1: identification (local, no MCP needed)
    compound_context = run_identification_phase(client)

    # Phase 2: graph execution (requires MCP / ORCA on VM)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("\nMCP tools available:", [t.name for t in tools.tools])

            result = await run_sp_graph(session, client)

    print_results(result, compound_context)


if __name__ == "__main__":
    asyncio.run(main())
