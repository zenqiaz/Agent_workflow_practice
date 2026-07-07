"""
Test: XYZ load identification flow.

Verifies that when a geometry is loaded into state (via 'load' command):
  1. render_xyz_image_rdkit  — renders 2D PNG from XYZ atom lines, opens in OS viewer
  2. fetch_compound_card_from_xyz — detects SMILES/formula via DetermineBonds + PubChem
  3. display_and_confirm_compound — shows card, auto-confirmed
  4. identify_and_confirm_compounds — picks up the loaded-but-unidentified geometry
     from state and processes it, even when user text contains no compound name

Uses water XYZ as the loaded geometry.
"""
import asyncio
from unittest.mock import patch
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from client_helpers import (
    render_xyz_image_rdkit,
    fetch_compound_card_from_xyz,
    display_and_confirm_compound,
    compounds_to_planner_context,
)
from nbo_agent_planning import identify_and_confirm_compounds, AgentState

WATER_XYZ = (
    "O 0.000000 0.000000 0.000000\n"
    "H 0.277400 0.892900 0.254400\n"
    "H 0.606800 -0.238300 -0.716900"
)
GEOM_ID = "water.xyz"


def test_render_xyz_image():
    print("=== render_xyz_image_rdkit ===")
    img = render_xyz_image_rdkit(WATER_XYZ, GEOM_ID, charge=0)
    assert img is not None, "Expected a PNG path, got None"
    print(f"  Image: {img} [opened]")
    print("  PASS\n")


def test_fetch_card_from_xyz():
    print("=== fetch_compound_card_from_xyz ===")
    card = fetch_compound_card_from_xyz(WATER_XYZ, GEOM_ID, charge=0)
    print(f"  source  : {card['source']}")
    print(f"  smiles  : {card['smiles']}")
    print(f"  formula : {card['formula']}")
    print(f"  mw      : {card['mw']}")
    print(f"  charge  : {card['charge']}")
    print(f"  cid     : {card['cid']}")
    assert card["smiles"] is not None, "Expected SMILES from DetermineBonds"
    assert card["formula"] is not None, "Expected formula"
    assert card["charge"] == 0
    print("  PASS\n")
    return card


def test_display_and_confirm(card, client):
    print("=== display_and_confirm_compound (auto-confirm) ===")
    with patch("builtins.input", return_value="y"):
        result = display_and_confirm_compound(card, client)
    assert result is not None, "Expected confirmed card"
    assert result["smiles"] is not None
    print(f"  confirmed name   : {result['name']}")
    print(f"  confirmed source : {result['source']}")
    print("  PASS\n")
    return result


async def test_identify_from_state(client):
    """Simulate: user loaded a geometry, then typed a planning request with no name."""
    print("=== identify_and_confirm_compounds (geometry in state, no name in text) ===")

    state: AgentState = {
        "files": {},
        "geometries": {GEOM_ID: WATER_XYZ},
        "name_to_geom": {},
        "identifiers": {},
        "geom_meta": {GEOM_ID: {"charge": 0, "multiplicity": 1}},
        "cached_props": {},
        "plans": {},
        "runs": [],
        "run_log": [],
        "artifacts": {},
        "node_results": {},
        "default_charge": 0,
        "default_multiplicity": 1,
    }

    user_text = "calculate the sp energy of the loaded molecule"

    with patch("builtins.input", return_value="y"):
        confirmed = await identify_and_confirm_compounds(user_text, client, state)

    print(f"\n  confirmed count : {len(confirmed)}")
    assert len(confirmed) >= 1, "Expected at least one confirmed card from loaded geometry"

    ctx = compounds_to_planner_context(confirmed)
    print("\n  Planner context:")
    print(ctx)

    assert GEOM_ID in ctx or "H2O" in ctx or "water" in ctx.lower(), \
        "Planner context should reference the loaded geometry"

    # Check state caches were populated
    assert GEOM_ID in state.get("cached_props", {}), \
        "cached_props should contain the geom_id"

    print("  PASS\n")


def main():
    client = OpenAI()

    test_render_xyz_image()
    card = test_fetch_card_from_xyz()
    test_display_and_confirm(card, client)
    asyncio.run(test_identify_from_state(client))

    print("=" * 50)
    print("All XYZ load identification tests PASSED")


if __name__ == "__main__":
    main()
