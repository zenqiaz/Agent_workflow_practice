# nbo_agent_repl.py
import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict, TypedDict, List
import traceback
from dataclasses import asdict
from pprint import pprint

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent
from openai import OpenAI
from urllib.request import urlretrieve

from client_helpers import (
    NEEDS_GEOM_SINGLE,
    TOOLS_RETURNING_STRUCTURE,
    load_xyz_as_geometry,
    summarize_geometries_prompt,
    geom_key_from_path,
    name_to_geometry_xyz,
    build_dimer_xyz,
    set_geometry_xyz,
    build_approach_scan_geometries,
    structure_add_remove_proton,
    print_tool_output,
    get_trivial_properties,
    pubchem_get_basic_properties,
    ALLOWED_STATE_KEYS,
    state_update,
    state_get_tool_args,
    _maybe_store_geometry_from_payload,
    xyz_to_no_header,
    print_session_state,
    handle_planimg_command,
    handle_plansave_command,
    summarize_workflow_state,
    parse_json_only,
    build_plan_review,
    print_plan_review,
    check_struct,
    format_plan_validation_feedback,
    #run_node_via_existing_executor,
    #run_plan_deterministically,
    run_tool_node,
    result_dict_to_prompt,
    extract_compound_names_llm,
    fetch_compound_card,
    fetch_compound_card_from_xyz,
    display_and_confirm_compound,
    compounds_to_planner_context,
    render_xyz_image_rdkit,
    auto_display_spectra,
    render_spectrum_image,
    render_pes_plot,
)
from build_graph_from_plan import build_graph_from_plan, build_state, expand_template
from prompts import SYSTEM_PROMPT, CALCULATOR_SYSTEM_PROMPT, REPORTER_SYSTEM_PROMPT
from skills import run_planning_skills, collect_skill_client_tools



load_dotenv(os.environ.get("ENV_FILE", ".env"))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or None

OPENAI_TOOLS = json.loads(Path("openai_tools_geom.json").read_text(encoding="utf-8"))
CLIENT_SIDE_TOOL_FUNCS = {
    "name_to_geometry_xyz": name_to_geometry_xyz,
    "build_dimer_xyz": build_dimer_xyz,
    "set_geometry_xyz": set_geometry_xyz,
    "build_approach_scan_geometries": build_approach_scan_geometries,
    "pubchem_get_basic_properties": pubchem_get_basic_properties,
    "structure_add_remove_proton": structure_add_remove_proton,
    # Deterministic tool functions declared by skills (e.g. PKaSkill's
    # compute_pka_calibrated) are installed here automatically — a new skill
    # that needs its own calculator never requires editing this file.
    **collect_skill_client_tools(),
}


class AgentState(TypedDict, total=False):
    # -------- session / planning data --------
    files: Dict[str, str]
    geometries: Dict[str, str]
    current_geom: Optional[str]
    name_to_geom: Dict[str, str]
    identifiers: Dict[str, Dict[str, Any]]
    geom_meta: Dict[str, Dict[str, Any]]
    cached_props: Dict[str, Dict[str, Any]]

    default_charge: int
    default_multiplicity: int

    latest_plan_name: Optional[str]
    approved_plan_name: Optional[str]
    plans: Dict[str, Dict[str, Any]]      # plan_name -> plan_pkg
    last_run_id: Optional[str]
    runs: List[Dict[str, Any]]
    last_user_text: str

    # -------- execution / graph bookkeeping not used--------
    run_log: List[Dict[str, Any]]
    last_status: str
    last_tool_result: Dict[str, Any]

    artifacts: Dict[str, Any]
    node_results: Dict[str, Any]

    final_report: Dict[str, Any]
    result: Any

def store_plan(state: AgentState, plan_pkg: dict) -> None:
    plan = plan_pkg.get("plan") or {}
    name = plan.get("name") or plan.get("title") or f"plan_{len(state.get('plans', {}))+1}"
    plan.setdefault("name", name)
    state.setdefault("plans", {})[name] = plan_pkg
    state["latest_plan_name"] = name

def approve_latest_plan(state: AgentState) -> None:
    state["approved_plan_name"] = state.get("latest_plan_name")

def tools_for_server(tool_names: list[str]) -> list[dict]:
    return [OPENAI_TOOLS[name] for name in tool_names if name in OPENAI_TOOLS]



def _record_usage(state: dict, role: str, resp) -> None:
    """Accumulate token usage from an OpenAI response into state['token_usage']."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    tu = state.setdefault("token_usage", {"planner": 0, "calculator": 0, "reporter": 0, "total": 0})
    tu[role]  = tu.get(role, 0)  + (usage.total_tokens or 0)
    tu["total"] = tu.get("total", 0) + (usage.total_tokens or 0)


def _print_token_usage(state: dict) -> None:
    tu = state.get("token_usage") or {}
    if not tu or tu.get("total", 0) == 0:
        return
    parts = [f"{k}={v}" for k, v in tu.items() if k != "total" and v]
    print(f"  [tokens] {' | '.join(parts)} | total={tu.get('total', 0)}")


async def handle_user_turn(session, client, state, user_text: str, tools_for_this_call: list,
                           compound_context: str = "", skill_contexts: list = [],
                           valid_tools: set = frozenset()):
    # Build message list: general prompt → skills → state → compound identity → user
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for skill_ctx in skill_contexts:
        messages.append({"role": "system", "content": "SKILL:\n" + skill_ctx})

    messages.append({"role": "system", "content": "STATE:\n" + summarize_geometries_prompt(state)})
    messages.append({"role": "system", "content": "WORKFLOW_STATE:\n" + summarize_workflow_state(state)})

    if compound_context:
        messages.append({"role": "system", "content": "CONFIRMED_COMPOUNDS:\n" + compound_context})

    messages.append({"role": "user", "content": user_text})
    state["last_user_text"] = user_text

    # MAX_TOOL_ROUNDS is the plan-validation retry budget: on structural-check
    # failure, the failed checks are fed back to the planner and it gets
    # another attempt, up to this many total planner calls. Matches the
    # paper's own precedent of "a corrected plan generated in a second
    # planner call" (the pKa Level A missing-opt-step catch), now automatic.
    MAX_TOOL_ROUNDS = 4
    plan = {}
    checks: Dict[str, bool] = {}
    retry_attempts = 0
    retry_fired    = False
    retry_stalled  = False
    prev_failed_signature = None
    for _round in range(MAX_TOOL_ROUNDS):
        #print("TOOL NAMES FOR MODEL:", tool_names_for_model)
        #print("TOOLS SENT:", [t["function"]["name"] for t in tools_for_this_call])
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools_for_this_call,
            tool_choice="none",
        )
        _record_usage(state, "planner", resp)
        msg = resp.choices[0].message
        messages.append(msg)
        raw = (msg.content or "").strip()
        print(raw)
        plan = parse_json_only(raw)

        expand_error = ""
        try:
            plan = expand_template(plan)
        except Exception as exc_expand:
            # Template expansion (e.g. duplicate node IDs from a missing
            # applies_to:"once") is itself a retryable planner mistake — catch
            # it here (before the human ever sees the plan) instead of letting
            # it surface later as a hard failure at execution time.
            expand_error = str(exc_expand)

        checks = ({"template_expansion_ok": False} if expand_error
                   else (check_struct(plan, valid_tools) if valid_tools else {}))
        if not checks or all(checks.values()):
            review = build_plan_review(plan)
            plan_pkg = {"plan": plan, "review": review, "ui": {"saved_files": {}},
                        "retry_attempts": retry_attempts, "retry_fired": retry_fired,
                        "retry_stalled": retry_stalled}
            store_plan(state, plan_pkg)
            print_plan_review(plan_pkg)
            return plan_pkg

        retry_attempts = _round + 1
        failed_signature = frozenset(k for k, v in checks.items() if not v)

        if failed_signature == prev_failed_signature:
            # Same failure as last attempt — not converging, stop early
            # rather than spending the rest of the retry budget on a loop
            # that isn't making progress.
            retry_stalled = True
            print(f"  [plan_retry] attempt={_round+1} "
                  f"failed={sorted(failed_signature)} "
                  f"-- identical to previous attempt, stopping early (not converging)")
            break

        print(f"  [plan_retry] attempt={_round+1} "
              f"failed={sorted(failed_signature)}"
              + (f" error={expand_error}" if expand_error else ""))
        if _round < MAX_TOOL_ROUNDS - 1:
            retry_fired = True
            prev_failed_signature = failed_signature
            messages.append({"role": "user",
                              "content": format_plan_validation_feedback(checks, expand_error)})

    # Retries either exhausted or stalled (same failure recurred) and the plan
    # still fails structural validation. Fail-open, not fail-silent: return
    # the plan (the existing "Execute this plan? (y/N)" human confirmation is
    # still the final safety net) but make the failure impossible to miss in
    # the review the human sees.
    review = build_plan_review(plan)
    reason = "the same failure recurred" if retry_stalled else f"{MAX_TOOL_ROUNDS} attempts were exhausted"
    review["validation_warning"] = (
        f"WARNING: This plan failed automatic structural validation "
        f"({reason}): {[k for k, v in checks.items() if not v]}"
    )
    plan_pkg = {"plan": plan, "review": review, "ui": {"saved_files": {}},
                "retry_attempts": retry_attempts, "retry_fired": retry_fired,
                "retry_stalled": retry_stalled}
    store_plan(state, plan_pkg)
    print_plan_review(plan_pkg)
    print(review["validation_warning"])
    return plan_pkg


def _cache_confirmed_card(state: AgentState, name: str, card: dict) -> None:
    """Store a confirmed compound card in state caches."""
    state.setdefault("cached_props", {})[name] = card
    if card.get("inchi") or card.get("smiles"):
        state.setdefault("identifiers", {})[name] = {
            "smiles": card.get("smiles"),
            "inchi": card.get("inchi"),
            "inchikey": card.get("inchikey"),
            "formula": card.get("formula"),
            "charge": card.get("charge", 0),
        }


_COMPOUND_MODES = ("full", "name", "smiles", "xyz")


async def identify_and_confirm_compounds(
    user_text: str, client, state: AgentState, mode: str = "full"
) -> list:
    """Pre-planning phase: identify all compounds (by name and/or loaded geometries).

    mode:
      "full"   (default) — LLM extracts names → PubChem → structure image → user confirms.
                           CONFIRMED_COMPOUNDS includes formula, SMILES, MW, CID, charge.
      "name"   — LLM extracts names only; no PubChem lookup, no image, no confirmation.
                 CONFIRMED_COMPOUNDS has name + charge only.
      "smiles" — LLM extracts names → PubChem to get SMILES; no image, auto-confirmed.
                 CONFIRMED_COMPOUNDS has name + SMILES + charge.
      "xyz"    — Skip all identification. Planner reasons from geometry state only.
                 CONFIRMED_COMPOUNDS is empty.
    """
    if mode == "xyz":
        return []

    print(f"\n[Identifying compounds... mode={mode}]")
    confirmed = []

    # --- Source 1: names mentioned in the planning request ---
    names = extract_compound_names_llm(user_text, client)
    for name in names:
        if mode == "name":
            card = {"name": name, "formula": None, "smiles": None,
                    "mw": None, "cid": None, "charge": 0, "source": "name_only"}
            print(f"  {name}  (name only)")
            confirmed.append(card)
            _cache_confirmed_card(state, name, card)
        elif mode == "smiles":
            card = fetch_compound_card(name)
            smiles_info = f"SMILES={card['smiles']}" if card.get("smiles") else "SMILES not found"
            print(f"  {name}: {smiles_info}")
            confirmed.append(card)
            _cache_confirmed_card(state, name, card)
        else:  # "full"
            card = fetch_compound_card(name)
            result = display_and_confirm_compound(card, client)
            if result is None:
                continue
            confirmed.append(result)
            _cache_confirmed_card(state, name, result)

    # --- Source 2: geometries loaded in state with no cached identity ---
    if mode != "name":  # name mode skips geometry identification
        geometries = state.get("geometries") or {}
        identifiers = state.get("identifiers") or {}
        cached_props = state.get("cached_props") or {}
        geom_meta = state.get("geom_meta") or {}

        unidentified = [
            gid for gid in geometries
            if gid not in identifiers and gid not in cached_props
        ]

        if unidentified and mode == "smiles":
            print(f"\n[{len(unidentified)} loaded geometry/geometries — deriving SMILES]")
        elif unidentified and mode == "full":
            print(f"\n[{len(unidentified)} loaded geometry/geometries not yet identified]")

        for geom_id in unidentified:
            xyz = geometries[geom_id]
            charge = (geom_meta.get(geom_id) or {}).get("charge", 0) or 0

            card = fetch_compound_card_from_xyz(xyz, geom_id, charge=charge)

            if mode == "smiles":
                if card.get("smiles"):
                    print(f"  Loaded {geom_id}: SMILES={card['smiles']}")
                confirmed.append(card)
                _cache_confirmed_card(state, geom_id, card)
            else:  # "full"
                print(f"\n── Loaded geometry: {geom_id} ──")
                if card.get("smiles"):
                    print(f"  Detected SMILES : {card['smiles']}")
                if card.get("formula"):
                    print(f"  Detected formula: {card['formula']}")
                result = display_and_confirm_compound(card, client)
                if result is None:
                    continue
                confirmed.append(result)
                _cache_confirmed_card(state, geom_id, result)

    if not confirmed:
        print("[No compounds identified — proceeding to planning with geometry state only]")

    return confirmed


def _build_mcp_server_params():
    """Build StdioServerParameters for the MCP server.

    All connection settings come from the active env file (ENV_FILE, default
    .env). No personal key paths, hosts, or server commands are baked into
    source — the SSH-specific values must be supplied by the env file.

    MCP_MODE=local  — spawn the server as a local subprocess (no SSH).
                      Set MCP_SERVER_CMD (default 'python server_with_product.py').
    MCP_MODE=ssh    — (default) connect via SSH, controlled by
                      MCP_SSH_BIN / MCP_SSH_KEY / MCP_SSH_HOST / MCP_SERVER_CMD.
    """
    from mcp import StdioServerParameters

    def _require(name: str) -> str:
        val = os.getenv(name, "").strip()
        if not val:
            raise RuntimeError(
                f"{name} is not set. Define it in your env file "
                f"(ENV_FILE={os.environ.get('ENV_FILE', '.env')})."
            )
        return val

    _mode = os.getenv("MCP_MODE", "ssh").strip().lower()
    if _mode == "local":
        _cmd = os.getenv("MCP_SERVER_CMD", "python server_with_product.py")
        parts = _cmd.split()
        return StdioServerParameters(
            command=parts[0],
            args=parts[1:],
            env=dict(os.environ),
        )
    # ssh mode (default) — SSH target/key/command come only from the env file
    _ssh_bin  = os.getenv("MCP_SSH_BIN", "ssh")
    _ssh_key  = _require("MCP_SSH_KEY")
    _ssh_host = _require("MCP_SSH_HOST")
    _ssh_cmd  = _require("MCP_SERVER_CMD")
    return StdioServerParameters(
        command=_ssh_bin,
        args=["-i", _ssh_key, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _ssh_host, _ssh_cmd],
        env=dict(os.environ),  # MCP's default env filter strips vars SSH needs
    )


async def main():
    parser = argparse.ArgumentParser(description="QC Agent REPL")
    parser.add_argument(
        "--compound-mode",
        choices=_COMPOUND_MODES,
        default="full",
        help=(
            "Compound identification mode for contrast experiments. "
            "'full' (default): name + PubChem + image + user confirm. "
            "'name': name only, no lookup. "
            "'smiles': name + SMILES, no image. "
            "'xyz': skip identification, planner reasons from geometry state only."
        ),
    )
    args = parser.parse_args()
    compound_mode: str = args.compound_mode

    server_params = _build_mcp_server_params()

    _client_kwargs = {"base_url": _LLM_BASE_URL} if _LLM_BASE_URL else {}
    client = OpenAI(**_client_kwargs)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            state: AgentState = {
    "files": {},
    "geometries": {},
    "name_to_geom": {},
    "identifiers": {},
    "geom_meta": {},
    "cached_props": {},
    "plans": {},
    "runs": [],
    "run_log": [],
    "artifacts": {},
    "node_results": {},
    "default_charge": 0,
    "default_multiplicity": 1,
    "token_usage": {"planner": 0, "calculator": 0, "reporter": 0, "total": 0},
            }

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            # Authoritative "is this a real tool" set for the plan-retry loop's
            # structural validator — every MCP tool plus every client-side tool
            # (including skill-owned ones, via CLIENT_SIDE_TOOL_FUNCS), not the
            # OPENAI_TOOLS-filtered subset below (that filtering is unrelated:
            # it's for the native tool-calling schema, which the planner never
            # actually invokes since tool_choice="none").
            VALID_TOOLS = set(tool_names) | set(CLIENT_SIDE_TOOL_FUNCS.keys())
            print("MCP tools available:", tool_names)
            print(f"Compound mode: {compound_mode}  "
                  f"(change with: mode full|name|smiles|xyz)")
            tool_names_for_model = [n for n in tool_names if n in OPENAI_TOOLS]
            for n in CLIENT_SIDE_TOOL_FUNCS:
                if n in OPENAI_TOOLS and n not in tool_names_for_model:
                    tool_names_for_model.append(n)

            tools_for_this_call = [OPENAI_TOOLS[n] for n in tool_names_for_model]

            while True:
                line = input("> ").strip()
                if not line:
                    continue
                if line.lower() in {"quit", "exit"}:
                    break

                if line.lower().startswith("mode "):
                    parts = line.split()
                    if len(parts) == 2 and parts[1] in _COMPOUND_MODES:
                        compound_mode = parts[1]
                        print(f"Compound mode set to: {compound_mode}")
                    else:
                        print(f"Usage: mode <{'|'.join(_COMPOUND_MODES)}>  "
                              f"(current: {compound_mode})")
                    continue

                if line.lower().startswith("load "):
                    path = line.split(maxsplit=1)[1]
                    geometry_xyz = load_xyz_as_geometry(path)
                    key = geom_key_from_path(path)
                    state.setdefault("files", {})[key] = path
                    state.setdefault("geometries", {})[key] = geometry_xyz
                    state["current_geom"] = key
                    print(f"Loaded geometry from {path!r} as {key!r}")
                    charge = (state.get("geom_meta") or {}).get(key, {}).get("charge", 0) or 0
                    img_path = render_xyz_image_rdkit(geometry_xyz, key, charge=charge)
                    if img_path:
                        print(f"Structure: {img_path} [opened]")
                    else:
                        print("Structure: (could not render — check RDKit / XYZ format)")
                    continue

                if line.strip().lower() == "state":
                    print_session_state(state, preview_lines=6, show_full_current=True, max_full_lines=None)
                    continue

                if line.strip().lower().startswith("planimg"):
                    handle_planimg_command(state, line)
                    continue

                if line.strip().lower().startswith("plansave"):
                    handle_plansave_command(state, line)
                    continue

                if line.strip().lower().startswith("planload"):
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: planload <path-to-plan.json>")
                        continue
                    plan_path = Path(parts[1].strip())
                    if not plan_path.exists():
                        print(f"File not found: {plan_path}")
                        continue
                    try:
                        plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"Failed to load plan: {e}")
                        continue
                    review = build_plan_review(plan)
                    plan_pkg = {"plan": plan, "review": review, "ui": {"saved_files": {}}}
                    store_plan(state, plan_pkg)
                    print(f"Loaded plan from {plan_path}")
                    print_plan_review(plan_pkg)
                    continue
                
                
                if line.strip().lower() == "showspec":
                    arts = state.get("last_artifacts") or {}
                    spec_keys = [
                        k for k in arts
                        if k.lower().startswith("ir_spectrum")
                        or k.lower().startswith("excited_states")
                        or k.lower().startswith("scan_results")
                    ]
                    if not spec_keys:
                        print("No spectrum/PES artifacts found. Run a calculation first.")
                    else:
                        for k in spec_keys:
                            v = arts.get(k)
                            if isinstance(v, list) and v:
                                kl = k.lower()
                                if kl.startswith("ir_spectrum"):
                                    mol = k[len("ir_spectrum_"):] if kl.startswith("ir_spectrum_") else "molecule"
                                    path = render_spectrum_image(v, "ir", f"IR_{mol}")
                                elif kl.startswith("excited_states"):
                                    mol = k[len("excited_states_"):] if kl.startswith("excited_states_") else "molecule"
                                    path = render_spectrum_image(v, "uvvis", f"UVVis_{mol}")
                                else:
                                    scan_lbl = k[len("scan_results_"):] if kl.startswith("scan_results_") else "scan"
                                    path = render_pes_plot(v, f"PES_{scan_lbl}")
                                if path:
                                    print(f"[spectrum] {k} → {path}  [opened]")
                    continue

                if line.strip().lower() == "run":
                    # 1) pick plan (prefer approved)
                    plan_name = state.get("approved_plan_name") or state.get("latest_plan_name")
                    plan_pkg = (state.get("plans") or {}).get(plan_name)

                    if not plan_pkg:
                        print("No plan found. Use 'plan' first.")
                        continue

                    # 2) show review
                    print_plan_review(plan_pkg)
                    yn = input("Execute this plan? (y/N): ").strip().lower()
                    if yn not in ("y", "yes"):
                        print("Cancelled.")
                        continue


                    # 3) execute
                    graph = build_graph_from_plan(plan_pkg["plan"], get_tool_args=state_get_tool_args,
                             run_tool_node=run_tool_node, openai_client=client)

                    init_state = build_state(plan_pkg["plan"], session, seed={
                        "client_side_tools": CLIENT_SIDE_TOOL_FUNCS,  # optional but recommended
                    },client_side_tools=CLIENT_SIDE_TOOL_FUNCS)
                    result = await graph.ainvoke(init_state)   # or graph.invoke(...)
                    print("Done:", result.get("status", "unknown"))
                    pprint(result.get("artifacts"))
                    state["last_artifacts"] = result.get("artifacts") or {}
                    # Merge calculator token usage from LangGraph result back into REPL state
                    _result_tu = result.get("token_usage") or {}
                    if _result_tu:
                        _tu = state.setdefault("token_usage", {"planner": 0, "calculator": 0, "reporter": 0, "total": 0})
                        for _k, _v in _result_tu.items():
                            _tu[_k] = _tu.get(_k, 0) + _v
                    auto_display_spectra(state["last_artifacts"])
                    user_payload = (
                        f"Original user request:\n{state.get('last_user_text','(not provided)')}\n\n"
                        f"Run result:\n{result_dict_to_prompt(result=result, user_text=state.get('last_user_text'))}"
                    )
                    report_messages = [
                        {"role": "system", "content": REPORTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ]
                    
                    resp = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=report_messages,
                    )
                    _record_usage(state, "reporter", resp)
                    msg = resp.choices[0].message
                    print(msg.content)
                    _print_token_usage(state)
                    #print(json.dumps(result, ensure_ascii=False, indent=2))
                    continue
                
                compounds = await identify_and_confirm_compounds(line, client, state, mode=compound_mode)
                compound_context = compounds_to_planner_context(compounds, mode=compound_mode)
                skill_contexts = run_planning_skills(line, state)
                await handle_user_turn(session, client, state, line, tools_for_this_call,
                                       compound_context=compound_context,
                                       skill_contexts=skill_contexts,
                                       valid_tools=VALID_TOOLS)

if __name__ == "__main__":
    asyncio.run(main())
