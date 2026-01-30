from __future__ import annotations

import asyncio
from typing import TypedDict, Any, Dict, Callable, List, Awaitable, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import ast
import math
import json
import time
from datetime import datetime, timezone
from orchestrator_reporter import report_bug_and_persist, report_runtime_and_persist
from typing import Tuple

class State(TypedDict, total=False):
    session: Any
    # Optional registry of purely local tools (not exposed by the MCP server).
    # If present, the deterministic tool runner can resolve e.g. name_to_geometry_xyz
    # without going through the session.
    client_side_tools: Dict[str, Any]
    run_id: str
    run_started_utc: str
    run_finished_utc: str
    last_status: str
    last_tool_result: Dict[str, Any]
    run_log: List[dict]

    artifacts: Dict[str, Any]
    node_results: Dict[str, Any]

    # GeometryRegistry-backed storage (dict-state)
    geometries: Dict[str, str]
    geom_meta: Dict[str, Any]
    name_to_geom: Dict[str, str]
    current_geom: Optional[str]
    final_report: Dict[str, Any]
    result: Any



def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


ToolCaller = Callable[[str, Dict[str, Any]], Dict[str, Any]]
LLMCaller = Callable[[str, State, Dict[str, Any]], Dict[str, Any]]

# New-style node runners (can wrap an LLM+tool-loop executor)
ToolNodeRunner = Callable[[State, Dict[str, Any]], "ToolNodeResult"]
AsyncToolNodeRunner = Callable[[State, Dict[str, Any]], Awaitable["ToolNodeResult"]]

ToolNodeResult = Dict[str, Any]


ALLOWED_FUNCS = {
    "log": math.log,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "abs": abs,
}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Name, ast.Constant,
    ast.Call, ast.Load,
)


def build_state(
    plan: Dict[str, Any],
    session: Any,
    *,
    seed: Optional[Dict[str, Any]] = None,
    client_side_tools: Optional[Dict[str, Any]] = None,
) -> State:
    """Build an initial LangGraph state from a plan JSON.

    Key behavior:
    - Pre-seed state['artifacts'] with all artifact keys implied by the plan.
      This lets executors/UI treat absent values as "pending" rather than
      missing keys.
    - Does NOT overwrite any non-None artifact values present in `seed`.
    """
    st: Dict[str, Any] = dict(seed or {})
    st["session"] = session
    st["plan"] = plan

    # If callers don't pass a seed containing client-side tools, allow passing them
    # explicitly via build_state(..., client_side_tools=...).
    if "client_side_tools" not in st and client_side_tools is not None:
        st["client_side_tools"] = client_side_tools

    st.setdefault("run_log", [])
    st.setdefault("node_results", {})
    st.setdefault("last_status", "init")
    st.setdefault("last_tool_result", {})
    st.setdefault("final_report", {})

    # Collect artifact keys from multiple plan sections
    artifact_keys: set[str] = set()
    artifact_keys.update(plan.get("artifacts_to_save") or [])

    fr = plan.get("final_report") or {}
    artifact_keys.update(fr.get("fields") or [])

    summ = plan.get("summarizer") or {}
    artifact_keys.update(summ.get("report_fields") or [])

    by_node: Dict[str, Any] = {}
    for n in (plan.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not nid:
            continue

        product = n.get("product") or {}
        if isinstance(product, dict):
            artifact_keys.update(product.keys())

        expect = n.get("expect") or {}
        if isinstance(expect, dict):
            artifact_keys.update(expect.get("artifacts_required") or [])

        if n.get("kind") == "tool":
            by_node[nid] = {
                "tool": n.get("tool"),
                "artifacts_required": (expect.get("artifacts_required") or []) if isinstance(expect, dict) else [],
                "properties_required": (expect.get("properties_required") or []) if isinstance(expect, dict) else [],
                "product_keys": list(product.keys()) if isinstance(product, dict) else [],
            }

    artifacts = dict(st.get("artifacts") or {})
    for k in sorted(artifact_keys):
        artifacts.setdefault(k, None)
    st["artifacts"] = artifacts
    st.setdefault("artifacts_spec", {"artifact_keys": sorted(artifact_keys), "by_node": by_node})

    return st  # type: ignore[return-value]


def compile_expr(expr: str) -> Callable[[Dict[str, float]], float]:
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("Disallowed function call")
        if isinstance(node, ast.Name):
            # Names are validated at runtime via the env dict;
            # this just blocks dunder / attribute tricks.
            if node.id.startswith("__"):
                raise ValueError("Disallowed name")

    code = compile(tree, "<expr>", "eval")

    def f(env: Dict[str, float]) -> float:
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(ALLOWED_FUNCS)
        safe_locals.update(env)
        return float(eval(code, safe_globals, safe_locals))

    return f

def build_graph_from_plan(
    plan: Dict[str, Any],
    call_tool: Optional[ToolCaller] = None,
    call_llm_task: Optional[LLMCaller] = None,
    get_tool_args: Optional[Callable[[Any, str, Dict[str, Any]], Dict[str, Any]]] = None,
    run_tool_node: Optional[ToolNodeRunner | AsyncToolNodeRunner] = None,
    run_llm_node: Optional[ToolNodeRunner | AsyncToolNodeRunner] = None,
):
    plan_nodes: List[Dict[str, Any]] = plan.get("nodes") or []
    if not plan_nodes:
        raise ValueError("plan['nodes'] is empty")

    nodes_by_id = {n["id"]: n for n in plan_nodes if n.get("id")}
    if not nodes_by_id:
        raise ValueError("plan has no nodes with 'id' fields")

    ordered_ids = [n["id"] for n in plan_nodes if n.get("id") in nodes_by_id]
    entry = plan.get("entry") or ordered_ids[0]

    FINAL_NODE_ID = "__final_report__"

    # sequential fallback routing (when spec lacks 'next')
    next_in_order: Dict[str, str] = {}
    for i, nid in enumerate(ordered_ids):
        next_in_order[nid] = ordered_ids[i + 1] if (i + 1) < len(ordered_ids) else FINAL_NODE_ID

    artifacts_to_save = set(plan.get("artifacts_to_save") or [])

    async def _maybe_await(v: Any) -> Any:
        if asyncio.iscoroutine(v):
            return await v
        return v

    def _ensure_state_defaults(state: State) -> Dict[str, Any]:
        upd: Dict[str, Any] = {}
        if "run_id" not in state or not state.get("run_id"):
            upd["run_id"] = _utc_run_id()
        if "run_started_utc" not in state or not state.get("run_started_utc"):
            upd["run_started_utc"] = datetime.now(timezone.utc).isoformat()
        if "run_log" not in state:
            upd["run_log"] = []
        if "artifacts" not in state or not isinstance(state.get("artifacts"), dict):
            upd["artifacts"] = {}
        if "node_results" not in state or not isinstance(state.get("node_results"), dict):
            upd["node_results"] = {}
        return upd

    def _get_by_path(obj: Any, path: Any) -> Any:
        if obj is None or path is None:
            return None
        cur = obj
        if isinstance(path, str):
            parts = [p for p in path.split(".") if p]
        elif isinstance(path, list):
            parts = path
        else:
            parts = [path]

        for p in parts:
            if cur is None:
                return None
            if isinstance(p, int):
                if isinstance(cur, list) and 0 <= p < len(cur):
                    cur = cur[p]
                else:
                    return None
            else:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
        return cur

    def _summarize_value_for_log(v: Any) -> Any:
        """Keep run_log compact + JSON-safe."""
        if v is None or isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, str):
            if len(v) <= 200:
                return v
            return v[:200] + f"...<len={len(v)}>"
        if isinstance(v, dict):
            ks = list(v.keys())
            return {"_type": "dict", "n_keys": len(ks), "keys_head": ks[:20]}
        if isinstance(v, list):
            return {"_type": "list", "len": len(v)}
        return {"_type": type(v).__name__}

    def _stash_artifacts(node_id: str, spec: Dict[str, Any], result: Dict[str, Any], state: State) -> Dict[str, Any]:
        upd: Dict[str, Any] = {}
        upd.update(_ensure_state_defaults(state))

        node_results = dict(state.get("node_results", {}))
        node_results[node_id] = result
        result = result.get("last_tool_result")
        upd["node_results"] = node_results

        artifacts = dict(state.get("artifacts", {}))

        # Convention: tools may return a primary output via a "product" field.
        # Example: {"status":"ok","product":"energy","energy":-76.32,...}
        product = result.get("product")#product key of result dict
        product_spec = spec.get("product") or {}# artifact key to store the product
        #print("product:", product,"result:", result, "product_spec:",product_spec)
        if isinstance(product_spec, dict):
            for out_key, src in product_spec.items():
                v = _get_by_path(result, product)
                if v is not None:
                    artifacts[out_key] = v

        upd["artifacts"] = artifacts
        #print("artifacts:", artifacts)
        return upd
    
    def _set_by_path_root_update(state: State, path: str, value: Any) -> Dict[str, Any]:
        if not path or not isinstance(path, str):
            return {}
        parts = [p for p in path.split(".") if p]
        if not parts:
            return {}
        if len(parts) == 1:
            return {parts[0]: value}

        root = parts[0]
        tail = parts[1:]

        root_obj = state.get(root)
        if not isinstance(root_obj, dict):
            root_obj = {}
        new_root = dict(root_obj)
        cur = new_root
        for p in tail[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
            nxt = dict(nxt)
            cur[p] = nxt
            cur = nxt
        cur[tail[-1]] = value
        return {root: new_root}
    
    def _compile_final_report(state: State) -> Dict[str, Any]:
        fr = plan.get("final_report") or {}
        collect_from = fr.get("collect_from") or ordered_ids
        fields = fr.get("fields") or []

        node_results = state.get("node_results", {}) or {}
        artifacts = state.get("artifacts", {}) or {}

        report: Dict[str, Any] = {}
        for field in fields:
            val = artifacts.get(field)
            if val is None:
                for nid in collect_from:
                    res = node_results.get(nid)
                    if not isinstance(res, dict):
                        continue
                    if field in res:
                        val = res[field]
                        break
                    props = res.get("properties")
                    if isinstance(props, dict) and field in props:
                        val = props[field]
                        break
                    art = res.get("artifacts")
                    if isinstance(art, dict) and field in art:
                        val = art[field]
                        break
            report[field] = val

        report.setdefault("status", state.get("last_status", "ok"))
        report.setdefault("run_log", state.get("run_log", []))
        return report

    builder = StateGraph(State)

    async def _maybe_await(x: Any) -> Any:
        # LangGraph nodes may be sync or async; we support both runner styles.
        if asyncio.iscoroutine(x):
            return await x
        return x

    def make_node(node_id: str):
        spec = dict(nodes_by_id[node_id])

        async def _node(state: State) -> Command:
            updates: Dict[str, Any] = {}
            updates.update(_ensure_state_defaults(state))

            node_started_utc = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()

            try:
                kind = (spec.get("kind") or "tool")
                if kind == "llm_task":
                    kind = "llm"
                status = "ok"

                if kind == "tool":
                    tool_name = spec["tool"]
                    # Prefer plan schema "args"; keep "overrides" for backward compatibility.
                    node_args = spec.get("args")
                    if node_args is None:
                        node_args = spec.get("overrides")
                    node_args = node_args or {}

                    if run_tool_node is not None:
                        # Delegate to your existing OpenAI-tool-loop executor (LLM decides tool calls),
                        # or any higher-level runner you provide.
                        node_spec = dict(spec)
                        node_spec["id"] = node_id
                        result = await _maybe_await(run_tool_node(state, node_spec))
                        if not isinstance(result, dict):
                            result = {"status": "error", "error": "tool node runner returned non-dict"}
                    else:
                        args = get_tool_args(state["session"], tool_name, node_args)
                        result = call_tool(tool_name, args) or {}
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except Exception:
                            result = {"status": "error", "error": "tool returned non-JSON string", "raw": result}

                    status = (result.get("status") or "error")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    tool_payload = result.get("last_tool_result", result)
                    status = result.get("last_status", tool_payload.get("status") if isinstance(tool_payload, dict) else None) or "unknown"
                    #print("result:", result)

                    # 1) merge the returned state updates first (this is what makes geometry persistent)
                    updates.update(result)

                    # 2) ensure per-node node_results stores the tool payload (not the whole state update dict)
                    node_results = dict(state.get("node_results") or {})
                    node_results[node_id] = tool_payload
                    updates["node_results"] = node_results

                    # 3) standard bookkeeping
                    updates.update({
                        "last_status": status,
                        "last_tool_result": tool_payload,
                        "run_log": (state.get("run_log") or []) + [{
                            "node": node_id,
                            "kind": kind,
                            "tool": tool_name,
                            "status": status,
                            "started_utc": node_started_utc,
                            "ended_utc": ended_utc,
                            "duration_ms": duration_ms,
                        }],
                    })
                    if isinstance(result, dict):
                        updates.update(_stash_artifacts(node_id, spec, result, state))
                elif kind == "llm":
                    task = spec.get("task") or spec.get("pattern") or "llm_task"
                    if run_llm_node is not None:
                        node_spec = dict(spec)
                        node_spec["id"] = node_id
                        out = await _maybe_await(run_llm_node(state, node_spec))
                        if not isinstance(out, dict):
                            out = {"status": "error", "error": "llm node runner returned non-dict"}
                    else:
                        out = call_llm_task(task, state, spec) or {}

                    status = out.get("status", "ok")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    updates["last_status"] = status
                    updates["run_log"] = state.get("run_log", []) + [{
                        "node": node_id,
                        "kind": kind,
                        "llm_task": task,
                        "status": status,
                        "started_utc": node_started_utc,
                        "ended_utc": ended_utc,
                        "duration_ms": duration_ms,
                    }]
                    if isinstance(out, dict):
                        updates.update(_stash_artifacts(node_id, spec, result, state))
                elif kind in ("calc_expr", "expr", "calc"):
                    expr = spec.get("expr")
                    inputs = spec.get("inputs") or {}
                    constants = spec.get("constants") or {}
                    output_key = spec.get("output_key")  # optional, e.g. "results.pKa"

                    result: Dict[str, Any] = {"status": "error"}
                    try:
                        if not isinstance(expr, str) or not expr.strip():
                            raise ValueError("missing expr")

                        env: Dict[str, float] = {}

                        if not isinstance(inputs, dict):
                            raise ValueError("inputs must be a dict")
                        for sym, path in inputs.items():
                            v = _get_by_path(state, path)
                            if v is None:
                                raise KeyError(f"input {sym!r} not found at path {path!r}")
                            env[str(sym)] = float(v)

                        if not isinstance(constants, dict):
                            raise ValueError("constants must be a dict")
                        for k, v in constants.items():
                            env[str(k)] = float(v)

                        fn = compile_expr(expr)
                        value = fn(env)

                        result = {"status": "ok", "expr": expr, "value": value, "env": env}

                        if isinstance(output_key, str) and output_key.strip():
                            updates.update(_set_by_path_root_update(state, output_key, value))

                    except Exception as e:
                        result = {"status": "error", "error": str(e), "expr": expr, "inputs": inputs}

                    status = (result.get("status") or "error")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    updates.update({
                        "last_status": status,
                        "last_tool_result": result,
                        "run_log": state.get("run_log", []) + [{
                            "node": node_id,
                            "kind": kind,
                            "status": status,
                            "started_utc": node_started_utc,
                            "ended_utc": ended_utc,
                            "duration_ms": duration_ms,
                        }],
                    })
                    updates.update(_stash_artifacts(node_id, spec, result, state))

                else:
                    status = "error"

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    updates["last_status"] = status
                    updates["run_log"] = state.get("run_log", []) + [{
                        "node": node_id,
                        "kind": kind,
                        "status": status,
                        "error": f"unknown kind: {kind}",
                        "started_utc": node_started_utc,
                        "ended_utc": ended_utc,
                        "duration_ms": duration_ms,
                    }]

                # routing
                if "next" in spec:
                    nxt = spec.get("next")
                    if isinstance(nxt, dict):
                        goto = nxt.get(status) or nxt.get("default") or next_in_order.get(node_id, FINAL_NODE_ID)
                    else:
                        goto = nxt
                else:
                    goto = next_in_order.get(node_id, FINAL_NODE_ID)

                # always run FINAL_REPORT if present/configured
                if goto in ("END", END, None):
                    goto = FINAL_NODE_ID

                if isinstance(goto, str) and goto not in nodes_by_id and goto != FINAL_NODE_ID:
                    updates["last_status"] = "error"
                    updates["run_log"] = state.get("run_log", []) + [{"node": node_id, "status": "error", "error": f"unknown next node: {goto!r}"}]
                    goto = FINAL_NODE_ID

                return Command(update=updates, goto=goto)

            except Exception as e:
                # Unexpected runtime exception: persist a bug report and force-finalize.
                node_spec = dict(spec)
                node_spec["id"] = node_id

                report: Dict[str, Any]
                json_path: Optional[str]
                md_path: Optional[str]

                try:
                    report, json_path, md_path = report_bug_and_persist(
                        node_id=node_id,
                        node_spec=node_spec,
                        state=state,
                        error=e,
                        stage="node_execution",
                    )
                except Exception as e2:
                    report = {
                        "type": "orchestrator_bug_reporter_failed",
                        "node_id": node_id,
                        "original_error": str(e),
                        "reporter_error": str(e2),
                    }
                    json_path, md_path = None, None

                # Make sure artifacts exist and stash paths.
                artifacts = state.get("artifacts")
                if not isinstance(artifacts, dict):
                    artifacts = {}
                else:
                    artifacts = dict(artifacts)

                artifacts["bug_report"] = report
                if json_path:
                    artifacts["bug_report_json_path"] = json_path
                if md_path:
                    artifacts["bug_report_md_path"] = md_path

                updates.update(_ensure_state_defaults(state))
                updates["artifacts"] = artifacts
                updates["last_status"] = "error"
                updates["last_tool_result"] = {"status": "error", "error": str(e)}

                ended_utc = datetime.now(timezone.utc).isoformat()
                duration_ms = int((time.perf_counter() - t0) * 1000)
                updates["run_log"] = state.get("run_log", []) + [{
                    "node": node_id,
                    "kind": spec.get("kind", "tool"),
                    "status": "error",
                    "error": str(e),
                    "bug_report_json_path": json_path,
                    "bug_report_md_path": md_path,
                    "started_utc": node_started_utc,
                    "ended_utc": ended_utc,
                    "duration_ms": duration_ms,
                }]

                return Command(update=updates, goto=FINAL_NODE_ID)

        return _node

    for node_id in ordered_ids:
        builder.add_node(node_id, make_node(node_id))

    def _final_report_node(state: State) -> Command:
        updates: Dict[str, Any] = {}
        updates.update(_ensure_state_defaults(state))

        # Mark end-of-run timestamp (used by runtime report writer).
        if not state.get("run_finished_utc"):
            updates["run_finished_utc"] = datetime.now(timezone.utc).isoformat()

        report = _compile_final_report(state)
        updates["final_report"] = report
        updates["result"] = report
        updates["last_status"] = report.get("status", state.get("last_status", "ok"))

        # Persist a full runtime report (JSON + Markdown) for the entire graph.
        merged_state = dict(state)
        merged_state.update(updates)
        merged_state["final_report"] = report

        rt_report: Dict[str, Any]
        rt_json_path: Optional[str]
        rt_md_path: Optional[str]
        try:
            rt_report, rt_json_path, rt_md_path = report_runtime_and_persist(
                plan=plan,
                state=merged_state,
                final_report=report,
                stage="graph_complete",
            )
        except Exception as e:
            rt_report = {"type": "orchestrator_runtime_report_failed", "error": str(e)}
            rt_json_path, rt_md_path = None, None

        artifacts = merged_state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        else:
            artifacts = dict(artifacts)

        artifacts["runtime_report"] = rt_report
        if rt_json_path:
            artifacts["runtime_report_json_path"] = rt_json_path
        if rt_md_path:
            artifacts["runtime_report_md_path"] = rt_md_path

        updates["artifacts"] = artifacts
        updates["run_log"] = (state.get("run_log", []) or []) + [{
            "node": FINAL_NODE_ID,
            "kind": "final",
            "status": "ok",
            "runtime_report_json_path": rt_json_path,
            "runtime_report_md_path": rt_md_path,
        }]
        return Command(update=updates, goto=END)

    builder.add_node(FINAL_NODE_ID, _final_report_node)
    builder.add_edge(START, entry)

    return builder.compile()
