"""Guardrails for plan JSON shape and tool-call validity.

A single, pure entry point — `validate_plan(plan)` — returns a list of
`Violation`s without raising. It is the one source of truth used in three places:

  1. Runtime gate:   build_graph_from_plan can refuse to execute a plan whose
                     validation has ERROR-level violations, before any ORCA job
                     spends wall-clock time.
  2. FT reward:      finetune/harvest_dataset.py can keep only plans that pass
                     validation (rejection sampling).
  3. Constrained gen: the same rules describe the grammar a fine-tuned or
                     schema-constrained planner must satisfy.

What it checks
--------------
Plan shape
  - required top-level keys; `nodes` non-empty
  - every node has a unique `id` and a known `kind` ∈ {tool, llm, calc_expr}
  - `geom_ids` / `artifacts_to_save` are lists of strings
Tool-call validity (against openai_tools_geom.json)
  - tool name exists
  - no hallucinated args (not in the tool's JSON schema, excluding injected/MCP params)
  - planner-supplied required args present (warning — many are executor-injected)
Project invariants (from CLAUDE.md + hard-won runtime lessons)
  - `ncores`, if present, must be 1 (MPI aborts under kubectl exec)
  - charge / multiplicity must NOT be hardcoded in tool args (executor injects them)
  - `scan_coords` must be a JSON *string*, not a list (survives MCP serialisation)
  - product mappings exist; artifacts_to_save are actually produced (warning)

Severity: ERROR blocks execution / rejects from the dataset; WARN is advisory.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from typing import Any

_SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "openai_tools_geom.json")

VALID_NODE_KINDS = frozenset({"tool", "llm", "calc_expr"})

# Client-side tools handled locally (client_helpers.py), not via MCP, and
# deliberately absent from openai_tools_geom.json. Known-valid, so not "unknown".
CLIENT_SIDE_TOOLS = frozenset({
    "pubchem_get_basic_properties", "state_update", "state_get_tool_args",
})

# Args the executor injects from session state or strips before the MCP call,
# so the planner is NOT expected to provide them and they are NOT hallucinations.
_INJECTED_ARGS = frozenset({
    "xyz", "geometry", "geom", "geom_id", "input_id", "output_id",
    "charge", "multiplicity", "mult", "spin",
})
_MCP_ONLY_ARGS = frozenset({"wall_timeout_seconds", "ncores", "job_label"})

# Keys that must never be hardcoded in a tool node's args.
_FORBIDDEN_HARDCODED = frozenset({"charge", "multiplicity", "mult", "spin"})

_REQUIRED_PLAN_KEYS = ("nodes",)


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    severity: str          # "ERROR" | "WARN"
    code: str              # stable machine-readable id
    node_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"[{self.node_id}] " if self.node_id else ""
        return f"{self.severity:5s} {self.code:24s} {where}{self.message}"


@dataclass(frozen=True)
class ValidationResult:
    violations: tuple[Violation, ...]

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "ERROR")

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "WARN")

    @property
    def ok(self) -> bool:
        """True when there are no ERROR-level violations."""
        return not self.errors


# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_tool_schemas(path: str = _SCHEMA_FILE) -> dict[str, dict[str, Any]]:
    """Return {tool_name: parameters_schema} from the OpenAI tools file.

    The file is a dict keyed by tool name → {type, function:{parameters:{...}}}.
    """
    raw = json.load(open(path, encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    entries = raw.values() if isinstance(raw, dict) else raw
    for entry in entries:
        fn = entry.get("function", entry)
        name = fn.get("name")
        if name:
            out[name] = fn.get("parameters") or {}
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_template_ref(value: Any) -> bool:
    """True for placeholder strings like '$(settings.K)' or '$(input.x)'."""
    return isinstance(value, str) and value.strip().startswith("$(")


def _emit(out: list[Violation], severity: str, code: str, node_id: str | None, msg: str) -> None:
    out.append(Violation(severity, code, node_id, msg))


# ---------------------------------------------------------------------------
# Plan-shape checks
# ---------------------------------------------------------------------------


def _check_plan_shape(plan: dict[str, Any], out: list[Violation]) -> None:
    for key in _REQUIRED_PLAN_KEYS:
        if key not in plan:
            _emit(out, "ERROR", "missing_plan_key", None, f"plan missing required key '{key}'")

    nodes = plan.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        _emit(out, "ERROR", "empty_nodes", None, "plan['nodes'] must be a non-empty list")
        return

    for lst_key in ("geom_ids", "artifacts_to_save"):
        val = plan.get(lst_key)
        if val is not None and not (isinstance(val, list) and all(isinstance(x, str) for x in val)):
            _emit(out, "ERROR", "bad_list_field", None, f"plan['{lst_key}'] must be a list of strings")

    seen: set[str] = set()
    for i, node in enumerate(nodes):
        nid = node.get("id")
        if not nid:
            _emit(out, "ERROR", "missing_node_id", None, f"node #{i} has no 'id'")
            continue
        if nid in seen:
            _emit(out, "ERROR", "duplicate_node_id", nid, "duplicate node id")
        seen.add(nid)
        kind = node.get("kind")
        if kind not in VALID_NODE_KINDS:
            _emit(out, "ERROR", "bad_node_kind", nid, f"unknown kind {kind!r} (allowed: {sorted(VALID_NODE_KINDS)})")


# ---------------------------------------------------------------------------
# Tool-node checks
# ---------------------------------------------------------------------------


def _check_tool_node(node: dict[str, Any], schemas: dict[str, dict[str, Any]], out: list[Violation]) -> None:
    nid = node.get("id")
    tool = node.get("tool")
    if not tool:
        _emit(out, "ERROR", "tool_node_no_tool", nid, "tool node has no 'tool' name")
        return
    if tool in CLIENT_SIDE_TOOLS:
        # Known local tool with no MCP schema — nothing further to validate here.
        _check_project_invariants(tool, nid, node.get("args") or {}, out)
        return
    if tool not in schemas:
        _emit(out, "ERROR", "unknown_tool", nid, f"unknown tool '{tool}' (not in openai_tools_geom.json)")
        return

    params = schemas[tool]
    properties = params.get("properties") or {}
    required = params.get("required") or []
    allowed = set(properties) | _INJECTED_ARGS | _MCP_ONLY_ARGS
    args = node.get("args") or {}
    if not isinstance(args, dict):
        _emit(out, "ERROR", "bad_args_type", nid, "node 'args' must be a dict")
        return

    # Hallucinated args
    for key in args:
        if key not in allowed:
            _emit(out, "WARN", "unknown_arg", nid, f"arg '{key}' is not in the schema for '{tool}'")

    # Missing planner-supplied required args (exclude executor-injected ones)
    for req in required:
        if req in _INJECTED_ARGS:
            continue
        if req not in args:
            _emit(out, "WARN", "missing_required_arg", nid, f"required arg '{req}' for '{tool}' not supplied")

    _check_project_invariants(tool, nid, args, out)


def _check_project_invariants(tool: str, nid: str | None, args: dict[str, Any], out: list[Violation]) -> None:
    # ncores must be 1 — MPI aborts under kubectl exec for ALL job types.
    if "ncores" in args and not _is_template_ref(args["ncores"]):
        try:
            if int(args["ncores"]) != 1:
                _emit(out, "ERROR", "ncores_not_one", nid, f"ncores must be 1, got {args['ncores']}")
        except (TypeError, ValueError):
            _emit(out, "ERROR", "ncores_not_int", nid, f"ncores must be int 1, got {args['ncores']!r}")

    # charge / multiplicity: state defaults are "fallback only" (client_helpers.py),
    # so per-node values are legal and often required (pKa anions, ions, complexes).
    # Flag as advisory only — a hardcoded value that contradicts the compound's real
    # charge is a likely error, but the executor tolerates explicit values.
    for key in _FORBIDDEN_HARDCODED:
        if key in args and not _is_template_ref(args[key]):
            _emit(out, "WARN", "explicit_charge_mult", nid,
                  f"'{key}' is set explicitly in args; confirm it matches the species "
                  f"(state defaults are fallback only)")

    # scan_coords must be a JSON string, not a list (survives MCP serialisation).
    if "scan_coords" in args:
        sc = args["scan_coords"]
        if isinstance(sc, list):
            _emit(out, "ERROR", "scan_coords_not_string", nid,
                  "scan_coords must be a JSON *string*, not a list")
        elif isinstance(sc, str) and not _is_template_ref(sc):
            try:
                parsed = json.loads(sc)
            except ValueError:
                _emit(out, "ERROR", "scan_coords_bad_json", nid, "scan_coords is not valid JSON")
            else:
                _check_scan_indices(parsed, nid, out)


def _check_scan_indices(parsed: Any, nid: str | None, out: list[Violation]) -> None:
    if not isinstance(parsed, list):
        return
    for spec in parsed:
        atoms = spec.get("atoms") if isinstance(spec, dict) else None
        if isinstance(atoms, list):
            for a in atoms:
                if not isinstance(a, int) or a < 0:
                    _emit(out, "WARN", "scan_atom_index", nid,
                          f"scan atom index {a!r} should be a non-negative int")


# ---------------------------------------------------------------------------
# calc_expr safety (mirrors the executor's AST allow-list)
# ---------------------------------------------------------------------------

_ALLOWED_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant, ast.Name,
    ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.Call, ast.keyword,
)
_ALLOWED_EXPR_FUNCS = frozenset({"abs", "min", "max", "round", "log", "log10", "exp", "sqrt"})


def _check_calc_expr_node(node: dict[str, Any], out: list[Violation]) -> None:
    nid = node.get("id")
    expr = node.get("expr") or (node.get("args") or {}).get("expr")
    if not expr or not isinstance(expr, str):
        _emit(out, "ERROR", "calc_expr_missing", nid, "calc_expr node has no 'expr' string")
        return
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        _emit(out, "ERROR", "calc_expr_syntax", nid, f"expr does not parse: {e}")
        return
    for sub in ast.walk(tree):
        if not isinstance(sub, _ALLOWED_EXPR_NODES):
            _emit(out, "ERROR", "calc_expr_disallowed", nid,
                  f"disallowed syntax {type(sub).__name__} in expr")
        if isinstance(sub, ast.Call):
            fname = getattr(sub.func, "id", None)
            if fname not in _ALLOWED_EXPR_FUNCS:
                _emit(out, "ERROR", "calc_expr_bad_func", nid, f"disallowed function {fname!r} in expr")


# ---------------------------------------------------------------------------
# Cross-reference checks
# ---------------------------------------------------------------------------


def _check_artifacts(plan: dict[str, Any], out: list[Violation]) -> None:
    produced: set[str] = set()
    for node in plan.get("nodes", []):
        product = node.get("product") or {}
        if isinstance(product, dict):
            produced.update(product.keys())
    for key in plan.get("artifacts_to_save") or []:
        if key not in produced and key not in (plan.get("settings") or {}):
            _emit(out, "WARN", "artifact_not_produced", None,
                  f"artifacts_to_save['{key}'] is not produced by any node 'product' mapping")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_plan(plan: dict[str, Any], schemas: dict[str, dict[str, Any]] | None = None) -> ValidationResult:
    """Validate a plan dict. Never raises — returns all violations found."""
    out: list[Violation] = []
    if not isinstance(plan, dict):
        return ValidationResult((Violation("ERROR", "not_a_dict", None, "plan is not a JSON object"),))
    if schemas is None:
        schemas = load_tool_schemas()

    _check_plan_shape(plan, out)
    for node in plan.get("nodes", []):
        if not isinstance(node, dict):
            _emit(out, "ERROR", "bad_node", None, "node is not a JSON object")
            continue
        kind = node.get("kind")
        if kind == "tool":
            _check_tool_node(node, schemas, out)
        elif kind == "calc_expr":
            _check_calc_expr_node(node, out)
    _check_artifacts(plan, out)
    return ValidationResult(tuple(out))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validate_file(path: str, schemas: dict[str, dict[str, Any]]) -> ValidationResult:
    doc = json.load(open(path, encoding="utf-8"))
    plan = doc.get("plan", doc) if isinstance(doc, dict) else doc
    return validate_plan(plan, schemas)


def main() -> None:
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="Validate plan JSON / tool-call guardrails")
    ap.add_argument("paths", nargs="*", help="plan or report JSON files (glob ok)")
    ap.add_argument("--quiet", action="store_true", help="only print files with violations")
    args = ap.parse_args()

    schemas = load_tool_schemas()
    files: list[str] = []
    for p in args.paths or ["runtime_reports/*.json"]:
        files.extend(sorted(glob.glob(p)))

    n_ok = n_err = 0
    for path in files:
        try:
            res = _validate_file(path, schemas)
        except (ValueError, OSError) as e:
            print(f"SKIP {os.path.basename(path)}: {e}")
            continue
        if res.ok:
            n_ok += 1
            if not args.quiet:
                print(f"OK   {os.path.basename(path)} ({len(res.warnings)} warnings)")
        else:
            n_err += 1
            print(f"FAIL {os.path.basename(path)} — {len(res.errors)} errors:")
            for v in res.errors:
                print(f"     {v}")
    print(f"\n{n_ok} ok / {n_err} with errors / {len(files)} total")


if __name__ == "__main__":
    main()
