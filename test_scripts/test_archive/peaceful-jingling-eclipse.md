# Plan: Implement patch_and_retry for on_error blocks

## Context

The planner LLM generates `on_error` blocks on tool nodes with deterministic retry
rules (e.g. increase `scf_max_iter` on SCF convergence failure). The executor
(`build_graph_from_plan.py`) completely ignores these blocks — `on_error` is not
referenced anywhere. If a node fails, the graph just fails with no retry. This plan
wires the retry loop into the executor.

## Scope — 3 files

1. `build_graph_from_plan.py` — 3 new helpers + retry loop in `make_node`
2. `server_with_product.py` — add `code` field to error returns
3. `prompts.py` — canonicalize on_error schema so future plans are consistent

## Key design decisions

**Error code detection (dual-source)**
- Server adds `"code": "SCF_NOT_CONVERGED"` etc. to error payloads (primary)
- Client-side `_infer_error_code()` also parses `error_summary`/`tail` text (fallback)
- Codes: `SCF_NOT_CONVERGED`, `IMAG_FREQ`, `GEOM_INVALID`, `RESOURCE_LIMIT`

**Schema normalization** — planner generates 3 inconsistent variants:
- condition key: `"if"` vs `"when"`
- action nesting: direct vs wrapped under `"then"`
- `_normalize_on_error_rule()` handles all → canonical `{codes, action, patch, max_attempts}`

**Patch → tool arg mapping** — only safe known params applied:
- `"scf": {"maxiter": N}` → `scf_max_iter=N`
- `"resources": {"ncores": N}` → `ncores=N`
- Flat keys in `_PATCHABLE_TOOL_ARGS` set: `method`, `basis`, `use_ri`, `scf_max_iter`,
  `opt_max_iter`, `ncores`, `wall_timeout_seconds`, `n_states`, `calc_hess`
- Unknown keys silently dropped (avoids TypeError in strict MCP tool signatures)
- `"preopt"`, `"opt_then_repeat"` patches: NOT implemented (require extra nodes)

**Retry loop location**: Inside `make_node._node`, wrapping the tool execution in the
`kind == "tool"` branch. Retry uses local `attempt` counter — no new state fields.

**Geometry between retries**: `current_state` is updated with geometry keys from the
previous attempt's result so retry sees any geometry changes.

---

## Implementation

### 1. `build_graph_from_plan.py` — three module-level helpers (before `build_graph_from_plan()`)

```python
_PATCHABLE_TOOL_ARGS = {
    "method", "basis", "use_ri", "scf_max_iter", "opt_max_iter",
    "ncores", "wall_timeout_seconds", "n_states", "calc_hess",
}

def _infer_error_code(tool_payload: Dict[str, Any]) -> Optional[str]:
    """Return structured error code from tool result (server-supplied or parsed from text)."""
    code = (tool_payload or {}).get("code")
    if code:
        return str(code)
    if tool_payload.get("status") == "timeout":
        return "RESOURCE_LIMIT"
    text = " ".join([
        str(tool_payload.get("error_summary") or ""),
        str(tool_payload.get("error") or ""),
        str(tool_payload.get("tail") or ""),
    ]).upper()
    if "SCF NOT CONVERGED" in text:
        return "SCF_NOT_CONVERGED"
    if "IMAGINARY" in text and ("MODE" in text or "FREQ" in text):
        return "IMAG_FREQ"
    if "GEOM" in text and ("INVALID" in text or "FAILED" in text or "BAD" in text):
        return "GEOM_INVALID"
    if "RESOURCE" in text or "TIME LIMIT" in text:
        return "RESOURCE_LIMIT"
    return None


def _normalize_on_error_rule(rule: dict) -> Dict[str, Any]:
    """Normalize one on_error entry to {codes, action, patch, max_attempts}."""
    condition = rule.get("if") or rule.get("when") or {}
    inner = rule.get("then") if "then" in rule else rule
    return {
        "codes":        list(condition.get("code_in") or []),
        "action":       (inner.get("action") or ""),
        "patch":        dict(inner.get("patch") or {}),
        "max_attempts": int(inner.get("max_attempts") or 1),
    }


def _apply_patch_to_args(current_args: dict, patch: dict) -> dict:
    """Merge patch into tool args, mapping nested structures to known param names."""
    new_args = dict(current_args)
    for key, val in patch.items():
        if key == "scf" and isinstance(val, dict):
            if "maxiter" in val:
                new_args["scf_max_iter"] = int(val["maxiter"])
        elif key == "resources" and isinstance(val, dict):
            if "ncores" in val:
                new_args["ncores"] = int(val["ncores"])
        elif key in _PATCHABLE_TOOL_ARGS:
            new_args[key] = val
        # else: silently drop unknown keys
    return new_args
```

### 2. `build_graph_from_plan.py` — replace single tool call with retry loop

Inside `make_node._node`, in the `kind == "tool"` branch, replace the current block
that calls `run_tool_node` (lines ~383-435) with:

```python
# --- build retry-capable rules list ---
on_error_rules = [
    _normalize_on_error_rule(r)
    for r in (spec.get("on_error") or [])
]
on_error_rules = [r for r in on_error_rules if r["action"] == "patch_and_retry"]

attempt       = 0
attempt_args  = dict(spec.get("args") or spec.get("overrides") or {})
current_state = state

while True:
    retry_spec = {**spec, "args": attempt_args,
                  "_plan_settings": plan.get("settings") or {}}

    if run_tool_node is not None:
        result = await _maybe_await(run_tool_node(current_state, retry_spec))
        if not isinstance(result, dict):
            result = {"status": "error", "error": "tool node runner returned non-dict"}
    else:
        args = get_tool_args(state["session"], tool_name, attempt_args)
        result = call_tool(tool_name, args) or {}

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            result = {"status": "error", "error": "non-JSON string", "raw": result}

    tool_payload = result.get("last_tool_result", result)
    status = result.get("last_status",
                        tool_payload.get("status") if isinstance(tool_payload, dict) else None
                       ) or "unknown"

    if status == "ok" or not on_error_rules:
        break

    error_code = _infer_error_code(tool_payload if isinstance(tool_payload, dict) else {})
    matched_rule = next(
        (r for r in on_error_rules
         if error_code in r["codes"] and attempt < r["max_attempts"]),
        None,
    )
    if matched_rule is None:
        break  # no rule matches or max_attempts exhausted

    attempt += 1
    attempt_args = _apply_patch_to_args(attempt_args, matched_rule["patch"])
    current_state = {**state, **{
        k: result[k] for k in ("geometries", "geom_meta", "name_to_geom", "current_geom")
        if k in result
    }}
    print(f"  [retry] node={node_id!r} attempt={attempt} "
          f"code={error_code!r} patch_keys={list(matched_rule['patch'].keys())}")

# --- rest of existing bookkeeping (updates.update(result), run_log, etc.) ---
# Add to run_log entry: "retry_attempts": attempt
```

### 3. `server_with_product.py` — add `_classify_orca_error_code()` + `code` field

After `_error_summary()` (~line 284), add:

```python
def _classify_orca_error_code(out_text: str) -> Optional[str]:
    upper = out_text.upper()
    if "SCF NOT CONVERGED" in upper or "FAILED TO CONVERGE" in upper:
        return "SCF_NOT_CONVERGED"
    if "***IMAGINARY MODE***" in upper:
        return "IMAG_FREQ"
    return None
```

In each error/not-converged return dict that has `out_text` in scope, add:
```python
"code": _classify_orca_error_code(out_text),
```
Target error returns (by approximate line):
- `run_opt_job` abnormal termination block (~line 789)
- `run_opt_job` exception catch (~line 767)
- `run_freq_job` error returns (~lines 1018, 1023)
- `run_sp_energy` error returns (~lines 961, 966)
- `run_spectrum_job` error returns (~lines 1102, 1108)
- `run_scan_job` error returns (~lines 1443, 1449)
- `run_ts_opt_job` error returns (~lines 1570, 1584)

### 4. `prompts.py` — canonicalize on_error schema

Replace the current error handling policy paragraph with the canonical schema
and supported patch keys. The new text should list:
- Canonical condition key: `"if"` with `"code_in"`
- Direct `"action"`, `"patch"`, `"max_attempts"` (no `"then"` wrapper)
- Supported error codes and their matching patch keys

---

## Verification

1. **Transparent on success**: Run existing full-mode test (H2O2 TS or pKa).
   Confirm `[retry]` log lines do NOT appear.

2. **Plan-mode regression**: `python test_success_rate.py --checker pka --runs 3`
   Confirm on_error blocks still generated and 100% pass.

3. **Deploy server**: `scp + kubectl cp server_with_product.py` to pod.

4. **(Optional) synthetic retry test**: Manually craft a plan that calls a tool
   with `scf_max_iter=1` (guaranteed SCF fail) + `on_error: [{if: {code_in:
   [SCF_NOT_CONVERGED]}, action: patch_and_retry, patch: {scf_max_iter: 300},
   max_attempts: 1}]` and run in full mode to confirm retry fires.

## Architecture
### Geometry passing (uses existing mechanism, no new code)
- run_scan_job returns "geometry_xyz": <XYZ at PES max>
- _normalize_payload_to_contract picks it up automatically
- Plan node sets output_id: "ts_candidate" on scan node -> stored in state["geometries"]["ts_candidate"]
- run_ts_opt_job is in NEEDS_GEOM_SINGLE -> geometry injected via input_id: "ts_candidate"

### Per-step geometry extraction (new helper _extract_scan_geometries)
- Splits .out text by "RELAXED SURFACE SCAN STEP N" markers
- Calls existing extract_final_geometry_from_out(chunk) on each chunk
- Returns List[str] of XYZ strings, indexed by (step-1)

## Files to modify

### 1. server_with_product.py (local + deploy to pod)

a. _build_calc: add "ts_opt" job type
   - task_kw = "OptTS" when job_type == "ts_opt"
   - BlockGeom(maxiter=opt_max_iter) for both "opt" and "ts_opt"
   - Update Literal type hint

b. New helper _extract_scan_geometries(out_text, n_steps) -> List[str]
   - Split by RELAXED SURFACE SCAN STEP N markers
   - extract_final_geometry_from_out on each chunk -> optimized geom per step

c. run_scan_job: add max_geometry_xyz to return dict
   - Also compute max_energy (in addition to existing min_energy/min_value)
   - add "geometry_xyz": max_geom and "max_geometry_xyz": max_geom to return

d. New run_ts_opt_job MCP tool
   - Signature: geometry_xyz, charge, multiplicity, method, basis, use_ri,
     scf_max_iter, opt_max_iter, calc_hess=True, wall_timeout_seconds, job_label, ncores
   - _build_calc with job_type="ts_opt"
   - If calc_hess: add_arbitrary_string("%geom\n  Calc_Hess true\nend")
   - Returns: {"status", "label", "geometry_xyz", "energy_eh", "text"}
   - extract_final_geometry_from_out for TS geometry

### 2. client_helpers.py
   - Add "run_ts_opt_job" to NEEDS_GEOM_SINGLE

### 3. openai_tools_geom.json
   - Add run_ts_opt_job schema (input_geom_id, method, basis, calc_hess, opt_max_iter, etc.)

### 4. skills.py
   - Add TSSearchSkill (priority 27, between Thermo=25 and Solvation=30)
   - Keywords: "transition state", "ts search", "ts opt", "saddle point", "activation barrier", "optts"
   - Plan pattern: scan (output_id: ts_candidate) -> ts_opt (input_id: ts_candidate, output_id: ts_structure) -> freq
   - Notes: calc_hess=true required; 1 negative freq in ir_spectrum confirms TS

### 5. prompts.py
   - Add run_ts_opt_job to tool list

### 6. test_success_rate.py
   - Add check_ts_plan: has_scan_node, has_ts_opt_node, has_freq_node, no_llm_node
   - Add "ts" to checker map, auto keywords, choices

## Plan pattern
{
  nodes: [
    {load_mol},
    {scan_rc: run_scan_job, output_id: "ts_candidate",
     product: {"scan_results_rc": "scan_results"}},
    {ts_opt: run_ts_opt_job, input_id: "ts_candidate", output_id: "ts_structure",
     product: {"ts_energy_eh": "energy_eh"}},
    {ts_freq: run_freq_job, input_id: "ts_structure",
     product: {"ts_ir_spectrum": "ir_spectrum", "ts_gibbs_eh": "gibbs_free_energy_eh"}}
  ]
}

## Deployment
scp server_with_product.py zhang@qcl:/tmp/
kubectl cp /tmp/server_with_product.py ns-general/zhang-ollama:/data/zhang/ollama/nbo_agent/server_with_product.py

## Verification
1. Plan test: python test_success_rate.py --checker ts --mode plan --runs 3 --message "find transition state..."
2. Full test: --mode full --runs 1 (will take ~30+ min for scan+OptTS+freq)
3. Check: ts_ir_spectrum has exactly 1 negative frequency entry
