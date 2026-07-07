

## Adding a New MCP Tool (checklist)

Use this to avoid re-reading the whole codebase. Grep for line numbers, then targeted reads.

### Find insertion points

```
Grep "def _build_calc"               → line ~157  (extend Literal + task_kw if new ORCA keyword)
Grep "async def run_opt_job"         → template to copy (~130 lines)
Grep "async def run_solvator_cluster\b" → insert new @mcp.tool() just above this
Grep "NEEDS_GEOM_SINGLE"             → client_helpers.py line ~33
Grep "run_scan_job:" SYSTEM_PROMPT   → prompts.py tool list (add one line)
Grep "check_scan\|_CHECKER_MAP\|_AUTO_KEYWORDS\|parse_args" test_success_rate.py → 4 spots
```

### 6-file checklist

| # | File | Change |
|---|------|--------|
| 1 | `server_with_product.py` | New `@mcp.tool()` + optionally extend `_build_calc` |
| 2 | `client_helpers.py` | Add to `NEEDS_GEOM_SINGLE` set; add output fields to `TOOL_OUTPUT_FIELDS` |
| 3 | `openai_tools_geom.json` | Add schema entry before closing `}` |
| 4 | `skills.py` | New `PlannerSkill` class + add to `SKILL_REGISTRY` |
| 5 | `prompts.py` | Add `run_MY_job: field_1, field_2` to tool result list |
| 6 | `test_success_rate.py` | `check_my()` + `_CHECKER_MAP` + `_AUTO_KEYWORDS` + CLI choices |

### server_with_product.py tool skeleton

```python
@mcp.tool()
async def run_MY_job(
    geometry_xyz: str, charge: int = 0, multiplicity: int = 1,
    method: str = "B3LYP", basis: str = "def2-SVP", use_ri: bool = True,
    scf_max_iter: int = 150, wall_timeout_seconds: int = 1800,
    job_label: Optional[str] = None, ncores: int = 1,
) -> str:
    if not geometry_xyz.strip(): raise ValueError("geometry_xyz is empty")
    if job_label is None:
        job_label = f"myjob_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    workdir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs")) / job_label

    calc = _build_calc(label=job_label, workdir=workdir, geometry_xyz=geometry_xyz,
        charge=charge, multiplicity=multiplicity, method=method, basis=basis,
        job_type="sp", use_ri=use_ri, scf_max_iter=scf_max_iter,
        opt_max_iter=1, nbo=False, ncores=ncores)
    # calc.input.add_arbitrary_string("...extra blocks...")

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        return json.dumps({"status": "timeout", "label": job_label})
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({"status": "error", "label": job_label,
                           "tail": "\n".join(out_text.splitlines()[-120:])})

    energy = extract_total_energy(out_text)
    return json.dumps({"status": "ok", "label": job_label,
                       "energy_eh": energy, "product": "energy_eh",
                       "text": f"Status: OK\nEnergy: {energy} Eh"})
```

### _build_calc: add new job_type (only if new ORCA keyword needed)

```python
# Literal (~line 165):
job_type: Literal["sp", "opt", "freq", "scan", "ts_opt", "MY_NEW_TYPE"],

# task_kw block (~line 189):
if job_type == "MY_NEW_TYPE":
    task_kw = "MyORCAKeyword"

# BlockGeom rule: if custom %geom options needed, keep "opt" only and build
# the block manually in the tool (one %geom block total — ORCA rejects duplicates).
```

### skills.py priority ladder

`10`=method · `15`=protonation · `20`=pka · `21`=scan · `22`=spectrum ·
`23`=tddft · `25`=thermo · `27`=ts_search · `30`=solvation · `40`=nbo

### test_success_rate.py checker template

```python
def check_my(plan: dict) -> Dict[str, bool]:
    nodes     = plan.get("nodes", [])
    tools_set = {n.get("tool") for n in nodes if n.get("kind") == "tool"}
    all_keys  = set(plan.get("artifacts_to_save") or []) | _all_product_keys(plan)
    llm_nodes = [n for n in nodes if n.get("kind") == "llm"]
    return {
        "has_my_node":  "run_MY_job" in tools_set,
        "my_artifact":  any("expected_key" in k.lower() for k in all_keys),
        "no_llm_node":  len(llm_nodes) == 0,
    }
```

### TOOL_OUTPUT_FIELDS registry (client_helpers.py)

When adding a new tool, register its output fields in `TOOL_OUTPUT_FIELDS` near the top of
`client_helpers.py`. This dict is the single source of truth for:
- Checkers in `test_success_rate.py` / `test_context_scaling.py` (tool-aware artifact checking)
- Future: auto-generating planner prompt tool list, plan validation before execution

```python
TOOL_OUTPUT_FIELDS: Dict[str, List[str]] = {
    "run_sp_energy":              ["energy_eh", "homo_lumo_gap_ev", "dipole_moment_debye", "mulliken_charges"],
    "run_opt_job":                ["energy_eh"],
    "run_freq_job":               ["energy_eh", "enthalpy_eh", "gibbs_free_energy_eh", "ir_spectrum"],
    "run_spectrum_job":           ["energy_eh", "enthalpy_eh", "gibbs_free_energy_eh", "ir_spectrum", "raman_spectrum"],
    "run_tddft_job":              ["energy_ground_state_eh", "homo_lumo_gap_ev", "excited_states"],
    "run_scan_job":               ["scan_results", "min_energy_eh", "min_value", "n_points", "geometry_xyz"],
    "run_ts_opt_job":             ["energy_eh", "geometry_xyz", "ts_converged"],
    "run_nbo_job":                ["nbo_section"],
    "run_casscf_job":             ["energy_eh", "energies_eh"],
    "run_solvator_cluster_thermo": ["energy_eh", "enthalpy_eh", "gibbs_free_energy_eh"],
    # Add new tool here: "run_MY_job": ["field_1", "field_2"]
}
```

Add a new entry alongside each new `@mcp.tool()`. Field names must exactly match the JSON
keys returned by the tool (same names used in explicit `product` dicts and auto-artifact keys).

### Key conventions

- Geometry never in artifacts — flows via `input_id` / `output_id` only
- `product`: `{"artifact_key": "result_field"}` — maps planner key → JSON result field; omit entirely for auto-artifact mode
- Auto-artifact mode: no `product` dict → orchestrator saves all fields as `{node_id}.{field}`; use `final_report.fields: ["node_id.field", ...]` to declare outputs
- No `kind:"llm"` node for structured outputs (spectra, scan, NBO text, geometries)
- `scan` / `ts_opt` ncores must be 1 (MPI + relaxed scan broken in ORCA 6.1.1)
- Use `energy_eh` (not `energy`) in new tools for consistency

