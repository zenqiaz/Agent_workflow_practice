---
name: qc-agent-dev
description: This skill should be used when the user wants to "add a new ORCA tool", "add a new MCP tool", "add a calculation type", "extend the QC agent", "add a skill to the planner", "debug a plan node", "modify the workflow", "write a checker", "add to NEEDS_GEOM_SINGLE", "fix a server tool", or is working on the QC agent codebase at D:/brick/D/20260217/working/. Provides the 6-file checklist, insertion grep patterns, key conventions, and architecture context without requiring the user to read multiple files.
version: 1.0.0
---

# QC Agent Development Skill

## Architecture at a Glance

```
nbo_agent_planning.py   — REPL entrypoint; pre-planning pipeline
build_graph_from_plan.py — LangGraph executor; runs JSON plan as DAG
client_helpers.py       — geometry helpers, state, PubChem, plan UI
skills.py               — PlannerSkill classes injected as system msgs
prompts.py              — SYSTEM_PROMPT, QC-CALCULATOR, REPORTER prompts
openai_tools_geom.json  — OpenAI tool schemas for all tools
server_with_product.py  — FastMCP server on VM; all ORCA/SOLVATOR tools
server_helpers.py       — low-level OPI helpers (shared by server)
new_tools.md            — ALWAYS read this first when adding a new tool
```

## 6-File Checklist (new ORCA tool)

| # | File | Change |
|---|------|--------|
| 1 | `server_with_product.py` | New `@mcp.tool()` + optionally extend `_build_calc` |
| 2 | `client_helpers.py` | Add tool name to `NEEDS_GEOM_SINGLE` set |
| 3 | `openai_tools_geom.json` | Add schema entry before closing `}` |
| 4 | `skills.py` | New `PlannerSkill` class + append to `SKILL_REGISTRY` |
| 5 | `prompts.py` | Add `run_MY_job: field_1, field_2` to tool result list |
| 6 | `test_success_rate.py` | `check_my()` + `_CHECKER_MAP` + `_AUTO_KEYWORDS` + CLI choices |

## Grep Patterns for Insertion Points

```
Grep "def _build_calc"               → server_with_product.py ~157
Grep "async def run_opt_job"         → template to copy (~130 lines)
Grep "async def run_solvator_cluster\b" → insert new tool just above
Grep "NEEDS_GEOM_SINGLE"             → client_helpers.py ~33
Grep "run_scan_job:" SYSTEM_PROMPT   → prompts.py tool list
Grep "check_scan\|_CHECKER_MAP\|_AUTO_KEYWORDS\|parse_args" test_success_rate.py
```

## Key Conventions

- **Artifact field name**: always `energy_eh` (never bare `energy`)
- **Geometry flow**: via `input_id` / `output_id` only — never store XYZ in artifacts
- **product spec**: `{"artifact_key": "result_field"}` — maps planner key to JSON field
- **No `kind:"llm"` node** for structured outputs (spectra, NBO, scan, freq results)
- **ncores must be 1** for `scan` and `ts_opt` (MPI + relaxed scan broken in ORCA 6.1.1)
- **Charge/multiplicity**: never hardcode in plan args; executor injects from state defaults
- **`_MCP_ONLY_PARAMS`**: `wall_timeout_seconds`, `ncores`, `job_label` are stripped before
  client-side tool calls — do not add them to client-side function signatures

## Skills Priority Ladder (skills.py)

```
10 = method_selection   (always active)
15 = protonation
20 = pka
21 = scan
22 = spectrum
23 = tddft
24 = casscf
25 = thermochemistry
27 = ts_search
30 = solvation
40 = nbo
```

New skills: pick a priority in the right range, append to `SKILL_REGISTRY`.

## Server Tool Skeleton

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
                       "energy_eh": energy, "product": "energy_eh"})
```

## Test Checker Template

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

## Deploy to Pod

```bash
# Windows OpenSSH required — Git Bash ssh/scp fail with publickey
"C:/Windows/System32/OpenSSH/scp.exe" -i ~/.ssh/qclab_zhang server_with_product.py zhang@<host>:/tmp/
"C:/Windows/System32/OpenSSH/ssh.exe" -i ~/.ssh/qclab_zhang zhang@<host> \
  "kubectl cp /tmp/server_with_product.py ns-general/zhang-ollama:/data/zhang/ollama/nbo_agent/server_with_product.py"
```

## Reference Files

- `D:/brick/D/20260217/working/new_tools.md` — canonical tool-creation guide (read first)
- `D:/brick/D/20260217/working/CLAUDE.md` — project overview, REPL commands, schema
