# Q-Planner — Project Context (session handoff)

> Standalone context file for the **Q-Planner** effort (the QC-agent's LLM planner +
> calculator + reporter). Load this alongside the repo's `CLAUDE.md`. It describes the
> system **as it currently stands** and, importantly, **what we demand from the base
> model** that drives it. Fine-tuning/data plans live separately in `finetune/`.

---

## 1. What Q-Planner is

An agentic quantum-chemistry workflow system. A local Windows client takes a
natural-language request, an LLM **planner** turns it into a deterministic **JSON plan**,
and a **LangGraph** executor runs that plan node-by-node against **ORCA 6.1.1** on a
remote Linux node, over **MCP-stdio-over-SSH**. Three LLM roles, one shared model:

| Role | Maps | Prompt |
|------|------|--------|
| **Planner** | `(system + skills + state + compounds + request)` → plan JSON | `QC-PLANNER` |
| **Calculator** | `(artifact values)` → derived numeric quantity + verification | `QC-CALCULATOR` |
| **Reporter** | `(artifact dict)` → markdown report | `REPORTER` |

```
Local (Windows)                         Lab k8s (zhang-ollama pod)
agent.py (REPL)                         server_with_product.py (FastMCP)
  planner LLM → JSON plan                 ORCA 6.1.1 + OPI + SOLVATOR
  LangGraph executor                      36-CPU node
  └──── SSH (kubectl exec) ───────────────►
```

---

## 2. Pipeline (per request)

```
user text
  ├─ compound identification (client_helpers.py): names/XYZ → PubChem/OPSIN → 2D image → confirm
  ├─ skill dispatch (skills.py): matched domain blocks injected as system messages
  ├─ planner LLM → plan JSON
  ├─ (optional) plan_validator.py: static checks before any compute
  └─ build_graph_from_plan.py: LangGraph executes nodes → artifacts → reporter
```

Node kinds: `tool` (MCP/client-side call), `llm` (calculator), `calc_expr` (safe AST math).

---

## 3. Current capabilities (confirmed working)

ORCA tools, all executor-verified end-to-end on the lab pod:

- `run_sp_energy`, `run_opt_job`, `run_freq_job` (E/H/G), `run_nbo_job`
- `run_spectrum_job` (IR/Raman), `run_tddft_job` (UV-Vis, `nroots`)
- `run_scan_job` (relaxed PES), `run_ts_opt_job` (OptTS), `run_casscf_job` (+AVAS)
- `run_solvator_cluster` / `_thermo` (explicit solvent), `structure_add_remove_proton`

Client-side: `name_to_geometry_xyz` (incl. bare ions/atoms), `pubchem_get_basic_properties`,
`build_coordination_complex`, `build_approach_scan_geometries`, `state_*`.

Visualization: IR/UV-Vis spectra + PES plots auto-rendered (`showspec` REPL command).

> Detailed per-tool notes (parameters, parser quirks, test results) are in the auto-memory
> `MEMORY.md`. Consult it before modifying any tool.

---

## 4. Hard invariants (do not break)

- **`ncores=1` for ALL ORCA jobs.** MPI aborts under `kubectl exec`. Parallelism comes
  from running multiple LangGraph nodes concurrently, never MPI within a job.
- **Never hardcode charge/multiplicity in plan args** — the executor injects them from
  state defaults (per-node overrides are allowed and by design for ions/anions/complexes).
- **`scan_coords` is a JSON *string***, not a list (survives MCP serialization).
- **Geometry storage:** headerless XYZ in `state["geometries"]`, keyed by stable `geom_id`
  that must match `product`/`input_id`/`output_id` references in the plan.
- **Atom indices are 0-based** and the planner must get dihedral/bond indices right.

---

## 5. What we demand from the BASE MODEL

This is the core of the handoff: the qualities a base (or future tuned) model must have to
drive Q-Planner well. Each is grounded in an existing mechanism in the repo.

### 5.1 Robust numerical calculation & self-verification
The **calculator** role does unit-bearing arithmetic (pKa, ΔG, K, conversions). We demand:
- **Self-checking via reverse / round-trip verification** — invert the formula, substitute
  the output back, recover the original input within tolerance, and emit the *residual*.
  Enforced in `CALCULATOR_SYSTEM_PROMPT` (prompts.py): the required `checks` list with
  `reverse_recovers_inputs` as the PRIMARY record.
- **The model must distinguish "effective checking" from "checking in vain"** — a check that
  shows a real residual and trips on a real mismatch, vs. a vacuous restatement or a
  miscalibrated halt. Checks must be *decision-relevant*.
- **Honest limitation we accept:** round-trip catches execution errors (arithmetic/sign/
  unit-transcription), **not** specification errors (wrong formula applied consistently).
  Pure arithmetic should therefore migrate to deterministic `calc_expr`; the LLM is for
  judgment, not for being a calculator.

### 5.2 Domain knowledge — method selection
The **planner** must choose chemically sound methods without being told. We demand built-in
QC competence equivalent to what `skills.py` currently injects:
- **DFT method/basis by task** (e.g. B3LYP/def2-SVP opt → PBE0/def2-TZVP SP), adapting to
  molecule size and atom count.
- **Protocol fluency**: pKa (remove proton → freq on HA & A⁻ → ΔG with `G_H_plus_ref_eh =
  -0.01372` Eh), thermochemistry (use `gibbs_free_energy_eh`), solvation (`nsolv=3`,
  `run_solvator_cluster_thermo`), TDDFT, NBO (opt→NBO), TS (scan→OptTS→freq, confirm 1
  imaginary mode), CASSCF active-space sizing (Fe(III) d⁵ → CAS(5,5); AVAS for d-block).
- A capable base model lets these skill blocks shrink toward *triggers* rather than full
  tutorials; a weak one needs every protocol spelled out.

### 5.3 Error handling & conservative failure
We demand a model that **fails safely** rather than confidently producing garbage:
- **Conservative error-out**: missing/degenerate inputs → `status="error"` with a reason,
  never an invented value or a ranking over identical energies (calculator data-integrity
  checks).
- **Schema/contract fidelity**: emit only valid tool names, argument names, and types — the
  action space is defined by `openai_tools_geom.json`; `plan_validator.py` statically
  catches unknown tools, missing/hallucinated args, and artifact-reference mismatches before
  any compute. The model should ideally not produce what the validator must reject.
- **Three planning-error categories to avoid** (from the paper's Discussion): silent-null
  artifact dependency, argument-specification (TypeError), and type-coercion.

---

## 6. Infrastructure access (current)

- **SSH:** active key is `~/.ssh/qclab_auto` (passphrase-free). The old `qclab_zhang` is
  retired — its passphrase was lost and is unrecoverable. `Host qcl` in `~/.ssh/config`
  points at `qclab_auto`. On Windows use **Windows OpenSSH** (`C:/Windows/System32/OpenSSH/`),
  not Git Bash ssh.
- **`_build_mcp_server_params`** (agent.py) takes SSH key/host/command **only from the env
  file** — no hardcoded personal/VM defaults; missing vars raise `RuntimeError`.
- **Pod status:** the `zhang-ollama` pod may need recreating (`kubectl apply -f
  ~/ollama-pod.yml`); data persists on the node's hostPath.
- **Env files:** `.env` (gpt-5.2 + lab k8s, default), `.env.local` (qwen2.5:14b + lab),
  `.env.vm` (gpt-5.2 + DigitalOcean fallback). `.env` holds a plaintext API key — never echo it.

---

## 7. Key files

| File | Role |
|------|------|
| `agent.py` | REPL; plan/run loop; MCP-over-SSH setup |
| `build_graph_from_plan.py` | Builds & executes LangGraph from plan JSON (node kinds) |
| `prompts.py` | `QC-PLANNER`, `QC-CALCULATOR` (with `checks`/round-trip), `REPORTER` |
| `skills.py` | Planner domain-knowledge skill modules (method selection, pKa, …) |
| `client_helpers.py` | Geometry, state, PubChem, compound ID, arg injection |
| `plan_validator.py` | Static plan validation (tool/arg/artifact checks) before compute |
| `openai_tools_geom.json` | Tool schemas = action space + validator/grammar source |
| `server_with_product.py` | FastMCP server on the pod; all ORCA/SOLVATOR tools |

When **adding a tool**, read `new_tools.md` first (6-file checklist). When extending the
**planner's knowledge**, add a skill to `skills.py`.
