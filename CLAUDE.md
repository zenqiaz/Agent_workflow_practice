# QC Agent Project — CLAUDE.md

## Project Overview

An agentic quantum-chemistry (QC) workflow system that orchestrates ORCA calculations
via a Model Context Protocol (MCP) server running on a remote Linux VM (DigitalOcean
droplet). The local client (Windows) sends natural-language requests, plans workflows
as JSON, and executes them deterministically via a LangGraph graph.

---

## Architecture

```
Local (Windows)                       Lab k8s (zhang-ollama pod, default)
─────────────────────────────────     ─────────────────────────────────
agent.py  (REPL)         server_with_product.py  (FastMCP)
  │  OpenAI gpt-5.2 (planner)           │  ORCA 6.1.1 + OPI
  │  LangGraph orchestrator             │  SOLVATOR
  │  MCP stdio over SSH                 │  36-CPU node, fast ORCA
  └──────── SSH (kubectl exec) ────────►│  /data/zhang/ollama/
```

### Key Files (local — `D:/brick/D/20260217/working/`)

| File | Role |
|------|------|
| `agent.py` | Main REPL entrypoint; plan/run loop |
| `build_graph_from_plan.py` | Builds & executes LangGraph from JSON plan |
| `client_helpers.py` | Geometry helpers, state management, PubChem, plan UI, compound identification |
| `prompts.py` | General system prompts: QC-PLANNER (structural rules only), QC-CALCULATOR, REPORTER |
| `skills.py` | Planner skill modules — domain chemistry injected as context (see Skills section) |
| `geometry_helpers.py` | XYZ parsing, proton add/remove, bond inference |
| `orchestrator_reporter.py` | Bug/runtime report persistence (JSON+MD) |
| `openai_tools_geom.json` | OpenAI tool schemas for all MCP + client-side tools |
| `server_with_product.py` | FastMCP server (runs on VM); all ORCA/SOLVATOR tools |
| `server_helpers.py` | Low-level OPI helpers (shared by server) |

### Key Directories

| Directory | Contents |
|-----------|----------|
| `archive/` | Old test/patch scripts (do not use) |
| `bug_reports/` | Auto-generated per-node error reports (JSON + MD) |
| `runtime_reports/` | Auto-generated per-run reports (JSON + MD) |

---

## Backend Connections

All connection parameters are in env files loaded via `ENV_FILE` (default: `.env`).
Infrastructure details live in `D:\brick\D\lab_local_llm\working\CLAUDE.md`.

| Config | LLM | ORCA server | Use when |
|--------|-----|-------------|----------|
| `.env` **(default)** | OpenAI `gpt-5.2` | Lab k8s `zhang-ollama` pod | Normal development |
| `.env.local` | Ollama `qwen2.5:14b` via `localhost:11434` | Lab k8s `zhang-ollama` pod | Local LLM testing (needs port-forward) |
| `.env.vm` | OpenAI `gpt-5.2` | DigitalOcean VM `188.166.232.163` | Fallback if lab cluster unavailable |

---

## MCP Tools (exposed by server)

| Tool | Description |
|------|-------------|
| `run_sp_energy` | Single-point ORCA energy |
| `run_opt_job` | Geometry optimization with ORCA |
| `run_freq_job` | Frequency + thermochemistry (E/H/G in Eh) |
| `run_nbo_job` | NBO analysis |
| `run_solvator_cluster` | Build explicit-solvent cluster via SOLVATOR |
| `run_solvator_cluster_thermo` | SOLVATOR cluster + ORCA freq in one call |
| `structure_add_remove_proton` | Add/remove proton from geometry |

### Client-Side Tools (handled locally, not via MCP)

| Tool | Description |
|------|-------------|
| `name_to_geometry_xyz` | Resolve name → XYZ via OPSIN/PubChem |
| `pubchem_get_basic_properties` | Fetch formula/MW/CID from PubChem |
| `state_update` | Update session state |
| `state_get_tool_args` | Inject geometry + defaults into tool args |

---

## Workflow Plan JSON Schema

The planner LLM (`QC-PLANNER`) outputs a plan JSON consumed by `build_graph_from_plan`.

```json
{
  "user_text": "...",
  "name": "plan_name",
  "version": "1.3",
  "geom_ids": ["ha_neutral", "a_anion"],
  "artifacts_to_save": ["G_HA_eh", "G_A_minus_eh", "pka"],
  "settings": { "G_H_plus_ref_eh": -0.01372 },
  "nodes": [ ... ],
  "final_report": { "format": "markdown", "fields": [...] }
}
```

### Node Kinds

| Kind | Description |
|------|-------------|
| `tool` | Calls an MCP or client-side tool; result stored in `artifacts` |
| `llm` | Calculator LLM (`QC-CALCULATOR`) computes derived quantities (pKa, ΔG) |
| `calc_expr` | Safe AST-evaluated math expression with artifact inputs |

### Artifact Flow

- Tool `product` dict maps artifact keys → tool result field paths.
- `artifacts_to_save` lists keys expected in final `state["artifacts"]`.
- LLM nodes use `needs_artifacts` to gather values and `product` to store results.

---

## Pre-Planning Pipeline (REPL flow)

Every non-command user input goes through two phases before the planner LLM is called:

```
user text
  ├─ Compound identification (client_helpers.py)
  │    ├─ Source 1: extract names via LLM → PubChem/OPSIN lookup → 2D image → user confirms
  │    └─ Source 2: geometries in state with no identity → XYZ bond detection → PubChem → confirm
  │
  └─ Skill dispatch (skills.py)
       └─ run_planning_skills(user_text, state) → matched skill blocks injected as system messages
```

Planner message stack (in order):
```
[system] SYSTEM_PROMPT            ← workflow rules, JSON schema, node kinds
[system] SKILL: method_selection  ← always active
[system] SKILL: pka_calibration   ← if pKa keywords detected
[system] SKILL: thermochemistry   ← if ΔG/freq keywords detected
[system] SKILL: solvation         ← if solvation keywords detected
[system] SKILL: nbo_analysis      ← if NBO keywords detected
[system] STATE: ...               ← loaded geometries summary
[system] CONFIRMED_COMPOUNDS: ... ← confirmed compound cards (formula, SMILES, charge)
[user]   <user request>
```

---

## Planner Skills (`skills.py`)

Skills inject domain-specific chemical reasoning into the planner on demand.
`SYSTEM_PROMPT` contains only orchestration rules; all chemistry lives in skills.

| Skill | Priority | Trigger keywords | Content |
|-------|----------|-----------------|---------|
| `MethodSelectionSkill` | 10 | always | DFT method/basis recommendations by task; adapts to loaded geometry atom count |
| `PKaSkill` | 20 | pka, acidity, deprotonation, … | Full pKa protocol: G(H⁺) ref (−0.01372 Eh), formula, systematic error, calibration benchmarks |
| `ThermochemistrySkill` | 25 | gibbs, free energy, enthalpy, freq, … | What `run_freq_job` returns (E/H/G), when to use G vs E, unit conventions |
| `SolvationSkill` | 30 | solv, aqueous, microsolvat, nsolv, … | `nsolv` selection guide, `run_solvator_cluster_thermo` pattern, artifact naming |
| `NBOSkill` | 40 | nbo, natural bond, wiberg, … | opt→NBO pattern, what the output contains, product field mapping |

### Extending skills

Add a new skill class to `skills.py` and append an instance to `SKILL_REGISTRY`:

```python
class MySkill(PlannerSkill):
    name = "my_skill"
    priority = 35
    def matches(self, user_text, state): ...
    def render(self, user_text, state) -> str: ...

SKILL_REGISTRY.append(MySkill())
```

---

## Compound Identification (`client_helpers.py`)

Helpers called before planning to show the user what molecule the agent will work on:

| Function | What it does |
|----------|-------------|
| `extract_compound_names_llm(text, client)` | LLM extracts compound names from free-form text |
| `fetch_compound_card(name)` | PubChem lookup; OPSIN fallback for SMILES |
| `fetch_compound_card_from_xyz(xyz, geom_id, charge)` | `DetermineBonds` → SMILES → PubChem lookup from loaded geometry |
| `render_compound_image_rdkit(smiles, name)` | SMILES → RDKit 2D PNG → `os.startfile` |
| `render_xyz_image_rdkit(xyz, label, charge)` | XYZ → `DetermineBonds` → RDKit 2D PNG → open |
| `display_and_confirm_compound(card, client)` | Print card, open image, prompt `[y / n / rename to <x>]` |
| `compounds_to_planner_context(compounds)` | Format confirmed cards as `CONFIRMED_COMPOUNDS` string |

Structure images cached at: `%TEMP%\qcagent_structures\<name>.png`

---

## REPL Commands

| Command | Effect |
|---------|--------|
| `load <path>` | Load XYZ file as current geometry |
| `state` | Print current session state |
| `planimg` | Visualize current plan as image |
| `plansave <path>` | Save plan JSON to file |
| `planload <path>` | Load plan from JSON file |
| `run` | Execute current/approved plan via LangGraph |
| `quit` / `exit` | Exit REPL |
| (any other text) | Send to QC-PLANNER to generate a new plan |

---

## LLM Models Used

All roles use the same model, configured via `LLM_MODEL` env var (default: `gpt-4.1-mini`).

| Role | Default model |
|------|--------------|
| Planner | `gpt-4.1-mini` |
| Calculator (pKa etc.) | `gpt-4.1-mini` |
| Reporter | `gpt-4.1-mini` |

### Local Ollama deployment

Set `ENV_FILE=.env.local` to switch to **qwen2.5:14b** on the k8s `zhang-ollama` pod.

Requires `~/.ssh/config` entry for `qcl` (see `D:\brick\D\lab_local_llm\qcl_LLM_connection.md`).

Port-forward setup:
```bash
# Step 1 — on qcl login node (run once; already persisted as nohup):
nohup kubectl port-forward -n ns-general pod/zhang-ollama 11434:11434 > /tmp/ollama-pf.log 2>&1 &

# Step 2 — on laptop (keep this terminal open while using the agent):
ssh -L 11434:localhost:11434 qcl -N
```

Verify tunnel is alive: `curl http://localhost:11434/v1/models`

Then run the agent:
```bash
ENV_FILE=.env.local python agent.py
```

---

## Key Chemistry Context

> Full protocols now live in `skills.py` and are injected into the planner at runtime.
> The notes below are quick-reference summaries for developers.

- **pKa workflow**: load HA → remove proton → `run_freq_job` on HA and A⁻ → pKa via ΔG = G(A⁻) + G(H⁺_ref) − G(HA) → pKa = ΔG(J/mol) / (RT ln10). See `PKaSkill`.
- **Reference proton free energy**: `G_H_plus_ref_eh = -0.01372` Eh (Tissandier et al.; always put in plan `settings`)
- **Solvation**: `run_solvator_cluster_thermo` builds explicit water cluster + ORCA freq in one call; `nsolv=3` is standard. See `SolvationSkill`.
- **Thermochemistry**: `run_freq_job` returns `energy_eh`, `enthalpy_eh`, `gibbs_free_energy_eh`. Always use `gibbs_free_energy_eh` for ΔG/pKa. See `ThermochemistrySkill`.
- **Geometry storage**: XYZ without natoms/comment header in `state["geometries"]`; keyed by `geom_id`

---

## Deployment Workflow

When updating the server (lab k8s pod — primary):
1. Edit local `server_with_product.py` / `server_helpers.py`
2. Copy to pod (requires `~/.ssh/config` entry for `qcl`):
```bash
scp server_with_product.py qcl:/tmp/
ssh qcl "kubectl cp /tmp/server_with_product.py ns-general/zhang-ollama:/data/zhang/ollama/nbo_agent/server_with_product.py"
```

When updating the server (DigitalOcean VM — fallback):
```bash
scp -i C:/Users/zrqrc/.ssh/droplet1 <file> root@188.166.232.163:/root/nbo_agent/
```

---

## Common Issues & Notes

- **Slow freq jobs**: freq calculations on microsolvation clusters can take 30+ min; this is expected
- **Geometry keys**: must be stable, human-readable identifiers matching `product` keys in plan nodes
- **Charge/multiplicity**: never hardcode in plan args; the executor injects from state defaults
- **Solvator validation**: `run_solvator_cluster_thermo` checks that cluster actually gained atoms vs. input; returns error if not
- **Bug reports**: auto-saved to `bug_reports/` on node exceptions; runtime reports to `runtime_reports/`

---

## Tool Creating
When making new tools, please read new_tools.md first rather than reading the project files directly.

---

## Running Experiments

### Prefer test_success_rate.py
Always use `test_success_rate.py` for experiment runs when possible — it already records token
consumption, reporter response, plan JSON, and per-check results in structured JSON logs under
`test_logs/`.

### When a new script is required
If a new experiment script must be written (e.g. `run_phph_experiment.py`,
`test_context_scaling.py`), it **must** capture and persist:

| Field | Why |
|-------|-----|
| `token_usage` (planner, calculator, reporter, total) | Cost tracking and context scaling analysis |
| `reporter_response` / final report text | Qualitative result validation |
| `plan` JSON | Reproducibility; lets checkers re-run offline |
| `duration_ms` per node / wall time total | Performance benchmarking |
| `checks` dict (per-check pass/fail) | Structured quality signal |
| `model` + `env_file` used | Reproducibility across configs |

Save logs as JSON to `test_logs/` with a timestamped filename.
Use `time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())` for the timestamp.

---

# RAG Session Handoff — MOSAIC-for-QC

> Copied in verbatim from `D:\brick\D\202606\working\RAG_CLAUDE.md` (dataset state as of
> 2026-07-07). Note: file paths below (e.g. `something_like MOSIAC/...`, `agent.py` in this
> section's context) refer to the `D:\brick\D\202606\working\` project, a separate codebase
> from the QC-Agent project documented above. This section covers RAG-based method-selection
> design work, not this repo's ORCA/MCP pipeline.

## 1. What we have

### 1.1 Record pool (36,328 records, 4 sources)

| Source | Records | Notes |
|--------|---------|-------|
| ORCA / NOMAD | 12,922 | 63 uploads; PBE0, DLPNO-CCSD(T), WB97X-D3, BP86 |
| Gaussian / NOMAD-CCCBDB | 4,171 | 51 uploads; B3LYP, BLYP, CCSD, QCISD, M06-L, 200+ functionals |
| tmQM (CSD Pd/Pt complexes) | 18,971 | 1 source; TPSSh-D3BJ/def2-SVP SP only |
| ioChem-BD (catalysis) | 264 | 1 source; B3LYP / PBE |

Parsed record schema (every source, same fields):
```json
{
  "entry_id": "...", "upload_id": "...",
  "functional": "PBE0", "basis": "def2-SVP",
  "task_type": "SP",
  "n_atoms": 14, "elements": ["C","H","N","O"],
  "charge": 0, "multiplicity": 1,
  "solvent": "", "dispersion": "D3BJ"
}
```

### 1.2 Sampled corpus — `jsonl/methods.jsonl` (3,668 records)

Produced by `sample_methods.py` with `upload_cap=20`, `element_cap=50`, `target=300`.
Bucket key: `(specialist_cell, method_family, scale)`.

Key buckets (saturated [Y] = hit 300-record cap):
```
organic_general / DFT / M    300 [Y]   organic SP/OPT, hybrid, medium mol
organic_TDDFT   / DFT / M    300 [Y]   organic TDDFT, PBE0 + range-sep
metal_general   / DFT / L    300 [Y]   TM/heavy, TPSSh (tmQM Pd/Pt)
metal_general   / DFT / S    300 [Y]   TM/heavy, PBE0/B3LYP small cplx
organic_general / DFT / S    191       organic, small, hybrid/GGA
organic_general / DFT / L    180       organic, large, hybrid
organic_general / CCSD / S   161       organic, correlated
metal_general   / CCSD / S   184       TM/heavy, DLPNO-CCSD(T)/MP2
...  + 25 smaller buckets
```

TM elements covered (14): Au Co Cr Cu Fe Mn Mo Ni Pd Pt Rh Ti V Zn
TM still missing: Ag Ir Os Re Ru W

### 1.3 TDDFT corpus (pending parse)

82,665 ORCA TDDFT entries downloaded (35 uploads, download complete 2026-07-07).
Files are in `orca_corpus/organic_TDDFT/`. They have not yet been parsed.
After parsing + resampling, `organic_TDDFT/DFT/M` is expected to grow from 3K → ~30K.

Run when ready:
```bash
python nomad_parse_corpus.py --corpus orca_corpus --out jsonl/orca_methods.jsonl
python sample_methods.py --inputs jsonl/orca_methods.jsonl jsonl/gaussian_methods.jsonl \
    tmQM_methods.jsonl iochem_methods.jsonl \
    --out jsonl/methods.jsonl --upload-cap 20 --element-cap 50 --target 300
```

---

## 2. Expert architecture (3 LoRA specialists)

| Expert | Trigger | Training data |
|--------|---------|---------------|
| `organic_general` | organic (C/H/N/O/F only), SP / OPT / Thermo | organic_general buckets, ~1,550 records |
| `organic_TDDFT`   | organic, excited-state / spectroscopy | organic_TDDFT buckets, ~404 records (will grow after TDDFT parse) |
| `metal_general`   | any TM or heavy element (Br+), any task | metal_general buckets, ~1,700 records |

Routing is deterministic (`canonical_ir.py : classify_cell()`):
- Check `elements` for TM/heavy → `metal_general`
- Check `task_type` == TDDFT AND organic → `organic_TDDFT`
- Otherwise → `organic_general`

SFT files (already generated):
- `jsonl/train_organic_general.jsonl`
- `jsonl/train_organic_TDDFT.jsonl`
- `jsonl/train_metal_general.jsonl`

---

## 3. Coverage grid and RAG scope

The 4×4 architecture (system × task) maps to the 3 experts as follows.
Color coding from the group presentation (slide 9):

```
                  SP / OPT          Thermo           Spectroscopy      TS / Scan
Organic           [SFT] organic_gen  → organic_gen    [SFT] org_TDDFT   skills.py
TM closed-shell   [SFT] metal_gen    → metal_gen      [RAG] no corpus   skills.py
Open-shell        [SFT] metal_gen    → metal_gen      [RAG] scarce      skills.py
Heavy element     [RAG] missing TM   → metal_gen      no data           skills.py
```

**[SFT]** = trained expert covers this cell.
**[RAG]** = retrieval fallback; enough pool records exist but no dedicated expert.
**skills.py** = TS/Scan bypassed entirely — handled by hard-coded protocol rules, not a model.

RAG serves two roles:
1. **Baseline** — evaluate whether a simple BM25 retrieval (k=5 in-context examples) can
   match or challenge the trained LoRA specialists. This is the key comparison the paper needs.
2. **Fallback** — for amber cells (TM spectroscopy, heavy SP/OPT, open-shell spectroscopy),
   RAG provides a functional path even without a trained expert.

---

## 4. RAG architecture sketch

### 4.1 Retrieval corpus

Use the **full pool** (36,328 records), not the sampled 3,668.
Rationale: RAG benefits from diversity; the sampling cap exists to prevent training imbalance,
not to limit retrieval.

Filter at query time: retrieve only from the same `specialist_cell` as the query
(i.e. do not retrieve metal examples for an organic query).

### 4.2 Query representation

For each incoming request, the planner already computes:
- `elements`: list of element symbols
- `task_type`: SP / OPT / TDDFT / FREQ / SCAN / TS
- `n_atoms`: integer
- `charge`, `multiplicity`
- `solvent`: CPCM / SMD / ""

Encode these as a structured string for BM25:
```
elements:C H N O  task:SP  scale:M  charge:0  mult:1  solvent:CPCM
```

For dense retrieval: embed using a small chemistry-aware model (e.g. ChemBERTa or
sentence-transformers on the structured string). Start with BM25; upgrade if needed.

### 4.3 Retrieved examples → prompt

Retrieved k=5 records are formatted as few-shot examples in the planner system message:

```
# Retrieved examples (similar systems)
## Example 1
System: C14H20N2O3, 39 atoms, charge 0, SP
Method chosen: PBE0/def2-TZVP, RIJCOSX, D3BJ

## Example 2
...
```

Integration point: `prompts.py` — inject after skills.py blocks, before the user request.

### 4.4 Retrieval index

Build with `rank_bm25` (pure Python, no server needed) or `faiss` for dense.
Index file: `jsonl/rag_index.pkl` — rebuild whenever `methods.jsonl` is updated.

Key functions to implement:
```python
build_index(records: list[dict]) -> Index
query(index: Index, query: dict, k: int, cell_filter: str) -> list[dict]
format_examples(records: list[dict]) -> str   # → few-shot block for prompts.py
```

### 4.5 Evaluation

Hold-out split: 10% stratified by `(specialist_cell, method_family, scale)` — already
planned in `sample_methods.py` but not yet written.

Metrics:
- **Primary**: ORCA run success rate on held-out queries (end-to-end via Q-Planner)
- **Secondary**: functional/basis exact match against ground-truth labels

Baselines to compare:
| Condition | Description |
|-----------|-------------|
| Zero-shot | Base model, no examples |
| Few-shot k=5 | Random 5 examples from same cell |
| RAG BM25 k=5 | Retrieved 5 nearest by BM25 |
| RAG dense k=5 | Retrieved 5 nearest by embedding |
| LoRA expert | Fine-tuned specialist (the trained model) |

The critical claim: **LoRA expert > RAG BM25** for in-distribution cells.
Expected RAG advantage over zero-shot: large. Expected LoRA advantage over RAG: moderate
but consistent, especially for edge cases (charge ≠ 0, unusual functional, large scale).

---

## 5. Key files (RAG/MOSAIC project — under `D:\brick\D\202606\working\`)

| File | Role |
|------|------|
| `something_like MOSIAC/jsonl/methods.jsonl` | Sampled corpus (3,668 records); RAG retrieval source |
| `something_like MOSIAC/jsonl/orca_methods.jsonl` | Full ORCA pool (12,922 records) |
| `something_like MOSIAC/canonical_ir.py` | `classify_cell()`, `TM`, `HEAVY` sets, functional family |
| `something_like MOSIAC/generate_sft.py` | SFT format; see `EXCLUDED_FUNCTIONALS` |
| `something_like MOSIAC/sample_methods.py` | Stratified sampler; needs `--split 0.9` flag for train/test |
| `something_like MOSIAC/sampling_process.md` | Full sampling pipeline documentation |
| `agent.py` (in `202606/working`) | Q-Planner REPL; RAG injection point is `_build_system_prompt()` |
| `prompts.py` (in `202606/working`) | `QC-PLANNER` system prompt; skills blocks injected here |
| `skills.py` (in `202606/working`) | Hard-coded domain rules; RAG replaces/augments these |

---

## 6. Immediate next steps for this session

1. **Implement `rag.py`** — `build_index()`, `query()`, `format_examples()`
   - Start with BM25 (`rank_bm25` package)
   - Use full pool (36,328 records) as retrieval corpus
   - Cell filter: only retrieve within same `specialist_cell`

2. **Add `--split` to `sample_methods.py`** — produce `methods_train.jsonl` +
   `methods_test.jsonl` (10% stratified hold-out) for evaluation

3. **Wire into `prompts.py`** — inject retrieved examples block into `QC-PLANNER` system
   prompt, after existing skills blocks

4. **Offline evaluation script** — given test set queries, score functional/basis match
   under zero-shot / few-shot / RAG / LoRA conditions

5. **After TDDFT parse** — re-run `sample_methods.py`, rebuild RAG index;
   `organic_TDDFT/DFT/M` will jump from 300 to 300 (saturated) but retrieval pool grows 8x