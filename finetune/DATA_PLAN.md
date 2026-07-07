# Fine-Tuning Data Plan

*Focus: (A) what data this agent can already provide, and (B) what data we still
need to tune a planner model that is genuinely suitable for this agent. Numbers
are measured from the current `test_logs/`, `runtime_reports/`, and
`bug_reports/` (June 2026).*

The target model is the **planner**: it maps
`(system prompt + matched skills + state summary + confirmed compounds + user request)`
→ a valid, verified **plan JSON**. Everything below is organized around that mapping.

---

## Part A — Data we can provide

### A1. Verified positive plans (primary SFT signal)
Source: `test_logs/` runs with `passed=True` + `runtime_reports/` with success status.

- **Format already captured:** `user_text → plan JSON`, plus `checks` and `token_usage`.
- **Volume today:** ~17 deduped *verified* plans (after removing scaling repeats).
- **Quality:** real, executor-tested; the strongest examples we have.
- **Limitation:** small and skewed (see coverage gap in B2).

### A2. Failure & repair data (negative + contrastive signal)
Source: `bug_reports/` (13 records) + plans the validator rejects.

- Each `bug_report` stores the **exact failing `node_spec`**, the `exception`, the
  `traceback`, and a `state_snapshot`.
- Measured error mix: **TypeError ×8, NoneType ×2, timeout ×2, ValueError ×1** —
  i.e. the paper's "argument-specification" and "silent-null artifact-dependency"
  categories, with ground-truth failing nodes attached.
- `plan_validator.py` adds more, e.g. four plans referencing the non-existent tool
  `geometry_to_xyz_artifact`.
- **Use:** build `(broken_node, error) → corrected_node` repair pairs and
  contrastive negatives. This is high-value and currently unused for training.

### A3. The conditioning context (input side of every example)
Source: `prompts.py` (orchestration system prompt) + `skills.py` (10 skill modules).

- Needed to reconstruct the **exact model input** for each plan — especially
  *which skills fired*, since that varies per request and changes the input.
- **Gap:** runs do not currently log the matched-skill set (see B1).

### A4. Action-space definition (output grammar)
Source: `openai_tools_geom.json` (18 tool schemas) + the plan JSON schema.

- Defines valid tool names, argument names/types, and node structure.
- Doubles as the constrained-decoding grammar and the validator's rule source.

### A5. Automatic labeler (the force multiplier)
Source: the success-checker harness + `plan_validator.py`.

- Can label **any** plan — harvested or synthetically generated — as
  pass/fail/violations, with **no human annotation**.
- This is what makes scalable data generation (B5) feasible: it converts cheap
  unlabeled generation into a verified dataset via rejection sampling.

### A6. Execution ground truth (gold tier)
Source: ORCA results in `runtime_reports/` (energies, spectra, thermochemistry).

- Lets us verify not just that a plan is *valid* but that it produced *correct
  numbers* — the basis for a small but trustworthy gold set.

---

## Part B — Data we need

### B1. Faithful input capture (prerequisite — do this first)
To train a model "suitable for this agent," each example's input must match
*production* input. Today we store `user_text` and the plan, but **not** the
matched-skill conditioning. We need to instrument the agent/test harness to log,
per run, the **full planner message stack**:

```
system prompt | matched skill names + text | state summary | confirmed compounds | user request  →  plan JSON
```

Without this, the model is trained on an input distribution that differs from
what it sees at inference. **Highest-leverage, low-cost change**, and it makes all
future runs compound into data automatically.

### B2. Coverage — the real gap
Target a balanced matrix of **task family × molecule class × phrasing**.

| Family | Verified now | Needed |
|---|---|---|
| pKa | 7 | balance up |
| single-point | 5 | balance up |
| optimization | 2 | ↑ |
| UV-Vis / TDDFT | 1 | ↑↑ |
| CASSCF | 1 | ↑↑ |
| thermochemistry / freq | 0 | from scratch |
| IR / Raman spectrum | 0 | from scratch |
| NBO | 0 | from scratch |
| PES scan | 0 | from scratch |
| TS search | 0 | from scratch |
| solvation cluster | 0 | from scratch |
| interaction scan | 0 | from scratch |
| coordination complex | 0 | from scratch |

Plus two cross-cutting axes the current data barely covers:
- **Molecule diversity per family:** neutral organics, anions/cations, radicals
  (open-shell multiplicity), transition-metal complexes, varied size.
- **Phrasing diversity per task:** several paraphrases per request, for
  robustness (the planner must not overfit one wording).
- **Composite / multi-compound requests:** the hardest reasoning cases (e.g.
  "rank these five by …"), where holistic planning matters most.

**Rough volume target:** for LoRA on a ~14B model, a few hundred to ~1k verified
examples *per major family*, balanced — i.e. ~2 orders of magnitude beyond the
current 17. Exact number to be set after a baseline (B6).

### B3. Quality tiers (label provenance, tracked per example)
1. **Gold** — ORCA-executed and results verified correct.
2. **Silver** — passes `plan_validator` + structural checker (no execution).
3. **Bronze** — schema-valid only (validator passes).
4. **Negative** — validator/checker/execution failure, with the error attached.

Train primarily on Gold+Silver; use Negative for repair pairs (B4); keep Bronze
for low-confidence augmentation only.

### B4. Negative & repair pairs (explicit target)
Build `(broken_plan, violation/error) → fixed_plan` triples from:
- `bug_reports/` (real failing nodes + exceptions),
- validator-rejected plans,
- deliberate corruption of gold plans (swap tool, wrong arg type, mismatched
  artifact key) — cheap synthetic negatives with known fixes.

Teaches the model both to avoid the three failure categories and to self-repair.

### B5. Data-generation strategy (how to close the gap)
1. **Teacher distillation + rejection sampling.** Use gpt-5.2 to generate plans
   over a deliberate molecule×task grid → filter through `plan_validator` +
   checker → keep only passing → label by tier. This is the main volume source.
2. **Paraphrase augmentation.** Rewrite each seed request N ways, re-plan, re-verify.
3. **Budgeted gold subset.** Actually execute a sampled fraction on ORCA for the
   Gold tier (expensive; reserve for held-out eval + a correctness anchor).
4. **Capture-in-production.** Once B1 lands, every real run adds a labeled example.

### B6. Splits, balance, and hygiene
- **Hold out by molecule and by phrasing**, not randomly — measures real
  generalization, not memorization.
- **Balance families** so the model doesn't collapse to pKa/SP.
- **Version each dataset** with the generator/model and validator version used.
- **Establish a baseline first:** run the checker on base local model vs gpt-5.2
  across families to size the gap before committing to volume.

### B7. Data to deliberately exclude (anti-requirements)
Tie to the harness design (the paper's "no autobiography / minimum live state"):
- **No execution transcripts** — they bloat examples and contradict the harness
  principle the system is built on.
- **Deduplicate scaling repeats** — many near-identical SP runs add little.
- **Don't bake absolute energies into planner targets** — the planner produces a
  *plan*, not numbers; numeric answers belong to execution, not the plan label.

---

## Part C — Calculator self-verification (discussion item — not yet implemented)

*Status: design discussed June 2026; `prompts.py` updated, no harvester/generator built.
Bring to collaborator before investing in data.*

### C1. Reframing: don't tune the arithmetic, tune the *checking policy*
We cannot meaningfully fine-tune the calculator to do float arithmetic more reliably —
that is `calc_expr`'s job (deterministic, exact, zero tokens), and any pure-arithmetic
calculator node should be migrated there. What *is* a legitimate fine-tune target is the
calculator's **verification behaviour**: deciding when a result is trustworthy and erroring
out conservatively when it is not.

### C2. The check method: reverse / round-trip verification
The verification a careful person does by hand: **invert the formula, substitute the output
back, and confirm it recovers the original input** within tolerance.
- e.g. forward `pKa = ΔG/(R·T·ln10)` → reverse `ΔG_rec = pKa·R·T·ln10` → compare to input ΔG.
- Non-invertible formula → recompute by an independent second path and compare.
- This emits a **residual number**, so it is **objectively gradable offline**: a labeler can
  re-derive the inverse and confirm the model's claimed residual is real. The check is
  *self-verifying* — the checker harness becomes an automatic reward with no human labeling.
- **Already in place:** `CALCULATOR_SYSTEM_PROMPT` now requires a structured `checks` list with
  `reverse_recovers_inputs` as the PRIMARY record (+ supplementary inputs_present /
  units_consistent / result_finite / result_plausible / inputs_distinct). These flow into
  `runtime_reports` via `node_results`, so future runs accumulate the raw signal.

### C3. Training objective: "effective checking" vs "checking in vain"
The contrast we want the model to learn:
- **Effective** — a check whose values *should* be distinct, whose pattern *discriminates*
  good from broken, and whose outcome *changes the action* (shows a real residual; trips on a
  real mismatch).
- **In vain** — *vacuous* (restates "I used the right formula", no residual) or *miscalibrated*
  (false halt on legitimately-close conformers; false pass on machine-precision-identical
  must-differ values).

This is naturally a **preference signal (DPO/KTO)**, not plain SFT: a vain check is a
*worse-but-plausible* completion, not invalid output. Strongest form = **contrastive near-miss
pairs** (near-identical inputs, opposite gold action) so the model attends to the discriminating
feature, not the spurious "values are close."

### C4. Data synthesis (we have no logged calculator negatives — perturb verified runs)
Generator parameterized by perturbation type, anchored on the ~7 verified pKa / multi-isomer runs:

| Example type | Construction | Gold |
|---|---|---|
| Effective halt | copy one geometry's energy into the other slot → exact equality | `error`, name pair, "duplicate geometry" |
| Proceed (near-miss) | perturb by real conformer-scale Δ (~0.2–0.5 kcal/mol) | `ok` + report |
| Vain over-halt (rejected) | the proceed case answered with a halt | dispreferred |
| Vacuous (rejected) | clean input answered with performative message spam | dispreferred |

Likely enough from 7 runs for a convincing contrastive *demo*, not a deployable checker; the
generator is the asset that scales as runs accumulate.

### C5. Eval — must be two-sided
- **False-halt rate** on clean/near-miss runs that should all proceed (catches trigger-happy).
- **Catch rate** on injected degenerate cases (catches asleep).
A single accuracy number hides this trade-off.

### C6. Honest limitation
Round-trip verifies **execution, not specification.** A consistently-wrong formula/unit recovers
the input perfectly (residual ≈ 0) and passes while the answer is wrong. The inverse check catches
arithmetic / sign / transcription slips only; `units_consistent` + `result_plausible` remain as
spec guards, and `calc_expr` is still the stronger long-run fix for the arithmetic itself.

### C7. Sequencing
Pin the calculator's output schema (Structured Outputs / `response_format`) **before** harvesting
`checks` for training — otherwise JSON-format noise (`_ALIAS_MAP` shims) contaminates the
effective-vs-vain signal.

---

## Other roles (lower priority)

- **Calculator (`llm` nodes):** verification-policy fine-tune is a live discussion — see Part C.
  Pure-arithmetic nodes should still migrate to deterministic `calc_expr`.
- **Reporter:** `(artifact dict → markdown report)`. Stylistic; harvestable from
  `runtime_reports[].final_report`, but low training priority.

---

## One-line bottom line
We can already provide a **verifier** and a trickle of verified plans + rich
failure data; what we *need* is **faithful input capture (B1)** and **balanced,
tier-labeled coverage across all task families (B2–B3)**, generated mostly by
**verifier-filtered teacher distillation (B5)**.
