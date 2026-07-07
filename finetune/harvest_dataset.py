"""Harvest a verified planner fine-tuning dataset from existing run logs.

This is an *exploration* tool: it converts the project's existing
`test_logs/` and `runtime_reports/` into

  1. a training-ready SFT file  (planner_sft_<ts>.jsonl, OpenAI `messages` format)
  2. a corpus-analysis report    (planner_corpus_<ts>.json + stdout summary)

so we can see how much verified `(user_text -> plan JSON)` data already exists,
how it is distributed across QC task families, and where the gaps are — before
committing to any particular fine-tuning direction.

Only *verified* examples are kept (rejection sampling):
  - test_logs:        run["passed"] is True
  - runtime_reports:  final_status in {"success", "ok", "completed"}

The checker harness is the reward signal; unverified plans are discarded.

Usage:
    python finetune/harvest_dataset.py
    python finetune/harvest_dataset.py --logs test_logs --reports runtime_reports \
        --out finetune/datasets
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Task-family inference
# ---------------------------------------------------------------------------

# Signature tools, in priority order: the first match names the family.
# A plan that opt's then freq's for a deprotonation is "pka"; a plan that only
# opt+sp is "single_point", etc.
_FAMILY_SIGNATURE: tuple[tuple[str, str], ...] = (
    ("run_ts_opt_job", "ts_search"),
    ("run_casscf_job", "casscf"),
    ("run_tddft_job", "uvvis_tddft"),
    ("run_scan_job", "pes_scan"),
    ("run_spectrum_job", "ir_spectrum"),
    ("run_nbo_job", "nbo"),
    ("run_solvator_cluster_thermo", "solvation"),
    ("run_solvator_cluster", "solvation"),
    ("build_approach_scan_geometries", "interaction_scan"),
)

_DEPROTONATION_TOOL = "structure_add_remove_proton"
_FREQ_TOOL = "run_freq_job"
_OPT_TOOL = "run_opt_job"
_SP_TOOL = "run_sp_energy"

_SUCCESS_STATES = {"success", "ok", "completed", "passed"}


def _tools_in_plan(plan: dict[str, Any]) -> list[str]:
    return [n.get("tool") for n in plan.get("nodes", []) if n.get("tool")]


def infer_family(plan: dict[str, Any]) -> str:
    """Derive a coarse QC task family from the tools a plan uses."""
    tools = set(_tools_in_plan(plan))
    for tool, family in _FAMILY_SIGNATURE:
        if tool in tools:
            return family
    if _DEPROTONATION_TOOL in tools and _FREQ_TOOL in tools:
        return "pka"
    if _FREQ_TOOL in tools:
        return "thermochemistry"
    if _SP_TOOL in tools and _OPT_TOOL not in tools:
        return "single_point"
    if _OPT_TOOL in tools:
        return "optimization"
    return "other"


# ---------------------------------------------------------------------------
# Example extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    """One verified (request -> plan) training pair plus provenance."""

    user_text: str
    plan: dict[str, Any]
    family: str
    tools: tuple[str, ...]
    n_nodes: int
    n_compounds: int
    source: str          # "test_log" | "runtime_report"
    source_file: str
    token_total: int | None

    def dedup_key(self) -> str:
        """Stable hash over request + plan structure (tool sequence)."""
        sig = self.user_text.strip().lower() + "|" + ">".join(self.tools)
        return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def _user_text_of(plan: dict[str, Any], fallback: str) -> str:
    return (plan.get("user_text") or fallback or "").strip()


def _example_from_plan(
    plan: dict[str, Any],
    *,
    source: str,
    source_file: str,
    fallback_text: str,
    token_total: int | None,
) -> Example | None:
    if not isinstance(plan, dict) or not plan.get("nodes"):
        return None
    user_text = _user_text_of(plan, fallback_text)
    if not user_text:
        return None
    tools = tuple(_tools_in_plan(plan))
    compounds = plan.get("compounds") or []
    return Example(
        user_text=user_text,
        plan=plan,
        family=infer_family(plan),
        tools=tools,
        n_nodes=len(plan.get("nodes", [])),
        n_compounds=len(compounds) if isinstance(compounds, list) else 0,
        source=source,
        source_file=os.path.basename(source_file),
        token_total=token_total,
    )


def harvest_test_logs(logs_dir: str) -> list[Example]:
    out: list[Example] = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.json"))):
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        fallback = doc.get("message", "")
        for run in doc.get("runs", []):
            if not run.get("passed"):
                continue
            tokens = (run.get("token_usage") or {}).get("total")
            ex = _example_from_plan(
                run.get("plan") or {},
                source="test_log",
                source_file=path,
                fallback_text=fallback,
                token_total=tokens,
            )
            if ex:
                out.append(ex)
    return out


def harvest_runtime_reports(reports_dir: str) -> list[Example]:
    out: list[Example] = []
    for path in sorted(glob.glob(os.path.join(reports_dir, "*.json"))):
        try:
            doc = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            continue
        status = str(doc.get("final_status", "")).strip().lower()
        if status not in _SUCCESS_STATES:
            continue
        ex = _example_from_plan(
            doc.get("plan") or {},
            source="runtime_report",
            source_file=path,
            fallback_text="",
            token_total=None,
        )
        if ex:
            out.append(ex)
    return out


def dedup(examples: list[Example]) -> list[Example]:
    """Keep one example per (request, tool-sequence); prefer test_log source."""
    best: dict[str, Example] = {}
    for ex in examples:
        key = ex.dedup_key()
        prior = best.get(key)
        if prior is None or (prior.source != "test_log" and ex.source == "test_log"):
            best[key] = ex
    return list(best.values())


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

# Placeholder system message. At train time, swap in the real SYSTEM_PROMPT
# from prompts.py plus the skills that matched the request, so the model is
# conditioned exactly as in production.
_SYSTEM_PLACEHOLDER = "<<QC_PLANNER_SYSTEM_PROMPT + matched skills go here>>"


def to_sft_record(ex: Example) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PLACEHOLDER},
            {"role": "user", "content": ex.user_text},
            {"role": "assistant", "content": json.dumps(ex.plan, ensure_ascii=False)},
        ],
        "metadata": {
            "family": ex.family,
            "tools": list(ex.tools),
            "n_nodes": ex.n_nodes,
            "n_compounds": ex.n_compounds,
            "source": ex.source,
            "source_file": ex.source_file,
            "token_total": ex.token_total,
        },
    }


@dataclass
class Corpus:
    examples: list[Example] = field(default_factory=list)

    def write_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for ex in self.examples:
                fh.write(json.dumps(to_sft_record(ex), ensure_ascii=False) + "\n")

    def analysis(self) -> dict[str, Any]:
        by_family = Counter(ex.family for ex in self.examples)
        by_source = Counter(ex.source for ex in self.examples)
        molecules: Counter[str] = Counter()
        for ex in self.examples:
            for comp in ex.plan.get("compounds") or []:
                name = (comp.get("name") or "").strip().lower()
                if name:
                    molecules[name] += 1
        tokens = sorted(ex.token_total for ex in self.examples if ex.token_total)
        return {
            "n_examples": len(self.examples),
            "by_family": dict(by_family.most_common()),
            "by_source": dict(by_source),
            "n_distinct_molecules": len(molecules),
            "top_molecules": dict(molecules.most_common(15)),
            "tokens_total": _percentiles(tokens),
            "node_counts": _percentiles(sorted(ex.n_nodes for ex in self.examples)),
        }


def _percentiles(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    n = len(values)
    return {
        "n": n,
        "min": values[0],
        "p50": values[n // 2],
        "p90": values[min(n - 1, int(n * 0.9))],
        "max": values[-1],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(analysis: dict[str, Any]) -> None:
    print(f"\nVerified planner examples: {analysis['n_examples']}")
    print(f"  by source : {analysis['by_source']}")
    print(f"  molecules : {analysis['n_distinct_molecules']} distinct")
    print("\n  by task family:")
    for fam, n in analysis["by_family"].items():
        bar = "#" * min(40, n)
        print(f"    {fam:18s} {n:4d}  {bar}")
    print(f"\n  plan size (nodes): {analysis['node_counts']}")
    print(f"  planner tokens   : {analysis['tokens_total']}")
    thin = [f for f, n in analysis["by_family"].items() if n < 5]
    if thin:
        print(f"\n  UNDER-COVERED families (<5 examples): {', '.join(thin)}")
        print("  -> these need synthetic generation before training is worthwhile.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", default="test_logs", help="test_logs directory")
    ap.add_argument("--reports", default="runtime_reports", help="runtime_reports directory")
    ap.add_argument("--out", default="finetune/datasets", help="output directory")
    args = ap.parse_args()

    raw = harvest_test_logs(args.logs) + harvest_runtime_reports(args.reports)
    corpus = Corpus(examples=dedup(raw))

    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    jsonl_path = os.path.join(args.out, f"planner_sft_{ts}.jsonl")
    analysis_path = os.path.join(args.out, f"planner_corpus_{ts}.json")

    corpus.write_jsonl(jsonl_path)
    analysis = corpus.analysis()
    with open(analysis_path, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, ensure_ascii=False)

    print(f"Harvested {len(raw)} raw -> {len(corpus.examples)} deduped verified examples.")
    print(f"  SFT dataset : {jsonl_path}")
    print(f"  Analysis    : {analysis_path}")
    _print_summary(analysis)


if __name__ == "__main__":
    main()
