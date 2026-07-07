"""
rag.py

BM25 retrieval over the MOSAIC-QC method corpus. Injects "similar system ->
method chosen" few-shot examples into the QC-PLANNER prompt, as a baseline
against (and fallback alongside) the LoRA specialists.

Retrieval corpus = the full per-source pool (orca/gaussian/tmQM/ioChem
*_methods.jsonl), NOT the sampled methods.jsonl. The sampling cap in
sample_methods.py exists to balance LoRA training data; RAG wants the
diversity, not the cap.

Cell routing reuses canonical_ir.classify_system / classify_cell directly, so
a query lands in the same specialist_cell a corpus record with the same
system_type/task_type would have -- no separate classification logic to keep
in sync.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from canonical_ir import TM, HEAVY, classify_system, classify_cell
from sample_methods import _scale

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCES = (
    "orca_methods.jsonl",
    "jsonl/gaussian_methods.jsonl",
    "tmQM_methods.jsonl",
    "iochem_methods.jsonl",
)

# BM25 score floor: a hit below this is treated as noise, not a real match --
# we'd rather hand the planner zero examples than a misleading one.
MIN_SCORE = 0.5

# Free-text keyword -> canonical task_type, checked in this order (first match wins).
_TASK_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TDDFT",  ("uv-vis", "uv/vis", "absorption", "excited state", "tddft", "spectrum")),
    ("TS_OPT", ("transition state", " ts ", "activation energy", "barrier")),
    ("SCAN",   ("scan", "dihedral", "rotational barrier", "torsion")),
    ("IRC",    ("irc", "reaction path", "intrinsic reaction coordinate")),
    ("NBO",    ("nbo", "natural bond", "wiberg")),
    ("NMR",    ("nmr", "chemical shift")),
    ("FREQ",   ("pka", "acidity", "deprotonation", "gibbs", "free energy", "enthalpy", "freq")),
    ("OPT",    ("optimi",)),
)


@dataclass
class Index:
    records: list[dict]
    bm25: object  # BM25Okapi instance, or None if rank_bm25 isn't installed
    tokenized_corpus: list[list[str]]


# ---------------------------------------------------------------------------
# Corpus loading + indexing
# ---------------------------------------------------------------------------

def load_pool(paths: list[str] | None = None, base_dir: str | Path | None = None) -> list[dict]:
    """Load and concatenate the full per-source record pool."""
    base = Path(base_dir) if base_dir is not None else BASE_DIR
    records: list[dict] = []
    for rel in (paths or DEFAULT_SOURCES):
        with (base / rel).open(encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


def _record_to_tokens(rec: dict) -> list[str]:
    """Structured-field encoding of one record, tokenized for BM25."""
    return [
        *(f"elem_{e}" for e in sorted(set(rec.get("elements") or []))),
        f"task_{rec.get('task_type') or 'unknown'}",
        f"scale_{_scale(rec.get('n_atoms') or rec.get('n_sites_index'))}",
        f"charge_{rec.get('charge')}",
        f"mult_{rec.get('multiplicity')}",
        f"solvent_{rec.get('solvent') or 'none'}",
    ]


def build_index(records: list[dict]) -> Index:
    tokenized = [_record_to_tokens(r) for r in records]
    bm25 = BM25Okapi(tokenized) if BM25Okapi is not None else None
    return Index(records=records, bm25=bm25, tokenized_corpus=tokenized)


def save_index(index: Index, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(index, f)


def load_index(path: str | Path) -> Index:
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Query-side feature extraction
# ---------------------------------------------------------------------------

def _guess_task_type(user_text: str) -> str:
    text = f" {user_text.lower()} "
    for task, keywords in _TASK_KEYWORDS:
        if any(kw in text for kw in keywords):
            return task
    return "SP"


def _guess_solvent(user_text: str) -> str | None:
    text = user_text.lower()
    if "water" in text or "aqueous" in text:
        return "water"
    return None


def build_query_features(user_text: str, state: dict) -> dict:
    """
    Derive the same fields the corpus is indexed on, from an in-flight agent
    request -- pulled from existing state/geometry, not re-parsed:
      - elements/n_atoms  <- current geometry (state already tracks this)
      - charge/mult       <- state defaults
      - solvent           <- state default, else keyword guess
      - task_type         <- keyword heuristic on user_text (approximate --
                              retrieval only needs "close enough"; the planner
                              determines the real task_type later)
    system_type/specialist_cell are then derived via canonical_ir, the same
    functions the corpus itself was labelled with, so query and corpus always
    agree on what a given cell means.
    """
    geom = state.get("current_geometry") or {}
    elements = geom.get("elements") or []
    n_atoms = geom.get("n_atoms") or len(elements)
    multiplicity = state.get("multiplicity", 1)

    task_type = _guess_task_type(user_text)
    method_family = "CASSCF" if "casscf" in user_text.lower() else "DFT"
    system_type = classify_system(elements, multiplicity)["system_type"]
    specialist_cell = classify_cell(system_type, task_type, method_family)

    return {
        "elements": elements,
        "n_atoms": n_atoms,
        "charge": state.get("charge", 0),
        "multiplicity": multiplicity,
        "solvent": state.get("solvent") or _guess_solvent(user_text),
        "task_type": task_type,
        "system_type": system_type,
        "specialist_cell": specialist_cell,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _cell_filter(features: dict, records: list[dict]) -> list[int]:
    """
    Candidate indices sharing the query's specialist_cell. Widens to the
    broad system_type (organic / TM_open / TM_closed / heavy_elem) if the
    exact cell is empty -- e.g. a heavy TM with no dedicated corpus records
    still gets metal_general examples rather than nothing.
    """
    exact = [i for i, r in enumerate(records) if r.get("specialist_cell") == features["specialist_cell"]]
    if exact:
        return exact
    return [i for i, r in enumerate(records) if r.get("system_type") == features["system_type"]]


def query(index: Index, features: dict, k: int = 5, min_score: float = MIN_SCORE) -> list[dict]:
    """
    Retrieve up to k examples similar to `features`. Returns fewer than k (or
    zero) rather than padding with weak matches -- a misleading few-shot
    example is worse than none. Low/no-match queries are logged to stderr
    with their cell, which doubles as a signal for where the corpus needs
    more data.
    """
    if index.bm25 is None:
        raise RuntimeError("rank_bm25 not installed -- pip install rank_bm25")

    candidate_idx = _cell_filter(features, index.records)
    if not candidate_idx:
        print(f"[rag] no candidates at all for cell={features['specialist_cell']!r} "
              f"system_type={features['system_type']!r}", file=sys.stderr)
        return []

    q_tokens = _record_to_tokens(features)
    scores = index.bm25.get_scores(q_tokens)
    ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)
    hits = [i for i in ranked[:k] if scores[i] >= min_score]

    if len(hits) < k:
        print(f"[rag] only {len(hits)}/{k} hits above min_score={min_score} "
              f"for cell={features['specialist_cell']!r}", file=sys.stderr)

    return [index.records[i] for i in hits]


def format_examples(records: list[dict]) -> str:
    """Few-shot block for injection into the QC-PLANNER system prompt."""
    if not records:
        return ""

    blocks = ["# Retrieved examples (similar systems)"]
    for n, r in enumerate(records, 1):
        formula = r.get("formula") or "".join(sorted(set(r.get("elements") or [])))
        method = r.get("functional") or r.get("method_family")
        extras = ", ".join(filter(None, [r.get("ri_approx"), r.get("dispersion"), r.get("solvent_model")]))
        blocks.append(
            f"## Example {n}\n"
            f"System: {formula}, {r.get('n_atoms')} atoms, charge {r.get('charge')}, {r.get('task_type')}\n"
            f"Method chosen: {method}/{r.get('basis')}" + (f", {extras}" if extras else "")
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CLI: build the index, optionally smoke-test a query
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the RAG index over the full method-record pool")
    parser.add_argument("--base-dir", default=None, help="Directory containing the *_methods.jsonl sources")
    parser.add_argument("--out", default="jsonl/rag_index.pkl")
    parser.add_argument("--query-text", default=None, help="If given, run a test query and print the result")
    args = parser.parse_args()

    pool = load_pool(base_dir=args.base_dir)
    print(f"Loaded {len(pool)} records from {len(DEFAULT_SOURCES)} sources")

    idx = build_index(pool)
    if idx.bm25 is None:
        print("WARNING: rank_bm25 not installed -- index built but query() will raise. "
              "pip install rank_bm25", file=sys.stderr)

    out_path = (Path(args.base_dir) if args.base_dir else BASE_DIR) / args.out
    save_index(idx, out_path)
    print(f"Saved index to {out_path}")

    if args.query_text and idx.bm25 is not None:
        features = build_query_features(args.query_text, state={})
        print("Query features:", features)
        hits = query(idx, features, k=5)
        print(f"{len(hits)} hit(s):\n")
        print(format_examples(hits))
