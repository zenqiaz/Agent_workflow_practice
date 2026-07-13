"""
rag_eval_holdout.py

Retrieval-only held-out evaluation for rag.py: no LLM calls, no API cost.

Splits rag.load_pool() (the exact pool rag.py uses at query time -- LDA
excluded, upload_cap applied) by upload_id into train/val (same mechanism as
generate_sft.py's split_by_upload: correlated same-batch records stay
together, so val queries can never retrieve their own source deposit).
Builds the BM25 index on TRAIN records only, queries with every VAL record,
and checks whether the retrieved functional/basis actually matches the val
record's own (real, corpus-labeled) functional/basis.

This answers "does RAG find a good precedent for a held-out query at all",
independent of whether an LLM planner then chooses to use it (see the
rag_ablation_a1_*.py / rag_ablation_novel_fe.py trials for that separate
question).

Usage:
    python rag_eval_holdout.py
    python rag_eval_holdout.py --cell metal_general
    python rag_eval_holdout.py --k 5 --val-frac 0.10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict

import rag
from canonical_ir import classify_cell


def _reclassify(records: list[dict]) -> list[dict]:
    """Recompute specialist_cell from the current classify_cell() logic.
    Some pool records carry a stale specialist_cell (e.g. "TM_general", a
    label classify_cell() no longer produces -- current logic folds every
    TM_open/TM_closed/heavy_elem record into "metal_general", matching what
    rag.py's own _cell_filter already does via its system_type-based
    special case). Evaluating against the stale label would isolate these
    records into their own tiny, artificial train/val pool instead of the
    unified pool a real query actually searches."""
    for r in records:
        r["specialist_cell"] = classify_cell(
            r.get("system_type"), r.get("task_type"), r.get("method_family")
        )
    return records


def split_by_upload(records: list[dict], val_frac: float = 0.10, seed: int = 42
                     ) -> tuple[list[dict], list[dict]]:
    """Split by upload_id so correlated entries stay in the same split.
    Mirrors generate_sft.py's split_by_upload (MOSAIC project) exactly, so
    this eval uses the same train/val boundary logic as the SFT data."""
    by_upload: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        uid = r.get("upload_id") or r.get("entry_id", "")
        by_upload[uid].append(r)

    uploads = list(by_upload.keys())
    rng = random.Random(seed)
    rng.shuffle(uploads)
    n_val = max(1, int(val_frac * len(uploads)))

    val_set = set(uploads[:n_val])
    train_set = set(uploads[n_val:])
    train = [r for uid in train_set for r in by_upload[uid]]
    val = [r for uid in val_set for r in by_upload[uid]]
    return train, val


def eval_record(train_idx, rec: dict, k: int) -> dict:
    features = {
        "elements": rec.get("elements") or [],
        "n_atoms": rec.get("n_atoms"),
        "charge": rec.get("charge"),
        "multiplicity": rec.get("multiplicity"),
        "solvent": rec.get("solvent"),
        "task_type": rec.get("task_type"),
        "system_type": rec.get("system_type"),
        "specialist_cell": rec.get("specialist_cell"),
    }
    hits = rag.query(train_idx, features, k=k, query_smiles=rec.get("smiles"))

    true_func = rec.get("functional")
    true_basis = rec.get("basis")

    if not hits:
        return {"n_hits": 0, "top1_func": False, "top1_basis": False,
                "top1_exact": False, "any5_func": False, "majority_func": False}

    top1 = hits[0]
    top1_func = top1.get("functional") == true_func
    top1_basis = top1.get("basis") == true_basis
    any5_func = any(h.get("functional") == true_func for h in hits)

    funcs = Counter(h.get("functional") for h in hits)
    majority_func_label, _ = funcs.most_common(1)[0]
    majority_func = majority_func_label == true_func

    return {
        "n_hits": len(hits),
        "top1_func": top1_func,
        "top1_basis": top1_basis,
        "top1_exact": top1_func and top1_basis,
        "any5_func": any5_func,
        "majority_func": majority_func,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell", default=None, help="Restrict eval to one specialist_cell")
    args = ap.parse_args()

    pool = _reclassify(rag.load_pool())
    train_records, val_records = split_by_upload(pool, val_frac=args.val_frac, seed=args.seed)
    print(f"[eval] pool={len(pool)}  train={len(train_records)}  val={len(val_records)}", file=sys.stderr)

    if args.cell:
        val_records = [r for r in val_records if r.get("specialist_cell") == args.cell]
        print(f"[eval] restricted to cell={args.cell!r}: {len(val_records)} val records", file=sys.stderr)

    train_idx = rag.build_index(train_records)

    per_cell_results: dict[str, list[dict]] = defaultdict(list)
    t0 = time.monotonic()
    for i, rec in enumerate(val_records):
        cell = rec.get("specialist_cell") or "?"
        result = eval_record(train_idx, rec, args.k)
        per_cell_results[cell].append(result)
        if (i + 1) % 50 == 0:
            print(f"[eval] {i+1}/{len(val_records)}", file=sys.stderr)
    duration_s = time.monotonic() - t0

    def summarize(results: list[dict]) -> dict:
        n = len(results)
        if n == 0:
            return {}
        no_hit = sum(1 for r in results if r["n_hits"] == 0)
        return {
            "n": n,
            "no_hit_pct": 100 * no_hit / n,
            "top1_func_pct": 100 * sum(r["top1_func"] for r in results) / n,
            "top1_basis_pct": 100 * sum(r["top1_basis"] for r in results) / n,
            "top1_exact_pct": 100 * sum(r["top1_exact"] for r in results) / n,
            "any5_func_pct": 100 * sum(r["any5_func"] for r in results) / n,
            "majority_func_pct": 100 * sum(r["majority_func"] for r in results) / n,
        }

    print("\n" + "=" * 100)
    print(f"RETRIEVAL-ONLY HELD-OUT EVAL  (k={args.k}, val_frac={args.val_frac}, seed={args.seed}, "
          f"{duration_s:.1f}s, no LLM calls)")
    print("=" * 100)
    print(f"{'cell':<20} {'n':>5} {'no_hit%':>8} {'top1_func%':>11} {'top1_basis%':>12} "
          f"{'top1_exact%':>12} {'any5_func%':>11} {'majority_func%':>15}")

    all_results = [r for rs in per_cell_results.values() for r in rs]
    summary_by_cell = {}
    for cell in sorted(per_cell_results):
        s = summarize(per_cell_results[cell])
        summary_by_cell[cell] = s
        print(f"{cell:<20} {s['n']:>5} {s['no_hit_pct']:>7.1f}% {s['top1_func_pct']:>10.1f}% "
              f"{s['top1_basis_pct']:>11.1f}% {s['top1_exact_pct']:>11.1f}% "
              f"{s['any5_func_pct']:>10.1f}% {s['majority_func_pct']:>14.1f}%")

    overall = summarize(all_results)
    print("-" * 100)
    print(f"{'OVERALL':<20} {overall['n']:>5} {overall['no_hit_pct']:>7.1f}% "
          f"{overall['top1_func_pct']:>10.1f}% {overall['top1_basis_pct']:>11.1f}% "
          f"{overall['top1_exact_pct']:>11.1f}% {overall['any5_func_pct']:>10.1f}% "
          f"{overall['majority_func_pct']:>14.1f}%")

    out = {
        "type": "rag_eval_holdout",
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "k": args.k, "val_frac": args.val_frac, "seed": args.seed,
        "pool_size": len(pool), "train_size": len(train_records), "val_size": len(val_records),
        "duration_s": duration_s,
        "summary_by_cell": summary_by_cell,
        "overall": overall,
    }
    import os
    os.makedirs("test_logs", exist_ok=True)
    log_path = f"test_logs/rag_eval_holdout_{out['timestamp_utc']}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
