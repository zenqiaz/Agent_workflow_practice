"""
sample_methods.py

Reads one or more *_methods.jsonl files, applies stratified reservoir sampling
to produce a diversity-balanced methods.jsonl.

Sampling strategy
-----------------
Bucket key: (specialist_cell, method_family, scale)
  scale derived from n_atoms: S ≤10, M 11-30, L 31-100, XL >100, unknown

Organic bucket (specialist_cell != "metal_general"):
  1. Group by upload_id
  2. Take min(UPLOAD_CAP, n) per upload   → upload-diverse pool
  3. Reservoir-sample TARGET from pool

Metal bucket (specialist_cell == "metal_general"):
  1. Group by metal element set (frozenset of TM ∪ HEAVY elements present)
  2. Per element group:
       a. Group by upload_id
       b. Take min(UPLOAD_CAP, n) per upload
       c. Take min(ELEMENT_CAP, n) from upload-capped pool
  3. Pool across element groups
  4. Reservoir-sample TARGET from pool

In all cases UPLOAD_CAP and ELEMENT_CAP apply unconditionally — even tiny
buckets with few entries are upload/element-diverse rather than dominated by
one group.

Usage
-----
    python sample_methods.py                          # default inputs + output
    python sample_methods.py --inputs orca_methods.jsonl gaussian_methods.jsonl
    python sample_methods.py --out methods.jsonl --target 300 --upload-cap 10 --element-cap 50
    python sample_methods.py --dry-run               # print stats, no write
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from canonical_ir import TM, HEAVY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale(n_atoms) -> str:
    if not n_atoms:   return "unknown"
    if n_atoms <= 10: return "S"
    if n_atoms <= 30: return "M"
    if n_atoms <= 100: return "L"
    return "XL"


def _is_metal(rec: dict) -> bool:
    """True if the entry contains any TM or heavy element — source of truth for
    metal branch selection, independent of how specialist_cell was labelled."""
    elems = set(rec.get("elements") or [])
    return bool(elems & (TM | HEAVY))


def _bucket_key(rec: dict) -> tuple:
    # Use element-based cell label so old records (TM_general, heavy_general)
    # and new records (metal_general) land in the same bucket.
    if _is_metal(rec):
        cell = "metal_general"
    else:
        cell = rec.get("specialist_cell") or "unclassified"
    family = rec.get("method_family") or "unknown"
    scale  = _scale(rec.get("n_atoms") or rec.get("n_sites_index"))
    return (cell, family, scale)


def _metal_group(rec: dict) -> frozenset:
    """Frozenset of TM/heavy elements present — the diversity axis for metals."""
    elems = set(rec.get("elements") or [])
    return frozenset(elems & (TM | HEAVY))


def _reservoir_sample(pool: list, k: int, rng: random.Random) -> list:
    """Algorithm R: uniform random sample of size min(k, len(pool)) from pool."""
    if k <= 0 or not pool:
        return []
    if len(pool) <= k:
        return list(pool)
    reservoir = pool[:k]
    for i in range(k, len(pool)):
        j = rng.randint(0, i)
        if j < k:
            reservoir[j] = pool[i]
    return reservoir


def _cap_by_key(records: list, key_fn, cap: int, rng: random.Random) -> list:
    """Group records by key_fn, shuffle each group, take min(cap, n) from each."""
    groups: dict = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out.extend(g[:cap])
    return out


# ---------------------------------------------------------------------------
# Per-bucket samplers
# ---------------------------------------------------------------------------

def _sample_organic(records: list, upload_cap: int, target: int,
                    rng: random.Random) -> list:
    pool = _cap_by_key(records, lambda r: r.get("upload_id", ""), upload_cap, rng)
    return _reservoir_sample(pool, target, rng)


def _sample_metal(records: list, upload_cap: int, element_cap: int,
                  target: int, rng: random.Random) -> list:
    # Group by metal element set
    by_elem: dict[frozenset, list] = defaultdict(list)
    for r in records:
        by_elem[_metal_group(r)].append(r)

    pool = []
    for elem_recs in by_elem.values():
        # Upload diversity within element group
        elem_pool = _cap_by_key(elem_recs,
                                lambda r: r.get("upload_id", ""),
                                upload_cap, rng)
        rng.shuffle(elem_pool)
        pool.extend(elem_pool[:element_cap])

    return _reservoir_sample(pool, target, rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                    n += 1
                except json.JSONDecodeError:
                    pass
        print(f"  Loaded {n:,} records from {p}")
    return records


def sample_all(records: list[dict], upload_cap: int, element_cap: int,
               target: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)

    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        buckets[_bucket_key(r)].append(r)

    sampled: list[dict] = []
    stats: dict[tuple, dict] = {}

    for key, recs in sorted(buckets.items()):
        is_metal = key[0] == "metal_general"  # set by _bucket_key via _is_metal()
        if is_metal:
            chosen = _sample_metal(recs, upload_cap, element_cap, target, rng)
        else:
            chosen = _sample_organic(recs, upload_cap, target, rng)

        if is_metal:
            # Record per-element counts in the sample for gap planning
            elem_counts: dict[str, int] = {}
            for r in chosen:
                for e in set(r.get("elements") or []) & (TM | HEAVY):
                    elem_counts[e] = elem_counts.get(e, 0) + 1
            elem_counts = dict(sorted(elem_counts.items(), key=lambda x: -x[1]))
            detail = f"elem_groups={len(elem_counts)}"
            extra  = {"elem_counts": elem_counts}
        else:
            uploads = len({r.get('upload_id') for r in chosen})
            detail  = f"uploads={uploads}"
            extra   = {}

        stats[key] = {
            "input":     len(recs),
            "output":    len(chosen),
            "saturated": len(chosen) >= target,
            "detail":    detail,
            **extra,
        }
        sampled.extend(chosen)

    return sampled, stats


def print_stats(stats: dict, upload_cap: int, element_cap: int, target: int) -> None:
    print(f"\n{'Bucket (cell / family / scale)':<50} {'in':>6} {'out':>6}  sat  detail")
    print("-" * 86)
    total_in = total_out = 0
    for (cell, family, scale), s in sorted(stats.items()):
        label = f"{cell} / {family} / {scale}"
        sat = "Y" if s["saturated"] else " "
        print(f"  {label:<48} {s['input']:>6} {s['output']:>6}  [{sat}]  {s['detail']}")
        total_in  += s["input"]
        total_out += s["output"]
    print("-" * 86)
    print(f"  {'TOTAL':<48} {total_in:>6} {total_out:>6}")
    print(f"\nParams: upload_cap={upload_cap}  element_cap={element_cap}  target={target}")
    print("  [Y] = saturated (bucket was larger than target; more data would help)")


def save_coverage(stats: dict, inputs: list[Path], upload_cap: int,
                  element_cap: int, target: int, out: Path) -> None:
    """Write coverage.json — machine-readable record for planning future downloads."""
    # Chemically important TM elements we want covered across metal buckets
    TARGET_TM = {
        "Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",   # 3d
        "Mo","Ru","Rh","Pd","Ag",                       # 4d
        "W","Re","Os","Ir","Pt","Au",                   # 5d
    }

    # Accumulate element counts across ALL metal buckets for global view
    all_elem_counts: dict[str, int] = {}
    for s in stats.values():
        for e, n in (s.get("elem_counts") or {}).items():
            all_elem_counts[e] = all_elem_counts.get(e, 0) + n
    covered_tm  = {e for e in all_elem_counts if e in TARGET_TM}
    missing_tm  = sorted(TARGET_TM - covered_tm)

    records = []
    for (cell, family, scale), s in sorted(stats.items()):
        rec = {
            "cell":          cell,
            "method_family": family,
            "scale":         scale,
            "n_input":       s["input"],
            "n_sampled":     s["output"],
            "saturated":     s["saturated"],
            "diversity":     s["detail"],
        }
        if "elem_counts" in s:
            rec["elem_counts"] = s["elem_counts"]
            rec["missing_tm"]  = sorted(TARGET_TM - set(s.get("elem_counts", {})))
        records.append(rec)
    payload = {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "inputs":      [str(p) for p in inputs],
        "params":      {"upload_cap": upload_cap, "element_cap": element_cap,
                        "target": target},
        "buckets":     records,
        "total_input":   sum(r["n_input"]   for r in records),
        "total_sampled": sum(r["n_sampled"] for r in records),
        "n_saturated":   sum(1 for r in records if r["saturated"]),
        "n_scarce":      sum(1 for r in records if not r["saturated"]),
        "metal_summary": {
            "present_tm":  sorted(covered_tm),
            "missing_tm":  missing_tm,
            "elem_counts": dict(sorted(all_elem_counts.items(), key=lambda x: -x[1])),
        },
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Coverage record → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified reservoir sampling of methods.jsonl sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs", nargs="+", type=Path,
        default=[Path("orca_methods.jsonl"), Path("gaussian_methods.jsonl")],
        help="Input *_methods.jsonl files (default: orca_methods.jsonl gaussian_methods.jsonl)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("methods.jsonl"),
        help="Output file (default: methods.jsonl)",
    )
    parser.add_argument(
        "--target", type=int, default=300,
        help="Max entries per bucket after sampling (default: 300)",
    )
    parser.add_argument(
        "--upload-cap", type=int, default=10,
        help="Max entries per upload_id per bucket (default: 10)",
    )
    parser.add_argument(
        "--element-cap", type=int, default=50,
        help="Max entries per metal element group per metal bucket (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing output",
    )
    parser.add_argument(
        "--coverage", type=Path, default=None,
        metavar="FILE",
        help="Write bucket coverage record to FILE (default: <out>.coverage.json)",
    )
    args = parser.parse_args()

    print("Loading records …")
    records = load_records(args.inputs)
    print(f"  Total input: {len(records):,} records\n")

    print("Sampling …")
    sampled, stats = sample_all(
        records,
        upload_cap=args.upload_cap,
        element_cap=args.element_cap,
        target=args.target,
        seed=args.seed,
    )

    print_stats(stats, args.upload_cap, args.element_cap, args.target)

    coverage_path = args.coverage or args.out.with_suffix("").with_suffix(".coverage.json")

    if args.dry_run:
        print("\n[dry-run] No output written.")
        save_coverage(stats, args.inputs, args.upload_cap, args.element_cap,
                      args.target, coverage_path)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled):,} records → {args.out}")
    save_coverage(stats, args.inputs, args.upload_cap, args.element_cap,
                  args.target, coverage_path)


if __name__ == "__main__":
    main()
