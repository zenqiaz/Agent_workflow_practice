"""
xyz.py — Extract XYZ geometries from an orchestrator runtime report (.md or .json).

Usage:
    python xyz.py orchestrator_20260312T100416Z.md
    python xyz.py orchestrator_20260312T100416Z.json

Writes one <geom_id>.xyz file per unique geometry found, in the same directory
as the input file (or current directory if the input has no parent).

XYZ source priority (most authoritative first):
  1. geometries dict in each node result (keyed by geom_id, e.g. "pt_en2_opt")
  2. last_tool_result.geometry_xyz / final_geometry_xyz (fallback label = node_id)
"""

import json
import re
import sys
from pathlib import Path


# ─── helpers ──────────────────────────────────────────────────────────────────

def _normalise_xyz_body(raw: str) -> list[str]:
    """Return list of 'El x y z' lines from a raw geometry_xyz string."""
    lines = []
    for line in raw.strip().splitlines():
        parts = line.split()
        if len(parts) == 4:
            el, x, y, z = parts
            lines.append(f"{el}  {float(x): .10f}  {float(y): .10f}  {float(z): .10f}")
    return lines


def _write_xyz(path: Path, label: str, body_lines: list[str]) -> None:
    n = len(body_lines)
    with open(path, "w") as fh:
        fh.write(f"{n}\n")
        fh.write(f"{label}\n")
        fh.write("\n".join(body_lines) + "\n")
    print(f"  wrote {path}  ({n} atoms)")


def _is_valid_xyz_str(s: str) -> bool:
    if not isinstance(s, str) or not s.strip():
        return False
    lines = [l for l in s.strip().splitlines() if l.strip()]
    return len(lines) >= 1 and len(lines[0].split()) == 4


# ─── extraction ───────────────────────────────────────────────────────────────

def extract_geometries(node_results: dict) -> dict[str, str]:
    """
    Return {geom_id: xyz_string} from node_results dict.
    Collects from .geometries dicts (preferred) and .last_tool_result.geometry_xyz (fallback).
    """
    collected: dict[str, str] = {}

    for node_id, node_data in node_results.items():
        if not isinstance(node_data, dict):
            continue

        # Source 1: geometries dict — most complete, keyed by geom_id
        geoms = node_data.get("geometries") or {}
        for geom_id, xyz in geoms.items():
            if _is_valid_xyz_str(xyz) and geom_id not in collected:
                collected[geom_id] = xyz

        # Source 2: last_tool_result.geometry_xyz / final_geometry_xyz — fallback
        ltr = node_data.get("last_tool_result") or {}
        if isinstance(ltr, dict):
            for field in ("final_geometry_xyz", "geometry_xyz"):
                xyz = ltr.get(field)
                if _is_valid_xyz_str(xyz):
                    # Use node_id as label if not already captured under a geom_id
                    label = f"{node_id}_{field.replace('_xyz', '')}"
                    if not any(v == xyz for v in collected.values()):
                        collected[label] = xyz
                    break  # prefer final_geometry_xyz

    return collected


def parse_md(md_path: Path) -> dict:
    """Extract node_results from ## Node results tail JSON block in the .md file."""
    text = md_path.read_text(encoding="utf-8")

    # Find the ```json block under "## Node results tail"
    pattern = r"##\s+Node results tail\s*\n```json\s*\n(.*?)```"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        # Try companion .json file
        json_path = md_path.with_suffix(".json")
        if json_path.exists():
            return parse_json(json_path)
        raise ValueError("No '## Node results tail' JSON block found in the .md file "
                         "and no companion .json file exists.")
    return json.loads(m.group(1))


def parse_json(json_path: Path) -> dict:
    """Extract node_results from the orchestrator .json report."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    # The JSON report stores node results under "node_results" or at top level
    node_results = data.get("node_results")
    if isinstance(node_results, dict):
        return node_results
    # Fallback: the whole object might be a flat node_results dict
    return data


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python xyz.py <orchestrator_report.md|.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    out_dir = input_path.parent if input_path.parent != Path(".") else Path(".")

    print(f"Reading: {input_path}")
    if input_path.suffix == ".json":
        node_results = parse_json(input_path)
    else:
        node_results = parse_md(input_path)

    geometries = extract_geometries(node_results)

    if not geometries:
        print("No geometries found.")
        sys.exit(0)

    print(f"Found {len(geometries)} geometry/geometries:\n")
    for geom_id, xyz_str in geometries.items():
        body = _normalise_xyz_body(xyz_str)
        if not body:
            print(f"  skipped {geom_id!r} (empty after parsing)")
            continue
        out_path = out_dir / f"{geom_id}.xyz"
        _write_xyz(out_path, geom_id, body)


if __name__ == "__main__":
    main()
