"""Fix _extract_cluster_geometry_from_workdir to prefer .solvator.xyz files."""

with open("server_helpers.py", "r") as f:
    content = f.read()

old = '''    # Prefer exact match, then shorter names (heuristic)
    candidates = sorted(
        {p for p in candidates if p.exists()},
        key=lambda p: (p.name != f"{label}.xyz", p.name != f"{label}.trj", len(p.name), p.name),
    )'''

new = '''    # Prefer: solvator output > exact label match > others
    # "solute.xyz" is the input; ".solvator.xyz" is the cluster output
    candidates = sorted(
        {p for p in candidates if p.exists()},
        key=lambda p: (
            "solvator" not in p.name,       # solvator files first
            p.name == "solute.xyz",          # input file last
            p.name != f"{label}.xyz",
            p.name != f"{label}.trj",
            len(p.name),
            p.name,
        ),
    )'''

if old not in content:
    print("ERROR: old text not found")
else:
    content = content.replace(old, new, 1)
    with open("server_helpers.py", "w") as f:
        f.write(content)
    print("OK: fixed cluster extraction to prefer solvator output")
