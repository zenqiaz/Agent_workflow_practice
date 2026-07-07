"""Patch server_with_product.py to add solvator validation."""

with open("server_with_product.py", "r") as f:
    content = f.read()

old = """    if not cluster_xyz_no_header:
        # Return solvator result as-is; thermo step skipped.
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Could not find cluster XYZ in SOLVATOR result."
        return json.dumps(solv_result)

    # 2) ORCA frequency job for thermochemistry"""

new = """    if not cluster_xyz_no_header:
        # Return solvator result as-is; thermo step skipped.
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Could not find cluster XYZ in SOLVATOR result."
        return json.dumps(solv_result)

    # Validate: check solvator actually added water molecules
    input_lines = [ln for ln in str(geometry_xyz).splitlines() if ln.strip()]
    cluster_lines = [ln for ln in cluster_xyz_no_header.splitlines() if ln.strip()]
    if len(cluster_lines) <= len(input_lines):
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = (
            f"SOLVATOR did not add solvent molecules: input has {len(input_lines)} atoms, "
            f"cluster has {len(cluster_lines)} atoms. Molecule may be too small for solvation."
        )
        solv_result["cluster_xyz"] = cluster_xyz_no_header
        return json.dumps(solv_result)

    # Guard: freq needs at least 2 atoms (no vibrations for a single atom)
    if len(cluster_lines) < 2:
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Cannot run frequency calculation on a single atom."
        solv_result["cluster_xyz"] = cluster_xyz_no_header
        return json.dumps(solv_result)

    # 2) ORCA frequency job for thermochemistry"""

if old not in content:
    print("ERROR: old text not found in server_with_product.py")
else:
    content = content.replace(old, new, 1)
    with open("server_with_product.py", "w") as f:
        f.write(content)
    print("OK: patched server_with_product.py with solvator validation")
