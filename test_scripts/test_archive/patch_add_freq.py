"""Add run_freq_job MCP tool to server_with_product.py (before run_solvator_cluster)."""

with open("server_with_product.py", "r") as f:
    content = f.read()

marker = """@mcp.tool()
async def run_solvator_cluster("""

new_tool = '''@mcp.tool()
async def run_freq_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA frequency calculation. Returns E, H, G (Gibbs free energy) in Eh."""
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"freq_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="freq",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        return json.dumps(
            {"status": "timeout", "label": job_label, "text": f"Status: TIMEOUT after {wall_timeout_seconds}s"}
        )
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({"status": "error", "label": job_label, "tail": "\\n".join(out_text.splitlines()[-120:])})

    energy = extract_total_energy(out_text)
    enthalpy = extract_total_enthalpy(out_text)
    gibbs = extract_gibbs_free_energy(out_text)

    return json.dumps({
        "status": "ok",
        "label": job_label,
        "energy": energy,
        "enthalpy_eh": enthalpy,
        "gibbs_free_energy_eh": gibbs,
        "product": "gibbs_free_energy_eh",
    })


'''

if marker not in content:
    print("ERROR: marker not found")
else:
    content = content.replace(marker, new_tool + marker, 1)
    with open("server_with_product.py", "w") as f:
        f.write(content)
    print("OK: added run_freq_job tool")
