import asyncio
import json
import os
import shutil
import sys
import re
from pathlib import Path
from typing import Literal, Optional, Dict, List

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import shutil
from types import SimpleNamespace



# ---- OPI imports ----
from opi.core import Calculator
from opi.input.structures.structure import Structure
from opi.input.blocks.block_scf import BlockScf
from opi.input.blocks.block_geom import BlockGeom

mcp = FastMCP("orca_tools_opi")

# ---- functions from helper ----

from server_helpers import (
    extract_total_energy,
    extract_nbo_section,
    extract_final_geometry_from_out,
    _ensure_clean_dir,
    _xyz_with_header,
    _set_charge_mult,
    _build_calc,
    _read_out_text,
    _run_calc_with_timeout,
    _run_solvator_sync,
    sanitize_label,
)

from geometry_helpers import (
    structure_proton_edit,
)

# -----------------------------
# Small text parsers (keep simple)
# -----------------------------
def extract_total_energy(output_text: str) -> Optional[float]:
    # ORCA usually prints: "FINAL SINGLE POINT ENERGY     -76.4..."
    for line in reversed(output_text.splitlines()):
        if "FINAL SINGLE POINT ENERGY" in line.upper():
            parts = line.split()
            try:
                return float(parts[-1])
            except Exception:
                return None
    return None


def extract_nbo_section(output_text: str) -> str:
    # Very conservative: return a chunk containing "NBO" and "NPA"/"NATURAL POPULATIONS"
    lines = output_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        u = line.upper()
        if ("NBO" in u and "ANALYSIS" in u) or ("NATURAL POPULATIONS" in u) or ("NPA" in u and "NATURAL" in u):
            start = i
            break
    if start is None:
        # fallback: last 200 lines (better than empty)
        return "\n".join(lines[-200:])

    # take a window after start
    return "\n".join(lines[start : min(len(lines), start + 400)])


def extract_final_geometry_from_out(output_text: str) -> str:
    """
    Parse a final CARTESIAN COORDINATES (ANGSTROEM) block.
    Works with both "index symbol x y z" and "symbol x y z".
    """
    lines = output_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper():
            start_idx = i

    if start_idx is None:
        return ""

    i = start_idx
    while i < len(lines) and "-----" not in lines[i]:
        i += 1
    # move to first atom line
    i += 1

    xyz_lines = []
    for j in range(i, len(lines)):
        parts = lines[j].split()
        if len(parts) < 4:
            break

        # case A: "1  O  0.0  0.0  0.0"
        if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
            symbol = parts[1]
            x, y, z = parts[2], parts[3], parts[4]
        # case B: "O  0.0  0.0  0.0"
        else:
            symbol = parts[0]
            x, y, z = parts[1], parts[2], parts[3]

        xyz_lines.append(f"{symbol} {x} {y} {z}")
    return "\n".join(xyz_lines)


# -----------------------------
# OPI helpers
# -----------------------------
def _ensure_clean_dir(d: Path, clean: bool = True):
    """Create workdir; optionally remove previous contents first."""
    if clean:
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _xyz_with_header(geometry_xyz_no_header: str) -> str:
    geom_lines = [ln for ln in geometry_xyz_no_header.splitlines() if ln.strip()]
    nat = len(geom_lines)
    return f"{nat}\nOPI\n" + "\n".join(geom_lines) + "\n"


def _set_charge_mult(calc: Calculator, charge: int, multiplicity: int):
    """
    OPI stores charge/multiplicity on input/structure depending on version.
    We set what exists, safely.
    """
    for attr in ["charge", "chg"]:
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, charge)
            break
    for attr in ["multiplicity", "mult", "spinmultiplicity"]:
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, multiplicity)
            break

    if hasattr(calc, "structure"):
        for attr in ["charge", "chg"]:
            if hasattr(calc.structure, attr):
                setattr(calc.structure, attr, charge)
                break
        for attr in ["multiplicity", "mult", "spinmultiplicity"]:
            if hasattr(calc.structure, attr):
                setattr(calc.structure, attr, multiplicity)
                break


def _build_calc(
    label: str,
    workdir: Path,
    geometry_xyz: str,
    charge: int,
    multiplicity: int,
    method: str,
    basis: str,
    job_type: Literal["sp", "opt", "freq"],
    use_ri: bool,
    scf_max_iter: int,
    opt_max_iter: int,
    nbo: bool,
    ncores: int,
    clean_workdir: bool = True,
) -> Calculator:
    _ensure_clean_dir(workdir, clean=clean_workdir)

    xyz_path = workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(geometry_xyz), encoding="utf-8")

    structure = Structure.from_xyz(xyz_path)

    calc = Calculator(basename=label, working_dir=workdir)
    calc.structure = structure
    _set_charge_mult(calc, charge, multiplicity)

    # Keep your “same keywords” style: build one main line.
    task_kw = "OPT" if job_type == "opt" else ("FREQ" if job_type == "freq" else "SP")
    ri_kw = "RIJCOSX" if use_ri else ""
    nbo_kw = "NBO" if nbo else ""

    main_line = f"! {method} {basis} {task_kw} {ri_kw} {nbo_kw}".strip()
    calc.input.add_arbitrary_string(main_line)

    # SCF control
    calc.input.add_blocks(BlockScf(maxiter=scf_max_iter))

    # OPT control
    if job_type == "opt":
        calc.input.add_blocks(BlockGeom(maxiter=opt_max_iter))

    # cores
    if hasattr(calc.input, "ncores"):
        calc.input.ncores = int(ncores)

    return calc


def _read_out_text(workdir: Path, label: str) -> str:
    out_path = workdir / f"{label}.out"
    if out_path.exists():
        return out_path.read_text(encoding="utf-8", errors="ignore")
    # fallback: read any .out in folder
    outs = sorted(workdir.glob("*.out"))
    if outs:
        return outs[-1].read_text(encoding="utf-8", errors="ignore")
    return ""



def _tail_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""

def _read_text_safe(p: Path, max_chars: int = 200_000) -> str:
    if not p.exists():
        return ""
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return t[-max_chars:] if len(t) > max_chars else t

def _list_dir(workdir: Path, max_files: int = 80):
    try:
        items = sorted(workdir.iterdir(), key=lambda x: x.name)
    except Exception:
        return []
    out = []
    for p in items[:max_files]:
        try:
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "is_dir": p.is_dir()})
        except Exception:
            out.append({"name": p.name, "size": None, "is_dir": p.is_dir()})
    return out

def _which(cmd: str) -> Optional[str]:
    import shutil as _sh
    return _sh.which(cmd)

def _collect_debug(workdir: Path, label: str, tail_lines: int = 120):
    out_path = workdir / f"{label}.out"
    inp_path = workdir / f"{label}.inp"
    err_path = workdir / f"{label}.err"
    return {
        "workdir": str(workdir),
        "orca_exe": _which("orca"),
        "files": _list_dir(workdir),
        "inp_tail": _tail_lines(_read_text_safe(inp_path), min(200, tail_lines)),
        "out_tail": _tail_lines(_read_text_safe(out_path), tail_lines),
        "err_tail": _tail_lines(_read_text_safe(err_path), tail_lines),
    }

def _error_summary(out_text: str) -> str:
    keys = ("ERROR", "ABORT", "FATAL", "TERMINAT", "did not converge", "SCF", "OPTIMIZATION")
    hits = []
    for ln in out_text.splitlines()[-1200:]:
        u = ln.upper()
        if any(k.upper() in u for k in keys):
            hits.append(ln)
    return "\n".join(hits[-60:])

async def _run_calc_with_timeout(calc: Calculator, wall_timeout_seconds: int):
    def _run_sync():
        calc.write_input()
        calc.run()
        return calc.get_output()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run_sync), timeout=wall_timeout_seconds)
    except asyncio.TimeoutError:
        # Best-effort kill if supported
        if hasattr(calc, "kill"):
            try:
                calc.kill()
            except Exception:
                pass
        raise



# -----------------------------
# Thermochemistry parsers (E/H/G) from ORCA output
# -----------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def _last_float_in_line(line: str) -> Optional[float]:
    vals = _FLOAT_RE.findall(line)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        return None

def extract_total_enthalpy(output_text: str) -> Optional[float]:
    # ORCA prints something like: "Total Enthalpy ...   -XXX.XXXX Eh"
    for line in reversed(output_text.splitlines()):
        u = line.upper()
        if "TOTAL ENTHALPY" in u:
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None

def extract_gibbs_free_energy(output_text: str) -> Optional[float]:
    # ORCA prints something like: "Final Gibbs free energy ...  -XXX.XXXX Eh"
    for line in reversed(output_text.splitlines()):
        u = line.upper()
        if "FINAL GIBBS" in u and "FREE" in u:
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None

def _get_cluster_xyz_from_solvator_result(result: dict) -> Optional[str]:
    # Be defensive: different helper versions use different keys.
    for k in [
        "cluster_xyz",
        "cluster_geometry_xyz",
        "cluster_geometry",
        "geometry_xyz",
        "cluster",
        "xyz",
    ]:
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


@mcp.tool()
async def run_solvator_cluster_thermo(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    nsolv: int = 3,
    thermo_engine: Literal["orca"] = "orca",
    # Reasonable cheap default for freq on a cluster:
    method: str = "r2scan-3c",
    basis: str = "",
    use_ri: bool = False,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 600,
    thermo_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """
    Build an explicit-solvent cluster with SOLVATOR and then compute E/H/G via an ORCA frequency job.

    Returns:
      - cluster geometry (XYZ, no header if that is what SOLVATOR returns)
      - E (FINAL SINGLE POINT ENERGY), H (Total Enthalpy), G (Final Gibbs free energy) in Eh
      - tails of logs for debugging
    """

    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"solvthermo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    # 1) SOLVATOR cluster build (sync helper in a thread)
    solv_result = await asyncio.to_thread(
        _run_solvator_sync,
        geometry_xyz,
        charge,
        multiplicity,
        nsolv,
        job_label,
    )

    if not isinstance(solv_result, dict):
        return json.dumps({"status": "error", "label": job_label, "error": "SOLVATOR helper returned non-dict."})

    cluster_xyz_no_header = _get_cluster_xyz_from_solvator_result(solv_result) or ""

    # Normalize: accept either XYZ-with-header or atom-lines-only.
    lines = [ln.strip() for ln in cluster_xyz_no_header.splitlines() if ln.strip()]
    if lines:
        try:
            nat = int(lines[0])
            if len(lines) >= nat + 2:
                cluster_xyz_no_header = "\n".join(lines[2 : 2 + nat])
        except Exception:
            pass
    if not cluster_xyz_no_header:
        # Return solvator result as-is; thermo step skipped.
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Could not find cluster XYZ in SOLVATOR result."
        return json.dumps(solv_result)

    # 2) ORCA frequency job for thermochemistry
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    thermo_label = f"{job_label}_freq"
    thermo_workdir = jobs_dir / thermo_label

    # Write cluster xyz (with header) into thermo workdir
    _ensure_clean_dir(thermo_workdir)
    (thermo_workdir / "cluster.xyz").write_text(_xyz_with_header(cluster_xyz_no_header), encoding="utf-8")

    # Build ORCA calc with FREQ task
    calc = _build_calc(
        label=thermo_label,
        workdir=thermo_workdir,
        geometry_xyz=cluster_xyz_no_header,  # _build_calc writes struc.xyz; OK
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

    # Ensure the calc uses the cluster.xyz geometry we wrote (more explicit)
    xyz_path = thermo_workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(cluster_xyz_no_header), encoding="utf-8")
    calc.structure = Structure.from_xyz(xyz_path)
    _set_charge_mult(calc, charge, multiplicity)

    try:
        output = await _run_calc_with_timeout(calc, thermo_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(thermo_workdir, thermo_label)
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "timeout",
                "E_eh": extract_total_energy(out_text),
                "H_eh": extract_total_enthalpy(out_text),
                "G_eh": extract_gibbs_free_energy(out_text),
                "thermo_tail": "\n".join(out_text.splitlines()[-120:]),
                "solvator": solv_result,
            }
        )
    except Exception as e:
        out_text = _read_out_text(thermo_workdir, thermo_label)
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "error",
                "thermo_error": str(e),
                "thermo_tail": "\n".join(out_text.splitlines()[-120:]),
                "solvator": solv_result,
            }
        )

    ok = output.terminated_normally()
    out_text = _read_out_text(thermo_workdir, thermo_label)

    if not ok:
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "error",
                "thermo_tail": "\n".join(out_text.splitlines()[-150:]),
                "solvator": solv_result,
                "text": "ORCA freq did not terminate normally.",
            }
        )

    E = extract_total_energy(out_text)
    H = extract_total_enthalpy(out_text)
    G = extract_gibbs_free_energy(out_text)

    return json.dumps(
        {
            "status": "ok",
            "label": job_label,
            "cluster_xyz": cluster_xyz_no_header,
            "E_eh": E,
            "H_eh": H,
            "G_eh": G,
            "thermo_status": "ok",
            "thermo_label": thermo_label,
            "solvator": solv_result,
            "text": f"Status: OK\nE={E} Eh\nH={H} Eh\nG={G} Eh",
        }
    )

# -----------------------------
# Tools
# -----------------------------

@mcp.tool()
async def run_opt_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 100,
    wall_timeout_seconds: int = 1800,
    job_label: Optional[str] = None,
    ncores: int = 1,
    clean_workdir: bool = True,
    debug: bool = False,
    debug_tail_lines: int = 160,
) -> str:
    """Geometry optimization with optional debug bundle.

    Debug tips:
      - set debug=True to include inp/out tails, file listing, ORCA path
      - set clean_workdir=False to preserve an existing directory for post-mortem comparison
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"opt_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        job_type="opt",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=False,
        ncores=ncores,
        clean_workdir=clean_workdir,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        resp = {
            "status": "timeout",
            "label": job_label,
            "workdir": str(workdir),
            "energy": extract_total_energy(out_text),
            "final_geometry_xyz": extract_final_geometry_from_out(out_text),
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        resp = {
            "status": "error",
            "label": job_label,
            "workdir": str(workdir),
            "error": str(e),
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    out_text = _read_out_text(workdir, job_label)
    ok = output.terminated_normally()

    # ORCA prints "*** THE OPTIMIZATION HAS CONVERGED ***" when converged.
    opt_converged = ("THE OPTIMIZATION HAS CONVERGED" in out_text.upper())

    energy = extract_total_energy(out_text)
    final_xyz = extract_final_geometry_from_out(out_text)

    if not ok:
        resp = {
            "status": "error",
            "label": job_label,
            "workdir": str(workdir),
            "opt_converged": opt_converged,
            "energy": energy,
            "final_geometry_xyz": final_xyz,
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
            "text": "ORCA did not terminate normally.",
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    # Terminated normally but may still hit maxoptiter (common case)
    status = "ok" if opt_converged else "not_converged"

    resp = resp = {
    "status": status,
    "label": job_label,
    "workdir": str(workdir),
    "opt_converged": opt_converged,
    "energy_eh": energy,  # rename is optional but nice
    "final_geometry_xyz": final_xyz,  # keep for backward compat
    "geometry_xyz": final_xyz,         # NEW: unified key for SessionState
    "provenance": {"geometry": "orca_opt"},
    "tail": _tail_lines(out_text, min(120, debug_tail_lines)),
    "text": f"Status: {status.upper()}\nFinal energy: {energy if energy is not None else 'N/A'} Eh",
    }
    if debug:
        resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
    return json.dumps(resp)




@mcp.tool()
async def run_nbo_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    job_type: Literal["sp", "opt"] = "sp",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 50,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"nbo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        job_type=job_type,
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=True,
        ncores=ncores,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {
                "status": "timeout",
                "label": job_label,
                "energy": extract_total_energy(out_text),
                "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
                "tail": "\n".join(out_text.splitlines()[-80:]),
            }
        )
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {"status": "error", "label": job_label, "error": str(e), "tail": "\n".join(out_text.splitlines()[-80:])}
        )

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)

    if not ok:
        return json.dumps(
            {
                "status": "error",
                "label": job_label,
                "tail": "\n".join(out_text.splitlines()[-120:]),
                "text": "ORCA did not terminate normally.",
            }
        )

    energy = extract_total_energy(out_text)
    nbo_section = extract_nbo_section(out_text)

    return json.dumps(
        {
            "status": "ok",
            "label": job_label,
            "energy": energy,
            "nbo_section": nbo_section,
            "text": "Status: OK\n=== NBO / NPA Section (excerpt) ===\n" + nbo_section,
        }
    )


@mcp.tool()
async def run_sp_energy(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"sp_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        job_type="sp",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=50,
        nbo=False,
        ncores=ncores,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {"status": "timeout", "label": job_label, "text": f"Status: TIMEOUT after {wall_timeout_seconds}s"}
        )
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({"status": "error", "label": job_label, "tail": "\n".join(out_text.splitlines()[-120:])})

    energy = extract_total_energy(out_text)
    return json.dumps({"status": "ok", "label": job_label, "energy": energy})
'''
@mcp.tool()
async def run_solvator_cluster(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    nsolv: int = 3,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
) -> str:
    """
    Build a small explicit-solvent cluster with SOLVATOR (water) around the given solute.

    - Uses XTB + ALPB(WATER)
    - Uses SOLVATOR with nsolv water molecules, docking mode, fixed solute
    - Returns: cluster geometry (XYZ, no header) + status text
    """

    if job_label is None:
        job_label = f"solv_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    # Run the sync helper in a thread; optionally you could add your own timeout logic here
    result = await asyncio.to_thread(
        _run_solvator_sync,
        geometry_xyz,
        charge,
        multiplicity,
        nsolv,
        job_label,
    )

    return json.dumps(result)



@mcp.tool()
async def inspect_job(
    job_label: str,
    tail_lines: int = 200,
) -> str:
    """Return a compact debug bundle for an existing job directory."""
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label
    if not workdir.exists():
        return json.dumps({"status": "error", "label": job_label, "error": "workdir not found"})
    dbg = _collect_debug(workdir, job_label, tail_lines=tail_lines)
    return json.dumps({"status": "ok", "label": job_label, "debug": dbg})

'''
from typing import Optional, Literal
@mcp.tool()
async def structure_add_remove_proton(
    geometry_xyz: str,
    mode: Literal["add", "remove"],
    charge: int = 0,
    multiplicity: int = 1,
    h_index: Optional[int] = None,
    target_atom_index: Optional[int] = None,
    geometry_name: Optional[str] = None,
    strategy: Literal["auto", "distance"] = "auto",
) -> str:
    res = structure_proton_edit(
        xyz=geometry_xyz,
        mode=mode,
        charge=charge,
        multiplicity=multiplicity,
        h_index=h_index,
        target_atom_index=target_atom_index,
        geometry_name=geometry_name,
        strategy=strategy,
        #status = 'ok'
    )
    return json.dumps(res)


mcp.run(transport="stdio")