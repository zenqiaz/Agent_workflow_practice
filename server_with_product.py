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
    job_type: Literal["sp", "opt", "freq", "scan", "ts_opt"],
    use_ri: bool,
    scf_max_iter: int,
    opt_max_iter: int,
    nbo: bool,
    ncores: int,
    clean_workdir: bool = True,
    raman: bool = False,
) -> Calculator:
    _ensure_clean_dir(workdir, clean=clean_workdir)

    xyz_path = workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(geometry_xyz), encoding="utf-8")

    structure = Structure.from_xyz(xyz_path)

    calc = Calculator(basename=label, working_dir=workdir)
    calc.structure = structure
    _set_charge_mult(calc, charge, multiplicity)

    # Keep your "same keywords" style: build one main line.
    # "scan" maps to OPT keyword (activates ORCA scan loop) but skips BlockGeom
    # to avoid a second %geom block conflicting with the %geom Scan block.
    # "ts_opt" uses OptTS keyword; caller adds its own %geom block.
    if job_type == "ts_opt":
        task_kw = "OptTS"
    elif job_type in ("opt", "scan"):
        task_kw = "OPT"
    elif job_type == "freq":
        task_kw = "FREQ"
    else:
        task_kw = "SP"
    ri_kw = "RIJCOSX" if use_ri else ""
    nbo_kw = "NBO" if nbo else ""

    main_line = " ".join(p for p in [f"! {method}", basis, task_kw, ri_kw, nbo_kw] if p.strip())
    calc.input.add_arbitrary_string(main_line)

    # Raman: requires polarizability derivatives via %elprop Polar 1
    if raman:
        calc.input.add_arbitrary_string("%elprop\n  Polar 1\nend")

    # SCF control
    calc.input.add_blocks(BlockScf(maxiter=scf_max_iter))

    # OPT control (skip for "scan" and "ts_opt" — callers add their own %geom blocks)
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

def extract_ir_spectrum(output_text: str) -> List[Dict]:
    """Parse IR SPECTRUM block from ORCA output.

    Returns [{"mode": int, "freq_cm1": float, "intensity_km_mol": float}, ...]
    for real vibrational modes only (|freq| > 10 cm-1).
    Uses the last occurrence of the block.

    ORCA format:
      Mode   freq       eps      Int      T**2   TX  TY  TZ
             cm**-1  L/(mol*cm)  km/mol   a.u.
      6:   2078.52   0.002020   10.21  0.000303  (...)
    """
    lines = output_text.splitlines()
    block_start = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r'\bIR\s+SPECTRUM\b', lines[i], re.IGNORECASE):
            block_start = i
            break
    if block_start is None:
        return []

    # Pattern: "  6:   2078.52   0.002020   10.21  ..."
    # columns after "N:": freq, eps, Int(km/mol), T**2, (TX TY TZ)
    pat = re.compile(r'^\s*(\d+):\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)')
    result = []
    for line in lines[block_start + 1: block_start + 100]:
        m = pat.match(line)
        if m:
            mode = int(m.group(1))
            freq = float(m.group(2))
            # m.group(3) = eps, m.group(4) = Int (km/mol)
            intensity = float(m.group(4))
            if abs(freq) > 10.0:
                result.append({"mode": mode, "freq_cm1": freq, "intensity_km_mol": intensity})
        elif result and line.strip() and not line.strip().startswith(('-', '*')):
            if not re.match(r'\s*(mode|freq|cm\*\*)', line.strip(), re.IGNORECASE):
                break
    return result


def extract_raman_spectrum(output_text: str) -> List[Dict]:
    """Parse RAMAN SPECTRUM block from ORCA output.

    Returns [{"mode": int, "freq_cm1": float, "activity": float, "depolarization": float}, ...]
    for real vibrational modes only (|freq| > 10 cm-1).
    Requires %elprop Polar 1 in the ORCA input.

    ORCA format:
      Mode    freq (cm**-1)   Activity   Depolarization
      6:      2078.52      8.498849      0.722060
    """
    lines = output_text.splitlines()
    block_start = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r'\bRAMAN\s+SPECTRUM\b', lines[i], re.IGNORECASE):
            block_start = i
            break
    if block_start is None:
        return []

    pat = re.compile(r'^\s*(\d+):\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)')
    result = []
    for line in lines[block_start + 1: block_start + 100]:
        m = pat.match(line)
        if m:
            mode = int(m.group(1))
            freq = float(m.group(2))
            activity = float(m.group(3))
            depol = float(m.group(4))
            if abs(freq) > 10.0:
                result.append({"mode": mode, "freq_cm1": freq,
                                "activity": activity, "depolarization": depol})
        elif result and line.strip() and not line.strip().startswith(('-', '*')):
            if not re.match(r'\s*(mode|freq|activity|depol)', line.strip(), re.IGNORECASE):
                break
    return result


_EV_PER_CM1 = 1.0 / 8065.54439  # 1 cm⁻¹ in eV


def extract_excited_states(output_text: str) -> List[Dict]:
    """Parse TD-DFT excited states from ORCA absorption spectrum table.

    Finds the last occurrence of the
    "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" block.

    ORCA 6 format (columns: Transition | Energy(eV) | Energy(cm-1) | Wavelength(nm) | fosc(D2) | ...):
        0-1A  ->  1-1A    4.077777   32889.5   304.0   0.000000000   ...
        0-1A  ->  2-1A    8.200851   66144.3   151.2   0.170248747   ...

    Returns list of:
        {"state": int, "energy_ev": float, "wavelength_nm": float,
         "oscillator_strength": float}
    """
    lines = output_text.splitlines()
    block_start = None
    for i, line in enumerate(lines):
        if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line.upper():
            block_start = i

    if block_start is None:
        return []

    # Match: "  0-1A  ->  N-XA    energy_ev   energy_cm1   wavelength_nm   fosc   ..."
    pat = re.compile(
        r'^\s+\d+-\S+\s*->\s*(\d+)-\S+\s+'  # transition label (group 1: target state #)
        r'([\d.]+)\s+'                         # energy_ev (group 2)
        r'[\d.]+\s+'                           # energy_cm1 (skip)
        r'([\d.]+)\s+'                         # wavelength_nm (group 3)
        r'([\d.]+(?:[eE][+-]?\d+)?)'           # fosc (group 4)
    )
    result = []
    for line in lines[block_start + 1: block_start + 200]:
        m = pat.match(line)
        if m:
            state = int(m.group(1))
            energy_ev = float(m.group(2))
            wavelength_nm = float(m.group(3))
            fosc = float(m.group(4))
            result.append({
                "state": state,
                "energy_ev": round(energy_ev, 4),
                "wavelength_nm": round(wavelength_nm, 2),
                "oscillator_strength": round(fosc, 6),
            })
        elif result and re.match(r'\s*-{10,}', line):
            break  # end-of-block separator
    return result


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
                "product": ("E_eh", "H_eh", "G_eh")
                
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
            "product": ("G_eh")
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
            "product": ("energy", "nbo_section")
            
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
    return json.dumps({"status": "ok", "label": job_label, "energy": energy, "product": "energy"})


@mcp.tool()
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
        return json.dumps({"status": "error", "label": job_label, "tail": "\n".join(out_text.splitlines()[-120:])})

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


@mcp.tool()
async def run_spectrum_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    spectrum_type: Literal["ir", "raman", "ir_raman"] = "ir",
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA frequency calculation and return IR and/or Raman spectrum data.

    spectrum_type:
      "ir"       – IR spectrum only (always free with FREQ; default)
      "raman"    – IR + Raman spectrum (adds %elprop Polar 1; ~2x cost)
      "ir_raman" – same as "raman"

    Returns E/H/G (same as run_freq_job) plus:
      ir_spectrum:    [{mode, freq_cm1, intensity_km_mol}, ...]
      raman_spectrum: [{mode, freq_cm1, activity, depolarization}, ...]  (if requested)
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    do_raman = spectrum_type in ("raman", "ir_raman")

    if job_label is None:
        job_label = f"spec_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        raman=do_raman,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "timeout", "label": job_label,
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
            "ir_spectrum": extract_ir_spectrum(out_text),
        })
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error", "label": job_label,
            "tail": "\n".join(out_text.splitlines()[-120:]),
        })

    energy = extract_total_energy(out_text)
    enthalpy = extract_total_enthalpy(out_text)
    gibbs = extract_gibbs_free_energy(out_text)
    ir_spec = extract_ir_spectrum(out_text)

    result = {
        "status": "ok",
        "label": job_label,
        "energy_eh": energy,
        "enthalpy_eh": enthalpy,
        "gibbs_free_energy_eh": gibbs,
        "ir_spectrum": ir_spec,
        "spectrum_type": spectrum_type,
        "product": "gibbs_free_energy_eh",
    }
    if do_raman:
        result["raman_spectrum"] = extract_raman_spectrum(out_text)

    return json.dumps(result)


@mcp.tool()
async def run_tddft_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    n_states: int = 5,
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA TD-DFT excited-state calculation on a pre-optimised geometry.

    Computes ground-state DFT energy and the lowest n_states singlet excited states.
    The input geometry must already be optimised (run run_opt_job first).

    Returns a dict with:
        energy_ground_state_eh: float  (ground-state DFT energy in Eh)
        excited_states: list of {state, energy_ev, wavelength_nm, oscillator_strength}

    Oscillator strength (fosc) interpretation:
        fosc >> 0  -> bright (electric-dipole-allowed) transition
        fosc ~  0  -> dark (forbidden) transition
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"tddft_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )
    n_states = max(1, int(n_states))
    calc.input.add_arbitrary_string(f"%tddft\n  nroots {n_states}\nend")

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "timeout",
            "label": job_label,
            "excited_states": extract_excited_states(out_text),
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
        })
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error",
            "label": job_label,
            "tail": "\n".join(out_text.splitlines()[-120:]),
        })

    energy = extract_total_energy(out_text)
    excited_states = extract_excited_states(out_text)

    return json.dumps({
        "status": "ok",
        "label": job_label,
        "energy_ground_state_eh": energy,
        "excited_states": excited_states,
        "product": "excited_states",
        "text": (
            f"Status: OK\nGround state energy: {energy} Eh\n"
            f"Excited states found: {len(excited_states)}"
        ),
    })


# ─── Rigid surface scan ────────────────────────────────────────────────────────

def _build_scan_block(scan_coords: List[Dict]) -> str:
    """Build ORCA %geom Scan block for a rigid scan.

    scan_coords: [{"type": "B"|"A"|"D",
                   "atoms": [i, j, ...],   # 0-based ORCA indices
                   "start": float,         # Å for B; degrees for A/D
                   "end":   float,
                   "n_points": int}]       # number of calculation points
    """
    lines = ["%geom", "  Scan"]
    for c in scan_coords:
        coord_type = str(c["type"]).upper()
        atoms = " ".join(str(int(a)) for a in c["atoms"])
        start = float(c["start"])
        end   = float(c["end"])
        n     = int(c["n_points"])
        lines.append(f"    {coord_type} {atoms} = {start:.6f}, {end:.6f}, {n}")
    lines += ["  end", "End"]
    return "\n".join(lines)


def _extract_scan_results(out_text: str, scan_coords: List[Dict]) -> List[Dict]:
    """Parse ORCA scan output (relaxed or rigid), returning [{step, value, energy_eh}, ...].

    Stage 1 — summary table: ORCA end-of-scan summary (relaxed or rigid).
    Stage 2 — per-step markers: "RELAXED SURFACE SCAN STEP N" with last energy per step.
    Stage 3 — fallback: all FINAL SINGLE POINT ENERGY lines with computed values.
    """
    lines = out_text.splitlines()
    energy_pat = re.compile(
        r'FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?\d+\.\d+)', re.IGNORECASE
    )

    def _computed_values(n_steps: int) -> List[float]:
        c0 = scan_coords[0] if scan_coords else None
        if c0 and int(c0.get("n_points", 0)) > 1:
            n = int(c0["n_points"])
            return [
                float(c0["start"]) + i * (float(c0["end"]) - float(c0["start"])) / (n - 1)
                for i in range(n_steps)
            ]
        return [float(i) for i in range(n_steps)]

    def _parse_summary_table(start_idx: int) -> List[Dict]:
        """Parse a scan summary table starting at start_idx."""
        # Rows: "  1   1.0000   -115.70900" or "  1  B(0,1): 1.0000  -115.70900"
        pat = re.compile(
            r'^\s+(\d+)\s+(?:\S+:\s*)?([-+]?\d+\.?\d*)\s+([-+]?\d+\.\d+)'
        )
        results = []
        for ln in lines[start_idx + 1 : start_idx + 400]:
            m = pat.match(ln)
            if m:
                results.append({
                    "step":      int(m.group(1)),
                    "value":     round(float(m.group(2)), 6),
                    "energy_eh": round(float(m.group(3)), 8),
                })
            elif results and re.match(r'\s*-{20,}', ln):
                break
        return results

    # Stage 1: find any scan summary table (relaxed or rigid, either keyword order)
    summary_patterns = [
        r'(RELAXED|RIGID)\s+SURFACE\s+SCAN\s+(SUMMARY|RESULTS)',
        r'(SUMMARY|RESULTS)\s+OF\s+(THE\s+)?(RELAXED|RIGID)\s+SURFACE\s+SCAN',
        r'THE\s+(RELAXED|RIGID)\s+SURFACE\s+SCAN\s+RESULTS',
    ]
    for spat in summary_patterns:
        for i, ln in enumerate(lines):
            if re.search(spat, ln, re.IGNORECASE):
                results = _parse_summary_table(i)
                if results:
                    return results

    # Stage 2: per-step parsing via "RELAXED SURFACE SCAN STEP N" markers
    # Each step ends just before the next step header (or EOF).
    step_pat = re.compile(r'RELAXED\s+SURFACE\s+SCAN\s+STEP\s+(\d+)', re.IGNORECASE)
    step_indices: List[tuple] = []  # (line_index, step_number)
    for i, ln in enumerate(lines):
        m = step_pat.search(ln)
        if m:
            step_indices.append((i, int(m.group(1))))

    if step_indices:
        results = []
        for k, (step_line_idx, step_num) in enumerate(step_indices):
            next_idx = step_indices[k + 1][0] if k + 1 < len(step_indices) else len(lines)
            # Collect the LAST FINAL SINGLE POINT ENERGY in this step's range
            step_energies = [
                float(m.group(1))
                for ln in lines[step_line_idx:next_idx]
                for m in [energy_pat.search(ln)] if m
            ]
            if step_energies:
                results.append({"step": step_num, "energy_eh": round(step_energies[-1], 8)})
        if results:
            vals = _computed_values(len(results))
            for k, r in enumerate(results):
                r["value"] = round(vals[k], 6)
            return results

    # Stage 3: fallback — all FINAL SINGLE POINT ENERGY lines with computed values
    energies = [float(m.group(1)) for m in energy_pat.finditer(out_text)]
    if not energies:
        return []
    vals = _computed_values(len(energies))
    return [
        {"step": i + 1, "value": round(vals[i], 6), "energy_eh": round(e, 8)}
        for i, e in enumerate(energies)
    ]


def _extract_scan_geometries(out_text: str, n_steps: int) -> List[str]:
    """Extract per-step optimised geometries from a relaxed scan output.

    Splits the output by "RELAXED SURFACE SCAN STEP N" markers and calls
    extract_final_geometry_from_out on each chunk.
    Returns a list of XYZ strings (atom-lines only, no header), indexed by (step-1).
    Steps with no parseable geometry return an empty string.
    """
    step_pat = re.compile(r'RELAXED\s+SURFACE\s+SCAN\s+STEP\s+(\d+)', re.IGNORECASE)
    lines = out_text.splitlines()

    step_indices: List[tuple] = []  # (line_idx, step_num)
    for i, ln in enumerate(lines):
        m = step_pat.search(ln)
        if m:
            step_indices.append((i, int(m.group(1))))

    if not step_indices:
        return []

    geometries = []
    for k, (start_idx, _step_num) in enumerate(step_indices):
        end_idx = step_indices[k + 1][0] if k + 1 < len(step_indices) else len(lines)
        chunk = "\n".join(lines[start_idx:end_idx])
        geom = extract_final_geometry_from_out(chunk)
        geometries.append(geom)
    return geometries


@mcp.tool()
async def run_scan_job(
    geometry_xyz: str,
    scan_coords: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 7200,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run a relaxed ORCA surface scan (geometry optimization at each scan point).

    scan_coords: JSON string with a list of coordinate dicts, e.g.:
        '[{"type":"B","atoms":[0,1],"start":0.8,"end":1.8,"n_points":11}]'
        type: "B" = bond (Å), "A" = angle (degrees), "D" = dihedral (degrees)
        atoms: 0-based ORCA atom indices (2 for B, 3 for A, 4 for D)
        n_points: total number of calculation points (inclusive)

    Returns:
        scan_results: [{step, value, energy_eh}, ...]
        min_energy_eh: float   (lowest energy found)
        min_value:     float   (coordinate value at minimum)
        n_points:      int
        scan_coords:   list    (echo of input coords)
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    try:
        coords = json.loads(scan_coords)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"status": "error", "error": f"scan_coords JSON parse failed: {e}"})
    if not isinstance(coords, list) or not coords:
        return json.dumps({"status": "error", "error": "scan_coords must be a non-empty JSON array"})

    if job_label is None:
        job_label = f"scan_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir  = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="scan",   # OPT keyword activates scan loop; no extra %geom block
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )
    calc.input.add_arbitrary_string(_build_scan_block(coords))

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        partial = _extract_scan_results(out_text, coords)
        return json.dumps({
            "status":      "timeout",
            "label":       job_label,
            "scan_results": partial,
            "text":        f"Status: TIMEOUT after {wall_timeout_seconds}s ({len(partial)} steps collected)",
        })
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error",
            "label":  job_label,
            "tail":   "\n".join(out_text.splitlines()[-120:]),
        })

    scan_results = _extract_scan_results(out_text, coords)

    min_energy = min((p["energy_eh"] for p in scan_results), default=None)
    min_value  = next(
        (p["value"] for p in scan_results if p["energy_eh"] == min_energy), None
    ) if min_energy is not None else None

    max_energy = max((p["energy_eh"] for p in scan_results), default=None)
    max_value  = next(
        (p["value"] for p in scan_results if p["energy_eh"] == max_energy), None
    ) if max_energy is not None else None

    # Extract geometry at the PES maximum (for TS candidate seeding)
    max_geom = ""
    if scan_results and max_energy is not None:
        step_geoms = _extract_scan_geometries(out_text, len(scan_results))
        max_step_idx = next(
            (k for k, r in enumerate(scan_results) if r["energy_eh"] == max_energy), None
        )
        if max_step_idx is not None and max_step_idx < len(step_geoms):
            max_geom = step_geoms[max_step_idx]

    return json.dumps({
        "status":           "ok",
        "label":            job_label,
        "scan_results":     scan_results,
        "n_points":         len(scan_results),
        "min_energy_eh":    min_energy,
        "min_value":        min_value,
        "max_energy_eh":    max_energy,
        "max_value":        max_value,
        "geometry_xyz":     max_geom,       # PES maximum geometry (TS candidate)
        "max_geometry_xyz": max_geom,       # alias
        "scan_coords":      coords,
        "product":          "scan_results",
        "text": (
            f"Status: OK\n{len(scan_results)} scan points\n"
            f"Min energy: {min_energy} Eh at value {min_value}\n"
            f"Max energy: {max_energy} Eh at value {max_value}"
        ),
    })


@mcp.tool()
async def run_ts_opt_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 100,
    calc_hess: bool = True,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA transition-state optimisation (OptTS).

    Requires a starting geometry near the transition state (e.g., the PES maximum
    from run_scan_job). calc_hess=True (default) adds Calc_Hess true to the %geom
    block, which is strongly recommended for reliable TS optimisation.

    Returns:
        status:       "ok" | "not_converged" | "error" | "timeout"
        geometry_xyz: optimised TS geometry (atom lines, no header)
        energy_eh:    final energy in Hartree
        ts_converged: bool — whether ORCA reported a converged optimisation
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"tsopt_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
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
        job_type="ts_opt",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=False,
        ncores=ncores,
    )

    # Build a single %geom block with maxiter and optional Calc_Hess
    geom_lines = ["%geom", f"  maxiter {opt_max_iter}"]
    if calc_hess:
        geom_lines.append("  Calc_Hess true")
    geom_lines.append("end")
    calc.input.add_arbitrary_string("\n".join(geom_lines))

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status":       "timeout",
            "label":        job_label,
            "energy_eh":    extract_total_energy(out_text),
            "geometry_xyz": extract_final_geometry_from_out(out_text),
            "text":         f"Status: TIMEOUT after {wall_timeout_seconds}s",
        })
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "error",
            "label":  job_label,
            "error":  str(e),
            "tail":   "\n".join(out_text.splitlines()[-80:]),
        })

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    ts_converged = "THE OPTIMIZATION HAS CONVERGED" in out_text.upper()
    energy = extract_total_energy(out_text)
    final_xyz = extract_final_geometry_from_out(out_text)

    if not ok:
        return json.dumps({
            "status":       "error",
            "label":        job_label,
            "ts_converged": ts_converged,
            "energy_eh":    energy,
            "geometry_xyz": final_xyz,
            "tail":         "\n".join(out_text.splitlines()[-120:]),
            "text":         "ORCA did not terminate normally.",
        })

    status = "ok" if ts_converged else "not_converged"
    return json.dumps({
        "status":       status,
        "label":        job_label,
        "ts_converged": ts_converged,
        "energy_eh":    energy,
        "geometry_xyz": final_xyz,
        "provenance":   {"geometry": "orca_tsopt"},
        "tail":         "\n".join(out_text.splitlines()[-60:]),
        "text":         f"Status: {status.upper()}\nTS energy: {energy} Eh",
    })


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

@mcp.tool()
async def structure_add_remove_proton(
    geometry_xyz: str,
    mode: Literal["add", "remove"],
    charge: int = 0,
    multiplicity: int = 1,
    site_selector: Optional[str] = None,
    variant: int = 0,
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
        site_selector=site_selector,
        variant=variant,
        h_index=h_index,
        target_atom_index=target_atom_index,
        geometry_name=geometry_name,
        strategy=strategy,
    )
    return json.dumps(res)


mcp.run(transport="stdio")