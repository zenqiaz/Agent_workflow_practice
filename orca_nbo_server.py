# orca_nbo_server.py
import asyncio
import os
import sys
from typing import Literal, Optional
import json

from mcp.server.fastmcp import FastMCP

from orca_toolset import (
    OrcaJobSettings,
    run_orca,
    extract_nbo_section,
    extract_total_energy,
)

mcp = FastMCP("orca_tools")


def _status_summary(result) -> str:
    """Generic status header used by all tools."""
    cmd_str = " ".join(result.command)
    base = f"Job label: {result.label}\nORCA command: {cmd_str}\n"

    if result.status == "timeout":
        tail = "\n".join(result.output_text.splitlines()[-80:])
        return (
            base
            + f"Status: TIMEOUT after wall-time limit.\n\nLast lines of output:\n{tail}"
        )

    if result.status == "error":
        tail = "\n".join(result.output_text.splitlines()[-80:])
        return (
            base
            + f"Status: ORCA EXITED WITH CODE {result.return_code}.\n\nLast lines of output:\n{tail}"
        )

    return base + "Status: OK\n"


def extract_final_geometry(output_text: str) -> str:
    """Get final coordinates as XYZ (very rough example for ORCA 4.x)."""
    lines = output_text.splitlines()
    # Look for "CARTESIAN COORDINATES (ANGSTROEM)" or "CARTESIAN COORDINATES (A.U.)"
    start_idx = None
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper():
            start_idx = i
    if start_idx is None:
        return ""

    # skip header lines until atom table
    i = start_idx
    while i < len(lines) and not lines[i].strip().startswith("------"):
        i += 1
    i += 1  # first atom line after dashes

    xyz_lines = []
    for j in range(i, len(lines)):
        parts = lines[j].split()
        #print(len(parts), parts, file=sys.stderr, flush=True)
        if len(parts) < 4:
            break
        # ORCA often prints: index, symbol, x, y, z
        # adjust indices if needed
        if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
            symbol = parts[1]
            x, y, z = parts[2], parts[3], parts[4]
        else:
            symbol = parts[0]
            x, y, z = parts[1], parts[2], parts[3]
        xyz_lines.append(f"{symbol} {x} {y} {z}")
    return "\n".join(xyz_lines)



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
) -> str:
    """Optimize geometry and return final xyz + energy."""
    ...
    
    if job_label is None:
        job_label = f"nbo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    settings = OrcaJobSettings(
        label=job_label,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="opt",
        use_ri=use_ri,
        extra_keywords=[],
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        wall_timeout_seconds=wall_timeout_seconds,
    )
    result = await run_orca(settings)
    
    base = _status_summary(result)
    final_xyz = extract_final_geometry(result.output_text)
    energy = extract_total_energy(result.output_text)

    return json.dumps({
        "status": result.status,
        "label": result.label,
        "energy": energy,
        "final_geometry_xyz": final_xyz,
        "text": base + ("\nFinal energy: %.8f Eh\n" % energy if energy is not None else ""),
    })
# ------------------------
# Tool 1: NBO/NPA job
# ------------------------

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
) -> str:
    """
    Run an ORCA NBO job (NPA/NBO) on the given geometry.

    geometry_xyz: multiline XYZ (no natoms/comment line).
    """

    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"nbo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"

    settings = OrcaJobSettings(
        label=job_label,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type=job_type,
        use_ri=use_ri,
        extra_keywords=["NBO"],
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        wall_timeout_seconds=wall_timeout_seconds,
    )

    
    result = await run_orca(settings)
    header = _status_summary(result)

    if result.status != "ok":
        return json.dumps({
            "status": result.status,
            "return_code": result.return_code,
            "label": result.label,
            "error_tail": "\n".join(result.output_text.splitlines()[-80:]),
            "text": header,
        })

    nbo_text = extract_nbo_section(result.output_text)
    energy = extract_total_energy(result.output_text)

    return json.dumps({
        "status": "ok",
        "label": result.label,
        "energy": energy,
        "nbo_section": nbo_text,
        "text": header + "\n=== NBO / NPA Section ===\n" + nbo_text,
    })


# ------------------------
# Tool 2: plain SP energy (no NBO) – template for further tools
# ------------------------

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
) -> str:
    """
    Run a single-point SCF energy calculation (no NBO).
    Useful as a building block tool.
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"sp_{os.getpid()}_{int(asyncio.get_event_loop().time())}"

    settings = OrcaJobSettings(
        label=job_label,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="sp",
        use_ri=use_ri,
        extra_keywords=[],  # no NBO
        scf_max_iter=scf_max_iter,
        opt_max_iter=50,
        wall_timeout_seconds=wall_timeout_seconds,
    )

    result = await run_orca(settings)

    header = _status_summary(result)
    if result.status != "ok":
        return header

    energy = extract_total_energy(result.output_text)
    energy_line = (
        f"SCF total energy: {energy:.8f} Eh" if energy is not None else "SCF energy not found."
    )

    return header + "\n" + energy_line + "\n"

# more tools later: run_opt_then_nbo, run_tddft, run_freq, etc.

mcp.run(transport="stdio")
