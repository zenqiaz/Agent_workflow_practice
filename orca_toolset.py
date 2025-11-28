# orca_toolset.py
import os
import textwrap
import subprocess
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List, Optional


# ------------------------
# Config
# ------------------------

ORCA_BINARY = os.environ.get("ORCA_BINARY", "orca")
JOBS_DIR = Path(os.environ.get("ORCA_JOBS_DIR", "./jobs")).expanduser()
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OrcaJobSettings:
    """High-level ORCA job configuration."""
    label: str
    geometry_xyz: str
    charge: int = 0
    multiplicity: int = 1

    method: str = "B3LYP"
    basis: str = "def2-SVP"
    job_type: Literal["sp", "opt"] = "sp"  # affects ! line + %geom block
    use_ri: bool = True
    extra_keywords: List[str] = field(default_factory=list)  # e.g. ["NBO"]

    scf_max_iter: int = 150
    opt_max_iter: int = 50
    wall_timeout_seconds: int = 600


@dataclass
class OrcaRunResult:
    """Result of an ORCA run."""
    label: str
    input_path: Path
    output_path: Path
    command: List[str]
    status: Literal["ok", "timeout", "error"]
    return_code: Optional[int]
    output_text: str


# ------------------------
# Input builder
# ------------------------

def build_orca_input(settings: OrcaJobSettings) -> str:
    """Build the ORCA input text from high-level settings."""

    # ! line
    bang_parts = [settings.method, settings.basis]

    if settings.job_type == "opt":
        bang_parts.append("Opt")

    bang_parts.extend(settings.extra_keywords)

    if settings.use_ri:
        bang_parts.extend(["RIJCOSX", "def2/J"])

    bang_parts.append("TightSCF")

    bang_line = "! " + " ".join(bang_parts)

    # blocks
    blocks = []

    blocks.append(
        textwrap.dedent(
            f"""\
            %scf
              MaxIter {settings.scf_max_iter}
            end
            """
        )
    )

    if settings.job_type == "opt":
        blocks.append(
            textwrap.dedent(
                f"""\
                %geom
                  MaxIter {settings.opt_max_iter}
                end
                """
            )
        )

    block_text = "\n".join(blocks).strip()

    # geometry block
    geom_block = textwrap.dedent(
        f"""\
        * xyz {settings.charge} {settings.multiplicity}
        {settings.geometry_xyz.strip()}
        *
        """
    )

    return textwrap.dedent(
        f"""\
        {bang_line}

        {block_text}

        {geom_block}
        """
    )


# ------------------------
# Running ORCA
# ------------------------

def _run_orca_sync(settings: OrcaJobSettings) -> OrcaRunResult:
    """Synchronous ORCA run; for async use asyncio.to_thread."""
    job_dir = JOBS_DIR / settings.label
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"{settings.label}.inp"
    output_path = job_dir / f"{settings.label}.out"

    inp_text = build_orca_input(settings)
    input_path.write_text(inp_text)

    cmd = [ORCA_BINARY, input_path.name]  # cwd = job_dir

    env = os.environ.copy()

    try:
        with output_path.open("w") as fout:
            proc = subprocess.run(
                cmd,
                stdout=fout,
                stderr=subprocess.STDOUT,
                cwd=str(job_dir),
                text=True,
                timeout=settings.wall_timeout_seconds,
                env=env,
            )
        status = "ok" if proc.returncode == 0 else "error"
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
        return_code = None

    if output_path.exists():
        out_text = output_path.read_text(encoding="utf-8", errors="ignore")
    else:
        out_text = ""

    return OrcaRunResult(
        label=settings.label,
        input_path=input_path,
        output_path=output_path,
        command=cmd,
        status=status,
        return_code=return_code,
        output_text=out_text,
    )


async def run_orca(settings: OrcaJobSettings) -> OrcaRunResult:
    """Async wrapper for ORCA run."""
    return await asyncio.to_thread(_run_orca_sync, settings)


# ------------------------
# Common parsers
# ------------------------

def extract_nbo_section(output_text: str) -> str:
    """Best-effort extract of NBO/NPA-related part of the output."""
    lines = output_text.splitlines()
    keep = []
    in_section = False

    markers_start = (
        "NATURAL POPULATION ANALYSIS",
        "NATURAL POPULATIONS",
        "NATURAL BOND ORBITALS",
        "NBO ANALYSIS",
    )
    markers_end = (
        "MULLIKEN POPULATION ANALYSIS",
        "LOEWDIN POPULATION ANALYSIS",
        "ORBITAL POPULATIONS",
        "Total charge on system",
    )

    for line in lines:
        if any(m in line for m in markers_start):
            in_section = True

        if in_section:
            keep.append(line)

        if in_section and any(m in line for m in markers_end):
            break

    if not keep:
        # fallback: last chunk
        keep = lines[-200:]

    return "\n".join(keep)


def extract_total_energy(output_text: str) -> Optional[float]:
    """Extract SCF total energy (Eh) if present."""
    for line in output_text.splitlines():
        line = line.strip()
        # ORCA 4.x example: "TOTAL SCF ENERGY       -76.4321"
        if line.upper().startswith("TOTAL SCF ENERGY"):
            parts = line.split()
            try:
                return float(parts[-1])
            except (ValueError, IndexError):
                continue
    return None
