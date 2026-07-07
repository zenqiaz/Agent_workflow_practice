# QC Agent — User Guide

A natural-language quantum-chemistry workflow system. You describe what you want to
compute; the agent plans and executes ORCA jobs on the lab cluster automatically.

---

## Prerequisites

### Local machine

1. **Python 3.11+** with the following packages:
   ```
   openai>=2.5
   langgraph>=1.0
   mcp>=1.18
   python-dotenv
   rdkit
   openbabel-wheel
   requests
   ```

2. **SSH key** with access to the lab login node (`zhang@130.54.50.100`)
   — contact the server admin to have your public key added.

3. **OpenAI API key** — obtain from [platform.openai.com](https://platform.openai.com).

### Lab cluster (already set up)

- `zhang-ollama` pod in `ns-general` namespace, 36 CPUs
- ORCA 6.1.1 + OPI, SOLVATOR, molSimplify
- MCP server: `/data/zhang/ollama/nbo_agent/server_with_product.py`

---

## Setup

### 1. Get the code

```bash
# Copy the working directory to your machine
# (ask the lab admin for the current snapshot or git clone if available)
```

### 2. Generate an SSH key (if you don't have one) and send the pub key to the administrator.

**macOS / Linux:**
```bash
ssh-keygen -t ed25519 -f ~/.ssh/qclab
# Send ~/.ssh/qclab.pub to the admin
```

**Windows:**
```powershell
ssh-keygen -t ed25519 -f C:\Users\<yourname>\.ssh\qclab
# Send C:\Users\<yourname>\.ssh\qclab.pub to the admin
```

### 3. Create your `.env` file

Create a file named `.env` in the working directory.

**macOS / Linux:**
```env
OPENAI_API_KEY=sk-proj-...

LLM_MODEL=gpt-5.2
LLM_BASE_URL=

MCP_SSH_BIN=ssh
MCP_SSH_KEY=/Users/<yourname>/.ssh/qclab
MCP_SSH_HOST=zhang@130.54.50.100
MCP_SERVER_CMD=kubectl exec -i -n ns-general zhang-ollama -- /data/zhang/ollama/start_mcp_server.sh
```

**Windows:**
```env
OPENAI_API_KEY=sk-proj-...

LLM_MODEL=gpt-4.1-mini
LLM_BASE_URL=

MCP_SSH_BIN=C:/Windows/System32/OpenSSH/ssh.exe
MCP_SSH_KEY=C:/Users/<yourname>/.ssh/qclab
MCP_SSH_HOST=zhang@130.54.50.100
MCP_SERVER_CMD=kubectl exec -i -n ns-general zhang-ollama -- /data/zhang/ollama/start_mcp_server.sh
```

> **Windows note:** use `C:/Windows/System32/OpenSSH/ssh.exe` — Git Bash ssh does not work.

Replace `<yourname>` with your username.

#### Local ORCA mode (no SSH)

If ORCA is installed on your local machine, set `MCP_MODE=local` to run the server
as a local subprocess instead of connecting via SSH:

```env
OPENAI_API_KEY=sk-proj-...

LLM_MODEL=gpt-4.1-mini
LLM_BASE_URL=

MCP_MODE=local
MCP_SERVER_CMD=python server_with_product.py

# Make sure ORCA is on your PATH, or set OPI_ORCA:
# OPI_ORCA=/path/to/orca
```

The server also requires OPI and its dependencies to be installed in the same
Python environment (`pip install opi molSimplify` on the local machine).

### 4. Verify SSH access

**macOS / Linux:**
```bash
ssh -i ~/.ssh/qclab zhang@130.54.50.100 "echo ok"
```

**Windows:**
```powershell
& "C:/Windows/System32/OpenSSH/ssh.exe" -i C:\Users\<yourname>\.ssh\qclab zhang@130.54.50.100 "echo ok"
```

Should print `ok`. If you get `Permission denied (publickey)`, your public key has not been added to the server — contact the admin.

### 5. Run the agent

```bash
python nbo_agent_planning.py
```

---

## Basic Usage

The agent runs as an interactive REPL. Type a natural-language request and press Enter.

### Example requests

```
calculate the pKa of acetone
optimize the geometry of ethanol and compute its IR spectrum
compare EAS reactivity of naphthalene and thiophene using DFT charges
compute the Gibbs free energy of water with 3 explicit water molecules
find the transition state for H2O2 cis-trans isomerization
```

### What happens after you type a request

1. **Compound identification** — the agent looks up each molecule on PubChem,
   shows you a 2D structure image, and asks you to confirm before proceeding.
   Reply `y` to confirm, `n` to skip, or `rename to <new name>` to correct the name.

2. **Plan generation** — the planner LLM produces a JSON workflow plan listing
   all ORCA jobs and derived calculations.

3. **Plan review** — the agent prints the plan and asks `[run / edit / cancel]`.
   Type `run` to execute, or `cancel` to abort.

4. **Execution** — jobs run on the lab cluster. Progress is printed in real time.
   A final report is shown when complete.

---

## REPL Commands

| Command | Effect |
|---------|--------|
| `load <path.xyz>` | Load an XYZ file as the current geometry |
| `state` | Print current session state (loaded geometries, artifacts) |
| `planimg` | Open a visual diagram of the current plan |
| `plansave <path.json>` | Save the current plan to a JSON file |
| `planload <path.json>` | Load a plan from a JSON file and skip planning |
| `run` | Execute the current plan immediately (skip review prompt) |
| `showspec` | Re-display spectra / PES plots from the last run |
| `quit` / `exit` | Exit the REPL |
| *(any other text)* | Send to the planner to generate a new plan |

---

## Loading Your Own Geometry

If you have an XYZ file from a previous calculation or another program:

```
load mystructure.xyz
```

The agent will detect the molecule's identity from its connectivity and confirm with
you. The loaded geometry is then used as the starting point for any optimization or
single-point job.

---

## Available Calculation Types

| Calculation | Example request |
|-------------|----------------|
| Single-point energy | `compute the energy of benzene at PBE0/def2-TZVP` |
| Geometry optimization | `optimize the geometry of aspirin` |
| IR spectrum | `calculate the IR spectrum of ethanol` |
| UV-Vis spectrum (TD-DFT) | `compute the UV-Vis absorption of formaldehyde` |
| pKa (alpha-CH) | `calculate the pKa of acetone` |
| NBO analysis | `run NBO analysis on water` |
| Microsolvation | `optimize caffeine with 3 explicit water molecules` |
| PES scan | `scan the H-C-C-H dihedral of ethane from 0 to 360 degrees` |
| Transition state search | `find the TS for H2O2 cis isomerization` |
| CASSCF | `run CASSCF(6,6) on benzene` |
| EAS reactivity | `rank EAS reactivity of pyrrole and thiophene` |
| Coordination complex | `build Fe(CN)6 3- and compute its energy` |

---

## Output

- **Terminal** — execution progress, node status, and final report are printed live.
- **Spectra / PES plots** — automatically opened as PNG images after runs that produce
  IR spectra, UV-Vis spectra, or potential energy surfaces.
  Re-display with `showspec`.
- **Bug reports** — if a node fails, a detailed report is saved to `bug_reports/`.
- **Runtime reports** — a summary of each run is saved to `runtime_reports/`.

---

## DFT Methods Used (defaults)

The agent selects method and basis set automatically based on the task and molecule size:

| Task | Method | Basis |
|------|--------|-------|
| Geometry optimization | B3LYP/RI | def2-SVP |
| Single-point energy | PBE0/RI | def2-TZVP |
| Frequency / thermochemistry | B3LYP/RI | def2-SVP |
| TD-DFT (UV-Vis) | PBE0 | def2-TZVP |
| NBO | B3LYP | def2-SVP |
| CASSCF | CASSCF | def2-SVP |

You can override by specifying in your request:
```
optimize ethanol at M06-2X/def2-TZVP
```

---

## Automatic Error Recovery

The agent retries failed jobs automatically:

| Error | Recovery |
|-------|----------|
| SCF not converged | Increases max SCF iterations |
| Geometry invalid | Enables xTB pre-optimization |
| Resource limit / timeout | Extends wall time; enables xTB pre-optimization |
| Imaginary frequency after optimization | Perturbs geometry and re-optimizes |

---

## Cost & Runtime

Jobs run on the lab cluster (36 CPUs, ncores=1 per job). Typical wall times:

| Job type | Typical time |
|----------|-------------|
| Single-point (small molecule) | 1–5 min |
| Geometry optimization | 5–20 min |
| Frequency calculation | 10–30 min |
| Microsolvation cluster + freq | 30–60 min |
| PES scan (10 points) | 30–90 min |
| TS search (scan + OptTS + freq) | 60–180 min |

**LLM cost** (OpenAI API): typically $0.01–0.05 per workflow with gpt-4.1-mini.

---

## Troubleshooting

**"Connection closed" / MCP connection error**
- Check that your SSH key is correct and has access to `zhang@130.54.50.100`
- **Windows only:** use `C:/Windows/System32/OpenSSH/ssh.exe` — Git Bash ssh does not work
- **macOS:** SSH is built-in; ensure `MCP_SSH_BIN=ssh` in your `.env`

**"Could not find ORCA" on the server**
- The server start script sets the ORCA path automatically; this should not occur
- If it does, contact the admin

**Compound not found**
- Try a more common name or the IUPAC name
- PubChem and OPSIN are used for lookup; exotic names may not resolve

**Plan looks wrong**
- At the plan review prompt, type `cancel` and rephrase your request with more detail
- E.g. specify the protonation site, charge, or method explicitly
