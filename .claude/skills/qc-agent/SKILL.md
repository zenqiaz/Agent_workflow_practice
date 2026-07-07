---
name: qc-agent
description: Quantum chemistry agent for ORCA-based molecular calculations. Use this skill when the user wants to compute molecular energies, optimise geometries, calculate vibrational spectra (IR/Raman), UV-Vis absorption, NBO charges/bond orders, pKa, free energies, potential energy surface scans, transition state searches, CASSCF multi-reference calculations, or any ORCA-based QC workflow. Covers how to launch the agent REPL, compose natural-language requests, interpret results, and understand what calculations are running under the hood.
license: MIT
metadata:
    skill-author: QC Agent Project
---

# ORCA QC Agent

## Overview

The QC agent orchestrates ORCA 6.1.1 calculations on a remote compute node (lab k8s /
DigitalOcean VM) via a Model Context Protocol (MCP) server over SSH. The user sends
natural-language requests; the planner LLM (gpt-4.1-mini) produces a JSON workflow plan;
LangGraph executes it deterministically, collecting results into a structured artifact store.

**Key facts:**
- DFT engine: ORCA 6.1.1 on a 36-CPU Linux node
- Default method: B3LYP/def2-SVP (opt); PBE0/def2-TZVP (single point)
- All energies returned in Hartree (Eh)
- Results stored in `state["artifacts"]` keyed by human-readable names

## Launching the Agent

```bash
# Default (OpenAI gpt-4.1-mini + lab k8s ORCA):
python nbo_agent_planning.py

# Local LLM (qwen2.5:14b, requires kubectl port-forward):
ENV_FILE=.env.local python nbo_agent_planning.py

# Fallback (DigitalOcean VM):
ENV_FILE=.env.vm python nbo_agent_planning.py
```

## REPL Commands

| Command | Effect |
|---------|--------|
| `load <path>` | Load an XYZ file as current geometry |
| `state` | Print session state (geometries, artifacts) |
| `planimg` | Visualise current plan as image |
| `plansave <path>` | Save plan JSON to file |
| `planload <path>` | Load plan JSON from file |
| `run` | Execute current plan via LangGraph |
| `showspec` | Re-display spectra / PES plots from last run |
| `quit` / `exit` | Exit the REPL |
| (any other text) | Send to planner → generate new plan |

## How a Calculation Runs

```
User request
  └─ Compound ID (LLM extracts names → PubChem → 2D image → user confirms)
       └─ Skill dispatch (chemistry context injected into planner)
            └─ Planner LLM → JSON plan
                 └─ User reviews → "run"
                      └─ LangGraph executes nodes
                           └─ Artifacts collected → report + spectra auto-displayed
```

---

## Tool Reference

### run_sp_energy — Single-Point Energy

```
method="B3LYP", basis="def2-SVP", use_ri=True, wall_timeout_seconds=600
```

Returns: `{"energy_eh": -228.456}`

Typical product_spec: `{"E_mol_eh": "energy_eh"}`

---

### run_opt_job — Geometry Optimisation

```
method="B3LYP", basis="def2-SVP", use_ri=True, opt_max_iter=100,
wall_timeout_seconds=1800
```

Returns: `{"energy_eh": -228.456, "geometry_xyz": "C  0.000 ..."}` — geometry flows
to next node via `output_id`, not stored in artifacts.

---

### run_freq_job — Frequency + Thermochemistry

```
method="B3LYP", basis="def2-SVP", use_ri=True, wall_timeout_seconds=3600
```

Returns:
```json
{"energy_eh": -228.456, "enthalpy_eh": -228.423, "gibbs_free_energy_eh": -228.401}
```

**Always use `gibbs_free_energy_eh` for ΔG and pKa.**

Typical product_spec: `{"G_mol_eh": "gibbs_free_energy_eh"}`

---

### run_spectrum_job — IR / Raman Spectrum

```
spectrum_type="ir"|"raman"|"ir_raman", method="B3LYP", basis="def2-SVP",
wall_timeout_seconds=3600
```

Returns: same as run_freq_job plus:
```json
{
  "ir_spectrum": [{"mode": 7, "freq_cm1": 1052.3, "intensity_km_mol": 45.2}, ...],
  "raman_spectrum": [{"mode": 7, "freq_cm1": 1052.3, "activity": 12.4, "depolarization": 0.75}]
}
```

Imaginary modes (TS confirmation) appear in `ir_spectrum` with negative `freq_cm1` and
`intensity_km_mol: 0.0`. Modes with |freq| ≤ 10 cm⁻¹ excluded.
Raman requires `spectrum_type="raman"` — roughly 2× cost.

Typical product_spec: `{"ir_spectrum_mol": "ir_spectrum", "G_mol_eh": "gibbs_free_energy_eh"}`

---

### run_tddft_job — TD-DFT UV-Vis Excited States

```
n_states=5, method="B3LYP", basis="def2-SVP", use_ri=True, wall_timeout_seconds=3600
```

**Input geometry must be pre-optimised** — run run_opt_job first.

Returns:
```json
{
  "energy_ground_state_eh": -114.123,
  "excited_states": [
    {"state": 1, "energy_ev": 3.89, "wavelength_nm": 319.0, "oscillator_strength": 0.032},
    {"state": 2, "energy_ev": 4.12, "wavelength_nm": 301.0, "oscillator_strength": 0.234}
  ]
}
```

`oscillator_strength >> 0` = bright (electric-dipole-allowed); `~ 0` = dark/forbidden.

Typical product_spec: `{"excited_states_mol": "excited_states"}`

---

### run_nbo_job — NBO Analysis

```
job_type="sp"|"opt", method="B3LYP", basis="def2-SVP", use_ri=True,
wall_timeout_seconds=600
```

Returns: `nbo_summary` (text), `wiberg_bond_orders` (list), `natural_charges` (list).

Typical product_spec: `{"nbo_mol": "nbo_summary", "charges_mol": "natural_charges"}`

---

### run_scan_job — Relaxed PES Scan

```
scan_coords="<JSON string>", method="B3LYP", basis="def2-SVP",
wall_timeout_seconds=7200, ncores=1 (REQUIRED)
```

`scan_coords` must be a **JSON string** (not a list), e.g.:
```
'[{"type":"D","atoms":[2,0,1,3],"start":0.0,"end":360.0,"n_points":13}]'
```
- `type`: `"B"` = bond (Å, 2 atoms), `"A"` = angle (°, 3 atoms), `"D"` = dihedral (°, 4 atoms)
- `atoms`: 0-based ORCA atom indices — **must be verified, do not guess**
- `n_points`: total points inclusive

**CRITICAL: ncores=1 always.** MPI + ORCA relaxed scan broken in ORCA 6.1.1.

Returns:
```json
{
  "scan_results": [{"step": 0, "value": 0.0, "energy_eh": -151.234}, ...],
  "min_energy_eh": -151.256,  "min_value": 180.0,
  "geometry_xyz": "(min geometry)",
  "max_geometry_xyz": "(max geometry = TS candidate)"
}
```

Typical product_spec: `{"scan_results_label": "scan_results"}`

---

### run_ts_opt_job — Transition State Optimisation (OptTS)

```
calc_hess=True (strongly recommended), opt_max_iter=100, wall_timeout_seconds=3600
```

Starting geometry must be near the TS — use `max_geometry_xyz` from run_scan_job.

Returns: `{"geometry_xyz": "...", "energy_eh": -151.234, "ts_converged": true}`

**Confirm TS**: run run_spectrum_job on the result. A true TS has **exactly one** negative
`freq_cm1` in `ir_spectrum`.

Typical plan pattern:
```
scan (output_id: ts_candidate)
  → ts_opt (input_id: ts_candidate, output_id: ts_structure)
  → spectrum (input_id: ts_structure) — confirms 1 imaginary freq
```

---

### run_casscf_job — CASSCF Multi-Reference

```
nel=2, norb=2, nroots=1, basis="def2-SVP", scf_max_iter=200,
wall_timeout_seconds=3600
```

Active space guide:
- Fe(III) d⁵ → CAS(5,5); Fe(II) d⁶ → CAS(6,5)
- Benzene π → CAS(6,6); butadiene → CAS(4,4)

Set `nroots > 1` for state-averaged SA-CASSCF.

Returns: `{"energy_eh": -1815.372}` (single state) or adds `"energies_eh": [...]` (SA).

---

### run_solvator_cluster_thermo — Explicit Solvation + Thermochemistry

```
nsolv=3, method="r2scan-3c", basis="" (leave empty), wall_timeout_seconds=600,
thermo_timeout_seconds=3600
```

Builds explicit water cluster with SOLVATOR + ORCA frequency in one call.
Standard choice: `nsolv=3`, `method="r2scan-3c"` (composite method, no basis needed).

Returns: same structure as run_freq_job (energy_eh, enthalpy_eh, gibbs_free_energy_eh).

---

### structure_add_remove_proton — Proton Edit (client-side, instant)

```
mode="add"|"remove", site_selector=None, variant=0,
h_index=None, target_atom_index=None
```

Returns new geometry_xyz with updated charge/multiplicity.

---

### name_to_geometry_xyz — Name to 3D Geometry (client-side)

Input: compound name, SMILES, or element symbol (e.g. "Fe", "Fe3+", "chloride").

Returns: `{"status": "ok", "geometry": "C  0.000 ...", "smiles": "CCO", "formula": "C2H6O"}`

---

### build_coordination_complex — Build Complex (client-side)

```python
build_coordination_complex(
    metal="Fe",           # element symbol, NOT a geometry ID
    ligands=["cyanide"]*6,
    geometry="octahedral",
    charge=-3,
    multiplicity=6,
    bond_length=2.0       # Angstrom
)
```

Built-in ligand names: chloride, cyanide, hydroxide, water, ammonia, CO, NO, acetate, etc.

---

## Common Workflows

### pKa Calculation

```
> Calculate the pKa of acetic acid
```

Plan outline:
1. `name_to_geometry_xyz("acetic acid")` → store as `ha_neutral`
2. `structure_add_remove_proton(mode="remove")` → `a_anion`
3. `run_freq_job` on `ha_neutral` → artifact `G_HA_eh` (gibbs_free_energy_eh)
4. `run_freq_job` on `a_anion` → artifact `G_A_minus_eh`
5. `calc_expr`: pKa formula below

**pKa formula:**
```
ΔG_eh = G_A_minus_eh + G_H_plus_ref_eh - G_HA_eh
ΔG_kJ = ΔG_eh * 2625.5
pKa = ΔG_kJ / (8.314e-3 * 298.15 * ln(10))
```

Reference proton: **G(H⁺) = −0.01372 Eh** (Tissandier et al.) — always in plan `settings`.

---

### IR / Raman Spectrum

```
> Get the IR spectrum of caffeine
```

Plan outline:
1. `name_to_geometry_xyz("caffeine")`
2. `run_opt_job` → optimised geometry
3. `run_spectrum_job(spectrum_type="ir")` → `ir_spectrum_caffeine`

Spectrum PNG auto-displayed. Re-display with `showspec`.

---

### UV-Vis Absorption (TD-DFT)

```
> UV-Vis spectrum of formaldehyde, 8 excited states
```

Plan outline:
1. `name_to_geometry_xyz("formaldehyde")`
2. `run_opt_job`
3. `run_tddft_job(n_states=8)` → `excited_states_formaldehyde`

UV-Vis PNG auto-displayed. For charge-transfer molecules, use `method="CAM-B3LYP"`.

---

### Transition State Search

```
> Find the TS for H2O2 cis conformation
```

Plan outline:
1. `name_to_geometry_xyz("H2O2")` — verify atom indices: O(0)O(1)H(2)H(3)
2. `run_scan_job(scan_coords='[{"type":"D","atoms":[2,0,1,3],"start":0.0,"end":180.0,"n_points":10}]')`
   → `max_geometry_xyz` stored as `ts_candidate`
3. `run_ts_opt_job(input_id="ts_candidate", output_id="ts_structure", calc_hess=True)`
4. `run_spectrum_job(input_id="ts_structure")` — confirm exactly 1 negative freq

---

### Explicit Microsolvation pKa

```
> pKa of acetic acid with 3 explicit water molecules
```

Replace `run_freq_job` nodes with `run_solvator_cluster_thermo(nsolv=3)`.
Use `method="r2scan-3c"` and leave `basis=""`.

---

### NBO Analysis

```
> NBO charges and bond orders of CO2
```

Plan outline:
1. `name_to_geometry_xyz("CO2")`
2. `run_opt_job`
3. `run_nbo_job(job_type="sp")` on the optimised geometry

NBO output contains natural charges, Wiberg bond orders, donor-acceptor interactions.

---

### CASSCF Multi-Reference

```
> CASSCF(5,5) energy of [Fe(CN)6]3- high-spin
```

Plan outline:
1. `build_coordination_complex(metal="Fe", ligands=["cyanide"]*6, geometry="octahedral", charge=-3, multiplicity=6)`
2. `run_opt_job` (B3LYP/def2-SVP)
3. `run_casscf_job(nel=5, norb=5, nroots=1, basis="def2-SVP")`

---

## DFT Method Recommendations

| Task | Method | Basis |
|------|--------|-------|
| Geometry optimisation | B3LYP | def2-SVP |
| Single-point energy | PBE0 | def2-TZVP |
| Thermochemistry (pKa, ΔG) | B3LYP | def2-SVP (freq at same level as opt) |
| UV-Vis (standard) | B3LYP | def2-SVP |
| UV-Vis (charge-transfer) | CAM-B3LYP | def2-SVP |
| Solvation clusters | r2scan-3c | (leave empty) |
| CASSCF | CASSCF | def2-SVP (use_ri=False) |

Wall-time defaults: SP=600s, Opt=1800s, Freq=3600s, Scan=7200s, TS=3600s, CASSCF=3600s

---

## Unit Conversions

| Quantity | Agent unit | Conversion |
|----------|-----------|------------|
| Energy | Hartree (Eh) | × 2625.5 = kJ/mol; × 627.5 = kcal/mol |
| pKa | dimensionless | ΔG(kJ/mol) / (R × T × ln10); R=8.314e-3 kJ/(mol·K), T=298.15 K |
| Frequencies | cm⁻¹ | negative = imaginary (TS confirmation mode) |
| UV-Vis | nm, eV | from TD-DFT excited_states list |

---

## Interpreting Results

```python
# After pKa calculation:
state["artifacts"] == {
    "G_HA_eh": -228.456,
    "G_A_minus_eh": -227.890,
    "G_H_plus_ref_eh": -0.01372,
    "delta_G_eh": -0.00045,
    "pka": 9.3,
}

# After spectrum calculation:
state["artifacts"] == {
    "ir_spectrum_caffeine": [
        {"mode": 7, "freq_cm1": 1052.3, "intensity_km_mol": 45.2},
        ...
    ],
    "gibbs_free_energy_caffeine_eh": -678.123,
}

# After TD-DFT:
state["artifacts"] == {
    "excited_states_mol": [
        {"state": 1, "energy_ev": 3.89, "wavelength_nm": 319.0, "oscillator_strength": 0.032},
        ...
    ],
}
```

Spectra (IR, UV-Vis, PES) are automatically plotted and opened as PNG images after `run`.
Re-display at any time with `showspec`.
