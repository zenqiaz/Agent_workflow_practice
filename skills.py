"""
Planner skill modules for the QC agent.

Each skill injects domain-specific chemical reasoning into the planner as a
focused system-message block, replacing a monolithic system prompt with
modular, task-triggered context.

Usage:
    from skills import run_planning_skills
    skill_contexts = run_planning_skills(user_text, state)
    # → list[str], one entry per matched skill
"""
from __future__ import annotations
from typing import Any, Dict, List


class PlannerSkill:
    """Base class for planner skills."""
    name: str = ""
    priority: int = 50  # lower = injected earlier in message list

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        raise NotImplementedError

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Individual skills
# ---------------------------------------------------------------------------

class MethodSelectionSkill(PlannerSkill):
    """Always-active skill: recommend DFT method/basis by task and molecule size."""

    name = "method_selection"
    priority = 10  # injected first — sets the general QC context

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        return True  # always inject; content is lightweight

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        # Estimate molecule size from loaded geometries
        geometries = state.get("geometries") or {}
        max_atoms = 0
        for xyz in geometries.values():
            n = sum(1 for ln in xyz.splitlines() if ln.strip() and ln.split()[0].isalpha())
            max_atoms = max(max_atoms, n)

        size_note = ""
        if max_atoms > 0:
            if max_atoms <= 10:
                size_note = f"Loaded geometry has {max_atoms} atoms (small molecule)."
            elif max_atoms <= 50:
                size_note = f"Loaded geometry has {max_atoms} atoms (medium molecule)."
            else:
                size_note = f"Loaded geometry has {max_atoms} atoms (large — prefer cheaper methods)."

        return f"""SKILL: Method & Basis Set Selection
{size_note}

Recommended levels of theory by task:
  Geometry optimisation : B3LYP/def2-SVP  (fast, adequate for most organics)
  Single-point energy   : B3LYP/def2-TZVP or PBE0/def2-TZVP  (more accurate)
  Thermochemistry (G/H) : run_freq_job at the same level as opt
  Microsolvation cluster: r2scan-3c (good accuracy/cost ratio for freq on clusters)
  NBO analysis          : B3LYP/def2-SVP with NBO keyword
  HOMO-LUMO gap (KS)    : run_sp_energy at PBE0/def2-TZVP; returns homo_lumo_gap_ev
  HOMO-LUMO + UV-Vis    : run_tddft_job; returns homo_lumo_gap_ev + excited_states

Basis set guidance:
  def2-SVP   → cheap, opt/freq, qualitative
  def2-TZVP  → accurate single-points, properties
  def2-TZVPP → high-accuracy; use only when def2-TZVP is insufficient

Performance flags:
  use_ri=True  reduces cost of hybrid DFT significantly (RIJCOSX in ORCA)
  ncores: default 1; increase for large molecules if VM allows

Wall-time limits (always set wall_timeout_seconds for every calc node):
  run_sp_energy         :  600 s
  run_opt_job           : 1200 s
  run_freq_job          : 1800 s
  run_scan_job          : 3600 s
  run_ts_opt_job        : 1800 s
  run_casscf_job        : 3600 s
  run_solvator_cluster_thermo: 3600 s
""".strip()


class PKaSkill(PlannerSkill):
    """pKa calculation protocol, reference values, calibration, and benchmarks."""

    name = "pka_calibration"
    priority = 20

    _KEYWORDS = ("pka", "pk_a", "acid dissociation", "acidity", "deprotonation",
                 "conjugate base", "ionization constant", "ka ")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: pKa Calculation Protocol
────────────────────────────────────────────────────────────

Reference proton free energy (gas phase, standard protocol):
  G(H⁺)_ref = -0.01372 Eh   (Tissandier et al. 1998; include in plan settings)

Required artifacts:
  G_HA_eh        — Gibbs free energy of the neutral acid (Hartree)
  G_A_minus_eh   — Gibbs free energy of the conjugate base (Hartree)
  Both MUST come from run_freq_job nodes, NOT from run_sp_energy.
  (SP energies lack zero-point and thermal corrections → unacceptable for pKa.)

Formula (for the LLM calc node):
  ΔG_eh  = G_A_minus_eh + G_H_plus_ref_eh − G_HA_eh
  ΔG_J   = ΔG_eh × 2625499.638        (Hartree → J/mol)
  pKa    = ΔG_J / (8.314462618 × 298.15 × 2.302585093)

Recommended level of theory:
  Gas phase    : B3LYP/def2-SVP  (opt + freq on same level)
  Higher acc.  : opt B3LYP/def2-SVP → SP B3LYP/def2-TZVP
  Aqueous pKa  : use run_solvator_cluster_thermo (nsolv ≥ 3) for HA and A⁻

Known systematic errors — gas-phase B3LYP/def2-SVP:
  Computed pKa values are typically 2–4 units too HIGH vs. aqueous experiment.
  Do NOT apply ad-hoc corrections in the plan; report raw computed value and
  note the systematic offset in the final_report fields.

Calibration benchmarks (gas-phase B3LYP/def2-SVP, for sanity check):
  Molecule      Exp. pKa(aq)   Typical computed gas-phase
  HF               3.2              ~5–6
  CH₃COOH          4.8              ~7–8
  H₂O             15.7              ~18–20
  NH₄⁺             9.2              ~11–12
  HCl             −7               ~−4 to −5

Plan validation rules:
  ✓ artifacts_to_save MUST include G_HA_eh and G_A_minus_eh
  ✓ Both freq nodes must set  product: {"G_XX_eh": "gibbs_free_energy_eh"}
  ✓ LLM calc node must list  needs_artifacts: ["G_HA_eh", "G_A_minus_eh"]
  ✗ Do NOT use SP energy as proxy for G in a pKa plan
""".strip()


class TSSearchSkill(PlannerSkill):
    """Transition-state search: relaxed scan → OptTS → freq verification."""

    name = "ts_search"
    priority = 27  # between ThermochemistrySkill (25) and SolvationSkill (30)

    _KEYWORDS = (
        "transition state", "ts search", "ts opt", "saddle point",
        "activation barrier", "activation energy", "optts", "ts structure",
        "find ts", "locate ts", "reaction barrier", "reaction pathway",
        "bond breaking", "bond forming",
    )

    def matches(self, user_text: str, state: dict) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: dict) -> str:
        return """SKILL: Transition State (TS) Search Protocol
────────────────────────────────────────────────────────────

Workflow: scan reaction coordinate → OptTS on PES maximum → freq verification

Step 1 — Relaxed PES scan (run_scan_job)
  Scan the reaction coordinate (bond/angle/dihedral) to locate the approximate TS.
  run_scan_job returns:
    scan_results:     [{step, value, energy_eh}, ...]
    geometry_xyz:     geometry at the PES MAXIMUM (TS candidate)
  Use output_id on the scan node to store the PES-maximum geometry as the TS candidate.

Step 2 — OptTS on PES maximum (run_ts_opt_job)
  Input:     PES-maximum geometry (from scan node output_id)
  calc_hess: ALWAYS set true (required for reliable TS optimisation)
  Returns:   geometry_xyz (optimised TS), energy_eh, ts_converged
  Use output_id to store the optimised TS geometry.

Step 3 — Freq verification (run_freq_job or run_spectrum_job)
  Run on the optimised TS geometry.
  A true transition state has EXACTLY ONE negative (imaginary) frequency.
  In ir_spectrum, imaginary modes appear as negative freq_cm1 values.
  Use run_spectrum_job (spectrum_type="ir") to get both E/H/G and the ir_spectrum.

Plan pattern (4 nodes):
  {load_mol}
  {scan_rc:    run_scan_job,    output_id: "ts_candidate",
               product: {"scan_results_rc": "scan_results"}}
  {ts_opt:     run_ts_opt_job,  input_id: "ts_candidate", output_id: "ts_structure",
               args: {calc_hess: true},
               product: {"ts_energy_eh": "energy_eh"}}
  {ts_freq:    run_spectrum_job, input_id: "ts_structure",
               product: {"ts_ir_spectrum": "ir_spectrum",
                         "ts_gibbs_eh": "gibbs_free_energy_eh"}}

scan_coords notes:
  Choose the coordinate that changes most along the reaction path.
  Use n_points = 10–15 for TS searches (coarser scan to locate max).
  Do NOT set ncores > 1 for scan (MPI scan mode has known issues).

CRITICAL — no kind:"llm" node:
  The TS structure and frequencies are returned as structured data.
  QC-CALCULATOR cannot process geometry or frequency data.
  Confirm TS by checking ts_ir_spectrum for exactly 1 negative freq_cm1 entry
  in the final_report fields — no LLM node needed.

Artifact naming convention:
  scan_results_rc    — PES scan results
  ts_energy_eh       — OptTS final energy
  ts_ir_spectrum     — IR spectrum of TS (check for 1 imaginary mode)
  ts_gibbs_eh        — Gibbs free energy of TS (for activation barrier ΔG‡)
""".strip()


class CasscfSkill(PlannerSkill):
    """CASSCF / SA-CASSCF single-point protocol."""

    name = "casscf"
    priority = 24

    _KEYWORDS = [
        "casscf", "cas(", "cas (", "active space", "multireference",
        "multi-reference", "sa-casscf", "state-averaged", "state averaged",
        "complete active space", "mcscf",
    ]

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        lt = user_text.lower()
        return any(kw in lt for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return _CASSCF_SKILL_TEXT


_CASSCF_SKILL_TEXT = """
SKILL: CASSCF / SA-CASSCF calculations

Tool
  run_casscf_job
  Required: nel (active electrons), norb (active orbitals)
  Optional: nroots (default 1), basis (default def2-SVP)
  Returns: energy_eh (total or SA-average), energies_eh list (only when nroots > 1)

Active space selection guide
  - Fe(III) d5 system:        CAS(5,5)  — 5 d-electrons in 5 d-orbitals
  - Fe(II)  d6 system:        CAS(6,5)  — 6 d-electrons in 5 d-orbitals
  - Ni(II)  d8 system:        CAS(8,5)  — 8 d-electrons in 5 d-orbitals
  - Simple bond dissociation: CAS(2,2)  — bonding + antibonding pair
  - Aromatic pi system:       CAS(N,N)  — N electrons in N pi orbitals (benzene: CAS(6,6))
  If unsure, ask the user or use the minimal valence active space.

Spin state comparison (SA-CASSCF)
  - To compare multiple spin states simultaneously, set nroots to the number of states.
  - The multiplicity parameter controls the spin of root 0; SA averages over all roots.
  - For spin-state energy differences, prefer running separate CASSCF jobs with the
    correct multiplicity for each state (single-state CASSCF, nroots=1).

Basis sets
  - def2-SVP: good starting point; affordable
  - def2-TZVP or cc-pVTZ: better accuracy for final results
  - ANO-RCC or def2-TZVPP recommended for high-accuracy metal active spaces

Typical plan pattern (spin-state gap)
  nodes:
    - {id: casscf_hs, kind: tool, tool: run_casscf_job,
       input_id: mol_hs, args: {nel: 5, norb: 5, nroots: 1},
       product: {e_hs_eh: energy_eh}}
    - {id: casscf_ls, kind: tool, tool: run_casscf_job,
       input_id: mol_ls, args: {nel: 5, norb: 5, nroots: 1},
       product: {e_ls_eh: energy_eh}}
    - {id: calc_gap, kind: llm, task: compute_spin_gap,
       needs: [casscf_hs, casscf_ls], needs_artifacts: [e_hs_eh, e_ls_eh],
       product: {delta_E_kcal: delta_E_kcal}}

No geometry optimization with CASSCF
  run_casscf_job is SP-only. If you need CASSCF-optimised geometry,
  first optimize with DFT (run_opt_job), then run run_casscf_job on the
  DFT-optimized structure.

on_error rules
  - SCF_NOT_CONVERGED → patch: {scf_max_iter: 500}, max_attempts: 1
""".strip()


class SolvationSkill(PlannerSkill):
    """Microsolvation / explicit solvation protocol and nsolv guidance."""

    name = "solvation"
    priority = 30

    _KEYWORDS = ("solv", "aqueous", "water cluster", "microsolvat", "explicit water",
                 "nsolv", "solvator", "hydrat")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: Explicit Solvation (Microsolvation) Protocol
────────────────────────────────────────────────────────────

Tool: run_solvator_cluster_thermo
  → Builds water cluster with SOLVATOR, then runs ORCA freq in one call.
  → Returns: cluster_xyz, energy_eh, enthalpy_eh, gibbs_free_energy_eh
  → Use this instead of separate run_solvator_cluster + run_freq_job.

nsolv selection guide:
  nsolv = 1–2  : qualitative, fast, suitable for screening
  nsolv = 3    : standard protocol for pKa and ΔG in polar environments
  nsolv = 4–6  : for molecules with multiple H-bond donors/acceptors
  nsolv ≥ 7    : large polar molecules; expect >30 min compute time

Workflow pattern for solvated pKa:
  load HA → solvator_thermo(HA, nsolv=3) → G_HA_cluster_eh
          → remove_proton → solvator_thermo(A⁻, nsolv=3) → G_A_cluster_eh
          → LLM calc pKa (same formula as gas phase)

Important constraints:
  - Charge of cluster = charge of solute (SOLVATOR keeps solute charge)
  - run_solvator_cluster_thermo validates that cluster gained atoms vs. input;
    returns error if SOLVATOR did not add water molecules.
  - Freq on cluster can take 30–60 min; set thermo_timeout_seconds accordingly.
  - Use r2scan-3c as method for cluster freq (better cost/accuracy than B3LYP).

Artifact naming convention:
  Solvated:   G_HA_solv_eh, G_A_minus_solv_eh
  Gas phase:  G_HA_eh, G_A_minus_eh
  Keep both sets if comparing gas vs. solution pKa.
""".strip()


class ThermochemistrySkill(PlannerSkill):
    """Thermochemistry conventions: freq vs SP, what run_freq_job returns, units."""

    name = "thermochemistry"
    priority = 25

    _KEYWORDS = ("gibbs", "free energy", "enthalpy", "thermochem", "freq",
                 "zero-point", "zpe", "delta g", "deltag", "delta_g",
                 "equilibrium", "reaction energy")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: Thermochemistry Conventions
────────────────────────────────────────────────────────────

run_freq_job returns (all in Hartree):
  energy_eh              — FINAL SINGLE POINT ENERGY (E_elec)
  enthalpy_eh            — Total Enthalpy  H = E + ZPE + thermal H
  gibbs_free_energy_eh   — Final Gibbs free energy  G = H − TS  (T=298.15 K)

Use the right quantity:
  For ΔG, pKa, equilibrium K  → use gibbs_free_energy_eh
  For ΔH, bond energies        → use enthalpy_eh
  For single-point comparisons → use energy_eh (only if no thermal correction needed)

Artifact naming convention (follow consistently):
  G_<species>_eh     — Gibbs free energy in Hartree
  H_<species>_eh     — Enthalpy in Hartree
  E_<species>_eh     — Electronic energy in Hartree

Unit conversions (available to the calculator LLM automatically):
  1 Hartree = 2625499.638 J/mol = 627.509 kcal/mol = 27.211 eV

Temperature:
  Default T = 298.15 K. Override via plan settings: {"temperature_K": 350}.

Plan rule:
  If the goal involves ΔG, K, or pKa, the plan MUST use run_freq_job — not
  run_sp_energy — unless the task explicitly states SP is acceptable.
""".strip()


class ProtonationSiteSkill(PlannerSkill):
    """Protonation site comparison: site_selector API, distinct sites, SP-only workflow."""

    name = "protonation_site"
    priority = 15  # after method selection, before pKa

    _KEYWORDS = ("protonation site", "site of protonation", "protoniz", "protonate",
                 "most basic site", "proton affinity", "site of proton")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: Protonation Site Analysis
────────────────────────────────────────────────────────────

Tool: structure_add_remove_proton  (mode="add" or "remove")

── site_selector kinds (STRING, not a dict) ──────────────────

1. Direct — line number (most explicit; no chemistry needed):
     site_selector="line:N"   →  Nth atom in the XYZ block (1-based)
     Works for both add and remove. For remove, if line N is a heavy atom,
     the tool automatically removes the bonded H.

2. Direct — atom marker (simplest; no naming needed):
     Add  *  as the 5th column on the target atom line, then pass site_selector="*":
       O   1.2  0.0  0.0  *
     The tool picks the first atom with any tag when selector is "*".
     For multiple marked atoms use variant=0, 1, … to step through them.

     Named tags also work if you want explicit labels:
       O   1.2  0.0  0.0  O_target
       O  -0.6  1.04 0.0  # O_hydroxyl    ← '#' prefix is stripped automatically
     Then pass  site_selector="O_target"  (case-insensitive; leading '@' stripped).
     Works for both add and remove with the same heavy-atom→bonded-H logic.

3. Semantic — bond-inference (when you know the chemical role):
     "oxygen_terminal"   → O atoms NOT bonded to any H  (e.g. N=O in HNO3)
     "oxygen_hydroxyl"   → O atoms bonded to H           (e.g. N-OH in HNO3)
     "nitrogen"          → N atoms

── variant parameter (int, default 0) ───────────────────────
  Selects among multiple atoms of the same type, ordered by atom index.
  Symmetry-equivalent atoms give identical energies — always use variant=0
  when the goal is to compare DISTINCT sites.
  Do NOT generate two nodes differing only in variant; they will give the same result.

── Distinct sites for common molecules ─────────────────────
  HNO3 → exactly 3 distinct sites:
    1. site_selector="oxygen_terminal"  (N=O oxygen)
    2. site_selector="oxygen_hydroxyl"  (N-OH oxygen)
    3. site_selector="nitrogen"
  Do NOT add a 4th node for the second N=O oxygen (symmetry-equivalent).

── CRITICAL: SP-only workflow ───────────────────────────────
  DO use run_sp_energy on each protonated isomer.
  DO NOT use run_opt_job before SP on protonated isomers —
    optimisation migrates the proton to the global minimum,
    collapsing all isomers to the same structure and energy.

Correct plan pattern:
  neutral → add_proton(site_selector="oxygen_terminal") → run_sp_energy → E_H_Oterminal_eh
  neutral → add_proton(site_selector="oxygen_hydroxyl") → run_sp_energy → E_H_Ohydroxyl_eh
  neutral → add_proton(site_selector="nitrogen")        → run_sp_energy → E_H_N_eh
  LLM node: compare energies, rank sites

LLM calc node prompt — MUST include this data integrity check:
  "Before ranking sites, check: if any two E_H_<site>_eh values are exactly equal
   or differ by less than 1e-6 Eh, set status='error' and report which values are
   identical and the likely cause (duplicate input structures, or opt was run before
   SP). Do NOT rank or recommend a site if any pair of energies is identical."
""".strip()


class ScanSkill(PlannerSkill):
    """Relaxed potential-energy-surface (PES) scan protocol."""

    name = "scan"
    priority = 21   # before SpectrumSkill (22) and TDDFTSkill (23)

    _KEYWORDS = (
        "scan", "pes scan", "potential energy surface", "reaction coordinate",
        "bond scan", "angle scan", "dihedral scan", "surface scan",
        "energy profile", "dissociation curve", "torsion scan", "rotational scan",
        "scan energy", "scan bond", "scan angle",
    )

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: Relaxed PES Surface Scan
────────────────────────────────────────────────────────────

Tool: run_scan_job
  Relaxed scan: ORCA optimises the geometry at each scan point (all degrees
  of freedom relaxed except the scanned coordinate) in a SINGLE ORCA job.
  Input: starting geometry (ideally pre-optimised).
  Returns: scan_results list, min_energy_eh, min_value.

Coordinate types (ORCA 0-based atom indices):
  B  atoms=[i, j]       → bond length (Å)
  A  atoms=[i, j, k]    → bond angle (degrees)
  D  atoms=[i, j, k, l] → dihedral angle (degrees)

scan_coords parameter (JSON string):
  '[{"type":"B", "atoms":[0,1], "start":0.8, "end":1.8, "n_points":11}]'
  n_points: total number of calculation points (inclusive of start and end).
  Typical: 10-20 for a 1D scan.

Atom indices are 0-based (first atom = 0). Check the XYZ geometry to identify
the correct atom numbers for the coordinate of interest.

Performance note:
  Relaxed scans run one geometry optimisation per point — keep n_points ≤ 13
  for quick jobs. Do NOT set ncores > 1 (MPI scan mode has known issues).

Plan pattern (2 nodes — NO LLM node):
  load → run_scan_job
  product: {"scan_results_<label>": "scan_results"}
  artifacts_to_save: ["scan_results_<label>"]

  Example for O-H bond scan of water:
    {"kind": "tool", "tool": "run_scan_job",
     "args": {"input_geom_id": "water",
              "scan_coords": "[{\\"type\\":\\"B\\",\\"atoms\\":[0,1],\\"start\\":0.8,\\"end\\":1.8,\\"n_points\\":11}]",
              "method": "B3LYP", "basis": "def2-SVP"},
     "product": {"scan_results_oh_stretch": "scan_results"}}

CRITICAL — no kind:"llm" node:
  scan_results is already structured JSON [{step, value, energy_eh}].
  QC-CALCULATOR cannot write tables or reports — never add an LLM node.
  The PES plot is generated automatically from the structured data.
""".strip()


class SpectrumSkill(PlannerSkill):
    """IR/Raman vibrational spectrum protocol."""

    name = "spectrum"
    priority = 22

    _KEYWORDS = ("spectrum", "ir spectrum", "infrared", "raman", "vibrational spectrum",
                 "absorption band", "ir band", "vibrational mode", "spectroscop")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: IR/Raman Spectrum Calculation
────────────────────────────────────────────────────────────

Tool: run_spectrum_job
  Extends run_freq_job: same E/H/G thermochemistry PLUS spectral data.

spectrum_type parameter:
  "ir"       → IR spectrum only (free; default)
  "raman"    → IR + Raman (adds %elprop Polar 1; roughly 2× cost)
  "ir_raman" → same as "raman"

Output fields:
  energy_eh, enthalpy_eh, gibbs_free_energy_eh  (identical to run_freq_job)
  ir_spectrum:    list of {mode, freq_cm1, intensity_km_mol}
  raman_spectrum: list of {mode, freq_cm1, activity, depolarization}
                  (only present when spectrum_type="raman" or "ir_raman")

IR intensity unit:  km/mol  (standard, proportional to peak area)
Raman activity unit: Å⁴/amu (relative; depolarization ratio 0–0.75)

Geometry requirement:
  Run run_spectrum_job on an already-optimised geometry.
  Running it on an unoptimised structure will produce meaningless frequencies
  (including imaginary modes). Standard workflow:
    run_opt_job → run_spectrum_job (input_id = output of opt)

Typical levels of theory:
  IR only : B3LYP/def2-SVP  (fast, qualitatively reliable)
  IR+Raman: B3LYP/def2-SVP  (same level; Raman adds polarizability CPSCF)
  Higher  : B3LYP/def2-TZVP for more quantitative frequencies

Frequency scaling:
  Computed harmonic frequencies are systematically too high.
  For B3LYP/def2-SVP: scale by ~0.97 for comparison with experiment.
  Report raw computed values; note scaling in the final_report.

Plan pattern for IR spectrum (3 nodes only — NO LLM node):
  load → run_opt_job (output_id: mol_opt) → run_spectrum_job(input_id: mol_opt, spectrum_type="ir")
  product: {"ir_spectrum_<species>": "ir_spectrum", "gibbs_free_energy_<species>_eh": "gibbs_free_energy_eh"}

CRITICAL — no kind:"llm" report node for spectrum:
  The ir_spectrum list is already fully structured JSON from run_spectrum_job.
  QC-CALCULATOR only handles numeric derivations (pKa, ΔG, etc.), NOT narrative text or tables.
  Adding a kind:"llm" report node will ALWAYS fail with status "error".
  Keep the plan to exactly 3 nodes: load, opt, spectrum.
  The final_report section captures the artifacts automatically via its fields list.

Artifact naming convention:
  ir_spectrum_<species>           — the ir_spectrum list artifact
  gibbs_free_energy_<species>_eh  — Gibbs free energy in Eh
  raman_spectrum_<species>        — Raman list artifact (if requested)

Do NOT put geometry strings in artifacts_to_save or product mappings.
""".strip()


class TDDFTSkill(PlannerSkill):
    """TD-DFT excited-state / UV-Vis absorption protocol."""

    name = "tddft"
    priority = 23  # between SpectrumSkill (22) and ThermochemistrySkill (25)

    _KEYWORDS = ("tddft", "td-dft", "excited state", "excitation energy",
                 "uv-vis", "uv/vis", "absorption spectrum", "electronic transition",
                 "oscillator strength", "charge transfer state", "singlet excited",
                 "s1 state", "vertical excitation", "optical gap",
                 "homo-lumo", "homo lumo", "frontier orbital", "band gap")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: TD-DFT Excited State / UV-Vis Absorption Calculation
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

Tool: run_tddft_job
  Single-point TD-DFT on an already-optimised geometry.
  Computes ground-state energy + vertical excitation energies for singlet excited states.

Parameters:
  n_states: number of excited states to compute (default 5; increase for wider coverage)

Output fields:
  energy_ground_state_eh: ground-state DFT energy (Eh)
  homo_lumo_gap_ev: KS orbital gap in eV (LUMO_ev - HOMO_ev from SCF; null if not parsed)
  excited_states: list of {state, energy_ev, wavelength_nm, oscillator_strength}
                  sorted by state index (ascending energy)

HOMO-LUMO gap:
  homo_lumo_gap_ev is the KS (Kohn-Sham) orbital energy gap — available from run_tddft_job
  AND from run_sp_energy (both parse ORBITAL ENERGIES block automatically).
  For the KS gap only (no excited states needed): use run_sp_energy, product homo_lumo_gap_ev.
  For the optical gap (S1 excitation energy): use excited_states[0].energy_ev from run_tddft_job.
  The KS gap underestimates the true gap; PBE0/def2-TZVP is more reliable than B3LYP.

UV/Vis conventions:
  oscillator_strength >> 0  -> bright (electric-dipole-allowed) transition
  oscillator_strength ~  0  -> dark (symmetry-forbidden) transition
  wavelength_nm is the vertical absorption peak for each excited state.

Geometry requirement:
  Always run on an optimised geometry (use run_opt_job first).
  Standard workflow:
    run_opt_job (output_id: mol_opt) -> run_tddft_job (input_id: mol_opt)

Recommended levels of theory:
  General UV/Vis:   B3LYP/def2-SVP    (fast, qualitatively correct for most organics)
  Better accuracy:  PBE0/def2-TZVP    (~2x cost, more quantitative excitation energies)
  Charge-transfer:  CAM-B3LYP/def2-TZVP (range-separated; needed for CT excited states)

Plan pattern (3 nodes -- NO LLM node):
  load -> run_opt_job (output_id: mol_opt) -> run_tddft_job (input_id: mol_opt, n_states=5)
  product: {"excited_states_<species>": "excited_states",
            "energy_<species>_ground_eh": "energy_ground_state_eh",
            "homo_lumo_gap_<species>_ev": "homo_lumo_gap_ev"}

CRITICAL -- no kind:"llm" report node:
  excited_states is already fully structured JSON. QC-CALCULATOR cannot write UV/Vis tables.
  Keep the plan to exactly 3 nodes: load, opt, tddft.
  Use final_report.fields: ["excited_states_<species>", "homo_lumo_gap_<species>_ev"]

Artifact naming convention:
  excited_states_<species>         -- the list of excited states
  energy_<species>_ground_eh       -- ground state DFT energy (Eh)
  homo_lumo_gap_<species>_ev       -- KS HOMO-LUMO gap in eV

For HOMO-LUMO gap only (no UV-Vis needed):
  Use run_sp_energy with product: {"homo_lumo_gap_<species>_ev": "homo_lumo_gap_ev"}
  Plan: load -> run_sp_energy  (2 nodes, fast)
""".strip()


class NBOSkill(PlannerSkill):
    """NBO analysis: what to request, what output to expect, charge transfer."""

    name = "nbo_analysis"
    priority = 40

    _KEYWORDS = ("nbo", "natural bond", "natural population", "npa", "charge transfer",
                 "donor acceptor", "hyperconjugat", "wiberg", "bond order")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: NBO Analysis Protocol
────────────────────────────────────────────────────────────

Tool: run_nbo_job
  Runs ORCA with NBO keyword; returns raw NBO section from output.

Recommended settings:
  method: B3LYP, basis: def2-SVP  (NBO is relatively basis-insensitive)
  Geometry should be pre-optimised before NBO (use run_opt_job first if needed).

What the output contains:
  - Natural Population Analysis (NPA): atomic charges and electron counts
  - Natural Bond Orbitals: σ, π, lone-pair occupancies
  - Second-order perturbation energies: donor–acceptor interactions (E2)
  - Wiberg Bond Indices: bond orders
  - NBO charges are more chemically meaningful than Mulliken charges

Plan pattern (opt → NBO):
  load → opt_job (output_id: mol_opt) → nbo_job (input_id: mol_opt)

Artifacts from run_nbo_job:
  The tool returns a text block (nbo_section). Store it as an artifact key
  (e.g. "nbo_output") for the final report. No numeric artifact is extracted
  automatically — the LLM report node summarises the text.

product field for NBO node:
  {"nbo_output": "nbo_section"}   ← maps artifact key to result field
""".strip()


# ---------------------------------------------------------------------------
# Coordination chemistry skill (priority 18 — before pKa/scan)
# ---------------------------------------------------------------------------

class CoordinationChemistrySkill(PlannerSkill):
    """Injected when the user asks about metal complexes or coordination compounds."""
    name     = "coordination_chemistry"
    priority = 18

    _KEYWORDS = {
        "complex", "coordination", "ligand", "metal", "octahedral", "tetrahedral",
        "square planar", "square_planar", "bipyridine", "bipy", "en ", "ethylenediamine",
        "ammonia complex", "transition metal", "cobalt", "iron", "ruthenium", "platinum",
        "copper", "nickel", "zinc", "chromium", "manganese", "palladium", "rhodium",
        "iridium", "molybdenum", "tungsten", "fe(", "co(", "ru(", "pt(", "cu(", "ni(",
    }

    def matches(self, user_text: str, state: dict) -> bool:
        t = user_text.lower()
        return any(kw in t for kw in self._KEYWORDS)

    def render(self, user_text: str, state: dict) -> str:
        return """
Coordination complex workflow
─────────────────────────────
Use build_coordination_complex (client-side) to generate the starting geometry, then
pipe it into run_opt_job via output_id/input_id.

build_coordination_complex parameters:
  metal        – element symbol: "Fe", "Ru", "Co", "Pt", "Cu", …
  ligands      – list of ligand names or SMILES (one entry per binding unit)
  geometry     – "linear" | "trigonal_planar" | "tetrahedral" | "square_planar"
               | "trigonal_bipyramidal" | "octahedral"
  charge       – total complex charge (integer)
  multiplicity – spin multiplicity 2S+1 (integer; use high spin for first-row TMs unless told otherwise)
  bond_length  – M–donor bond length in Å (optional, default 2.0)

Denticity rules:
  Monodentate ligands (NH3, H2O, Cl, CO, CN, SCN) → one entry per site.
  Bidentate ligands (en, bipyridine, acac, ox) → one entry per ligand; the tool
  auto-detects two donor atoms and fills two adjacent sites.
  Total denticity must equal coordination number:
    linear=2, trigonal_planar=3, tetrahedral/square_planar=4,
    trigonal_bipyramidal=5, octahedral=6.

Common bond lengths (Å):
  M–N  2.0 (first-row TM, e.g. Fe–NH3, Co–en)
  M–O  2.1 (e.g. Fe–H2O)
  M–Cl 2.3, M–P 2.3, M–C 1.9 (for CO/CN)
  Second-row TMs (Ru, Pd, Rh): add ~0.1 Å.

Multiplicity guidance:
  Fe(II) octahedral: 5 (high spin, d6 t2g4 eg2) or 1 (low spin, d6 t2g6) — depends on ligand field.
  Strong-field ligands (CO, CN, bipy) → low spin; weak-field (H2O, Cl, NH3) → high spin for Fe/Co.

Typical plan pattern:
  build_coordination_complex (output_id: complex_start)
    → run_opt_job            (input_id: complex_start, output_id: complex_opt)
    → run_sp_energy or run_freq_job

Geometry note: the template geometry is approximate. Always follow with run_opt_job.
""".strip()


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

SKILL_REGISTRY: List[PlannerSkill] = [
    MethodSelectionSkill(),
    CoordinationChemistrySkill(),
    ProtonationSiteSkill(),
    PKaSkill(),
    ScanSkill(),
    SpectrumSkill(),
    TDDFTSkill(),
    CasscfSkill(),
    ThermochemistrySkill(),
    TSSearchSkill(),
    SolvationSkill(),
    NBOSkill(),
]


def run_planning_skills(
    user_text: str,
    state: Dict[str, Any],
    registry: List[PlannerSkill] = SKILL_REGISTRY,
) -> List[str]:
    """Run all matching skills and return their rendered context strings.

    Skills are sorted by priority (ascending) before evaluation so that
    general context (MethodSelectionSkill) arrives before specific protocols.

    Returns a list of strings; each will be injected as a separate system
    message in handle_user_turn.
    """
    matched = [s for s in sorted(registry, key=lambda s: s.priority)
               if s.matches(user_text, state)]

    if matched:
        names = ", ".join(s.name for s in matched)
        print(f"[Skills activated: {names}]")
    else:
        print("[No domain skills matched]")

    return [s.render(user_text, state) for s in matched]
