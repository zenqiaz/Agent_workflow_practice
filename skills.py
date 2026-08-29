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
import functools
from typing import Any, Callable, Dict, List

from client_helpers import compute_pka_calibrated, rank_eas_sites, compute_spin_gap


class PlannerSkill:
    """Base class for planner skills."""
    name: str = ""
    priority: int = 50  # lower = injected earlier in message list

    # Deterministic tool functions this skill owns, e.g.
    # {"compute_pka_calibrated": compute_pka_calibrated}. Implementations may
    # live anywhere (client_helpers.py by convention, alongside every other
    # tool function) — declaring them here is what makes them discoverable and
    # auto-installable into the executor's tool registry via
    # collect_skill_client_tools(), instead of requiring a manual edit to
    # CLIENT_SIDE_TOOL_FUNCS in agent.py / test_success_rate.py for every skill
    # that needs its own calculation. Use this instead of kind:"llm" arithmetic
    # or ad-hoc global tool registrations whenever a skill needs to compute a
    # derived quantity deterministically.
    client_tools: Dict[str, Callable] = {}

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
  Dipole moment         : run_sp_energy (always returned, no extra cost)
  NBO charges           : run_sp_energy(properties=["nbo"]) — adds NBO keyword in same job
  HOMO-LUMO + UV-Vis    : run_sp_energy(properties=["tddft"]) or run_tddft_job standalone

Basis set guidance:
  def2-SVP   → cheap, opt/freq, qualitative
  def2-TZVP  → accurate single-points, properties
  def2-TZVPP → high-accuracy; use only when def2-TZVP is insufficient

Performance flags:
  use_ri=True  reduces cost of hybrid DFT significantly (RIJCOSX in ORCA)
  ncores: ALWAYS 1 — MPI is not available in this execution environment

Wall-time limits (always set wall_timeout_seconds for every calc node):
  run_sp_energy         :  600 s
  run_opt_job           : 1800 s  (3600 s for coordination complexes / large molecules)
  run_freq_job          : 1800 s
  run_scan_job          : 3600 s
  run_ts_opt_job        : 1800 s
  run_casscf_job        : 3600 s
  run_solvator_cluster_thermo: 3600 s

Geometry provenance — when to add run_opt_job:
  name_to_geometry_xyz        → PubChem/OPSIN 3D structure (reasonable quality).
                                 SKIP run_opt_job if task is SP-only or quick NBO/TDDFT scan.
                                 ADD  run_opt_job before freq / pKa / thermochemistry.
  build_coordination_complex  → force-field geometry. ALWAYS add run_opt_job before any QC job.
  structure_add_remove_proton → approximate geometry. ALWAYS add run_opt_job before SP/freq.
  geometry already in state (unknown source, loaded from file, user-provided XYZ)
                              → provenance unknown. ALWAYS add run_opt_job before any QC job.
""".strip()


class PKaSkill(PlannerSkill):
    """pKa calculation protocol, reference values, calibration, and benchmarks."""

    name = "pka_calibration"
    priority = 20

    # Owns compute_pka_calibrated: deterministic pKa arithmetic (plain and
    # reference-acid-calibrated, N >= 0 references) — replaces the kind:"llm"
    # arithmetic node this skill used to instruct the planner to emit, which
    # was found to silently produce arithmetic errors of >100 pKa units.
    client_tools = {"compute_pka_calibrated": compute_pka_calibrated}

    _KEYWORDS = ("pka", "pk_a", "acid dissociation", "acidity", "deprotonation",
                 "conjugate base", "ionization constant", "ka ")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: pKa Calculation Protocol
────────────────────────────────────────────────────────────

Solvation (READ FIRST — this is the most common planning error here):
  An aqueous pKa REQUIRES solvated Gibbs energies. Implicit solvation is
  requested by putting the solvation keyword INSIDE the method string:

      "method": "B3LYP CPCM(Water)"

  run_opt_job / run_freq_job / run_sp_energy have NO solvation parameter.
  Passing solvent, solvent_model, implicit_solvation_model, or similar as
  separate args is rejected by plan validation — and, worse, simply deleting
  those args leaves a gas-phase calculation silently mislabelled as aqueous.
  Move the keyword into "method"; do not drop it.
  A gas-phase deprotonation energy is ~350 kcal/mol; an aqueous one is
  ~6-25 kcal/mol. If ΔG comes out in the hundreds, solvation is missing.

Reference proton free energy — pick by whether you calibrate:
  CALIBRATED (references supplied): the proton term cancels exactly between
    target and references, so its value is irrelevant. Omit G_H_plus_ref_eh
    and let the tool choose.
  UNCALIBRATED (references: []): the proton term does NOT cancel and must be
    the full AQUEOUS proton free energy, -0.43074 Eh (Tissandier et al. 1998
    solvation, plus gas-phase G(H⁺) and the 1 atm→1 M correction). The
    gas-phase thermal value -0.01372 Eh is NOT valid here: it omits ~266
    kcal/mol of proton solvation and inflates pKa by ~195 units.

  PREFER CALIBRATION. Uncalibrated DFT pKa at these levels of theory carries
  errors of many pKa units even when set up correctly, because solvation-model
  error on the anion does not cancel. If the user names reference acids with
  known pKa, always use them.

Required artifacts:
  G_HA_eh        — Gibbs free energy of the neutral acid (Hartree)
  G_A_minus_eh   — Gibbs free energy of the conjugate base (Hartree)
  Both MUST come from run_freq_job nodes, NOT from run_sp_energy.
  (SP energies lack zero-point and thermal corrections → unacceptable for pKa.)

Formula (deterministic tool node — do NOT use kind:"llm" for this arithmetic;
verified that LLM-executed arithmetic on this exact formula can be off by
>100 pKa units, even though the underlying Gibbs energies are correct):

  Use ONE kind:"tool", tool:"compute_pka_calibrated" node per target compound:

  {
    "id": "compute_pka_{C}",
    "kind": "tool",
    "tool": "compute_pka_calibrated",
    "needs": ["freq_{C}_HA", "freq_{C}_A_minus"],
    "args": {
      "G_HA_eh": "$(artifacts.G_{C}_HA_eh)",
      "G_A_minus_eh": "$(artifacts.G_{C}_A_minus_eh)",
      "references": []
    },
    "product": {"pka_{C}": "pka"}
  }

  This computes the plain (uncalibrated) pKa: ΔG_eh = G_A_minus_eh +
  G_H_plus_ref_eh − G_HA_eh, converted to pKa via ΔG_J/(R·T·ln10). Leave
  "references" as an empty list for this case — the tool treats an empty list
  as "no calibration" and returns the raw formula pKa.

  Omit G_H_plus_ref_eh entirely, as shown. The tool then selects the correct
  proton reference for the call (aqueous when uncalibrated, thermal when
  calibrated). Only pass it explicitly if the user supplies a specific value;
  do NOT wire it to a thermal-only constant on an uncalibrated node.

Recommended level of theory:
  Gas phase    : B3LYP/def2-SVP  (opt + freq on same level)
  Higher acc.  : opt B3LYP/def2-SVP → SP B3LYP/def2-TZVP
  Aqueous pKa  : use run_solvator_cluster_thermo (nsolv ≥ 3) for HA and A⁻

Isodesmic / reference-acid calibration (REQUIRED for carbonyl alpha-H requests;
also usable for any pKa calibrated against known reference-acid pKa values):
  Gas-phase B3LYP/def2-SVP has a large systematic error vs. aqueous experiment
  (~200–250 pKa units too high) due to missing solvation. Cancel this error by
  calibrating against one or more reference acids with known experimental pKa
  (e.g. ethanal/acetaldehyde, pKa=17.0, for alpha-CH acidity):

  Plan must include opt+freq nodes for each reference acid's HA and A⁻ (same
  level as targets). Then populate the SAME compute_pka_{C} tool node's
  "references" list, one entry per reference acid, and add each reference's
  freq nodes to "needs":

  {
    "id": "compute_pka_{C}",
    "kind": "tool",
    "tool": "compute_pka_calibrated",
    "needs": ["freq_{C}_HA", "freq_{C}_A_minus",
              "freq_ethanal_HA", "freq_ethanal_A_minus"],
    "args": {
      "G_HA_eh": "$(artifacts.G_{C}_HA_eh)",
      "G_A_minus_eh": "$(artifacts.G_{C}_A_minus_eh)",
      "references": [
        {"G_HA_eh": "$(artifacts.G_ethanal_HA_eh)",
         "G_A_minus_eh": "$(artifacts.G_ethanal_A_minus_eh)",
         "pka_exp": 17.0}
      ]
    },
    "product": {"pka_{C}": "pka"}
  }

  This same pattern extends to N >= 2 reference acids (e.g. a multi-acid
  calibration such as fitting the proton solvation free energy against several
  carboxylic acids of known pKa) by adding more entries to "references" — no
  extra nodes needed; the tool averages the per-reference correction
  internally (epsilon_avg over all references).

  Do NOT put G_H_plus_ref_eh in plan settings or wire it into pKa nodes unless
  the user explicitly supplies a value. Omitting it lets the tool pick the
  reference that is correct for each call; a plan-level constant cannot, since
  the correct value differs between the calibrated and uncalibrated cases.

Deprotonation step — structure_add_remove_proton:
  Use  mode="remove"  with the correct  site_selector  (STRING, not a dict).
  DO NOT pass  geometry_xyz  in args — the executor injects it automatically via input_id.
  DO NOT use argument names 'site', 'site_hint', 'proton_site', or any other variant.
  The ONLY valid argument name is  site_selector.

  site_selector values for common pKa sites:
    "oxygen_hydroxyl"   → O-H bond (carboxylic acid, phenol, alcohol)
    "nitrogen"          → N-H bond (amine, amide)
    "alpha_carbon"      → alpha C-H bond (H on C adjacent to C=O; beta-ketoesters,
                          malonates, ketones, esters — any carbonyl alpha position)
    "line:N"            → Nth atom line in the XYZ (1-based; most explicit fallback)

  For carbonyl alpha-H acidity (pKa of C-H between two C=O groups):
    ALWAYS use  site_selector="alpha_carbon"
    NOT "alpha_to_carbonyl_C-H", NOT "alpha_H", NOT any other invented string.

xTB pre-optimization for anion geometry (REQUIRED for alpha-carbon deprotonation):
  After removing an alpha-C–H proton, the crude geometry retains a tetrahedral sp3
  carbon with a missing H — but the enolate is sp2 (planar). This large structural
  change means DFT optimization from the raw geometry is slow, often hits RESOURCE_LIMIT,
  and may fail to find the correct minimum.

  ALWAYS set  xtb_preopt: true  on the run_opt_job node for A⁻ whenever the deprotonation
  site is "alpha_carbon" (or any carbon site). This runs a cheap GFN2-xTB optimization
  first, relaxes the sp3→sp2 geometry, then passes the improved structure to ORCA.

  This is NOT required for O-H or N-H deprotonation (geometry change is minor).

  Example opt_A_minus node args:
    {"xtb_preopt": true, "wall_timeout_seconds": 1800}

Plan validation rules:
  ✓ artifacts_to_save MUST include G_HA_eh and G_A_minus_eh
  ✓ Both freq nodes must set  product: {"G_XX_eh": "gibbs_free_energy_eh"}
  ✓ compute_pka_{C} tool node must set  args: {"G_HA_eh": "$(artifacts...)",
    "G_A_minus_eh": "$(artifacts...)", "references": [...]}
    and  product: {"pka_{C}": "pka"}.  Omit G_H_plus_ref_eh — the tool picks
    the correct proton reference for calibrated vs uncalibrated calls.
  ✓ For an aqueous pKa, solvation goes in the method string
    ("method": "B3LYP CPCM(Water)"), never as a separate solvent argument
  ✓ structure_add_remove_proton node: use input_id (not geometry_xyz in args), site_selector only
  ✗ Do NOT use SP energy as proxy for G in a pKa plan
  ✗ Do NOT use kind:"llm" for the pKa calibration arithmetic — use the
    compute_pka_calibrated tool node above

Multi-compound pKa with calibration — use template mode (N >= 3 targets):
  Put all target compounds and ethanal in "compounds" list.
  Mark ethanal with role="reference".
  per_compound: load → deprot (alpha_carbon, xtb_preopt: true on A⁻ opt) → freq_HA → freq_A⁻
  post_template: one compute_pka_{C} kind:"tool" node per target
    (applies_to="targets_only"), tool:"compute_pka_calibrated", reading
    G_{C}_HA_eh, G_{C}_A_minus_eh via args as above, with "references" set to
    G_ethanal_HA_eh / G_ethanal_A_minus_eh (literal — not {C} — because
    ethanal is the fixed reference).

on_error patches for run_opt_job / run_freq_job:
  RESOURCE_LIMIT (geometry slow to converge, wall time exceeded):
    patch: {xtb_preopt: true, wall_timeout_seconds: 3600}   ← xTB pre-opt gets geometry near minimum
    NOT: {ncores: 1}  — ncores is already 1, this does nothing
  SCF_NOT_CONVERGED:
    patch: {scf_max_iter: 500}
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
    client_tools = {"compute_spin_gap": compute_spin_gap}

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
  Optional: nroots (default 1), basis (default def2-SVP), maxiter (default 300),
            avas_variant, scf_max_iter (default 200)
  Returns: energy_eh (total or SA-average), energies_eh list (only when nroots > 1)

  compute_spin_gap  (kind:"tool", NOT kind:"llm" -- this is arithmetic, not judgement)
  Required: E_hs_eh, E_ls_eh (the two run_casscf_job energy_eh artifacts)
  Returns: delta_E_kcal, delta_E_eh, ground_state ("high-spin"/"low-spin"/"degenerate")
  Subtracting two energies and converting Hartree to kcal/mol has no chemistry
  judgement in it -- do not route this through an llm node.

AVAS (strongly recommended for all transition metal and pi systems)
  ORCA 6 uses avas_variant as a simple keyword — no separate block, no orbital strings.
  AVAS rotates the best starting orbitals into the active space before CASSCF,
  dramatically reducing macro-iterations and avoiding saddle-point traps.

  avas_variant options:
    "VALENCE-D"  — valence d orbitals of all transition metals in the molecule (most common)
    "DOUBLE-D"   — 3d + 4d shells (for heavy 4d metals or double-shell CASSCF)
    "VALENCE-DS" — d + s valence
    "DOUBLE-DS"  — double d + s
    "VALENCE-F"  — valence f orbitals (lanthanides/actinides)
    "DOUBLE-F"   — double f shell

  For most d-block metal complexes: use "VALENCE-D".
  Omit avas_variant for simple non-metal systems (e.g. bond dissociation CAS(2,2)).

Active space selection guide
  - Fe(III) d5:               CAS(5,5),  avas_variant: "VALENCE-D"
  - Fe(II)  d6:               CAS(6,5),  avas_variant: "VALENCE-D"
  - Ni(II)  d8:               CAS(8,5),  avas_variant: "VALENCE-D"
  - Simple bond dissociation: CAS(2,2),  no AVAS needed
  - Benzene pi (non-metal):   CAS(6,6),  no AVAS (AVAS variants target metals)
  If unsure, use the minimal valence active space.

Convergence
  Use maxiter as the primary convergence knob; default 300 is generous.
  Do not increase maxiter in retries — change the orbital or basis strategy instead.

Spin state comparison
  Prefer separate single-state CASSCF jobs (nroots=1) for each spin state.
  Use SA-CASSCF (nroots > 1) only when states are strongly mixed.

Basis sets
  - def2-SVP: good starting point; affordable
  - def2-TZVP: better accuracy for final results
  - ANO-RCC or def2-TZVPP: high-accuracy metal active spaces

Typical plan pattern (Fe spin-state gap with AVAS)
  CRITICAL: run_casscf_job must publish energy_eh under a node-specific artifact
  key via product: -- a bare node-id reference (casscf_hs.energy_eh) is only
  valid inside an llm node's needs_artifacts list, NOT inside a tool node's
  args, which only expand $(artifacts.KEY) and $(settings.KEY).
  nodes:
    - {id: casscf_hs, kind: tool, tool: run_casscf_job,
       input_id: mol_hs, args: {nel: 5, norb: 5, avas_variant: "VALENCE-D"},
       product: {energy_eh_hs: energy_eh}}
    - {id: casscf_ls, kind: tool, tool: run_casscf_job,
       input_id: mol_ls, args: {nel: 5, norb: 5, avas_variant: "VALENCE-D"},
       product: {energy_eh_ls: energy_eh}}
    - {id: calc_gap, kind: tool, tool: compute_spin_gap,
       needs: [casscf_hs, casscf_ls],
       args: {E_hs_eh: "$(artifacts.energy_eh_hs)", E_ls_eh: "$(artifacts.energy_eh_ls)"},
       product: {delta_E_kcal: delta_E_kcal, ground_state: ground_state}}

No geometry optimization with CASSCF
  run_casscf_job is SP-only. Optimize with DFT first (run_opt_job), then run CASSCF.

on_error cascade (include ALL rules in every CASSCF node)
  Design principle: cheaper changes first. Each retry must change the strategy —
  never retry with only more iterations of the same approach.

  on_error:
    # Stage 1: HF pre-SCF failed → more SCF iterations (cheap, orthogonal problem)
    - if: {code_in: [SCF_NOT_CONVERGED]}
      patch: {scf_max_iter: 500}
      max_attempts: 1

    # Stage 2: CASSCF not converged → upgrade basis (reshapes virtual space,
    #   resolves near-degeneracies def2-SVP cannot represent)
    - if: {code_in: [CASSCF_NOT_CONVERGED]}
      patch: {basis: def2-TZVP}
      max_attempts: 1

    # Stage 3: Timeout → extend wall time (safety net only)
    - if: {code_in: [RESOURCE_LIMIT]}
      patch: {wall_timeout_seconds: 7200}
      max_attempts: 1

  If Stage 2 fails: the active space choice is wrong, not the numerics.
  Manual intervention needed — wrong nel/norb, wrong avas_variant, or the system
  requires a different approach (NEVPT2, DMRG, or larger CAS).
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

Two solvator tools exist. Pick by what the downstream node needs, and pass only
the arguments the chosen tool declares — they are NOT interchangeable.

Tool: run_solvator_cluster_thermo   (cluster + thermochemistry)
  → Builds water cluster with SOLVATOR, then runs ORCA freq in one call.
  → Returns: cluster_xyz, energy_eh, enthalpy_eh, gibbs_free_energy_eh
  → Accepts: geometry_xyz, nsolv, charge, multiplicity, method, basis, use_ri,
             ncores, scf_max_iter, thermo_engine, thermo_timeout_seconds,
             wall_timeout_seconds, job_label
  → Use when a Gibbs free energy or enthalpy of the cluster is required
    (pKa, ΔG, any thermochemistry). Prefer it over run_solvator_cluster +
    run_freq_job in that case: one call instead of two.

Tool: run_solvator_cluster          (cluster geometry only)
  → Builds the water cluster and stops. No frequency job, no thermochemistry.
  → Returns: cluster_geometry_xyz, energy_eh
  → Accepts ONLY: geometry_xyz, nsolv, charge, multiplicity,
             wall_timeout_seconds, job_label
  → Does NOT accept method, basis, use_ri, ncores, scf_max_iter, or any
    thermo_* argument. Passing one is rejected by plan validation.
  → Use when only the solvated geometry is needed downstream — for example
    a TD-DFT/UV-Vis spectrum on the cluster, or a single-point energy. Running
    the thermo variant there buys a frequency calculation nothing consumes.

If the request names one of these tools explicitly, use the one it names.

nsolv selection guide:
  nsolv = 1–2  : qualitative, fast, suitable for screening
  nsolv = 3    : standard protocol for pKa and ΔG in polar environments
  nsolv = 4–6  : for molecules with multiple H-bond donors/acceptors
  nsolv ≥ 7    : large polar molecules; expect >30 min compute time

Workflow pattern for solvated pKa:
  load HA → solvator_thermo(HA, nsolv=3) → G_HA_cluster_eh
          → remove_proton → solvator_thermo(A⁻, nsolv=3) → G_A_cluster_eh
          → kind:"tool" node, tool: compute_pka_calibrated
  Do the pKa arithmetic in that tool node, NOT in a kind:"llm" node. A language
  model asked to evaluate this expression does not reproduce its own answer
  across runs even when the input energies are byte-identical.

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


class InteractionScanSkill(PlannerSkill):
    """Intermolecular interaction-energy scan: monomer SP energies + dimer PES."""

    name = "interaction_scan"
    priority = 19   # just before ScanSkill (21)

    _KEYWORDS = (
        "interaction scan", "interaction energy", "binding energy scan",
        "approach curve", "dissociation curve", "intermolecular scan",
        "pi stacking", "π stacking", "cation pi", "cation-pi", "cation π",
        "electrophilic aromatic", "wheland", "adduct scan",
        "dimer scan", "complex scan", "host guest scan",
    )

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        geom_info = ""
        geoms = state.get("geometries", {})
        if geoms:
            sizes = {k: len([l for l in v.splitlines() if l.strip()]) for k, v in geoms.items()}
            geom_info = "\nLoaded geometries: " + ", ".join(
                f"{k}({n}atoms)" for k, n in sizes.items()
            )
        return ("""SKILL: Intermolecular Interaction-Energy Scan
────────────────────────────────────────────────────────────

Goal: compute ΔE_int(d) = E(dimer, d) − E(A) − E(B) as a function of
intermolecular distance d. This gives a proper dissociation curve with
the zero at infinite separation (two isolated monomers).

═══ Choose one of two workflows ═══

──── Workflow A: Rigid-body approach scan (PREFERRED for most cases) ────

  Translates mol_b as a rigid fragment toward mol_a along the centroid-
  to-centroid vector. No ORCA constraints, no atom index arithmetic.
  Works for any fragment size and orientation.

  Plan structure (4 + N nodes):

  [parallel] sp_a  — run_sp_energy(input_id: mol_a) → E_A_eh
  [parallel] sp_b  — run_sp_energy(input_id: mol_b) → E_B_eh

  build_scan  — build_approach_scan_geometries(
      mol_a_id: "mol_a",
      mol_b_id: "mol_b",          ← starting position already set in state
      n_steps: <N>,               ← e.g. 11
      step_ang: <Δ>,              ← e.g. 0.1 Å
      output_prefix: "approach"   ← geom_ids will be approach_00..approach_10
  )

  [parallel, N nodes]
  sp_approach_00  — run_sp_energy(input_id: "approach_00") → E_approach_00_eh
  sp_approach_01  — run_sp_energy(input_id: "approach_01") → E_approach_01_eh
  ...
  sp_approach_NN  — run_sp_energy(input_id: "approach_NN") → E_approach_NN_eh

  CRITICAL: geom_ids are deterministic — {output_prefix}_{00..n_steps-1}.
  Write them explicitly in the plan at plan time.

  CRITICAL: use EXACTLY ONE llm node for the entire curve — do NOT add one llm
  node per step. One node receives all monomer + approach energies and returns
  the full interaction_curve list.

  compute_delta_e  — kind:"llm"  (ONE node for ALL steps)
      needs_artifacts: ["E_A_eh", "E_B_eh",
                        "E_approach_00_eh", "E_approach_01_eh", ..., "E_approach_NN_eh",
                        "start_distance_ang", "step_ang"]
      prompt: "Given E_A_eh, E_B_eh, start_distance_ang, step_ang, and
               E_approach_00_eh through E_approach_NN_eh, compute for each step i:
               distance_ang = start_distance_ang - step_ang * i,
               delta_e_int_kcal = (E_approach_i_eh - E_A_eh - E_B_eh) * 627.509.
               Return JSON: {status,
               interaction_curve: [{step, distance_ang, delta_e_int_kcal}],
               d_eq_ang: distance at minimum delta_e_int_kcal,
               binding_energy_kcal: minimum delta_e_int_kcal}"
      product: {"interaction_curve": "interaction_curve",
                "d_eq_ang": "d_eq_ang",
                "binding_energy_kcal": "binding_energy_kcal"}

──── Workflow B: ORCA relaxed scan (use only for small rigid systems ≤ 20 atoms) ────

  Constrains ONE internal coordinate (the approach bond) and fully
  relaxes everything else at each step. Gives correct TS geometries
  but requires atom index arithmetic and ncores=1.

  build_dimer  — build_dimer_xyz(
      geom_a_id: "mol_a", geom_b_id: "mol_b",
      distance_ang: $(settings.d_start),
      ref_atom_b: $(settings.ref_atom_b),
      axis: "z", output_id: "dimer"
  )

  CRITICAL — scan_atom_b must be a LITERAL INTEGER:
    scan_atom_b = n_atoms_a + ref_atom_b
    Read n_atoms_a from STATE (the "n_atoms=N" field). Write the literal integer.

  scan_approach  — run_scan_job(
      input_id: "dimer",
      scan_coords: '[{"type":"B","atoms":[0,<scan_atom_b>],"start":<d_start>,"end":<d_end>,"n_points":<N>}]'
  ) → scan_results_dimer

  compute_delta_e  — kind:"llm"
      needs_artifacts: ["scan_results_dimer", "E_A_eh", "E_B_eh"]
      prompt: "For each point in scan_results_dimer compute
               delta_e_int_kcal = (energy_eh - E_A_eh - E_B_eh) * 627.509.
               Return JSON: {status, interaction_curve, d_eq_ang, binding_energy_kcal}"
      product: {"interaction_curve": "interaction_curve", ...}

═──── Workflow C: Parallel constrained-opt scan (bond-forming with relaxation) ────

  Use when: 1.4–2.0 Å bond-forming region AND geometry relaxation matters
  (e.g. EAS ipso puckering, SN2 backside attack, O-N-O angle narrowing).
  Faster than Workflow B (parallel instead of sequential ORCA).

  Step 1 — generate dimer geometries at each approach distance (same as Workflow A):
    build_scan — build_approach_scan_geometries(
        mol_a_id, mol_b_id, n_steps, step_ang, ref_atom_a, output_prefix="approach"
    )
    → geom_ids: approach_00 .. approach_{n_steps-1}
    → mol_a atoms are indices 0 .. N_a-1 in each dimer
    → mol_b atoms are indices N_a .. N_a+N_b-1 in each dimer

  Step 2 — parallel constrained opts (ONE node per scan point):
    opt_approach_00 — run_opt_job(
        input_geom_id: "approach_00",
        constraints: [{"type": "B", "atoms": [<ref_atom_a>, <ref_atom_b_in_dimer>]}]
        # omit "value" → ORCA fixes bond at current distance in that geometry
    ) → E_opt_approach_00_eh

  CRITICAL: ref_atom_b_in_dimer = N_a + ref_atom_b_in_mol_b
    Read N_a from STATE (the "n_atoms=N" field for mol_a). Write the literal integer.
    Example: mol_a=benzene (N_a=12), mol_b=NO2+ (N=atom 2) → ref_atom_b_in_dimer=14

  Step 3 — monomer SPs (parallel, unconstrained):
    sp_a — run_sp_energy(input_geom_id: mol_a) → E_A_eh
    sp_b — run_sp_energy(input_geom_id: mol_b) → E_B_eh

  Step 4 — ONE llm node for full curve (same as Workflow A).

═══ Scan range guidance ═══

  Interaction type               | d_start (Å) | d_end (Å) | n_steps | step_ang
  ────────────────────────────── | ----------- | --------- | ------- | --------
  H-bond (O-H···O/N)             |   3.5       |   1.5     |   11    |  0.2
  Ion–π approach (coarse)        |   5.0       |   2.4     |   11    |  0.26
  π–π stacking                   |   6.0       |   3.0     |   11    |  0.3
  van der Waals (noble gas)      |   6.0       |   3.0     |    9    |  0.375
  Bond-forming region (C-C/C-N/C-O, fine scan) | 2.0 | 1.4 | 13 | 0.05

  Bond-forming rule: C-C, C-N, C-O single bonds form at 1.47–1.54 Å.
  For EAS (Wheland intermediate), SN2 at carbonyl, Michael addition:
    • Coarse pass: d_start=2.5 → d_end=1.4, n_steps=12, step_ang=0.1
      (maps full approach curve, finds the repulsive wall onset)
    • Fine pass: d_start=2.0 → d_end=1.4, n_steps=13, step_ang=0.05
      (resolves the TS region where bond actually forms)
  Use the fine scan range when the question is specifically about the
  transition state geometry or activation barrier.

═══ Method / basis ═══

  B3LYP/def2-SVP: qualitative curves.
  PBE0/def2-TZVP: quantitative binding energies.

CRITICAL:
  - The kind:"llm" compute_delta_e node is the ONLY LLM node allowed.
  - Do NOT add a kind:"llm" node for report generation.
  - scan_results_dimer / individual E_approach energies are raw E(dimer) —
    ΔE_int requires monomer subtraction, so the LLM node IS necessary.
""" + geom_info).strip()


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
                 "oscillator strength", "charge transfer state",
                 "singlet excited", "triplet excited",
                 "s1 state", "t1 state", "s1 energy", "t1 energy",
                 "s0->s1", "s0-s1", "vertical excitation", "optical gap",
                 "uv absorption", "photon absorption", "photophysic",
                 "fluorescen", "phosphorescen")

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: TD-DFT Excited State / UV-Vis Absorption Calculation
\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

run_sp_energy with properties (preferred — single ORCA job):
  run_sp_energy always returns: energy_eh, dipole_moment_debye, homo_lumo_gap_ev
  Add properties=["tddft"] to also get excited_states in the same job.
  Add properties=["nbo"]   to also get nbo_section in the same job.
  Combine freely: properties=["tddft","nbo"] for all in one run.

Tool: run_tddft_job (standalone alternative)
  Use when you only need excited states and no other SP properties,
  or for a clean 3-node plan after run_opt_job.

Parameters:
  n_tddft_states (run_sp_energy) / nroots (run_tddft_job): number of roots (default 5)

Output fields:
  energy_eh / energy_ground_state_eh: ground-state DFT energy (Eh)
  dipole_moment_debye: dipole moment magnitude in Debye (always from run_sp_energy)
  homo_lumo_gap_ev: KS orbital gap in eV (LUMO_ev - HOMO_ev from SCF)
  excited_states: list of {state, energy_ev, wavelength_nm, oscillator_strength}

HOMO-LUMO gap:
  KS gap only (2-node plan, fast): run_sp_energy → homo_lumo_gap_ev
  Optical gap (S1): excited_states[0].energy_ev (from tddft)
  The KS gap underestimates the true gap; PBE0/def2-TZVP is more reliable than B3LYP.

UV/Vis conventions:
  oscillator_strength >> 0  -> bright (electric-dipole-allowed) transition
  oscillator_strength ~  0  -> dark (symmetry-forbidden) transition
  wavelength_nm is the vertical absorption peak for each excited state.

Geometry requirement:
  Always run on an optimised geometry (use run_opt_job first).

Recommended levels of theory:
  General UV/Vis:   B3LYP/def2-SVP    (fast, qualitatively correct for most organics)
  Better accuracy:  PBE0/def2-TZVP    (~2x cost, more quantitative excitation energies)
  Charge-transfer:  CAM-B3LYP/def2-TZVP (range-separated; needed for CT excited states)

Plan patterns (NO LLM node):

  UV-Vis + dipole + HOMO-LUMO in one job (preferred):
    load -> run_opt_job (output_id: mol_opt)
         -> run_sp_energy(input_id: mol_opt, properties=["tddft"], n_tddft_states=5)
    product: {"excited_states_<mol>": "excited_states",
              "homo_lumo_gap_<mol>_ev": "homo_lumo_gap_ev",
              "dipole_<mol>_debye": "dipole_moment_debye"}

  Standalone run_tddft_job:
    load -> run_opt_job (output_id: mol_opt) -> run_tddft_job(input_id: mol_opt, nroots=5)
    product: {"excited_states_<mol>": "excited_states",
              "homo_lumo_gap_<mol>_ev": "homo_lumo_gap_ev"}

Multi-compound comparison (e.g. "which compound absorbs best at 500 nm"):
  Collect excited_states for each compound — NEVER add a kind:"llm" comparison node.
  The reporter already has all excited_states artifacts and answers the comparison directly.
  Plan: for each compound: load -> run_opt_job -> run_tddft_job (or run_sp_energy with properties=["tddft"])
  artifacts_to_save: ["excited_states_mol1", "excited_states_mol2", ...]
  final_report.fields: ["excited_states_mol1", "excited_states_mol2", ...]

CRITICAL -- NEVER add any kind:"llm" node for TDDFT tasks:
  excited_states is already fully structured JSON. QC-CALCULATOR cannot write tables or comparisons.
  DO NOT add a "compare_absorption" or any ranking/comparison kind:"llm" node — the reporter does this.
  The ONLY valid kind:"llm" node is for computing scalar numbers (pKa, ΔG) not present in tool output.

Artifact naming convention:
  excited_states_<mol>      -- list of excited states
  homo_lumo_gap_<mol>_ev    -- KS HOMO-LUMO gap in eV
  dipole_<mol>_debye        -- dipole moment in Debye
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

Two ways to run NBO:

Option A — run_sp_energy with properties=["nbo"] (preferred, fewer nodes):
  load → run_opt_job (output_id: mol_opt)
       → run_sp_energy(input_id: mol_opt, properties=["nbo"])
  Returns: energy_eh, dipole_moment_debye, homo_lumo_gap_ev, nbo_section
  product: {"nbo_output": "nbo_section", "homo_lumo_gap_mol_ev": "homo_lumo_gap_ev"}

Option B — run_nbo_job (standalone):
  load → run_opt_job (output_id: mol_opt) → run_nbo_job(input_id: mol_opt)
  Returns: nbo_section only.
  product: {"nbo_output": "nbo_section"}

Recommended settings:
  method: B3LYP, basis: def2-SVP  (NBO is relatively basis-insensitive)
  Geometry should be pre-optimised before NBO (use run_opt_job first).

What the output contains:
  - Natural Population Analysis (NPA): atomic charges and electron counts
  - Natural Bond Orbitals: σ, π, lone-pair occupancies
  - Second-order perturbation energies: donor–acceptor interactions (E2)
  - Wiberg Bond Indices: bond orders
  - NBO charges are more chemically meaningful than Mulliken charges

Artifacts:
  nbo_section is a raw text block. Store as artifact key (e.g. "nbo_output").
  No numeric artifact is extracted automatically — the LLM report node summarises the text.
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
        "square planar", "square_planar", "bipyridine", "bipy", "ethylenediamine",
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
Use build_coordination_complex (server-side MCP tool, uses molSimplify) to generate the
starting geometry, then pipe into run_opt_job via output_id/input_id.

build_coordination_complex parameters:
  metal           – element symbol lowercase: "fe", "co", "ru", "cu", "pt", …
  ligands         – list of molSimplify ligand names.
                    Common names: "cl" (chloride), "water", "nh3", "co" (carbonyl),
                    "cn" (cyanide), "en" (ethylenediamine), "bipy" (bipyridine),
                    "acac" (acetylacetonate), "acetate", "ox" (oxalate), "ncs" (thiocyanate).
                    Use ONE ENTRY PER LIGAND MOLECULE (not per donor atom).
                    Coordination number is derived automatically from denticity:
                      Monodentate (cl, water, nh3, cn, co, ncs, acetate): 1 site each
                      Bidentate   (en, acac, bipy, phen, ox):             2 sites each
                    Examples:
                      [Fe(CN)₆]³⁻  → ligands=["cn","cn","cn","cn","cn","cn"], geometry="oct"
                      [Cu(en)₂]²⁺  → ligands=["en","en"],  geometry="sqp"   (2 × bidentate = coord 4)
                      Ni(acac)₂    → ligands=["acac","acac"], geometry="sqp" (2 × bidentate = coord 4)
                      cis-Pt(NH₃)₂Cl₂ → ligands=["nh3","nh3","cl","cl"], geometry="sqp"
                    WARNING: "dmg" (dimethylglyoximate) is NOT in molSimplify. Use acac as a substitute.
  geometry        – "oct" (octahedral, 6) | "thd" (tetrahedral, 4) |
                    "sqp" (square planar, 4) | "tbp" (trigonal bipyramidal, 5)
  oxidation_state – Roman numeral string: "II", "III", "IV", etc.
  spin            – spin multiplicity 2S+1 (integer)
  charge          – total complex charge (integer)
  force_field     – "uff" (default) | "mmff94" | "n" (skip FF)

CRITICAL: do NOT pass bond_length or coordination_number — not valid parameters.
  Coordination number is derived automatically from ligand denticities.

CRITICAL: ALWAYS set both spin and charge explicitly in every build_coordination_complex call.
  Never omit them — missing args cause wrong defaults and silent errors.

Spin (spin multiplicity 2S+1) — ALWAYS provide explicitly:
  d10 metals  Cu(I), Zn(II), Ag(I), Au(I)            → spin=1
  d9  metals  Cu(II)                                  → spin=2
  d8  sqp     Ni(II), Pd(II), Pt(II) square planar    → spin=1
  d8  thd     Ni(II), Pd(II), Pt(II) tetrahedral      → spin=3  (rare)
  d6  LS      Ru(II), Ir(III), Rh(III), Fe(II)+CN/CO  → spin=1
  d5  LS      Fe(III) + strong field (CN,CO,en,bipy)   → spin=2
  d5  HS      Fe(III) + weak field   (Cl,H2O,F,NCS)   → spin=6
  d6  HS      Fe(II)  + weak field                    → spin=5
  d4  LS      Co(III) + strong field                  → spin=1
  d4  HS      Co(III) + weak field                    → spin=5
  d3          Cr(III) oct                             → spin=4  (always HS)

  Strong-field ligands (low spin): CN⁻, CO, bipy, en, NH₃, NO₂⁻
  Weak-field  ligands (high spin): Cl⁻, F⁻, H₂O, NCS⁻, OAc⁻

charge — ALWAYS provide explicitly:
  Total complex charge = metal oxidation state + sum of ligand charges.
  Example: Fe(III) + 6 CN⁻ → charge = +3 + 6×(−1) = −3

Typical plan pattern:
  build_coordination_complex (output_id: complex_start)
    → run_opt_job            (input_id: complex_start, output_id: complex_opt)
    → run_freq_job or run_sp_energy

Geometry note: molSimplify uses UFF force-field pre-optimization. The result is a
reasonable starting geometry; always follow with run_opt_job for accurate structure.

CRITICAL run_opt_job settings for coordination complexes:
  use_ri=True               – REQUIRED for metal complexes; speeds SCF by 3-5x
  ncores=1                  – MPI is unavailable in this environment; always ncores=1
  wall_timeout_seconds=3600 – coordination complexes often need > 30 min to optimize
""".strip()


class GeometrySkill(PlannerSkill):
    """Teaches the planner how to construct and inject custom molecular geometries."""

    name = "geometry"
    priority = 18   # just before InteractionScanSkill (19)

    _KEYWORDS = (
        "above", "tilt", "tilted", "angle", "degree", "orient", "orientation",
        "approach", "axial", "equatorial", "perpendicular", "parallel",
        "set_geometry", "custom geometry", "write xyz", "starting geometry",
        "sigma complex", "sigma-complex", "wheland", "eas", "electrophilic aromatic",
        "above the ring", "above the plane", "off centre", "off-centre", "off center",
        "directly above", "carbon atom", "target atom",
        "dimer geometry", "interaction scan", "cation pi", "cation-pi",
    )

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: Molecular Geometry Construction
────────────────────────────────────────────────────────────

═══ Reading atom indices from the STATE block ═══

The STATE block shows each loaded geometry with element ranges, e.g.:
  - benzene: q=0, mult=1  n_atoms=12  atoms: C[0-5] H[6-11]
  - no2plus: q=1, mult=1  n_atoms=3   atoms: O[0-1] N[2]

Use these indices directly for scan_coords, target_atom_a, ref_atom_b, etc.
Do NOT guess — always read from the STATE block.

═══ Coordinate conventions ═══

- Geometries stored WITHOUT natoms/comment header
- One line per atom: El  x  y  z
- PubChem orientations (after centering at origin):
    Benzene C6H6:    ring in XY plane, centroid at (0,0,0)
                     C[0-5] roughly at radius 1.40 Å in XY, H[6-11] at 2.48 Å
    NO2+ (linear):   O-N-O along X axis, N at origin
                     O[0] at (-1.336, 0, 0), O[1] at (1.336, 0, 0), N[2] at (0,0,0)
    Water H2O:       O at origin, H-O-H in XY plane
    CO2 (linear):    O-C-O along X axis

═══ Tool choice ═══

  build_dimer_xyz  — use when:
    • Approach is straight along Z (or X/Y) above A's centroid
    • No tilt, no specific target atom on A needed
    • Perpendicular π-approach, symmetric van der Waals scan

  set_geometry_xyz  — use when:
    • Approach is tilted (θ > 0° from ring normal)
    • Target is a specific atom of A (not the centroid)
    • EAS / σ-complex starting geometry (N directly above a C)
    • Any custom dimer geometry the planner computes explicitly

═══ Computing a tilted or off-centre approach geometry ═══

Step 1 — Center A at origin (subtract centroid from all coordinates).
Step 2 — Identify target atom position t = (tx, ty, tz) in A.
Step 3 — Choose approach vector v at angle θ from Z, tilted toward t:
           v_horiz = normalize(tx, ty, 0)   # horizontal component toward t
           v = sin(θ)·v_horiz + cos(θ)·ẑ   # unit approach vector
Step 4 — Place ref_atom_b of B at:  p = t + d·v
Step 5 — Shift all atoms of B so ref_atom_b lands at p.
Step 6 — Write combined A+B coordinates into set_geometry_xyz.

Example — NO2+ N directly above C0 of benzene at d=2.4 Å (θ=0°, EAS):
  C0 of benzene (after centering) ≈ (0.000, 1.396, 0.000)
  v = ẑ = (0, 0, 1)
  N position = (0.000, 1.396, 2.400)
  O atoms: N ± (1.336, 0, 0) → (-1.336, 1.396, 2.400) and (1.336, 1.396, 2.400)
  → scan_coords atoms: [0, scan_atom_b]  where 0 = C0 index, scan_atom_b = 12+2 = 14

Example — NO2+ at 45° above C0 (tilted EAS approach):
  v_horiz = normalize(0, 1.396, 0) = (0, 1, 0)
  v = sin(45°)·(0,1,0) + cos(45°)·(0,0,1) = (0, 0.707, 0.707)
  N position = (0, 1.396, 0) + d·(0, 0.707, 0.707)
  At d=3.0 Å: N = (0, 1.396+2.121, 2.121) = (0, 3.517, 2.121)

═══ scan_atom_a for off-centre scans ═══

When the reference atom on A is NOT atom 0, set:
  scan_atom_a = index of target atom in A  (read from STATE atoms: block)
  scan_atom_b = n_atoms_a + ref_atom_b_local_index

scan_coords: '[{"type":"B","atoms":[$(settings.scan_atom_a),$(settings.scan_atom_b)],...}]'
Put both as LITERAL INTEGERS in settings.
"""


# ---------------------------------------------------------------------------
# EAS reactivity skill (priority 33 — between Solvation=30 and NBO=40)
# ---------------------------------------------------------------------------

class EASSkill(PlannerSkill):
    """EAS site reactivity via NPA charges and Fukui f⁻ indices."""

    name     = "eas_reactivity"
    priority = 33

    # Owns rank_eas_sites: deterministic filter-and-sort over the Mulliken charge
    # table — replaces the kind:"llm" node this skill used to instruct the
    # planner to emit, whose whole task was keeping the carbons and sorting by
    # charge. The chemistry is in the element predicate, which is fixed; the
    # sorting never needed a language model.
    client_tools = {"rank_eas_sites": rank_eas_sites}

    _KEYWORDS = (
        "eas", "electrophilic aromatic", "aromatic substitution",
        "site selectivity", "site reactivity", "activated site", "deactivated site",
        "ortho para", "o/p director", "meta director",
        "fukui", "fukui function",
        "charge analysis", "npa charge", "natural charge",
        "electron density", "charge distribution", "aromatic reactivity",
        "most reactive site", "least reactive site", "preferred site",
    )

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: EAS Reactivity Analysis Protocol
────────────────────────────────────────────────────────────

Goal: rank aromatic carbon sites by susceptibility to electrophilic attack.

IMPORTANT: NBO/NPA analysis is NOT available on this server (NBOEXE not set).
Use Mulliken charges from run_sp_energy (always available, no extra keyword needed).

══ Method: Mulliken charges (1 SP per molecule, NO properties=["nbo"]) ═══

  Step 1  opt → run_opt_job  (output_id: mol_opt)
  Step 2  run_sp_energy(input_id: mol_opt)  ← NO properties needed
          product: {"mulliken_mol": "mulliken_charges"}
  Step 3  kind:"tool" node, tool: rank_eas_sites
          args: {"mulliken_charges": "$(artifacts.mulliken_<mol>)"}
          Returns site_ranking (carbons and heteroatoms, sorted most-negative
          first), most_activated, n_sites.
          Do NOT emit a kind:"llm" node for this step. Filtering a charge table
          by element and sorting it is not a reasoning task, and a language model
          asked to do it costs ~1,500 tokens per molecule without reproducing its
          own output across runs.

  Chemistry: EAS preferentially attacks the most electron-rich (most negative) carbon.
  Mulliken charges are less absolute than NPA but correctly rank relative site reactivity
  within a molecule and qualitatively across similar molecules.

══ Multi-molecule comparison ═══════════════════════════════════════════════

  Run Method above for each molecule (separate opt → SP chains).
  Do NOT add a final node to compare molecules. Every eas_sites_<mol> artifact
  is already visible to the reporter once the plan finishes executing, and
  building the cross-molecule table from them -- take the top-ranked site per
  molecule, order the molecules by that site's charge -- is a report-writing
  task, not a plan step: it needs no artifact no earlier node has already
  produced, and no computation the reporter cannot do directly from what it is
  already given. Stop the plan at the last rank_eas_sites node.

══ Recommended settings ════════════════════════════════════════════════════

  method: B3LYP, basis: def2-SVP, use_ri: True
  ncores: 1  (MPI unavailable); wall_timeout_seconds: 600
  Always optimize geometry before charge analysis.

══ Parsing mulliken_charges in the llm node ═══════════════════════════════

  The artifact is a JSON list of dicts:
    [{"atom_index": 0, "symbol": "C", "charge": -0.037},
     {"atom_index": 1, "symbol": "C", "charge":  0.019}, ...]
  atom_index is 0-based (matches XYZ file order).
  Filter by symbol to get only C (and N, S for heteroaromatics).
  Sort by charge ascending for EAS ranking.

══ Required output format for the rank llm node ════════════════════════════

  Keep it simple — DO NOT attempt IUPAC mapping (no geometry available).
  Just filter carbons, sort by charge, and return atom_index + charge.

  CRITICAL: the top-level JSON key MUST be exactly "site_ranking" (not "site_ranking_json"
  or any other name). The product spec MUST use {"eas_sites_<mol>": "site_ranking"}.

  Return JSON:
  {
    "site_ranking": [
      {"atom_index": 2, "symbol": "C", "mulliken_charge": -0.180, "rank": 1},
      {"atom_index": 3, "symbol": "C", "mulliken_charge": -0.172, "rank": 2},
      ...
    ]
  }

  Key rules:
  • "site_ranking" is the ONLY required top-level key — do NOT wrap in "values" or "molecule".
  • Include ALL carbon atoms (and heteroatoms if relevant).
  • Sort by mulliken_charge ascending (most negative = rank 1 = most EAS-activated).
  • DO NOT invent IUPAC positions — only use atom_index from the input list.
  • The final compare node will handle cross-molecule interpretation.
  • Product spec for each rank node: {"eas_sites_<mol>": "site_ranking"}
""".strip()


# ---------------------------------------------------------------------------
# FSSH initial conditions skill (priority 35 — after Solvation=30, before NBO=40)
# ---------------------------------------------------------------------------

class FSSHInitialConditionsSkill(PlannerSkill):
    """MD-based FSSH initial conditions: MCPB.py force field → Amber MD → cpptraj → exyz."""

    name = "fssh_initial_conditions"
    priority = 35

    _KEYWORDS = (
        "fssh", "surface hopping", "fewest switches", "nonadiabatic", "non-adiabatic",
        "initial condition", "phase space sampling", "md sampling",
        "mcpb", "tleap", "sander", "cpptraj", "amber md",
        "solvent sampling", "snapshot sampling", "nuclear sampling", "thermal sampling",
        "p, q", "pq snapshot", "exyz", "nc_to_exyz",
    )

    def matches(self, user_text: str, state: Dict[str, Any]) -> bool:
        low = user_text.lower()
        return any(kw in low for kw in self._KEYWORDS)

    def render(self, user_text: str, state: Dict[str, Any]) -> str:
        return """SKILL: MD-Based FSSH Initial Conditions (p, q)
────────────────────────────────────────────────────────────

Goal: generate an ensemble of phase-space snapshots (positions q, momenta p) from
classical MD for use as initial conditions in FSSH nonadiabatic dynamics.
Example system: Cu(phen)₂Cl₂ complex in MeCN solvent.

══ Step 1 — Geometry optimisation for the metal complex ════════════════════

  Level of theory: match the planned FSSH electronic-structure level
  (e.g. SA-CASSCF(11,11)/PT2 for Cu(phen)₂Cl₂).
  Optimise in gas phase and keep geometry fixed during subsequent sampling
  if you intend to use a rigid-solute model.

══ Step 2 — Force field via MCPB.py ════════════════════════════════════════

  MCPB.py generates bonded force-field parameters for the metal centre
  using Gaussian 16 QM jobs. Required input files:

  Cu_Phen2_Cl2.MCPB.in keys:
    original_pdb   — PDB of the metal complex (HETATM records)
    ion_mol2files  — ElementSymbol.mol2 for each metal  (e.g. CU.mol2)
    naa_mol2files  — ResidueName.mol2 for non-standard ligands (e.g. RES.mol2)
    frcmod_files   — RES.frcmod (generated by parmchk2)
    cut_off        — coordination shell distance in Å (typically 3.2)
    smmodel_chg / smmodel_spin / lgmodel_chg / lgmodel_spin — charge & spin

  PDB and mol2 conventions:
    • Append numbers to all non-metal atom names (C1, N1, H1 …) to avoid
      name collisions with AMBER standard residues.
    • Metal mol2 filename MUST be ElementSymbol.mol2 (e.g. CU.mol2).
    • mol2 atom types must be GAFF format (CA, HA, …), not SYBYL (C.3).
      Convert with: antechamber -i RES.mol2 -fi mol2 -o RES.mol2 -fo mol2
                                -at gaff2 -c rc
    • parmchk2 -i RES.mol2 -f mol2 -o RES.frcmod -s gaff2

  MCPB.py run order:
    Step 1 → generates three Gaussian input files; run them in order:
               large_mk.com → small_opt.com → small_fc.com → formchk (.fchk)
    Steps 2–4 → MCPB.py -i Cu_Phen2_Cl2.MCPB.in -s 2 / -s 3 / -s 4
    Step 4 produces tleap input, mol2 files, and Cu_Phen2_Cl2_MCPB_mcpbpy.frcmod.

══ Step 3 — tleap: solvate with organic solvent ════════════════════════════

  Modify the tleap input generated by MCPB step 4:
    • source leaprc.gaff2   (replaces leaprc.protein.ff19SB + leaprc.water.opc)
    • Load solvent mol2 and frcmod (e.g. MeCN.mol2 / MeCN.frcmod)
    • solvatebox mol MECN 16.0   — random-fill with ≥ 16 Å padding
      (Ewald will crash if the box is too small; use a generous cutoff)
    • saveamberparm produces: complex.prmtop, complex.inpcrd

  Solvent force-field preparation (if no existing entry):
    antechamber -i MeCN.pdb -fi pdb -o MeCN.mol2 -fo mol2 -c bcc -at gaff2
    parmchk2    -i MeCN.mol2 -f mol2 -o MeCN.frcmod -s gaff2
    (The frcmod is usually nearly empty — GAFF2 already covers most MeCN parameters.)

══ Step 4 — Amber equilibration (sander) ═══════════════════════════════════

  Run three sequential sander stages:

  mmMIN.in  — energy minimisation (imin=1, maxcyc=2000)
    sander -O -i mmMIN.in -o mmMIN.out -p complex.prmtop
           -c complex.inpcrd -r mm.ncrst -x mmMIN.nc
           -inf mmMIN.info -ref complex.inpcrd

  mmHEAT.in — NVT heating 0 → 300 K (nstlim=10000, dt=0.0005, ntb=1)
    Use Langevin thermostat: ntt=3, gamma_ln=2.0, ig=-1
    Ramp via &wt type='TEMP0', istep1=0, istep2=10000, value1=0.0, value2=300
    sander -O -i mmHEAT.in … -c mm.ncrst -r mm.ncrst -x mmHEAT.nc -ref complex.inpcrd

  mmNPT.in  — NPT production (nstlim=100000, dt=0.001, ntb=2, ntp=1, taup=1)
    CRITICAL: set ntwv=-1 to write velocities to the trajectory
    (velocities are required for momenta p in FSSH initial conditions)
    Run an additional mmNPT.equi.in round for further equilibration.

  dt notes: use dt=0.0005 (no SHAKE) or dt=0.002 with ntc=ntf=2 (SHAKE on H).

══ Step 5 — cpptraj: extract solvent sphere per frame ══════════════════════

  After PBC equilibration, extract a finite sphere around the solute for QM/MM:

  1. Reimage: autoimage → center :CU1 mass origin → image origin center
     (saves mmNPT.reim.nc)

  2. Count solvent molecules within cutoff at frame 1:
     nativecontacts :CU1 :UNL distance 8.0 byresidue out contacts.dat

  3. Keep N closest solvent molecules per frame:
     closest <N> :CU1 solventmask :UNL
     (saves trajectory_closest_8angs.nc and .pdb)

══ Step 6 — Convert to extended XYZ with nc_to_exyz.py ════════════════════

  python nc_to_exyz.py -n trajectory_closest_8angs.nc
                       -p trajectory_closest_8angs.pdb
                       -o initial_conditions.exyz

  Output format: one frame per snapshot, each atom line contains:
    El  x  y  z  vx  vy  vz
  Coordinates in Å, velocities in Å/fs (converted from the NetCDF scale factor).
  These (q, p) pairs are the FSSH initial conditions ready for dynamics.

══ Key conventions and pitfalls ════════════════════════════════════════════

  • MCPB atom naming: non-metal atoms in PDB and mol2 MUST have numbered names
    (C1, N2 …); plain C/N/H will collide with AMBER standard residue names.
  • tleap solvatebox padding: use ≥ 16 Å; smaller boxes cause Ewald errors.
  • ntwv=-1 in mmNPT.in is mandatory — without it the .nc file contains no
    velocities and momenta p cannot be recovered.
  • If the geometry is fixed (rigid solute), omit ntwv for the solute atoms
    or keep track of momenta only for the solvent shell as needed.
  • Subsample with cpptraj "trajin mmNPT.reim.nc 1 last 5" for a sparser set.
""".strip()


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

SKILL_REGISTRY: List[PlannerSkill] = [
    MethodSelectionSkill(),
    CoordinationChemistrySkill(),
    ProtonationSiteSkill(),
    PKaSkill(),
    GeometrySkill(),
    InteractionScanSkill(),
    ScanSkill(),
    SpectrumSkill(),
    TDDFTSkill(),
    CasscfSkill(),
    ThermochemistrySkill(),
    TSSearchSkill(),
    SolvationSkill(),
    EASSkill(),
    FSSHInitialConditionsSkill(),
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


def collect_skill_client_tools(registry: List[PlannerSkill] = SKILL_REGISTRY) -> Dict[str, Callable]:
    """Merge client_tools declared by every registered skill into one dict.

    This is how a skill's deterministic tool functions get installed into the
    executor's tool registry (CLIENT_SIDE_TOOL_FUNCS in agent.py /
    test_success_rate.py) automatically — adding a new skill-specific
    calculator requires only implementing it and declaring it on the owning
    skill's client_tools, never editing agent.py or test_success_rate.py.

    Raises on a name collision between two different skills (fail loud rather
    than silently letting one skill's tool shadow another's).
    """
    merged: Dict[str, Callable] = {}
    for skill in registry:
        for tool_name, fn in (getattr(skill, "client_tools", None) or {}).items():
            if tool_name in merged and merged[tool_name] is not fn:
                raise ValueError(
                    f"client_tools collision: {tool_name!r} declared by multiple skills"
                )
            merged[tool_name] = _with_status(fn)
    return merged


def _with_status(fn: Callable) -> Callable:
    """Guarantee the executor's status contract for a skill-owned tool.

    The executor reads a tool result's "status" field and records the node as
    'unknown' when it is absent, which no node spec's expect.status_in can
    satisfy. That contract lived only in the example set by existing tools, so a
    new tool could satisfy every schema, compute the right answer, and still fail
    its node -- which is exactly what happened when rank_eas_sites was added: the
    chemistry ran, the rankings were correct, and all three benchmark repeats
    failed on execution_ok.

    Rather than ask each author to remember, the collector enforces it here. A
    dict returned without a status is a successful return by definition, since
    failures in these functions are raised, not encoded. Existing values are
    never overwritten, so a tool that reports its own status -- including
    "error" -- keeps it.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, dict) and "status" not in out:
            out = {"status": "ok", **out}
        return out

    wrapper.__wrapped__ = fn        # keep the original signature introspectable,
    return wrapper                  # build_tool_schemas reads it to make schemas
