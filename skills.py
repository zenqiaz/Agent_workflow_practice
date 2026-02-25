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

Basis set guidance:
  def2-SVP   → cheap, opt/freq, qualitative
  def2-TZVP  → accurate single-points, properties
  def2-TZVPP → high-accuracy; use only when def2-TZVP is insufficient

Performance flags:
  use_ri=True  reduces cost of hybrid DFT significantly (RIJCOSX in ORCA)
  ncores: default 1; increase for large molecules if VM allows
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
# Registry and dispatcher
# ---------------------------------------------------------------------------

SKILL_REGISTRY: List[PlannerSkill] = [
    MethodSelectionSkill(),
    ProtonationSiteSkill(),
    PKaSkill(),
    ThermochemistrySkill(),
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
