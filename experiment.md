# QC Agent Experiment Plan

Record for each experiment: **tokens** (planner/total), **elapsed time (s)**, **report quality**, **pass rate**.

---

## 1. Organic Compounds Basic (SP)

**Goal:** Test planner robustness under increasing context load.

**Task message template:**
```
Calculate single-point energy (B3LYP/def2-SVP) for each of the following N compounds.
Run them in parallel where possible. For each compound report: SP energy (energy_eh),
HOMO-LUMO gap (homo_lumo_gap_ev), and dipole moment (dipole_moment_debye).
All three are returned by run_sp_energy at no extra cost.
```

**Compound set:** `organic_compounds_100.csv` (random sample)

**Experiment runs:**

| N compounds | Runs | Mode | Notes |
|-------------|------|------|-------|
| 5 | 3 | plan | baseline |
| 10 | 3 | plan + full | artifact expansion verified |
| 20 | 3 | plan | robustness boundary |
| 30 | 3 | plan | problems expected here |
| 50 | 3 | plan | stress test |

**Pass criteria:**
- `has_sp_node` ✓
- `no_freq_job` ✓
- `has_energy_art` ✓
- Plan nodes = N × 2 exactly (geometry + SP per compound); extra nodes count as planner error

**Command:**
```bash
python test_context_scaling.py --sizes 5 10 20 30 50 --runs 3 --task sp
python test_success_rate.py --checker sp --runs 1 --csv organic_compounds_100.csv --n-compounds 10 --mode full
```

---

## 2. Coordination Compound Basic

**Goal:** Test geometry generation (molSimplify) + SP for transition-metal complexes.

**Compounds:**

| Complex | Formula | Notes |
|---------|---------|-------|
| Hexacyanoferrate(III) | Fe(CN)₆³⁻ | high-spin d⁵, CAS(5,5) recommended |
| Cisplatin | cis-Pt(NH₃)₂Cl₂ | square planar Pt(II) |
| Bis(ethylenediamine)copper(II) | Cu(en)₂²⁺ | square planar Cu(II) |
| Nickel acetylacetonate | Ni(acac)₂ | square planar Ni(II) |

*(Cu₂(AcO)₄(H₂O)₂ excluded — too difficult for current pipeline)*
*(Ni(dmgH)₂ excluded — "dmg" not in molSimplify ligand database; Ni(acac)₂ used instead)*

**Task message:**
```
Calculate properties of these compounds using ORCA with the B3LYP method and def2-SVP
basis set in the gas phase. For each compound report: final Cartesian coordinates (Å),
total energy (Hartrees), dipole moment (Debye), HOMO–LUMO gap (eV).
Use build_coordination_complex to generate initial geometries.
```

**Runs:** 3 per compound (plan), 1 full execution per compound

**Pass criteria:**
- `has_sp_node` or `has_opt_node` ✓
- `energy_negative` ✓
- `has_homo_lumo_art` ✓
- Geometry successfully built by molSimplify

**Notes:**
- Charge/multiplicity must be set correctly per complex
- Fe(CN)₆³⁻: charge=−3, mult=2 (LS) or mult=6 (HS)
- cisplatin: charge=0, mult=1
- Cu(en)₂²⁺: charge=+2, mult=2
- Ni(dmgH)₂: charge=0, mult=1

---

## 3. pKa — Carboxyl Alpha-C–H Acidity

**Goal:** Compute gas-phase pKa with isodesmic calibration; verify template-mode multi-compound pipeline.

**Target compounds:**

| Compound | Expected pKa (gas) | Notes |
|----------|--------------------|-------|
| Ethyl acetylacetate | ~13 | beta-ketoester |
| Barbituric acid | ~5 | very acidic CH₂ |
| Acetone | ~20 | simple ketone |
| Ethyl chloroacetate | ~16 | α-halo ester |

**Calibration reference:** ethanal (acetaldehyde), pKa_exp = 17.0

**Task message:**
```
Calculate the alpha-CH pKa of ethyl acetylacetate, barbituric acid, acetone,
and ethyl chloroacetate. Use ethanal (acetaldehyde, experimental pKa=17.0) as an
isodesmic calibration reference to correct the systematic gas-phase DFT error.
```

**Runs:** 3 full executions

**Pass criteria:**
- `pka_present` ✓ (at least one `pka_*` artifact)
- `pka_reasonable`: −20 < pKa < 60 ✓
- `G_eh_present`: ≥ 2 Gibbs free energy artifacts ✓
- `G_negative` ✓
- Calibrated pKa = pKa_raw − ε, where ε = pKa_calc(ethanal) − 17.0

**Known issues:**
- ethyl_chloroacetate A⁻ may hit RESOURCE_LIMIT → `patch_and_retry` handles automatically
- ethanal PubChem lookup: now has 3-retry backoff (fixed)
- `xtb_preopt=True` default on all opt nodes (fixed) — critical for sp3→sp2 anion

**Future:** Add microsolvation (nsolv=3) for aqueous pKa — not yet done.

**Command:**
```bash
python test_success_rate.py --checker pka --runs 3 --mode full
```

---

## 4. EAS — Aromatic Electrophilic Substitution Reactivity

**Goal:** Rank electrophilic reactivity of 6 heterocycles using Mulliken charges.

**Compounds:** naphthalene, pyrrole, imidazole, pyrazole, thiophene, thiazole

**Task message:**
```
Rank EAS reactivity of naphthalene, pyrrole, imidazole, pyrazole, thiophene, thiazole
using Mulliken charges from DFT.
```

**Runs:** 3 full executions

**Pass criteria:**
- SP or NBO node present ✓
- Mulliken charge artifacts present ✓
- Reporter ranks all 6 compounds ✓
- Reporter identifies **equivalent sites** (e.g. α vs β in naphthalene)
- Reporter uses IUPAC ring numbering (position 1, 2, 3…)

**Known issues / open items:**
1. Reporter must recognize equivalent (symmetry-equivalent) positions — e.g. α(1,4,5,8) and β(2,3,6,7) in naphthalene
2. Reporter should use standard IUPAC ring numbering per compound
3. These requirements are not yet enforced in `check_eas_result` — add checker criteria

**Command:**
```bash
python test_success_rate.py --checker eas --runs 3 --mode full
```

---

## Summary Table

| Experiment | Status | Pass rate | Notes |
|------------|--------|-----------|-------|
| Organic SP (N=3) | ✅ Done | 3/3 | artifact expansion verified |
| Organic SP (N=10) | ✅ Done | 2/2 | full execution confirmed |
| Organic SP (N=20–50) | 🔲 Pending | — | robustness sweep not yet run |
| Coordination basic | ✅ Done | 1/1 | all 4 complexes; ~97 min; 2026-03-28 |
| pKa (gas phase) | ✅ Done | 3/3 | ethyl_chloroacetate A- retried via patch_and_retry; 2026-03-28 |
| pKa (aqueous) | 🔲 Pending | — | add nsolv=3 solvation |
| EAS ranking | ✅ Done | 3/3 | duplicate-node fix confirmed; ~16 min/run; 2026-03-28 |
| EAS (improved reporter) | 🔲 Pending | — | enforce symmetry & IUPAC numbering in checker |
