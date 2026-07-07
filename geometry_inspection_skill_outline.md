# GeometryInspectionSkill — Design Outline

## Purpose

Pre-flight geometry check before expensive QM calculations. Detects structural
problems that cause SCF failures, geometry optimization traps, or bad dynamics,
and recommends remediation without performing a full optimization itself.

**Priority:** 17 — fires just before `CoordinationChemistrySkill` (18) and
`GeometrySkill` (18), so problems are flagged before any planning begins.

**Trigger keywords:** `inspect`, `check geometry`, `geometry quality`,
`bad structure`, `clash`, `bond length`, `bad bond`, `high energy`, `pre-check`,
`geometry issue`, `validate structure`

---

## Section 0 — Bond inference (prerequisite for all other sections)

### Format detection: when to run Section 0

Section 0 is only needed when the input does **not** carry explicit bond
connectivity. Decision table:

| Format | Bond connectivity | Bond orders | Section 0 |
|---|---|---|---|
| XYZ (`.xyz`) | None | None | **Run** — full N(N−1)/2 inference |
| Extended XYZ (`.exyz`) | None | None | **Run** |
| SDF / MOL (`.sdf`, `.mol`) | Explicit | Yes (1/2/3/ar) | **Skip** — read graph directly |
| mol2 (`.mol2`) | Explicit | Yes (SYBYL/GAFF types) | **Skip** — read graph directly |
| PDB (`.pdb`) — ATOM records | CONECT records (when present) | No | **Skip if CONECT present; otherwise Run** |
| PDB — HETATM records | CONECT often absent | No | **Run** (CONECT is typically missing for ligands) |
| SMILES (string) | Explicit | Yes (single/double/triple/aromatic) | **Skip** — parse graph from SMILES |

#### SMILES as bond source

SMILES encodes atom types, bond connectivity, bond orders, aromaticity, and
formal charges in a compact string — but contains **no 3D coordinates**.
It must therefore be paired with a coordinate source:

```
Input pair:  SMILES string  →  bond graph + bond orders + formal charges
             XYZ / exyz     →  3D coordinates for distance measurements
```

Workflow when SMILES is provided alongside XYZ:
1. Parse SMILES to extract the bond graph (e.g. via RDKit `MolFromSmiles`).
2. Atom ordering in SMILES may differ from XYZ — a mapping step is required:
   match by element type and connectivity pattern (or rely on the caller to
   supply a canonical ordering).
3. Use the SMILES-derived graph as the bond topology; skip Section 0.
4. Use XYZ coordinates for all distance, angle, and torsion measurements.

Advantages of SMILES over pure XYZ:
- Bond orders are unambiguous (no aromatic-bond guessing needed for Section A).
- Formal charges and implicit-H counts are explicit → Section E is more reliable.
- Aromaticity flags suppress false positives in Section C (aromatic C–C angles
  are always ~120°, not ~109°).

Limitation: if the SMILES and XYZ describe different conformers or tautomers,
the bond graph will be inconsistent with the coordinates. The inspector should
cross-check: for every SMILES-declared bond, verify that the XYZ distance falls
within the expected range (same as Section A). Any declared bond with a distance
> 1.20 × reference indicates a SMILES / XYZ mismatch.

### Algorithm (XYZ / exyz / PDB-without-CONECT only)

Two atoms *i* and *j* are assigned a bond if:

```
d(i, j) < ( r_cov(i) + r_cov(j) ) × threshold
```

- Default threshold: **1.20** for organic bonds; **1.30** for any pair
  involving a metal atom (M–L bonds are longer and more variable).
- The full N(N−1)/2 pair scan is done **once** and the result is classified
  into three bins simultaneously:
  - **bonded** → feeds Sections A, C, D
  - **non-bonded clash** → feeds Section B
  - **OK non-bonded** → no further action

### Covalent radii table (Å)

| Element | r_cov | Element | r_cov | Element | r_cov |
|---|---|---|---|---|---|
| H  | 0.31 | N  | 0.71 | S  | 1.05 |
| C  | 0.77 | O  | 0.66 | Cl | 1.02 |
| Cu | 1.32 | Fe | 1.32 | Ru | 1.46 |
| Ni | 1.24 | Pt | 1.36 | Zn | 1.22 |

### Known complications

| Situation | Problem | Mitigation |
|---|---|---|
| Metal–ligand bonds (e.g. Cu–N ~2.0 Å) | Borderline for threshold 1.20 | Use threshold 1.30 for any pair containing a metal |
| Van der Waals dimer (two molecules in contact) | Intermolecular contact may be inferred as a bond | Flag disconnected graph components; warn if threshold > 1.25 generated an inter-fragment bond |
| Aromatic C–C (bond order ambiguity) | Length ~1.40 Å lies between single (1.54) and double (1.34) reference | Treat as a third category "aromatic"; use 1.40 Å reference in Section A |
| H-bond donor H to acceptor O (~1.8–2.0 Å) | Could be inferred as a bond | Safe: (0.31+0.66)×1.20 = 1.16 Å threshold — H-bond is NOT inferred |

### H–H case in detail

H has the smallest covalent radius (0.31 Å). The bonding threshold for any
H–H pair is:

```
( 0.31 + 0.31 ) × 1.20 = 0.744 Å
```

The only real H–H bond (H₂ gas, 0.74 Å) sits exactly at this threshold.
In all practical molecular geometries H–H pairs are **never** assigned a bond.

For Section B, H has a vdW radius of 1.20 Å:

```
vdW sum(H–H) = 1.20 + 1.20 = 2.40 Å
clash threshold = 2.40 × 0.70 = 1.68 Å
```

Any two non-bonded H atoms within **1.68 Å** → flagged as a steric clash.
This is the most common artifact in PDB-derived or MCPB-prepared structures
(methylene H–H clashes at chain branch points).

---

## Section A — Bond length checks

Runs on every edge in the bond graph produced by Section 0.

Reference bond lengths (Å):

| Bond | Single | Double | Triple / Aromatic |
|---|---|---|---|
| C–C | 1.54 | 1.34 | 1.20 / 1.40 |
| C–N | 1.47 | 1.28 | — |
| C–O | 1.43 | 1.22 | — |
| C–H | 1.09 | — | — |
| N–H | 1.01 | — | — |
| O–H | 0.96 | — | — |
| M–N | 2.0–2.2 | — | — |
| M–Cl | 2.2–2.4 | — | — |
| M–O | 2.0–2.2 | — | — |

- Flag: measured bond < 0.85 × reference → likely clash or force-field artifact
- Flag: measured bond > 1.20 × reference → broken bond, missing atom, or
  wrong connectivity inferred by Section 0
- Special case: fullerene and small-ring C–C bonds are intentionally short/strained;
  do not apply organic sp³ reference

---

## Section B — Non-bonded clashes

Runs on all pairs classified as **non-bonded** by Section 0.

- Clash criterion: `d(i, j) < 0.70 × ( r_vdW(i) + r_vdW(j) )`
- vdW radii (Å): H 1.20, C 1.70, N 1.55, O 1.52, Cl 1.75, S 1.80, Cu 1.40

| Pair type | Clash threshold (Å) | Common cause |
|---|---|---|
| H–H | 1.68 | sp³ branch-point H's in PDB conversion |
| H–C | 1.96 | Missing H repositioning after protonation |
| H–N / H–O | 1.93 / 1.90 | Proton added to wrong site |
| C–C | 2.38 | Force-field artifact, ring flip |

Remediation: `xtb_preopt: true` resolves most H–H and H–heavy clashes.

---

## Section C — Bond angles

Enumerated from the bond graph: for each atom *j* with degree ≥ 2, check all
pairs of neighbours (i–j–k). Number of angles at atom *j* = C(deg(*j*), 2).

Expected ranges by hybridisation:

| Hybridisation | Expected | Flag if outside |
|---|---|---|
| sp³ (C, N, O) | ~109.5° | < 85° or > 135° |
| sp² (C, N) | ~120° | < 100° or > 140° |
| sp (C, N) | ~180° | < 160° |
| Octahedral M (cis) | ~90° | < 75° or > 105° |
| Octahedral M (trans) | ~180° | < 160° |
| Tetrahedral M | ~109° | < 90° or > 130° |
| Square planar M | ~90° / 180° | < 75° or > 105° |
| Chelate bite (phen) | ~82° | < 70° or > 95° |
| Chelate bite (en) | ~85° | < 72° or > 98° |

Note: small-ring and fullerene angles are intentionally strained — the
inspector must accept angles as low as 60° (cyclopropane) without flagging.

---

## Section D — Torsion / conformation

Enumerated as all paths of length 3 (four atoms) in the bond graph, for bonds
connecting two atoms each with degree ≥ 2.

- Eclipsed sp³–sp³ bonds (dihedral ~0° ± 10°): raises energy, can trap SCF
- Aromatic ring planarity: compute mean plane, flag any atom with deviation > 0.2 Å
- Inverted stereocenters relative to expected configuration (requires reference)

Skip torsion checks for:
- Any bond involving a metal (M–L rotation is not meaningful in this context)
- Bonds in aromatic rings (by definition planar; planarity is checked separately)

---

## Section E — Charge and spin consistency

One pass over all atoms (O(N)):

- Sum formal valence electrons from atomic numbers and bond graph degrees
- Compare with user-supplied `charge` and `multiplicity`
- Flag: valence sum inconsistent with charge → likely missing or extra H
- Flag: odd electron count with `multiplicity = 1` → must be open-shell
- Common issue: proton removed from structure but `charge` not decremented,
  or vice versa

---

## Section F — QM-specific hazards

Fixed checklist of 6 tests, independent of molecule size:

| # | Hazard | Detection | Action |
|---|---|---|---|
| F1 | Near-linear geometry (angle > 175°) | Scan Section C results | Add `%geom AngleConstraint` or use Cartesian coords in ORCA |
| F2 | Duplicate coordinates | Any pair with d < 0.01 Å | Reject immediately; geometry is corrupt |
| F3 | Near-symmetric but not exact | Point-group deviation > 0.05 Å | Use `nosym` or symmetrise before QC |
| F4 | Missing H atoms | Valence check from Section E | Re-protonate; check with `antechamber` |
| F5 | Metal with wrong coordination number | Count M–L bonds from Section 0 graph | Rebuild with `build_coordination_complex` |
| F6 | Disconnected graph (multiple fragments) | Check graph connectivity | Confirm intentional dimer; warn if unexpected |

---

## Remediation decision tree

| Issue severity | Recommendation |
|---|---|
| Minor H–H or H–heavy clash (Section B) | `xtb_preopt: true` on first `run_opt_job` |
| Eclipsed torsion, wrong conformation (Section D) | `xtb_preopt: true` or force-field pre-opt |
| Bond length out of range (Section A) | Check if bond inference is wrong first; if confirmed, rebuild geometry |
| Charge / spin mismatch (Section E) | Correct `charge` and `multiplicity` in plan before any QC node |
| Duplicate coordinates (F2) | Reject immediately; cannot proceed |
| Missing H atoms (F4) | Re-protonate with `antechamber` or `tleap` |
| Wrong coordination number (F5) | Rebuild with `build_coordination_complex` |

---

## Call-count estimate per section

Reference molecules:
- **C20 chain**: eicosane C₂₀H₄₂, N = 62 atoms, purely sp³
- **C20 aromatic**: pyrene C₁₆H₁₀ as proxy (C20 PAH), N = 26 atoms, fused rings
- **Coordination**: Cu(phen)₂Cl₂, N = 37 atoms (25 heavy + 12 H)

| Section | What is one "call" | C20 chain (N=62) | C20 aromatic (N=26) | Cu(phen)₂Cl₂ (N=37) | Driver |
|---|---|---|---|---|---|
| **0 — Bond inference** | One pair distance + classification | **1 891** (XYZ) / **0** (SDF / SMILES) | **325** (XYZ) / **0** (SDF / SMILES) | **666** (XYZ) / **0** (SDF / SMILES) | N(N−1)/2; skipped entirely for SDF, mol2, or SMILES input |
| **A — Bond lengths** | One bond vs. reference table | **61** bonds | **33** bonds | **46** bonds | = number of bonds (inferred or explicit) |
| **B — Non-bonded clashes** | One non-bonded pair vs. vdW threshold | **1 830** (XYZ) / **1 830** (SDF) | **292** (XYZ) / **292** (SDF) | **620** (XYZ) / **620** (SDF) | Always N(N−1)/2 − bonds; Section 0 pre-computes for XYZ, SDF reads bonds then scans remaining pairs |
| **C — Bond angles** | One angle vs. expected range | **110–120** | **58–65** | **78–88** | Σ C(deg(j),2) over all heavy-atom centres |
| **D — Torsions** | One dihedral vs. ±10° eclipsed / planarity | **~75** (17 backbone + ~58 H-C-C-H) | **~8** (ring planarity only) | **~18** (chelate ring + ligand orientation) | Chain: many sp³ bonds; aromatic: near-zero free rotation |
| **E — Charge/spin** | One valence check per atom | **62** → 1 verdict | **26** → 1 verdict | **37** + 1 metal → 1 verdict | O(N) |
| **F — QM hazards** | One test per hazard type | **6** tests | **6** tests | **6** tests | Fixed checklist |

### Key observations

- **Section 0 is entirely skipped for SDF, mol2, or SMILES input** — the bond
  graph is read directly from the file or parsed from the SMILES string. Bond
  orders are also known in all three cases, removing the aromatic bond
  ambiguity that affects XYZ inference. SMILES additionally provides formal
  charges explicitly, making Section E more reliable.
- **SMILES requires a consistency cross-check** — since SMILES carries no
  coordinates, every declared bond must be validated against the paired XYZ
  distances in Section A. A declared bond with d > 1.20 × reference signals a
  SMILES / XYZ mismatch that must be resolved before proceeding.
- **Section B always requires a non-bonded pair scan** regardless of input
  format — even with SDF, we must check all non-bonded pairs for clashes.
  For XYZ, Section 0 produces these distances as a byproduct; for SDF, a
  separate O(N²) scan is still needed for Section B.
- **Sections 0 and B share the same N² scan for XYZ input** — Section 0
  classifies all pairs into *bonded* / *clash* / *OK* in one pass; Section B
  consumes the already-computed *clash* bin at zero extra cost.
- **Section D is near-zero for aromatics.** Fused-ring systems have no freely
  rotating bonds; the only checks are ring planarity deviations (~8 atoms tested).
- **Coordination compounds sit between the two organics** in all sections.
  The phen rings suppress torsion count; the metal centre adds ~6 extra angle
  checks (L–M–L) and 4 extra bond checks (M–L), offset by the smaller N.
- **Section F is always 6 calls** regardless of molecule size.

### Overall complexity summary

| Molecule | Dominant cost | Total pair evaluations | Total bond/angle/torsion checks |
|---|---|---|---|
| C20 chain | Section 0 / B (N² scan) | 1 891 | ~246–256 |
| C20 aromatic | Section 0 / B (N² scan) | 325 | ~105–112 |
| Cu(phen)₂Cl₂ | Section 0 / B (N² scan) | 666 | ~148–158 |

All cases complete in milliseconds in pure Python. The real design cost lies in:
1. **Reference data coverage** — metal bond-length and angle entries, aromatic
   C–C category, strained-ring exceptions.
2. **Edge-case rules** — fullerene / small-ring angle tolerances to avoid false positives.
3. **Remediation routing** — the domain logic that maps flagged issues to the
   correct planner action (`xtb_preopt`, rebuild, charge fix, etc.).
