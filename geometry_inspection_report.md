# Geometry Inspection Module — Implementation Report

**Date:** 2026-05-20  
**Files modified:** `geometry_helpers.py`  
**Files created:** `test_geometry_inspection.py`, `geometry_inspection_skill_outline.md`

---

## 1. Motivation

The QC agent planner receives molecular geometries in XYZ, SDF, mol2, or SMILES form
before passing them to expensive ORCA or Amber calculations. Structural problems in the
input geometry — steric clashes, wrong bond lengths, distorted angles, spin/charge
mismatches — cause SCF failures, optimization traps, or incorrect dynamics without any
informative error message. The goal of this module is to catch these problems cheaply
before any QC job is submitted.

---

## 2. Design

### 2.1 Pipeline

```
Input (atoms + coords + optional bond_list)
        │
        ▼
[Section 0]  Bond inference            ← skipped if bond_list provided (SDF / mol2 / SMILES)
        │
        ├──▶ [Section A]  Bond length checks
        ├──▶ [Section B]  Non-bonded clash detection
        ├──▶ [Section C]  Bond angle checks
        ├──▶ [Section D]  Torsion / eclipsed-conformation checks
        ├──▶ [Section E]  Charge and spin consistency
        └──▶ [Section F]  QM-specific hazards (F1–F6)
                │
                ▼
        InspectionReport
```

### 2.2 Input format handling

| Format | Bond connectivity | Section 0 |
|---|---|---|
| XYZ / extended XYZ | None | **Run** — full N(N−1)/2 inference |
| SDF / MOL | Explicit bonds + bond orders | **Skip** — pass `bond_list` from file |
| mol2 | Explicit bonds + SYBYL/GAFF types | **Skip** |
| PDB (ATOM with CONECT) | CONECT records | **Skip if present; Run if absent** |
| PDB (HETATM) | CONECT often absent | **Run** |
| SMILES + XYZ | SMILES gives topology; XYZ gives coords | **Skip** — parse SMILES externally, pass `bond_list` |

When SMILES is used, every SMILES-declared bond is cross-checked against the paired
XYZ distances in Section A. A declared bond with `d > 1.20 × reference` signals a
SMILES/XYZ mismatch (wrong conformer or tautomer).

---

## 3. Implementation

All functions were added to `geometry_helpers.py` (existing file).
No new files were introduced for the library code.

### 3.1 New constants

| Name | Purpose |
|---|---|
| `COV_RAD` (extended) | Added 12 metals: Cu, Fe, Ru, Ni, Pt, Zn, Co, Mn, Cr, Pd, Rh, Ir |
| `VDW_RAD` | van der Waals radii for 18 elements including common metals |
| `BOND_REF` | Reference bond lengths (Å) for 28 element pairs (organic + metal–ligand) |
| `METALS` | Frozenset of 20 transition/post-transition metals |
| `ATOMIC_NUM` | Atomic numbers for electron-count check in Section E |

### 3.2 New dataclasses

| Class | Fields |
|---|---|
| `BondIssue` | `i, j, element_i, element_j, distance, reference, flag` |
| `ClashIssue` | `i, j, element_i, element_j, distance, threshold` |
| `AngleIssue` | `i, j (central), k, element_j, angle_deg, lo, hi` |
| `TorsionIssue` | `i, j, k, l, dihedral_deg, flag` |
| `InspectionReport` | Aggregates all issue lists + `n_atoms`, `n_bonds`, `is_clean()`, `summary()` |

### 3.3 New functions

| Function | Section | Notes |
|---|---|---|
| `bonds_to_adj(n, bond_list)` | 0 | Converts `(i,j)` pair list to adjacency list |
| `infer_bond_list(atoms, coords, organic_scale, metal_scale)` | 0 | Covalent-radii threshold; 1.20 for organic, 1.30 for metal–ligand |
| `inspect_bonds(...)` | A | Flags bonds outside `[0.85×, 1.20×]` reference; wider `[0.80×, 1.25×]` for metal–ligand |
| `inspect_clashes(...)` | B | Excludes 1–2 and 1–3 pairs; flags `d < 0.70 × (vdW_i + vdW_j)` |
| `inspect_angles(...)` | C | Hybridisation inferred from degree; O/S always sp³ |
| `inspect_torsions(...)` | D | Flags dihedrals within ±10° of 0° (eclipsed) on sp³–sp³ bonds |
| `inspect_charge_spin(atoms, charge, multiplicity)` | E | Electron-count parity check against multiplicity |
| `inspect_qm_hazards(atoms, coords, bond_list)` | F | Codes F1–F6 (see below) |
| `inspect_geometry(atoms, coords, charge, multiplicity, bond_list)` | All | Top-level; pass `bond_list` to skip Section 0 |

### 3.4 QM hazard codes

| Code | Hazard | Remediation |
|---|---|---|
| F1 | Bond angle > 175° (near-linear) | Add `%geom AngleConstraint` or use Cartesian coords in ORCA |
| F2 | Two atoms within 0.01 Å (duplicate coordinates) | Reject; geometry is corrupt |
| F4 | Organic atom with far fewer bonds than expected valence | Re-protonate; check with `antechamber` |
| F5 | Metal coordination number outside expected range | Rebuild with `build_coordination_complex` |
| F6 | Disconnected molecular graph | Confirm intentional dimer; warn if unexpected |

*F3 (near-symmetry detection) is deferred — requires point-group analysis.*

---

## 4. Complexity

All sections run in milliseconds for the molecule sizes encountered in this project.
The dominant cost is the N(N−1)/2 pair scan shared by Sections 0 and B.

| Section | C20 chain (N=62) | C20 aromatic (N=26) | Cu(phen)₂Cl₂ (N=37) |
|---|---|---|---|
| 0 — Bond inference | 1 891 pairs (XYZ) / 0 (SDF/SMILES) | 325 / 0 | 666 / 0 |
| A — Bond lengths | 61 checks | 33 checks | 46 checks |
| B — Non-bonded clashes | 1 830 pairs | 292 pairs | 620 pairs |
| C — Bond angles | 110–120 | 58–65 | 78–88 |
| D — Torsions | ~75 | ~8 | ~18 |
| E — Charge/spin | 62 atoms → 1 verdict | 26 → 1 | 37+1 → 1 |
| F — QM hazards | 6 fixed tests | 6 | 6 |

Sections 0 and B share the same pair distances for XYZ input — Section B consumes
the already-computed non-bonded bin at zero extra cost.

---

## 5. Unit tests

**File:** `test_geometry_inspection.py`  
**Framework:** pytest  
**Result:** 42 / 42 passed

### Test fixtures

| Fixture | Description |
|---|---|
| `_methane()` | C at origin, 4 H at tetrahedral positions (d = 1.09 Å) |
| `_water()` | O at origin, H–O–H = 104.5°, d(O–H) = 0.96 Å |
| `_staggered_ethane()` | H–C–C–H dihedral = 60° (no issues) |
| `_eclipsed_ethane()` | H–C–C–H dihedral = 0° (eclipsed) |
| `_co2()` | Linear O–C–O, angle = 180° |
| `_cu_complex()` | Cu(II) with 4 N ligands at 2.0 Å, square-planar |

### Test coverage by section

| Section | Tests | Key scenarios |
|---|---|---|
| 0 — Bond inference | 7 | Methane, water, ethane, Cu complex, H₂ not bonded, SDF bypass |
| A — Bond lengths | 6 | Clean methane/water, compressed C–H, stretched C–H, Cu–N normal, unknown pair skipped |
| B — Clashes | 4 | Clean methane, H–H clash detected, bonded pair not flagged, 1–3 pair not flagged |
| C — Angles | 4 | Clean methane/water, compressed angle, CO₂ linear no false positive |
| D — Torsions | 3 | Staggered (no issue), eclipsed (flagged), metal bond skipped |
| E — Charge/spin | 6 | Water/methane correct, CH₃ radical singlet wrong, doublet correct, anion/cation parity |
| F — QM hazards | 7 | Clean methane, CO₂ F1, duplicate coords F2, disconnected graph F6, missing H F4, wrong CN F5, correct CN no F5 |
| Top-level | 5 | Clean methane report, clean water, explicit bond list, multi-issue summary, Cu complex |

---

## 6. Bugs found during testing

### Bug 1 — `_angle_range`: O and S incorrectly treated as sp/linear

**Location:** `inspect_angles` → `_angle_range`  
**Symptom:** Water (H–O–H = 104.5°) was flagged as an angle issue because O with
degree 2 was assigned the sp/linear range (155–180°).  
**Root cause:** The original range function applied the same linear-range rule to all
elements with degree 2, including O and S. In reality, O and S are always bent due to
lone pairs — they are never sp regardless of formal degree.  
**Fix:** O and S now unconditionally return the sp³ range (85–135°). Only C with
degree 2 is treated as sp/linear (acetylene, CO₂ central carbon, nitrile).

```python
# Before (wrong):
if degree == 2:
    return (155.0, 180.0)   # applied to all elements

# After (correct):
if el == "C":
    if degree == 2: return (155.0, 180.0)   # sp only for carbon
if el in ("O", "S"):
    return (85.0, 135.0)    # always sp3-like (water 104.5°, ether ~111°)
```

### Bug 2 — `inspect_charge_spin`: parity logic inverted

**Location:** `inspect_charge_spin`  
**Symptom:** CH₃ radical (9 electrons) with `multiplicity=2` (doublet, correct) was
being flagged as an error.  
**Root cause:** The parity rule was backwards. The code flagged `odd electrons AND
even multiplicity`, but even multiplicity (2, 4, 6, ...) is exactly what an
odd-electron system requires.

| Electrons | Valid multiplicity | 2S+1 |
|---|---|---|
| Odd | Even (2, 4, 6, ...) | doublet, quartet, ... |
| Even | Odd (1, 3, 5, ...) | singlet, triplet, ... |

**Fix:** Inverted both conditions.

```python
# Before (wrong):
if odd and multiplicity % 2 == 0:   # flagged doublet for radical — incorrect
    ...

# After (correct):
if odd and multiplicity % 2 != 0:   # flags odd multiplicity for odd electrons
    ...
elif not odd and multiplicity % 2 == 0:   # flags even multiplicity for even electrons
    ...
```

---

## 7. Next steps

1. **Integrate into the planner as `GeometryInspectionSkill`** in `skills.py` —
   skill outline is in `geometry_inspection_skill_outline.md`.
2. **Add `inspect_geometry` call to the server** as a lightweight pre-check before
   any `run_opt_job` or `run_casscf_job` node.
3. **F3 (near-symmetry detection)** — deferred; requires point-group deviation
   analysis, out of scope for initial implementation.
4. **Aromatic bond order** — currently BOND_REF has no aromatic C–C entry; consider
   adding 1.40 Å reference for mol2/SDF input where bond type is known.
5. **Ring planarity check** — aromatic ring puckering > 0.2 Å is described in the
   outline but not yet implemented in `inspect_torsions`.
