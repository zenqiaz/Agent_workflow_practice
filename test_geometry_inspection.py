"""
Unit tests for geometry inspection functions in geometry_helpers.py.

Covers:
  Section 0  infer_bond_list / bonds_to_adj
  Section A  inspect_bonds
  Section B  inspect_clashes
  Section C  inspect_angles
  Section D  inspect_torsions
  Section E  inspect_charge_spin
  Section F  inspect_qm_hazards
  Top-level  inspect_geometry  (with and without explicit bond list)
"""

import math
import pytest
from geometry_helpers import (
    Vec3,
    infer_bond_list,
    bonds_to_adj,
    inspect_bonds,
    inspect_clashes,
    inspect_angles,
    inspect_torsions,
    inspect_charge_spin,
    inspect_qm_hazards,
    inspect_geometry,
    InspectionReport,
)

# ── Geometry fixtures ─────────────────────────────────────────────────────────

def _methane():
    """Clean sp3 methane. C at origin, 4 H at tetrahedral positions (d=1.09 Å)."""
    s = 1.09 / math.sqrt(3)
    atoms = ["C", "H", "H", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        ( s,  s,  s),
        (-s, -s,  s),
        (-s,  s, -s),
        ( s, -s, -s),
    ]
    return atoms, coords

def _water():
    """Clean water. O at origin, two H at 0.96 Å, H-O-H = 104.5°.
    H1 along +x; H2 at 104.5° from H1 measured at O."""
    atoms = ["O", "H", "H"]
    coords = [
        (0.0, 0.0, 0.0),
        (0.96, 0.0, 0.0),
        (0.96 * math.cos(math.radians(104.5)),
         0.96 * math.sin(math.radians(104.5)), 0.0),
    ]
    return atoms, coords

def _staggered_ethane():
    """Ethane with staggered conformation (H-C-C-H dihedral = 60°)."""
    # C–C along x, d=1.54 Å
    # Tetrahedral H's on C1 (x<0 side):
    #   reference H at (-0.363, 1.028, 0.0)
    # H's on C2 staggered by 60° (rotate by 60° around x):
    c1 = (-0.77, 0.0, 0.0)
    c2 = ( 0.77, 0.0, 0.0)
    r = 1.028
    dx = -0.363
    angles_c1 = [0.0, 2*math.pi/3, 4*math.pi/3]
    angles_c2 = [math.pi/3, math.pi, 5*math.pi/3]   # staggered: offset by 60°
    atoms = ["C", "C"]
    coords: list = [c1, c2]
    for a in angles_c1:
        coords.append((c1[0] + dx, r * math.cos(a), r * math.sin(a)))
        atoms.append("H")
    for a in angles_c2:
        coords.append((c2[0] - dx, r * math.cos(a), r * math.sin(a)))
        atoms.append("H")
    return atoms, coords

def _eclipsed_ethane():
    """Ethane with eclipsed conformation (H-C-C-H dihedral ≈ 0°)."""
    c1 = (-0.77, 0.0, 0.0)
    c2 = ( 0.77, 0.0, 0.0)
    r = 1.028
    dx = -0.363
    angles = [0.0, 2*math.pi/3, 4*math.pi/3]   # same angles on both carbons
    atoms = ["C", "C"]
    coords: list = [c1, c2]
    for a in angles:
        coords.append((c1[0] + dx, r * math.cos(a), r * math.sin(a)))
        atoms.append("H")
    for a in angles:
        coords.append((c2[0] - dx, r * math.cos(a), r * math.sin(a)))
        atoms.append("H")
    return atoms, coords

def _co2():
    """Linear CO2: O–C–O angle = 180°."""
    atoms = ["O", "C", "O"]
    coords = [(-1.16, 0.0, 0.0), (0.0, 0.0, 0.0), (1.16, 0.0, 0.0)]
    return atoms, coords

def _cu_complex():
    """Minimal Cu(II) complex: Cu with 4 N ligands at ~2.0 Å, square-planar.
    Represents a simplified Cu(phen)2 without the carbon backbone."""
    atoms = ["Cu", "N", "N", "N", "N"]
    coords = [
        (0.0, 0.0, 0.0),
        ( 2.0,  0.0,  0.0),
        (-2.0,  0.0,  0.0),
        ( 0.0,  2.0,  0.0),
        ( 0.0, -2.0,  0.0),
    ]
    return atoms, coords

# ── Section 0: bond inference ─────────────────────────────────────────────────

class TestInferBondList:
    def test_methane_bonds(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        # C bonded to all 4 H, no H–H bonds
        assert len(bonds) == 4
        for i, j in bonds:
            pair = {atoms[i], atoms[j]}
            assert pair == {"C", "H"}

    def test_water_bonds(self):
        atoms, coords = _water()
        bonds = infer_bond_list(atoms, coords)
        assert len(bonds) == 2
        for i, j in bonds:
            assert {atoms[i], atoms[j]} == {"O", "H"}

    def test_ethane_bonds(self):
        atoms, coords = _staggered_ethane()
        bonds = infer_bond_list(atoms, coords)
        # 1 C–C + 6 C–H = 7 bonds
        assert len(bonds) == 7

    def test_cu_complex_metal_bonds(self):
        """Metal bonds require higher threshold (1.30) to be inferred correctly."""
        atoms, coords = _cu_complex()
        bonds = infer_bond_list(atoms, coords)
        # All 4 N should be bonded to Cu at 2.0 Å
        cu_bonds = [(i, j) for i, j in bonds if "Cu" in {atoms[i], atoms[j]}]
        assert len(cu_bonds) == 4

    def test_bonds_to_adj_roundtrip(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        adj = bonds_to_adj(len(atoms), bonds)
        # C (index 0) should have 4 neighbors
        assert len(adj[0]) == 4
        # Each H should have exactly 1 neighbor (C)
        for i in range(1, 5):
            assert adj[i] == [0]

    def test_h2_not_inferred_as_bonded(self):
        """Two H atoms 1.5 Å apart should NOT be assigned a bond."""
        atoms = ["H", "H"]
        coords = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]
        bonds = infer_bond_list(atoms, coords)
        # threshold = 1.20 * (0.31+0.31) = 0.744 Å — 1.5 Å is well outside
        assert bonds == []

    def test_sdf_bond_list_skips_inference(self):
        """Passing explicit bond_list skips Section 0 entirely."""
        atoms, coords = _methane()
        explicit = [(0, 1), (0, 2), (0, 3), (0, 4)]
        report = inspect_geometry(atoms, coords, bond_list=explicit)
        assert report.n_bonds == 4

# ── Section A: bond length checks ────────────────────────────────────────────

class TestInspectBonds:
    def test_clean_methane_no_issues(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_bonds(atoms, coords, bonds)
        assert issues == []

    def test_clean_water_no_issues(self):
        atoms, coords = _water()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_bonds(atoms, coords, bonds)
        assert issues == []

    def test_compressed_ch_flagged(self):
        """C–H bond at 0.60 Å (< 0.85 × 1.09 = 0.927 Å) → too_short."""
        atoms = ["C", "H"]
        coords = [(0.0, 0.0, 0.0), (0.60, 0.0, 0.0)]
        bonds = [(0, 1)]
        issues = inspect_bonds(atoms, coords, bonds)
        assert len(issues) == 1
        assert issues[0].flag == "too_short"
        assert issues[0].element_i in ("C", "H")

    def test_stretched_ch_flagged(self):
        """C–H bond at 1.50 Å (> 1.20 × 1.09 = 1.308 Å) → too_long."""
        atoms = ["C", "H"]
        coords = [(0.0, 0.0, 0.0), (1.50, 0.0, 0.0)]
        bonds = [(0, 1)]
        issues = inspect_bonds(atoms, coords, bonds)
        assert len(issues) == 1
        assert issues[0].flag == "too_long"

    def test_cu_n_normal_no_issue(self):
        """Cu–N at 2.05 Å is within metal tolerance → no issue."""
        atoms, coords = _cu_complex()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_bonds(atoms, coords, bonds)
        assert issues == []

    def test_unknown_pair_skipped(self):
        """Pairs not in BOND_REF are silently skipped."""
        atoms = ["Si", "Si"]
        coords = [(0.0, 0.0, 0.0), (2.35, 0.0, 0.0)]
        bonds = [(0, 1)]
        issues = inspect_bonds(atoms, coords, bonds)
        assert issues == []

# ── Section B: non-bonded clashes ────────────────────────────────────────────

class TestInspectClashes:
    def test_clean_methane_no_clash(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_clashes(atoms, coords, bonds)
        assert issues == []

    def test_hh_clash_detected(self):
        """Two H atoms at 1.0 Å (< 0.70 × 2.40 = 1.68 Å) → clash."""
        atoms = ["C", "H", "H"]
        coords = [
            (0.0, 0.0, 0.0),
            (1.09, 0.0, 0.0),
            (1.09 + 1.0, 0.0, 0.0),  # 1.0 Å from first H, not bonded to anything
        ]
        bonds = [(0, 1)]  # only C–H bond; second H is isolated but close
        issues = inspect_clashes(atoms, coords, bonds)
        hh = [iss for iss in issues if {iss.element_i, iss.element_j} == {"H", "H"}]
        assert len(hh) >= 1

    def test_bonded_pair_not_flagged(self):
        """Bonded atoms should never appear as clashes even at normal bond distances."""
        atoms, coords = _water()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_clashes(atoms, coords, bonds)
        bonded_indices = {frozenset([i, j]) for i, j in bonds}
        for iss in issues:
            assert frozenset([iss.i, iss.j]) not in bonded_indices

    def test_one_three_pair_not_flagged(self):
        """Atoms sharing a common bond (1–3 pair) should not be flagged as clashes."""
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_clashes(atoms, coords, bonds)
        # H–H distance in methane ≈ 1.78 Å; threshold ≈ 1.68 Å → should be clean
        assert issues == []

# ── Section C: bond angles ────────────────────────────────────────────────────

class TestInspectAngles:
    def test_methane_angles_clean(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_angles(atoms, coords, bonds)
        assert issues == []

    def test_water_angles_clean(self):
        atoms, coords = _water()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_angles(atoms, coords, bonds)
        assert issues == []

    def test_compressed_angle_flagged(self):
        """Force an H–C–H angle of ~60° (well below sp3 lower bound of 85°)."""
        # Place two H's only 1.0 Å apart while bonded to the same C
        atoms = ["C", "H", "H"]
        coords = [
            (0.0, 0.0, 0.0),
            (1.09,  0.0,  0.0),
            (1.09 * math.cos(math.radians(60)),
             1.09 * math.sin(math.radians(60)), 0.0),
        ]
        bonds = [(0, 1), (0, 2)]
        issues = inspect_angles(atoms, coords, bonds)
        assert any(iss.angle_deg < 85.0 for iss in issues)

    def test_co2_linear_no_false_positive(self):
        """CO2 central C has degree 2 → sp range (155–180°); 180° should not be flagged."""
        atoms, coords = _co2()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_angles(atoms, coords, bonds)
        c_issues = [iss for iss in issues if iss.element_j == "C"]
        assert c_issues == []

# ── Section D: torsions ───────────────────────────────────────────────────────

class TestInspectTorsions:
    def test_staggered_ethane_no_issue(self):
        atoms, coords = _staggered_ethane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_torsions(atoms, coords, bonds)
        assert issues == []

    def test_eclipsed_ethane_flagged(self):
        atoms, coords = _eclipsed_ethane()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_torsions(atoms, coords, bonds)
        assert len(issues) >= 1
        assert all(iss.flag == "eclipsed" for iss in issues)
        assert all(abs(iss.dihedral_deg) < 10.0 for iss in issues)

    def test_metal_bonds_skipped(self):
        """Torsions across metal–ligand bonds should not be checked."""
        atoms, coords = _cu_complex()
        bonds = infer_bond_list(atoms, coords)
        issues = inspect_torsions(atoms, coords, bonds)
        assert issues == []

# ── Section E: charge and spin ────────────────────────────────────────────────

class TestInspectChargeSpin:
    def test_water_singlet_correct(self):
        atoms, _ = _water()
        issues = inspect_charge_spin(atoms, charge=0, multiplicity=1)
        assert issues == []

    def test_methane_singlet_correct(self):
        atoms, _ = _methane()
        issues = inspect_charge_spin(atoms, charge=0, multiplicity=1)
        assert issues == []

    def test_odd_electron_singlet_flagged(self):
        """Methyl radical CH3 (charge=0) has 9 electrons — must be doublet."""
        atoms = ["C", "H", "H", "H"]  # 6+3 = 9 electrons
        issues = inspect_charge_spin(atoms, charge=0, multiplicity=1)
        assert len(issues) >= 1
        assert "open-shell" in issues[0].lower() or "radical" in issues[0].lower()

    def test_odd_electron_doublet_correct(self):
        """CH3 with multiplicity=2 is correct."""
        atoms = ["C", "H", "H", "H"]
        issues = inspect_charge_spin(atoms, charge=0, multiplicity=2)
        assert issues == []

    def test_anion_even_singlet_correct(self):
        """Water anion [H2O]⁻ has 11 electrons → odd → doublet; singlet should flag."""
        atoms, _ = _water()  # 10 electrons neutral
        issues = inspect_charge_spin(atoms, charge=-1, multiplicity=1)
        assert len(issues) >= 1

    def test_cation_even_singlet_correct(self):
        """CH4 cation [CH4]⁺ has 9 electrons → odd → singlet should flag."""
        atoms, _ = _methane()  # 10 electrons neutral; cation = 9
        issues = inspect_charge_spin(atoms, charge=+1, multiplicity=1)
        assert len(issues) >= 1

# ── Section F: QM hazards ─────────────────────────────────────────────────────

class TestInspectQMHazards:
    def test_clean_methane_no_hazards(self):
        atoms, coords = _methane()
        bonds = infer_bond_list(atoms, coords)
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        assert hazards == []

    def test_co2_near_linear_flagged(self):
        """CO2 O–C–O angle is 180° → F1 hazard."""
        atoms, coords = _co2()
        bonds = infer_bond_list(atoms, coords)
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f1 = [h for h in hazards if h[0] == "F1"]
        assert len(f1) >= 1

    def test_duplicate_coordinates_flagged(self):
        """Two atoms at the same position → F2 hazard."""
        atoms = ["C", "C", "H", "H", "H", "H"]
        coords = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),  # duplicate of atom 0
            (1.09, 0.0, 0.0),
            (-1.09, 0.0, 0.0),
            (0.0, 1.09, 0.0),
            (0.0, -1.09, 0.0),
        ]
        bonds = [(0, 2), (0, 3), (0, 4), (0, 5)]
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f2 = [h for h in hazards if h[0] == "F2"]
        assert len(f2) >= 1
        assert 0 in f2[0][2] and 1 in f2[0][2]

    def test_disconnected_graph_flagged(self):
        """Two separate water molecules with no bonds between them → F6 hazard."""
        atoms = ["O", "H", "H", "O", "H", "H"]
        coords = [
            (0.0,  0.0, 0.0),
            (0.96, 0.0, 0.0),
            (-0.24, 0.93, 0.0),
            (5.0,  0.0, 0.0),  # far away
            (5.96, 0.0, 0.0),
            (4.76, 0.93, 0.0),
        ]
        bonds = [(0, 1), (0, 2), (3, 4), (3, 5)]
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f6 = [h for h in hazards if h[0] == "F6"]
        assert len(f6) == 1

    def test_missing_h_flagged(self):
        """A carbon with only 1 bond (needs ~4) → F4 hazard."""
        atoms = ["C", "H"]
        coords = [(0.0, 0.0, 0.0), (1.09, 0.0, 0.0)]
        bonds = [(0, 1)]
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f4 = [h for h in hazards if h[0] == "F4"]
        assert len(f4) >= 1

    def test_metal_wrong_cn_flagged(self):
        """Cu with only 2 bonds (expected 4–6) → F5 hazard."""
        atoms = ["Cu", "N", "N"]
        coords = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)]
        bonds = [(0, 1), (0, 2)]
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f5 = [h for h in hazards if h[0] == "F5"]
        assert len(f5) >= 1

    def test_cu_complex_correct_cn_no_f5(self):
        """Cu with 4 N bonds (within 4–6 range) → no F5 hazard."""
        atoms, coords = _cu_complex()
        bonds = infer_bond_list(atoms, coords)
        hazards = inspect_qm_hazards(atoms, coords, bonds)
        f5 = [h for h in hazards if h[0] == "F5"]
        assert f5 == []

# ── Top-level inspect_geometry ────────────────────────────────────────────────

class TestInspectGeometry:
    def test_clean_methane_report(self):
        atoms, coords = _methane()
        report = inspect_geometry(atoms, coords, charge=0, multiplicity=1)
        assert isinstance(report, InspectionReport)
        assert report.n_atoms == 5
        assert report.n_bonds == 4
        assert report.is_clean()
        assert report.summary() == "clean"

    def test_clean_water_report(self):
        atoms, coords = _water()
        report = inspect_geometry(atoms, coords, charge=0, multiplicity=1)
        assert report.is_clean()

    def test_explicit_bond_list_skips_inference(self):
        """With explicit bond list (SDF/SMILES path), Section 0 is not run."""
        atoms, coords = _methane()
        explicit_bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
        report = inspect_geometry(atoms, coords, bond_list=explicit_bonds)
        assert report.n_bonds == 4
        assert report.is_clean()

    def test_summary_reports_issues(self):
        """A molecule with clashes and charge issues should have a non-clean summary."""
        # Two H atoms very close (clash) + odd electrons with singlet
        atoms = ["C", "H", "H", "H"]  # 9 electrons, singlet is wrong
        coords = [
            (0.0, 0.0, 0.0),
            (1.09, 0.0, 0.0),
            (1.09 + 0.8, 0.0, 0.0),   # H–H at 0.8 Å = clash
            (0.0, 1.09, 0.0),
        ]
        bonds = [(0, 1), (0, 3)]  # atom 2 is unconnected and close to atom 1
        report = inspect_geometry(atoms, coords, charge=0, multiplicity=1,
                                  bond_list=bonds)
        assert not report.is_clean()
        assert "clash" in report.summary() or "charge" in report.summary()

    def test_cu_complex_clean(self):
        atoms, coords = _cu_complex()
        report = inspect_geometry(atoms, coords, charge=2, multiplicity=2)
        # Expect no bond, clash, torsion, or angle issues for this idealised geometry
        assert report.bond_issues == []
        assert report.clash_issues == []
        assert report.torsion_issues == []
