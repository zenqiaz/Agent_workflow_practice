# Test Criteria — test_success_rate.py

Each test run applies two layers of checks:
1. **Structural checks** (`check_struct`) — applied to every plan regardless of task type
2. **Task-specific plan checks** — verify the plan is chemically correct for the task
3. **Execution result checks** (full mode only) — verify artifacts produced by ORCA are physically reasonable

A run PASSES only if all checks in all layers return True and no exception was raised.

---

## Structural Checks (all tasks)

Applied to every plan. These verify the JSON is a valid plan skeleton.

| Check | What it tests |
|-------|--------------|
| `is_dict` | Plan is a JSON object (not a list or string) |
| `has_name` | Plan has a non-empty `name` field |
| `has_nodes` | Plan contains at least one node |
| `has_artifacts` | `artifacts_to_save` list is non-empty |
| `nodes_have_id` | Every node has a non-empty `id` field |
| `nodes_have_kind` | Every node has a `kind` field (`tool`, `llm`, or `calc_expr`) |
| `tools_valid` | Every tool node references a known tool name |

---

## Task-Specific Plan Checks

### `pka` — pKa Calculation

| Check | What it tests |
|-------|--------------|
| `uses_freq_not_sp` | Uses `run_freq_job`; does NOT use `run_sp_energy` (SP lacks zero-point corrections) |
| `has_h_plus_ref` | `G_H_plus_ref_eh` key is present in plan `settings` |
| `artifacts_G_eh` | At least one `G_*_eh` Gibbs energy artifact in `artifacts_to_save` |
| `pka_in_artifacts` | At least one artifact name contains "pka" |
| `llm_node_exists` | At least one LLM node exists (to compute pKa from ΔG) |
| `freq_arts_in_needs` | LLM node's `needs_artifacts` includes at least one `G_*_eh` key |

### `sp` — Single-Point Energy

| Check | What it tests |
|-------|--------------|
| `has_sp_node` | Uses `run_sp_energy` |
| `no_freq_job` | Does NOT use `run_freq_job` (would be wasteful/wrong for a pure SP task) |
| `has_energy_art` | At least one energy artifact (`E_*`, `*_eh`, or containing "energy") in plan |

### `nbo` — NBO Analysis

| Check | What it tests |
|-------|--------------|
| `has_opt_node` | Uses `run_opt_job` (geometry must be optimized before NBO) |
| `has_nbo_node` | Uses `run_nbo_job` |
| `nbo_after_opt` | `run_nbo_job` appears after `run_opt_job` in node order |
| `nbo_artifact` | At least one artifact key contains "nbo" |

### `solvation` — Microsolvation

| Check | What it tests |
|-------|--------------|
| `has_solvator` | Uses `run_solvator_cluster_thermo` or `run_solvator_cluster` |
| `nsolv_arg_set` | At least one node passes `nsolv` in its args (number of solvent molecules) |
| `uses_freq` | Uses `run_freq_job` or `run_solvator_cluster_thermo` (thermochemistry required) |

### `protonation` — Protonation Site Comparison

| Check | What it tests |
|-------|--------------|
| `has_sp_node` | Uses `run_sp_energy` (SP energies for each protonated site) |
| `has_proton_tool` | Uses `structure_add_remove_proton` (to generate each protonated isomer) |
| `multiple_sp_nodes` | At least 2 SP nodes (one per protonation site) |
| `has_energy_artifacts` | At least one energy artifact in plan |
| `has_comparison` | Plan has a comparison artifact or an LLM node to pick the preferred site |

### `spectrum` — IR Spectrum

| Check | What it tests |
|-------|--------------|
| `has_spectrum_node` | Uses `run_spectrum_job` (not `run_freq_job` — spectrum tool gives IR output) |
| `no_plain_freq` | Does NOT use `run_freq_job` (wrong tool for IR spectrum tasks) |
| `has_opt_node` | Uses `run_opt_job` (geometry must be optimized first) |
| `spectrum_after_opt` | `run_spectrum_job` appears after `run_opt_job` in node order |
| `ir_spectrum_art` | At least one artifact key contains "ir_spectrum" |
| `no_llm_node` | No LLM nodes (spectrum is raw data; no derived quantity needed) |

### `tddft` — UV-Vis / TD-DFT

| Check | What it tests |
|-------|--------------|
| `has_tddft_node` | Uses `run_tddft_job` |
| `has_opt_node` | Uses `run_opt_job` (optimized geometry required) |
| `tddft_after_opt` | `run_tddft_job` appears after `run_opt_job` in node order |
| `excited_states_art` | At least one artifact key contains "excited_states" |
| `no_llm_node` | No LLM nodes (excited states are raw data) |

> When geometries are pre-loaded (`--preload`), `has_opt_node`, `tddft_after_opt`, and `no_llm_node` are relaxed (a comparison LLM node is allowed).

### `scan` — PES Scan

| Check | What it tests |
|-------|--------------|
| `has_scan_node` | Uses `run_scan_job` |
| `scan_results_art` | At least one artifact key contains "scan_results" |
| `no_llm_node` | No LLM nodes (scan is raw data; PES is inspected visually) |

### `ts` — Transition State Search

| Check | What it tests |
|-------|--------------|
| `has_scan_node` | Uses `run_scan_job` (scan to locate TS candidate) |
| `has_ts_opt_node` | Uses `run_ts_opt_job` (OptTS refinement) |
| `has_freq_node` | Uses `run_freq_job` or `run_spectrum_job` (verify exactly 1 imaginary frequency) |
| `ts_opt_after_scan` | `run_ts_opt_job` appears after `run_scan_job` in node order |
| `freq_after_ts_opt` | Frequency node appears after `run_ts_opt_job` in node order |
| `no_llm_node` | No LLM nodes |

### `casscf` — CASSCF Multi-Reference

| Check | What it tests |
|-------|--------------|
| `has_casscf_node` | Uses `run_casscf_job` |
| `has_active_space` | Every CASSCF node specifies both `nel` (electrons) and `norb` (orbitals) in args |
| `has_energy_artifact` | At least one energy artifact in plan |

### `eas` — Electrophilic Aromatic Substitution Reactivity

| Check | What it tests |
|-------|--------------|
| `has_opt_node` | Uses `run_opt_job` (optimized geometry for charge analysis) |
| `has_charge_calc` | Uses `run_sp_energy` (Mulliken charges) or `run_nbo_job` (NPA charges) |
| `has_ranking_llm` | At least one LLM node (to rank sites by charge) |
| `charge_artifact` | At least one artifact key contains "npa", "nbo", "charge", "fukui", or "mulliken" |
| `ranking_artifact` | At least one artifact key contains "ranking", "site", "eas", or "fukui" |

### `interaction_scan` — Intermolecular Interaction Scan

| Check | What it tests |
|-------|--------------|
| `has_dimer_build` | Uses `build_dimer_xyz` (constructs dimer geometry) |
| `has_sp_monomer` | Uses `run_sp_energy` (monomer energies for ΔE_int) |
| `has_scan_node` | Uses `run_scan_job` (approach curve) |
| `has_delta_e_llm` | Exactly one LLM node (to compute ΔE_int from SP energies) |
| `interaction_curve_art` | At least one artifact key contains "interaction_curve" or "binding_energy" |
| `scan_results_art` | At least one artifact key contains "scan_results" |

### `coordination_sp` — Coordination Complex SP

| Check | What it tests |
|-------|--------------|
| `has_build_node` | Uses `build_coordination_complex` |
| `has_opt_node` | Uses `run_opt_job` |
| `has_sp_node` | Uses `run_sp_energy` |
| `build_before_opt` | Build node appears before opt node |
| `opt_before_sp` | Opt node appears before SP node |
| `all_built_are_opted` | Every complex built by `build_coordination_complex` feeds into an opt node |
| `has_energy_art` | At least one `*energy*_eh` artifact |
| `has_homo_lumo_art` | At least one HOMO/LUMO/gap artifact |
| `has_dipole_art` | At least one dipole artifact |
| `charge_<compound>` | Charge in build args matches expected value (for known test compounds) |
| `spin_<compound>` | Multiplicity in build args matches expected value (for known test compounds) |

---

## Execution Result Checks (full mode only)

These run after ORCA jobs complete and check that the artifacts contain physically reasonable values.

### `pka` result

| Check | What it tests |
|-------|--------------|
| `pka_present` | At least one `pka` or `pka_*` artifact exists |
| `pka_reasonable` | At least one pKa value is in the range −20 to 60 (calibrated gas-phase) |
| `G_eh_present` | At least 2 `G_*_eh` Gibbs energy artifacts exist |
| `G_negative` | All `G_*_eh` values are negative (correct sign for absolute DFT energies in Hartree) |

### `sp` result

| Check | What it tests |
|-------|--------------|
| `energy_present` | At least one energy artifact exists |
| `energy_negative` | All energy values are negative (correct sign for DFT total energies) |

### `spectrum` result

| Check | What it tests |
|-------|--------------|
| `ir_spectrum_present` | IR spectrum artifact is a non-empty list |
| `has_positive_freqs` | At least one mode with positive frequency (real vibration) |
| `has_intensity_field` | Each mode entry contains `intensity_km_mol` field |

### `tddft` result

| Check | What it tests |
|-------|--------------|
| `excited_states_present` | Excited states artifact is a non-empty list |
| `has_energy_ev` | Each state entry contains `energy_ev` |
| `has_wavelength_nm` | Each state entry contains `wavelength_nm` |
| `has_oscillator_strength` | Each state entry contains `oscillator_strength` |

### `scan` result

| Check | What it tests |
|-------|--------------|
| `scan_results_present` | Scan results artifact is a non-empty list |
| `has_step` | Each scan point entry contains `step` |
| `has_value` | Each scan point entry contains `value` (coordinate value) |
| `has_energy_eh` | Each scan point entry contains `energy_eh` |

### `interaction_scan` result

| Check | What it tests |
|-------|--------------|
| `interaction_curve_present` | Interaction curve artifact is a non-empty list |
| `has_distance` | Each curve point contains `distance_ang` or `value` |
| `has_delta_e` | Each curve point contains `delta_e_int_kcal` |
| `binding_energy_present` | A `binding_energy_kcal` artifact exists |

---

## Common Checks Added During Execution (full mode)

| Check | What it tests |
|-------|--------------|
| `execution_ok` | LangGraph finished with `last_status == "ok"` (no node failed fatally) |
| `exception` | Set to `False` if any unhandled Python exception occurred during planning or execution |

---

## Checker Auto-Detection

When `--checker auto` (default), the checker is selected by keyword matching on the message:

| Checker | Trigger keywords |
|---------|-----------------|
| `pka` | pka, acidity, deprotonation, acid dissociation |
| `protonation` | protonation, protonization, protonation site |
| `sp` | single-point, single point, sp energy |
| `nbo` | nbo, natural bond, wiberg |
| `solvation` | solvat, aqueous, microsolvat, nsolv, in water |
| `spectrum` | ir spectrum, infrared, raman, vibrational spectrum |
| `tddft` | tddft, td-dft, excited state, uv-vis, absorption spectrum |
| `scan` | scan, pes scan, potential energy surface, dihedral scan |
| `ts` | transition state, ts search, saddle point, activation barrier |
| `casscf` | casscf, active space, multireference, sa-casscf |
| `interaction_scan` | interaction energy, binding energy scan, pi stacking, dimer scan |
| `eas` | eas, electrophilic aromatic, site selectivity, ortho para |
| `coordination_sp` | coordination compound, metal complex, hexacyanide |
| `struct` | *(fallback — no keyword matched)* |
