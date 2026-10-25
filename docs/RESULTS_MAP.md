# Results map

Which script produces which number in the manuscript. All aggregate outputs listed here are
in `results/`.

## Original pipeline

| Script | Purpose | Main outputs |
|---|---|---|
| `01_extract_corrected_cohorts.py` | Rebuilds the harmonised MIMIC-IV and eICU-CRD surgical cohorts: age from anchor year, active surgical service at ICU entry, surgical ICU, eICU `unitadmitsource`, exclusion of unknown hospital outcome, first eligible stay per hospitalisation | `derived_rewarming_v3/*_rewarming_cohort.csv.gz` (local only), extraction summary text |
| `02_run_primary_analysis.py` | Core association models, cohort descriptives, trajectory profiles, forest plot, quintiles | `Table0_cohort_and_AUC_scaling.csv`, `Table1_descriptive_by_trajectory.csv`, `Table2_adjusted_models.csv`, `Table3_sensitivity_analyses.csv`, figures |
| `03_run_enhanced_analyses.py` | Natural cubic spline, nonlinearity test, eICU hospital-conditional model | `Table_Enhanced_AUC_Models.csv`, `Table_Nonlinearity_Tests.csv`, `Table_eICU_Hospital_Conditional_Model.csv` |
| `04_run_severity_adjustment.py` | Database-specific severity adjustment (MIMIC-IV early MAP/GCS/lactate/creatinine; eICU APACHE IVa and Acute Physiology Score) | `Table_severity_adjusted_models.csv`, `Table_severity_coverage.csv` |
| `05_complete_tables.py` | Descriptive tables by survival status and AUC quintile | `Table_V3_By_Mortality.csv`, `Table_V3_AUC_Quintiles.csv`, `Table_V3_Quintile_Linearity_Diagnostic.csv` |

## Reviewer-requested analyses

| Script | Manuscript element | Output |
|---|---|---|
| `R01_core_association_and_subgroups.py` | Core and per-1 °C·h ORs, alternative-interpolation models, subgroup strata, out-of-fold prediction | `association_and_subgroups.csv`, `incremental_prediction_core_model.csv`, `incremental_prediction_severity_models.csv` |
| `R02_extract_severity.py` | Table 2 severity rows, Figure 3 | `Table_severity_adjusted_models.csv`, `Table_severity_coverage.csv` |
| `R03_transfusion_sensitivity.py` | Blood-product sensitivity (Table 2 text, Supplementary Table S5) | `transfusion_sensitivity.csv` |
| `R04_vasopressor_sensitivity.py` | Vasoactive-infusion sensitivity and interaction | `vasopressor_adjusted_models.csv`, `vasopressor_subgroup_interaction.csv` |
| `R05_subgroup_interactions.py` | Age, emergency admission, cardiothoracic/vascular, sparse-sampling interactions (Supplementary Table S4) | `subgroup_interactions_corrected.csv` |
| `R06_reclassification_idi_nri.py` | Out-of-fold AUROC, Brier, IDI, continuous NRI (Supplementary Table S4) | `reclassification_idi_continuous_nri.csv` |
| `R07_mimic_warming_sensitivity.py` | MIMIC-IV active warming-device sensitivity | `mimic_warming_sensitivity.csv` |
| `R08_aki_strict_timing.py` | Post-landmark creatinine-defined AKI with ordered 48-h / 7-day comparisons and ICU-discharge censoring | `incident_aki_models.csv`, `aki_event_definition_audit.csv` |
| `R09_eicu_elective_surgery.py` | eICU APACHE elective vs non-elective surgery interaction (Supplementary Table S4) | `eicu_elective_surgery_interaction.csv` |
| `R10_mimic_surg_emergency_scan.py` | Audit of MIMIC-IV APACHE surgical-emergency chart items (all empty in the study cohort) | console output only |
| `R11_eicu_variable_feasibility.py` | Coverage audit of eICU source fields | `eicu_fresh_feasibility_aggregate.json` |
| `F01`–`F04` | Field-availability audits for MIMIC chart items and eICU intake/output, nursing and rhythm tables | local audit CSVs (not tracked) |

## Manuscript element → file

| Manuscript element | Source file |
|---|---|
| Table 1 (cohort characteristics) | `Table1_descriptive_by_trajectory.csv` (step 02) consolidated with `Table_V3_By_Mortality.csv` (step 05) |
| Table 2 (core and severity-adjusted ORs) | `association_and_subgroups.csv` (R01) and `Table_severity_adjusted_models.csv` (R02) |
| Table 3 (secondary metrics) | `Table_V3_AUC_Quintiles.csv`, `Table_Nonlinearity_Tests.csv`, `association_and_subgroups.csv` |
| Figure 1 (cohort flow) | counts from the step 01 extraction summary |
| Figure 2 (mortality by quintile) | `Table_V3_AUC_Quintiles.csv` (step 05) |
| Figure 3 (adjusted forest plot) | `Table_severity_adjusted_models.csv` (R02) |
| Supplementary Table S1 | `Table_V3_By_Mortality.csv` (step 05) |
| Supplementary Table S2 | `Table3_sensitivity_analyses.csv` (step 02) |
| Supplementary Table S3 | `Table_Nonlinearity_Tests.csv`, `Table_eICU_Hospital_Conditional_Model.csv` (step 03) |
| Supplementary Table S4 | `interpolation_auc_summary.csv`, `subgroup_interactions_corrected.csv`, `eicu_elective_surgery_interaction.csv`, `reclassification_idi_continuous_nri.csv`, `incremental_prediction_*.csv` |
| Supplementary Table S5 | `transfusion_sensitivity.csv`, `vasopressor_adjusted_models.csv`, `vasopressor_subgroup_interaction.csv`, `mimic_warming_sensitivity.csv`, `incident_aki_models.csv`, `aki_event_definition_audit.csv` |
| Supplementary Figure S1 | `Figure1_trajectory_profiles.png` (step 02) |
| Supplementary Figure S2 | `Figure_Spline_AUC_Mortality.png` (step 03) |

## Note on the pooled estimate

`Table4_fixed_effect_summary.csv` (fixed-effect pooled OR) is intentionally not archived: the pooled estimate was removed from the manuscript during revision, and the numbers are no longer reported.

## Known gaps

* `interpolation_auc_summary.csv` is archived as a result table; the short helper that
  summarised the paired per-patient linear and zero-order-hold AUC values was not kept as a
  standalone script. The underlying model estimates are reproducible with
  `R01_core_association_and_subgroups.py`.
* Scripts that assembled the manuscript and response-letter Word files are not included;
  they operate on the outputs above and contain no analysis logic.
