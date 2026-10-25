# Early Temperature-Deficit Burden and Mortality in Hypothermic Surgical ICU Patients

Analysis code and aggregate result tables for the manuscript

> **Early Temperature-Deficit Burden and In-Hospital Mortality Among Hypothermic Surgical ICU
> Patients: A Two-Database Cohort Study**

(MIMIC-IV v3.1, 1,422 patients; eICU-CRD v2.0, 2,592 patients; 6-hour landmark design;
exposure = area below 36 °C during hours 0–6).

This repository contains **code and aggregate tables only**. It contains no patient-level
records. MIMIC-IV and eICU-CRD are available from PhysioNet to credentialed users who
complete the required training and sign the data-use agreements; they cannot be redistributed.

## Contents

```
analysis/    Python scripts (original pipeline 01–05, reviewer-requested analyses R01–R11,
             variable-feasibility scans F01–F04, and config.py)
results/     Aggregate CSV/JSON outputs that back the tables and figures in the manuscript
docs/        RESULTS_MAP.md (manuscript element -> script -> output file) and RUNNING.md
```

## Data sources

| Database | Version | Access |
|---|---|---|
| MIMIC-IV | v3.1 | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD | v2.0 | https://physionet.org/content/eicu-crd/2.0/ |

Both require completion of the CITI "Data or Specimens Only Research" training and a signed
data-use agreement.

## Environment

Python 3.10 or newer. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

No path is hard-coded. Every script reads its locations from `analysis/config.py`, which
takes them from environment variables:

| Variable | Meaning |
|---|---|
| `MIMIC_ROOT` | folder containing the MIMIC-IV `hosp/` and `icu/` directories |
| `EICU_ROOT` | folder containing `patient.csv.gz`, `apachePredVar.csv.gz`, ... |
| `REVISION_SOURCE` | folder holding the derived cohort files written by step 01 |
| `RESULTS_DIR` | output folder for aggregate tables (defaults to `results/`) |

Run `python analysis/config.py` to print the active paths and check that they exist.

## Running the pipeline

```bash
python analysis/01_extract_corrected_cohorts.py     # builds the two cohorts from the raw databases
python analysis/02_run_primary_analysis.py          # primary models, descriptive tables, trajectory figures
python analysis/03_run_enhanced_analyses.py         # spline, quintile and hospital-conditional analyses
python analysis/04_run_severity_adjustment.py       # database-specific severity adjustment
python analysis/05_complete_tables.py               # manuscript tables by survival status and quintile
```

Steps 02–05 read the patient-level files that step 01 writes to
`REVISION_SOURCE` (`mimic_rewarming_cohort.csv.gz`, `eicu_rewarming_cohort.csv.gz`). Those
files must stay on the local machine and are excluded by `.gitignore`.

The scripts prefixed `R` implement the analyses added during peer review (severity
comparison, transfusion / vasoactive / warming sensitivities, subgroup interactions,
alternative interpolation, AKI with strict event timing, incremental discrimination, and
the eICU elective-surgery interaction). The scripts prefixed `F` are variable-availability
audits that document which source fields could and could not be harmonised; they write
patient-level audit files that are deliberately not tracked.

## Reproducibility notes

* `analysis/config.py` centralises all input and output locations.
* Results in `results/` are the exact files from which the manuscript numbers were taken.
* Patient-level intermediates (`*_internal.csv`, `derived_rewarming_v3/`) are generated
  locally and are never committed.
* Some console messages in the original pipeline scripts are in Chinese; the code and all
  output files are in English.

## Manuscript mapping

See [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) for the file-by-file mapping from each
manuscript table and figure to the script that produces it.

## Licence

MIT. See `LICENSE`.

## Citation

If you use this code, please cite the article and this archived release. Machine-readable
metadata is in `CITATION.cff`.

## Contact

Fuyang Cao, Department of Anesthesiology, The Sixth Medical Center of the Chinese PLA
General Hospital, Beijing, China — caofuyang840723@163.com
