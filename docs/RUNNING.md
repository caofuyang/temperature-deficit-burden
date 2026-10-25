# Running the analysis

## 1. Get the data

Download MIMIC-IV v3.1 and eICU-CRD v2.0 from PhysioNet after completing the required
training and signing the data-use agreements. Extract them so that:

```
<MIMIC_ROOT>/hosp/...
<MIMIC_ROOT>/icu/...
<EICU_ROOT>/patient.csv.gz
<EICU_ROOT>/apachePredVar.csv.gz
...
```

## 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 3. Point the code at the data

```bash
export MIMIC_ROOT=/data/mimic-iv-3.1        # Windows: $env:MIMIC_ROOT = "D:\data\mimic-iv-3.1"
export EICU_ROOT=/data/eicu-crd-2.0
export REVISION_SOURCE=/data/work/revision_source
python analysis/config.py                   # verifies that each path exists
```

## 4. Run

```bash
python analysis/01_extract_corrected_cohorts.py
python analysis/02_run_primary_analysis.py
python analysis/03_run_enhanced_analyses.py
python analysis/04_run_severity_adjustment.py
python analysis/05_complete_tables.py
```

Steps 02–05 prompt for the folder that step 01 wrote (`REVISION_SOURCE`); pressing Enter
accepts the configured default.

The reviewer-requested scripts (`R01`–`R11`) can be run in any order after step 01, except:

* `R01` must run before `R06`, because `R06` reuses its out-of-fold predictions;
* `R03`, `R04` and `R07` read the raw event tables as well as the derived cohort files and
  take 1–3 hours each on a full MIMIC-IV extract because `inputevents` and `chartevents` are
  scanned in chunks.

## 5. Check the outputs

Aggregate tables land in `RESULTS_DIR` (default `results/`). Compare them against the
archived copies in this repository; the values in the manuscript were taken from those files.

## Data protection

`*_internal.csv`, `*_cohort.csv.gz` and the `derived_rewarming_v3/` folder contain
patient-level rows. They are excluded by `.gitignore` and must never be pushed, uploaded or
shared. Only aggregate tables belong in a public repository.
