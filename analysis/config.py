"""Path configuration for the temperature-deficit burden analysis.

Every script imports this module and takes its input/output locations from
here, so no absolute path is hard-coded anywhere in the repository.

Set the locations with environment variables (all optional). Example
PowerShell session:

    $env:MIMIC_ROOT       = "D:\\physionet\\mimic-iv-3.1"
    $env:EICU_ROOT        = "D:\\physionet\\eicu-crd-2.0"
    $env:REVISION_SOURCE  = "D:\\work\\revision_source"
    python analysis\\02_run_primary_analysis.py

or Bash:

    export MIMIC_ROOT=/data/mimic-iv-3.1
    export EICU_ROOT=/data/eicu-crd-2.0
    python analysis/02_run_primary_analysis.py

If a variable is not set, the default is a folder inside this repository
(`data/`, `revision_source/`, `results/`). Those folders are not tracked by
git and must never contain identifiable data.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip().strip('"').strip("'")
    if not raw:
        return default
    return Path(raw).expanduser()


#: Root of the MIMIC-IV v3.1 extract (the folder that contains hosp/ and icu/).
MIMIC_ROOT = _env_path("MIMIC_ROOT", REPO_ROOT / "data" / "mimic-iv-3.1")

#: Root of the eICU-CRD v2.0 extract (the folder that contains patient.csv.gz ...).
EICU_ROOT = _env_path("EICU_ROOT", REPO_ROOT / "data" / "eicu-crd-2.0")

#: Folder holding the derived per-patient cohort files written by
#: 01_extract_corrected_cohorts.py (previously named derived_rewarming_v3).
REVISION_SOURCE = _env_path("REVISION_SOURCE", REPO_ROOT / "revision_source")

#: Where aggregate result tables are written. Only aggregate files belong here.
RESULTS_DIR = _env_path("RESULTS_DIR", REPO_ROOT / "results")


def describe() -> str:
    """Return a human-readable summary of the active paths."""
    lines = [
        f"repository root : {REPO_ROOT}",
        f"MIMIC_ROOT      : {MIMIC_ROOT}  (exists: {MIMIC_ROOT.exists()})",
        f"EICU_ROOT       : {EICU_ROOT}  (exists: {EICU_ROOT.exists()})",
        f"REVISION_SOURCE : {REVISION_SOURCE}  (exists: {REVISION_SOURCE.exists()})",
        f"RESULTS_DIR     : {RESULTS_DIR}  (exists: {RESULTS_DIR.exists()})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
