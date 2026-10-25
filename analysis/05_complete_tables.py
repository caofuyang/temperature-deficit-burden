# -*- coding: utf-8 -*-
"""Create V3 quintile models and mortality-stratified descriptive summaries."""
from __future__ import annotations

import math
import sys
import traceback
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "v3_completion_results"
ERROR = HERE / "V3_completion_error.txt"

try:
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    from scipy.stats import chi2, norm
except ImportError as exc:
    ERROR.write_text(
        f"Missing package: {exc}\nRun: py -3 -m pip install pandas numpy scipy\n",
        encoding="utf-8-sig",
    )
    print(ERROR.read_text(encoding="utf-8-sig"))
    input("Press Enter to close...")
    sys.exit(1)


def find_derived():
    choices = [HERE / "derived_rewarming_v3", HERE.parent / "derived_rewarming_v3"]
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        try:
            choices.extend(downloads.rglob("derived_rewarming_v3"))
        except OSError:
            pass
    valid = [
        p for p in choices
        if (p / "mimic_rewarming_cohort.csv.gz").exists()
        and (p / "eicu_rewarming_cohort.csv.gz").exists()
    ]
    return max(valid, key=lambda p: (p / "mimic_rewarming_cohort.csv.gz").stat().st_mtime) if valid else None


def broad_surgery(value):
    text = str(value).lower()
    if any(x in text for x in ("cardiac", "cabg", "coronary", "valve", "aortic", "vascular", "cardiothoracic")):
        return "cardiothoracic_vascular"
    if any(x in text for x in ("neuro", "crani", "brain", "subdural", "spine")):
        return "neurosurgery"
    if any(x in text for x in ("trauma", "fracture", "orthop", "hip", "femur")):
        return "orthopedic_trauma"
    if any(x in text for x in ("abdom", "bowel", "colon", "gastric", "hepatic", "liver", "pancrea", "general")):
        return "general_abdominal"
    return "other_surgery"


def broad_icu(value):
    text = str(value).lower()
    if any(x in text for x in ("cardiac", "cticu", "csicu", "ccu", "cvicu")):
        return "cardiac_icu"
    if any(x in text for x in ("sicu", "surgical", "trauma", "vic")):
        return "surgical_icu"
    if "neuro" in text:
        return "neuro_icu"
    if any(x in text for x in ("micu", "medical")):
        return "medical_icu"
    return "other_icu"


def zscore(series):
    sd = series.std(ddof=0)
    return (series - series.mean()) / (sd if pd.notna(sd) and sd > 0 else 1.0)


def prepare(frame):
    d = frame.copy()
    numeric = [
        "age", "first_temperature_c", "n_temperature",
        "temperature_deficit_auc_c_hours", "rewarming_slope_c_per_hour",
        "icu_los_days", "hospital_los_days", "hospital_mortality",
    ]
    for column in numeric:
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d["age"] = d["age"].fillna(d["age"].median())
    d["sex_male"] = d["sex"].astype(str).str.lower().str.startswith("m").astype(int)
    d["emergency_admission"] = pd.to_numeric(d["emergency_admission"], errors="coerce").fillna(0).astype(int)
    d["hospital_mortality"] = d["hospital_mortality"].astype(int)
    d["hospital_id"] = d["hospital_id"].astype(str)
    d["surgery_broad"] = d["surgery_group"].map(broad_surgery)
    d["icu_broad"] = d["icu_type"].map(broad_icu)
    for column in ("age", "first_temperature_c", "n_temperature", "temperature_deficit_auc_c_hours"):
        d[column + "_z"] = zscore(d[column])
    return d


def design(d, exposures):
    base = ["age_z", "sex_male", "emergency_admission", "first_temperature_c_z", "n_temperature_z"]
    x = d[list(exposures) + base].astype(float).copy()
    categories = pd.get_dummies(
        d[["surgery_broad", "icu_broad"]].fillna("Unknown"),
        drop_first=True,
        dtype=float,
    )
    x = pd.concat([x, categories], axis=1)
    x.insert(0, "Intercept", 1.0)
    return x


def logistic(x, y, clusters=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, p = x.shape
    beta = np.zeros(p)
    for _ in range(100):
        probability = expit(np.clip(x @ beta, -30, 30))
        weight = np.clip(probability * (1 - probability), 1e-8, None)
        hessian = x.T @ (x * weight[:, None]) + np.eye(p) * 1e-8
        step = np.linalg.pinv(hessian) @ (x.T @ (y - probability))
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    probability = expit(np.clip(x @ beta, -30, 30))
    weight = np.clip(probability * (1 - probability), 1e-8, None)
    bread = np.linalg.pinv(x.T @ (x * weight[:, None]))
    residual = y - probability
    if clusters is not None and len(set(clusters)) > 1:
        groups = np.asarray(clusters)
        meat = np.zeros((p, p))
        for group in np.unique(groups):
            score = x[groups == group].T @ residual[groups == group]
            meat += np.outer(score, score)
        count = len(np.unique(groups))
        meat *= count / (count - 1) * (n - 1) / max(n - p, 1)
    else:
        scores = x * residual[:, None]
        meat = scores.T @ scores * n / max(n - p, 1)
    covariance = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    loglik = float(np.sum(
        y * np.log(np.clip(probability, 1e-12, 1))
        + (1 - y) * np.log(np.clip(1 - probability, 1e-12, 1))
    ))
    return beta, se, loglik


def fit(d, database, exposures):
    required = list(exposures) + [
        "hospital_mortality", "age_z", "sex_male", "emergency_admission",
        "first_temperature_c_z", "n_temperature_z",
    ]
    q = d.dropna(subset=required).copy()
    x = design(q, exposures)
    groups = q["hospital_id"].to_numpy() if database == "eICU" else None
    beta, se, loglik = logistic(x, q["hospital_mortality"], groups)
    return q, x, beta, se, loglik


def quintile_results(d, database):
    d = d.copy()
    d["auc_quintile"] = pd.qcut(
        d["temperature_deficit_auc_c_hours"], 5, labels=False, duplicates="drop"
    ) + 1
    dummy = pd.get_dummies(d["auc_quintile"], prefix="AUC_Q", dtype=float)
    d = pd.concat([d, dummy], axis=1)
    available_quintiles = sorted(int(x) for x in d["auc_quintile"].dropna().unique())
    if len(available_quintiles) < 2:
        raise RuntimeError(f"{database}: fewer than two distinct AUC quantile groups")
    reference_quintile = available_quintiles[0]
    exposures = [f"AUC_Q_{q}" for q in available_quintiles if q != reference_quintile]
    q, x, beta, se, categorical_loglik = fit(d, database, exposures)
    names = list(x.columns)
    rows = []
    counts = d.groupby("auc_quintile")["hospital_mortality"].agg(["size", "sum", "mean"])
    for quintile in available_quintiles:
        observed = counts.loc[quintile]
        if quintile == reference_quintile:
            rows.append({
                "database": database, "quintile": quintile,
                "n": int(observed["size"]), "deaths": int(observed["sum"]),
                "mortality_percent": 100 * observed["mean"],
                "odds_ratio": 1.0, "ci_lower": 1.0, "ci_upper": 1.0,
                "p_value": np.nan, "reference": 1,
            })
            continue
        column = f"AUC_Q_{quintile}"
        index = names.index(column)
        b, s = beta[index], se[index]
        rows.append({
            "database": database, "quintile": quintile,
            "n": int(observed["size"]), "deaths": int(observed["sum"]),
            "mortality_percent": 100 * observed["mean"],
            "odds_ratio": math.exp(b), "ci_lower": math.exp(b - 1.96 * s),
            "ci_upper": math.exp(b + 1.96 * s),
            "p_value": 2 * norm.sf(abs(b / s)), "reference": 0,
        })

    _, _, _, _, linear_loglik = fit(d, database, ["temperature_deficit_auc_c_hours_z"])
    _, _, _, _, augmented_loglik = fit(
        d, database, ["temperature_deficit_auc_c_hours_z"] + exposures
    )
    lr = max(0.0, 2 * (augmented_loglik - linear_loglik))
    test = {
        "database": database,
        "linear_loglik": linear_loglik,
        "categorical_loglik": categorical_loglik,
        "augmented_loglik": augmented_loglik,
        "lr_statistic": lr,
        "df": len(exposures),
        "p_value": chi2.sf(lr, len(exposures)),
        "note": "Exploratory augmentation of the linear model by Q2-Q5 indicators; natural spline is the primary functional-form diagnostic.",
    }
    return rows, test


def mean_sd(series):
    return f"{series.mean():.2f} ({series.std(ddof=1):.2f})"


def median_iqr(series):
    return f"{series.median():.2f} [{series.quantile(.25):.2f}-{series.quantile(.75):.2f}]"


def mortality_descriptive(d, database):
    rows = []
    for mortality, group in d.groupby("hospital_mortality"):
        rows.append({
            "database": database,
            "outcome": "non-survivor" if mortality else "survivor",
            "n": len(group),
            "age_mean_sd": mean_sd(group["age"]),
            "female_n_percent": f"{int((1-group.sex_male).sum())} ({100*(1-group.sex_male).mean():.1f}%)",
            "emergency_n_percent": f"{int(group.emergency_admission.sum())} ({100*group.emergency_admission.mean():.1f}%)",
            "first_temperature_mean_sd": mean_sd(group["first_temperature_c"]),
            "temperature_measurements_median_iqr": median_iqr(group["n_temperature"]),
            "auc_median_iqr": median_iqr(group["temperature_deficit_auc_c_hours"]),
            "rewarming_slope_mean_sd": mean_sd(group["rewarming_slope_c_per_hour"]),
            "persistent_n_percent": f"{int(group.persistent_hypothermia_6h.sum())} ({100*group.persistent_hypothermia_6h.mean():.1f}%)",
            "overshoot_n_percent": f"{int(group.overshoot_hyperthermia.sum())} ({100*group.overshoot_hyperthermia.mean():.1f}%)",
            "icu_los_median_iqr": median_iqr(group["icu_los_days"]),
            "hospital_los_median_iqr": median_iqr(group["hospital_los_days"]),
        })
    return rows


def main():
    if ERROR.exists():
        ERROR.unlink()
    source = find_derived()
    raw = input(f"derived_rewarming_v3 folder (Enter to use {source or 'not found'}): ").strip().strip('"')
    source = Path(raw) if raw else source
    if source is None or not source.exists():
        raise FileNotFoundError("derived_rewarming_v3 was not found")
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {
        "MIMIC-IV": prepare(pd.read_csv(source / "mimic_rewarming_cohort.csv.gz", low_memory=False)),
        "eICU": prepare(pd.read_csv(source / "eicu_rewarming_cohort.csv.gz", low_memory=False)),
    }
    quintiles, tests, descriptive = [], [], []
    for database, frame in frames.items():
        rows, test = quintile_results(frame, database)
        quintiles.extend(rows)
        tests.append(test)
        descriptive.extend(mortality_descriptive(frame, database))
    pd.DataFrame(quintiles).to_csv(OUT / "Table_V3_AUC_Quintiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).to_csv(OUT / "Table_V3_Quintile_Linearity_Diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(descriptive).to_csv(OUT / "Table_V3_By_Mortality.csv", index=False, encoding="utf-8-sig")
    with (OUT / "UPLOAD_THIS_SUMMARY.txt").open("w", encoding="utf-8-sig") as output:
        output.write("V3 final table-completion summary\n\n")
        for database in frames:
            output.write(f"[{database}]\n")
            for row in [r for r in quintiles if r["database"] == database]:
                if row["reference"]:
                    output.write(f"Q1: {row['deaths']}/{row['n']} deaths (reference)\n")
                else:
                    output.write(
                        f"Q{row['quintile']}: {row['deaths']}/{row['n']} deaths; adjusted OR "
                        f"{row['odds_ratio']:.3f} (95% CI {row['ci_lower']:.3f}-{row['ci_upper']:.3f}), "
                        f"P={row['p_value']:.4g}\n"
                    )
            test = next(x for x in tests if x["database"] == database)
            output.write(f"Exploratory quintile-versus-linear LR test: P={test['p_value']:.4g}\n\n")
        output.write("All files are aggregate and may be uploaded.\n")
    print("Completed:", OUT)
    print("Upload the v3_completion_results folder.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc()
        ERROR.write_text(message, encoding="utf-8-sig")
        print(message)
        print("Error log:", ERROR)
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        sys.exit(1)
