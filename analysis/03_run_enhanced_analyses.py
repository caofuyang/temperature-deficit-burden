# -*- coding: utf-8 -*-
"""V3 aggregate-only enhanced analyses for the temperature-burden study.

Reads the local patient-level derived_rewarming files, but writes only aggregate
model tables, diagnostics, and a spline figure. Do not upload derived_rewarming.
"""
from __future__ import annotations

import csv
import gzip
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "v3_analysis_results" / "enhanced"
ERROR = HERE / "V3_enhanced_analysis_error.txt"
SURGICAL = {"SURG", "CSURG", "NSURG", "ORTHO", "ENT", "GU", "GYN", "OBS", "OMED", "PSURG", "TSURG", "VSURG", "DENT", "TRAUM"}

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import patsy
    import statsmodels.api as sm
    from scipy.stats import chi2
    from statsmodels.discrete.conditional_models import ConditionalLogit
except ImportError as exc:
    message = (
        f"Missing Python package: {exc}\n\n"
        "Run this command, then run the script again:\n"
        "py -3 -m pip install pandas numpy scipy matplotlib patsy statsmodels\n"
    )
    ERROR.write_text(message, encoding="utf-8-sig")
    print(message)
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
    sys.exit(1)


def find_derived() -> Path | None:
    candidates = [HERE / "derived_rewarming_v3", HERE.parent / "derived_rewarming_v3"]
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        try:
            candidates.extend(downloads.rglob("derived_rewarming_v3"))
        except OSError:
            pass
    valid = [
        p for p in candidates
        if (p / "mimic_rewarming_cohort.csv.gz").exists()
        and (p / "eicu_rewarming_cohort.csv.gz").exists()
        and (p / "mimic_temperature_long.csv.gz").exists()
        and (p / "eicu_temperature_long.csv.gz").exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda p: (p / "mimic_rewarming_cohort.csv.gz").stat().st_mtime)


def canon(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def open_text(path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="")
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def get(row, *names):
    low = {str(k).lower(): (v or "").strip() for k, v in row.items() if k is not None}
    for name in names:
        if low.get(name.lower(), ""):
            return low[name.lower()]
    return ""


def parse_dt(value):
    value = (value or "").strip().strip('"')
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("T", " ").replace("Z", ""))
    except ValueError:
        return None


def find_file(root, names):
    targets = {canon(n) for n in names}; hits = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and "archive" not in d.lower()]
        for filename in files:
            stem = filename.lower()
            for suffix in (".csv.gz", ".csv"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]; break
            if canon(stem) in targets:
                path = Path(base) / filename; hits.append((path.stat().st_size, path))
    return sorted(hits, reverse=True)[0][1] if hits else None


def is_surgical(service):
    service = (service or "").upper().strip()
    return service in SURGICAL or "SURG" in service


def is_surgical_icu(unit):
    unit = (unit or "").upper()
    return any(x in unit for x in ("SICU", "SURGICAL", "CVICU", "CARDIAC VASCULAR", "TRAUMA"))


def corrected_mimic_ids(root, cohort_ids):
    """Reproduce the final active-service + surgical-ICU alignment."""
    icustays = find_file(root, ["icustays"])
    services = find_file(root, ["services"])
    if icustays is None or services is None:
        raise FileNotFoundError("MIMIC root must contain icustays.csv(.gz) and services.csv(.gz)")
    service_history = defaultdict(list)
    with open_text(services) as handle:
        for row in csv.DictReader(handle):
            service_history[get(row, "hadm_id")].append((parse_dt(get(row, "transfertime")), get(row, "curr_service")))
    selected = set()
    with open_text(icustays) as handle:
        for row in csv.DictReader(handle):
            stay = get(row, "stay_id")
            if stay not in cohort_ids:
                continue
            intime = parse_dt(get(row, "intime")); hadm = get(row, "hadm_id")
            if intime is None:
                continue
            eligible = [(t, s) for t, s in service_history.get(hadm, []) if t and t <= intime]
            active_service = max(eligible, default=(None, ""), key=lambda x: x[0])[1]
            if is_surgical(active_service) and is_surgical_icu(get(row, "first_careunit")):
                selected.add(stay)
    return selected


def broad_surgery(value):
    s = str(value).lower()
    if any(x in s for x in ("cardiac", "cabg", "coronary", "valve", "aortic", "vascular", "cardiothoracic")):
        return "cardiothoracic_vascular"
    if any(x in s for x in ("neuro", "crani", "brain", "subdural", "spine")):
        return "neurosurgery"
    if any(x in s for x in ("trauma", "fracture", "orthop", "hip", "femur")):
        return "orthopedic_trauma"
    if any(x in s for x in ("abdom", "bowel", "colon", "gastric", "hepatic", "liver", "pancrea", "general")):
        return "general_abdominal"
    return "other_surgery"


def broad_icu(value):
    s = str(value).lower()
    if any(x in s for x in ("cardiac", "cticu", "csicu", "ccu", "cvicu")):
        return "cardiac_icu"
    if any(x in s for x in ("sicu", "surgical", "trauma", "vic")):
        return "surgical_icu"
    if "neuro" in s:
        return "neuro_icu"
    if any(x in s for x in ("micu", "medical")):
        return "medical_icu"
    return "other_icu"


def prepare(df):
    d = df.copy()
    numeric = [
        "age", "first_temperature_c", "first_temperature_minute",
        "last_temperature_minute", "n_temperature",
        "temperature_deficit_auc_c_hours", "hospital_mortality",
    ]
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["encounter_id"] = d["encounter_id"].astype(str)
    d["hospital_id"] = d["hospital_id"].astype(str)
    d["age"] = d["age"].fillna(d["age"].median())
    d["sex_male"] = d["sex"].astype(str).str.lower().str.startswith("m").astype(int)
    d["emergency_admission"] = pd.to_numeric(d["emergency_admission"], errors="coerce").fillna(0).astype(int)
    d["hospital_mortality"] = d["hospital_mortality"].astype(int)
    d["surgery_broad"] = d["surgery_group"].map(broad_surgery)
    d["icu_broad"] = d["icu_type"].map(broad_icu)
    return d


def zscore(series):
    sd = series.std(ddof=0)
    return (series - series.mean()) / (sd if pd.notna(sd) and sd > 0 else 1.0)


def design_frame(d, exposure="auc_z", extra=()):
    columns = [exposure, "age_z", "sex_male", "emergency_admission", "first_temp_z", "n_temp_z"] + list(extra)
    x = d[columns].astype(float)
    cats = pd.get_dummies(d[["surgery_broad", "icu_broad"]].fillna("Unknown"), drop_first=True, dtype=float)
    x = pd.concat([x, cats], axis=1)
    # Remove constants before adding an intercept. This also makes restricted
    # subgroup analyses robust when a category has only one observed level.
    keep = [c for c in x.columns if c == exposure or x[c].nunique(dropna=False) > 1]
    return x[keep]


def full_rank(x, forced=()):
    """Keep a deterministic full-rank subset while prioritizing target terms."""
    order = list(dict.fromkeys([c for c in forced if c in x.columns] + list(x.columns)))
    kept = []
    rank = 0
    for c in order:
        candidate = x[kept + [c]].to_numpy(float)
        new_rank = np.linalg.matrix_rank(candidate)
        if new_rank > rank:
            kept.append(c); rank = new_rank
    return x[kept]


def standardize(d, exposure):
    q = d.copy()
    q["auc_z"] = zscore(q[exposure])
    q["age_z"] = zscore(q["age"])
    q["first_temp_z"] = zscore(q["first_temperature_c"])
    q["n_temp_z"] = zscore(q["n_temperature"])
    return q


def fit_glm(d, database, label, exposure, extra_cont=()):
    q = standardize(d.dropna(subset=[exposure]).copy(), exposure)
    extra_z = []
    for c in extra_cont:
        q = q.dropna(subset=[c]).copy()
        name = c + "_z"; q[name] = zscore(q[c]); extra_z.append(name)
    x = sm.add_constant(design_frame(q, extra=extra_z), has_constant="add")
    x = full_rank(x, forced=("const", "auc_z"))
    model = sm.GLM(q["hospital_mortality"], x, family=sm.families.Binomial())
    if database == "eICU" and q["hospital_id"].nunique() > 1:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": q["hospital_id"]})
        covariance = "hospital-clustered"
    else:
        result = model.fit(cov_type="HC1")
        covariance = "HC1"
    b = float(result.params["auc_z"])
    se = float(result.bse["auc_z"])
    return {
        "database": database, "analysis": label, "n": len(q),
        "deaths": int(q["hospital_mortality"].sum()),
        "hospitals": int(q["hospital_id"].nunique()), "covariance": covariance,
        "odds_ratio_per_sd": math.exp(b), "ci_lower": math.exp(b - 1.96 * se),
        "ci_upper": math.exp(b + 1.96 * se), "p_value": float(result.pvalues["auc_z"]),
        "log_likelihood": float(result.llf),
    }


def integrate_observed(group):
    g = group.sort_values("minute_from_icu_admission")
    x = pd.to_numeric(g["minute_from_icu_admission"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(g["temperature_c"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= 360)
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return pd.Series({"auc_observed": np.nan, "observed_hours": np.nan})
    order = np.argsort(x); x, y = x[order], y[order]
    unique = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].median()
    x, y = unique["x"].to_numpy(), unique["y"].to_numpy()
    if len(x) < 2:
        return pd.Series({"auc_observed": np.nan, "observed_hours": np.nan})
    deficit = np.maximum(0.0, 36.0 - y)
    return pd.Series({
        "auc_observed": float(np.trapezoid(deficit, x / 60.0)),
        "observed_hours": float((x[-1] - x[0]) / 60.0),
    })


def add_observed_auc(cohort, long_df):
    long_df = long_df.copy()
    long_df["encounter_id"] = long_df["encounter_id"].astype(str)
    metrics = long_df.groupby("encounter_id", sort=False).apply(integrate_observed, include_groups=False).reset_index()
    return cohort.merge(metrics, on="encounter_id", how="left")


def spline_analysis(d, database):
    q = standardize(d.dropna(subset=["temperature_deficit_auc_c_hours"]).copy(), "temperature_deficit_auc_c_hours")
    cov = design_frame(q).drop(columns=["auc_z"])
    spline = patsy.dmatrix("cr(auc_z, df=4) - 1", q, return_type="dataframe")
    spline.columns = [f"spline_{i}" for i in range(spline.shape[1])]
    x_linear = sm.add_constant(pd.concat([q[["auc_z"]].reset_index(drop=True), cov.reset_index(drop=True)], axis=1), has_constant="add")
    x_spline = sm.add_constant(pd.concat([spline.reset_index(drop=True), cov.reset_index(drop=True)], axis=1), has_constant="add")
    x_linear = full_rank(x_linear, forced=("const", "auc_z"))
    x_spline = full_rank(x_spline, forced=("const", "spline_0", "spline_1", "spline_2", "spline_3"))
    y = q["hospital_mortality"].reset_index(drop=True)
    linear = sm.GLM(y, x_linear, family=sm.families.Binomial()).fit()
    nonlinear = sm.GLM(y, x_spline, family=sm.families.Binomial()).fit()
    lr = max(0.0, 2.0 * (nonlinear.llf - linear.llf))
    df_diff = max(1, int(nonlinear.df_model - linear.df_model))
    p = float(chi2.sf(lr, df_diff))

    grid = np.linspace(q["auc_z"].quantile(.01), q["auc_z"].quantile(.99), 120)
    reference = {c: 0.0 for c in cov.columns}
    pred_base = pd.DataFrame([reference] * len(grid))
    grid_spline = patsy.build_design_matrices([spline.design_info], {"auc_z": grid})[0]
    pred_x = sm.add_constant(pd.concat([pd.DataFrame(grid_spline, columns=spline.columns), pred_base], axis=1), has_constant="add")
    pred_x = pred_x.reindex(columns=x_spline.columns, fill_value=0.0)
    sf = nonlinear.get_prediction(pred_x).summary_frame()
    scale = q["temperature_deficit_auc_c_hours"].std(ddof=0)
    center = q["temperature_deficit_auc_c_hours"].mean()
    curve = pd.DataFrame({
        "database": database, "auc_c_hours": grid * scale + center,
        "predicted_mortality": sf["mean"], "ci_lower": sf["mean_ci_lower"], "ci_upper": sf["mean_ci_upper"],
    })
    test = {"database": database, "linear_loglik": linear.llf, "spline_loglik": nonlinear.llf,
            "lr_statistic": lr, "df": df_diff, "p_nonlinearity": p, "n": len(q),
            "note": "Likelihood-ratio comparison of linear AUC with a 4-df natural cubic spline"}
    return test, curve


def conditional_hospital_model(d):
    q = standardize(d.dropna(subset=["temperature_deficit_auc_c_hours"]).copy(), "temperature_deficit_auc_c_hours")
    variation = q.groupby("hospital_id")["hospital_mortality"].agg(["min", "max", "size"])
    eligible = variation.index[(variation["min"] == 0) & (variation["max"] == 1)]
    q = q[q["hospital_id"].isin(eligible)].copy()
    x = full_rank(design_frame(q).astype(float), forced=("auc_z",))
    try:
        result = ConditionalLogit(
            q["hospital_mortality"].to_numpy(), x.to_numpy(),
            groups=q["hospital_id"].to_numpy(),
        ).fit(disp=False, maxiter=500)
        i = list(x.columns).index("auc_z")
        b, se = float(result.params[i]), float(result.bse[i])
        p_value = float(result.pvalues[i])
        analysis = "Hospital-stratified conditional logistic regression"
        note = "Controls all hospital-level factors constant within hospital; hospitals without outcome variation are omitted by design."
    except Exception as exc:
        # Explicit hospital indicators are a transparent fallback if the
        # conditional-likelihood Hessian cannot be inverted.
        hospitals = pd.get_dummies(q["hospital_id"], prefix="hospital", drop_first=True, dtype=float)
        xf = sm.add_constant(pd.concat([x.reset_index(drop=True), hospitals.reset_index(drop=True)], axis=1), has_constant="add")
        xf = full_rank(xf, forced=("const", "auc_z"))
        result = sm.GLM(q["hospital_mortality"].reset_index(drop=True), xf, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": q["hospital_id"].reset_index(drop=True)}
        )
        b, se = float(result.params["auc_z"]), float(result.bse["auc_z"])
        p_value = float(result.pvalues["auc_z"])
        analysis = "Hospital fixed-effect logistic regression (indicator fallback)"
        note = f"Conditional-likelihood fit failed ({type(exc).__name__}); hospital indicators were used instead."
    return {
        "database": "eICU", "analysis": analysis,
        "n": len(q), "deaths": int(q["hospital_mortality"].sum()),
        "hospitals": len(eligible), "odds_ratio_per_sd": math.exp(b),
        "ci_lower": math.exp(b - 1.96 * se), "ci_upper": math.exp(b + 1.96 * se),
        "p_value": p_value, "note": note,
    }


def plot_splines(curves):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (database, g) in zip(axes, curves.groupby("database", sort=False)):
        ax.plot(g["auc_c_hours"], 100 * g["predicted_mortality"], color="#1f5a85", lw=2)
        ax.fill_between(g["auc_c_hours"], 100 * g["ci_lower"], 100 * g["ci_upper"], color="#8fbad1", alpha=.35)
        ax.set_title(database); ax.set_xlabel("Temperature-deficit AUC (°C·h)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Model-based mortality probability (%)")
    fig.tight_layout(); fig.savefig(OUT / "Figure_Spline_AUC_Mortality.png", dpi=300); plt.close(fig)


def main():
    if ERROR.exists():
        ERROR.unlink()
    auto = find_derived()
    raw = input(f"derived_rewarming folder (Enter to use {auto or 'not found'}): ").strip().strip('"')
    source = Path(raw) if raw else auto
    if source is None or not source.exists():
        raise FileNotFoundError("derived_rewarming folder was not found")
    OUT.mkdir(parents=True, exist_ok=True)

    specs = {
        "MIMIC-IV": ("mimic_rewarming_cohort.csv.gz", "mimic_temperature_long.csv.gz"),
        "eICU": ("eicu_rewarming_cohort.csv.gz", "eicu_temperature_long.csv.gz"),
    }
    cohorts = {}; rows = []; spline_tests = []; curves = []
    for database, (cohort_name, long_name) in specs.items():
        cohort = prepare(pd.read_csv(source / cohort_name, low_memory=False))
        long_df = pd.read_csv(source / long_name, low_memory=False)
        cohort = add_observed_auc(cohort, long_df)
        cohorts[database] = cohort

        rows.append(fit_glm(cohort, database, "Statsmodels verification of final boundary-carried primary AUC", "temperature_deficit_auc_c_hours"))
        tight = cohort[(cohort["first_temperature_minute"] <= 60) & (cohort["last_temperature_minute"] >= 300)].copy()
        rows.append(fit_glm(tight, database, "Tight boundary coverage (first <=1 h; last >=5 h), original AUC", "temperature_deficit_auc_c_hours"))
        rows.append(fit_glm(
            tight, database,
            "Observed-interval AUC without boundary extrapolation (adjusted for observed duration)",
            "auc_observed", extra_cont=("observed_hours",),
        ))
        test, curve = spline_analysis(cohort, database)
        spline_tests.append(test); curves.append(curve)

    hospital_result = conditional_hospital_model(cohorts["eICU"])
    pd.DataFrame(rows).to_csv(OUT / "Table_Enhanced_AUC_Models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(spline_tests).to_csv(OUT / "Table_Nonlinearity_Tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([hospital_result]).to_csv(OUT / "Table_eICU_Hospital_Conditional_Model.csv", index=False, encoding="utf-8-sig")
    curve_df = pd.concat(curves, ignore_index=True)
    curve_df.to_csv(OUT / "Spline_Predictions.csv", index=False, encoding="utf-8-sig")
    plot_splines(curve_df)

    with (OUT / "UPLOAD_THIS_SUMMARY.txt").open("w", encoding="utf-8-sig") as f:
        f.write("V3 enhanced-analysis summary\n\n")
        f.write("All models use the same five-category harmonized surgery classification.\n\n")
        f.write("[AUC sensitivity and software verification]\n")
        for r in rows:
            f.write(f"{r['database']} | {r['analysis']} | OR/SD {r['odds_ratio_per_sd']:.3f} "
                    f"(95% CI {r['ci_lower']:.3f}-{r['ci_upper']:.3f}), P={r['p_value']:.4g}, "
                    f"N={r['n']}, deaths={r['deaths']}\n")
        f.write("\n[Natural-spline nonlinearity tests]\n")
        for r in spline_tests:
            f.write(f"{r['database']} | LR={r['lr_statistic']:.3f}, df={r['df']}, P={r['p_nonlinearity']:.4g}, N={r['n']}\n")
        f.write("\n[eICU hospital-stratified model]\n")
        r = hospital_result
        f.write(f"OR/SD {r['odds_ratio_per_sd']:.3f} (95% CI {r['ci_lower']:.3f}-{r['ci_upper']:.3f}), "
                f"P={r['p_value']:.4g}, N={r['n']}, deaths={r['deaths']}, hospitals={r['hospitals']}\n")
        f.write("\nPrivacy: this folder contains aggregate outputs only. Do not upload derived_rewarming_v3.\n")
    print(f"\nCompleted. After both scripts finish, upload the parent v3_analysis_results folder: {OUT.parent}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc()
        ERROR.write_text(message, encoding="utf-8-sig")
        print(message)
        print("Error log:", ERROR)
        sys.exit(1)
