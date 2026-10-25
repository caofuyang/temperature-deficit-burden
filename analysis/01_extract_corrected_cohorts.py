# -*- coding: utf-8 -*-
"""V3: rebuild harmonized MIMIC-IV/eICU postoperative hypothermia cohorts.

Corrections implemented:
1. MIMIC age is age at hospital admission, derived from anchor age/year.
2. MIMIC surgery group uses the active service at ICU admission.
3. MIMIC requires an active surgical service and a surgical ICU.
4. eICU postoperative status uses unit admission source only.
5. eICU excludes missing/unknown hospital discharge status.
6. Both databases retain the first *eligible* ICU stay per hospitalization.

Patient-level files stay on the user's computer. Only the aggregate summary
and the later aggregate analysis-results folder should be uploaded.
"""

from __future__ import annotations

import csv
import gzip
import os
import re
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "derived_rewarming_v3"
SUMMARY = HERE / "V3_最终队列提取汇总.txt"
ERROR = HERE / "V3_队列提取_错误日志.txt"

SURGICAL_SERVICES = {
    "SURG", "CSURG", "NSURG", "ORTHO", "ENT", "GU", "GYN", "OBS",
    "OMED", "PSURG", "TSURG", "VSURG", "DENT", "TRAUM",
}

POSTOPERATIVE_UNIT_SOURCES = (
    "operating room", "operating rm", "recovery room", "pacu",
    "post-anesthesia", "post anesthesia", "perioperative",
)

COHORT_FIELDS = [
    "database", "encounter_id", "patient_id", "hospital_id", "age", "sex",
    "race_ethnicity", "surgery_group", "icu_type", "emergency_admission",
    "first_temperature_c", "first_temperature_minute", "n_temperature",
    "last_temperature_minute", "minimum_temperature_c", "maximum_temperature_c",
    "mean_temperature_c", "rewarming_slope_c_per_hour",
    "time_to_normothermia_hours", "temperature_deficit_auc_c_hours",
    "persistent_hypothermia_6h", "overshoot_hyperthermia", "icu_los_days",
    "hospital_los_days", "hospital_mortality",
]

LONG_FIELDS = [
    "database", "encounter_id", "minute_from_icu_admission", "temperature_c"
]


def canon(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def open_text(path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="")
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def find_file(root, names):
    targets = {canon(n) for n in names}
    hits = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and "archive" not in d.lower()]
        for filename in files:
            stem = filename.lower()
            for suffix in (".csv.gz", ".csv"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            if canon(stem) in targets:
                path = Path(base) / filename
                hits.append((path.stat().st_size, path))
    return sorted(hits, reverse=True)[0][1] if hits else None


def get(row, *names):
    lower = {str(k).lower(): (v or "").strip() for k, v in row.items() if k is not None}
    for name in names:
        value = lower.get(name.lower(), "")
        if value:
            return value
    return ""


def parse_dt(value):
    value = (value or "").strip().strip('"')
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("T", " ").replace("Z", ""))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M:%S",
        "%m/%d/%y %H:%M",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def num(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", (value or "").replace(",", ""))
    try:
        return float(match.group()) if match else None
    except ValueError:
        return None


def temp_c(value, unit="", label=""):
    result = num(value)
    if result is None:
        return None
    text = f"{unit} {label}".lower()
    if "fahren" in text or re.search(r"(^|\W)f($|\W)", text) or result > 50:
        result = (result - 32) * 5 / 9
    return result if 25 <= result <= 45 else None


def core_mimic_label(label):
    text = label.lower()
    if any(x in text for x in ("site", "skin", "cerebral", "change", "apache")):
        return False
    return (
        "temperature celsius" in text
        or "temperature fahrenheit" in text
        or "blood temperature cco" in text
    )


def is_surgical_service(value):
    service = (value or "").upper().strip()
    return service in SURGICAL_SERVICES or "SURG" in service


def is_surgical_icu(value):
    unit = (value or "").upper()
    return any(x in unit for x in ("SICU", "SURGICAL", "CVICU", "CARDIAC VASCULAR", "TRAUMA"))


def surgery_group(service):
    service = (service or "").upper().strip()
    if service in ("CSURG", "TSURG", "VSURG"):
        return "cardiothoracic_vascular"
    if service == "NSURG":
        return "neurosurgery"
    if service in ("ORTHO", "OMED", "TRAUM"):
        return "orthopedic_trauma"
    if service in ("GYN", "OBS"):
        return "gynecology_obstetric"
    if service in ("ENT", "DENT"):
        return "head_neck"
    if service == "GU":
        return "urologic"
    return "general_other_surgery"


def active_service(service_rows, when):
    prior = [(timepoint, service) for timepoint, service in service_rows if timepoint and timepoint <= when]
    return max(prior, default=(None, ""), key=lambda x: x[0])[1]


def age_at_admission(anchor_age, anchor_year, admission_time):
    if anchor_age is None:
        return None
    if anchor_year is None or admission_time is None:
        return anchor_age
    return anchor_age + admission_time.year - anchor_year


def dedupe_temperatures(values):
    by_minute = defaultdict(list)
    for minute, value in values:
        by_minute[int(round(minute))].append(value)
    return [(minute, statistics.median(group)) for minute, group in sorted(by_minute.items())]


def temperature_metrics(values):
    values = dedupe_temperatures(values)
    if len(values) < 3:
        return None, "fewer_than_3"
    if values[0][0] > 120:
        return None, "first_after_2h"
    if not (32 <= values[0][1] < 36):
        return None, "not_initial_hypothermia"
    if values[-1][0] < 240:
        return None, "no_late_measurement"

    hours = [minute / 60 for minute, _ in values]
    temperatures = [temperature for _, temperature in values]
    mean_x = sum(hours) / len(hours)
    mean_y = sum(temperatures) / len(temperatures)
    denominator = sum((x - mean_x) ** 2 for x in hours)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(hours, temperatures)) / denominator
        if denominator else 0.0
    )
    normothermia = [minute for minute, temperature in values if temperature >= 36]
    time_to_normothermia = (
        (normothermia[0] - values[0][0]) / 60 if normothermia else ""
    )

    # Primary estimand: trapezoidal AUC below 36 C over 0-6 h, carrying the
    # first observation back to time zero and the last observation to 6 h.
    curve = [(0, values[0][1])]
    curve += [(minute, temperature) for minute, temperature in values if 0 < minute < 360]
    curve += [(360, values[-1][1])]
    auc = 0.0
    for (minute1, temp1), (minute2, temp2) in zip(curve, curve[1:]):
        deficit1 = max(0, 36 - temp1)
        deficit2 = max(0, 36 - temp2)
        auc += (deficit1 + deficit2) / 2 * ((minute2 - minute1) / 60)

    return {
        "first_temperature_c": round(values[0][1], 3),
        "first_temperature_minute": values[0][0],
        "n_temperature": len(values),
        "last_temperature_minute": values[-1][0],
        "minimum_temperature_c": round(min(temperatures), 3),
        "maximum_temperature_c": round(max(temperatures), 3),
        "mean_temperature_c": round(sum(temperatures) / len(temperatures), 3),
        "rewarming_slope_c_per_hour": round(slope, 4),
        "time_to_normothermia_hours": (
            "" if time_to_normothermia == "" else round(time_to_normothermia, 3)
        ),
        "temperature_deficit_auc_c_hours": round(auc, 4),
        "persistent_hypothermia_6h": int(not normothermia),
        "overshoot_hyperthermia": int(max(temperatures) >= 38),
        "_long": values,
    }, "eligible"


def progress(label, rows, started):
    if rows and rows % 5_000_000 == 0:
        speed = rows / max(time.time() - started, 0.1)
        print(f"  {label}: {rows:,}行（{speed:,.0f}行/秒）", flush=True)


def write_gzip_csv(path, fields, rows):
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finalize(database, candidates, temperatures):
    eligible = []
    reasons = Counter()
    for encounter_id, base in candidates.items():
        measurements, reason = temperature_metrics(temperatures.get(encounter_id, []))
        reasons[reason] += 1
        if measurements is None:
            continue
        eligible.append((base.get("_order", ()), encounter_id, base, measurements))

    # Retain the first eligible ICU stay within each hospitalization.
    selected = []
    seen_hospital_stays = set()
    duplicate_eligible = 0
    for order, encounter_id, base, measurements in sorted(eligible, key=lambda x: x[0]):
        hospital_stay = base["_hospital_stay_id"]
        if hospital_stay in seen_hospital_stays:
            duplicate_eligible += 1
            continue
        seen_hospital_stays.add(hospital_stay)
        selected.append((encounter_id, base, measurements))

    cohort = []
    long_rows = []
    deaths = 0
    for encounter_id, base, measurements in selected:
        row = {k: v for k, v in base.items() if not k.startswith("_")}
        row.update({k: v for k, v in measurements.items() if not k.startswith("_")})
        cohort.append(row)
        deaths += int(row["hospital_mortality"])
        for minute, value in measurements["_long"]:
            long_rows.append({
                "database": database,
                "encounter_id": encounter_id,
                "minute_from_icu_admission": minute,
                "temperature_c": round(value, 3),
            })

    stats = {
        "candidate": len(candidates),
        "eligible_before_dedup": len(eligible),
        "duplicate_eligible": duplicate_eligible,
        "eligible": len(cohort),
        "deaths": deaths,
        "long_rows": len(long_rows),
    }
    return cohort, long_rows, reasons, stats


def build_mimic(root):
    required = {
        name: find_file(root, [name])
        for name in ("services", "icustays", "admissions", "patients", "d_items", "chartevents")
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError("MIMIC-IV缺少：" + ", ".join(missing))

    patients = {}
    with open_text(required["patients"]) as handle:
        for row in csv.DictReader(handle):
            patients[get(row, "subject_id")] = {
                "anchor_age": num(get(row, "anchor_age")),
                "anchor_year": num(get(row, "anchor_year")),
                "sex": get(row, "gender"),
            }

    admissions = {}
    with open_text(required["admissions"]) as handle:
        for row in csv.DictReader(handle):
            hadm_id = get(row, "hadm_id")
            admit = parse_dt(get(row, "admittime"))
            discharge = parse_dt(get(row, "dischtime"))
            admissions[hadm_id] = {
                "subject": get(row, "subject_id"),
                "admit": admit,
                "race": get(row, "race"),
                "type": get(row, "admission_type"),
                "mortality": int(get(row, "hospital_expire_flag") == "1"),
                "hosp_los": (
                    (discharge - admit).total_seconds() / 86400
                    if admit and discharge else ""
                ),
            }

    services = defaultdict(list)
    with open_text(required["services"]) as handle:
        for row in csv.DictReader(handle):
            hadm_id = get(row, "hadm_id")
            if hadm_id:
                services[hadm_id].append((
                    parse_dt(get(row, "transfertime")), get(row, "curr_service").upper()
                ))

    candidates = {}
    cohort_reasons = Counter()
    with open_text(required["icustays"]) as handle:
        for row in csv.DictReader(handle):
            hadm_id = get(row, "hadm_id")
            stay_id = get(row, "stay_id")
            start = parse_dt(get(row, "intime"))
            end = parse_dt(get(row, "outtime"))
            if not hadm_id or not stay_id or not start or not end:
                cohort_reasons["missing_icu_identifiers_or_times"] += 1
                continue
            if (end - start).total_seconds() < 21600:
                cohort_reasons["icu_stay_under_6h"] += 1
                continue
            service = active_service(services.get(hadm_id, []), start)
            if not is_surgical_service(service):
                cohort_reasons["not_active_surgical_service"] += 1
                continue
            icu_type = get(row, "first_careunit")
            if not is_surgical_icu(icu_type):
                cohort_reasons["not_surgical_icu"] += 1
                continue
            admission = admissions.get(hadm_id, {})
            patient = patients.get(admission.get("subject", ""), {})
            age = age_at_admission(
                patient.get("anchor_age"), patient.get("anchor_year"), admission.get("admit")
            )
            if age is not None and age < 18:
                cohort_reasons["age_under_18"] += 1
                continue
            candidates[stay_id] = {
                "database": "MIMIC-IV",
                "encounter_id": stay_id,
                "patient_id": admission.get("subject", ""),
                "hospital_id": "MIMIC",
                "age": "" if age is None else age,
                "sex": patient.get("sex", ""),
                "race_ethnicity": admission.get("race", ""),
                "surgery_group": surgery_group(service),
                "icu_type": icu_type,
                "emergency_admission": int(any(
                    token in admission.get("type", "").upper()
                    for token in ("EMER", "URGENT", "TRAUMA")
                )),
                "icu_los_days": round((end - start).total_seconds() / 86400, 4),
                "hospital_los_days": admission.get("hosp_los", ""),
                "hospital_mortality": admission.get("mortality", 0),
                "_start": start,
                "_order": (start, stay_id),
                "_hospital_stay_id": hadm_id,
            }

    item_labels = {}
    with open_text(required["d_items"]) as handle:
        for row in csv.DictReader(handle):
            label = get(row, "label")
            if core_mimic_label(label):
                item_labels[get(row, "itemid")] = label

    temperatures = defaultdict(list)
    markers = [f",{item_id}," for item_id in item_labels]
    rows = 0
    started = time.time()
    print("\n[MIMIC-IV] 扫描0–6小时体温……")
    with open_text(required["chartevents"]) as handle:
        header = handle.readline()
        columns = next(csv.reader([header]))
        index = {name.lower(): i for i, name in enumerate(columns)}
        for line in handle:
            rows += 1
            progress("MIMIC chartevents", rows, started)
            if not any(marker in line for marker in markers):
                continue
            try:
                row = next(csv.reader([line]))
                stay_id = row[index["stay_id"]].strip()
                item_id = row[index["itemid"]].strip()
            except (KeyError, IndexError, csv.Error):
                continue
            if stay_id not in candidates or item_id not in item_labels:
                continue
            when = parse_dt(row[index["charttime"]])
            unit = row[index["valueuom"]]
            raw = row[index["valuenum"]] or row[index["value"]]
            value = temp_c(raw, unit, item_labels[item_id])
            if when is None or value is None:
                continue
            minute = (when - candidates[stay_id]["_start"]).total_seconds() / 60
            if 0 <= minute <= 360:
                temperatures[stay_id].append((minute, value))

    result = finalize("MIMIC-IV", candidates, temperatures)
    result[2].update(cohort_reasons)
    return result


def valid_eicu_outcome(value):
    status = (value or "").strip().lower()
    if status in ("alive", "expired"):
        return int(status == "expired")
    return None


def build_eicu(root):
    patient_path = find_file(root, ["patient"])
    nurse_path = find_file(root, ["nurseCharting"])
    if not patient_path or not nurse_path:
        raise FileNotFoundError("eICU-CRD缺少 patient 或 nurseCharting")

    candidates = {}
    cohort_reasons = Counter()
    with open_text(patient_path) as handle:
        for row in csv.DictReader(handle):
            stay_id = get(row, "patientunitstayid")
            hospital_stay_id = get(row, "patienthealthsystemstayid")
            if not stay_id or not hospital_stay_id:
                cohort_reasons["missing_stay_identifier"] += 1
                continue
            age = num(get(row, "age").replace(">", ""))
            if age is not None and age < 18:
                cohort_reasons["age_under_18"] += 1
                continue
            unit_source = get(row, "unitadmitsource").lower()
            if not any(term in unit_source for term in POSTOPERATIVE_UNIT_SOURCES):
                cohort_reasons["not_direct_or_pacu_to_icu"] += 1
                continue
            mortality = valid_eicu_outcome(get(row, "hospitaldischargestatus"))
            if mortality is None:
                cohort_reasons["missing_or_unknown_hospital_outcome"] += 1
                continue
            unit_los = num(get(row, "unitdischargeoffset"))
            hospital_admit_offset = num(get(row, "hospitaladmitoffset"))
            hospital_discharge_offset = num(get(row, "hospitaldischargeoffset"))
            if unit_los is None or unit_los < 360:
                cohort_reasons["icu_stay_under_6h"] += 1
                continue
            unit_visit = num(get(row, "unitvisitnumber"))
            # ICU admission occurs -hospitaladmitoffset minutes after hospital admission.
            hospital_minute = -hospital_admit_offset if hospital_admit_offset is not None else float("inf")
            order = (hospital_minute, unit_visit if unit_visit is not None else float("inf"), stay_id)
            hospital_source = get(row, "hospitaladmitsource").lower()
            admission_type = get(row, "unitadmittype").lower()
            diagnosis = get(row, "apacheadmissiondx")
            candidates[stay_id] = {
                "database": "eICU",
                "encounter_id": stay_id,
                "patient_id": get(row, "uniquepid"),
                "hospital_id": get(row, "hospitalid"),
                "age": "" if age is None else age,
                "sex": get(row, "gender"),
                "race_ethnicity": get(row, "ethnicity"),
                "surgery_group": diagnosis or "postoperative_unspecified",
                "icu_type": get(row, "unittype"),
                "emergency_admission": int(
                    "emergency" in hospital_source
                    or "emergency" in admission_type
                    or "urgent" in admission_type
                ),
                "icu_los_days": round(unit_los / 1440, 4),
                "hospital_los_days": (
                    round((hospital_discharge_offset - hospital_admit_offset) / 1440, 4)
                    if hospital_admit_offset is not None and hospital_discharge_offset is not None
                    else ""
                ),
                "hospital_mortality": mortality,
                "_order": order,
                "_hospital_stay_id": hospital_stay_id,
            }

    temperatures = defaultdict(list)
    rows = 0
    started = time.time()
    print("\n[eICU] 扫描0–6小时体温……")
    with open_text(nurse_path) as handle:
        header = handle.readline()
        columns = next(csv.reader([header]))
        index = {name.lower(): i for i, name in enumerate(columns)}
        for line in handle:
            rows += 1
            progress("eICU nurseCharting", rows, started)
            if "temp" not in line.lower():
                continue
            try:
                row = next(csv.reader([line]))
                stay_id = row[index["patientunitstayid"]].strip()
            except (KeyError, IndexError, csv.Error):
                continue
            if stay_id not in candidates:
                continue
            label = (
                row[index["nursingchartcelltypevallabel"]]
                + " "
                + row[index["nursingchartcelltypevalname"]]
            ).strip()
            label_lower = label.lower()
            if "temperature" not in label_lower or "location" in label_lower:
                continue
            minute = num(row[index["nursingchartoffset"]])
            value = temp_c(row[index["nursingchartvalue"]], "", label)
            if minute is not None and value is not None and 0 <= minute <= 360:
                temperatures[stay_id].append((minute, value))

    result = finalize("eICU", candidates, temperatures)
    result[2].update(cohort_reasons)
    return result


def root_prompt(name, default):
    while True:
        raw = input(f"{name}根目录（直接回车使用 {default}）：").strip().strip('"')
        path = Path(raw or default)
        if path.exists() and path.is_dir():
            return path
        print("找不到目录，请重新输入。")


def write_summary(results):
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as output:
        output.write("V3最终校正队列提取汇总\n")
        output.write(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        output.write("已实施的关键校正：\n")
        output.write("1. MIMIC年龄按anchor_age + 入院年份 - anchor_year计算。\n")
        output.write("2. MIMIC使用ICU入科时最新活动外科服务确定手术类别，并限定外科ICU。\n")
        output.write("3. eICU仅使用unitadmitsource判定OR/PACU/恢复室直接转入ICU。\n")
        output.write("4. eICU排除缺失或不明确的院内结局。\n")
        output.write("5. 两库均在满足全部条件后，每次住院仅保留首次符合条件的ICU stay。\n")
        output.write("6. 合理体温范围为25–45°C；研究队列首个体温须为32–<36°C。\n")
        output.write("7. AUC为0–6小时低于36°C的梯形面积；边界使用最近观察值延伸。\n")
        output.write("8. 复温时间从首次符合条件的低温测量开始计算。\n")
        output.write("\nPrivacy: this summary contains aggregate counts only.\n")

        for name, (_, _, reasons, stats) in results:
            output.write(f"\n[{name}]\n")
            output.write(f"候选ICU stay：{stats['candidate']:,}\n")
            output.write(f"温度条件合格（去重前）：{stats['eligible_before_dedup']:,}\n")
            output.write(f"同次住院重复合格stay排除：{stats['duplicate_eligible']:,}\n")
            output.write(f"最终队列：{stats['eligible']:,}\n")
            output.write(f"院内死亡：{stats['deaths']:,}\n")
            output.write(f"体温长表行数：{stats['long_rows']:,}\n")
            output.write("流程计数：\n")
            for reason, count in reasons.most_common():
                output.write(f"  {reason}: {count:,}\n")

        output.write("\nNext: keep derived_rewarming_v3 local and use this summary for downstream steps.\n")


def main():
    if ERROR.exists():
        ERROR.unlink()
    OUTDIR.mkdir(exist_ok=True)
    print("=" * 72)
    print("温度负荷研究 V3 最终队列校正")
    print("本步骤会重新扫描MIMIC-IV和eICU；患者级文件仅保存在本机。")
    print("=" * 72)
    mimic_root = root_prompt("MIMIC-IV", str(config.MIMIC_ROOT.parent))
    eicu_root = root_prompt("eICU-CRD", str(config.EICU_ROOT.parent))

    mimic = build_mimic(mimic_root)
    eicu = build_eicu(eicu_root)
    for filename, fields, rows in (
        ("mimic_rewarming_cohort.csv.gz", COHORT_FIELDS, mimic[0]),
        ("mimic_temperature_long.csv.gz", LONG_FIELDS, mimic[1]),
        ("eicu_rewarming_cohort.csv.gz", COHORT_FIELDS, eicu[0]),
        ("eicu_temperature_long.csv.gz", LONG_FIELDS, eicu[1]),
    ):
        write_gzip_csv(OUTDIR / filename, fields, rows)

    write_summary((("MIMIC-IV", mimic), ("eICU", eicu)))
    print("\n完成。结果汇总：", SUMMARY)
    print("Patient-level intermediates stay local; only the aggregate summary is for distribution.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        message = traceback.format_exc()
        ERROR.write_text(message, encoding="utf-8-sig")
        print(message)
        print("错误日志：", ERROR)
        input("按回车关闭……")
        sys.exit(1)

