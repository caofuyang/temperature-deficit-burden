# -*- coding: utf-8 -*-
"""V3 severity-adjusted sensitivity analysis for the temperature-burden study."""
from __future__ import annotations
import csv,gzip,math,os,re,sys,time,traceback
from collections import defaultdict
from datetime import datetime,timedelta
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HERE=Path(__file__).resolve().parent
OUT=HERE/"v3_severity_results"
ERR=HERE/"V3_severity_error.txt"
SURGICAL={"SURG","CSURG","NSURG","ORTHO","ENT","GU","GYN","OBS","OMED","PSURG","TSURG","VSURG","DENT","TRAUM"}
try:
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    from scipy.stats import norm
except ImportError as exc:
    ERR.write_text("缺少统计包："+str(exc)+"\n运行：py -3 -m pip install pandas numpy scipy\n",encoding="utf-8-sig")
    print(ERR.read_text(encoding="utf-8-sig"));input("按回车关闭……");sys.exit(1)

def canon(s):return re.sub(r"[^a-z0-9]","",str(s).lower())
def open_text(p):return gzip.open(p,"rt",encoding="utf-8-sig",errors="replace",newline="") if p.name.lower().endswith(".gz") else p.open("r",encoding="utf-8-sig",errors="replace",newline="")
def get(r,*names):
    low={str(k).lower():(v or "").strip() for k,v in r.items() if k is not None}
    for n in names:
        if low.get(n.lower(),""):return low[n.lower()]
    return ""
def num(s):
    try:return float(str(s).replace(",","").strip())
    except (TypeError,ValueError):return None
def parse_dt(s):
    s=(s or "").strip().strip('"')
    if not s:return None
    try:return datetime.fromisoformat(s.replace("T"," ").replace("Z",""))
    except ValueError:return None
def find_file(root,names):
    targets={canon(n) for n in names};hits=[]
    for base,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if not d.startswith(".") and "archive" not in d.lower()]
        for fn in files:
            stem=fn.lower()
            for suf in (".csv.gz",".csv"):
                if stem.endswith(suf):stem=stem[:-len(suf)];break
            if canon(stem) in targets:
                p=Path(base)/fn;hits.append((p.stat().st_size,p))
    return sorted(hits,reverse=True)[0][1] if hits else None
def find_derived():
    choices=[HERE/"derived_rewarming_v3",HERE.parent/"derived_rewarming_v3"];d=Path.home()/"Downloads"
    if d.exists():
        try:choices.extend(d.rglob("derived_rewarming_v3"))
        except OSError:pass
    valid=[p for p in choices if (p/"mimic_rewarming_cohort.csv.gz").exists() and (p/"eicu_rewarming_cohort.csv.gz").exists()]
    return max(valid,key=lambda p:(p/"mimic_rewarming_cohort.csv.gz").stat().st_mtime) if valid else None
def progress(label,n,start):
    if n and n%5_000_000==0:print(f"  {label}: {n:,}行（{n/max(time.time()-start,.1):,.0f}行/秒）",flush=True)
def is_surgical(s):
    s=(s or "").upper().strip();return s in SURGICAL or "SURG" in s
def is_surgical_icu(s):
    s=(s or "").upper();return any(x in s for x in ("SICU","SURGICAL","CVICU","CARDIAC VASCULAR","TRAUMA"))

def map_mimic_stays(root,cohort_ids):
    """Map the already-final V3 cohort to admission IDs and ICU admission times."""
    path=find_file(root,["icustays"])
    if path is None:raise FileNotFoundError("MIMIC缺少icustays")
    selected={};hadm_map={}
    with open_text(path) as f:
        for r in csv.DictReader(f):
            stay=get(r,"stay_id");when=parse_dt(get(r,"intime"));hadm=get(r,"hadm_id")
            if stay not in cohort_ids or when is None:continue
            selected[stay]={"hadm_id":hadm,"intime":when};hadm_map[hadm]=stay
    missing=cohort_ids-set(selected)
    if missing:raise RuntimeError(f"MIMIC V3队列中有{len(missing)}个stay无法在icustays中匹配")
    return selected,hadm_map

def item_target(label):
    s=label.lower()
    if "gcs" in s or "glasgow coma" in s:
        if "total" in s:return "gcs_total"
        if "eye" in s:return "gcs_eye"
        if "motor" in s:return "gcs_motor"
        if "verbal" in s:return "gcs_verbal"
    if any(x in s for x in ("arterial blood pressure mean","non invasive blood pressure mean","manual blood pressure mean","nbp mean","abp mean")):
        if "pulmonary" not in s:return "map"
    return None
def lab_target(label):
    s=label.lower().strip()
    if "lactate" in s and "dehydrogenase" not in s:return "lactate"
    if s=="creatinine" or s.startswith("creatinine, serum"):return "creatinine"
    return None

def extract_mimic_severity(root,selected,hadm_map):
    paths={n:find_file(root,[n]) for n in ("d_items","chartevents","d_labitems","labevents")}
    missing=[n for n,p in paths.items() if p is None]
    if missing:raise FileNotFoundError("MIMIC缺少："+", ".join(missing))
    chart_items={}
    with open_text(paths["d_items"]) as f:
        for r in csv.DictReader(f):
            target=item_target(get(r,"label"))
            if target:chart_items[get(r,"itemid")]=target
    labs={}
    with open_text(paths["d_labitems"]) as f:
        for r in csv.DictReader(f):
            target=lab_target(get(r,"label"))
            if target:labs[get(r,"itemid")]=target
    values={s:defaultdict(list) for s in selected};gcs_parts={s:defaultdict(dict) for s in selected}
    markers=[f",{x}," for x in chart_items];rows=0;started=time.time()
    print("\n[MIMIC] 扫描0–2小时MAP和GCS……")
    with open_text(paths["chartevents"]) as f:
        header=f.readline();cols=next(csv.reader([header]));ix={n.lower():i for i,n in enumerate(cols)}
        for line in f:
            rows+=1;progress("chartevents",rows,started)
            if not any(m in line for m in markers):continue
            try:r=next(csv.reader([line]));stay=r[ix["stay_id"]].strip();item=r[ix["itemid"]].strip()
            except Exception:continue
            if stay not in selected or item not in chart_items:continue
            when=parse_dt(r[ix["charttime"]]);v=num(r[ix.get("valuenum",ix.get("value"))])
            if when is None or v is None:continue
            minute=(when-selected[stay]["intime"]).total_seconds()/60
            if not 0<=minute<=120:continue
            target=chart_items[item]
            if target=="map" and 20<=v<=200:values[stay]["map"].append(v)
            elif target=="gcs_total" and 3<=v<=15:values[stay]["gcs"].append(v)
            elif target.startswith("gcs_"):
                key=int(round(minute));gcs_parts[stay][key][target]=v
    for stay,times in gcs_parts.items():
        for parts in times.values():
            if all(k in parts for k in ("gcs_eye","gcs_motor","gcs_verbal")):
                total=parts["gcs_eye"]+parts["gcs_motor"]+parts["gcs_verbal"]
                if 3<=total<=15:values[stay]["gcs"].append(total)
    markers=[f",{x}," for x in labs];rows=0;started=time.time()
    print("\n[MIMIC] 扫描入科前6小时至后2小时乳酸和肌酐……")
    with open_text(paths["labevents"]) as f:
        header=f.readline();cols=next(csv.reader([header]));ix={n.lower():i for i,n in enumerate(cols)}
        for line in f:
            rows+=1;progress("labevents",rows,started)
            if not any(m in line for m in markers):continue
            try:r=next(csv.reader([line]));hadm=r[ix["hadm_id"]].strip();item=r[ix["itemid"]].strip()
            except Exception:continue
            stay=hadm_map.get(hadm)
            if not stay or item not in labs:continue
            when=parse_dt(r[ix["charttime"]]);v=num(r[ix.get("valuenum",ix.get("value"))])
            if when is None or v is None:continue
            minute=(when-selected[stay]["intime"]).total_seconds()/60
            if not -360<=minute<=120:continue
            target=labs[item]
            if target=="lactate" and .1<=v<=30:values[stay][target].append(v)
            elif target=="creatinine" and .1<=v<=20:values[stay][target].append(v)
    rows=[]
    for stay in selected:
        v=values[stay];rows.append({"encounter_id":stay,"severity_map_min":min(v["map"]) if v["map"] else np.nan,
            "severity_gcs_min":min(v["gcs"]) if v["gcs"] else np.nan,
            "severity_lactate_max":max(v["lactate"]) if v["lactate"] else np.nan,
            "severity_creatinine_max":max(v["creatinine"]) if v["creatinine"] else np.nan})
    return pd.DataFrame(rows)

def broad_surgery(v):
    s=str(v).lower()
    if any(x in s for x in ("cardiac","cabg","coronary","valve","aortic","vascular","cardiothoracic")):return "cardiothoracic_vascular"
    if any(x in s for x in ("neuro","crani","brain","subdural","spine")):return "neurosurgery"
    if any(x in s for x in ("trauma","fracture","orthop","hip","femur")):return "orthopedic_trauma"
    if any(x in s for x in ("abdom","bowel","colon","gastric","hepatic","liver","pancrea","general")):return "general_abdominal"
    return "other_surgery"
def broad_icu(v):
    s=str(v).lower()
    if any(x in s for x in ("cardiac","cticu","csicu","ccu","cvicu")):return "cardiac_icu"
    if any(x in s for x in ("sicu","surgical","trauma","vic")):return "surgical_icu"
    if "neuro" in s:return "neuro_icu"
    if any(x in s for x in ("micu","medical")):return "medical_icu"
    return "other_icu"
def prepare(d):
    d=d.copy()
    for c in ("age","first_temperature_c","n_temperature","temperature_deficit_auc_c_hours","hospital_mortality"):
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d.age=d.age.fillna(d.age.median());d["sex_male"]=d.sex.astype(str).str.lower().str.startswith("m").astype(int)
    d["emergency_admission"]=pd.to_numeric(d.emergency_admission,errors="coerce").fillna(0).astype(int);d.hospital_mortality=d.hospital_mortality.astype(int)
    d["surgery_broad"]=d.surgery_group.map(broad_surgery);d["icu_broad"]=d.icu_type.map(broad_icu);d["encounter_id"]=d.encounter_id.astype(str)
    return d
def z(d,c):
    sd=d[c].std(ddof=0);return (d[c]-d[c].mean())/(sd if pd.notna(sd) and sd>0 else 1)
def base_prepare(d):
    d=d.copy()
    for c in ("age","first_temperature_c","n_temperature","temperature_deficit_auc_c_hours"):d[c+"_z"]=z(d,c)
    return d
def design(d,extra):
    cols=["temperature_deficit_auc_c_hours_z","age_z","sex_male","emergency_admission","first_temperature_c_z","n_temperature_z"]+list(extra)
    X=d[cols].astype(float).copy();X=pd.concat([X,pd.get_dummies(d[["surgery_broad","icu_broad"]].fillna("Unknown"),drop_first=True,dtype=float)],axis=1);X.insert(0,"Intercept",1.0);return X
def logistic(X,y,clusters=None):
    X=np.asarray(X,float);y=np.asarray(y,float);n,p=X.shape;b=np.zeros(p)
    for _ in range(100):
        pr=expit(np.clip(X@b,-30,30));w=np.clip(pr*(1-pr),1e-8,None);h=X.T@(X*w[:,None])+np.eye(p)*1e-8;step=np.linalg.pinv(h)@(X.T@(y-pr));b2=b+step
        if np.max(np.abs(step))<1e-8:b=b2;break
        b=b2
    pr=expit(np.clip(X@b,-30,30));w=np.clip(pr*(1-pr),1e-8,None);bread=np.linalg.pinv(X.T@(X*w[:,None]));res=y-pr
    if clusters is not None and len(set(clusters))>1:
        g=np.asarray(clusters);meat=np.zeros((p,p))
        for k in np.unique(g):score=X[g==k].T@res[g==k];meat+=np.outer(score,score)
        G=len(np.unique(g));meat*=G/(G-1)*(n-1)/max(n-p,1)
    else:
        scores=X*res[:,None];meat=scores.T@scores*n/max(n-p,1)
    cov=bread@meat@bread;se=np.sqrt(np.clip(np.diag(cov),0,None));return b,se
def fit(d,db,label,extra):
    needed=["hospital_mortality","temperature_deficit_auc_c_hours_z","age_z","sex_male","emergency_admission","first_temperature_c_z","n_temperature_z"]+list(extra)
    q=d.dropna(subset=needed).copy();X=design(q,extra);clusters=q.hospital_id.astype(str).values if db=="eICU" else None;b,se=logistic(X,q.hospital_mortality,clusters)
    i=list(X.columns).index("temperature_deficit_auc_c_hours_z");p=2*norm.sf(abs(b[i]/se[i]))
    return {"database":db,"model":label,"n":len(q),"deaths":int(q.hospital_mortality.sum()),"odds_ratio":math.exp(b[i]),"ci_lower":math.exp(b[i]-1.96*se[i]),"ci_upper":math.exp(b[i]+1.96*se[i]),"p_value":p}
def impute_with_missing(d,cols):
    d=d.copy();extra=[]
    for c in cols:
        d[c]=pd.to_numeric(d[c],errors="coerce");miss=c+"_missing";d[miss]=d[c].isna().astype(int);median=d[c].median();d[c]=d[c].fillna(median if pd.notna(median) else 0);zc=c+"_z";d[zc]=z(d,c);extra.extend([zc,miss])
    return d,extra
def load_apache(root,e):
    p=find_file(root,["apachePatientResult"])
    if p is None:raise FileNotFoundError("eICU缺少apachePatientResult")
    a=pd.read_csv(p,low_memory=False);a.columns=[canon(c) for c in a.columns];a["patientunitstayid"]=a.patientunitstayid.astype(str)
    a["_pref"]=a.apacheversion.astype(str).str.lower().map(lambda x:0 if x=="iva" else 1);a=a.sort_values(["patientunitstayid","_pref"]).drop_duplicates("patientunitstayid")
    keep=a[["patientunitstayid","apachescore","acutephysiologyscore"]].rename(columns={"patientunitstayid":"encounter_id"})
    return e.merge(keep,on="encounter_id",how="left")

def main():
    if ERR.exists():ERR.unlink()
    der=find_derived();raw=input(f"derived_rewarming_v3文件夹（直接回车使用：{der or '未找到'}）：").strip().strip('"');der=Path(raw) if raw else der
    raw=input(r"MIMIC-IV根目录（直接回车使用 {config.MIMIC_ROOT.parent}）：").strip().strip('"');mroot=Path(raw) if raw else config.MIMIC_ROOT
    raw=input(r"eICU-CRD根目录（直接回车使用 {config.EICU_ROOT.parent}）：").strip().strip('"');eroot=Path(raw) if raw else config.EICU_ROOT
    if der is None or not der.exists():raise FileNotFoundError("找不到derived_rewarming_v3")
    m=prepare(pd.read_csv(der/"mimic_rewarming_cohort.csv.gz"));e=prepare(pd.read_csv(der/"eicu_rewarming_cohort.csv.gz"))
    selected,hadm_map=map_mimic_stays(mroot,set(m.encounter_id));sev=extract_mimic_severity(mroot,selected,hadm_map);m=m.merge(sev,on="encounter_id",how="left");m=base_prepare(m)
    m.to_csv(HERE/"mimic_severity_internal.csv",index=False)
    severity_cols=["severity_map_min","severity_gcs_min","severity_lactate_max","severity_creatinine_max"]
    coverage=[]
    for c in severity_cols:coverage.append({"database":"MIMIC-IV","variable":c,"available_n":int(m[c].notna().sum()),"cohort_n":len(m),"coverage_percent":100*m[c].notna().mean()})
    m_imp,m_extra=impute_with_missing(m,severity_cols)
    results=[fit(m,"MIMIC-IV","Base corrected model",[]),fit(m_imp,"MIMIC-IV","Expanded admission-severity model",m_extra)]
    complete=m.dropna(subset=severity_cols).copy()
    if len(complete)>=100 and complete.hospital_mortality.sum()>=20:
        for c in severity_cols:complete[c+"_z"]=z(complete,c)
        results.append(fit(complete,"MIMIC-IV","Complete-case admission-severity model",[c+"_z" for c in severity_cols]))
    e=base_prepare(load_apache(eroot,e))
    aps_path=find_file(eroot,["apacheApsVar"])
    if aps_path is not None:
        aps=pd.read_csv(aps_path,usecols=["patientunitstayid","temperature"],low_memory=False)
        aps["patientunitstayid"]=aps.patientunitstayid.astype(str)
        aps=aps.rename(columns={"patientunitstayid":"encounter_id","temperature":"apache_worst_temp_c"})
        e=e.merge(aps,on="encounter_id",how="left")
    e.to_csv(HERE/"eicu_severity_internal.csv",index=False)
    for c in ("apachescore","acutephysiologyscore"):
        e[c]=pd.to_numeric(e[c],errors="coerce").where(lambda x:x>0);coverage.append({"database":"eICU","variable":c,"available_n":int(e[c].notna().sum()),"cohort_n":len(e),"coverage_percent":100*e[c].notna().mean()})
    results.append(fit(e,"eICU","Base model",[]))
    for score,label in (("apachescore","APACHE IVa score"),("acutephysiologyscore","Acute Physiology Score")):
        imp,extra=impute_with_missing(e,[score]);results.append(fit(imp,"eICU",label+" with missing indicator",extra))
        cc=e.dropna(subset=[score]).copy();cc[score+"_z"]=z(cc,score);results.append(fit(cc,"eICU",label+" complete case",[score+"_z"]))
    OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(coverage).to_csv(OUT/"Table_severity_coverage.csv",index=False,encoding="utf-8-sig");pd.DataFrame(results).to_csv(OUT/"Table_severity_adjusted_models.csv",index=False,encoding="utf-8-sig")
    with (OUT/"严重程度校正结果汇总.txt").open("w",encoding="utf-8-sig") as out:
        out.write("V3双库温度负荷研究严重程度校正结果\nMIMIC严重程度窗口：ICU入科0–2小时MAP/GCS及入科前6小时至后2小时乳酸/肌酐；不把体温作为严重程度变量。\n")
        out.write("eICU使用APACHE IVa及Acute Physiology Score；评分本身可能包含体温，因此属于可能过度校正的保守支持性分析。\n")
        out.write("MIMIC扩展模型采用中位数填补加缺失指示变量，因死亡事件数有限，仅作为支持性敏感性分析。\n\n[覆盖率]\n")
        for x in coverage:out.write(f"{x['database']} | {x['variable']}: {x['available_n']:,}/{x['cohort_n']:,} ({x['coverage_percent']:.1f}%)\n")
        out.write("\n[AUC每SD与院内死亡]\n")
        for x in results:out.write(f"{x['database']} | {x['model']}: OR {x['odds_ratio']:.3f} (95%CI {x['ci_lower']:.3f}–{x['ci_upper']:.3f}), P={x['p_value']:.4g}, N={x['n']}, deaths={x['deaths']}\n")
        out.write("\n本文件仅含汇总。若严重程度模型在两库仍同向且置信区间不跨1，可进入论文定稿。\n")
    print("\n分析完成：",OUT);print("请压缩并上传 v3_severity_results；不要上传患者级数据。")

if __name__=="__main__":
    try:main()
    except Exception:
        msg=traceback.format_exc();ERR.write_text(msg,encoding="utf-8-sig");print(msg);print("错误日志：",ERR);input("按回车关闭……");sys.exit(1)
