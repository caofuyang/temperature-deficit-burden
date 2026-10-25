# -*- coding: utf-8 -*-
"""V3 aggregate primary analysis for the dual-database temperature-burden study."""

from __future__ import annotations
import math, sys, traceback
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

HERE=Path(__file__).resolve().parent
OUT=HERE/"v3_analysis_results"/"primary"
ERROR=HERE/"V3_primary_analysis_error.txt"

try:
    import numpy as np
    import pandas as pd
    from scipy.special import expit
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    ERROR.write_text("缺少Python统计包："+str(exc)+"\n请运行：py -3 -m pip install pandas numpy scipy matplotlib seaborn\n",encoding="utf-8-sig")
    print(ERROR.read_text(encoding="utf-8-sig"));input("按回车关闭……");sys.exit(1)


def find_derived():
    direct=[HERE/"derived_rewarming_v3",HERE.parent/"derived_rewarming_v3"]
    downloads=Path.home()/"Downloads"
    if downloads.exists():
        try: direct.extend(downloads.rglob("derived_rewarming_v3"))
        except OSError: pass
    valid=[]
    for p in direct:
        if (p/"mimic_rewarming_cohort.csv.gz").exists() and (p/"eicu_rewarming_cohort.csv.gz").exists():
            valid.append(p)
    if not valid:return None
    return max(valid,key=lambda p:(p/"mimic_rewarming_cohort.csv.gz").stat().st_mtime)


def broad_surgery(value):
    s=str(value).lower()
    if any(x in s for x in ("cardiac","cabg","coronary","valve","aortic","vascular","cardiothoracic")):return "cardiothoracic_vascular"
    if any(x in s for x in ("neuro","crani","brain","subdural","spine")):return "neurosurgery"
    if any(x in s for x in ("trauma","fracture","orthop","hip","femur")):return "orthopedic_trauma"
    if any(x in s for x in ("abdom","bowel","colon","gastric","hepatic","liver","pancrea","general")):return "general_abdominal"
    return "other_surgery"


def broad_icu(value):
    s=str(value).lower()
    if any(x in s for x in ("cardiac","cticu","csicu","ccu")):return "cardiac_icu"
    if any(x in s for x in ("sicu","surgical","trauma","vic")):return "surgical_icu"
    if any(x in s for x in ("neuro",)):return "neuro_icu"
    if any(x in s for x in ("micu","medical")):return "medical_icu"
    return "other_icu"


def trajectory_group(row):
    if int(row.get("overshoot_hyperthermia",0))==1:return "overshoot"
    if int(row.get("persistent_hypothermia_6h",0))==1:return "persistent"
    t=row.get("time_to_normothermia_hours",np.nan)
    if pd.notna(t) and float(t)<=2:return "rapid_after_first_low"
    return "delayed"


def prepare(df):
    df=df.copy()
    numeric=["age","first_temperature_c","n_temperature","temperature_deficit_auc_c_hours","rewarming_slope_c_per_hour",
             "time_to_normothermia_hours","icu_los_days","hospital_los_days","hospital_mortality","first_temperature_minute"]
    for c in numeric:df[c]=pd.to_numeric(df[c],errors="coerce")
    df["age"]=df["age"].fillna(df["age"].median())
    df["sex_male"]=df["sex"].astype(str).str.lower().str.startswith("m").astype(int)
    df["emergency_admission"]=pd.to_numeric(df["emergency_admission"],errors="coerce").fillna(0).astype(int)
    df["hospital_mortality"]=df["hospital_mortality"].astype(int)
    df["surgery_broad"]=df["surgery_group"].map(broad_surgery)
    df["icu_broad"]=df["icu_type"].map(broad_icu)
    df["trajectory_group"]=df.apply(trajectory_group,axis=1)
    for c in ("age","first_temperature_c","n_temperature","temperature_deficit_auc_c_hours","rewarming_slope_c_per_hour"):
        sd=df[c].std(ddof=0);df[c+"_z"]=(df[c]-df[c].mean())/(sd if sd and np.isfinite(sd) else 1)
    return df


def design(df,exposures):
    base=["age_z","sex_male","emergency_admission","first_temperature_c_z","n_temperature_z"]
    X=df[list(exposures)+base].astype(float).copy()
    cats=pd.get_dummies(df[["surgery_broad","icu_broad"]].fillna("Unknown"),drop_first=True,dtype=float)
    X=pd.concat([X,cats],axis=1)
    X.insert(0,"Intercept",1.0)
    return X


def logistic_fit(X,y,clusters=None,max_iter=100):
    X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float);n,p=X.shape;beta=np.zeros(p)
    for _ in range(max_iter):
        prob=expit(np.clip(X@beta,-30,30));w=np.clip(prob*(1-prob),1e-8,None)
        h=X.T@(X*w[:,None])+np.eye(p)*1e-8;g=X.T@(y-prob)
        step=np.linalg.pinv(h)@g;beta_new=beta+step
        if np.max(np.abs(step))<1e-8:beta=beta_new;break
        beta=beta_new
    prob=expit(np.clip(X@beta,-30,30));w=np.clip(prob*(1-prob),1e-8,None)
    bread=np.linalg.pinv(X.T@(X*w[:,None]))
    resid=y-prob
    if clusters is not None and len(set(clusters))>1:
        groups=np.asarray(clusters);meat=np.zeros((p,p))
        for g in np.unique(groups):
            score=X[groups==g].T@resid[groups==g];meat+=np.outer(score,score)
        G=len(np.unique(groups));meat*=G/(G-1)*(n-1)/max(n-p,1)
    else:
        scores=X*resid[:,None];meat=scores.T@scores*n/max(n-p,1)
    cov=bread@meat@bread;se=np.sqrt(np.clip(np.diag(cov),0,None))
    ll=float(np.sum(y*np.log(np.clip(prob,1e-12,1))+(1-y)*np.log(np.clip(1-prob,1e-12,1))))
    return beta,se,ll


def fit_one(df,database,model_name,exposure_cols,report_cols=None):
    needed=list(exposure_cols)+["hospital_mortality","age_z","sex_male","emergency_admission","first_temperature_c_z","n_temperature_z"]
    d=df.dropna(subset=needed).copy();X=design(d,exposure_cols);y=d["hospital_mortality"]
    clusters=d["hospital_id"].astype(str).values if database=="eICU" else None
    beta,se,ll=logistic_fit(X,y,clusters)
    names=list(X.columns);report_cols=report_cols or list(exposure_cols);rows=[]
    for col in report_cols:
        i=names.index(col);b=beta[i];s=se[i];z=b/s if s>0 else np.nan;p=2*norm.sf(abs(z)) if np.isfinite(z) else np.nan
        rows.append({"database":database,"model":model_name,"exposure":col,"n":len(d),"deaths":int(y.sum()),
            "log_or":b,"se":s,"odds_ratio":math.exp(b),"ci_lower":math.exp(b-1.96*s),"ci_upper":math.exp(b+1.96*s),"p_value":p,"log_likelihood":ll})
    return rows


def model_results(df,database):
    rows=[]
    rows+=fit_one(df,database,"Primary: temperature deficit AUC per SD",["temperature_deficit_auc_c_hours_z"])
    rows+=fit_one(df,database,"Rewarming slope per SD",["rewarming_slope_c_per_hour_z"])
    rows+=fit_one(df,database,"Persistent hypothermia",["persistent_hypothermia_6h"])
    rows+=fit_one(df,database,"Overshoot hyperthermia",["overshoot_hyperthermia"])
    g=pd.get_dummies(df["trajectory_group"],prefix="trajectory",dtype=float)
    for col in ("trajectory_delayed","trajectory_persistent","trajectory_overshoot"):
        if col not in g:g[col]=0.0
    d=pd.concat([df,g],axis=1)
    rows+=fit_one(d,database,"Trajectory groups (reference: normothermia within 2 h after first low temperature)",["trajectory_delayed","trajectory_persistent","trajectory_overshoot"])
    return rows


def sensitivity(df,database):
    specs={"At least 4 temperatures":df[df.n_temperature>=4],"First temperature within 1 h":df[df.first_temperature_minute<=60],
           "Exclude overshoot":df[df.overshoot_hyperthermia==0],"Non-cardiothoracic surgery":df[df.surgery_broad!="cardiothoracic_vascular"]}
    rows=[]
    for name,d in specs.items():rows+=fit_one(d,database,name,["temperature_deficit_auc_c_hours_z"])
    return rows


def meta(models):
    out=[]
    primary=models[(models.model=="Primary: temperature deficit AUC per SD")]
    for exposure,grp in primary.groupby("exposure"):
        if len(grp)!=2:continue
        w=1/grp.se.pow(2);pooled=(w*grp.log_or).sum()/w.sum();se=math.sqrt(1/w.sum())
        q=float((w*(grp.log_or-pooled)**2).sum());i2=max(0,(q-(len(grp)-1))/q)*100 if q>0 else 0
        out.append({"exposure":exposure,"pooled_odds_ratio":math.exp(pooled),"ci_lower":math.exp(pooled-1.96*se),
                    "ci_upper":math.exp(pooled+1.96*se),"p_value":2*norm.sf(abs(pooled/se)),"Q":q,"I2_percent":i2,
                    "note":"Fixed-effect summary; database-specific estimates are primary"})
    return pd.DataFrame(out)


def descriptive(frames):
    rows=[]
    for db,df in frames.items():
        for group,g in [("Overall",df)]+list(df.groupby("trajectory_group")):
            rows.append({"database":db,"group":group,"n":len(g),"deaths":int(g.hospital_mortality.sum()),
                "mortality_percent":100*g.hospital_mortality.mean(),"age_mean":g.age.mean(),"female_percent":100*(1-g.sex_male).mean(),
                "emergency_percent":100*g.emergency_admission.mean(),"first_temp_mean":g.first_temperature_c.mean(),
                "auc_median":g.temperature_deficit_auc_c_hours.median(),"slope_mean":g.rewarming_slope_c_per_hour.mean(),
                "icu_los_median":g.icu_los_days.median(),"hospital_los_median":g.hospital_los_days.median()})
    return pd.DataFrame(rows)


def plot_profiles(frames,long_frames):
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    palette={"rapid_after_first_low":"#2b8cbe","delayed":"#7bccc4","persistent":"#d95f0e","overshoot":"#7a0177"}
    for ax,(db,df) in zip(axes,frames.items()):
        long=long_frames[db].merge(df[["encounter_id","trajectory_group"]],on="encounter_id",how="inner")
        long["time_bin"]=(long.minute_from_icu_admission/30).round()*0.5
        prof=long.groupby(["trajectory_group","time_bin"],as_index=False).temperature_c.mean()
        for group,g in prof.groupby("trajectory_group"):
            ax.plot(g.time_bin,g.temperature_c,marker="o",ms=3,label=group,color=palette.get(group))
        ax.axhline(36,color="black",ls="--",lw=1);ax.set_title(db);ax.set_xlabel("Hours from ICU admission");ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean temperature (°C)");axes[1].legend(title="Trajectory")
    fig.tight_layout();fig.savefig(OUT/"Figure1_trajectory_profiles.png",dpi=300);plt.close(fig)


def plot_forest(models):
    d=models[(models.model.isin(["Primary: temperature deficit AUC per SD","Rewarming slope per SD","Persistent hypothermia","Overshoot hyperthermia"]))].copy()
    labels=[];ys=[];fig,ax=plt.subplots(figsize=(8,6));y=0
    for model in d.model.unique():
        for db in ("MIMIC-IV","eICU"):
            r=d[(d.model==model)&(d.database==db)]
            if r.empty:continue
            r=r.iloc[0];ax.errorbar(r.odds_ratio,y,xerr=[[r.odds_ratio-r.ci_lower],[r.ci_upper-r.odds_ratio]],fmt="o",capsize=3)
            labels.append(f"{model} — {db}");ys.append(y);y+=1
        y+=.5
    ax.axvline(1,color="black",ls="--",lw=1);ax.set_xscale("log");ax.set_yticks(ys);ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel("Adjusted odds ratio for hospital mortality (95% CI)");ax.grid(axis="x",alpha=.2);fig.tight_layout()
    fig.savefig(OUT/"Figure2_adjusted_forest_plot.png",dpi=300);plt.close(fig)


def plot_auc_bins(frames):
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(db,df) in zip(axes,frames.items()):
        d=df.copy();d["quintile"]=pd.qcut(d.temperature_deficit_auc_c_hours,5,duplicates="drop")
        g=d.groupby("quintile",observed=True).hospital_mortality.agg(["mean","count","sum"]).reset_index()
        ax.plot(range(1,len(g)+1),100*g["mean"],marker="o");ax.set_title(db);ax.set_xlabel("Temperature-deficit AUC quintile")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Observed hospital mortality (%)");fig.tight_layout();fig.savefig(OUT/"Figure3_auc_quintile_mortality.png",dpi=300);plt.close(fig)


def main():
    if ERROR.exists():ERROR.unlink()
    auto=find_derived();default=str(auto) if auto else "未自动找到"
    raw=input(f"derived_rewarming文件夹（直接回车使用：{default}）：").strip().strip('"')
    source=Path(raw) if raw else auto
    if source is None or not source.exists():raise FileNotFoundError("找不到derived_rewarming文件夹")
    OUT.mkdir(parents=True,exist_ok=True)
    frames={"MIMIC-IV":prepare(pd.read_csv(source/"mimic_rewarming_cohort.csv.gz")),"eICU":prepare(pd.read_csv(source/"eicu_rewarming_cohort.csv.gz"))}
    long_frames={"MIMIC-IV":pd.read_csv(source/"mimic_temperature_long.csv.gz"),"eICU":pd.read_csv(source/"eicu_temperature_long.csv.gz")}
    desc=descriptive(frames);desc.to_csv(OUT/"Table1_descriptive_by_trajectory.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([
        {"database":db,"n":len(df),"deaths":int(df.hospital_mortality.sum()),
         "auc_mean":df.temperature_deficit_auc_c_hours.mean(),
         "auc_sd_population":df.temperature_deficit_auc_c_hours.std(ddof=0),
         "auc_median":df.temperature_deficit_auc_c_hours.median(),
         "auc_q1":df.temperature_deficit_auc_c_hours.quantile(.25),
         "auc_q3":df.temperature_deficit_auc_c_hours.quantile(.75)}
        for db,df in frames.items()
    ]).to_csv(OUT/"Table0_cohort_and_AUC_scaling.csv",index=False,encoding="utf-8-sig")
    model_rows=[];sens_rows=[]
    for db,df in frames.items():model_rows+=model_results(df,db);sens_rows+=sensitivity(df,db)
    models=pd.DataFrame(model_rows);sens=pd.DataFrame(sens_rows)
    models.to_csv(OUT/"Table2_adjusted_models.csv",index=False,encoding="utf-8-sig")
    sens.to_csv(OUT/"Table3_sensitivity_analyses.csv",index=False,encoding="utf-8-sig")
    meta(models).to_csv(OUT/"Table4_fixed_effect_summary.csv",index=False,encoding="utf-8-sig")
    plot_profiles(frames,long_frames);plot_forest(models);plot_auc_bins(frames)
    with (OUT/"正式统计分析汇总.txt").open("w",encoding="utf-8-sig") as out:
        out.write("V3双数据库术后温度负荷正式统计分析汇总\n")
        out.write("说明：所有模型分别在MIMIC-IV和eICU拟合；eICU标准误按医院聚类。效应为关联，不解释为因果。\n")
        for db,df in frames.items():
            out.write(f"\n[{db}] N={len(df):,}, deaths={int(df.hospital_mortality.sum()):,}\n")
            out.write(f"温度负荷AUC的总体标准差：{df.temperature_deficit_auc_c_hours.std(ddof=0):.4f} °C·h\n")
            out.write("轨迹："+", ".join(f"{k}={v}" for k,v in df.trajectory_group.value_counts().to_dict().items())+"\n")
            for _,r in models[models.database==db].iterrows():
                out.write(f"{r['model']} | {r['exposure']}: OR {r.odds_ratio:.3f} (95%CI {r.ci_lower:.3f}–{r.ci_upper:.3f}), P={r.p_value:.4g}, n={int(r.n)}\n")
        out.write("\n[重要限制]\n当前模型已调整年龄、性别、急诊入院、初始体温、测量频次、手术大类和ICU类型。\n")
        out.write("数据库特异的急性严重度评分将在下一步作为支持性敏感性分析；当前结果不得表述为因果效应。\n")
        out.write("\n输出文件均为汇总表或图片，不包含患者标识，可整体压缩后上传。\n")
    print("\n分析完成。结果文件夹：",OUT)
    print("Aggregate outputs written to v3_analysis_results. Patient-level intermediates are never written to results/.")


if __name__=="__main__":
    try:main()
    except Exception:
        msg=traceback.format_exc();ERROR.write_text(msg,encoding="utf-8-sig");print(msg);print("错误日志：",ERROR);input("按回车关闭……");sys.exit(1)
