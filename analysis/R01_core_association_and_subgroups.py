# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import bootstrap

ROOT=Path(__file__).resolve().parent
SRC=Path(config.REVISION_SOURCE)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)

def paired_prediction_bootstrap(y, pred0, pred1, groups=None, seed=20261021, n_boot=2000):
    """Paired percentile bootstrap of fixed out-of-fold predictions; resample hospitals for multicentre eICU."""
    rng=np.random.default_rng(seed); da=[]; dbri=[]
    if groups is None:
        clusters=None
    else:
        groups=np.asarray(groups).astype(str); clusters=np.unique(groups)
        members={g:np.flatnonzero(groups==g) for g in clusters}
    n=len(y)
    for _ in range(n_boot):
        if groups is None: ix=rng.integers(0,n,n)
        else:
            chosen=rng.choice(clusters,size=len(clusters),replace=True)
            ix=np.concatenate([members[g] for g in chosen])
        if len(np.unique(y[ix]))<2: continue
        da.append(roc_auc_score(y[ix],pred1[ix])-roc_auc_score(y[ix],pred0[ix]))
        dbri.append(brier_score_loss(y[ix],pred1[ix])-brier_score_loss(y[ix],pred0[ix]))
    return np.percentile(da,2.5),np.percentile(da,97.5),np.percentile(dbri,2.5),np.percentile(dbri,97.5)

def broad_surgery(v):
    s=str(v).lower()
    if any(x in s for x in ('cardiac','cabg','coronary','valve','aortic','vascular','cardiothoracic')): return 'cardiothoracic_vascular'
    if any(x in s for x in ('neuro','crani','brain','subdural','spine')): return 'neurosurgery'
    if any(x in s for x in ('trauma','fracture','orthop','hip','femur')): return 'orthopedic_trauma'
    if any(x in s for x in ('abdom','bowel','colon','gastric','hepatic','liver','pancrea','general')): return 'general_abdominal'
    return 'other_surgery'

def broad_icu(v):
    s=str(v).lower()
    if any(x in s for x in ('cardiac','cticu','csicu','ccu','cvicu')): return 'cardiac_icu'
    if any(x in s for x in ('sicu','surgical','trauma','vic')): return 'surgical_icu'
    if 'neuro' in s: return 'neuro_icu'
    if any(x in s for x in ('micu','medical')): return 'medical_icu'
    return 'other_icu'

def prepare(path,db):
    d=pd.read_csv(path)
    numeric=['age','first_temperature_c','n_temperature','temperature_deficit_auc_c_hours','hospital_mortality','emergency_admission']
    for c in numeric: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['age']=d.age.fillna(d.age.median()); d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int)
    d['emergency_admission']=d.emergency_admission.fillna(0).astype(int)
    d['surgery_broad']=d.surgery_group.map(broad_surgery); d['icu_broad']=d.icu_type.map(broad_icu)
    d['database']=db
    d['auc_z']=(d.temperature_deficit_auc_c_hours-d.temperature_deficit_auc_c_hours.mean())/d.temperature_deficit_auc_c_hours.std(ddof=0)
    d['age75']=(d.age>=75).astype(int); d['cardiothoracic']=(d.surgery_broad=='cardiothoracic_vascular').astype(int)
    return d

frames={
 'MIMIC-IV':prepare(SRC/'mimic_rewarming_cohort.csv.gz','MIMIC-IV'),
 'eICU-CRD':prepare(SRC/'eicu_rewarming_cohort.csv.gz','eICU-CRD')}

def fit(formula,d,db):
    if db=='eICU-CRD':
        return smf.logit(formula,d).fit(disp=0,cov_type='cluster',cov_kwds={'groups':d.hospital_id.astype(str)})
    return smf.logit(formula,d).fit(disp=0,cov_type='HC1')

base='hospital_mortality ~ age + sex_male + emergency_admission + first_temperature_c + n_temperature + C(surgery_broad) + C(icu_broad)'
rows=[]
for db,d in frames.items():
    d=d.dropna(subset=['hospital_mortality','auc_z']).copy()
    sd=d.temperature_deficit_auc_c_hours.std(ddof=0)
    for scale,expr in [('per database SD','auc_z'),('per 1 C-hour','temperature_deficit_auc_c_hours')]:
        m=fit(base+f' + {expr}',d,db); term=expr; b=m.params[term]; se=m.bse[term]
        rows.append({'analysis':'primary core adjusted','database':db,'scale':scale,'n':len(d),'deaths':int(d.hospital_mortality.sum()),'OR':np.exp(b),'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':m.pvalues[term],'AUC_SD_C_hours':sd})

# Alternative within-interval hold-last-value integration, with primary boundary carry retained.
for db,d in frames.items():
    long=pd.read_csv(SRC/('mimic_temperature_long.csv.gz' if db=='MIMIC-IV' else 'eicu_temperature_long.csv.gz'))
    long['minute_from_icu_admission']=pd.to_numeric(long.minute_from_icu_admission,errors='coerce')
    long['temperature_c']=pd.to_numeric(long.temperature_c,errors='coerce')
    long=long.dropna(subset=['minute_from_icu_admission','temperature_c'])
    vals={}
    for eid,g in long.groupby('encounter_id',sort=False):
        g=g[(g.minute_from_icu_admission>=0)&(g.minute_from_icu_admission<=360)].sort_values('minute_from_icu_admission')
        if len(g)<2: continue
        t=g.minute_from_icu_admission.to_numpy(float); temp=g.temperature_c.to_numpy(float)
        if t[0]>0: t=np.r_[0,t]; temp=np.r_[temp[0],temp]
        if t[-1]<360: t=np.r_[t,360]; temp=np.r_[temp,temp[-1]]
        # Left-continuous zero-order hold: temperature at interval start is carried to next observation.
        vals[str(eid)]=float(np.sum(np.maximum(0,36-temp[:-1])*np.diff(t)/60))
    d=d.copy(); d['auc_locf']=d.encounter_id.astype(str).map(vals)
    d=d.dropna(subset=['auc_locf']).copy(); d['auc_locf_z']=(d.auc_locf-d.auc_locf.mean())/d.auc_locf.std(ddof=0)
    d['auc_linear_z']=(d.temperature_deficit_auc_c_hours-d.temperature_deficit_auc_c_hours.mean())/d.temperature_deficit_auc_c_hours.std(ddof=0)
    for term,label in [('auc_linear_z','linear/trapezoid'),('auc_locf_z','interior LOCF')]:
        m=fit(base+f' + {term}',d,db);b=m.params[term];se=m.bse[term]
        raw_col='temperature_deficit_auc_c_hours' if term=='auc_linear_z' else 'auc_locf'
        rows.append({'analysis':'interpolation sensitivity','database':db,'scale':label,'n':len(d),'deaths':int(d.hospital_mortality.sum()),'OR':np.exp(b),'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':m.pvalues[term],'AUC_SD_C_hours':d[raw_col].std(ddof=0)})

# Prespecified clinically interpretable subgroups and multiplicative interaction tests.
for db,d in frames.items():
    d=d.dropna(subset=['hospital_mortality','auc_z']).copy()
    d['age75']=(d.age>=75).astype(int); d['cardiothoracic']=(d.surgery_broad=='cardiothoracic_vascular').astype(int)
    d['sparse_sampling']=(d.n_temperature<=4).astype(int)
    for name,term in [('age >=75','age75'),('emergency admission','emergency_admission'),('cardiothoracic/vascular surgery','cardiothoracic'),('sparse sampling (≤4 readings)','sparse_sampling')]:
        try:
            m=fit(base+f' + auc_z*{term}',d,db)
            b=m.params['auc_z'];se=m.bse['auc_z']; inter=f'auc_z:{term}' if f'auc_z:{term}' in m.params else f'{term}:auc_z'
            rows.append({'analysis':'interaction','database':db,'scale':name,'n':len(d),'deaths':int(d.hospital_mortality.sum()),'OR':np.exp(b),'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':m.pvalues.get(inter,np.nan),'AUC_SD_C_hours':np.nan})
            for value,label in [(0,'reference stratum'),(1,'comparison stratum')]:
                bb=b+(m.params.get(inter,0) if value else 0)
                vv=m.cov_params().loc['auc_z','auc_z']+(m.cov_params().loc[inter,inter]+2*m.cov_params().loc['auc_z',inter] if value else 0)
                ss=float(np.sqrt(max(vv,0)))
                rows.append({'analysis':'subgroup-specific OR','database':db,'scale':name+'; '+label,'n':int((d[term]==value).sum()),'deaths':int(d.loc[d[term]==value,'hospital_mortality'].sum()),'OR':np.exp(bb),'CI_low':np.exp(bb-1.96*ss),'CI_high':np.exp(bb+1.96*ss),'p':m.pvalues.get(inter,np.nan),'AUC_SD_C_hours':np.nan})
        except Exception as e:
            rows.append({'analysis':'interaction','database':db,'scale':name,'error':str(e)[:200]})

pd.DataFrame(rows).to_csv(OUT/'association_and_subgroups.csv',index=False)

# Exploratory cross-validated incremental discrimination over the existing core covariates.
perf=[]
for db,d in frames.items():
    y=d.hospital_mortality.astype(int).to_numpy()
    numeric=['age','first_temperature_c','n_temperature','emergency_admission','sex_male']
    cat=['surgery_broad','icu_broad']
    predictions={k:np.full(len(d),np.nan) for k in ['clinical','clinical_plus_auc']}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261021)
    for tr,te in cv.split(d,y):
        for model,cols in [('clinical',numeric+cat),('clinical_plus_auc',numeric+cat+['temperature_deficit_auc_c_hours'])]:
            pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),[c for c in cols if c not in cat]),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')), [c for c in cols if c in cat])])
            clf=make_pipeline(pre,LogisticRegression(max_iter=2000,C=1.0))
            clf.fit(d.iloc[tr][cols],y[tr]); predictions[model][te]=clf.predict_proba(d.iloc[te][cols])[:,1]
    a=predictions['clinical']; b=predictions['clinical_plus_auc']; n=len(y)
    auc0=roc_auc_score(y,a);auc1=roc_auc_score(y,b);br0=brier_score_loss(y,a);br1=brier_score_loss(y,b)
    bg=d.hospital_id.astype(str).to_numpy() if db=='eICU-CRD' else None
    alo,ahi,blo,bhi=paired_prediction_bootstrap(y,a,b,bg)
    perf.append({'database':db,'n':n,'deaths':int(y.sum()),'clinical_AUROC':auc0,'clinical_plus_AUC_AUROC':auc1,'delta_AUROC':auc1-auc0,'delta_AUROC_lo':alo,'delta_AUROC_hi':ahi,'clinical_Brier':br0,'clinical_plus_AUC_Brier':br1,'delta_Brier':br1-br0,'delta_Brier_lo':blo,'delta_Brier_hi':bhi,'bootstrap':'patient-level' if bg is None else 'hospital-cluster paired'})
pd.DataFrame(perf).to_csv(OUT/'incremental_prediction_core_model.csv',index=False)

# Severity-adjusted incremental performance; patient-level severity files stay in work only.
severity_frames={
 'MIMIC-IV':pd.read_csv(ROOT/'mimic_severity_internal.csv',dtype={'encounter_id':str}),
 'eICU-CRD':pd.read_csv(ROOT/'eicu_severity_internal.csv',dtype={'encounter_id':str})}
severity_perf=[]
for db,d0 in frames.items():
    d=severity_frames[db].copy(); d['encounter_id']=d.encounter_id.astype(str)
    d0=d0.copy();d0['encounter_id']=d0.encounter_id.astype(str)
    d=d.merge(d0[['encounter_id','surgery_broad','icu_broad','sex_male','auc_z']],on='encounter_id',how='inner',suffixes=('','_x'))
    y=pd.to_numeric(d.hospital_mortality,errors='coerce').fillna(0).astype(int).to_numpy()
    core_num=['age','first_temperature_c','n_temperature','emergency_admission','sex_male']
    severity_num=(['severity_map_min','severity_gcs_min','severity_lactate_max','severity_creatinine_max'] if db=='MIMIC-IV' else ['apachescore'])
    if db=='eICU-CRD' and 'apachescore' not in d: raise KeyError('APACHE score absent')
    cat=['surgery_broad','icu_broad']
    sev_mean=np.zeros(len(d)); full_mean=np.zeros(len(d)); repetitions=3
    for rep in range(repetitions):
        if db=='eICU-CRD':
            from sklearn.model_selection import StratifiedGroupKFold
            splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261021+rep)
            splits=splitter.split(d,y,groups=d.hospital_id.astype(str))
        else:
            splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261021+rep)
            splits=splitter.split(d,y)
        for tr,te in splits:
            for is_full,target in [(False,sev_mean),(True,full_mean)]:
                nums=core_num+severity_num+(['temperature_deficit_auc_c_hours'] if is_full else [])
                pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat)])
                clf=make_pipeline(pre,LogisticRegression(max_iter=2500,C=1.0))
                clf.fit(d.iloc[tr][nums+cat],y[tr]); target[te]+=clf.predict_proba(d.iloc[te][nums+cat])[:,1]/repetitions
    auc0=roc_auc_score(y,sev_mean); auc1=roc_auc_score(y,full_mean); br0=brier_score_loss(y,sev_mean);br1=brier_score_loss(y,full_mean)
    bg=d.hospital_id.astype(str).to_numpy() if db=='eICU-CRD' else None
    alo,ahi,blo,bhi=paired_prediction_bootstrap(y,sev_mean,full_mean,bg)
    severity_perf.append({'database':db,'n':len(d),'deaths':int(y.sum()),'cv':'3 repeats of 5-fold; hospital grouped for eICU','severity_model_AUROC':auc0,'severity_plus_AUC_AUROC':auc1,'delta_AUROC':auc1-auc0,'delta_AUROC_lo':alo,'delta_AUROC_hi':ahi,'severity_model_Brier':br0,'severity_plus_AUC_Brier':br1,'delta_Brier':br1-br0,'delta_Brier_lo':blo,'delta_Brier_hi':bhi,'bootstrap':'patient-level' if bg is None else 'hospital-cluster paired'})
pd.DataFrame(severity_perf).to_csv(OUT/'incremental_prediction_severity_models.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(perf).to_string(index=False))
print(pd.DataFrame(severity_perf).to_string(index=False))
