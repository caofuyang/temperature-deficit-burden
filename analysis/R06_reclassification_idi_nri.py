"""Exploratory out-of-fold IDI and continuous NRI for reviewer R1-6.

Predictions use the same 3x5-fold severity models as reviewer_analyses.py.
Bootstrap intervals condition on the fitted out-of-fold predictions.
"""
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,brier_score_loss

ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'revision_source'
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
def metrics(y,a,b):
 event=y==1; nonevent=~event
 idi=(b[event].mean()-b[nonevent].mean())-(a[event].mean()-a[nonevent].mean())
 delta=b-a
 nri_event=np.mean(delta[event]>0)-np.mean(delta[event]<0)
 nri_nonevent=np.mean(delta[nonevent]<0)-np.mean(delta[nonevent]>0)
 return idi,nri_event,nri_nonevent,nri_event+nri_nonevent
out=[]
for db,stem in [('MIMIC-IV','mimic'),('eICU-CRD','eicu')]:
 c=pd.read_csv(SRC/f'{stem}_rewarming_cohort.csv.gz',dtype={'encounter_id':str})
 d=pd.read_csv(ROOT/f'{stem}_severity_internal.csv',dtype={'encounter_id':str})
 c['surgery_broad']=c.surgery_group.map(broad_surgery)
 c['icu_broad']=c.icu_type.map(broad_icu)
 c['sex_male']=c.sex.astype(str).str.lower().str.startswith('m').astype(int)
 d=d.merge(c[['encounter_id','surgery_broad','icu_broad','sex_male']],on='encounter_id',how='inner',suffixes=('','_x'))
 y=pd.to_numeric(d.hospital_mortality,errors='coerce').fillna(0).astype(int).to_numpy()
 core=['age','first_temperature_c','n_temperature','emergency_admission','sex_male']
 sev=['severity_map_min','severity_gcs_min','severity_lactate_max','severity_creatinine_max'] if db=='MIMIC-IV' else ['apachescore']
 cat=['surgery_broad','icu_broad']
 p0=np.zeros(len(d));p1=np.zeros(len(d)); reps=3
 for rep in range(reps):
  if db=='eICU-CRD':
   split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261021+rep).split(d,y,d.hospital_id.astype(str))
  else:
   split=StratifiedKFold(n_splits=5,shuffle=True,random_state=20261021+rep).split(d,y)
  for tr,te in split:
   for use_auc,target in [(False,p0),(True,p1)]:
    nums=core+sev+(['temperature_deficit_auc_c_hours'] if use_auc else [])
    pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),nums),('cat',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore')),cat)])
    model=make_pipeline(pre,LogisticRegression(max_iter=2500,C=1.0))
    model.fit(d.iloc[tr][nums+cat],y[tr]);target[te]+=model.predict_proba(d.iloc[te][nums+cat])[:,1]/reps
 if np.isnan(p0).any() or np.isnan(p1).any(): raise ValueError('Missing predictions')
 value=metrics(y,p0,p1)
 rng=np.random.default_rng(20261023)
 groups=d.hospital_id.astype(str).to_numpy() if db=='eICU-CRD' else None
 members={g:np.where(groups==g)[0] for g in np.unique(groups)} if groups is not None else None
 boots=[]
 for _ in range(2000):
  if members is None: ix=rng.integers(len(d),size=len(d))
  else: ix=np.concatenate([members[g] for g in rng.choice(list(members),size=len(members),replace=True)])
  if len(np.unique(y[ix]))<2: continue
  boots.append(metrics(y[ix],p0[ix],p1[ix]))
 lo,hi=np.percentile(boots,[2.5,97.5],axis=0)
 row={'database':db,'n':len(d),'deaths':int(y.sum()),'IDI':value[0],'IDI_lo':lo[0],'IDI_hi':hi[0],
      'NRI_event':value[1],'NRI_event_lo':lo[1],'NRI_event_hi':hi[1],
      'NRI_nonevent':value[2],'NRI_nonevent_lo':lo[2],'NRI_nonevent_hi':hi[2],
      'continuous_NRI':value[3],'continuous_NRI_lo':lo[3],'continuous_NRI_hi':hi[3],
      'mean_risk_event_baseline':p0[y==1].mean(),'mean_risk_event_plus_auc':p1[y==1].mean(),
      'mean_risk_nonevent_baseline':p0[y==0].mean(),'mean_risk_nonevent_plus_auc':p1[y==0].mean(),
      'severity_auroc':roc_auc_score(y,p0),'plus_auc_auroc':roc_auc_score(y,p1),
      'severity_brier':brier_score_loss(y,p0),'plus_auc_brier':brier_score_loss(y,p1)}
 prior=pd.read_csv(ROOT/'results'/'incremental_prediction_severity_models.csv')
 check=prior[prior.database.eq(db)].iloc[0]
 for actual,reported in [('severity_auroc','severity_model_AUROC'),('plus_auc_auroc','severity_plus_AUC_AUROC'),
                         ('severity_brier','severity_model_Brier'),('plus_auc_brier','severity_plus_AUC_Brier')]:
  if abs(row[actual]-check[reported])>1e-10:raise ValueError(f'{db} {actual} does not reproduce prior performance result')
 out.append(row)
 print(pd.DataFrame([row]).to_string(index=False),flush=True)
pd.DataFrame(out).to_csv(config.RESULTS_DIR/'reclassification_idi_continuous_nri.csv',index=False)
