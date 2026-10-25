"""Reassess post-landmark creatinine AKI with true temporal ordering and ICU censoring.

Uses exact blood creatinine items/units. The 48-hour +0.3 mg/dL comparison
includes both pre- and post-landmark earlier values. Pre-landmark prevalence
requires a rise from an earlier measurement rather than max/min in any order.
"""
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

R=Path(__file__).resolve().parent; S=config.REVISION_SOURCE; O=config.RESULTS_DIR
M=config.MIMIC_ROOT; E=Path(config.EICU_ROOT)

def broad_surgery(v):
 s=str(v).lower()
 if any(x in s for x in ('cardiac','cabg','coronary','valve','aortic','vascular','cardiothoracic')):return 'cardiothoracic_vascular'
 if any(x in s for x in ('neuro','crani','brain','subdural','spine')):return 'neurosurgery'
 if any(x in s for x in ('trauma','fracture','orthop','hip','femur')):return 'orthopedic_trauma'
 if any(x in s for x in ('abdom','bowel','colon','gastric','hepatic','liver','pancrea','general')):return 'general_abdominal'
 return 'other_surgery'
def broad_icu(v):
 s=str(v).lower()
 if any(x in s for x in ('cardiac','cticu','csicu','ccu','cvicu')):return 'cardiac_icu'
 if any(x in s for x in ('sicu','surgical','trauma','vic')):return 'surgical_icu'
 if 'neuro' in s:return 'neuro_icu'
 if any(x in s for x in ('micu','medical')):return 'medical_icu'
 return 'other_icu'

m=pd.read_csv(S/'mimic_rewarming_cohort.csv.gz',dtype={'encounter_id':str})
e=pd.read_csv(S/'eicu_rewarming_cohort.csv.gz',dtype={'encounter_id':str,'hospital_id':str})
st=pd.read_csv(M/'icu'/'icustays.csv.gz',usecols=['stay_id','subject_id','hadm_id','intime','outtime'],dtype={'stay_id':str,'subject_id':str,'hadm_id':str})
st=st[st.stay_id.isin(set(m.encounter_id))].copy()
st['intime']=pd.to_datetime(st.intime,errors='coerce'); st['outtime']=pd.to_datetime(st.outtime,errors='coerce')
assert st.stay_id.nunique()==len(m)
st['icu_end_min']=(st.outtime-st.intime).dt.total_seconds()/60
subj=set(st.subject_id); hadm=set(st.hadm_id); parts=[]
for i,c in enumerate(pd.read_csv(M/'hosp'/'labevents.csv.gz',
    usecols=['subject_id','hadm_id','itemid','charttime','valuenum','valueuom'],
    dtype={'subject_id':str,'hadm_id':str,'itemid':str,'valueuom':str},chunksize=1_500_000)):
 c=c[c.subject_id.isin(subj)&c.hadm_id.isin(hadm)&c.itemid.isin(['50912','52546'])]
 c=c[c.valueuom.fillna('').str.lower().isin(['mg/dl',''])]
 if len(c):parts.append(c)
 if i and i%20==0:print('MIMIC labevents chunks',i,'selected',sum(map(len,parts)),flush=True)
ml=pd.concat(parts,ignore_index=True)
ml['charttime']=pd.to_datetime(ml.charttime,errors='coerce')
ml['creatinine']=pd.to_numeric(ml.valuenum,errors='coerce')
ml=ml.merge(st[['stay_id','subject_id','hadm_id','intime','icu_end_min']],on=['subject_id','hadm_id'],how='inner')
ml['minute']=(ml.charttime-ml.intime).dt.total_seconds()/60
ml=ml[ml.creatinine.between(.1,30)&ml.minute.between(-2880,10080)&(ml.minute<=ml.icu_end_min)]
ml=ml.rename(columns={'stay_id':'encounter_id'})[['encounter_id','minute','creatinine']]
print('MIMIC retained within ICU',len(ml),'stays',ml.encounter_id.nunique(),flush=True)

pat=pd.read_csv(E/'patient.csv.gz',usecols=['patientunitstayid','unitdischargeoffset'],dtype={'patientunitstayid':str})
pat=pat[pat.patientunitstayid.isin(set(e.encounter_id))].set_index('patientunitstayid')
pat['unitdischargeoffset']=pd.to_numeric(pat.unitdischargeoffset,errors='coerce')
parts=[]
for i,c in enumerate(pd.read_csv(E/'lab.csv.gz',
    usecols=['patientunitstayid','labresultoffset','labname','labresult','labmeasurenamesystem','labmeasurenameinterface'],
    dtype=str,chunksize=700000)):
 c=c[c.patientunitstayid.isin(set(e.encounter_id))]
 if c.empty:continue
 name=c.labname.fillna('').str.strip().str.casefold()
 system=c.labmeasurenamesystem.fillna('').str.strip().str.casefold()
 interface=c.labmeasurenameinterface.fillna('').str.strip().str.casefold()
 c=c[(name=='creatinine')&(system=='mg/dl')&(interface.isin(['mg/dl','']))]
 if len(c):parts.append(c)
 if i and i%20==0:print('eICU labs chunks',i,'selected',sum(map(len,parts)),flush=True)
el=pd.concat(parts,ignore_index=True)
el['minute']=pd.to_numeric(el.labresultoffset,errors='coerce')
el['creatinine']=pd.to_numeric(el.labresult,errors='coerce')
el['icu_end_min']=el.patientunitstayid.map(pat.unitdischargeoffset)
el=el[el.creatinine.between(.1,30)&el.minute.between(-2880,10080)&(el.minute<=el.icu_end_min)]
el=el.rename(columns={'patientunitstayid':'encounter_id'})[['encounter_id','minute','creatinine']]
print('eICU retained within ICU',len(el),'stays',el.encounter_id.nunique(),flush=True)

def assess(g):
 g=g.sort_values(['minute','creatinine']); t=g.minute.to_numpy();v=g.creatinine.to_numpy()
 pre=np.where(t<=360)[0];post=np.where(t>360)[0]
 if not len(pre) or not len(post):return None
 base=float(v[pre].min()); prevalent=False
 for j in pre:
  earlier_48h=pre[(t[pre]<t[j])&(t[j]-t[pre]<=2880)]
  earlier_7d=pre[(t[pre]<t[j])&(t[j]-t[pre]<=10080)]
  if ((len(earlier_48h) and v[j]-v[earlier_48h].min()>=.3-1e-9)
      or (len(earlier_7d) and v[j]>=1.5*v[earlier_7d].min()-1e-9)):
   prevalent=True;break
 if prevalent:return {'baseline':base,'prevalent':1,'aki':np.nan,'n_pre':len(pre),'n_post':len(post)}
 event=False
 for j in post:
  earlier_48h=np.where((t<t[j])&(t[j]-t<=2880))[0]
  earlier_7d=np.where((t<t[j])&(t[j]-t<=10080))[0]
  if len(earlier_48h) and v[j]-v[earlier_48h].min()>=.3-1e-9:event=True
  if len(earlier_7d) and v[j]>=1.5*v[earlier_7d].min()-1e-9:event=True
 return {'baseline':base,'prevalent':0,'aki':int(event),'n_pre':len(pre),'n_post':len(post)}

audit=[];models=[]
for db,cohort,labs,stem in [('MIMIC-IV',m,ml,'mimic'),('eICU-CRD',e,el,'eicu')]:
 status=[]
 for enc,g in labs.groupby('encounter_id',sort=False):
  x=assess(g)
  if x is not None:x['encounter_id']=str(enc);status.append(x)
 a=pd.DataFrame(status)
 linked=cohort.merge(a,on='encounter_id',how='inner',validate='one_to_one')
 risk=linked[linked.prevalent==0].copy()
 risk.to_csv(O/f'{stem}_aki_analysis_internal.csv',index=False)
 audit.append({'database':db,'cohort_n':len(cohort),'linked_pre_and_post_n':len(linked),
               'prevalent_observed_by_6h':int(linked.prevalent.sum()),'at_risk_n':len(risk),'incident_events':int(risk.aki.sum()),
               'lab_rows_within_icu':len(labs)})
 d=risk.copy();d['age']=pd.to_numeric(d.age,errors='coerce');d['age']=d.age.fillna(d.age.median())
 d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int)
 d['emergency_admission']=pd.to_numeric(d.emergency_admission,errors='coerce').fillna(0).astype(int)
 d['baseline_creatinine']=pd.to_numeric(d.baseline,errors='coerce')
 full_sd=pd.to_numeric(cohort.temperature_deficit_auc_c_hours).std(ddof=0)
 d['exposure_z']=(pd.to_numeric(d.temperature_deficit_auc_c_hours)-pd.to_numeric(cohort.temperature_deficit_auc_c_hours).mean())/full_sd
 d['surg']=d.surgery_group.map(broad_surgery);d['icu']=d.icu_type.map(broad_icu)
 formula='aki ~ exposure_z + age + sex_male + emergency_admission + first_temperature_c + n_temperature + baseline_creatinine + C(surg) + C(icu)'
 if db=='eICU-CRD':fit=smf.glm(formula,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.hospital_id.astype(str)})
 else:fit=smf.glm(formula,d,family=sm.families.Binomial()).fit(cov_type='HC1')
 b=fit.params.exposure_z;se=fit.bse.exposure_z
 models.append({'database':db,'n_at_risk':len(d),'aki_events':int(d.aki.sum()),'OR_per_full_cohort_SD':np.exp(b),
                'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':fit.pvalues.exposure_z,
                'full_cohort_auc_SD':full_sd,'definition':'creatinine-only KDIGO; ordered 48h comparisons; ICU-censored to day 7; core broad covariates plus baseline creatinine'})
 print(db,'linked',len(linked),'early observed AKI',int(linked.prevalent.sum()),'risk',len(d),'events',int(d.aki.sum()),
       'OR',models[-1]['OR_per_full_cohort_SD'],'CI',models[-1]['CI_low'],models[-1]['CI_high'],'p',models[-1]['p'],flush=True)
pd.DataFrame(audit).to_csv(O/'aki_event_definition_audit.csv',index=False)
pd.DataFrame(models).to_csv(O/'incident_aki_models.csv',index=False)
