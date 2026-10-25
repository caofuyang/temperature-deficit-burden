# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd,numpy as np,re
import statsmodels.formula.api as smf,statsmodels.api as sm
R=Path(__file__).resolve().parent; S=config.REVISION_SOURCE; O=config.RESULTS_DIR
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
pat=re.compile(r'packed red|red blood|blood product|plasma|fresh frozen|platelet|cryoprecip|transfus',re.I)
m=pd.read_csv(S/'mimic_rewarming_cohort.csv.gz',dtype={'encounter_id':str,'hospital_id':str})
ids=set(m.encounter_id)
items=pd.read_csv(str(config.MIMIC_ROOT/'icu'/'d_items.csv.gz'))
mi=set(items.loc[items.label.astype(str).str.contains(pat,na=False),'itemid'].astype(str))
# keep only canonical blood components and auto-transfusion from item dictionary
mi=set(items.loc[items.label.astype(str).str.contains(r'fresh frozen plasma|packed red|packed red blood|autotransfusion|platelets$|cryoprecipitate',case=False,na=False),'itemid'].astype(str))
mi.update(['220970','220996','221193','225168','225170','225171','226369','226371','227071'])
# Match only positive-amount component records overlapping ICU hours 0–6.
cache=R/'mimic_transfusion_ids_positive_internal.csv'
st=pd.read_csv(str(config.MIMIC_ROOT/'icu'/'icustays.csv.gz'),dtype={'stay_id':str},usecols=['stay_id','intime'])
st=st[st.stay_id.isin(ids)].copy();st.intime=pd.to_datetime(st.intime,errors='coerce'); start=st.set_index('stay_id').intime
flagm=set()
for c in pd.read_csv(str(config.MIMIC_ROOT/'icu'/'inputevents.csv.gz'),usecols=['stay_id','itemid','starttime','endtime','amount'],dtype={'stay_id':str,'itemid':str},chunksize=1000000):
 c=c[c.stay_id.isin(ids)&c.itemid.isin(mi)]
 if c.empty:continue
 c['starttime']=pd.to_datetime(c.starttime,errors='coerce');c['endtime']=pd.to_datetime(c.endtime,errors='coerce');c['amount']=pd.to_numeric(c.amount,errors='coerce');a=c.stay_id.map(start)
 keep=c.starttime.notna()&(c.starttime<=a+pd.Timedelta(hours=6))&(c.endtime>=a)&(c.amount>0)
 flagm.update(c.loc[keep,'stay_id'].astype(str))
pd.DataFrame({'stay_id':sorted(flagm)}).to_csv(cache,index=False)
print('MIMIC positive-amount 0-6h transfusion flags',len(flagm),flush=True)
# eICU intakeOutput identifies Blood Products in flowsheet cell path/label.
e=pd.read_csv(S/'eicu_rewarming_cohort.csv.gz',dtype={'encounter_id':str,'hospital_id':str});eids=set(e.encounter_id);flage=set()
p=str(config.EICU_ROOT/'intakeOutput.csv.gz')
for c in pd.read_csv(p,usecols=['patientunitstayid','intakeoutputoffset','cellpath','celllabel','cellvaluetext','cellvaluenumeric'],dtype={'patientunitstayid':str,'cellpath':str,'celllabel':str,'cellvaluetext':str},chunksize=700000):
 c=c[c.patientunitstayid.isin(eids)]
 if c.empty:continue
 desc=(c.cellpath.fillna('')+' '+c.celllabel.fillna('')+' '+c.cellvaluetext.fillna(''))
 c=c[desc.str.contains(r'blood products|transfus|packed red|\bpRBC\b|\bFFP\b|platelet|cryoprecip',case=False,na=False)]
 off=pd.to_numeric(c.intakeoutputoffset,errors='coerce')
 amount=pd.to_numeric(c.cellvaluenumeric,errors='coerce').fillna(0)
 flage.update(c.loc[(off>=0)&(off<=360)&(amount>0),'patientunitstayid'].astype(str))
print('eICU 0-6h transfusion flags',len(flage),flush=True)
rows=[]
for db,d,flags in [('MIMIC-IV',m,flagm),('eICU-CRD',e,flage)]:
 d=d.copy(); d['transfusion_0_6h']=d.encounter_id.astype(str).isin(flags).astype(int); a=pd.to_numeric(d.temperature_deficit_auc_c_hours);d['exposure_z']=(a-a.mean())/a.std(ddof=0)
 d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int);d['age']=pd.to_numeric(d.age,errors='coerce');d['age']=d.age.fillna(d.age.median());d['emergency_admission']=pd.to_numeric(d.emergency_admission,errors='coerce').fillna(0).astype(int)
 d['surg']=d.surgery_group.map(broad_surgery);d['icu']=d.icu_type.map(broad_icu)
 f='hospital_mortality ~ exposure_z + age + sex_male + emergency_admission + first_temperature_c + n_temperature + transfusion_0_6h + C(surg) + C(icu)'
 if db=='eICU-CRD': fit=smf.glm(f,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.hospital_id.astype(str)})
 else: fit=smf.glm(f,d,family=sm.families.Binomial()).fit(cov_type='HC1')
 b=fit.params.exposure_z;se=fit.bse.exposure_z
 rows.append({'database':db,'n':len(d),'deaths':int(d.hospital_mortality.sum()),'transfusion_n':int(d.transfusion_0_6h.sum()),'transfusion_pct':100*d.transfusion_0_6h.mean(),'OR_per_SD':np.exp(b),'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':fit.pvalues.exposure_z})
pd.DataFrame(rows).to_csv(O/'transfusion_sensitivity.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False),flush=True)
