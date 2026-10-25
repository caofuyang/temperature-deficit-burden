# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import re
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

ROOT=Path(__file__).resolve().parent
SRC=Path(config.REVISION_SOURCE)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
PAT=re.compile(r'norepinephrine|noradrenaline|epinephrine|adrenaline|phenylephrine|dopamine|vasopressin|dobutamine|milrinone',re.I)
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

# MIMIC-IV: flag any vasoactive infusion overlapping ICU hours 0-6.
m=pd.read_csv(SRC/'mimic_rewarming_cohort.csv.gz',dtype={'encounter_id':str})
stay_ids=set(m.encounter_id)
items=pd.read_csv(str(config.MIMIC_ROOT/'icu'/'d_items.csv.gz'))
namecol='label'
drug_items=set(items.loc[items[namecol].astype(str).str.contains(PAT,na=False),'itemid'].astype(str))
intime=pd.read_csv(str(config.MIMIC_ROOT/'icu'/'icustays.csv.gz'),usecols=['stay_id','intime'],dtype={'stay_id':str})
intime=intime[intime.stay_id.isin(stay_ids)].copy(); intime['intime']=pd.to_datetime(intime.intime,errors='coerce')
start_map=intime.set_index('stay_id').intime.to_dict(); mflag=set(); nscan=0
for ch in pd.read_csv(str(config.MIMIC_ROOT/'icu'/'inputevents.csv.gz'),usecols=['stay_id','itemid','starttime','endtime'],dtype={'stay_id':str,'itemid':str},chunksize=750_000):
    nscan+=len(ch); ch=ch[ch.stay_id.isin(stay_ids)&ch.itemid.isin(drug_items)]
    if ch.empty: continue
    ch['starttime']=pd.to_datetime(ch.starttime,errors='coerce'); ch['endtime']=pd.to_datetime(ch.endtime,errors='coerce')
    adm=ch.stay_id.map(start_map); end=adm+pd.Timedelta(hours=6)
    active=(ch.starttime<=end)&(ch.endtime>=adm)
    mflag.update(ch.loc[active,'stay_id'].dropna().astype(str))

# eICU-CRD: flag a named infusion overlapping ICU offsets 0-360 min.
e=pd.read_csv(SRC/'eicu_rewarming_cohort.csv.gz',dtype={'encounter_id':str})
eids=set(e.encounter_id); eflag=set()
inf=str(config.EICU_ROOT/'infusionDrug.csv.gz')
for ch in pd.read_csv(inf,usecols=['patientunitstayid','drugname','infusionoffset'],dtype={'patientunitstayid':str},chunksize=500_000):
    ch=ch[ch.patientunitstayid.isin(eids)&ch.drugname.astype(str).str.contains(PAT,na=False)]
    if ch.empty: continue
    a=pd.to_numeric(ch.infusionoffset,errors='coerce')
    # The released eICU infusion table has offsets for recorded rate entries but no stop offset.
    eflag.update(ch.loc[(a>=0)&(a<=360),'patientunitstayid'].dropna().astype(str))

rows=[]
subgroup_rows=[]
for db,d,flags in [('MIMIC-IV',m,mflag),('eICU-CRD',e,eflag)]:
    d['vasopressor_0_6h']=d.encounter_id.astype(str).isin(flags).astype(int)
    d['age']=pd.to_numeric(d.age,errors='coerce').fillna(pd.to_numeric(d.age,errors='coerce').median())
    d['exposure_z']=(pd.to_numeric(d.temperature_deficit_auc_c_hours)-pd.to_numeric(d.temperature_deficit_auc_c_hours).mean())/pd.to_numeric(d.temperature_deficit_auc_c_hours).std(ddof=0)
    d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int)
    d['surg']=d.surgery_group.map(broad_surgery)
    d['icu']=d.icu_type.map(broad_icu)
    formula='hospital_mortality ~ exposure_z + age + sex_male + emergency_admission + first_temperature_c + n_temperature + vasopressor_0_6h + C(surg) + C(icu)'
    q=d.dropna(subset=['hospital_mortality','exposure_z']).copy()
    if db=='eICU-CRD': fit=smf.glm(formula,q,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':q.hospital_id.astype(str)})
    else: fit=smf.glm(formula,q,family=sm.families.Binomial()).fit(cov_type='HC1')
    b=fit.params.exposure_z; se=fit.bse.exposure_z
    rows.append({'database':db,'n':len(q),'deaths':int(q.hospital_mortality.sum()),'vasopressor_n':int(q.vasopressor_0_6h.sum()),'vasopressor_pct':100*q.vasopressor_0_6h.mean(),'adjusted_OR':np.exp(b),'CI_low':np.exp(b-1.96*se),'CI_high':np.exp(b+1.96*se),'p':fit.pvalues.exposure_z,'input_rows_scanned':nscan if db=='MIMIC-IV' else np.nan})
    # Exploratory modification by early vasoactive support as a shock proxy.
    f_int=formula+' + exposure_z:vasopressor_0_6h'
    if db=='eICU-CRD': fi=smf.glm(f_int,q,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':q.hospital_id.astype(str)})
    else: fi=smf.glm(f_int,q,family=sm.families.Binomial()).fit(cov_type='HC1')
    term='exposure_z:vasopressor_0_6h'; pint=float(fi.pvalues[term])
    for g,label in [(0,'No vasoactive infusion documented'),(1,'Vasoactive infusion documented')]:
        z=q[q.vasopressor_0_6h==g].copy()
        fsub='hospital_mortality ~ exposure_z + age + sex_male + emergency_admission + first_temperature_c + n_temperature + C(surg) + C(icu)'
        if db=='eICU-CRD': fs=smf.glm(fsub,z,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':z.hospital_id.astype(str)})
        else: fs=smf.glm(fsub,z,family=sm.families.Binomial()).fit(cov_type='HC1')
        bs=fs.params.exposure_z; ss=fs.bse.exposure_z
        subgroup_rows.append({'database':db,'subgroup':label,'n':len(z),'deaths':int(z.hospital_mortality.sum()),'OR_per_SD':np.exp(bs),'CI_low':np.exp(bs-1.96*ss),'CI_high':np.exp(bs+1.96*ss),'interaction_p':pint})
pd.DataFrame(rows).to_csv(OUT/'vasopressor_adjusted_models.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame([{'database':'MIMIC-IV','encounter_id':x} for x in sorted(mflag)]+[{'database':'eICU-CRD','encounter_id':x} for x in sorted(eflag)]).to_csv(ROOT/'vasopressor_flags_internal.csv',index=False)
pd.DataFrame(subgroup_rows).to_csv(OUT/'vasopressor_subgroup_interaction.csv',index=False)
print(pd.DataFrame(subgroup_rows).to_string(index=False))
