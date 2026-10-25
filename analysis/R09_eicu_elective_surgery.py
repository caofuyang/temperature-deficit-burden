# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import gzip,csv
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
R=Path(__file__).resolve().parent
S=config.REVISION_SOURCE
d=pd.read_csv(S/'eicu_rewarming_cohort.csv.gz')
assert len(d)==2592 and d.encounter_id.is_unique
keys=set(d.encounter_id.astype(str));flags={}
with gzip.open(str(config.EICU_ROOT/'apachePredVar.csv.gz'),'rt',encoding='utf-8-sig',newline='') as f:
 for row in csv.DictReader(f):
  k=row['patientunitstayid']
  if k in keys:
   v=row['electivesurgery']
   if k in flags and flags[k]!=v:raise ValueError('conflicting elective values: '+k)
   flags[k]=v
d['elective_surgery']=pd.to_numeric(d.encounter_id.astype(str).map(flags),errors='coerce')
assert set(d.elective_surgery.dropna().unique())=={0,1}
a=pd.to_numeric(d.temperature_deficit_auc_c_hours,errors='coerce')
d['exposure_z']=(a-a.mean())/a.std(ddof=0)
for c in ['age','first_temperature_c','n_temperature','emergency_admission']:d[c]=pd.to_numeric(d[c],errors='coerce')
d['emergency_admission']=d.emergency_admission.fillna(0)
d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int)
def surg(v):
 s=str(v).lower()
 if any(x in s for x in ('cardiac','cabg','coronary','valve','aortic','vascular','cardiothoracic')):return 'cardiothoracic_vascular'
 if any(x in s for x in ('neuro','crani','brain','subdural','spine')):return 'neurosurgery'
 if any(x in s for x in ('trauma','fracture','orthop','hip','femur')):return 'orthopedic_trauma'
 if any(x in s for x in ('abdom','bowel','colon','gastric','hepatic','liver','pancrea','general')):return 'general_abdominal'
 return 'other_surgery'
def icu(v):
 s=str(v).lower()
 if any(x in s for x in ('cardiac','cticu','csicu','ccu','cvicu')):return 'cardiac_icu'
 if any(x in s for x in ('sicu','surgical','trauma','vic')):return 'surgical_icu'
 if 'neuro' in s:return 'neuro_icu'
 if any(x in s for x in ('micu','medical')):return 'medical_icu'
 return 'other_icu'
d['surgery_broad']=d.surgery_group.map(surg);d['icu_broad']=d.icu_type.map(icu)
f='hospital_mortality ~ exposure_z * elective_surgery + age + sex_male + emergency_admission + first_temperature_c + n_temperature + C(surgery_broad) + C(icu_broad)'
required=['hospital_mortality','exposure_z','elective_surgery','age','sex_male','emergency_admission','first_temperature_c','n_temperature','surgery_broad','icu_broad','hospital_id']
z=d.dropna(subset=required).copy()
fit=smf.glm(f,z,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':z.hospital_id.astype(str)})
term='exposure_z:elective_surgery';cov=fit.cov_params();rows=[]
for v,name in [(0,'non-elective surgery'),(1,'elective surgery')]:
 subset=z[z.elective_surgery==v];b=fit.params['exposure_z']+v*fit.params[term]
 variance=cov.loc['exposure_z','exposure_z']+v*(cov.loc[term,term]+2*cov.loc['exposure_z',term]);se=np.sqrt(max(variance,0))
 rows.append(dict(database='eICU-CRD',subgroup=name,n=len(subset),deaths=int(subset.hospital_mortality.sum()),OR_per_full_cohort_SD=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),interaction_p=float(fit.pvalues[term]),total_cohort_n=len(d),modeled_n=len(z),missing_flag_n=int(d.elective_surgery.isna().sum()),formula=f))
out=config.RESULTS_DIR/'eicu_elective_surgery_interaction.csv';pd.DataFrame(rows).to_csv(out,index=False)
print(pd.DataFrame(rows).drop(columns='formula').to_string(index=False));print('flag counts',d.elective_surgery.value_counts(dropna=False).to_dict())
