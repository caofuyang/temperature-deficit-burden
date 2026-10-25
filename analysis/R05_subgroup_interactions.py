# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd,numpy as np,statsmodels.formula.api as smf,statsmodels.api as sm
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
base='hospital_mortality ~ exposure_z + age + sex_male + emergency_admission + first_temperature_c + n_temperature + C(surgery_broad) + C(icu_broad)'
rows=[]
for db,file in [('MIMIC-IV','mimic_rewarming_cohort.csv.gz'),('eICU-CRD','eicu_rewarming_cohort.csv.gz')]:
 d=pd.read_csv(S/file);d['age']=pd.to_numeric(d.age,errors='coerce');d['first_temperature_c']=pd.to_numeric(d.first_temperature_c,errors='coerce');d['n_temperature']=pd.to_numeric(d.n_temperature,errors='coerce');d['emergency_admission']=pd.to_numeric(d.emergency_admission,errors='coerce').fillna(0);a=pd.to_numeric(d.temperature_deficit_auc_c_hours);d['exposure_z']=(a-a.mean())/a.std(ddof=0);d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int);d['surgery_broad']=d.surgery_group.map(broad_surgery);d['icu_broad']=d.icu_type.map(broad_icu);d['age75']=(d.age>=75).astype(int);d['cardiothoracic']=(d.surgery_broad=='cardiothoracic_vascular').astype(int);d['sparse_sampling']=(d.n_temperature<=4).astype(int)
 for name,term in [('age >=75','age75'),('emergency admission','emergency_admission'),('cardiothoracic/vascular surgery','cardiothoracic'),('sparse sampling (<=4 readings)','sparse_sampling')]:
  # Keep the original core adjustment. Terms already represented in the core model
  # (emergency status and surgery category) enter only as interactions; otherwise
  # include their main effects as well.
  if term=='emergency_admission' or term=='cardiothoracic': formula=base+f' + exposure_z:{term}'
  else: formula=base+f' + {term} + exposure_z:{term}'
  try:
   if db=='eICU-CRD':fit=smf.glm(formula,d,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.hospital_id.astype(str)})
   else:fit=smf.glm(formula,d,family=sm.families.Binomial()).fit(cov_type='HC1')
   inter=f'exposure_z:{term}' if f'exposure_z:{term}' in fit.params else f'{term}:exposure_z'
   p=float(fit.pvalues[inter]); b=fit.params['exposure_z']
   for val,label in [(0,'reference stratum'),(1,'comparison stratum')]:
    est=b+(fit.params[inter] if val else 0);v=fit.cov_params().loc['exposure_z','exposure_z']+(fit.cov_params().loc[inter,inter]+2*fit.cov_params().loc['exposure_z',inter] if val else 0);se=np.sqrt(max(v,0))
    rows.append({'analysis':name,'database':db,'stratum':label,'n':int((d[term]==val).sum()),'deaths':int(d.loc[d[term]==val,'hospital_mortality'].sum()),'OR_per_SD':np.exp(est),'CI_low':np.exp(est-1.96*se),'CI_high':np.exp(est+1.96*se),'interaction_p':p,'formula':formula})
  except Exception as ex: print('ERROR',db,name,ex)
out=pd.DataFrame(rows);out.to_csv(O/'subgroup_interactions_corrected.csv',index=False);print(out.drop(columns='formula').to_string(index=False))
