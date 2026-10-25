"""MIMIC-only exploratory warming-device sensitivity, not a treatment effect."""
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

root=Path(__file__).resolve().parent
d=pd.read_csv(root.parent/'revision_source'/'mimic_rewarming_cohort.csv.gz',dtype={'encounter_id':str})
c=pd.read_csv(root/'feasibility_mimic_charts_internal.csv',dtype={'stay_id':str,'itemid':str})
c=c[(c.offset_min>=0)&(c.offset_min<=360)].copy()
device=c[((c.itemid=='225050') & c.value.eq('On')) |
         ((c.itemid=='224105') & c.value.isin(['Bair Hugger','Blanket','Hot Line','Level One'])) |
         ((c.itemid=='229686') & c.value.isin(['Bair Hugger','Warming Blanket','Blood Warmer']))]
ids=set(device.stay_id)
d['warming_record']=d.encounter_id.isin(ids).astype(int)
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
d['surgery_broad']=d.surgery_group.map(broad_surgery)
d['icu_broad']=d.icu_type.map(broad_icu)
d['sex_male']=d.sex.astype(str).str.lower().str.startswith('m').astype(int)
d['age']=pd.to_numeric(d.age,errors='coerce').fillna(pd.to_numeric(d.age,errors='coerce').median())
d['emergency_admission']=pd.to_numeric(d.emergency_admission,errors='coerce').fillna(0).astype(int)
d['auc_z']=(d.temperature_deficit_auc_c_hours-d.temperature_deficit_auc_c_hours.mean())/d.temperature_deficit_auc_c_hours.std(ddof=0)
f='hospital_mortality ~ age + sex_male + emergency_admission + first_temperature_c + n_temperature + C(surgery_broad) + C(icu_broad) + auc_z'
out=[]
for label,formula in [('core',f),('plus_warming_record',f+' + warming_record')]:
 m=smf.glm(formula,d,family=sm.families.Binomial()).fit(cov_type='HC1')
 term='auc_z';b=m.params[term];se=m.bse[term]
 out.append({'model':label,'n':len(d),'deaths':int(d.hospital_mortality.sum()),'warming_record_count':int(d.warming_record.sum()),'auc_or':np.exp(b),'auc_ci_lo':np.exp(b-1.96*se),'auc_ci_hi':np.exp(b+1.96*se),'auc_p':m.pvalues[term]})
 if label=='plus_warming_record':
  out[-1].update(warming_or=np.exp(m.params['warming_record']),warming_p=m.pvalues['warming_record'])
print(pd.DataFrame(out).to_string(index=False))
print('record flag by mortality',pd.crosstab(d.warming_record,d.hospital_mortality).to_string())
pd.DataFrame(out).to_csv(config.RESULTS_DIR/'mimic_warming_sensitivity.csv',index=False)
