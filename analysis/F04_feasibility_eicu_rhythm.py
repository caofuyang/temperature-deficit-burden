# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd
root=Path(__file__).resolve().parent
e=Path(config.EICU_ROOT)
ids=set(pd.read_csv(root.parent/'revision_source'/'eicu_rewarming_cohort.csv.gz',dtype={'encounter_id':str},usecols=['encounter_id']).encounter_id)
specs=[('nurseAssessment.csv.gz','nurseassessoffset',['cellattributepath','celllabel','cellattribute','cellattributevalue']),('physicalExam.csv.gz','physicalexamoffset',['physicalexampath','physicalexamvalue','physicalexamtext'])]
pattern=r'rhythm|arrhyth|fibrill|tachycard|flutter|ventricular|cardiac'
for file,offset,fields in specs:
 parts=[]
 for ch in pd.read_csv(e/file,usecols=['patientunitstayid',offset]+fields,dtype=str,chunksize=500000):
  d=ch[ch.patientunitstayid.isin(ids)]
  mask=d[fields].fillna('').apply(lambda col:col.str.contains(pattern,case=False,regex=True)).any(axis=1)
  if mask.any():parts.append(d[mask])
 d=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['patientunitstayid',offset]+fields)
 d['offset_min']=pd.to_numeric(d[offset],errors='coerce')
 d.to_csv(root/('feasibility_'+file.replace('.csv.gz','')+'_internal.csv'),index=False)
 print('\n',file,'keyword rows',len(d),'stays',d.patientunitstayid.nunique(),flush=True)
 print('<=6h',d[d.offset_min<=360].patientunitstayid.nunique(),'>6h',d[d.offset_min>360].patientunitstayid.nunique(),flush=True)
 for field in fields:
  print(field,d[field].value_counts(dropna=False).head(30).to_string(),flush=True)
