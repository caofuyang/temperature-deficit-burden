# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd

root=Path(__file__).resolve().parent
ids=set(pd.read_csv(root.parent/'revision_source'/'eicu_rewarming_cohort.csv.gz',usecols=['encounter_id'],dtype=str).encounter_id)
p=Path(str(config.EICU_ROOT/'intakeOutput.csv.gz'))
cols=['patientunitstayid','intakeoutputoffset','intakeoutputentryoffset','cellpath','celllabel','cellvaluenumeric']
parts=[]
for i,ch in enumerate(pd.read_csv(p,usecols=cols,dtype=str,chunksize=1000000)):
    d=ch[ch.patientunitstayid.isin(ids)]
    d=d[d.cellpath.fillna('').str.contains(r'Intake \(ml\)',case=False,regex=True)]
    if len(d): parts.append(d)
    if i%20==0: print('chunks',i,'selected',sum(map(len,parts)),flush=True)
d=pd.concat(parts,ignore_index=True)
d['offset']=pd.to_numeric(d.intakeoutputoffset,errors='coerce')
d['amount']=pd.to_numeric(d.cellvaluenumeric,errors='coerce')
early=d[(d.offset>=0)&(d.offset<=360)&(d.amount>0)]
print('All intake rows',len(d),'early positive rows',len(early),'stays',early.patientunitstayid.nunique(),flush=True)
print(early.cellpath.value_counts().head(80).to_string(),flush=True)
print('early categories',early.cellpath.str.extract(r'Intake \(ml\)\|([^|]+)',expand=False).value_counts().to_string(),flush=True)
print('blood all positive first6',early[early.cellpath.str.contains('blood|red blood|plasma|platelet',case=False,regex=True)].patientunitstayid.nunique(),flush=True)
