# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = config.REVISION_SOURCE
EICU = Path(config.EICU_ROOT)
ids = set(pd.read_csv(SOURCE / 'eicu_rewarming_cohort.csv.gz', dtype={'encounter_id': str}).encounter_id)
columns = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypecat',
           'nursingchartcelltypevallabel', 'nursingchartcelltypevalname', 'nursingchartvalue']
pattern = r'rhythm|arrhyth|fibrill|tachycard|warm|blanket|infection|wound'
parts = []
for i, chunk in enumerate(pd.read_csv(EICU / 'nurseCharting.csv.gz', usecols=columns,
                                      dtype=str, chunksize=1_000_000)):
    cohort = chunk[chunk.patientunitstayid.isin(ids)]
    mask = cohort[['nursingchartcelltypecat','nursingchartcelltypevallabel',
                   'nursingchartcelltypevalname']].fillna('').apply(
        lambda col: col.str.contains(pattern,case=False,regex=True)).any(axis=1)
    selected = cohort[mask]
    if len(selected): parts.append(selected)
    if i % 30 == 0: print('chunks', i, 'matched', sum(len(x) for x in parts), flush=True)
data = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
data['offset_min'] = pd.to_numeric(data.nursingchartoffset, errors='coerce')
data.to_csv(ROOT / 'feasibility_eicu_nurse_internal.csv', index=False)
for keys, group in data.groupby(['nursingchartcelltypecat','nursingchartcelltypevallabel','nursingchartcelltypevalname'],dropna=False):
    if len(group)<5: continue
    print('\nKEYS', keys, 'rows', len(group), 'stays', group.patientunitstayid.nunique())
    print('<=6h stays',group[group.offset_min<=360].patientunitstayid.nunique(),
          '>6h stays',group[group.offset_min>360].patientunitstayid.nunique())
    print(group.nursingchartvalue.value_counts(dropna=False).head(20).to_string())
