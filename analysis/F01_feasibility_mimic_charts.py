# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = config.REVISION_SOURCE
MIMIC = config.MIMIC_ROOT
cohort = pd.read_csv(SOURCE / 'mimic_rewarming_cohort.csv.gz', dtype={'encounter_id': str})
stays = set(cohort.encounter_id)
icu = pd.read_csv(MIMIC / 'icu' / 'icustays.csv.gz', usecols=['stay_id', 'intime'], dtype={'stay_id': str})
icu = icu[icu.stay_id.isin(stays)].set_index('stay_id')
icu['intime'] = pd.to_datetime(icu.intime)
itemids = {'220048', '224105', '225050', '229686'}
parts = []
for i, chunk in enumerate(pd.read_csv(MIMIC / 'icu' / 'chartevents.csv.gz',
                                      usecols=['stay_id', 'charttime', 'itemid', 'value'],
                                      dtype=str, chunksize=1_000_000)):
    rows = chunk[chunk.stay_id.isin(stays) & chunk.itemid.isin(itemids)]
    if len(rows): parts.append(rows)
    if i % 30 == 0: print('chunks', i, 'matched', sum(len(x) for x in parts), flush=True)
data = pd.concat(parts, ignore_index=True)
data['offset_min'] = (pd.to_datetime(data.charttime) - data.stay_id.map(icu.intime)).dt.total_seconds() / 60
data.to_csv(ROOT / 'feasibility_mimic_charts_internal.csv', index=False)
for itemid, group in data.groupby('itemid'):
    print('\nITEM', itemid, 'rows', len(group), 'stays', group.stay_id.nunique())
    print('at/before 6 h stays', group[group.offset_min <= 360].stay_id.nunique(),
          'after 6 h stays', group[group.offset_min > 360].stay_id.nunique())
    print(group.value.value_counts(dropna=False).head(35).to_string())
