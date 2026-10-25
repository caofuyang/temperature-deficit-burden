import gzip,csv,collections,time
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
cohort=set()
with gzip.open(str(config.REVISION_SOURCE/'mimic_rewarming_cohort.csv.gz'),'rt',encoding='utf-8-sig',newline='') as f:
 for row in csv.DictReader(f):cohort.add(row['encounter_id'])
p=str(config.MIMIC_ROOT/'icu'/'chartevents.csv.gz')
ids={b'226741',b'226742'}
result={x.decode():{'rows':0,'stays':set(),'values':collections.Counter()} for x in ids}
start=time.monotonic()
with gzip.open(p,'rb') as f:
 header=f.readline().decode().strip().split(',')
 print('header',header,flush=True)
 key=header.index('stay_id');item=header.index('itemid');val=header.index('value')
 for n,line in enumerate(f,1):
  if b'226741' not in line and b'226742' not in line:continue
  parts=line.rstrip(b'\r\n').split(b',',max(item,val,key)+1)
  if len(parts)<=max(item,val,key):continue
  it=parts[item]
  if it not in ids:continue
  stay=parts[key].decode()
  if stay not in cohort:continue
  z=result[it.decode()];z['rows']+=1;z['stays'].add(stay);z['values'][parts[val].decode('utf-8',errors='replace')]+=1
  if z['rows']<=3:print('sample matched item',it.decode(),'value',parts[val][:50],flush=True)
print('elapsed_s',round(time.monotonic()-start,1),'rows_scanned',n)
for k,z in result.items():print(k,'rows',z['rows'],'cohort_stays',len(z['stays']),'values',z['values'].most_common(15))
