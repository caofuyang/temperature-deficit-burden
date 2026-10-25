import gzip,csv,re,json,collections
# Path configuration: the repository root is derived from this file,
# and database locations are read from environment variables at import time.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
base=Path(config.EICU_ROOT)
cohort=set()
with gzip.open(str(config.REVISION_SOURCE/'eicu_rewarming_cohort.csv.gz'),'rt',encoding='utf-8-sig',newline='') as f:
 for row in csv.DictReader(f):cohort.add(row['encounter_id'])
terms={
 'sepsis_shock':re.compile(r'sepsis|septic|shock',re.I),
 'bleeding':re.compile(r'bleed|haemorr|hemorr|blood loss',re.I),
 'surgical_site_infection':re.compile(r'surgical.site.infect|post.?op(?:erative)?.{0,20}infect|wound.{0,12}infect|infect.{0,12}wound',re.I),
 'arrhythmia':re.compile(r'arrhyth|atrial fibrill|atrial flutter|ventricular tachy|irregular rhythm',re.I),
}
tables={
 'admissionDx':('admitdxenteredoffset',['admitdxpath','admitdxname','admitdxtext']),
 'diagnosis':('diagnosisoffset',['diagnosisstring','icd9code']),
 'treatment':('treatmentoffset',['treatmentstring']),
 'note':('noteoffset',['notepath','notevalue','notetext']),
 'nurseCare':('nursecareoffset',['celllabel','cellattributepath','cellattribute','cellattributevalue']),
 'carePlanInfectiousDisease':('cplinfectdiseaseoffset',['infectdiseasesite','infectdiseaseassessment','treatment']),
 'microLab':('culturetakenoffset',['culturesite','organism']),
}
out={'cohort_n':len(cohort),'tables':{}}
for tab,(off,cols) in tables.items():
 path=base/(tab+'.csv.gz');res={'rows_in_cohort':0,'any_stays':set(),'terms':{k:{'rows':0,'all':set(),'by6h':set(),'after6h':set(),'negative':set(),'no_offset':set(),'labels':collections.Counter()} for k in terms}}
 with gzip.open(path,'rt',encoding='utf-8-sig',errors='replace',newline='') as f:
  for row in csv.DictReader(f):
   sid=row.get('patientunitstayid','')
   if sid not in cohort:continue
   res['rows_in_cohort']+=1;res['any_stays'].add(sid)
   body=' | '.join(row.get(c,'') or '' for c in cols)
   try:minute=float(row.get(off,''))
   except:minute=None
   for k,pat in terms.items():
    if not pat.search(body):continue
    z=res['terms'][k];z['rows']+=1;z['all'].add(sid)
    bucket='no_offset' if minute is None else 'negative' if minute<0 else 'by6h' if minute<=360 else 'after6h'
    z[bucket].add(sid)
    if tab in ('admissionDx','diagnosis','treatment'):
     lab=(row.get(cols[0],'') or '')[-100:]
     z['labels'][lab]+=1
 summary={'rows_in_cohort':res['rows_in_cohort'],'any_stays':len(res['any_stays']),'terms':{}}
 for k,z in res['terms'].items():summary['terms'][k]={q:(len(v) if isinstance(v,set) else v) for q,v in z.items() if q!='labels'}|{'top_labels':z['labels'].most_common(8)}
 out['tables'][tab]=summary
 print(tab,'cohort_stays',summary['any_stays'],{k:v['all'] for k,v in summary['terms'].items()},flush=True)
p=config.RESULTS_DIR/'eicu_fresh_feasibility_aggregate.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('saved',p)
