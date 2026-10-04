"""Finish the owner-approved one-run experiment; never starts another training run."""
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from phase5_evaluation import evaluate
from phase5_report import report,assess

training=ROOT/'artifacts/phase5/runs/full-17-approved'
output=ROOT/'docs/benchmarks/phase5/single-seed-17'
output.mkdir(parents=True,exist_ok=True)
print('Waiting for the approved 32,768-transition run to finish.',flush=True)
while not (training/'training.json').is_file():time.sleep(10)
record=json.loads((training/'training.json').read_text())
assert record['config']['steps']==32768 and record['config']['seed']==17 and not record['config']['quick']
checkpoint=Path('artifacts/phase5/runs/full-17-approved/policy.pt')
validation=evaluate(output/'validation',100,[checkpoint],validation=True)
curve=[]
for steps in (8192,16384,24576):
    path=Path('artifacts/phase5/runs/full-17-approved')/f'checkpoint-{steps}.pt'
    measured=evaluate(output/f'curve-{steps}',20,[path],validation=True,include_baselines=False)
    curve.append({'steps':steps,**measured['metrics'][str(path)]['ID']})
final_rows=[json.loads(line) for line in (output/'validation/matches.jsonl').read_text().splitlines()
            if json.loads(line)['method']==str(checkpoint) and json.loads(line)['seed']<620020]
curve.append({'steps':32768,'matches':len(final_rows),'crew_win_rate':sum(r['crew_win'] for r in final_rows)/len(final_rows),
              'mean_own_tasks':sum(r['own_tasks'] for r in final_rows)/len(final_rows)})
(output/'validation-curve.json').write_text(json.dumps(curve,indent=2)+'\n')
(output/'validation-assessment.json').write_text(json.dumps(assess(validation,str(checkpoint)),indent=2)+'\n')
print('Validation assessment recorded before opening final tests.',flush=True)
evaluate(output/'test',200,[checkpoint])
shutil.copyfile(training/'training.json',output/'training.json')
summary=report(training,output)
print(json.dumps({'complete':True,'assessment':summary['assessment'],'report':str(output/'README.md')}),flush=True)
