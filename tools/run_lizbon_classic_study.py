"""Owner-approved, three-hour bounded pilot. No automatic retries or final tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT=Path(__file__).resolve().parents[1]


def write(path,value,exclusive=False):
    path=Path(path)
    if exclusive:
        with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
    else:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def select(validation):
    """Only complete validation rows enter selection; reference wins exact ties."""
    scores={}
    for name,rows in validation.items():
        if not rows or any(r['truncated'] or r['winner'] not in ('crewmate','impostor') for r in rows):
            raise ValueError('Selection requires every registered validation match complete')
        scores[name]=(sum(r['winner']=='impostor' for r in rows)/len(rows),
            sum(r['metrics']['eliminations'] for r in rows)/len(rows),int(name=='reference'))
    if not scores:raise ValueError('No validated candidates')
    return sorted(scores,key=lambda n:(scores[n],n),reverse=True)[0]


def lock_selection(path,selected,checkpoint,validation):
    value=dict(schema='lizbon-classic-selection-v2',locked_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected,checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),validation=validation,
        rule='Natural win fraction, mean eliminations, reference on exact tie; lexical candidate ID last',
        immutable_after_confirmation_opens=True)
    write(path,value,True);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=monotonic();started=datetime.now(timezone.utc);deadline=start+10800
    python=sys.executable
    registration=dict(schema='lizbon-classic-pilot-v2',started_utc=started.isoformat(),
        deadline_utc=(started+timedelta(hours=3)).isoformat(),budget_seconds=10800,training_cutoff_seconds=5400,
        workers=2,profile='jane/configs/classic-v2/default_12p.json',initialization='V1 checkpoint001 weights; fresh Adam',
        streams={'101':dict(train_scenarios=list(range(9710001,9710005))),
                 '202':dict(train_scenarios=list(range(9710101,9710105)))},
        same_starting_weights=True,independent_initializations=False,
        candidates='Reference starting weights plus last complete full-PPO checkpoint of each stream',
        validation_scenarios=[9720001,9720002],validation_selection='Sampled; wins, mean eliminations, reference on exact tie',
        confirmation_scenarios={'sampled':[9720101,9720102],'greedy':[9720103,9720104]},
        confirmation='Paired selected versus starting reference after immutable selection lock; no retuning',
        match_wall_seconds=600,training_match_wall_seconds=900,
        failure_policy='Preserve prefix, stop study on child failure; never retry automatically',
        reserved='9730001..9731000 and every historical final partition remain sealed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'registration.json',registration,True)
    def status(phase,**info):write(out/'status.json',dict(phase=phase,wall_seconds=monotonic()-start,
        deadline_utc=registration['deadline_utc'],**info))

    def job(name,mode,episodes,offset,end,checkpoint=None,warm=False,deterministic=False):
        remaining=end-monotonic()
        if remaining<5:return dict(name=name,exit_code=None,budget_cut=True)
        target=out/name
        cmd=[python,str(ROOT/'run_lizbon_classic.py'),'--source-root',str(args.source_root.resolve()),
            '--output',str(target),'--mode',mode,'--episodes',str(episodes),'--scenario-offset',str(offset),
            '--max-minutes',str(max(.01,(remaining-3)/60)),
            '--match-wall-seconds','900' if mode=='train' else '600']
        if warm:cmd+=['--warm-start-v1',str(ROOT/'artifacts/lizbon-v1/pilot-20261030/checkpoint-001.pt')]
        if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
        if deterministic:cmd+=['--deterministic']
        with (out/(name+'.console.log')).open('x',encoding='utf-8') as console:
            try:
                result=subprocess.run(cmd,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,
                    timeout=remaining,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                record=dict(name=name,exit_code=result.returncode,budget_cut=False)
            except subprocess.TimeoutExpired:
                record=dict(name=name,exit_code=None,budget_cut=True)
        write(out/(name+'.job.json'),record);return record

    def group(specs,end):
        jobs=[]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures={executor.submit(job,**s,end=end):s['name'] for s in specs}
            for future in as_completed(futures):
                r=future.result();jobs.append(r)
                status('processing',completed_jobs=jobs)
        if any(r.get('exit_code')!=0 for r in jobs):raise RuntimeError('Child failure or registered deadline; see job logs')
        return jobs

    try:
        status('training')
        group([dict(name='train101',mode='train',episodes=4,offset=0,warm=True),
               dict(name='train202',mode='train',episodes=4,offset=100,warm=True)],start+5400)
        candidates={'reference':out/'train101/initial.pt'}
        for name in ('train101','train202'):
            rows=[json.loads(line) for line in (out/name/'matches.jsonl').read_text().splitlines()]
            eligible=[r for r in rows if not r['truncated'] and r['winner'] and not r.get('optimization_budget_cut',True)]
            if eligible:candidates[name]=out/name/f"checkpoint-{eligible[-1]['episode']:03d}.pt"
        if len(candidates)==1:raise RuntimeError('No complete new optimizer candidate; retain reference without confirmation')
        status('validation',candidates={k:str(v) for k,v in candidates.items()})
        eval_end=min(deadline,monotonic()+5400)
        val_end=min(eval_end,monotonic()+2700)
        group([dict(name='validation-'+n,mode='evaluate',episodes=2,offset=0,checkpoint=path)
            for n,path in candidates.items()],val_end)
        validation={n:json.loads((out/('validation-'+n)/'results.json').read_text())['matches'] for n in candidates}
        if any(len(rows)!=2 for rows in validation.values()):raise RuntimeError('Validation incomplete; no selection')
        selected=select(validation)
        lock=lock_selection(out/'selection-lock.json',selected,candidates[selected],validation)
        status('confirmation',selected=selected)
        # The lock is written before any fresh confirmation run opens.
        group([dict(name=f'confirmation-{name}-{mode}',mode='evaluate',episodes=2,
            offset=100 if mode=='sampled' else 102,checkpoint=path,deterministic=mode=='greedy')
            for name,path in (('reference',candidates['reference']),('selected',candidates[selected]))
            for mode in ('sampled','greedy')],eval_end)
        confirmation={f'{name}-{mode}':json.loads((out/f'confirmation-{name}-{mode}/results.json').read_text())
            for name in ('reference','selected') for mode in ('sampled','greedy')}
        complete=all(r['games']==2 and r['counters']['technical_cuts']==0 for r in confirmation.values())
        result=dict(phase='finished' if complete else 'confirmation-incomplete',selection=lock,
            confirmation=confirmation,wall_seconds=monotonic()-start,
            started_utc=started.isoformat(),finished_utc=datetime.now(timezone.utc).isoformat(),
            final_tests_opened=False,learning_sufficiency='Small pilot, two same-start sampling streams; no replicated initialization or broad skill claim')
        write(out/'results.json',result);status(result['phase'],selected=selected)
    except BaseException as error:
        write(out/'failure.json',dict(error=type(error).__name__,message=str(error),wall_seconds=monotonic()-start,
            selection_locked=(out/'selection-lock.json').exists(),automatic_retry=False))
        status('failed',error=str(error));raise
    finally:
        # Offline report/plots only. Failure cannot trigger new matches or change selection.
        report=ROOT/'tools/report_lizbon_classic_study.py'
        if report.exists():subprocess.run([python,str(report),str(out)],cwd=ROOT,check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)


if __name__=='__main__':main()
