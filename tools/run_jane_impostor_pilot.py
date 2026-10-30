"""Owner-approved 90-minute pilot: <=60 train, <=30 paired development, no finals."""
import argparse
from datetime import datetime,timedelta,timezone
from collections import Counter
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from time import monotonic,sleep

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def wilson(wins,n):
    if not n:return None
    z=1.96;p=wins/n;scale=1+z*z/n
    centre=(p+z*z/(2*n))/scale
    margin=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/scale
    return [max(0,centre-margin),min(1,centre+margin)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=731)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=monotonic();utc=datetime.now(timezone.utc)
    study_deadline=started+90*60
    registration=dict(protocol='jane-impostor-first-pilot-v1',approved_budget_minutes=90,
        training_minutes=60,max_training_matches=4,development_minutes=30,
        development_seeds=[9320001,9320002],selection='last complete PPO boundary; locked before development',
        initialization_seed=args.seed,started_utc=utc.isoformat(),deadline_utc=(utc+timedelta(minutes=90)).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        no_reserved_final_access=True,no_outcome_based_extension=True,
        model_name='Lizbon',actual_engine='pinned Patrick Jane A2; 10 frozen crew + 2 new neural impostors',
        hardware=dict(processor=platform.processor(),logical_cpus=os.cpu_count(),
                      python=platform.python_version(),device='cpu',torch_threads=1))
    write(args.output/'pilot-registration.json',registration)
    status_path=args.output/'pilot-status.json'
    environment=dict(os.environ,PYGAME_HIDE_SUPPORT_PROMPT='1',MPLCONFIGDIR=str(args.output/'plot-cache'))
    def child(folder,mode,minutes,episodes,checkpoint=None):
        command=[sys.executable,'-m','jane.impostor.v1.run','--source-root',str(args.source_root.resolve()),
            '--mode',mode,'--episodes',str(episodes),'--max-minutes',str(max(.05,minutes)),
            '--match-wall-seconds','1200','--seed',str(args.seed),'--output',str(folder)]
        if checkpoint:command+=['--checkpoint',str(checkpoint)]
        log=folder.parent/(folder.name+'.console.log')
        with log.open('w',encoding='utf-8') as handle:
            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=handle,stderr=subprocess.STDOUT)
        previous=None
        while process.poll() is None:
            value={}
            try:value=json.loads((folder/'status.json').read_text())
            except (OSError,ValueError):pass
            state=dict(stage=folder.name,phase=value.get('phase','starting'),pid=process.pid,
                elapsed_seconds=monotonic()-started,deadline_utc=registration['deadline_utc'],
                remaining_seconds=max(0,study_deadline-monotonic()),child_status=value)
            write(status_path,state)
            signature=(state['stage'],state['phase'],value.get('counters',{}).get('complete_matches'),value.get('simulation_seconds'))
            if signature!=previous:
                print(json.dumps(state),flush=True);previous=signature
            if monotonic()>study_deadline:
                process.terminate();process.wait(timeout=20)
                raise RuntimeError('Global safety deadline; incomplete run preserved, no success claimed')
            sleep(5)
        if process.returncode:
            raise RuntimeError(f'{folder.name} failed with exit {process.returncode}; inspect {log.name}')
        return json.loads((folder/'results.json').read_text())

    training=child(args.output/'train','train',59.5,4)
    train_root=args.output/'train'
    completed_updates=training['counters']['optimizer_steps']
    final=train_root/'final.pt';initial=train_root/'initial.pt'
    lock=dict(selection='last completed PPO boundary, independent of development outcomes',
        trained_checkpoint=final.name,trained_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        optimizer_steps=completed_updates,locked_utc=datetime.now(timezone.utc).isoformat())
    with (args.output/'selection-lock.json').open('x',encoding='utf-8') as handle:json.dump(lock,handle,indent=2)
    dev={}
    dev_deadline=min(study_deadline-150,monotonic()+27*60)
    if completed_updates:
        for i,(label,checkpoint) in enumerate((('initial',initial),('trained',final))):
            remaining=dev_deadline-monotonic()
            if remaining<15:break
            # Each method gets the same scenarios and a predeclared equal first
            # allocation. Technical cuts stay missing outcomes, never losses.
            allowance=min(13.5*60,remaining/(2-i))
            dev[label]=child(args.output/('development-'+label),'evaluate',allowance/60,2,checkpoint)
    plots={}
    for folder in (train_root,*[args.output/('development-'+label) for label in dev]):
        try:
            result=subprocess.run([sys.executable,'tools/plot_jane_impostor.py',str(folder)],cwd=ROOT,
                                  env=environment,capture_output=True,text=True,timeout=40)
            plots[folder.name]=dict(exit_code=result.returncode,output=result.stdout,error=result.stderr[-1000:])
        except subprocess.TimeoutExpired:plots[folder.name]=dict(error='plot deadline exceeded')
    comparisons=[]
    if 'initial' in dev and 'trained' in dev:
        a={r['seed']:r for r in dev['initial']['matches']}
        b={r['seed']:r for r in dev['trained']['matches']}
        for seed in registration['development_seeds']:
            ra,rb=a.get(seed),b.get(seed)
            valid=bool(ra and rb and not ra['truncated'] and not rb['truncated'] and ra['winner'] and rb['winner'])
            comparisons.append(dict(seed=seed,paired_complete=valid,
                initial=ra['winner'] if ra else None,trained=rb['winner'] if rb else None,
                win_difference=int(rb['winner']=='impostor')-int(ra['winner']=='impostor') if valid else None))
    summary=dict(registration=registration,training=training,development=dev,paired=comparisons,
        development_intervals={label:wilson(r['wins'],r['games']) for label,r in dev.items()},
        plots=plots,elapsed_seconds=monotonic()-started,selection_lock=lock,
        sufficient_for_best_model_claim=False,independent_initializations=1,
        learning_assessment='Engineering/pilot evidence only; four training games and two paired scenarios cannot establish a strong general model',
        reserved_final_opened=False)
    write(args.output/'pilot-results.json',summary)
    lines=['# Lizbon: first pilot','',f"Initialization seed: {args.seed}. One independent initialization.",
        f"Training: {training['wins']}/{training['games']} natural training outcomes; {completed_updates} optimizer passes.",
        f"Elapsed: {summary['elapsed_seconds']/60:.2f} minutes. Technical cuts are excluded from win denominators.",
        '', '## Paired development comparison', '', '| Method | Wins / natural games | Technical cuts |', '|---|---|---|']
    for label,r in dev.items():lines.append(f"| {label} | {r['wins']}/{r['games']} | {r['counters']['technical_cuts']} |")
    lines+=['','Training wins are not test accuracy. No reserved final set was opened.',
        'The selected checkpoint was locked before development. No tuning or budget extension followed outcomes.',
        'This pilot does not establish a globally best model or reliable team strategy.',
        'Next: inspect failure replays and opportunity metrics, then propose replicated training with stronger frozen crew.',
        '', '## Collaboration', '',
        'Owner/impostor work: review search/fake/report/vote behavior and approve the next measured study.',
        'Crew developer: improve body reporting/evidence/ballots; publish a fixed source/settings/checkpoint release.',
        'Both: keep separate branches, exchange frozen releases, coordinate shared rule/schema changes.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(status_path,dict(stage='finished',elapsed_seconds=summary['elapsed_seconds'],
        training_games=training['games'],training_wins=training['wins'],optimizer_steps=completed_updates,
        paired_complete=sum(r['paired_complete'] for r in comparisons)))
    print(json.dumps(dict(finished=True,output=str(args.output),elapsed_seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
