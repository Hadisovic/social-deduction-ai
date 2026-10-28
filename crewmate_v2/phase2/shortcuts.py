"""Tiny leave-scenario-out assay using public pre-ballot behavior only.

Reporter flag and delivered claim kind are legitimate public information. Role
labels enter fitting/scoring offline, never an actor or feature. Scenario folds
prevent treating actors/meetings from the same seed as independent holdouts.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def public_rows(folder):
    rows=[]
    for path in sorted(Path(folder).rglob('*.jsonl')):
        header={};reporter=None;meeting=None;claims={};participants=[];captured=set()
        for line in path.open(encoding='utf-8'):
            e=json.loads(line)
            if e['record']=='header':header=e;continue
            if e['record']!='event':continue
            if e['kind']=='report':reporter=e['reporter_id']
            if e['kind']=='emergency_called':reporter=None
            if e['kind']=='meeting_started':
                meeting=e['meeting_id'];participants=e['participants'];claims={}
            if e['kind']=='claim' and e.get('channel')=='public':claims[e['speaker_id']]=e.get('claim_kind','none')
            if e['kind']=='voting_started' and meeting not in captured:
                captured.add(meeting)
                for pid in participants:
                    rows.append(dict(seed=header['scenario_seed'],arm=header['config']['opponent'],
                        context=header['config']['context'],meeting=meeting,actor=pid,
                        features=(str(pid==reporter),claims.get(pid,'none')),
                        label=header['agent_roles'][pid]))
            if e['kind'] in ('meeting_ended','exploration_resumed'):reporter=None;meeting=None
    return rows


def assay(rows):
    output={}
    for arm in sorted({r['arm'] for r in rows}):
        data=[r for r in rows if r['arm']==arm];predictions=[]
        for seed in sorted({r['seed'] for r in data}):
            train=[r for r in data if r['seed']!=seed];test=[r for r in data if r['seed']==seed]
            if not train or len({r['label'] for r in train})<2:continue
            classes=('crewmate','impostor');counts={label:[Counter(),Counter()] for label in classes};n=Counter()
            vocab=[set(r['features'][j] for r in train) for j in (0,1)]
            for r in train:
                n[r['label']]+=1
                for j,value in enumerate(r['features']):counts[r['label']][j][value]+=1
            for r in test:
                logs={label:sum(math.log((counts[label][j][value]+1)/(n[label]+len(vocab[j])+1)) for j,value in enumerate(r['features'])) for label in classes}
                # Balanced class priors; frequency alone is not an identity input.
                delta=logs['impostor']-logs['crewmate'];prob=1/(1+math.exp(-delta))
                predictions.append(dict(r,probability=prob))
        tp=sum(r['label']=='impostor' and r['probability']>=.5 for r in predictions)
        tn=sum(r['label']=='crewmate' and r['probability']<.5 for r in predictions)
        imps=sum(r['label']=='impostor' for r in predictions);crew=len(predictions)-imps
        meetings=defaultdict(list)
        for r in predictions:meetings[(r['seed'],r['meeting'])].append(r)
        top=[]
        for values in meetings.values():
            best=max(r['probability'] for r in values);ties=[r for r in values if abs(r['probability']-best)<1e-10]
            top.append(sum(r['label']=='impostor' for r in ties)/len(ties))
        output[arm]=dict(scenario_clusters=len({r['seed'] for r in data}),held_out_actor_meetings=len(predictions),
            impostor_recall=tp/imps if imps else None,crew_specificity=tn/crew if crew else None,
            balanced_accuracy=(tp/imps+tn/crew)/2 if imps and crew else None,
            meeting_macro_top1=statistics.mean(top) if top else None,meetings=len(top),
            limits='Exploratory public-behavior classifier; correlated actor/meeting rows, tiny seed folds, same frozen crew policy. Fit is not a trained game agent or proof of legitimate general deduction.')
    return output
