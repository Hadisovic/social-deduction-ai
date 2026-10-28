"""Aggregate-only curation of immutable Phase 2.1 diagnostics; no model/game run."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def calibration(games,name):
    total=sum(g['calibration'][name]['actor_meetings'] for g in games);out={'actor_meetings':total}
    for metric in ('nll','brier','top1'):
        out[metric]=sum(g['calibration'][name]['actor_meetings']*(g['calibration'][name][metric] or 0) for g in games)/total if total else None
    bins=[dict(n=0,confidence=0.,accuracy=0.) for _ in range(5)]
    for game in games:
        for b in game['calibration'][name]['bins']:
            target=bins[min(4,int(b['confidence']*5))];target['n']+=b['n'];target['confidence']+=b['n']*b['confidence'];target['accuracy']+=b['n']*b['accuracy']
    for b in bins:
        if b['n']:b['confidence']/=b['n'];b['accuracy']/=b['n']
    out['ece']=sum(b['n']/total*abs(b['confidence']-b['accuracy']) for b in bins) if total else None
    out['bins']=bins;out['high_confidence_wrong']=sum(g['calibration'][name]['high_confidence_wrong'] for g in games)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'local-runs/phase21-comparison-v1';mode_names=('unrestricted','guard','influence03','influence10','ordered10','witness')
    social={};privacy={'applicable_contexts':0,'original_last_is_impostor':0,'confidence_last_is_impostor':0}
    for mode in mode_names:
        games=[load(f) for f in sorted(source.glob('social-*-'+mode+'.json'))];counts=Counter();actions=Counter()
        for g in games:counts.update(g['counts']);actions.update(g['chosen_actions'])
        social[mode]=dict(contexts=len(games),counts=dict(counts),chosen_actions=dict(actions),
            complete_matches=sum(g['complete_match'] for g in games),unfinished_segments=sum(not g['complete_match'] for g in games),
            base_calibration=calibration(games,'base'),adjusted_calibration=calibration(games,'adjusted'),
            maximum_voting_L1_influence=max(g['max_voting_L1_influence'] for g in games))
        if mode=='ordered10':
            for g in games:
                if not g['order']:continue
                replay=source/'replays/social'/mode/(g['fixture']+'.jsonl')
                if not replay.exists():replay=Path(str(replay)+'.writing')
                with replay.open(encoding='utf-8') as f:header=json.loads(next(f));events=[json.loads(line) for line in f]
                initial=next(e['participants'] for e in events if e.get('kind')=='meeting_started')
                privacy['applicable_contexts']+=1
                privacy['original_last_is_impostor']+=header['agent_roles'][initial[-1]]=='impostor'
                privacy['confidence_last_is_impostor']+=header['agent_roles'][g['order'][0]['order'][-1]]=='impostor'
    opponents={}
    for arm in ('baseline','v1','v2'):
        games=[load(f) for f in sorted(source.glob('opponent-*-'+arm+'.json'))];counts=Counter();roles={r:Counter() for r in ('crewmate','impostor')};events=Counter()
        for g in games:
            counts.update(g['counts'])
            for r in roles:
                v=g['opportunities']['by_role'][r]
                for key in ('accepted_reports','legal_actor_body_pairs','accepted_kills','self_reports'):roles[r][key]+=v[key]
                for key in ('report_legal_decision_windows','kill_legal_decision_windows'):roles[r][key]+=v['counts'].get(key,0)
            for key in ('meetings','discovered_bodies','unreported_kills'):events[key]+=g['opportunities'][key]
        opponents[arm]=dict(games=len(games),endings=dict(Counter(g['reason'] for g in games)),counts=dict(counts),
            events=dict(events),opportunities={r:dict(v) for r,v in roles.items()},
            retained_goal_windows=sum(g['retained_goal_windows'] or 0 for g in games),
            chase_updates=sum(g['chase_updates'] or 0 for g in games),
            mean_simulated_seconds=sum(g['simulated_seconds'] for g in games)/len(games))
    witness=[]
    for path in sorted(source.glob('witness-*.json')):
        g=load(path);r=g['witness_reports'][g['focal_witness']]
        accepted=[v['time'] for v in r['accepted_times'] if v['victim']==g['primary_body']]
        witness.append(dict(case=g['fixture'],mode=g['mode'],ending=g['reason'],complete_match=g['complete_match'],
            direct_witness_time=r['first_witness'],first_legal_report_time=r['first_legal'],
            focal_submitted_reports=r['submissions'],focal_accepted_reports=len(accepted),focal_approaches=r['approaches'],
            latency_from_witness=accepted[0]-r['first_witness'] if accepted and r['first_witness'] is not None else None,
            latency_from_first_legal=accepted[0]-r['first_legal'] if accepted and r['first_legal'] is not None else None,
            counts=g['counts']))
    metrics=dict(status='exploratory_engineering_only',acceptance='pending',execution=load(source/'execution.json'),
        social=social,speaking_order_privacy=privacy,opponents=opponents,witness=witness,
        public_shortcuts=load(ROOT/'local-runs/phase21-public-shortcuts.json'),
        historical=load(ROOT/'local-runs/historical-analysis-v2/registered-aggregate.json'),
        phase2_clustering=load(ROOT/'local-runs/historical-analysis-v2/phase2-aggregate.json'),
        phase21_clustering=load(ROOT/'local-runs/historical-analysis-v2/phase21-aggregate.json'),
        verification=load(ROOT/'local-runs/phase21-recording-verification.json'),
        preservation=load(ROOT/'local-runs/phase21-preservation.json'),
        precedence_repair_equivalence=load(ROOT/'local-runs/phase21-precedence-equivalence.json'),
        reproducibility=load(ROOT/'local-runs/phase21-repeat-v2.json'),
        synthetic_boundary_stress=[dict(influence=v,target_probability=p,vote_eligible=allowed,skip_retained=True)
            for v,p,allowed in ((0,.65,False),(.03,.68,False),(.10,.75,True))],
        provenance=dict(base='8c6592f',comparison_code='0cae29e',comparison_version='crewmate-phase21-v1',
            final_version='crewmate-phase21-v1.1',raw_observations_or_role_datasets_included=False),
        limits=['Six constructed social contexts share four engineering seeds; witness conditions share one seed; four natural scenario clusters.',
            'Mechanisms change choices externally; no neural learning. Incomplete segments are not match outcomes.',
            'Social adjustments are heuristic and can double-count claims already processed by the frozen predictor.',
            'Heterogeneous confidence providers confound speaker-order role leakage; not a recommended default.',
            'Three/four meetings per opponent make shortcut classifiers unstable; failure is not proof of unpredictability.'])
    with (a.output/'metrics.json').open('x',encoding='utf-8') as f:json.dump(metrics,f,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15.4,4.6),layout='constrained')
    x=list(range(6));correct=[social[m]['counts'].get('correct_votes',0) for m in mode_names];wrong=[social[m]['counts'].get('incorrect_votes',0) for m in mode_names];skip=[social[m]['counts'].get('published_skips',0) for m in mode_names]
    axes[0].bar(x,correct,color='#227c9d',label='Correct');axes[0].bar(x,wrong,bottom=correct,color='#b34b3b',label='Incorrect');axes[0].bar(x,skip,bottom=[a+b for a,b in zip(correct,wrong)],color='#999999',label='Skip')
    axes[0].set(xticks=x,xticklabels=['Raw','Guard','+3pp','+10pp','Order','Witness'],ylabel='Crew ballots',title='Six social contexts / mode');axes[0].tick_params(axis='x',labelrotation=45);axes[0].legend(fontsize=8)
    ys=[social[m]['adjusted_calibration']['nll'] for m in ('guard','influence03','influence10')]
    axes[1].bar([0,1,2],ys,color=['#227c9d','#47a39a','#b34b3b']);axes[1].set(xticks=[0,1,2],xticklabels=['0pp','3pp','10pp'],ylabel='Negative log-likelihood',title='Influence: 20 correlated forecasts')
    for i,y in enumerate(ys):axes[1].text(i,y+.04,f'{y:.2f}',ha='center',fontsize=9)
    for i,arm in enumerate(('baseline','v1','v2')):
        ends=opponents[arm]['endings'];win=ends.get('IMPOSTOR_EJECTED',0)+ends.get('TASKS_COMPLETED',0);loss=ends.get('PARITY',0);draw=ends.get('TIMEOUT',0)
        for bottom,n,color in ((0,win,'#227c9d'),(win,loss,'#b66a19'),(win+loss,draw,'#999999')):
            axes[2].bar(i,n,bottom=bottom,color=color)
            if n:axes[2].text(i,bottom+n/2,str(n),ha='center',va='center',color='white',weight='bold')
    from matplotlib.patches import Patch
    axes[2].set(xticks=[0,1,2],xticklabels=['Original','v1','v2'],ylim=(0,4.7),ylabel='Games',title='Four paired natural scenarios')
    axes[2].legend(handles=[Patch(color=c,label=l) for c,l in zip(['#227c9d','#b66a19','#999999'],['Crew win','Parity','Timeout'])],fontsize=8)
    n=privacy['applicable_contexts'];vals=[privacy['original_last_is_impostor']/n,privacy['confidence_last_is_impostor']/n]
    axes[3].bar([0,1],vals,color=['#227c9d','#b34b3b']);axes[3].set(xticks=[0,1],xticklabels=['Original','Confidence'],ylim=(0,1.16),ylabel='Last speaker is impostor',title='Order leakage: four fixtures')
    for i,v in enumerate(vals):axes[3].text(i,v+.035,f'{round(v*n)}/{n}',ha='center',fontsize=9)
    fig.suptitle('Frozen policy + explicit experimental mechanisms: engineering effects, not learned improvement',fontsize=12)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/'comparison.png',dpi=150);fig.savefig(a.output/'comparison.svg');plt.close(fig)
    svg=a.output/'comparison.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print('Curated aggregate-only metrics and scientific comparison figure created.')


if __name__=='__main__':main()
