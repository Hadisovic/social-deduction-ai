"""Post-hoc descriptive reporting-style audit and aggregate figure; no game execution."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def summarize(source,output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    aggregate=json.loads((source/'aggregate.json').read_text())
    styles=defaultdict(Counter);opportunities=defaultdict(Counter);before=defaultdict(Counter);seen=set()
    ballots=[]
    for path in sorted((source/'private').glob('diverse-*-?.json')):
        doc=json.loads(path.read_text());row=doc['metrics'];condition=row['condition']
        roles={e['player_id']:e['role'] for e in doc['replay']['events'] if e['kind']=='spawn'}
        if row['seed'] not in seen:
            for pid,style in doc['styles'].items():styles[roles[pid]][style]+=1
            seen.add(row['seed'])
        for pair in doc['exposures']['pairs']:
            key=pair['role']+'_'+doc['styles'][pair['actor']];c=opportunities[key]
            c['visible_pairs']+=1;c['visible_windows']+=pair['visible_windows']
            c['legal_pairs']+=pair['first_legal'] is not None;c['legal_windows']+=pair['legal_windows']
            c['submitting_pairs']+=bool(pair['submissions']);c['accepted']+=pair['accepted_time'] is not None
            c['censored_'+pair['censor_reason']]+=pair['accepted_time'] is None
        for b in doc['exposures']['ballots']:
            c=before[condition];c['ballots']+=1;c['before_full_bar']+=b['before_full_bar']
            informative=b['direct_witness'] or b['body_info']
            c['direct_crime_exposure']+=informative
            c['crime_exposure_before_full_bar']+=informative and b['before_full_bar']
            ballots.append(b)
    supplement=dict(analysis='Post-hoc descriptive; no new games or policy selection.',
        unique_scenarios=len(seen),styles_by_role_once_per_scenario={k:dict(v) for k,v in styles.items()},
        opportunities_by_role_style_pooled_AB={k:dict(v) for k,v in opportunities.items()},
        crew_ballots_by_condition={k:dict(v) for k,v in before.items()})
    (output/'supplement.json').write_text(json.dumps(supplement,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    conditions=('baseline_A','diverse_A','baseline_B','diverse_B')
    labels=['Original A','Diverse A','Original B','Diverse B']
    for index,key in enumerate(conditions):
        group=aggregate['groups'][key];crew=group['reporting']['crewmate']['counts']
        for offset,(field,label,color) in zip((-.23,0,.23),(
            ('visible_unique','Body seen','#3b78b7'),('legal_unique','Legal range','#8d68a6'),('accepted','Report accepted','#dc8a47'))):
            value=crew[field]
            axes[0].bar(index+offset,value,width=.22,color=color,label=label if index==0 else None)
            axes[0].text(index+offset,value+.08,str(value),ha='center',fontsize=9)
        score=group['reporter_heuristics']['event_weighted']
        for offset,(field,label,color) in zip((-.23,0,.23),(
            ('accuse','Accuse reporter','#c34e4d'),('avoid','Avoid reporter','#549784'),('uniform','Uniform','#7c8493'))):
            value=score[field]*100
            axes[1].bar(index+offset,value,width=.22,color=color,label=label if index==0 else None)
    for ax in axes:
        ax.set_xticks(range(4),labels,rotation=15);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=8,loc='upper left');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='Crew reporting opportunities',ylabel='Actor-body pairs / accepted reports',ylim=(0,8))
    axes[1].set(title='Reporter-only identification at crew ballots',ylabel='Tie-averaged accuracy (%)',ylim=(0,115))
    fig.suptitle('Reporting diversity helps exposure; the identity shortcut remains',fontsize=14)
    fig.savefig(output/'reporting-comparison.png',dpi=170);plt.close(fig)
    return supplement


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.source,args.output)
