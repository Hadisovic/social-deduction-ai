"""Post-hoc descriptive summaries/figure from local diagnostic output; no games."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(source,output):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    raw=json.loads((source/'aggregate.json').read_text())
    summaries={}
    for profile in ('baseline','staggered'):
        for condition in ('A','B'):
            latencies=[];terminal_witness_pairs=0;lag=[];nonwait=0;matches=0
            for path in sorted((source/'private').glob(f'{profile}-*-{condition}.json')):
                doc=json.loads(path.read_text());row=doc['metrics'];events=doc['replay']['events']
                roles={e['player_id']:e['role'] for e in events if e['kind']=='spawn'}
                kills={e['victim_id']:e['time'] for e in events if e['kind']=='elimination'}
                for event in events:
                    if event['kind']=='report':
                        latencies.append(event['time']-kills[event['victim_id']])
                    if (event['kind']=='elimination_witnessed' and roles[event['observer_id']]=='crewmate'
                        and event['observer_id']!=event['victim_id'] and row['reason']=='PARITY'
                        and event['time']==row['duration']):
                        terminal_witness_pairs+=1
                if row['first_kill'] is not None and row['first_kill_legal'] is not None:
                    lag.append(row['first_kill']-row['first_kill_legal'])
                nonwait+=row['counts']['actionable_windows']-row['counts'].get('selected_wait_with_actions',0)
                matches+=1
            summaries[profile+'_'+condition]=dict(matches=matches,reported_bodies=len(latencies),
                kill_to_report_seconds=dict(median=statistics.median(latencies) if latencies else None,
                    min=min(latencies) if latencies else None,max=max(latencies) if latencies else None),
                physical_crew_witness_pairs_at_terminal_parity_kill=terminal_witness_pairs,
                median_first_physical_opportunity_to_kill=statistics.median(lag) if lag else None,
                actual_nonwait_decisions=nonwait)
    document=dict(development_only=True,analysis='post-hoc descriptive; no new execution or selection',groups=summaries)
    (output/'supplement.json').write_text(json.dumps(document,indent=2)+'\n')
    # Standard static research figure, derived solely from aggregate counts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    labels=['Ordinary A','Deduction-only B'];colors=['#2864b5','#8661a8','#dc8845']
    for ax,field,title in zip(axes,('phase_simulated_seconds','phase_wall_seconds'),
                            ('Simulated time by phase','Recorded wall cost by phase')):
        bottoms=[0.,0.]
        for phase,color in zip(('roaming','discussion','voting'),colors):
            values=[raw['groups']['baseline_'+c][field].get(phase,0) for c in ('A','B')]
            ax.bar(labels,values,bottom=bottoms,color=color,label=phase.capitalize(),width=.56)
            for index,(value,bottom) in enumerate(zip(values,bottoms)):
                if phase!='discussion':ax.text(index,bottom+value/2,f'{value:.0f}s',ha='center',va='center',color='white',fontsize=11)
            bottoms=[b+v for b,v in zip(bottoms,values)]
        ax.set(title=title,ylabel='Seconds across eight scenarios')
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False,loc='upper left')
    fig.suptitle('V2 scripted diagnostic: longer games, substantial voting cost',fontsize=14)
    fig.savefig(output/'phase-costs.png',dpi=170)
    plt.close(fig)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summarize(args.source,args.output)
