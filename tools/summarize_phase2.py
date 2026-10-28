"""Curate aggregate-only Phase 2 material from immutable local diagnostics."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    def load(name):return json.loads((ROOT/'local-runs'/name).read_text())
    experiment=load('opponent-assay-v1/aggregate.json');history=load('historical-social-v3/aggregate.json')
    raw=load('historical-social-v3/private-reconstruction.json');cases=[c for g in raw if 2790000<=g['seed']<=2790007 for c in g['cases']]
    from crewmate_v2.phase2.social import action_probability
    gaps=[action_probability(c['defend_neural'],'claim_defend',c['target'])-action_probability(c['defend_neural'],'claim_suspect',c['target']) for c in cases]
    social=dict(validation_matches=48,same_agent_same_meeting_links=len(cases),
        defended_impostor_then_voted_impostor=sum(c['target_is_impostor'] for c in cases),
        defended_innocent_then_voted_innocent=sum(not c['target_is_impostor'] for c in cases),
        defenses_argmax=sum(c['defense_was_argmax'] is True for c in cases),
        target_suspicion_at_least_80pct=sum(c['target_belief_at_defense']>=.8 for c in cases),
        target_top_ranked_suspicion=sum(c['target_belief_at_defense']>=max(c['defend_neural']['suspicion_probabilities'].values())-1e-9 for c in cases),
        no_new_public_claims=sum(not any(e['kind']=='claim' for e in c['intervening_public_events']) for c in cases),
        belief_unchanged_within_1e6=sum(c['max_belief_change']<1e-6 for c in cases),
        defense_probability_min=min(c['selected_defense_probability'] for c in cases),
        defense_probability_max=max(c['selected_defense_probability'] for c in cases),
        median_same_target_defend_minus_accuse_probability=statistics.median(gaps),
        private_memory_available=False)
    public=copy.deepcopy(experiment)
    for groups in public['groups'].values():
        for g in groups.values():
            nav=g.pop('navigation_choices');victims=g.pop('victim_ids_offline')
            g['navigation_summary']=dict(total_commands=sum(nav.values()),distinct_command_labels=len(nav))
            g['victim_summary']=dict(accepted_kills=sum(victims.values()),distinct_recorded_ids=len(victims),
                limitation='IDs are not stable identities across seeds; legal per-candidate kill ranks were not retained in this assay.')
    metrics=dict(status='exploratory_development_only',acceptance='pending',historical_groups=history,
        social_consistency=social,opponent_comparison=public,
        public_shortcuts=load('opponent-assay-v1/public-shortcuts-stratified.json'),
        evidence_probes=load('evidence-probe-v2/aggregate.json'),preservation=load('preservation-final-01.json'),
        provenance=dict(actor_base='fab5c70',comparison_code_snapshot='91f7692',
            exact_local_source_hashes='local protocol.json; benchmark source recoverable at comparison_code_snapshot',
            final_actor_sources_unchanged=True,raw_observation_records_included=False))
    with (a.output/'metrics.json').open('x',encoding='utf-8') as f:json.dump(metrics,f,indent=2,allow_nan=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
    colors=['#227c9d','#b66a19'];labels=['Original script','Mixture v1']
    for i,arm in enumerate(('baseline','mixture')):
        groups=public['groups'][arm]
        roles={role:Counter() for role in ('crewmate','impostor')}
        for g in groups.values():
            for role in roles:
                for key in ('accepted_reports','legal_actor_body_pairs'):roles[role][key]+=g['by_role'][role][key]
        for j,role in enumerate(roles):
            r=roles[role];rate=r['accepted_reports']/r['legal_actor_body_pairs']
            x=j+(i-.5)*.32;axes[0].bar(x,rate,width=.30,color=colors[i],label=labels[i] if j==0 else None)
            axes[0].text(x,rate+.025,f"{r['accepted_reports']}/{r['legal_actor_body_pairs']}",ha='center',fontsize=9)
        endings=Counter()
        for g in groups.values():endings.update(g['endings'])
        win=sum(g['crew_wins'] for g in groups.values());draw=endings['TIMEOUT'];loss=endings['PARITY']
        axes[1].bar(i,win,color='#227c9d');axes[1].bar(i,loss,bottom=win,color='#b66a19');axes[1].bar(i,draw,bottom=win+loss,color='#929292')
        for bottom,n in ((0,win),(win,loss),(win+loss,draw)):
            if n:axes[1].text(i,bottom+n/2,str(n),ha='center',va='center',color='white',weight='bold')
    axes[0].set(xticks=[0,1],xticklabels=['Crew','Impostor'],ylim=(0,1),ylabel='Accepted reports / legal actor–body pairs',title='Opportunity-adjusted reporting')
    axes[0].legend(fontsize=8);axes[1].set(xticks=[0,1],xticklabels=labels,ylim=(0,8.5),ylabel='Games',title='Eight scenarios per arm')
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=c,label=l) for c,l in zip(['#227c9d','#b66a19','#929292'],['Crew wins','Parity losses','Timeout draws'])],fontsize=8)
    classifier=metrics['public_shortcuts']['all']
    values=[classifier[arm]['meeting_macro_top1'] for arm in ('baseline','mixture')]
    axes[2].bar([0,1],values,color=colors)
    for i,arm in enumerate(('baseline','mixture')):
        n=classifier[arm]['meetings'];axes[2].text(i,values[i]+.025,f"{round(values[i]*n)}/{n} meetings",ha='center',fontsize=9)
    axes[2].set(xticks=[0,1],xticklabels=labels,ylim=(0,1.15),ylabel='Held-out meeting top-1',title='Public reporter + claim shortcut')
    fig.suptitle('Phase 2 engineering diagnostic — one frozen policy; small, correlated samples',fontsize=12)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(a.output/'comparison.png',dpi=160);fig.savefig(a.output/'comparison.svg');plt.close(fig)
    svg=a.output/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print('Aggregate JSON and scientific figures created; no raw actor/role records exported.')


if __name__=='__main__':main()
