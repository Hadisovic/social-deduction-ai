"""Isolated native integration; bounded inference only, no new training study."""
import os
from pathlib import Path
from dataclasses import replace
import gzip
import json
import pytest

SOURCE=os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
if not SOURCE:pytest.skip('Explicit pinned source required',allow_module_level=True)
from jane.impostor.v1.bootstrap import activate,frozen_crew
activate(SOURCE)
import torch
torch.set_num_threads(1)
from jane.impostor.v1.model import ImpostorPolicy
from jane.impostor.v1.run import tensor_hash
from jane.classic_v2.settings import load_profile,HostProfile
ROOT=Path(__file__).resolve().parents[1]


def test_v2_has_checkpoint_migration_and_profile_locked_roundtrip(tmp_path):
    from jane.impostor.v2.checkpoint import load,save,warm_start
    profile=load_profile(ROOT/'jane/configs/classic-v2/default_12p.json')
    model=ImpostorPolicy(731)
    parent=ROOT/'artifacts/lizbon-v1/pilot-20261030/checkpoint-001.pt'
    ancestry=warm_start(parent,model,ROOT/'jane/configs/patrick_jane_a2_12p.json')
    assert ancestry['transfer'] and ancestry['parent_counters']['optimizer_steps']>0
    path=tmp_path/'new.pt';save(path,model,None,profile,{'optimizer_steps':0},731,ancestry)
    restored=ImpostorPolicy(2);load(path,restored,profile)
    assert tensor_hash(restored)==tensor_hash(model)
    with pytest.raises(FileExistsError):save(path,model,None,profile,{},731,ancestry)
    changed=HostProfile(replace(profile.settings,tasks=replace(profile.settings.tasks,short=3)))
    with pytest.raises(ValueError,match='mismatch'):load(path,restored,changed)
    record=torch.load(path,weights_only=True);record['sources']['missing.py']='changed'
    torch.save(record,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='mismatch'):load(tmp_path/'bad.pt',restored,profile)


@pytest.mark.parametrize('preset',['small_4p','full_15p_1imp','full_15p_3imp'])
def test_custom_match_emits_complete_local_decision_audit(tmp_path,preset):
    from jane.impostor.v2.session import run_match
    profile=load_profile(ROOT/f'jane/configs/classic-v2/{preset}.json')
    if preset=='full_15p_1imp':
        profile=HostProfile(replace(profile.settings,tasks=replace(profile.settings.tasks,common=2,long=3,short=5)))
    model=ImpostorPolicy(731);crew=frozen_crew(Path(SOURCE));before=tensor_hash(crew)
    replay=tmp_path/'match.jsonl'
    summary,_=run_match(model,crew,profile,9700010,output=replay,max_observations=6,
        max_wall_seconds=30,deterministic=True)
    with gzip.open(replay.with_suffix('.decisions.jsonl.gz'),'rt',encoding='utf-8') as f:
        records=[json.loads(line) for line in f]
    choices=[r for r in records if r['record']=='decision']
    assert len(choices)==summary['impostor_observations']>0
    assert summary['host_profile_sha256']==profile.fingerprint()
    assert summary['truncated'] and tensor_hash(crew)==before
    assert any(r['record']=='effects' for r in records)
    assert records[-1]['record']=='closed'
    from jane.impostor.v2.analysis import summarize_decisions
    assert len(summarize_decisions(replay.with_suffix('.decisions.jsonl.gz'))['actors'])==profile.settings.lobby.impostors
    for row in choices:
        assert len(row['self_features'])==192 and len(row['players'])==14
        assert sum(c['probability'] for c in row['candidates'])==pytest.approx(1.)
        assert row['candidates'][row['selected']]['kind']==row['selected_kind']
        assert row['input_scope']=='actor-local'
        assert set(row['local']['visible_players'])<=set(row['local']['roster'])
        assert 'world' not in row['local']


def test_audit_summary_separates_contiguous_opportunities(tmp_path):
    from jane.impostor.v2.analysis import summarize_decisions
    path=tmp_path/'decisions.jsonl.gz'
    with gzip.open(path,'wt') as f:
        for tick,kinds,chosen in ((0,['wait','kill'],0),(3,['wait','kill'],1),(6,['wait'],0),(9,['wait','kill'],1)):
            f.write(json.dumps(dict(record='decision',owner='p0',tick=tick,time=tick*.2,phase='roaming',
                selected=chosen,selected_kind=kinds[chosen],local={'cooldown':0,'navigation':'idle','door_blocked':False},
                candidates=[dict(kind=k,key='p1',probability=1/len(kinds)) for k in kinds]))+'\n')
    result=summarize_decisions(path)['actors']['p0']
    assert result['kill_eligible_decisions']==3
    assert result['kill_windows']==2
    assert result['selected/kill']==2


def test_transferred_initial_metadata_counts_preserved_training_passes():
    from jane.impostor.v2.run import known_passes
    assert known_passes(dict(counters={'optimizer_steps':0},ancestry={'parent_counters':{'optimizer_steps':8}}))==8
    assert known_passes(dict(counters={'optimizer_steps':8},ancestry={'parent_total_steps':16}))==24
