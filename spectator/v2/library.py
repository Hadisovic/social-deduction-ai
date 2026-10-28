"""Bounded background JSONL indexing/loading; never writes to a replay folder."""
from collections import Counter,deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from spectator.replay_format import SCHEMA_VERSION,parse_replay
from .data import family,event_text,time_of


def inspect_file(path):
    path=Path(path);header={};result=None;counts=Counter();last={};rewards={};moments=[];reports=0;report_options=0;errors=0;wrong=0;neural_count=0;defended=set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:r=json.loads(line)
            except (ValueError,TypeError):continue
            if not isinstance(r,dict):continue
            record=r.get('record');kind=r.get('kind')
            if record=='header':header=r
            elif record=='tick':last=r
            elif record=='result':result=r
            elif record=='reward':
                if isinstance(r.get('cumulative_reward'),(int,float)):rewards[r.get('player_id')]=r['cumulative_reward']
            elif record=='neural':
                neural_count+=1
                counts['action_'+family(r.get('action_kind','unknown'))]+=1
                probs=r.get('suspicion_probabilities',{})
                if isinstance(probs,dict) and probs and header.get('agent_roles',{}).get(r.get('player_id'))=='crewmate':
                    best=max(probs.values());leaders=[p for p,v in probs.items() if v==best]
                    if best>=.9 and leaders and all(header.get('agent_roles',{}).get(p)=='crewmate' for p in leaders):
                        errors+=1
                        if len(moments)<128:moments.append(dict(time=time_of(r),player=r.get('player_id'),label='High-confidence belief error (offline truth)'))
                legal=r.get('legal_actions',[])
                if any(family(a)=='Reporting' for a in legal):
                    report_options+=1
                    if r.get('action_kind')!='report' and len(moments)<128:moments.append(dict(time=time_of(r),player=r.get('player_id'),label='Legal report option not selected (review)'))
            elif record=='event':
                counts[kind]+=1
                if kind=='report':
                    reports+=1
                    roles=header.get('agent_roles',{})
                    if roles.get(r.get('reporter_id'))=='impostor':counts['impostor_reports']+=1
                    handoff=header.get('config',{}).get('prefix_handoff_seconds',0)
                    if time_of(r)<handoff-1e-8:counts['prefix_reports']+=1
                    elif roles.get(r.get('reporter_id'))=='crewmate' and header.get('controller_types',{}).get(r.get('reporter_id')) in ('ppo','frozen'):counts['neural_reports']+=1
                if kind=='claim' and 'defend' in str(r.get('claim_kind','')).lower():defended.add((r.get('speaker_id'),r.get('subject_id')))
                if kind=='vote_submitted' and 'target_id' in r and r['target_id'] is None and header.get('agent_roles',{}).get(r.get('voter_id'))=='crewmate':counts['skips']+=1
                if kind=='vote' and r.get('channel')=='public':
                    target=r.get('target_id');voter=r.get('voter_id')
                    counts['all_ballots']+=1
                    if header.get('agent_roles',{}).get(voter)!='crewmate':continue
                    if 'target_id' not in r:counts['unscored_ballots']+=1
                    elif target is None:counts['published_skips']+=1
                    elif header.get('agent_roles',{}).get(target)=='crewmate':
                        wrong+=1
                        if len(moments)<128:moments.append(dict(time=time_of(r),player=voter,label='Incorrect ballot (offline truth)'))
                    elif header.get('agent_roles',{}).get(target)=='impostor':counts['correct_ballots']+=1
                    else:counts['unscored_ballots']+=1
                    if (voter,target) in defended:
                        counts['position_changes']+=1
                        if len(moments)<128:moments.append(dict(time=time_of(r),player=voter,label='Defense then vote (review, not a verified bug)'))
    if header.get('schema')!=SCHEMA_VERSION or result is None or not last:return None
    cfg=header.get('config',{});players=last.get('players',{})
    tasks=[p for p in players.values() if isinstance(p.get('tasks_total'),(int,float)) and p.get('tasks_total',0)>0]
    tags=[]
    if wrong:tags.append('Wrong ballots')
    if errors:tags.append('Confident errors')
    if report_options:tags.append('Report decisions')
    if counts['impostor_reports']:tags.append('Impostor reports')
    if counts['position_changes']:tags.append('Position changes')
    if result.get('reason')=='TASKS_COMPLETED':tags.append('Task victory')
    if neural_count==0:tags.append('No neural telemetry')
    return dict(path=str(path.resolve()),name=path.stem,header=header,condition=cfg.get('training_condition','Historical / scripted'),
        context=cfg.get('context','Unspecified'),seed=header.get('scenario_seed'),session=header.get('training_session',''),
        duration=time_of(result),winner=result.get('winner'),reason=result.get('reason',''),
        kills=counts['elimination'] or counts['kill'],reports=reports,neural_reports=counts['neural_reports'],prefix_reports=counts['prefix_reports'],meetings=counts['meeting_started'],
        wrong_ballots=wrong,correct_ballots=counts['correct_ballots'],skips=counts['skips'],
        confident_errors=errors,report_decision_windows=report_options,neural_count=neural_count,
        tasks_completed=sum(p.get('tasks_completed',0) or 0 for p in tasks) if tasks else None,
        tasks_total=sum(p['tasks_total'] for p in tasks) if tasks else None,
        rewards=rewards,tags=tags,moments=moments,action_families={k[7:]:v for k,v in counts.items() if str(k).startswith('action_')},mtime=path.stat().st_mtime)


class Library:
    """At most two metadata jobs, one full load and 5,000 small entries."""
    def __init__(self):
        self.directory=None;self.entries={};self.signatures={};self.errors={};self.queue=deque()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='replay-metadata')
        self.loader=ThreadPoolExecutor(max_workers=1,thread_name_prefix='replay-loader')
        self.scan=None;self.pending={};self.loading=None;self.requested=None;self.load_result=None
        self.last_scan=0.;self.loading_path=None;self.closed=False;self.cap_reached=False
        self.revision=0;self._filter_signature=None;self._filtered_cache=[]

    def set_directory(self,path):
        self.directory=Path(path).resolve();self.entries.clear();self.signatures.clear();self.errors.clear();self.queue.clear();self.last_scan=0.
        self.revision+=1;self._filter_signature=None;self._filtered_cache=[]

    def _discover(self):
        result={}
        if not self.directory or not self.directory.exists():return result
        for path in self.directory.rglob('*.jsonl'):
            if len(result)>=5000:break
            try:s=path.stat();result[str(path.resolve())]=(s.st_mtime_ns,s.st_size)
            except OSError:pass
        return result

    def poll(self):
        now=time.monotonic();changed=False
        if self.scan is None and now-self.last_scan>=5 and self.directory:
            self.scan=self.pool.submit(self._discover);self.last_scan=now
        if self.scan and self.scan.done():
            try:found=self.scan.result()
            except OSError:found={}
            self.scan=None
            found={p:s for p,s in found.items() if self.directory and Path(p).is_relative_to(self.directory)}
            self.cap_reached=len(found)>=5000
            for path in self.entries.keys()-found.keys():self.entries.pop(path,None);self.signatures.pop(path,None);changed=True
            queued={p for p,_ in self.queue}|{v[0] for v in self.pending.values()}
            for path,sig in found.items():
                if self.signatures.get(path)!=sig and path not in queued:self.queue.append((path,sig))
        for future,(path,sig) in list(self.pending.items()):
            if not future.done():continue
            self.pending.pop(future)
            try:
                item=future.result()
                if self.directory and Path(path).is_relative_to(self.directory):
                    if item:self.entries[path]=item;changed=True
                    self.signatures[path]=sig
            except (OSError,ValueError,TypeError) as exc:self.errors[path]=str(exc);self.signatures[path]=sig
        while self.queue and len(self.pending)<2:
            path,sig=self.queue.popleft();self.pending[self.pool.submit(inspect_file,path)]=(path,sig)
        if self.loading and self.loading.done():
            path=self.loading_path
            try:
                data=self.loading.result()
                if self.requested==path:self.load_result=(path,data,None);self.requested=None
            except Exception as exc:
                if self.requested==path:self.load_result=(path,None,str(exc));self.requested=None
            self.loading=None
        if not self.loading and self.requested:
            self.loading_path=self.requested;self.loading=self.loader.submit(self._load,self.requested)
        if changed:self.revision+=1
        return changed

    @staticmethod
    def _load(path):
        if Path(path).stat().st_size>128*1024*1024:raise ValueError('Replay exceeds the 128 MiB interactive loading limit')
        data=parse_replay(path)
        if not data['ticks'] or not data['complete']:raise ValueError('No completed compatible replay snapshots; incomplete files are not played')
        return data

    def request(self,path):self.requested=str(Path(path).resolve())

    def take_loaded(self):
        result=self.load_result;self.load_result=None;return result

    @property
    def busy(self):return bool(self.queue or self.scan or self.pending)

    def close(self):
        if self.closed:return
        self.closed=True;self.pool.shutdown(wait=False,cancel_futures=True);self.loader.shutdown(wait=False,cancel_futures=True)

    def filtered(self,query='',condition='All',context='All',tag='All',sort='Newest',favorites=None,recent=None,bookmarks=None):
        rows=[];favorites=set(favorites or []);recent_set=set(recent or []);marked={b['path'] for b in bookmarks or []}
        signature=(self.revision,query,condition,context,tag,sort,frozenset(favorites),tuple(recent or []),frozenset(marked))
        if signature==self._filter_signature:return self._filtered_cache
        for e in self.entries.values():
            if condition!='All' and e['condition']!=condition:continue
            if context!='All' and e['context']!=context:continue
            if tag=='Favorites' and e['path'] not in favorites:continue
            if tag=='Recent' and e['path'] not in recent_set:continue
            if tag=='Bookmarked' and e['path'] not in marked:continue
            if tag not in ('All','Favorites','Recent','Bookmarked') and tag not in e['tags']:continue
            text=' '.join(str(e.get(k,'')) for k in ('name','condition','context','seed','session','reason'))+' '+' '.join(e['tags'])
            if query.casefold() not in text.casefold():continue
            rows.append(e)
        keys={'Newest':lambda e:-e['mtime'],'Seed':lambda e:(e['seed'] is None,e['seed'] or 0),'Duration':lambda e:-e['duration'],'Model':lambda e:e['condition'],'Outcome':lambda e:e['reason']}
        self._filtered_cache=sorted(rows,key=(lambda e:(recent or []).index(e['path'])) if tag=='Recent' else keys.get(sort,keys['Newest']))
        self._filter_signature=signature;return self._filtered_cache
