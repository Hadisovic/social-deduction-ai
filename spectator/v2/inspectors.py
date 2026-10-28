"""Selected-agent views: recorded choices/beliefs/rewards, never explanations."""
from collections import Counter
import math
import pygame
from phase3_renderer import player_color
from .data import MISSING,time_of,family,event_text
from .widgets import *


class Inspector:
    def draw(self,surface,rect,app,ui):
        index=app.index
        if not index or app.selected not in index.frame().players:return
        pid=app.selected;player=index.frame().players[pid];scale=app.prefs['scale'];s=lambda v:round(v*scale)
        x=rect.x+18;w=rect.w-36;y=rect.y+16
        role=index.role(pid) if app.prefs['omniscient'] and app.prefs['reveal_roles'] else 'Role hidden'
        color=player_color(player.color)
        pygame.draw.circle(surface,color,(x+8,y+13),7)
        text(surface,pid.upper(),x+26,y,24,TEXT,True)
        text(surface,str(role).upper(),x+26,y+s(31),12,RED if role=='impostor' else CYAN)
        controller=index.header.get('controller_types',{}).get(pid,MISSING);label=controller
        if index.config.get('training_condition')=='zero' and label=='ppo':label='frozen policy · zero updates'
        if label=='ppo' and app.prefs['neural']:label='PPO · selected-actor private debug'
        if player.status!='alive':label+=' · inactive / automatic ghost'
        elif index.frame().sim_time<index.config.get('prefix_handoff_seconds',0):label='scripted prefix · no learning'
        text(surface,label,x,y+s(61),12,MUTED,width=w)
        tabs=['Mind','Belief','Rewards','Tasks','Evidence'] if controller in ('ppo','frozen') or pid in index.neurals else ['Controller','Tasks','Evidence','Rewards']
        if role=='impostor':tabs=['Impostor','Evidence','Rewards']
        if app.tab not in tabs:app.tab=tabs[0]
        ty=y+s(88);columns=3 if len(tabs)>4 else len(tabs);rows=math.ceil(len(tabs)/columns);tw=(w-4*(columns-1))//columns
        for i,name in enumerate(tabs):
            ui.button(surface,'inspect-tab-'+name,name,pygame.Rect(x+(i%columns)*(tw+4),ty+(i//columns)*s(36),tw,s(32)),('tab',name),app.tab==name,tooltip='Selected actor only')
        body=pygame.Rect(rect.x+12,ty+s(43+(rows-1)*36),rect.w-24,max(10,rect.bottom-ty-s(49+(rows-1)*36)))
        old=surface.get_clip();surface.set_clip(body)
        self.surface=surface;self.ui=ui;self.app=app;self.index=index;self.pid=pid;self.x=body.x+7;self.w=body.w-14;self.s=s
        self.y=body.y-app.scrolls.get('inspector',0);self.start=self.y
        if app.tab=='Mind':self.mind()
        elif app.tab=='Belief':self.belief()
        elif app.tab=='Rewards':self.rewards()
        elif app.tab=='Evidence':self.evidence()
        elif app.tab=='Tasks':self.tasks()
        else:self.impostor(role=='impostor')
        total=max(0,self.y-self.start-body.h)
        app.scroll_regions.append((body,'inspector',total))
        surface.set_clip(old)
        if total>0:
            track=pygame.Rect(rect.right-6,body.y,3,body.h);progress(surface,track,None)
            thumb=pygame.Rect(track.x,track.y+round((body.h-34)*min(1,app.scrolls.get('inspector',0)/total)),3,34)
            pygame.draw.rect(surface,CYAN,thumb,border_radius=1)

    def title(self,value,color=CYAN):
        self.y+=self.s(14);text(self.surface,value.upper(),self.x,self.y,12,color,True,width=self.w)
        self.y+=self.s(26)

    def line(self,label,value,color=TEXT):
        if value==MISSING and not self.app.prefs['unavailable']:return
        text(self.surface,label,self.x,self.y,13,MUTED,width=self.w*.48)
        text(self.surface,str(value),self.x+self.w*.48,self.y,13,color,width=self.w*.52,mono=True)
        self.y+=self.s(26)

    def note(self,value,color=MUTED):
        f=font(13);words=str(value).split();line=''
        for word in words:
            if line and f.size(line+' '+word)[0]>self.w:
                text(self.surface,line,self.x,self.y,13,color);self.y+=self.s(21);line=word
            else:line=(line+' '+word).strip()
        if line:text(self.surface,line,self.x,self.y,13,color);self.y+=self.s(24)

    def bar(self,label,value,color=CYAN):
        text(self.surface,label,self.x,self.y,13,TEXT,width=self.w-65)
        text(self.surface,f'{value*100:.1f}%',self.x+self.w-60,self.y,13,color,mono=True)
        self.y+=self.s(22);progress(self.surface,pygame.Rect(self.x,self.y,self.w,self.s(5)),value,color);self.y+=self.s(16)

    def button(self,key,label,action,accent=CYAN):
        self.ui.button(self.surface,key,label,pygame.Rect(self.x,self.y,self.w,self.s(30)),action,accent=accent)
        self.y+=self.s(38)

    def chart(self,series,color=CYAN,height=90,range_y=None):
        r=pygame.Rect(self.x,self.y,self.w,self.s(height));box(self.surface,r,CARD,LINE,6)
        self.y=r.bottom+self.s(12)
        if len(series)<2:
            text(self.surface,'More recorded points needed',r.x+10,r.y+18,12,MUTED,width=r.w-15);return
        lo,hi=range_y or (min(y for x,y in series),max(y for x,y in series));span=max(1e-9,hi-lo);t0,t1=series[0][0],series[-1][0]
        points=[(r.x+8+(x-t0)/max(1e-9,t1-t0)*(r.w-16),r.bottom-14-(y-lo)/span*(r.h-28)) for x,y in series]
        pygame.draw.lines(self.surface,color,False,points,2)
        text(self.surface,f'{lo:.2f} — {hi:.2f}',r.x+8,r.y+5,11,MUTED,mono=True)

    def mind(self):
        if not self.app.prefs['neural']:self.note('Neural details are disabled in Settings.');return
        if self.index.config.get('patrick_jane_schema'):
            self.patrick_jane();return
        n=self.index.neural(self.pid)
        if not n:self.note('No neural decision is recorded at or before this moment. Scripted prefixes and inactive ghosts do not invent policy decisions.');return
        self.title('Recorded decision')
        self.note(str(n.get('action_kind') or MISSING).replace('_',' ').upper(),LIME)
        self.line('Target',n.get('action_target') if n.get('action_target') is not None else 'None recorded')
        self.line('Timestamp',f'{time_of(n):.2f}s')
        self.line('Age',f'{self.index.frame().sim_time-time_of(n):.2f}s')
        self.line('Family',family(n.get('action_kind','')))
        self.line('Selection mode',n.get('selection_mode',MISSING))
        self.line('Next decision',n.get('next_decision_time',MISSING))
        self.line('Critic value',f"{n['value_estimate']:.3f}" if isinstance(n.get('value_estimate'),(int,float)) else MISSING)
        self.line('Policy entropy',f"{n['policy_entropy']:.3f} nats" if isinstance(n.get('policy_entropy'),(int,float)) else MISSING)
        if self.index.frame().players[self.pid].status!='alive':self.note('Last living decision. This policy is now inactive.',AMBER)
        self.title('Candidate action probabilities')
        probabilities=n.get('action_probabilities',{})
        items=sorted(((k,v) for k,v in probabilities.items() if isinstance(v,(int,float)) and math.isfinite(v)),key=lambda p:-p[1])
        selected=n.get('selected_action_index')
        for key,value in items[:len(items) if self.app.all_actions else 7]:self.bar(key,value,LIME if isinstance(selected,int) and key.startswith(f'{selected:02d}:') else CYAN)
        if items and abs(sum(v for _,v in items)-1)>.001:self.note('Candidate mass is incomplete or unnormalized in this historical record.',AMBER)
        if len(items)>7:self.button('all-actions','Show fewer' if self.app.all_actions else f'Show all {len(items)} candidates',('all_actions',))
        self.title('Grouped action mass')
        grouped=Counter()
        for key,value in items:grouped[family(key)]+=value
        for key,value in grouped.most_common():self.bar(key,value,PURPLE)
        self.note('Grouped action mass sums candidate probabilities. It is not an impostor belief or a causal explanation.')
        self.title('Legal / masked actions')
        legal=n.get('legal_actions');self.line('Legal candidates',len(legal) if isinstance(legal,list) else MISSING)
        for key in (legal or [])[:16]:self.note(key)
        self.line('Masked entries',len(n['masked_actions']) if isinstance(n.get('masked_actions'),list) else MISSING)
        self.title('Recent owner decisions')
        for record in self.index.history(self.pid)[-12:][::-1]:
            self.button('decision-'+str(time_of(record)),f"{time_of(record):.1f}s  {record.get('action_kind','decision')}",('seek',time_of(record)))
        self.title('Model provenance')
        self.note(str(self.index.config.get('training_condition',MISSING)))
        version=self.index.config.get('policy_version',MISSING)
        self.line('Policy update','0 (frozen reference)' if version==-1 and self.index.config.get('training_condition')=='zero' else version)
        self.note(self.index.header.get('checkpoint_path') or 'Physical checkpoint path not recorded; runtime/model labels are recorded.')

    def patrick_jane(self):
        n=self.index.neural(self.pid)
        self.title('Patrick Jane · recurrent PPO')
        if not n:self.note('No living neural decision recorded yet.');return
        self.line('Actual action',n.get('action_kind',MISSING));self.line('Target',n.get('action_target') or 'None')
        self.line('Decision time',f'{time_of(n):.2f}s');self.line('Update',n.get('policy_update',MISSING))
        self.line('Critic value',n.get('value_estimate',MISSING));self.line('Joint entropy',n.get('policy_entropy',MISSING))
        self.note('Shared weights; independent 256-unit GRU and timestamped memory for this actor.')
        probabilities=n.get('action_probabilities',{})
        for head in ('interaction','navigation','communication','voting','target'):
            values=[(key.split('/',1)[1],p) for key,p in probabilities.items() if key.startswith(head+'/')]
            if not values:continue
            self.title(head+' · conditional distribution')
            for key,p in sorted(values,key=lambda x:-x[1])[:len(values) if self.app.all_actions else 6]:self.bar(key,p)
            self.note('Normalized within this sampled head. Different heads are not one flat action distribution.')
        self.button('all-actions','Show fewer targets' if self.app.all_actions else 'Show all targets',('all_actions',))
        routes=[e for e in self.index.events if e.get('kind')=='patrick_jane_route' and e.get('player_id')==self.pid and time_of(e)<=self.index.frame().sim_time]
        if routes:
            self.title('Recorded navigation');self.line('Destination',routes[-1].get('destination') or 'None')
            self.line('Route points',len(routes[-1].get('route',[])))
        self.title('Model provenance');self.note(self.index.header.get('model_architecture',MISSING))
        self.note(self.index.header.get('checkpoint_path',MISSING))

    def belief(self):
        if self.index.config.get('patrick_jane_schema'):
            self.title('No separate belief output')
            self.note('Patrick Jane learns from private player/event encoders and recurrent memory. Its vote and target action probabilities are not calibrated impostor probabilities. The historical frozen belief predictor is not used.');return
        if not self.app.prefs['evidence']:self.note('Suspicion and evidence are disabled in Settings.');return
        n=self.index.neural(self.pid);probs=n.get('suspicion_probabilities',{}) if n else {}
        self.title('Private suspicion · frozen Phase 4')
        self.note('One shared frozen predictor, evaluated on this actor’s history. These are not the PPO action probabilities.')
        if not probs:self.note(MISSING);return
        for pid,value in sorted(probs.items(),key=lambda p:-p[1]):self.bar(pid,value,RED if value>=.9 else AMBER)
        self.title('Recorded suspicion history')
        history=self.index.history(self.pid)
        targets=sorted(probs)
        for i,pid in enumerate(targets):
            points=[(time_of(n),n['suspicion_probabilities'][pid]) for n in history if pid in n.get('suspicion_probabilities',{})]
            self.line('Candidate',pid)
            self.chart(points,(CYAN,AMBER,PURPLE,LIME,RED)[i%5],65,(0.,1.))
        if self.app.prefs['omniscient']:
            best=max(probs.values());leaders=[pid for pid,value in probs.items() if value==best]
            wrong=all(self.index.role(pid)=='crewmate' for pid in leaders)
            self.title('Spectator-only truth check',AMBER)
            self.note('High-confidence wrong prediction (>=90%)' if best>=.9 and wrong else 'No >=90% wrong prediction in this record.',RED if best>=.9 and wrong else MUTED)
        self.note('Private observations and numeric suspicion are distinct. This replay cannot prove which evidence caused a probability change.')

    def rewards(self):
        if not self.app.prefs['rewards']:self.note('Reward details are disabled in Settings.');return
        self.title('Recorded scoring profile')
        self.note(self.index.reward_label(),AMBER)
        self.note('Model condition: '+str(self.index.config.get('training_condition',MISSING)))
        reward=self.index.reward(self.pid)
        if not reward:self.note('Reward values are not recorded at this moment. No values are inferred from kills, tasks, votes or team outcomes.');return
        r=reward['latest'];self.line('Last reward',f"{r['reward']:+.4f}" if isinstance(r.get('reward'),(int,float)) else MISSING)
        self.line('Cumulative',f"{r['cumulative_reward']:+.4f}" if isinstance(r.get('cumulative_reward'),(int,float)) else MISSING,LIME)
        self.line('Recorded at',f'{time_of(r):.2f}s')
        self.title('Components over recorded intervals')
        for key,value in sorted(reward['components'].items(),key=lambda p:-abs(p[1])):
            self.line(key.replace('_',' '),f'{value:+.4f}',RED if value<0 else LIME)
        balance=reward['unattributed_balance']
        if balance is not None and abs(balance)>1e-6:
            self.line('Unattributed balance',f'{balance:+.4f}',AMBER)
            self.note('The recorded cumulative total includes a baseline or interval not decomposed by these reward records. Its component attribution is not inferred.',AMBER)
        self.title('Reward timeline')
        self.chart([(time_of(r),r['cumulative_reward']) for r in reward['history'] if isinstance(r.get('cumulative_reward'),(int,float))],LIME,115)
        self.note('Components sum only recorded reward intervals; they may not cover the whole cumulative total. Team/ghost credit is never invented.')
        profile=self.index.config.get('reward_profile')
        if isinstance(profile,dict):
            self.title('Recorded coefficients')
            for key,value in profile.items():
                if isinstance(value,(int,float)):self.line(key.replace('_',' '),value)

    def evidence(self):
        if not self.app.prefs['evidence']:self.note('Evidence details are disabled in Settings.');return
        self.title('Knowledge provenance')
        self.note('Public events below were available by the displayed tick. Ground-truth positions and hidden ballots are not treated as private evidence.')
        evidence=self.index.evidence(self.pid)
        self.line('Private packet',MISSING if evidence['private'] is None else 'Recorded')
        self.line('Last-seen memory',MISSING)
        self.line('Private sightings',MISSING)
        self.line('Task stages / readiness',MISSING)
        self.note('Historical recordings omit full observation/memory packets. Missing knowledge cannot be reconstructed from spectator truth.',AMBER)
        n=self.index.neural(self.pid)
        if n:
            legal=n.get('legal_actions')
            self.line('Report option at decision',('Yes' if any(family(a)=='Reporting' for a in legal) else 'No') if isinstance(legal,list) else MISSING)
            self.line('Decision time',f'{time_of(n):.2f}s')
        self.title('Compare two actual decisions')
        self.button('pin-a','Pin current owner decision as A',('pin','A'))
        self.button('pin-b','Pin current owner decision as B',('pin','B'))
        a,b=self.app.pins.get('A'),self.app.pins.get('B')
        if a and b and a[0]==b[0]==self.pid:
            future=max(time_of(a[1]),time_of(b[1]))>self.index.frame().sim_time+1e-8
            if future and not self.app.retrospective_compare:
                self.note('A pinned decision is later than the displayed tick; it is withheld.')
                self.button('retrospective-compare','Open retrospective comparison',('retrospective_compare',))
            else:
                delta=self.index.decision_delta(self.pid,a[1],b[1])
                self.line('Recorded interval',f"{delta['start']:.1f}–{delta['end']:.1f}s")
                self.line('Distribution TV',f"{delta['total_variation']:.4f}")
                for pid,value in delta['suspicion_delta'].items():self.line('Suspicion Δ '+pid,f'{value:+.3f}')
                self.line('Public events between',len(delta['public_events']))
                self.note(delta['interpretation'])
                if future:self.note('RETROSPECTIVE COMPARISON: explicitly selected future pins.',AMBER)
        else:self.note('Jump to a claim or vote, pin its preceding owner decision, then pin a second moment. Records stay actor-specific.')
        self.title('Public event history')
        for e in evidence['public'][-16:][::-1]:
            label=f'{time_of(e):.1f}s · '+event_text(e)
            self.button('evidence-event-'+str(id(e)),label,('seek',time_of(e)))

    def tasks(self):
        self.title('Recorded owned tasks')
        progress=self.index.task_progress(self.pid)
        if progress is None:self.note('Owned-task totals and progress are not recorded for this player. No completion percentage is guessed.');return
        self.bar('Completed owned tasks',progress['fraction'],LIME)
        self.line('Completed / assigned',f"{progress['completed']} / {progress['total']}")
        self.line('Remaining',progress['total']-progress['completed'])
        self.line('Active interaction',progress['active'] or 'None recorded')
        if isinstance(progress['stage'],(int,float)):self.bar('Current task progress',progress['stage'],CYAN)
        else:self.line('Active task progress',MISSING)
        self.line('Task stage details',MISSING)
        self.line('Readiness / task cooldowns',MISSING)
        counts=self.index.actor_stats(self.pid)
        self.title('Recorded completion events')
        self.line('Owner task events',counts['tasks'])
        self.line('Ghost completions',counts['ghost_tasks'])
        self.note('Ghost event counts are derived from recorded status at completion. They are contributions, not invented personal rewards. Automatic ghost behavior is separate from PPO decisions.')

    def impostor(self,revealed):
        self.title('Recorded controller')
        self.note(self.index.header.get('controller_types',{}).get(self.pid,MISSING))
        raw=self.index.raw(self.pid);p=self.index.frame().players[self.pid]
        self.line('Location',self.app.world.region(p.position))
        self.line('Action',raw.get('intention',MISSING))
        self.line('Destination',raw.get('destination',MISSING))
        self.line('Explicit target',raw.get('target_player',MISSING))
        if revealed:
            self.title('Kill cooldown',RED)
            cd=raw.get('cooldown');maximum=self.index.config.get('kill_cooldown')
            self.line('Recorded remaining',f'{cd:.1f}s' if isinstance(cd,(int,float)) else MISSING)
            if isinstance(cd,(int,float)) and isinstance(maximum,(int,float)) and maximum>0:
                self.bar('Cooldown ready fraction',1-max(0,min(1,cd/maximum)),RED)
            self.line('Cooldown ready',str(cd<=0) if isinstance(cd,(int,float)) else MISSING)
            self.line('Legal kill actions',MISSING)
            self.note('Cooldown readiness alone does not establish legal range, line of sight, target eligibility or full kill legality.')
            self.line('Actor-visible players',raw.get('visible_players',MISSING))
            self.line('Actor-visible bodies',raw.get('visible_bodies',MISSING))
            nearest=[(math.dist(p.position,o.position),pid) for pid,o in self.index.frame().players.items() if pid!=self.pid and o.status=='alive']
            if nearest:
                dist,pid=min(nearest);self.line('Nearest alive (truth)',f'{pid} / {dist:.2f}m')
                self.note('Nearest distance is spectator geometry, not an actor visibility assertion.')
        stats=self.index.actor_stats(self.pid)
        self.title('Events up to this tick')
        for key in ('kills','reports','ballots','tasks'):self.line(key.title(),stats[key])
        for e in self.index.events_until()[-80:][::-1]:
            if self.pid in (e.get('killer_id'),e.get('reporter_id'),e.get('voter_id'),e.get('speaker_id')):
                self.button('script-event-'+str(id(e)),f'{time_of(e):.1f}s · '+event_text(e,self.app.prefs['omniscient'] and self.app.prefs['reveal_ballots']),('seek',time_of(e)))
        self.title('Script strategy / sabotage')
        self.line('Script profile',self.index.config.get('script_profile',MISSING))
        self.note('Sabotage: Not implemented by the current simulator interface. No controls or fake events are added.',AMBER)
