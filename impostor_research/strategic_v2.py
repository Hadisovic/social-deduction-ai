"""Learned action-family choices, actor-local geometry and bounded travel options."""
from dataclasses import asdict
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind,WitnessedElimination
from .env import ImpostorEnv,RewardConfig
from .features import KINDS,GLOBAL_COUNT,PLAYER_COUNT,ACTION_COUNT,MAX_ACTIONS

SCHEMA='impostor-strategic-v2'
FAMILIES=('continue','travel','cancel','fake_task','kill','approach_body','report','emergency','claim','vote','skip','interact')
EXTRA_COUNT=8
ACTION_COUNT_V2=ACTION_COUNT+3


def family(kind):
    if kind=='wait':return 'continue'
    if kind in ('go_to_room','go_to_task','go_to_location'):return 'travel'
    if kind=='cancel_navigation':return 'cancel'
    if kind=='go_to_body':return 'approach_body'
    if kind=='call_meeting':return 'emergency'
    if kind.startswith('claim_'):return 'claim'
    return kind


class FamilyPolicy(nn.Module):
    """Joint p(family|state)*p(candidate|family,state), without scripted choices."""
    def __init__(self,seed=71):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.player=nn.Sequential(nn.Linear(PLAYER_COUNT,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
            self.context=nn.Sequential(nn.Linear(GLOBAL_COUNT+EXTRA_COUNT+32,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
            self.candidate=nn.Sequential(nn.Linear(128+32+ACTION_COUNT_V2,64),nn.Tanh())
            self.candidate_score=nn.Linear(64,1)
            self.family_embedding=nn.Embedding(len(FAMILIES),16)
            self.family_score=nn.Sequential(nn.Linear(128+64+16,64),nn.Tanh(),nn.Linear(64,1))
            # Critic has its own encoder: value fitting cannot swamp actor gradients.
            self.critic_player=nn.Sequential(nn.Linear(PLAYER_COUNT,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
            self.critic=nn.Sequential(nn.Linear(GLOBAL_COUNT+EXTRA_COUNT+32,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
            nn.init.orthogonal_(self.candidate_score.weight,.05);nn.init.zeros_(self.candidate_score.bias)
            nn.init.orthogonal_(self.family_score[-1].weight,.05);nn.init.zeros_(self.family_score[-1].bias)
        self.register_buffer('kind_family',torch.tensor([FAMILIES.index(family(k)) for k in KINDS]))

    def actor_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith('critic')]
    def critic_parameters(self):return [p for n,p in self.named_parameters() if n.startswith('critic')]

    def evaluate(self,packet):
        players=self.player(packet['players'].float())
        globals_=torch.cat((packet['global_'].float(),packet['extra'].float()),dim=-1)
        context=self.context(torch.cat((globals_,players.mean(1)),dim=-1))
        targets=packet['targets'].long();mask=packet['mask'].bool()
        gathered=players.gather(1,targets.clamp_min(0).unsqueeze(-1).expand(-1,-1,32))*(targets>=0).unsqueeze(-1)
        h=self.candidate(torch.cat((context.unsqueeze(1).expand(-1,MAX_ACTIONS,-1),gathered,packet['actions'].float()),dim=-1))
        families=self.kind_family[packet['actions'][...,:len(KINDS)].argmax(-1)]
        hot=torch.nn.functional.one_hot(families,len(FAMILIES)).float()*mask.unsqueeze(-1)
        count=hot.sum(1);pooled=torch.einsum('baf,bah->bfh',hot,h)/count.clamp_min(1).unsqueeze(-1)
        embeddings=self.family_embedding.weight.unsqueeze(0).expand(h.shape[0],-1,-1)
        family_logits=self.family_score(torch.cat((context.unsqueeze(1).expand(-1,len(FAMILIES),-1),pooled,embeddings),dim=-1)).squeeze(-1)
        family_distribution=Categorical(logits=family_logits.masked_fill(count==0,-1e9))
        candidate_logits=self.candidate_score(h).squeeze(-1)
        grouped=candidate_logits.unsqueeze(-1).expand(-1,-1,len(FAMILIES)).masked_fill(hot==0,-1e9)
        normalizer=torch.logsumexp(grouped,dim=1)
        conditional_logp=candidate_logits-normalizer.gather(1,families)
        joint=family_distribution.logits.gather(1,families)+conditional_logp
        distribution=Categorical(logits=joint.masked_fill(~mask,-1e9))
        conditional_p=conditional_logp.exp()*mask
        conditional_entropy=torch.einsum('ba,baf->bf',-conditional_p*conditional_logp,hot)
        normalized=conditional_entropy/count.clamp_min(1).log().clamp_min(1)
        conditional_bonus=(normalized*family_distribution.probs).sum(-1)
        value=self.critic(torch.cat((globals_,self.critic_player(packet['players'].float()).mean(1)),dim=-1)).squeeze(-1)
        return distribution,value,family_distribution,families,conditional_bonus

    def forward(self,packet):
        distribution,value,*_=self.evaluate(packet)
        return distribution,value

    @torch.inference_mode()
    def act(self,packet,*,generator=None,deterministic=True):
        data={k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()}
        distribution,value,families,ids,_=self.evaluate(data)
        if deterministic:
            chosen_family=families.probs.argmax(-1)
            action=distribution.logits.masked_fill(ids!=chosen_family.unsqueeze(-1),-1e9).argmax(-1)
        else:
            if generator is None:raise ValueError('Sampling requires a private RNG')
            action=torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
        return int(action.item()),float(distribution.log_prob(action).item()),float(value.item())


class StrategicEnv(ImpostorEnv):
    """Same V2 world. Hold chosen travel up to 3s; offer every legal kill window.

    No action is automatically chosen. Reward gives +2 per successful kill, with
    terminal wins/losses dominant and an extra -2 for own ejection. No witness-truth
    shaping, survival, fake-task or reporting reward is used.
    """
    def __init__(self,*,models=None,trace=False,**kwargs):
        self.goal=None;self.policy_decisions=0;self.strategic_return=0.;self.render_trace=trace
        self.visual_frames=[];self.last_kind='continue';self.legal_kill_windows=0
        super().__init__(models=models,reward=RewardConfig(timeout=-12.,per_second=-.002,gamma_per_second=1.),**kwargs)
        from gymnasium import spaces
        self.observation_space.spaces['extra']=spaces.Box(-1,1,(EXTRA_COUNT,),np.float32)
        self.observation_space.spaces['actions']=spaces.Box(-1,1,(MAX_ACTIONS,ACTION_COUNT_V2),np.float32)

    def reset(self,**kwargs):
        self.goal=None;self.policy_decisions=0;self.strategic_return=0.;self.last_kind='continue';self.legal_kill_windows=0
        result=super().reset(**kwargs)
        self.visual_frames=[self._frame()]
        return result

    def _frame(self):return self.match.game.time,{pid:(p.position,p.velocity) for pid,p in self.match.game.players.items()}

    def _observe(self):
        super()._observe();obs=self.envelope.actor
        if obs.context.phase is Phase.FINISHED:
            self.packet['extra']=np.zeros(EXTRA_COUNT,np.float32)
            return
        kills=[e for e in self.memory.events if type(e.payload) is WitnessedElimination and e.payload.killer_id==obs.own.player_id]
        x,y=obs.own.position.x,obs.own.position.y
        last=max((e.tick for e in kills),default=None)
        body_distance=min((math.hypot(b.position.x-x,b.position.y-y)/60 for b in obs.visible_bodies),default=0.)
        self.packet['extra']=np.asarray([float(bool(obs.visible_bodies)),min(body_distance,1.),len(kills)/3,
            min((obs.tick-last)*obs.settings.dt/900,1.) if last is not None else 1.,
            (self.goal.x-x)/60 if self.goal else 0.,(self.goal.y-y)/60 if self.goal else 0.,
            float(self.goal is not None and obs.navigation_status=='moving'),float(any(a.kind is IntentKind.KILL for a in obs.actions))],np.float32)
        geometry=np.zeros((MAX_ACTIONS,3),np.float32)
        seen={p.player_id:p for p in obs.visible_players}
        for i,c in enumerate(self.options):
            target=seen.get(c.target)
            if target:
                distances=[math.hypot(p.position.x-target.position.x,p.position.y-target.position.y) for p in obs.visible_players if p.player_id!=target.player_id]
                geometry[i]=[1.,min(min(distances,default=20.)/20,1.),sum(d<=4.5 for d in distances)/3]
        self.packet['actions']=np.concatenate((self.packet['actions'],geometry),axis=1)
        self.actor_trace.update(self.packet['extra'].tobytes());self.actor_trace.update(geometry.tobytes())

    def _advance(self,intent):
        # Tick orchestration only; unchanged game engine owns all physics/rules.
        reward=raw=elapsed=0.;forced=0
        while True:
            for index in range(self.match.decision_ticks):
                self.match.tick(intent if elapsed==0 and index==0 else None)
                if self.render_trace:self.visual_frames.append(self._frame())
                dt=self.match.game.config.dt;r=self.reward_config.per_second*dt
                if self.match.game.is_terminal:
                    result=self.match.game.result
                    from social_deduction.actor import Role
                    r+=self.reward_config.win if result.winner is Role.IMPOSTOR else self.reward_config.loss if result.winner is Role.CREWMATE else self.reward_config.timeout
                reward+=r;raw+=r;elapsed+=dt
                if self.match.game.is_terminal:break
            self.windows+=1;self._observe()
            if self.match.game.is_terminal or not self.coalesce or len(self.options)>1:break
            forced+=1
        self.finished=self.match.game.is_terminal;self.total_reward+=raw
        return self.packet,float(reward),self.finished,False,dict(elapsed_seconds=elapsed,discount=1.,undiscounted_reward=raw,forced_windows_coalesced=forced)

    def step(self,action):
        if self.finished:raise RuntimeError('Reset after terminal')
        if not self.action_space.contains(action) or not self.packet['mask'][int(action)]:raise ValueError('Invalid masked selection')
        choice=self.options[int(action)];self.last_kind=choice.kind;self.policy_decisions+=1
        if family(choice.kind)=='travel':self.goal=choice.position
        elif choice.kind=='cancel_navigation':self.goal=None
        initial=self.envelope.actor;known_players={p.player_id for p in initial.visible_players};known_bodies={b.victim_id for b in initial.visible_bodies}
        reward=elapsed=0.;initial_kills=self.match.game.metrics['eliminations']
        self.visual_frames=[self._frame()] if self.render_trace else []
        extended=family(choice.kind)=='travel'
        while True:
            _,r,done,_,info=self._advance(choice.intent if elapsed==0 else None)
            elapsed+=info['elapsed_seconds'];reward+=r
            obs=self.envelope.actor
            if any(a.kind is IntentKind.KILL for a in obs.actions):self.legal_kill_windows+=1
            if (done or not extended or elapsed>=3.-1e-8 or obs.context.phase is not Phase.ROAMING
                or obs.context.phase is not initial.context.phase or obs.navigation_status!='moving'
                or {p.player_id for p in obs.visible_players}-known_players
                or {b.victim_id for b in obs.visible_bodies}-known_bodies
                or any(a.kind is IntentKind.KILL for a in obs.actions)):
                break
        kill_bonus=2.*(self.match.game.metrics['eliminations']-initial_kills)
        ejection_penalty=-2. if done and self.match.game.result.reason=='IMPOSTOR_EJECTED' else 0.
        reward+=kill_bonus+ejection_penalty;self.strategic_return+=reward
        info.update(elapsed_seconds=elapsed,discount=1.,undiscounted_reward=reward,kill_bonus=kill_bonus,ejection_penalty=ejection_penalty)
        if done:info['episode']=self.summary()
        return self.packet,float(reward),done,False,info

    def summary(self):
        result=super().summary()
        result.update(return_=self.strategic_return,meaningful_decisions=self.policy_decisions,legal_kill_windows=self.legal_kill_windows,
                      schema=SCHEMA)
        return result
