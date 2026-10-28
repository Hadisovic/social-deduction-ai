"""Learned action-family choices, actor-local geometry and bounded travel options."""
from dataclasses import asdict
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from social_deduction.actor import Phase
from social_deduction.phase3_api import IntentKind,WitnessedElimination
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
