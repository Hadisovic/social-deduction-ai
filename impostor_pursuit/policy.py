"""A learned pursuit family separates visible-player goals from public destinations."""
import torch
from torch import nn
from torch.distributions import Categorical
from impostor_research.strategic_v2 import FamilyPolicy,FAMILIES as V2_FAMILIES
from impostor_research.features import KINDS,MAX_ACTIONS

SCHEMA='impostor-pursuit-v3'
FAMILIES=V2_FAMILIES+('pursuit',)


class PursuitPolicy(FamilyPolicy):
    def __init__(self,seed=71):
        super().__init__(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed^0x73C1)
            self.family_embedding=nn.Embedding(len(FAMILIES),16)
        self.kind_family[KINDS.index('go_to_location')]=FAMILIES.index('pursuit')

    def warm_start(self,parent):
        """Continue only this project's learned impostor; never copy crew weights."""
        parent_state=parent.state_dict();state=self.state_dict()
        for name,value in parent_state.items():
            if name not in ('family_embedding.weight','kind_family'):state[name]=value.clone()
        state['family_embedding.weight'][:len(V2_FAMILIES)]=parent_state['family_embedding.weight']
        state['family_embedding.weight'][-1]=parent_state['family_embedding.weight'][V2_FAMILIES.index('travel')]
        self.load_state_dict(state,strict=True)

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
        conditional_logp=candidate_logits-torch.logsumexp(grouped,dim=1).gather(1,families)
        distribution=Categorical(logits=(family_distribution.logits.gather(1,families)+conditional_logp).masked_fill(~mask,-1e9))
        conditional_p=conditional_logp.exp()*mask
        conditional_entropy=torch.einsum('ba,baf->bf',-conditional_p*conditional_logp,hot)
        conditional_bonus=(conditional_entropy/count.clamp_min(1).log().clamp_min(1)*family_distribution.probs).sum(-1)
        value=self.critic(torch.cat((globals_,self.critic_player(packet['players'].float()).mean(1)),dim=-1)).squeeze(-1)
        return distribution,value,family_distribution,families,conditional_bonus
