"""New randomly initialized recurrent candidate policy, inspired by Jane's design."""
import torch
from torch import nn
from .features import SELF_WIDTH, PLAYER_WIDTH, EVENT_WIDTH, ACTION_WIDTH, FAMILIES


def block(width, middle, output):
    return nn.Sequential(nn.Linear(width, middle), nn.Tanh(), nn.Linear(middle, output), nn.Tanh())


class ImpostorPolicy(nn.Module):
    def __init__(self, seed=731):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.own = block(SELF_WIDTH, 256, 128)
            self.people = block(PLAYER_WIDTH, 128, 128)
            self.people_pool = nn.MultiheadAttention(128, 4, batch_first=True)
            self.history = block(EVENT_WIDTH, 128, 128)
            self.history_pool = nn.MultiheadAttention(128, 4, batch_first=True)
            self.merge = nn.Sequential(nn.Linear(384, 256), nn.Tanh())
            self.memory = nn.GRU(256, 256, batch_first=True)
            self.actor = block(256, 256, 128)
            self.critic = block(256, 256, 128)
            self.family = nn.Linear(128, len(FAMILIES))
            self.keys = nn.Linear(ACTION_WIDTH, 128)
            self.candidate_pool = nn.MultiheadAttention(128, 4, batch_first=True)
            self.query = nn.Linear(128, 128)
            self.family_query = nn.Embedding(len(FAMILIES), 128)
            self.value = nn.Linear(128, 1)
            nn.init.orthogonal_(self.family.weight, .01)
            nn.init.zeros_(self.family.bias)

    @staticmethod
    def attend(module, query, keys, valid):
        empty = ~valid.any(-1); safe = valid.clone(); safe[empty, 0] = True
        keys = keys.masked_fill(~valid[..., None], 0.)
        result, _ = module(query, keys, keys, key_padding_mask=~safe, need_weights=False)
        return result.masked_fill(empty[:, None, None], 0.)

    def forward(self, packets, hidden, sequence=False):
        device = next(self.parameters()).device
        stack = lambda name: torch.as_tensor(__import__('numpy').stack([getattr(p,name) for p in packets]), device=device)
        own = self.own(stack('self_features'))
        pm = stack('player_mask').bool(); people = self.people(stack('players'))
        people = self.attend(self.people_pool, people, people, pm)
        people = (people*pm[...,None]).sum(1)/pm.sum(1).clamp_min(1)[:,None]
        events = self.history(stack('events'))
        events = self.attend(self.history_pool, own[:,None], events, stack('event_mask').bool())[:,0]
        merged = self.merge(torch.cat((own, people, events), -1))
        if sequence:
            recurrent, next_hidden = self.memory(merged[None], hidden[None])
            states = recurrent[0]
        else:
            recurrent, next_hidden = self.memory(merged[:,None], hidden[None])
            states = recurrent[:,0]
        actor = self.actor(states); values = self.value(self.critic(states))[:,0]
        distributions = []
        for i, packet in enumerate(packets):
            actions = torch.as_tensor(packet.actions, device=device)
            families = torch.as_tensor(packet.families, device=device)
            keys = torch.tanh(self.keys(actions))
            retrieved, _ = self.candidate_pool(actor[i:i+1,None], keys[None], keys[None], need_weights=False)
            q = self.query(actor[i]) + retrieved[0,0]
            scores = (keys*(q[None]+self.family_query(families))).sum(-1)/128**.5
            available = torch.zeros(len(FAMILIES), dtype=torch.bool, device=device)
            available[families] = True
            family_lp = torch.log_softmax(self.family(actor[i]).masked_fill(~available, -torch.inf), -1)
            normalizers = torch.stack([torch.logsumexp(scores[families==f],0) if available[f]
                                       else scores.sum()*0 for f in range(len(FAMILIES))])
            joint = family_lp[families]+scores-normalizers[families]
            distributions.append(torch.distributions.Categorical(logits=joint))
        return distributions, values, next_hidden[0]

    def architecture(self):
        from . import MODEL_NAME
        return dict(name=MODEL_NAME,schema='jane-impostor-recurrent-a2-v1', parameters=sum(p.numel() for p in self.parameters()),
                    self_width=192, player_shape=[14,24], events_shape=[64,16], candidate_width=48,
                    families=list(FAMILIES), hidden=256, actor_critic='shared recurrent backbone, separate trunks',
                    weights='fresh initialization; not copied from crew or old impostor')
