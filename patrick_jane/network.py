"""Locked three-encoder, attention, shared-GRU actor/critic; five actor heads."""
import torch
from torch import nn
from .features import SELF_WIDTH, PLAYER_WIDTH, EVENT_WIDTH, TARGET_WIDTH


def mlp(left,middle,right):
    return nn.Sequential(nn.Linear(left,middle),nn.Tanh(),nn.Linear(middle,right),nn.Tanh())


def safe_attention(module,query,keys,mask):
    """All-empty rows use a zero dummy; returned representation stays exactly zero."""
    mask=mask.bool();empty=~mask.any(-1)
    safe=mask.clone();safe[empty,0]=True
    keys=keys.masked_fill(~mask.unsqueeze(-1),0.)
    out,_=module(query,keys,keys,key_padding_mask=~safe,need_weights=False)
    return out.masked_fill(empty[:,None,None],0.)


class PatrickJane(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_encoder=mlp(SELF_WIDTH,256,128)
        self.player_encoder=mlp(PLAYER_WIDTH,128,128)
        self.player_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.event_encoder=mlp(EVENT_WIDTH,128,128)
        self.event_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(384,256),nn.Tanh())
        self.gru=nn.GRU(256,256,batch_first=True)
        self.actor=mlp(256,256,128)
        self.critic=mlp(256,256,128)
        self.navigation=nn.Linear(128,3)
        self.interaction=nn.Linear(128,5)
        self.communication=nn.Linear(128,6)
        self.voting=nn.Linear(128,15)
        self.target_encoder=nn.Linear(TARGET_WIDTH,128)
        self.target_attention=nn.MultiheadAttention(128,4,batch_first=True)
        self.target_query=nn.Linear(128,128)
        self.value=nn.Linear(128,1)
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.orthogonal_(module.weight,gain=2**.5);nn.init.zeros_(module.bias)
        for head in (self.navigation,self.interaction,self.communication,self.voting):
            nn.init.orthogonal_(head.weight,gain=.01)
        nn.init.orthogonal_(self.value.weight,gain=1.)

    def encode(self,batch):
        a=self.self_encoder(batch['self_features'])
        p=self.player_encoder(batch['players'])
        # Self attention among players, then a mask-safe mean pool.
        p=safe_attention(self.player_attention,p,p,batch['player_mask'])
        pm=batch['player_mask'].unsqueeze(-1)
        p=(p*pm).sum(1)/pm.sum(1).clamp_min(1)
        e=self.event_encoder(batch['events'])
        e=safe_attention(self.event_attention,a[:,None,:],e,batch['event_mask']).squeeze(1)
        fused=self.fusion(torch.cat((a,p,e),-1))
        return fused

    def decode(self,batch,shared):
        actor=self.actor(shared);critic=self.critic(shared)
        targets=torch.tanh(self.target_encoder(batch['targets']))
        retrieval=safe_attention(self.target_attention,actor[:,None,:],targets,batch['target_mask']).squeeze(1)
        query=self.target_query(actor)+retrieval
        scores=(targets*query[:,None,:]).sum(-1)/(128**.5)
        return dict(navigation=self.navigation(actor),interaction=self.interaction(actor),
                    communication=self.communication(actor),voting=self.voting(actor),target=scores,
                    value=self.value(critic).squeeze(-1))

    def forward(self,batch,hidden):
        recurrent,new_hidden=self.gru(self.encode(batch)[:,None,:],hidden.unsqueeze(0))
        return self.decode(batch,recurrent[:,0,:]),new_hidden.squeeze(0)

    def forward_sequence(self,batch,hidden):
        # One owner, contiguous time, one initial state; no cross-actor recurrence.
        recurrent,new_hidden=self.gru(self.encode(batch)[None,:,:],hidden.unsqueeze(0))
        return self.decode(batch,recurrent[0]),new_hidden.squeeze(0)

    def forward_sequences(self,batch,hidden,lengths):
        """Batch independent owners; padding follows their last used output only."""
        encoded=self.encode(batch)
        padded=nn.utils.rnn.pad_sequence(encoded.split(lengths),batch_first=True)
        recurrent,new_hidden=self.gru(padded,hidden.unsqueeze(0))
        shared=torch.cat([row[:n] for row,n in zip(recurrent,lengths)])
        final=torch.stack([row[n-1] for row,n in zip(recurrent,lengths)])
        return self.decode(batch,shared),final

    def architecture(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
            modules={name:sum(p.numel() for p in module.parameters()) for name,module in self.named_children()},
            linear_modules=sum(isinstance(m,nn.Linear) for m in self.modules()),
            attention_modules=3,attention_heads=4,attention_width=128,gru_layers=1,gru_hidden=256,
            longest_path='self/player/event encoder -> attention where applicable -> fusion -> shared GRU -> actor/critic MLP -> head',
            self_width=192,player_shape=[14,16],events_shape=['capacity=64',16],target_shape=['dynamic',16],
            outputs=dict(navigation=3,interaction=5,communication=6,voting=15,target='dynamic attention selector',value=1))


def collate(packets,device='cpu'):
    import numpy as np
    result={}
    for key in ('self_features','players','player_mask','events','event_mask'):
        result[key]=torch.as_tensor(np.stack([getattr(p,key) for p in packets]),device=device)
    n=max(len(p.targets) for p in packets)
    targets=np.zeros((len(packets),n,TARGET_WIDTH),np.float32);mask=np.zeros((len(packets),n),bool)
    for i,p in enumerate(packets):targets[i,:len(p.targets)]=p.targets;mask[i,:len(p.targets)]=True
    result['targets']=torch.as_tensor(targets,device=device);result['target_mask']=torch.as_tensor(mask,device=device)
    return result
