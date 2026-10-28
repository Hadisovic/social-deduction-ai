"""New V2 checkpoint schema; pooled actor/task encoders and strict legal masks."""
from pathlib import Path
import torch
from torch import nn
from torch.distributions import Categorical
from crewmate_v2.config import LobbySettings
from . import SCHEMA
from .provenance import runtime_lock

SHAPES={'global':(42,),'players':(4,29),'own_tasks_v2':(4,8),'actions':(96,23),
        'targets':(96,),'task_targets':(96,),'mask':(96,)}


def check_packet(packet,batched=False):
    if set(packet)!=set(SHAPES):raise ValueError('Only allowlisted actor tensors may enter the policy')
    for key,shape in SHAPES.items():
        actual=tuple(packet[key].shape)
        if (actual[1:] if batched else actual)!=shape:raise ValueError('V2 policy dimension mismatch: '+key)
        if not torch.isfinite(torch.as_tensor(packet[key])).all():raise ValueError('Nonfinite actor input')
    for key in ('targets','task_targets'):
        value=torch.as_tensor(packet[key])
        if not ((value>=-1)&(value<4)&(value==value.long())).all():raise ValueError('Invalid actor row indices')
    mask=torch.as_tensor(packet['mask'])
    if not ((mask==0)|(mask==1)).all() or not mask.bool().any(-1).all():raise ValueError('Invalid legal-action mask')


def tensors(packet):
    check_packet(packet)
    return {key:torch.as_tensor(value).unsqueeze(0) for key,value in packet.items()}


class V2Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.player=nn.Sequential(nn.Linear(29,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.task=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU())
        self.context=nn.Sequential(nn.Linear(90,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.actor=nn.Sequential(nn.Linear(199,64),nn.ReLU(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.zeros_(self.actor[-1].bias)

    def forward(self,packet):
        check_packet(packet,True)
        players=self.player(packet['players'].float());tasks=self.task(packet['own_tasks_v2'].float())
        context=self.context(torch.cat((packet['global'].float(),players.mean(1),tasks.mean(1)),dim=-1))
        def gather(rows,key,width):
            ids=packet[key].long()
            values=rows.gather(1,ids.clamp_min(0).unsqueeze(-1).expand(-1,-1,width))
            return values*(ids>=0).unsqueeze(-1)
        p=gather(players,'targets',32);t=gather(tasks,'task_targets',16)
        logits=self.actor(torch.cat((context.unsqueeze(1).expand(-1,96,-1),p,t,packet['actions'].float()),dim=-1)).squeeze(-1)
        logits=logits.masked_fill(~packet['mask'].bool(),-1e9)
        return Categorical(logits=logits),self.critic(context).squeeze(-1)

    @torch.no_grad()
    def act(self,packet,deterministic=False):
        distribution,value=self(tensors(packet))
        action=distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()),float(distribution.log_prob(action).item()),float(value.item())

    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if metadata.get('runtime')!=runtime_lock(LobbySettings()):raise ValueError('Checkpoint runtime lock missing/mismatched')
        # Exclusive creation prevents accidental replacement of any checkpoint.
        with path.open('xb') as stream:
            torch.save(dict(schema=SCHEMA,shapes=SHAPES,metadata=metadata,state_dict=self.state_dict()),stream)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('schema')!=SCHEMA or data.get('shapes')!=SHAPES:raise ValueError('Historical/incompatible policy schema')
        metadata=data.get('metadata',{})
        if metadata.get('runtime')!=runtime_lock(LobbySettings()):raise ValueError('Learning checkpoint source/runtime mismatch')
        model=cls();model.load_state_dict(data['state_dict'],strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
        return model.eval(),metadata
