"""Permutation-equivariant candidate scorer and masked categorical PPO policy."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from torch.distributions import Categorical
from phase5_features import SCHEMA,GLOBAL_NAMES,PLAYER_FEATURES,ACTION_FEATURES,KINDS


def tensors(packet):
    return {k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()}


def validate_runtime(metadata):
    """Reject a changed map/backbone at deployment; hashes are never input features."""
    root=Path(__file__).resolve().parent
    for key,path,normalize in [('map_sha256','assets/skeld/among_us_map.json',False),
                               ('belief_sha256','artifacts/phase4/belief.pt',False),
                               ('features_sha256','phase5_features.py',True),
                               ('geometry_sha256','among_us_map.py',True)]:
        if key in metadata:
            data=(root/path).read_bytes()
            if normalize:data=data.replace(b'\r\n',b'\n')
            if hashlib.sha256(data).hexdigest()!=metadata[key]:raise ValueError(f'Checkpoint runtime mismatch: {path}')


class StrategicPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.player=nn.Sequential(nn.Linear(PLAYER_FEATURES,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.context=nn.Sequential(nn.Linear(len(GLOBAL_NAMES)+32,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.actor=nn.Sequential(nn.Linear(128+32+ACTION_FEATURES,64),nn.ReLU(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.zeros_(self.actor[-1].bias)

    def forward(self,packet):
        players=self.player(packet['players'].float())
        context=self.context(torch.cat((packet['global'].float(),players.mean(1)),dim=-1))
        targets=packet['targets'].long()
        gathered=players.gather(1,targets.clamp_min(0).unsqueeze(-1).expand(-1,-1,32))
        gathered=gathered*(targets>=0).unsqueeze(-1)
        context_rows=context.unsqueeze(1).expand(-1,targets.shape[1],-1)
        logits=self.actor(torch.cat((context_rows,gathered,packet['actions'].float()),dim=-1)).squeeze(-1)
        mask=packet['mask'].bool()
        if not mask.any(-1).all():raise ValueError('Every state needs an explicit no-submission option')
        logits=logits.masked_fill(~mask,-1e9)
        return Categorical(logits=logits),self.critic(context).squeeze(-1)

    @torch.no_grad()
    def act(self,packet,deterministic=False):
        distribution,value=self(tensors(packet))
        action=distribution.probs.argmax(-1) if deterministic else distribution.sample()
        return int(action.item()),float(distribution.log_prob(action).item()),float(value.item())

    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'schema':SCHEMA,'global_names':GLOBAL_NAMES,'kinds':KINDS,
                    'state_dict':self.state_dict(),'metadata':metadata or {}},path)

    @classmethod
    def load(cls,path):
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('schema')!=SCHEMA or tuple(data.get('global_names',()))!=GLOBAL_NAMES or tuple(data.get('kinds',()))!=KINDS:
            raise ValueError('Incompatible strategic checkpoint')
        model=cls();model.load_state_dict(data['state_dict'],strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite checkpoint')
        validate_runtime(data.get('metadata',{}))
        model.eval();return model,data.get('metadata',{})
