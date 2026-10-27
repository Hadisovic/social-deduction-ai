"""New randomly initialized impostor scorer; no optimizer or training entrypoint."""
import torch
from torch import nn
from torch.distributions import Categorical
from .features import ACTION_COUNT, GLOBAL_COUNT, PLAYER_COUNT


class ImpostorPolicy(nn.Module):
    def __init__(self, seed=71):
        super().__init__()
        # Construction must not perturb opponents or other experiments' RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.player = nn.Sequential(nn.Linear(PLAYER_COUNT,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
            self.context = nn.Sequential(nn.Linear(GLOBAL_COUNT+32,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
            self.actor = nn.Sequential(nn.Linear(128+32+ACTION_COUNT,64),nn.Tanh(),nn.Linear(64,1))
            self.critic = nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
            nn.init.orthogonal_(self.actor[-1].weight,.01); nn.init.zeros_(self.actor[-1].bias)

    def forward(self, packet):
        players = self.player(packet['players'].float())
        context = self.context(torch.cat((packet['global_'].float(),players.mean(1)),dim=-1))
        targets = packet['targets'].long()
        gathered = players.gather(1,targets.clamp_min(0).unsqueeze(-1).expand(-1,-1,32)) * (targets>=0).unsqueeze(-1)
        logits = self.actor(torch.cat((context.unsqueeze(1).expand(-1,targets.shape[1],-1),gathered,packet['actions'].float()),dim=-1)).squeeze(-1)
        mask = packet['mask'].bool()
        if not mask.any(-1).all(): raise ValueError('Empty legal mask')
        return Categorical(logits=logits.masked_fill(~mask,-1e9)),self.critic(context).squeeze(-1)

    @torch.inference_mode()
    def act(self, packet, *, generator=None, deterministic=True):
        data = {k:torch.as_tensor(v).unsqueeze(0) for k,v in packet.items()}
        distribution,value = self(data)
        if not deterministic and generator is None:
            raise ValueError('Stochastic impostor inference requires a private RNG')
        action = distribution.probs.argmax(-1) if deterministic else torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1)
        return int(action.item()),float(distribution.log_prob(action).item()),float(value.item())
