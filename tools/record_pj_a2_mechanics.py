"""Explicit engineering fixture, not a natural-match evaluation or training run."""
from pathlib import Path
import json


def main():
    import torch,numpy as np
    from dataclasses import replace
    from patrick_jane.a2_config import A2Experiment
    from patrick_jane.worker import Match
    from patrick_jane.api import Action
    from patrick_jane.a2_transition import import_update102
    from patrick_jane.network import PatrickJane,collate
    from patrick_jane.actions import choose
    from social_deduction.actor import Role
    from social_deduction.phase3_api import Intent,IntentKind
    folder=Path('local-runs/a2-mechanics-replay-v1');folder.mkdir(parents=True,exist_ok=False)
    origin=Path('local-runs/imported/task-pursuit-v1-update102/run')
    # One-time verified weights; no optimizer steps, learned weights never saved.
    parent=torch.load(origin/'checkpoints/attempt-0001-update-000102.pt',map_location='cpu',weights_only=True)
    policy=PatrickJane();policy.load_state_dict(parent['state_dict']);torch.set_num_threads(1)
    exp=replace(A2Experiment(opponent_regime='historical'),technical_max_seconds=14.)
    m=Match(2920005,exp,replay=folder/'engineering-doors-vents.jsonl',metadata=dict(session='A2 ENGINEERING FIXTURE - not evaluation',checkpoint='original-update102',update=102,training_step=2813879))
    g=m.game;pid=next(p for p in g.players if g.players[p].role is Role.IMPOSTOR);point=g.vents['3']['position']
    g.players[pid].motion.spawn(point);g.players[m.crew[0]].motion.spawn(point);g._invalidate()
    class Script:
        def __init__(self):self.entered=False;self.moved=False;self.exited=False;self.closed=False
        def decide(self,obs):
            if not self.entered and obs.vent_actions:self.entered=True;return Action('vent_enter','3')
            if obs.vent_state and obs.vent_transition_ready:
                if not self.moved:self.moved=True;return Action('vent_move',obs.vent_links[0])
                return Action('vent_exit',obs.vent_state)
            if obs.door_actions and not self.closed:self.closed=True;return Action('doors','Electrical')
            return Intent(IntentKind.WAIT)
    m.opponents={pid:Script()};m.prepare();hidden={pid:torch.zeros(256) for pid in m.crew};generator=torch.Generator().manual_seed(17)
    while not g.is_terminal:
        records={}
        for owner,packet in m.packets.items():
            with torch.no_grad():out,h=policy(collate([packet],'cpu'),hidden[owner].unsqueeze(0))
            out={k:v[0] for k,v in out.items()};choice,lp,en,prob=choose(out,packet,generator=generator);hidden[owner]=h[0]
            records[owner]=dict(choice=choice,logp=float(lp),entropy=float(en),value=float(out['value']),probabilities=prob,update=102,training_step=2813879)
        m.advance(records)
    assert g.result.reason=='TRUNCATED';summary=m.summary()
    (folder/'fixture.json').write_text(json.dumps(dict(engineering_only=True,normal_complete_games=0,weights_updated=False,summary=summary),indent=2)+'\n')
    print(folder/'engineering-doors-vents.jsonl');return 0

if __name__=='__main__':raise SystemExit(main())
