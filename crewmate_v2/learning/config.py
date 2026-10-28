"""Versioned experimental hypotheses and a small owner-gated pilot."""
from dataclasses import asdict,dataclass
import json
import math
from pathlib import Path
from . import SCHEMA


@dataclass(frozen=True)
class RewardConfig:
    schema:str='crewmate-v2-rewards-v1'
    alive_task:float=1.
    report:float=1.25
    correct_vote:float=2.
    incorrect_vote:float=-1.
    skip:float=0.
    ghost_task:float=0.
    death:float=-5.
    win:float=10.
    loss:float=-10.
    draw:float=-5.
    time_per_decision:float=-.005

    def __post_init__(self):
        if self.schema!='crewmate-v2-rewards-v1':raise ValueError('Reward schema mismatch')
        for name,value in asdict(self).items():
            if name!='schema' and (type(value) not in (int,float) or not math.isfinite(value)):
                raise ValueError('Reward coefficients must be finite numbers')

    @classmethod
    def legacy(cls):
        return cls(report=0.,correct_vote=0.,incorrect_vote=0.,ghost_task=1.)

    def to_dict(self):return asdict(self)

    @classmethod
    def from_file(cls,path):return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


PARTITIONS={'train':(2700000,2709999),'validation':(2710000,2710003),
            'smoke':(2720000,2720003),'fixtures':(2730000,2730099)}


def match_seed(partition,index):
    if partition not in PARTITIONS or type(index) is not int or index<0:raise ValueError('Unknown development partition/index')
    lo,hi=PARTITIONS[partition]
    if lo+index>hi:raise ValueError('Development seed partition exhausted')
    return lo+index


def pilot_plan():
    return dict(schema=SCHEMA,initializations=[17],replication_after_review=[29],arms=['original','shaped'],decisions=512,
        architecture=dict(global_features=42,candidate_shape=[4,29],task_shape=[4,8],action_shape=[96,23],
            candidate_encoder=[29,32,32],task_encoder=[8,16,16],context=[90,128,128],actor=[199,64,1],critic=[128,64,1]),
        rollout=128,learning_rate=.0003,entropy=.02,gamma=.99,gae_lambda=.95,
        ppo_epochs=4,minibatch=64,clip=.2,value_coefficient=.5,max_gradient_norm=.5,
        checkpoint_decisions=[128,256,384,512],selection='fixed final budget 512; no best-of-validation selection',
        deduction_frequency=1.,coalesce_forced_wait=True,opponents='varied',partitions=PARTITIONS,
        validation=dict(seeds=4,deduction_report_rotations=[0,1,2],ordinary_report_rotations=[0],
            original_opponent_controls='deduction only',games_per_model=20),
        reward_original=RewardConfig.legacy().to_dict(),reward_shaped=RewardConfig().to_dict(),
        total_wall_ceiling_seconds=10800,smoke_steps_per_arm=32,smoke_wall_ceiling_seconds=600)
