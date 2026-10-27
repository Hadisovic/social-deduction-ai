"""New reward profile. Historical focal-agent profiles remain unchanged."""
from dataclasses import dataclass,asdict
import math
from crewmate_v2.learning.config import RewardConfig


@dataclass(frozen=True)
class Rewards(RewardConfig):
    schema:str='crewmate-fourcrew-rewards-v2'
    report:float=1.5

    def __post_init__(self):
        if self.schema!='crewmate-fourcrew-rewards-v2':raise ValueError('Unknown four-crew reward schema')
        for name,value in asdict(self).items():
            if name!='schema' and (type(value) not in (int,float) or not math.isfinite(value)):
                raise ValueError('Nonfinite reward coefficient')
        if self.ghost_task!=0:raise ValueError('Automatic ghost work cannot earn individual reward')

    @classmethod
    def original(cls):return cls(report=0.,correct_vote=0.,incorrect_vote=0.)


PARTITIONS={'engineering':(2760000,2760099),'registered_train':(2780000,2780031),
            'registered_validation':(2790000,2790007)}


def check_seed(seed,partition='engineering'):
    if partition not in PARTITIONS:raise ValueError('Unknown development partition')
    lo,hi=PARTITIONS[partition]
    if type(seed) is not int or not lo<=seed<=hi:raise ValueError('Seed outside explicit development partition')
