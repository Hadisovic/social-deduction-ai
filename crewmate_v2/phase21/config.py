from dataclasses import dataclass,asdict
import math


@dataclass(frozen=True)
class Features:
    schema:str='crewmate-phase21-features-v1'
    social_guard:bool=False
    influence:float=0.
    defense_max:float=.30
    accusation_min:float=.70
    strong_min:float=.85
    vote_min:float=.70
    margin:float=.10
    influence_cap:float=.20
    protect_direct:bool=True
    speaking_order:str='original'
    witness_priority:bool=False
    witness_logit_bonus:float=4.
    witness_approach_max:float=4.5
    witness_fresh_seconds:float=3.
    opponent:str='baseline'

    def __post_init__(self):
        if self.schema!='crewmate-phase21-features-v1':raise ValueError('Unknown experimental feature schema')
        for key in ('social_guard','protect_direct','witness_priority'):
            if type(getattr(self,key)) is not bool:raise ValueError('Experimental switches must be boolean')
        for key in ('influence','defense_max','accusation_min','strong_min','vote_min','margin','influence_cap'):
            x=getattr(self,key)
            if type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('Invalid bounded experimental parameter')
        if not self.defense_max<self.accusation_min<=self.strong_min:raise ValueError('Inconsistent social thresholds')
        for key in ('witness_logit_bonus','witness_approach_max','witness_fresh_seconds'):
            x=getattr(self,key)
            if type(x) not in (int,float) or not math.isfinite(x) or not 0<x<=20:raise ValueError('Invalid witness parameter')
        if self.speaking_order not in ('original','confidence'):raise ValueError('Unknown speaker mechanism')
        if self.opponent not in ('baseline','v1','v2'):raise ValueError('Unknown opponent version')

    def to_dict(self):return asdict(self)


def modes():
    return {'unrestricted':Features(),'guard':Features(social_guard=True),
            'influence03':Features(social_guard=True,influence=.03),
            'influence10':Features(social_guard=True,influence=.10),
            'ordered10':Features(social_guard=True,influence=.10,speaking_order='confidence'),
            'witness':Features(social_guard=True,witness_priority=True)}
