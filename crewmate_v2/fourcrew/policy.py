"""Same actor architecture; separate fail-closed multi-agent checkpoint contract."""
from pathlib import Path
import hashlib
import torch
from crewmate_v2.learning.policy import V2Policy,SHAPES
from crewmate_v2.learning.provenance import runtime_lock,digest
from crewmate_v2.config import LobbySettings
from . import SCHEMA


def source_lock():
    folder=Path(__file__).parent
    return dict(base=runtime_lock(LobbySettings()),fourcrew={p.name:hashlib.sha256(
        p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sorted(folder.glob('*.py'))})


class FourCrewPolicy(V2Policy):
    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            torch.save(dict(schema=SCHEMA,shapes=SHAPES,runtime=source_lock(),metadata=metadata,
                            state_dict=self.state_dict()),stream)

    @classmethod
    def load(cls,path):
        record=torch.load(path,weights_only=True,map_location='cpu')
        if record.get('schema')!=SCHEMA or record.get('shapes')!=SHAPES or record.get('runtime')!=source_lock():
            raise ValueError('Four-crew checkpoint schema/source mismatch; no implicit historical migration')
        policy=cls();policy.load_state_dict(record['state_dict'],strict=True)
        if any(not torch.isfinite(x).all() for x in policy.parameters()):raise ValueError('Nonfinite policy')
        return policy.eval(),record['metadata']
