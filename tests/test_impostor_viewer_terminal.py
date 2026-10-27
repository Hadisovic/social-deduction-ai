"""Controlled end-of-game display, without training or evaluation scenarios."""
import pytest
import torch
from social_deduction.actor import Role


@pytest.mark.parametrize('winner,reason',[(Role.CREWMATE,'TASKS_COMPLETED'),(Role.IMPOSTOR,'PARITY')])
def test_viewer_displays_absorbing_end_state(tmp_path,monkeypatch,winner,reason):
    from impostor_research.frozen import FrozenModels
    from impostor_research.learning_v2 import Session
    from impostor_research.strategic_v2 import StrategicEnv
    import impostor_research.frozen as frozen_module
    import impostor_research.strategic_v2 as strategic_module
    from run_impostor import main
    torch.set_num_threads(1)
    models=FrozenModels();session=Session(71,models=models,development=True)
    checkpoint=tmp_path/'preflight.pt';session.save(checkpoint,False)

    class FinishedEnv(StrategicEnv):
        def reset(self,**kwargs):
            super().reset(**kwargs)
            self.match.game._finish(winner,reason)
            self.finished=True;self._observe()
            assert self.observation_space.contains(self.packet)
            assert self.packet['actions'].shape==(256,51)
            return self.packet,{}

    monkeypatch.setattr(strategic_module,'StrategicEnv',FinishedEnv)
    monkeypatch.setattr(frozen_module,'FrozenModels',lambda:models)
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    assert main(['--checkpoint',str(checkpoint),'--max-frames','2','--capture',str(tmp_path/'capture')])==0
    assert (tmp_path/'capture/impostor.png').is_file()
    models.assert_unchanged()
