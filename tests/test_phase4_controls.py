"""Exercise real demo event handlers without depending on desktop input injection."""
import pytest


def test_demo_keyboard_routes_reset_step_focal_memory_truth(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy')
    import pygame
    from belief.model import DEFAULT_CHECKPOINT
    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip('Release checkpoint required for integrated demo event test')
    from phase3_renderer import Phase3Renderer
    from phase4_renderer import BeliefPanel
    from run_phase4 import main
    phases = [[], [pygame.K_SPACE,pygame.K_RIGHT], [pygame.K_f,pygame.K_m],
              [pygame.K_m,pygame.K_PAGEUP], [pygame.K_1], [pygame.K_r], [pygame.K_n], [pygame.K_ESCAPE]]
    frames = iter(phases)
    monkeypatch.setattr(pygame.event,'get',lambda:[pygame.event.Event(pygame.KEYDOWN,key=key) for key in next(frames,[pygame.K_ESCAPE])])
    renders,panels = [],[]
    def draw(self,game,options,paused,speed,fps):
        renders.append((game.seed,game.tick,paused,options.roles))
        return pygame.Surface((1440,900))
    def panel(self,memory,belief,active,expanded=True,event_offset=0):
        panels.append((memory.player_id,expanded,event_offset))
        return pygame.Surface((360,900))
    monkeypatch.setattr(Phase3Renderer,'draw',draw)
    monkeypatch.setattr(BeliefPanel,'draw',panel)
    assert main(['--max-frames','12'])==0
    assert renders[1][1:3] == (1,True)
    assert panels[0][0] != panels[2][0] and panels[2][1] is False
    assert panels[3][1] is True
    assert renders[4][3] is True and renders[0][3] is False
    assert renders[5][0]==15 and renders[6][0]==16
