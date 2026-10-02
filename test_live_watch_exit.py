"""Regression: initial render may consume close before a checkpoint is loaded."""
def test_empty_viewer_closed_during_initial_render_never_starts_episode(monkeypatch, tmp_path):
    import sys
    import pygame
    import live_watch_training as viewer
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    closed = []
    class Environment:
        screen = None
        metadata = {}
        def __init__(self, **kwargs):
            pass
        def render(self):
            self.screen = None  # Same state as a close consumed by env.render.
        def reset(self):
            raise AssertionError('No checkpoint was loaded: an episode must not start')
        def close(self):
            closed.append(True)
    monkeypatch.setattr(viewer, 'StealthGymEnv', Environment)
    monkeypatch.setattr(sys, 'argv', ['live_watch_training.py', '--checkpoint-dir', str(tmp_path)])
    viewer.main()
    assert closed == [True]
    assert not pygame.get_init()
