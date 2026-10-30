"""Presentation regressions: replay seeking and visibility, never model training."""
import os
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from lizbon_viewer.doors import DoorTimeline,DoorLayer,ANIMATION_SECONDS,ASSETS


def state(t,closed=(),local=()):
    return dict(time=t,doors={p:99 for p in closed},local_doors={'a':list(local)})


def test_close_hold_open_and_arbitrary_seek_are_simulation_time_based():
    timeline=DoorTimeline([state(0),state(2,['d']),state(12)])
    for when,expected in ((1,0),(2,0),(2.125,.5),(3,1),(12,1),(12.125,.5),(13,0),(2.125,.5)):
        assert timeline.fraction('d',when)==pytest.approx(expected)
    assert timeline.fraction('missing',3)==0


def test_pending_door_request_does_not_animate_until_actual_state_changes():
    # A command at t1 is absent from physical state until the doorway clears at t3.
    timeline=DoorTimeline([state(0),state(1),state(2),state(3,['d'])])
    assert timeline.fraction('d',2.99)==0
    assert timeline.fraction('d',3.125)==pytest.approx(.5)


def test_selected_view_does_not_reveal_unseen_global_closures_or_deadlines():
    states=[state(0),state(1,['d']),state(2,['d'],['d']),state(3,(),['d']),state(4)]
    local=DoorTimeline(states,omniscient=False,selected='a')
    assert local.fraction('d',1.8)==0
    assert local.fraction('d',3.8)==1  # Retain only the actor's recorded knowledge.
    assert local.fraction('d',4.125)==pytest.approx(.5)
    assert DoorTimeline(states,omniscient=False,selected='other').fraction('d',3)==0


def test_mid_closure_live_window_is_closed_without_invented_transition():
    timeline=DoorTimeline([state(50,['d']),state(51,['d']),state(52)])
    assert timeline.fraction('d',50)==1
    assert timeline.fraction('d',52.125)==pytest.approx(.5)


def test_reduced_motion_and_rapid_reversal():
    timeline=DoorTimeline([state(0),state(1,['d']),state(1.1)])
    assert timeline.fraction('d',1.1)==pytest.approx(.4)
    assert timeline.fraction('d',1.225)==pytest.approx(.2)
    assert timeline.fraction('d',1.125,True)==0
    assert timeline.fraction('d',1.05,True)==1


def test_sprite_windows_are_original_and_closing_fills_both_orientations():
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    import pygame
    pygame.init();pygame.display.set_mode((64,64))
    try:
        layer=DoorLayer();layer.load()
        for kind,frames in layer.frames.items():
            counts=[pygame.mask.from_surface(frame).count() for frame in frames]
            assert counts[0]<counts[-1]
            assert counts==sorted(counts)
            assert counts[-1]>frames[-1].get_width()*frames[-1].get_height()*.8
        for n in range(160):layer.sprite('horizontal',1,(20+n,30))
        assert len(layer.scaled)==128
    finally:pygame.quit()


def test_native_adapter_hides_only_debug_door_bars_without_mutating_state():
    from lizbon_viewer.native import _IndexView
    source=state(2,['d'],['d']);source['visible']={'a':['b']};source['vented']={'c':'0'}
    painted=[];index=SimpleNamespace(a2_state=lambda:source)
    view=_IndexView(index,lambda:painted.append(True));result=view.a2_state()
    assert painted==[True] and result['doors']=={} and result['local_doors']=={}
    assert source['doors']=={'d':99} and source['local_doors']=={'a':['d']}
    assert result['visible']==source['visible'] and result['vented']==source['vented']
