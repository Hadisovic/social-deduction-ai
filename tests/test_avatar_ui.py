"""Visible sprite size and actor-only HUD regressions, including offline use."""
from dataclasses import replace

import numpy as np
import pygame
import pytest

from phase3_renderer import character_sprite, COLORS
from phase4_avatar_hud import AvatarHUD, available_actions
from spectator_display import fitted_window_size
from phase3_runner import ScriptedMatch
from social_deduction.phase3_api import Intent, IntentKind


def test_small_secondary_monitor_does_not_shrink_primary_window():
    assert fitted_window_size([(2560,1440),(800,600)]) == (1800,900)
    assert fitted_window_size([(2560,1440),(800,600)],display=1) == (720,360)
    assert fitted_window_size([(2560,1440)],scale=1.25) == (2250,1125)


@pytest.mark.parametrize('display,scale',[(1,None),(0,.1),(0,float('nan'))])
def test_bad_window_arguments_are_rejected(display,scale):
    with pytest.raises(ValueError): fitted_window_size([(1920,1080)],display,scale)


def test_offline_body_is_visible_half_suit_with_bone_and_transparency():
    sprite=character_sprite(COLORS['red'],32,body=True)
    rgb=pygame.surfarray.array3d(sprite); alpha=pygame.surfarray.array_alpha(sprite)
    # Visible body retains > half the living height, not a 7px collapsed oval.
    assert sprite.get_height()>=20
    assert sprite.get_bounding_rect().height>=18
    assert ((rgb[:,:,0]>220)&(rgb[:,:,1]>220)&(rgb[:,:,2]>190)&(alpha>200)).any()
    assert (alpha==0).any() and (alpha>200).any()


def test_hud_uses_legal_actions_and_does_not_mutate_actor_or_match():
    pygame.font.init()
    match=ScriptedMatch(15); obs=match.game.observe('p0')
    before=match.trajectory_hash
    assert available_actions(obs)==(('REPORT',False),('USE',False),('MEETING',False))
    # These immutable candidate packets test the display projection. The engine,
    # not the HUD, remains responsible for generating legitimate candidates.
    usable=replace(obs,actions=(Intent(IntentKind.REPORT,'p1'),Intent(IntentKind.INTERACT,'t0')))
    assert available_actions(usable)==(('REPORT',True),('USE',True),('MEETING',False))
    inactive=replace(usable,own=replace(usable.own,active=False))
    assert all(not enabled for _,enabled in available_actions(inactive))
    a=pygame.Surface((1800,900)); a.fill((0,0,0)); b=a.copy()
    hud=AvatarHUD(); hud.draw(a,obs); hud.draw(b,usable)
    assert np.any(pygame.surfarray.array3d(a)!=pygame.surfarray.array3d(b))
    assert match.trajectory_hash==before and match.game.observe('p0')==obs


def test_hud_renders_without_any_downloaded_art(monkeypatch):
    pygame.font.init()
    import phase4_avatar_hud
    monkeypatch.setattr(phase4_avatar_hud,'character_image',lambda *a,**kw:None)
    obs=ScriptedMatch(15).game.observe('p0')
    surface=pygame.Surface((1800,900)); surface.fill((0,0,0))
    AvatarHUD().draw(surface,obs)
    # Portrait has visible colored pixels in its reserved bottom slot.
    assert pygame.surfarray.array3d(surface.subsurface((30,760,60,64))).std()>20


@pytest.mark.parametrize('color',['red','blue','green','yellow','purple','orange','black','pink','white','brown'])
def test_installed_colors_have_four_distinct_walking_poses(color):
    from phase3_assets import character_image
    for direction in ('left','right','up','down'):
        frames=[character_image(color,64,direction,i) for i in range(4)]
        if any(frame is None for frame in frames):pytest.skip('Optional local art pack is absent')
        assert len({pygame.image.tobytes(frame,'RGBA') for frame in frames})==4


def test_meeting_banner_identifies_current_caller_and_reporter():
    from phase3_renderer import Phase3Renderer
    from phase3_engine import Body
    match=ScriptedMatch(15);game=match.game
    caller,victim=list(game.players.values())[:2]
    game._start_meeting(caller,Body(victim.player_id,victim.position,game.tick,caller.player_id))
    label=Phase3Renderer.meeting_origin(game)
    assert caller.color_name in label and victim.color_name in label and 'reported' in label
    from social_deduction.actor import Phase
    game.phase=Phase.ROAMING
    game._start_meeting(victim)
    label=Phase3Renderer.meeting_origin(game)
    assert victim.color_name in label and 'emergency' in label and 'reported' not in label


def test_render_interpolation_uses_physical_segments_and_does_not_mutate_trace():
    from run_phase5 import animation_frame
    trace=[(0.,{'p0':((0.,0.),(0.,0.))}),(.2,{'p0':((1.,0.),(5.,0.))}),(.4,{'p0':((1.,1.),(0.,5.))})]
    assert animation_frame(trace,.1)==(.1,{'p0':((.5,0.),(5.,0.))})
    time,frame=animation_frame(trace,.3)
    assert frame['p0'][0]==pytest.approx((1.,.5))
    assert trace[0][1]['p0'][0]==(0.,0.)
