"""Focal avatar and action availability, derived only from its actor packet.

The cards are spectator indicators, not manual-control buttons. No game, role
labels, belief thresholds, controller or world truth is accepted by this module.
"""
import pygame
from phase3_assets import character_image
from phase3_renderer import character_sprite, player_color, TEXT, MUTED, ACCENT
from social_deduction.phase3_api import IntentKind, validate_observation


def available_actions(observation):
    validate_observation(observation)
    kinds = {action.kind for action in observation.actions} if observation.own.active else set()
    return (('REPORT', IntentKind.REPORT in kinds),
            ('USE', IntentKind.INTERACT in kinds),
            ('MEETING', IntentKind.CALL_MEETING in kinds))


class AvatarHUD:
    def __init__(self, map_width=1088, height=900):
        self.map_width, self.height = map_width, height
        self.font = pygame.font.SysFont('segoeui', 16, bold=True)
        self.small = pygame.font.SysFont('segoeui', 12)

    def draw(self, surface, observation):
        states = available_actions(observation)
        own = observation.own
        color = next(i.display_name for i in observation.roster if i.player_id == own.player_id)
        y = self.height-155
        rect = pygame.Rect(18, y, self.map_width-36, 86)
        pygame.draw.rect(surface, (16,29,44), rect, border_radius=12)
        pygame.draw.rect(surface, (56,86,107), rect, 1, border_radius=12)
        # Identity portrait stays an intact suit even when the observer is out.
        # Elimination does not justify revealing whether it was killed/ejected.
        sprite = character_image(color, 58, 'right', 0)
        if sprite is None:
            sprite = character_sprite(player_color(color), 58)
        surface.blit(sprite, sprite.get_rect(midbottom=(57,y+76)))
        surface.blit(self.font.render(f'{color} / {own.player_id}',True,TEXT),(93,y+17))
        label = 'FOCAL CREWMATE' if own.active else 'OBSERVER INACTIVE'
        surface.blit(self.small.render(label,True,ACCENT if own.active else MUTED),(93,y+42))
        surface.blit(self.small.render('Action availability / spectator',True,MUTED),(93,y+60))
        left = max(330, self.map_width-535)
        for index, (label, enabled) in enumerate(states):
            card = pygame.Rect(left+index*166, y+13, 154, 60)
            pygame.draw.rect(surface,(30,65,67) if enabled else (28,40,55),card,border_radius=9)
            pygame.draw.rect(surface,ACCENT if enabled else (62,77,95),card,1,border_radius=9)
            text = self.font.render(label,True,TEXT if enabled else MUTED)
            surface.blit(text,text.get_rect(midtop=(card.centerx,card.y+9)))
            text = self.small.render('AVAILABLE' if enabled else 'UNAVAILABLE',True,ACCENT if enabled else MUTED)
            surface.blit(text,text.get_rect(midtop=(card.centerx,card.y+34)))
