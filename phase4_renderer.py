"""Actor-only belief sidebar; deliberately accepts no game/world/controller."""
import math
import pygame
from phase3_renderer import player_color
from social_deduction.actor import BodySeen, Ejection, Meeting, PlayerSeen, Report, Vote
from social_deduction.phase3_api import StructuredClaim, WitnessedElimination


class BeliefPanel:
    def __init__(self, size=(360,900)):
        self.size = size
        self.title = pygame.font.SysFont('segoeui',22,bold=True)
        self.font = pygame.font.SysFont('segoeui',16)
        self.small = pygame.font.SysFont('segoeui',13)

    def draw(self,memory,belief,active,expanded=True,event_offset=0):
        surface = pygame.Surface(self.size)
        surface.fill((15,26,41))
        names = {i.player_id:i.display_name for i in memory.roster}
        def text(value,y,color=(221,233,242),small=False,title=False):
            font = self.title if title else self.small if small else self.font
            # Wrap real evidence labels rather than silently truncating identity/time.
            words = str(value).split(); line = ''; rows = []
            for word in words:
                if font.size((line+' '+word).strip())[0] > self.size[0]-32 and line:
                    rows.append(line); line = word
                else:
                    line = (line+' '+word).strip()
            rows.append(line)
            for line in rows:
                surface.blit(font.render(line,True,color),(16,y)); y += font.get_linesize()
            return y
        text('CREWMATE BELIEF',18,title=True)
        text(f'{names[memory.player_id]} / {memory.player_id}  •  F to change',51)
        text(f'Observed through {memory.tick*memory.dt:.1f}s | {memory.phase.value}',77,small=True)
        if not active:
            text('Observer eliminated/ejected: memory frozen',99,(243,180,108),small=True)
        else:
            text('Learned observer • all actions remain scripted',99,(136,171,195),small=True)
        y = 132
        for pid,p in belief.by_player.items():
            text(f'{names[pid]} / {pid}',y)
            percent = self.font.render(f'{p:.1%}',True,(233,244,251))
            surface.blit(percent,(self.size[0]-percent.get_width()-18,y))
            pygame.draw.rect(surface,(36,53,73),(16,y+26,self.size[0]-32,10),border_radius=4)
            pygame.draw.rect(surface,player_color(names[pid]),(16,y+26,max(2,round((self.size[0]-32)*p)),10),border_radius=4)
            y += 53
        uncertainty = belief.entropy/math.log(4)
        text(f'Entropy {belief.entropy:.3f} / {math.log(4):.3f} nats',351)
        text('High uncertainty' if uncertainty > .8 else 'Evidence favors fewer candidates',376,
             (119,213,195) if uncertainty > .8 else (243,196,120),small=True)
        pygame.draw.line(surface,(47,66,88),(16,404),(self.size[0]-16,404))
        text('LEGITIMATE MEMORY  [M]',418,title=True)
        if not expanded:
            text('M opens sightings, conflicts and evidence.',455,small=True)
            return surface
        y = 454
        for pid in memory.candidates:
            last = memory.last_seen.get(pid)
            if last:
                label = f'{names[pid]}: {last.payload.room}, {(memory.tick-last.tick)*memory.dt:.1f}s ago'
            else:
                label = f'{names[pid]}: never directly seen'
            y = text(label,y,(167,191,211),small=True)+2
        conflicts = memory.contradictions()
        hard = sum(c.strength=='direct' for c in conflicts)
        weak = sum(c.strength=='claim' for c in conflicts)
        y = text(f'Conflicts: {hard} direct / {weak} hearsay',y+8,(243,180,108),small=True)+8
        text('EVIDENCE • PgUp/PgDn history',y,small=True); y += 25
        def who(pid):
            return names.get(pid,pid or 'skip')
        timeline = sorted(memory.events,key=lambda e:e.tick)
        end = max(0,len(timeline)-event_offset)
        for e in timeline[max(0,end-8):end]:
            p = e.payload
            if type(p) is PlayerSeen:
                line = f'saw {who(p.player_id)} in {p.room}'
            elif type(p) is BodySeen:
                line = f'saw {who(p.victim_id)} body in {p.room}'
            elif type(p) is StructuredClaim:
                line = f'{who(p.speaker_id)}: {p.kind.value} {who(p.subject_id)} / {p.region} @ {p.asserted_tick*memory.dt:.1f}s'
            elif type(p) is WitnessedElimination:
                line = f'witnessed {who(p.killer_id)} eliminate {who(p.victim_id)}'
            elif type(p) is Report:
                line = f'{who(p.reporter_id)} reported {who(p.victim_id)} / {p.room}'
            elif type(p) is Vote:
                line = f'{who(p.voter_id)} voted {who(p.target_id)}'
            elif type(p) is Ejection:
                line = f'ejected {who(p.player_id)}; role not revealed'
            elif type(p) is Meeting:
                line = f'meeting: {len(p.participants)} participants'
            else:
                continue
            if y > 831:
                break
            prefix = {'direct':'SAW','public':'PUBLIC','claim':'CLAIM'}[e.provenance.value]
            color = (233,194,135) if prefix=='CLAIM' else (158,207,226)
            y = text(f'{e.tick*memory.dt:.1f} {prefix} • {line}',y,color,small=True)+5
        text('1: separate spectator truth overlay',869,(136,171,195),small=True)
        return surface
