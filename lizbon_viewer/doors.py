"""Original Skeld sprite animation driven only by recorded door-state edges."""
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

ASSETS=Path(__file__).resolve().parents[1]/'assets/skeld/doors'
ANIMATION_SECONDS=.25  # Reconstructed cadence; the texture contains no clip FPS.


@dataclass(frozen=True)
class Edge:
    time: float
    start: float
    target: float


class Track:
    def __init__(self,initial=False):
        self.initial=float(initial);self.edges=[];self.times=[]

    def at(self,when,reduced=False):
        i=bisect_right(self.times,when)-1
        if i<0:return self.initial
        edge=self.edges[i]
        if reduced:return edge.target
        alpha=max(0.,min(1.,(when-edge.time)/ANIMATION_SECONDS))
        return edge.start+(edge.target-edge.start)*alpha

    def change(self,when,closed):
        start=self.at(when)
        self.edges.append(Edge(when,start,float(closed)));self.times.append(when)


class DoorTimeline:
    """Deterministic seeking, including rapid reversals and cropped live history."""
    def __init__(self,states,*,selected=None,omniscient=True):
        self.tracks={};previous=set();first=True
        for state in sorted(states,key=lambda s:float(s.get('time',s.get('sim_time',0.)))):
            when=float(state.get('time',state.get('sim_time',0.)))
            if not math.isfinite(when):continue
            current=set(state.get('doors',{})) if omniscient else set(state.get('local_doors',{}).get(selected,[]))
            if first:
                # A live window may start mid-closure: do not invent a new close.
                for key in current:self.tracks[key]=Track(True)
                previous=current;first=False;continue
            for key in current.symmetric_difference(previous):
                track=self.tracks.setdefault(key,Track())
                track.change(when,key in current)
            previous=current

    def fraction(self,key,when,reduced=False):
        track=self.tracks.get(key)
        return track.at(when,reduced) if track else 0.


class DoorLayer:
    def __init__(self):
        self._index=None;self._scope=None;self.timeline=None
        self.frames=None;self.scaled=OrderedDict()

    def load(self):
        if self.frames is not None:return
        import pygame
        config=json.loads((ASSETS/'atlas.json').read_text(encoding='utf-8'))
        path=ASSETS/'original_doors.png'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:
            raise ValueError('Original door sprite checksum mismatch')
        atlas=pygame.image.load(str(path)).convert_alpha()
        self.frames={kind:[atlas.subsurface(pygame.Rect(box)).copy() for box in boxes]
                     for kind,boxes in config['frames'].items()}

    def timeline_for(self,index,omniscient,selected):
        scope=(omniscient,selected if not omniscient else None)
        if self._index is not index or self._scope!=scope:
            states=getattr(index,'a2_states',None)
            if states is None:states=[e for e in index.events if e.get('kind')=='a2_world_state']
            self.timeline=DoorTimeline(states,selected=selected,omniscient=omniscient)
            self._index=index;self._scope=scope
        return self.timeline

    def sprite(self,kind,fraction,size):
        import pygame
        self.load();frames=self.frames[kind]
        number=min(len(frames)-1,max(0,round(fraction*(len(frames)-1))))
        key=(kind,number,*size)
        if key not in self.scaled:
            self.scaled[key]=pygame.transform.smoothscale(frames[number],size)
            if len(self.scaled)>128:self.scaled.popitem(last=False)
        self.scaled.move_to_end(key)
        return self.scaled[key]

    def draw(self,surface,view,app):
        index=app.index
        if not index:return
        import pygame
        timeline=self.timeline_for(index,app.prefs['omniscient'],app.selected)
        when=float(index.player.current_sim_time)
        old=surface.get_clip();surface.set_clip(view.rect)
        try:
            for door in view.world.data.get('doors',[]):
                if door.get('geometry_supported',True) is False:continue
                fraction=timeline.fraction(door['id'],when,app.prefs['reduced_motion'])
                if fraction<=1e-8:continue  # Existing map artwork already has open jambs.
                xs,ys=zip(*door['points']);left,right=min(xs),max(xs);bottom,top=min(ys),max(ys)
                width,height=right-left,top-bottom;center=((left+right)/2,(bottom+top)/2)
                if width>height:
                    # Front-facing doors rise above the floor-level collider.
                    w=max(2,round(width*view.scale));h=max(2,round(w*93/116))
                    foot=view.point((center[0],bottom));rect=pygame.Rect(0,0,w,h);rect.midbottom=foot
                    kind='horizontal'
                else:
                    h=max(2,round(height*view.scale));w=max(2,round(max(width,height*28/81)*view.scale))
                    rect=pygame.Rect(0,0,w,h);rect.center=view.point(center);kind='vertical'
                surface.blit(self.sprite(kind,fraction,rect.size),rect)
                if app.prefs['collisions']:
                    pygame.draw.polygon(surface,(249,204,116),[view.point(p) for p in door['points']],1)
        finally:surface.set_clip(old)
