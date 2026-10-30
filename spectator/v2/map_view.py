"""Camera/presentation only; exact pinned Skeld artwork and original sprites."""
from bisect import bisect_right
import math
import pygame
from among_us_map import ASSET_DIR
from phase3_renderer import player_color
from spectator.ui_renderer import draw_character
from .widgets import BG,LINE,TEXT,MUTED,LIME,CYAN,AMBER,RED,text,progress


class MapView:
    def __init__(self,world):
        self.world=world;self.source=pygame.image.load(str(ASSET_DIR/'reference_map.png'))
        l,b,r,t=world.bounds;self.home=((l+r)/2,(b+t)/2);self.center=self.home
        self.zoom=1.;self.rect=pygame.Rect(0,0,1,1);self.scale=1.;self._art_key=None;self.art=None;self.player_points={}

    def fit(self):self.zoom=1.;self.center=self.home

    def transform(self,rect):
        self.rect=rect;self.scale=max(.1,min((rect.w-30)/self.world.width,(rect.h-28)/self.world.height)*self.zoom)

    def point(self,p):return (round(self.rect.centerx+(p[0]-self.center[0])*self.scale),round(self.rect.centery-(p[1]-self.center[1])*self.scale))

    def unproject(self,p):return (self.center[0]+(p[0]-self.rect.centerx)/self.scale,self.center[1]-(p[1]-self.rect.centery)/self.scale)

    def zoom_at(self,factor,point=None):
        point=point or self.rect.center;before=self.unproject(point)
        self.zoom=max(.6,min(3.5,self.zoom*factor));self.transform(self.rect);after=self.unproject(point)
        self.center=(self.center[0]+before[0]-after[0],self.center[1]+before[1]-after[1])

    def pan(self,dx,dy):self.center=(self.center[0]-dx/self.scale,self.center[1]+dy/self.scale)

    def draw(self,surface,rect,app,ui):
        self.transform(rect);old=surface.get_clip();surface.set_clip(rect)
        pygame.draw.rect(surface,(8,13,20),rect)
        index=app.index;frame=index.frame() if index else None;prefs=app.prefs
        if frame and prefs['follow'] and app.selected in frame.players:
            p=frame.players[app.selected];self.center=tuple(p.ghost_position or p.position)
        dims=(max(1,round(self.world.width*self.scale)),max(1,round(self.world.height*self.scale)))
        if self._art_key!=dims:self.art=pygame.transform.smoothscale(self.source,dims);self._art_key=dims
        left,_,_,top=self.world.bounds;surface.blit(self.art,self.point((left,top)))
        a2=index.a2_state() if index else None
        if a2:
            shown=set(a2.get('doors',{})) if prefs['omniscient'] else set(a2.get('local_doors',{}).get(app.selected,[]))
            for door in self.world.data.get('doors',[]):
                if door['id'] in shown:
                    pygame.draw.polygon(surface,RED,[self.point(p) for p in door['points']])
                    if prefs['omniscient']:
                        center=tuple(sum(p[i] for p in door['points'])/len(door['points']) for i in (0,1));pt=self.point(center)
                        text(surface,f"{max(0,a2['doors'].get(door['id'],0)-index.player.current_sim_time):.1f}s",pt[0],pt[1],10,AMBER)
        if prefs['collisions']:
            for wall in self.world.data.get('walls',[]):
                pygame.draw.lines(surface,AMBER,False,[self.point(p) for p in wall['points']],1)
        if prefs['tasks']:
            for d in self.world.destinations:
                if d.category=='task':pygame.draw.circle(surface,(236,203,117),self.point(d.position),3,1)
        if prefs['rooms']:
            from among_us_map import ROOMS
            for name,region in self.world.regions:
                if name in ROOMS:
                    p=region.intersection(self.world.center_domain).representative_point()
                    x,y=self.point(p.coords[0]);text(surface,name,x,y,12,MUTED)
        self.player_points={}
        if not frame:
            text(surface,'Open a genuine recording to begin',rect.x+32,rect.y+34,23,TEXT,True)
            text(surface,'The map and character artwork are preserved.',rect.x+32,rect.y+72,16,MUTED)
            surface.set_clip(old);return
        if prefs['trails'] and app.selected:
            stop=bisect_right(index.tick_times,frame.sim_time);start=max(0,stop-180)
            points=[]
            for snapshot in index.ticks[start:stop:2]:
                raw=snapshot.get('players',{}).get(app.selected,{})
                if raw and raw.get('status')=='alive':points.append(self.point(raw['position']))
            if len(points)>1:pygame.draw.lines(surface,CYAN,False,points,2)
            if index.config.get('patrick_jane_schema'):
                routes=[e for e in index.events if e.get('kind')=='patrick_jane_route' and e.get('player_id')==app.selected and e.get('time',0)<=frame.sim_time]
                if routes:
                    route=[self.point(p) for p in routes[-1].get('route',[])]
                    if len(route)>1:pygame.draw.lines(surface,AMBER,False,route,1)
        if prefs['bodies']:
            for body in frame.bodies.values():
                if body.reported:continue
                color=frame.players[body.victim_id].color if body.victim_id in frame.players else 'red'
                draw_character(surface,color,self.point(body.position),max(24,round(self.scale*.9)),body=True)
        for pid,p in sorted(frame.players.items(),key=lambda item:-item[1].position[1]):
            if a2 and pid in a2.get('vented',{}):
                if prefs['omniscient']:
                    key=a2['vented'][pid];point=self.point(a2['vent_positions'][key]);pygame.draw.circle(surface,AMBER,point,8,2);text(surface,pid+' in vent',point[0]+10,point[1],11,AMBER)
                continue
            if a2 and frame.phase=='roaming' and not prefs['omniscient'] and app.selected and pid!=app.selected and p.status=='alive' and pid not in a2.get('visible',{}).get(app.selected,[]):continue
            ghost=p.status!='alive' and p.ghost_position is not None
            if p.status!='alive' and not ghost:continue
            position=tuple(p.ghost_position if ghost else p.position)
            i=index.player._current_tick_index
            if prefs['interpolate'] and not prefs['reduced_motion'] and frame.phase=='roaming' and i+1<len(index.ticks):
                nxt=index.ticks[i+1];n=nxt.get('players',{}).get(pid,{})
                span=index.tick_times[i+1]-frame.sim_time
                if span>0 and nxt.get('phase')=='roaming' and n.get('status')==p.status:
                    target=n.get('ghost_position') if ghost else n.get('position')
                    if target:
                        alpha=max(0.,min(1.,(index.player.current_sim_time-frame.sim_time)/span))
                        # Never animate a long discontinuity or a segment outside navigable space.
                        if math.dist(position,target)<=1.:
                            from shapely.geometry import LineString
                            if self.world.center_domain.covers(LineString([position,target])):
                                position=(position[0]+(target[0]-position[0])*alpha,position[1]+(target[1]-position[1])*alpha)
            point=self.point(position);self.player_points[pid]=point;color=player_color(p.color)
            if prefs['visibility']:
                role=index.role(pid);key='impostor_sight_range' if role=='impostor' else 'crewmate_sight_range'
                radius=index.config.get(key)
                if isinstance(radius,(int,float)) and pid==app.selected:pygame.draw.circle(surface,color,point,max(1,round(radius*self.scale)),1)
            moving=math.hypot(*p.velocity)>1e-4 and frame.phase=='roaming'
            vx,vy=p.velocity;direction=('up' if vy>0 else 'down') if abs(vy)>abs(vx) else ('left' if vx<0 else 'right')
            anim=int(index.player.current_sim_time*8)%4 if moving and not prefs['reduced_motion'] else 0
            height=max(25,min(62,round(self.scale*.98)))
            if pid==app.selected:
                radius=max(17,round(self.scale*.5));pygame.draw.circle(surface,LIME,point,radius,2)
                pygame.draw.circle(surface,(89,122,67),point,radius+5,1)
            draw_character(surface,p.color,point,height,direction,anim,alpha=90 if ghost else 255)
            if p.interaction_task and isinstance(p.task_progress,(int,float)):
                progress(surface,pygame.Rect(point[0]-18,point[1]+8,36,4),p.task_progress,CYAN)
            hit=pygame.Rect(point[0]-20,point[1]-height,40,height+15)
            ui.hit('map-player-'+pid,hit,('select',pid),'Select '+pid,surface=surface)
            if prefs['labels']:
                text(surface,pid+(' · ghost' if ghost else ''),point[0]+12,point[1]-height+3,13,TEXT)
            if prefs['omniscient'] and prefs['reveal_roles'] and index.role(pid):
                pygame.draw.circle(surface,RED if index.role(pid)=='impostor' else CYAN,(point[0],point[1]-height-7),4)
        if frame.phase in ('discussion','voting'):
            label='MEETING · '+frame.phase.upper()+' · movement frozen'
            text(surface,label,rect.x+18,rect.y+14,14,AMBER,True,width=rect.w-40)
        if frame.result:
            from spectator.outcomes import outcome_label
            label=outcome_label(frame.result)
            pill=pygame.Rect(rect.centerx-180,rect.bottom-55,360,38)
            pygame.draw.rect(surface,(19,38,32),pill,border_radius=8)
            color=LIME if frame.result.get('winner')=='crewmate' else RED if frame.result.get('winner')=='impostor' else AMBER
            text(surface,label,pill.x+15,pill.y+9,16,color,True,width=pill.w-25)
        surface.set_clip(old)
