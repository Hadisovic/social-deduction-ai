"""Independent Pygame application. Only JSON recordings and local UI files are read."""
import argparse
from dataclasses import dataclass,field
import json
import os
from pathlib import Path
import time
import pygame

from .data import ReplayIndex,time_of
from .preferences import Preferences
from .library import Library
from .renderer import Renderer,SPEEDS
from .widgets import Widgets,font
from .shared_io import read_json,write_json

ROOT=Path(__file__).resolve().parents[2]
REGISTERED=ROOT.parent/'_v2_fourcrew_registration'
WATCH=REGISTERED/'local-runs/live-watch/fourcrew-reporting-2048-seed17-v1'
METRICS=REGISTERED/'docs/crewmate-v2-fourcrew/results/registered-2048-seed17-v1/metrics.json'


@dataclass
class State:
    prefs:object
    world:object
    library:object
    index:object=None
    selected:str|None=None
    current_path:str|None=None
    scene:str='Match'
    tab:str='Mind'
    playing:bool=False
    speed:float=1.
    mode:str='manual'
    fps:float=0.
    scrolls:dict=field(default_factory=dict)
    scroll_regions:list=field(default_factory=list)
    pins:dict=field(default_factory=dict)
    retrospective_compare:bool=False
    all_actions:bool=False
    meeting_open:bool=False
    meeting_collapsed:bool=False
    dropdown:object=None
    log_open:bool=False
    help_open:bool=False
    text_focus:str|None=None
    library_query:str=''
    event_query:str=''
    seek_text:str=''
    condition:str='All'
    context:str='All'
    tag:str='All'
    sort:str='Newest'
    settings_category:str='Appearance'
    study_tab:str='Registered study'
    study_context:str='ordinary'
    metrics:dict=field(default_factory=dict)
    status:dict=field(default_factory=dict)
    live_environment:int|None=None
    message:str=''
    message_until:float=0.
    fullscreen:bool=False
    auto_wait:bool=False
    opened_monotonic:float=field(default_factory=time.monotonic)


class Application:
    def __init__(self,preferences,world):
        # Pygame Font objects become invalid after font.quit()/reinitialization.
        font.cache_clear()
        self.state=State(preferences,world,Library(),speed=preferences['speed'],mode=preferences['mode'])
        self.renderer=Renderer(world);self.ui=Widgets();self.screen=None;self.running=True
        self.drag=None;self.scrubbing=False;self.last_mouse=None;self.saved_size=None
        self.last_follow_mtime=0.;self.capture_request=None;self.first_seek=None;self.loaded_frames=0
        self.last_live_poll=0.;self.last_live_stamp=0.

    def tell(self,message):self.state.message=str(message);self.state.message_until=time.monotonic()+6

    def open(self,path):
        self.state.live_environment=None
        self.state.library.request(path);self.tell('Loading completed replay in background…')

    def load_metrics(self,path):
        try:
            data=read_json(path)
            groups=data.get('registered_aggregate',data).get('groups',{})
            if not groups or not all(label in groups for label in ('original','shaped_v2','zero')):raise ValueError('Unsupported aggregate metrics groups')
            self.state.metrics=data;self.tell('Loaded recorded aggregate study metrics; no experiment executed.')
        except (OSError,ValueError,TypeError,AttributeError) as exc:self.tell('Metrics unavailable: '+str(exc))

    def ordered(self):return sorted(self.state.library.entries.values(),key=lambda e:(e['mtime'],e['path']))

    def neighbor(self,direction):
        rows=self.ordered();paths=[e['path'] for e in rows]
        if not paths:return None
        if self.state.current_path not in paths:return paths[0] if direction>0 else paths[-1]
        i=paths.index(self.state.current_path)+direction
        return paths[i] if 0<=i<len(paths) else None

    def poll(self):
        app=self.state;changed=app.library.poll();loaded=app.library.take_loaded()
        if loaded:
            path,data,error=loaded
            if error:self.tell('Replay not loaded: '+error)
            else:
                app.index=ReplayIndex(data);app.current_path=path;self.loaded_frames=0
                ids=list(app.index.frame().players);default=app.prefs['default_player']
                app.selected=default if default in ids else next((p for p in ids if app.index.header.get('controller_types',{}).get(p) in ('ppo','frozen','patrick-jane-ppo')),ids[0] if ids else None)
                app.playing=app.prefs['autoplay'];app.auto_wait=False;app.pins={};app.retrospective_compare=False;app.scrolls['inspector']=0;app.meeting_open=False;app.meeting_collapsed=False
                self.renderer.map.fit();app.prefs.remember(path)
                self.last_follow_mtime=Path(path).stat().st_mtime
                if self.first_seek is not None:app.index.player.seek_to_time(self.first_seek);self.first_seek=None
                if getattr(self,'initial_paused',False):app.playing=False;self.initial_paused=False
                if getattr(self,'initial_selected',None) in ids:app.selected=self.initial_selected;self.initial_selected=None
                if getattr(self,'initial_tab',None):app.tab=self.initial_tab;self.initial_tab=None
                self.tell('Recorded replay loaded. Private neural data stays selected-actor only.')
        if changed and app.mode=='follow_latest' and app.library.entries and app.live_environment is None:
            latest=max(app.library.entries.values(),key=lambda e:e['mtime'])
            if latest['mtime']>self.last_follow_mtime and latest['path']!=app.current_path:
                self.last_follow_mtime=latest['mtime'];self.open(latest['path'])
        if app.auto_wait and app.mode=='auto_next' and not app.library.loading and not app.library.requested:
            path=self.neighbor(1)
            if path:self.open(path);app.auto_wait=False
        if app.message and time.monotonic()>app.message_until:app.message=''
        if app.live_environment is not None and time.monotonic()-self.last_live_poll>=.5:
            self.last_live_poll=time.monotonic()
            streams=app.status.get('environment_progress',{}).get('streams',[])
            stream=next((s for s in streams if s['environment']==app.live_environment),None)
            if stream and stream.get('snapshot'):
                try:
                    path=Path(stream['snapshot']);data=read_json(path)
                    same_seed=data.get('header',{}).get('scenario_seed')==stream['seed']
                    expected_update=app.status.get('environment_progress',{}).get('update')
                    same_update=expected_update is None or data.get('header',{}).get('config',{}).get('update')==expected_update
                    if not same_seed or not same_update:
                        app.index=None;app.current_path=None
                    elif data.get('updated_unix',0)>self.last_live_stamp and data.get('ticks'):
                        previous=app.selected;app.index=ReplayIndex(data);app.current_path=str(path)
                        app.index.player.seek_to_time(app.index.player.duration)
                        ids=list(app.index.frame().players)
                        app.selected=previous if previous in ids else next((p for p in ids if app.index.header.get('controller_types',{}).get(p)=='patrick-jane-ppo'),ids[0])
                        app.playing=False;self.last_live_stamp=data['updated_unix']
                except (OSError,ValueError,TypeError,KeyError):pass

    def set_pref(self,key,value):
        app=self.state
        if key=='mode':app.live_environment=None  # Re-selecting follow_latest also leaves live snapshots.
        if not app.prefs.set(key,value):return
        if key=='speed':app.speed=app.prefs['speed']
        if key=='mode':
            app.mode=app.prefs['mode'];app.live_environment=None
            app.auto_wait=bool(app.index and app.index.player.at_end and app.mode=='auto_next')
        if key in ('roster','inspector') and value and self.screen:
            w=self.screen.get_width();needed=(app.prefs['left_width']+app.prefs['right_width'])*app.prefs['scale']+470
            if w<needed:app.prefs.set('inspector' if key=='roster' else 'roster',False)
        app.prefs.save()

    def dispatch(self,action,pos=None):
        if not action:return
        command,*args=action;app=self.state;index=app.index;player=index.player if index else None
        if command!='focus':app.text_focus=None;pygame.key.stop_text_input()
        if command not in ('dropdown','layers'):app.dropdown=None
        if command=='page':
            app.scene=args[0];app.playing=False;app.auto_wait=False
        elif command=='select':app.selected=args[0];app.scrolls['inspector']=0;app.pins={};app.retrospective_compare=False
        elif command=='tab':app.tab=args[0];app.scrolls['inspector']=0
        elif command=='play' and player:
            if player.at_end:player.seek_to_time(0.)
            app.playing=not app.playing;app.auto_wait=False
        elif command=='step' and player:
            (player.step_forward if args[0]>0 else player.step_backward)();app.playing=False;app.auto_wait=False
        elif command=='restart' and player:player.seek_to_time(0.);app.playing=app.prefs['autoplay'];app.auto_wait=False
        elif command in ('seek','seek_match') and player:
            player.seek_to_time(float(args[0]));app.playing=False;app.auto_wait=False
            if command=='seek_match':
                app.scene='Match';app.log_open=False
                if len(args)>1 and args[1] in index.frame().players:app.selected=args[1]
        elif command=='event' and player:
            candidates=self.renderer.visible_markers(app,include_future=True);when=player.current_sim_time
            times=[m.sim_time for m in candidates if m.sim_time>when+.01] if args[0]>0 else [m.sim_time for m in candidates if m.sim_time<when-.01]
            if times:player.seek_to_time(min(times) if args[0]>0 else max(times));app.playing=False
        elif command in ('previous','next'):
            path=self.neighbor(-1 if command=='previous' else 1)
            if path:self.open(path)
            else:self.tell('No next completed recording in the indexed queue.')
        elif command=='speed':self.set_pref('speed',args[0])
        elif command=='toggle':self.set_pref(args[0],not app.prefs[args[0]])
        elif command=='set_pref':self.set_pref(args[0],args[1])
        elif command=='load':self.open(args[0]);app.scene='Match'
        elif command=='live_environment':
            app.live_environment=int(args[0]);app.scene='Match';app.playing=False
            app.index=None;app.current_path=None;app.auto_wait=False
            self.last_live_stamp=0.
            directory=app.status.get('monitor_dir')
            if directory:
                try:
                    write_json(Path(directory)/'focus.json',dict(environment=app.live_environment))
                    self.tell('Watching real throttled snapshots · bounded history · no simulation control')
                except OSError as exc:self.tell('Live selection unavailable: '+str(exc))
        elif command=='live_menu':
            streams=app.status.get('environment_progress',{}).get('streams',[])
            app.dropdown=dict(pos=pos or pygame.mouse.get_pos(),options=[
                (f"Env {s['environment']:02d} · {s.get('phase','unknown')} · {s['simulation_seconds']:.1f}s",('live_environment',s['environment'])) for s in streams])
        elif command=='favorite':app.prefs.favorite(args[0])
        elif command=='bookmark' and player:
            app.prefs.bookmark(app.current_path,player.current_sim_time,app.selected);self.tell('Local decision/time bookmark saved.')
        elif command=='bookmarks':
            options=[(f"{b['time']:.2f}s · {Path(b['path']).name}",('load_bookmark',b)) for b in reversed(app.prefs['bookmarks'][-11:])]
            app.dropdown=dict(pos=pos or pygame.mouse.get_pos(),options=options or [('No saved local bookmarks',('notify','Save one with B or the Bookmark control.'))])
        elif command=='load_bookmark':
            b=args[0];self.first_seek=b['time'];self.initial_selected=b.get('player');self.initial_paused=True;self.open(b['path']);app.scene='Match'
        elif command=='notify':self.tell(args[0])
        elif command=='pin' and index and app.selected:
            n=index.neural(app.selected)
            if n:app.pins[args[0]]=(app.selected,n);self.tell('Pinned recorded owner decision '+args[0])
            else:self.tell('No neural decision is recorded for this actor by this tick.')
        elif command=='retrospective_compare':app.retrospective_compare=True
        elif command=='all_actions':app.all_actions=not app.all_actions
        elif command=='settings_category':app.settings_category=args[0];app.scrolls['settings']=0
        elif command=='study_tab':app.study_tab=args[0];app.scrolls['study']=0
        elif command=='study_context':app.study_context=args[0]
        elif command=='set_default_player':self.set_pref('default_player',app.selected or '')
        elif command=='reset_settings':
            app.prefs.reset();app.speed=app.prefs['speed'];app.mode=app.prefs['mode'];app.scrolls={};self.renderer.map.fit();self.tell('Viewer defaults restored. No research/replay file was changed.')
        elif command=='fit':self.renderer.map.fit();self.set_pref('follow',False)
        elif command=='focus_player' and index and app.selected in index.frame().players:
            p=index.frame().players[app.selected];self.renderer.map.center=tuple(p.ghost_position or p.position)
        elif command=='zoom':self.renderer.map.zoom_at(args[0])
        elif command=='resize':self.drag=args[0]
        elif command=='scrub':self.scrubbing=True;self.seek_mouse(pos or pygame.mouse.get_pos())
        elif command=='log':app.log_open=not app.log_open
        elif command=='help':app.help_open=not app.help_open
        elif command=='meeting' and index:
            if index.frame().phase in ('discussion','voting'):app.meeting_collapsed=not app.meeting_collapsed
            else:app.meeting_open=not app.meeting_open
        elif command=='focus':
            app.text_focus=args[0]
            if args[0]=='seek_text':app.seek_text=f'{player.current_sim_time:.2f}' if player else '0'
            pygame.key.start_text_input()
        elif command=='dropdown':
            key,options=args;commands=[]
            for option in options:
                target=('set_pref',key[5:],option) if key.startswith('pref:') else ('speed',option) if key=='speed' else ('set_pref','mode',option) if key=='mode' else ('filter',key,option)
                commands.append((str(option),target))
            app.dropdown=dict(pos=pos or pygame.mouse.get_pos(),options=commands)
        elif command=='filter':setattr(app,args[0],args[1]);app.scrolls['library']=0
        elif command=='layers':
            keys=[('Player labels','labels'),('Room labels','rooms'),('Task consoles','tasks'),('Bodies','bodies'),('Sight radius','visibility'),('Recorded trail','trails'),('Collisions','collisions'),('Reveal roles','reveal_roles'),('Omniscient mode','omniscient'),('Hidden ballots','reveal_ballots')]
            app.dropdown=dict(pos=pos or pygame.mouse.get_pos(),options=[(('ON  ' if app.prefs[k] else 'OFF  ')+label,('toggle',k)) for label,k in keys])
        elif command=='close_dropdown':app.dropdown=None
        elif command=='fullscreen':self.fullscreen()
        elif command=='capture':self.capture_request=ROOT/'local-runs/spectator-v2-captures'/('view-'+time.strftime('%Y%m%d-%H%M%S')+'.png')
        elif command in ('open_file','open_folder','open_metrics'):
            self.file_dialog(command)

    def file_dialog(self,kind):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root=tk.Tk();root.withdraw()
            if kind=='open_folder':path=filedialog.askdirectory(title='Choose a completed replay folder')
            else:path=filedialog.askopenfilename(title='Choose aggregate metrics JSON' if kind=='open_metrics' else 'Choose completed replay',filetypes=[('JSON metrics','*.json')] if kind=='open_metrics' else [('Replay','*.jsonl')])
            root.destroy()
            if not path:return
            if kind=='open_folder':self.state.library.set_directory(path);self.set_pref('watch_dir',str(path));self.tell('Indexing folder in background.')
            elif kind=='open_metrics':self.load_metrics(path)
            else:self.open(path);self.state.scene='Match'
        except Exception as exc:self.tell('File picker unavailable: '+str(exc))

    def fullscreen(self):
        if self.screen is None:return
        if not self.state.fullscreen:
            self.saved_size=self.screen.get_size();self.screen=pygame.display.set_mode(pygame.display.get_desktop_sizes()[0],pygame.FULLSCREEN)
        else:self.screen=pygame.display.set_mode(self.saved_size or (1600,960),pygame.RESIZABLE)
        self.state.fullscreen=not self.state.fullscreen

    def seek_mouse(self,pos):
        r=self.renderer.timeline_rect
        if self.state.index and r:
            fraction=max(0,min(1,(pos[0]-r.x)/r.w));self.state.index.player.seek_to_time(fraction*self.state.index.player.duration)
            self.state.playing=False;self.state.auto_wait=False

    def handle(self,event):
        app=self.state
        if event.type==pygame.QUIT:self.running=False
        elif event.type==pygame.VIDEORESIZE and not app.fullscreen:self.screen=pygame.display.set_mode((max(1000,event.w),max(700,event.h)),pygame.RESIZABLE)
        elif event.type==pygame.TEXTINPUT and app.text_focus:
            value=getattr(app,app.text_focus);addition=event.text
            if app.text_focus=='seek_text':addition=''.join(c for c in addition if c in '0123456789.')
            setattr(app,app.text_focus,(value+addition)[:160])
            app.scrolls['library' if app.text_focus=='library_query' else 'log']=0
        elif event.type==pygame.KEYDOWN:
            if app.text_focus:
                if event.key==pygame.K_BACKSPACE:setattr(app,app.text_focus,getattr(app,app.text_focus)[:-1])
                elif event.key==pygame.K_RETURN:
                    if app.text_focus=='seek_text':
                        try:self.dispatch(('seek',float(app.seek_text)))
                        except ValueError:self.tell('Enter a valid simulation time in seconds.')
                    app.text_focus=None;pygame.key.stop_text_input()
                elif event.key==pygame.K_ESCAPE:app.text_focus=None;pygame.key.stop_text_input()
                return
            key=event.key
            if key==pygame.K_TAB:self.ui.focus_next(bool(event.mod&pygame.KMOD_SHIFT))
            elif key==pygame.K_RETURN:self.dispatch(self.ui.activate())
            elif key==pygame.K_SPACE:self.dispatch(('play',))
            elif key in (pygame.K_LEFT,pygame.K_RIGHT):self.dispatch(('step',-1 if key==pygame.K_LEFT else 1))
            elif key in (pygame.K_COMMA,pygame.K_PERIOD):self.dispatch(('event',-1 if key==pygame.K_COMMA else 1))
            elif key==pygame.K_F11:self.dispatch(('fullscreen',))
            elif key==pygame.K_F12:self.dispatch(('capture',))
            elif key==pygame.K_F1:self.dispatch(('help',))
            elif pygame.K_1<=key<=pygame.K_9 and app.index:
                ids=list(app.index.frame().players);slot=key-pygame.K_1
                if slot<len(ids):self.dispatch(('select',ids[slot]))
            elif key==pygame.K_r:self.dispatch(('restart',))
            elif key==pygame.K_n:self.dispatch(('next',))
            elif key==pygame.K_p:self.dispatch(('previous',))
            elif key==pygame.K_b:self.dispatch(('bookmark',))
            elif key==pygame.K_v:self.dispatch(('meeting',))
            elif key==pygame.K_s:self.dispatch(('page','Settings'))
            elif key==pygame.K_l:self.dispatch(('page','Library'))
            elif key==pygame.K_m:
                modes=['manual','auto_next','follow_latest'];self.set_pref('mode',modes[(modes.index(app.mode)+1)%3])
            elif key in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_MINUS):
                i=min(range(len(SPEEDS)),key=lambda n:abs(SPEEDS[n]-app.speed));self.dispatch(('speed',SPEEDS[max(0,min(6,i+(-1 if key==pygame.K_MINUS else 1)))]))
            elif key==pygame.K_ESCAPE:
                if app.dropdown:app.dropdown=None
                elif app.help_open:app.help_open=False
                elif app.log_open:app.log_open=False
                elif app.scene!='Match':app.scene='Match'
                else:app.playing=False;self.tell('Playback paused. Close the window or press Q to exit.')
            elif key==pygame.K_q:self.running=False
        elif event.type==pygame.MOUSEBUTTONDOWN:
            if event.button==1:self.dispatch(self.ui.click(event.pos),event.pos)
            elif event.button==2 and self.renderer.map.rect.collidepoint(event.pos):self.drag='pan';self.last_mouse=event.pos;self.set_pref('follow',False)
            elif event.button==3 and self.state.scene=='Match':
                nearest=[(math_dist(event.pos,p),pid) for pid,p in self.renderer.map.player_points.items()]
                if nearest and min(nearest)[0]<40:
                    pid=min(nearest)[1];app.selected=pid
                    app.dropdown=dict(pos=event.pos,options=[('Focus '+pid,('focus_player',)),('Toggle camera follow',('toggle','follow')),('Bookmark this moment',('bookmark',)),('Pin owner decision A',('pin','A'))])
        elif event.type==pygame.MOUSEBUTTONUP:
            if self.drag in ('left','right'):app.prefs.save()
            self.drag=None;self.scrubbing=False
        elif event.type==pygame.MOUSEMOTION:
            if self.scrubbing:self.seek_mouse(event.pos)
            if self.drag=='pan':self.renderer.map.pan(*event.rel)
            elif self.drag=='left':app.prefs.set('left_width',event.pos[0]/app.prefs['scale']-12)
            elif self.drag=='right' and self.screen:app.prefs.set('right_width',(self.screen.get_width()-event.pos[0]-12)/app.prefs['scale'])
        elif event.type==pygame.MOUSEWHEEL:
            pos=pygame.mouse.get_pos()
            for rect,key,maximum in reversed(app.scroll_regions):
                if rect.collidepoint(pos):app.scrolls[key]=max(0,min(maximum,app.scrolls.get(key,0)-event.y*44));return
            if app.dropdown or app.log_open or app.help_open:return
            if app.scene=='Match' and self.renderer.map.rect.collidepoint(pos):self.renderer.map.zoom_at(1.15**event.y,pos)

    def advance(self,dt):
        app=self.state
        if app.index and app.playing and app.scene=='Match' and app.live_environment is None:
            if app.index.player.advance_sim_time(max(0,min(.25,dt))*app.speed):app.playing=False;app.auto_wait=app.mode=='auto_next'

    def draw(self):
        self.ui.begin(pygame.mouse.get_pos());self.renderer.draw(self.screen,self.state,self.ui)

    def close(self):self.state.prefs.save();self.state.library.close()


def math_dist(a,b):return ((a[0]-b[0])**2+(a[1]-b[1])**2)**.5


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='CrewLab Spectator V2 — read-only replay and research dashboard')
    p.add_argument('--replay',type=Path);p.add_argument('--watch',type=Path);p.add_argument('--metrics',type=Path)
    p.add_argument('--preferences',type=Path,default=ROOT/'local-runs/spectator-v2-preferences.json')
    p.add_argument('--status-file',type=Path);p.add_argument('--state-file',type=Path)
    p.add_argument('--width',type=int,default=0);p.add_argument('--height',type=int,default=0)
    p.add_argument('--speed',type=float);p.add_argument('--mode',choices=['manual','auto_next','follow_latest'])
    p.add_argument('--seek',type=float);p.add_argument('--paused',action='store_true');p.add_argument('--selected')
    p.add_argument('--live-environment',type=int,help='Follow genuine throttled snapshots of this instance')
    p.add_argument('--page',choices=['Match','Library','Study','Settings'],default='Match');p.add_argument('--tab')
    p.add_argument('--omniscient',action='store_true');p.add_argument('--reveal-ballots',action='store_true')
    p.add_argument('--fullscreen',action='store_true');p.add_argument('--capture',type=Path);p.add_argument('--max-frames',type=int)
    p.add_argument('--fps-limit',type=int,default=60)
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv);pygame.init()
    from among_us_map import get_map
    prefs=Preferences(args.preferences)
    if args.speed is not None:prefs.set('speed',args.speed)
    if args.mode:prefs.set('mode',args.mode)
    if args.omniscient:prefs.set('omniscient',True);prefs.set('reveal_roles',True)
    if args.reveal_ballots:prefs.set('reveal_ballots',True)
    desktop=pygame.display.get_desktop_sizes()[0]
    size=(args.width or min(1800,max(1000,desktop[0]-100)),args.height or min(1050,max(700,desktop[1]-140)))
    app=Application(prefs,get_map());app.screen=pygame.display.set_mode(size,pygame.RESIZABLE)
    pygame.display.set_caption('CrewLab | Spectator V2 | Research Dashboard')
    app.state.scene=args.page;app.first_seek=args.seek;app.initial_paused=args.paused;app.initial_selected=args.selected;app.initial_tab=args.tab
    if args.fullscreen:app.fullscreen()
    watch=args.watch or (Path(prefs['watch_dir']) if prefs['watch_dir'] else WATCH/'replays' if (WATCH/'replays').exists() else ROOT/'replays')
    app.state.library.set_directory(watch)
    default=WATCH/'replays/shaped_v2/ordinary-2790005.jsonl'
    replay=args.replay or (default if not args.watch and default.exists() else None)
    if replay:app.open(replay)
    app.load_metrics(args.metrics or METRICS)
    clock=pygame.time.Clock();frames=0;started=time.monotonic();last_state=last_status=0.
    try:
        while app.running:
            dt=clock.tick(max(1,args.fps_limit))/1000.;frames+=1
            for event in pygame.event.get():app.handle(event)
            app.poll();app.advance(dt);app.state.fps=clock.get_fps();app.draw()
            if app.state.index:app.loaded_frames+=1
            pygame.display.flip();now=time.monotonic()
            status_file=args.status_file or WATCH/'status.json'
            if now-last_status>=1:
                last_status=now
                try:app.state.status=read_json(status_file)
                except (OSError,ValueError):pass
                if args.live_environment is not None and app.state.status.get('monitor_dir'):
                    app.dispatch(('live_environment',args.live_environment));args.live_environment=None
            if args.state_file and now-last_state>=1:
                last_state=now;args.state_file.parent.mkdir(parents=True,exist_ok=True)
                record=dict(pid=os.getpid(),updated_unix=time.time(),scene=app.state.scene,replay=app.state.current_path,
                    selected=app.state.selected,tab=app.state.tab,sim_time=app.state.index.player.current_sim_time if app.state.index else None,
                    live_environment=app.state.live_environment,
                    display_driver=pygame.display.get_driver(),
                    playing=app.state.playing,speed=app.state.speed,fullscreen=app.state.fullscreen,window=list(app.screen.get_size()),
                    indexed=len(app.state.library.entries),indexing=app.state.library.busy,omniscient=prefs['omniscient'],reveal_ballots=prefs['reveal_ballots'],
                    controls=[dict(key=c.key,rect=list(c.rect)) for c in app.ui.controls],fps=app.state.fps)
                try:write_json(args.state_file,record)
                except PermissionError:pass  # Retry next diagnostic tick; never close the viewer for a reader lock.
            target=app.capture_request or (args.capture if app.state.index and app.loaded_frames>=3 else None)
            if target:
                target.parent.mkdir(parents=True,exist_ok=True);pygame.image.save(app.screen,str(target));app.capture_request=None
                print('Saved actual GUI frame:',target,flush=True)
                if args.capture:break
            if args.max_frames and frames>=args.max_frames:break
            if args.capture and now-started>35:raise RuntimeError('Could not load a completed replay for capture within 35 seconds')
    finally:app.close();pygame.quit()
    return 0
