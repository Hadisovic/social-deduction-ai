"""Lobby page installed over the immutable native spectator shell."""
from pathlib import Path
import time

from .lobby import BUDGETS, LIMITS, ROOT, LobbyController, read_json


def draw_lobby(surface, rect, app, ui):
    import pygame
    from spectator.v2.widgets import box, text, wrap, scrollbar, TEXT, MUTED, CYAN, LIME, RED, AMBER, CARD
    c=app.lobby; d=c.draft
    maximum=max(0,640-rect.h)
    offset=max(0,min(maximum,app.scrolls.get('lobby',0)));app.scrolls['lobby']=offset
    old=surface.get_clip();surface.set_clip(rect.clip(old))
    x=rect.x+24;y=rect.y+18-offset;w=rect.w-48;col=(w-32)//2
    text(surface,'CLASSIC LOBBY',x,y,23,TEXT,True)
    text(surface,'Choose the next AI game. Recorded games keep their original rules.',x,y+36,14,MUTED,width=w)
    text(surface,'POPULATION',x,y+78,12,CYAN,True)
    text(surface,'TASKS PER CREWMATE',x+col+32,y+78,12,CYAN,True)

    def row(key,label,hint,xx,yy):
        area=pygame.Rect(xx,yy,col,57);box(surface,area,CARD)
        text(surface,label,xx+12,yy+6,16,TEXT,True,width=col-150)
        text(surface,hint,xx+12,yy+33,11,MUTED,width=col-145)
        value=getattr(d,key);lo,hi=LIMITS[key]
        ui.button(surface,'lobby-'+key+'-minus','−',pygame.Rect(area.right-133,yy+10,34,35),
            ('lobby_delta',key,-1),enabled=value>lo)
        text(surface,str(value),area.right-85,yy+13,19,LIME,True,mono=True)
        ui.button(surface,'lobby-'+key+'-plus','+',pygame.Rect(area.right-45,yy+10,34,35),
            ('lobby_delta',key,1),enabled=value<hi)

    row('players','Total players','Includes the impostors · 4–15',x,y+103)
    row('impostors','Impostors','1–3 · minimum total: 4 / 7 / 9',x,y+170)
    for i,(key,label,hint) in enumerate((('common','Common','Same common types for all crew · 0–2'),
            ('long','Long','Unique task families per owner · 0–3'),('short','Short','Unique task families per owner · 0–5'))):
        row(key,label,hint,x+col+32,y+103+i*67)
    text(surface,f'{d.players-d.impostors} CREWMATES  +  {d.impostors} IMPOSTORS',x,y+244,17,LIME,True,width=col)
    tasks=d.common+d.long+d.short
    text(surface,f'{tasks or 1} task families per crewmate',x,y+273,14,MUTED,width=col)
    if not tasks:text(surface,'All zero → engine assigns one Short task.',x,y+297,12,AMBER,width=col)

    text(surface,'LIZBON WEIGHTS',x,y+330,12,CYAN,True)
    if c.models:
        for i,m in enumerate(c.models):
            ui.button(surface,'lobby-model-'+m.key,m.label,
                pygame.Rect(x+i*(col+32),y+352,col,33),('lobby_set','model',m.key),d.model==m.key,size=13)
    else:
        text(surface,'No known checkpoint available. Restore Lizbon V1 or the migrated Classic checkpoint.',x,y+357,14,AMBER,width=w)
    text(surface,'Existing weights transferred to this lobby; performance in new settings is unvalidated.',x,y+395,12,MUTED,width=w)
    ui.button(surface,'lobby-budget',f'Wall limit: {d.minutes} min',pygame.Rect(x,y+424,170,32),('lobby_budget',),accent=CYAN,
        tooltip='Real-time safety limit. A cut match is not a win or loss.')
    ui.button(surface,'lobby-policy','Greedy choices' if d.greedy else 'Sampled choices',pygame.Rect(x+182,y+424,170,32),
        ('lobby_set','greedy',not d.greedy),d.greedy,accent=CYAN,tooltip='Sampled is the training-style policy; greedy selects its highest probability.')
    text(surface,'Frozen learned crew · engineering scenario · no training',x+368,y+433,12,MUTED,width=w-368)
    error=c.error()
    ui.button(surface,'lobby-start','Start AI game',pygame.Rect(x,y+474,166,39),('lobby_start',),
        active=not c.running and not error and bool(c.models),enabled=not c.running and not error and bool(c.models))
    ui.button(surface,'lobby-save','Save lobby',pygame.Rect(x+178,y+474,132,39),('lobby_save',),enabled=not error)
    ui.button(surface,'lobby-stop','Stop game',pygame.Rect(x+322,y+474,120,39),('lobby_stop',),enabled=c.running,accent=AMBER)
    ui.button(surface,'lobby-watch','Watch game',pygame.Rect(x+454,y+474,132,39),('lobby_watch',),enabled=c.output is not None)
    label=error or c.notice
    wrap(surface,label,pygame.Rect(x,y+530,w,48),14,RED if error else LIME if c.running else TEXT,22)
    if c.running:
        elapsed=max(0,time.monotonic()-c.started)
        text(surface,f"{elapsed/60:.1f} min elapsed  ·  sim {c.progress.get('simulation_seconds',0):.1f}s  ·  inference only",x,y+578,12,CYAN,width=w)
    path=c.run_dir or c.saved
    if path:text(surface,'Files: '+str(path),x,y+605,11,MUTED,width=w-20)
    surface.set_clip(old)
    app.scroll_regions.append((rect,'lobby',maximum));scrollbar(surface,rect,offset,maximum)


def install(source, *, storage=None, start_lobby=False):
    """Extend only local viewer processes; pinned engine and training stay untouched."""
    import pygame
    from spectator.v2 import application, renderer
    from spectator.v2.data import ReplayIndex
    from spectator.v2.widgets import PANEL, CYAN, AMBER
    if getattr(application.Application,'lizbon_lobby',False):return
    NativeApp=application.Application;NativeRenderer=application.Renderer
    lobby_capture=None

    class LobbyRenderer(NativeRenderer):
        def header(self,surface,rect,app,ui):
            super().header(surface,rect,app,ui)
            if app.scene=='Lobby':
                from spectator.v2.widgets import text, MUTED
                pygame.draw.rect(surface,PANEL,pygame.Rect(260,17,rect.w-530,47))
                text(surface,'LIZBON / CLASSIC GAME',270,20,14,CYAN,True,width=rect.w-600)
                text(surface,'Configure the next game or watch a saved recording',270,45,12,MUTED,width=rect.w-600)
            # Replace only the navigation strip. Keep the original transport,
            # inspectors, map, door art, overlays and screenshot controls.
            ui.controls[:]=[c for c in ui.controls if not (c.key.startswith(('page-','panel-')) or c.key=='help')]
            yy=rect.bottom-30
            pygame.draw.rect(surface,PANEL,pygame.Rect(20,yy,min(rect.w-290,910),31))
            xx=28
            for name in ('Match','Library','Study','Settings','Lobby'):
                ui.button(surface,'page-'+name,name,pygame.Rect(xx,yy,95,30),('page',name),app.scene==name);xx+=103
            ui.button(surface,'help','Help',pygame.Rect(xx+8,yy,58,30),('help',));xx+=78
            if app.scene=='Match':
                for key,label in (('roster','Roster'),('inspector','Inspector')):
                    if xx+92>rect.w-280:break
                    ui.button(surface,'panel-'+key,label,pygame.Rect(xx,yy,92,30),('toggle',key),app.prefs[key],accent=CYAN);xx+=100

        def draw(self,surface,app,ui):
            native=self.pages.draw
            if app.scene=='Lobby':
                self.pages.draw=lambda surf,area,state,widgets:draw_lobby(surf,area,state,widgets)
            try:super().draw(surface,app,ui)
            finally:self.pages.draw=native
            if app.scene=='Match' and getattr(app,'lobby_attached',False):
                # Persistent provenance for genuine live snapshots and completed playback.
                from spectator.v2.widgets import text
                label='LIVE AI GAME' if app.lobby.running else ('PARTIAL PLAYBACK / ' if app.lobby_partial else 'PLAYBACK / ')+app.lobby.notice
                if app.index is None:
                    label='LOADING AI GAME…' if app.lobby.running else 'NO GAME SNAPSHOT' if app.lobby_partial else 'LOADING COMPLETED REPLAY…'
                pygame.draw.rect(surface,PANEL,pygame.Rect(260,17,surface.get_width()-530,26))
                text(surface,label,270,20,13,AMBER,width=surface.get_width()-600)

    class LobbyApp(NativeApp):
        lizbon_lobby=True
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.state.lobby=LobbyController(ROOT,source,storage=storage)
            self.state.lobby_attached=False
            self.state.lobby_partial=False
            self.lobby_stamp=0.;self.lobby_polled=0.;self.lobby_replay_opened=False
            if lobby_capture:self.capture_request=lobby_capture

        def open(self,path):
            c=self.state.lobby
            if c.output is None or Path(path).resolve()!=(c.output/'match-001.jsonl').resolve():
                self.state.lobby_attached=False
            super().open(path)

        def watch_lobby(self):
            c=self.state.lobby
            if not c.output:return
            # Native Library discards a finished background load unless its
            # path still matches requested. Invalidate the previous attachment.
            self.state.library.requested=None;self.state.library.load_result=None
            self.state.library.set_directory(c.storage)
            self.state.lobby_attached=True;self.state.scene='Match'
            replay=c.output/'match-001.jsonl'
            if c.finished and replay.exists() and not c.stop_requested and (c.output/'results.json').exists():
                self.state.lobby_partial=False
                self.open_completed_game(replay)
            elif c.finished:
                self.state.lobby_partial=True
                self.state.index=None;self.state.current_path=None;self.state.live_environment=None
                if not self.load_lobby_snapshot(live=False):
                    self.tell('No snapshot was published before this game ended. See its console log.')
            else:
                self.state.lobby_partial=False
                self.state.live_environment=0;self.state.playing=False
                self.state.index=None;self.state.current_path=None;self.lobby_stamp=0

        def open_completed_game(self,replay):
            self.initial_paused=True;self.first_seek=0.
            self.state.index=None;self.state.current_path=None
            self.open(replay);self.lobby_replay_opened=True

        def load_lobby_snapshot(self,live=True):
            c=self.state.lobby;data=read_json(c.output/'live/env-00.json')
            if not data.get('ticks') or live and data.get('updated_unix',0)<=self.lobby_stamp:return False
            selected=self.state.selected
            index=ReplayIndex(data);ids=list(index.frame().players)
            if not ids:return False
            self.state.index=index;self.state.current_path=str(c.output/'live/env-00.json')
            index.player.seek_to_time(index.player.duration)
            self.state.selected=selected if selected in ids else next((p for p in ids if data['header']['agent_roles'].get(p)=='impostor'),ids[0])
            self.state.live_environment=0 if live else None;self.state.playing=False
            c.progress=dict(simulation_seconds=index.player.duration)
            self.lobby_stamp=data.get('updated_unix',0)
            return True

        def dispatch(self,action,pos=None):
            if action and action[0].startswith('lobby_'):
                command,*args=action;c=self.state.lobby;self.state.dropdown=None
                try:
                    if command=='lobby_delta':
                        key,delta=args;lo,hi=LIMITS[key]
                        setattr(c.draft,key,max(lo,min(hi,getattr(c.draft,key)+delta)))
                    elif command=='lobby_set':setattr(c.draft,args[0],args[1])
                    elif command=='lobby_budget':
                        self.state.dropdown=dict(pos=pos or pygame.mouse.get_pos(),options=[
                            (f'{n} minutes wall time',('lobby_set','minutes',n)) for n in BUDGETS])
                    elif command=='lobby_save':c.save();self.tell(c.notice)
                    elif command=='lobby_start':
                        c.start();self.lobby_replay_opened=False
                        self.state.library.set_directory(c.storage);self.state.mode='manual'
                        self.state.status={};self.watch_lobby()
                    elif command=='lobby_stop':c.stop();self.watch_lobby();self.tell(c.notice)
                    elif command=='lobby_watch':self.watch_lobby()
                except (OSError,ValueError) as exc:
                    c.notice=str(exc);self.tell(c.notice)
                return
            if action and action[0] in ('load','live_environment','previous','next','open_file','open_folder'):
                self.state.lobby_attached=False
            super().dispatch(action,pos)

        def poll(self):
            c=self.state.lobby;c.poll()
            if self.state.lobby_attached:self.state.status=dict(c.progress)
            super().poll()
            if not self.state.lobby_attached or c.output is None:return
            if c.finished:
                replay=c.output/'match-001.jsonl'
                if not self.lobby_replay_opened and not c.stop_requested and (c.output/'results.json').exists() and replay.exists():
                    self.state.lobby_partial=False
                    self.open_completed_game(replay)
                    self.tell(c.notice+' Playback paused at start.')
                elif not self.lobby_replay_opened and (c.stop_requested or (c.output/'failure.json').exists()):
                    self.state.lobby_partial=True
                    self.load_lobby_snapshot(live=False);self.lobby_replay_opened=True
                return
            if time.monotonic()-self.lobby_polled<.5:return
            self.lobby_polled=time.monotonic()
            self.load_lobby_snapshot()

        def close(self):
            try:self.state.lobby.close()
            finally:super().close()

    application.Renderer=LobbyRenderer
    application.Application=LobbyApp
    if start_lobby:
        original=application.parse_args
        def parse(argv=None):
            nonlocal lobby_capture
            args=original(argv);args.page='Lobby'
            # An empty local library prevents an unrelated historical replay opening.
            if not args.watch:args.watch=Path(storage) if storage else ROOT/'local-runs/lizbon-dashboard'
            if args.capture:
                lobby_capture=args.capture;args.capture=None;args.max_frames=1
            return args
        application.parse_args=parse
