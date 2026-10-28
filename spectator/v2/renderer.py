"""Map-centered research shell and clickable transport/meeting/event surfaces."""
import math
import time
import pygame
from phase3_renderer import player_color
from spectator.ui_renderer import draw_character
from .data import category,time_of,event_text
from .map_view import MapView
from .inspectors import Inspector
from .pages import Pages
from .widgets import *

SPEEDS=[.25,.5,1.,2.,4.,8.,16.]
EVENTS=['All','Kills','Reports','Meetings','Votes','Claims','Tasks','Rewards','Result']


class Renderer:
    def __init__(self,world):self.map=MapView(world);self.inspector=Inspector();self.pages=Pages();self.layout={};self.timeline_rect=None

    def compute_layout(self,size,prefs):
        w,h=size;s=prefs['scale'];top=round(106*s);bottom=round(158*s)
        left=round(prefs['left_width']*s) if prefs['roster'] else 0
        right=round(prefs['right_width']*s) if prefs['inspector'] else 0
        if w-left-right-48<420:left=0
        if w-right-36<380:right=0
        body=pygame.Rect(12,top,w-24,max(180,h-top-bottom-12))
        center=pygame.Rect(body.x+(left+10 if left else 0),body.y,body.w-(left+10 if left else 0)-(right+10 if right else 0),body.h)
        self.layout=dict(header=pygame.Rect(0,0,w,top-10),body=body,left=pygame.Rect(body.x,body.y,left,body.h),
            right=pygame.Rect(body.right-right,body.y,right,body.h),center=center,bottom=pygame.Rect(0,h-bottom,w,bottom))
        return self.layout

    def draw(self,surface,app,ui):
        set_scale(app.prefs['scale']);surface.fill((6,12,27) if app.prefs['theme']=='midnight' else BG)
        lay=self.compute_layout(surface.get_size(),app.prefs);app.scroll_regions=[]
        self.header(surface,lay['header'],app,ui)
        if app.scene=='Match':
            center=lay['center'];box(surface,center,PANEL,LINE,12)
            view=pygame.Rect(center.x+1,center.y+45,center.w-2,center.h-46)
            meeting=app.index.meeting(app.prefs['omniscient'],app.prefs['reveal_ballots']) if app.index else None
            show_meeting=bool(meeting and ((app.index.frame().phase in ('discussion','voting') and not app.meeting_collapsed) or app.meeting_open))
            if show_meeting and center.h>=390:view.h-=190
            self.map.draw(surface,view,app,ui);self.map_toolbar(surface,center,app,ui)
            if show_meeting and center.h>=390:self.meeting(surface,pygame.Rect(center.x+1,view.bottom,center.w-2,190),meeting,app,ui)
            if lay['left'].w:box(surface,lay['left']);self.roster(surface,lay['left'],app,ui)
            if lay['right'].w:box(surface,lay['right']);self.inspector.draw(surface,lay['right'],app,ui)
            if lay['left'].w:ui.hit('resize-left',pygame.Rect(lay['left'].right+1,lay['body'].y,8,lay['body'].h),('resize','left'),'Drag to resize roster')
            if lay['right'].w:ui.hit('resize-right',pygame.Rect(lay['right'].x-9,lay['body'].y,8,lay['body'].h),('resize','right'),'Drag to resize inspector')
        else:
            box(surface,lay['body']);self.pages.draw(surface,lay['body'],app,ui)
        self.transport(surface,lay['bottom'],app,ui)
        if app.message:
            r=pygame.Rect(24,surface.get_height()-lay['bottom'].h-47,min(surface.get_width()-48,850),35)
            box(surface,r,(36,45,58),CYAN,6);text(surface,app.message,r.x+12,r.y+8,13,TEXT,width=r.w-24)
        if app.log_open:self.event_log(surface,app,ui)
        if app.help_open:self.help(surface,app,ui)
        if app.dropdown:self.dropdown(surface,app,ui)
        ui.tooltip(surface)

    def header(self,surface,r,app,ui):
        box(surface,pygame.Rect(12,10,r.w-24,r.h-4),PANEL,LINE,11)
        pygame.draw.circle(surface,LIME,(33,33),5)
        text(surface,'CREWLAB',48,18,21,TEXT,True)
        text(surface,'SPECTATOR V2 / RESEARCH',48,47,11,MUTED)
        index=app.index
        if index:
            config=index.config;label=f"{config.get('training_condition','Historical').upper()}   /   {config.get('context','REPLAY').replace('_',' ').upper()}   /   SEED {index.header.get('scenario_seed','—')}"
            text(surface,label,265,20,14,CYAN,True,width=r.w-600)
            phase=index.frame().phase.upper();warning=f' · {len(index.issues)} telemetry warnings' if index.issues else ''
            text(surface,f"{phase}   ·   t {index.player.current_sim_time:07.2f}s   ·   tick {index.frame().tick}"+warning,265,46,13,AMBER if warning else MUTED,width=r.w-600,mono=True)
        else:
            text(surface,'A read-only instrument for AI research',265,23,16,MUTED,width=r.w-550)
        ui.button(surface,'screenshot','Capture',pygame.Rect(r.w-238,20,92,32),('capture',),tooltip='F12: save this actual interface locally')
        ui.button(surface,'fullscreen','Fullscreen',pygame.Rect(r.w-136,20,110,32),('fullscreen',),tooltip='F11 toggles fullscreen')
        x=28;y=r.bottom-30
        for name in ('Match','Library','Study','Settings'):
            ui.button(surface,'page-'+name,name,pygame.Rect(x,y,95,30),('page',name),app.scene==name);x+=103
        if app.scene=='Match':
            for key,label in (('roster','Roster'),('inspector','Inspector')):
                ui.button(surface,'panel-'+key,label,pygame.Rect(x+16,y,92,30),('toggle',key),app.prefs[key],accent=CYAN);x+=100
        ui.button(surface,'help','Help',pygame.Rect(x+18,y,58,30),('help',),tooltip='F1: keyboard controls and data provenance')
        mode='OMNISCIENT' if app.prefs['omniscient'] else 'CHRONOLOGICAL'
        text(surface,mode,r.w-260,r.bottom-25,11,AMBER if app.prefs['omniscient'] else MUTED,width=200)
        if app.status:
            d=app.status.get('decisions',{});v=app.status.get('validation_games');target=app.status.get('validation_target')
            label=('A '+str(d.get('original','—'))+' / B '+str(d.get('shaped_v2','—'))) if isinstance(d,dict) else 'Decisions '+str(d)
            if v is not None:label+=' · validation '+str(v)+('/'+str(target) if target is not None else '')
            text(surface,label,r.w-390,51,10,MUTED,width=360)

    def map_toolbar(self,surface,r,app,ui):
        text(surface,'THE SKELD',r.x+16,r.y+13,13,MUTED,True,width=105)
        x=r.right-480
        if r.w<650:x=r.x+126
        for label,key,action,width in [('−','zoom-out',('zoom',.8),30),('+','zoom-in',('zoom',1.25),30),('Fit','fit',('fit',),42),('Focus','focus',('focus_player',),58),('Follow','follow',('toggle','follow'),64),('Layers','layers',('layers',),64),('Meeting','meeting',('meeting',),79)]:
            if x+width>r.right-8:break
            ui.button(surface,'map-'+key,label,pygame.Rect(x,r.y+7,width,30),action,app.prefs['follow'] if key=='follow' else False,accent=CYAN,tooltip='Mouse wheel zooms; middle-drag pans' if key.startswith('zoom') else label);x+=width+6

    def roster(self,surface,r,app,ui):
        index=app.index
        text(surface,'PLAYERS',r.x+17,r.y+14,13,MUTED,True)
        if not index:return
        players=index.frame().players;task=index.team_tasks()
        text(surface,f"{sum(p.status=='alive' for p in players.values())} alive / {len(players)} total",r.x+17,r.y+40,13,TEXT)
        if task:
            text(surface,f"TEAM TASKS   {task['completed']}/{task['total']}",r.x+17,r.y+66,12,LIME,True)
            progress(surface,pygame.Rect(r.x+17,r.y+90,r.w-34,6),task['fraction'])
        else:text(surface,'Team task totals not recorded',r.x+17,r.y+67,12,AMBER,width=r.w-30)
        body=pygame.Rect(r.x+8,r.y+113,r.w-16,r.h-121);old=surface.get_clip();surface.set_clip(body)
        rh=113 if app.prefs['density']=='compact' else 126;rh=round(rh*app.prefs['scale'])
        for i,(pid,p) in enumerate(players.items()):
            card=pygame.Rect(body.x,body.y+i*rh-app.scrolls.get('roster',0),body.w,rh-7)
            if not card.colliderect(body):continue
            selected=app.selected==pid;box(surface,card,(29,43,42) if selected else CARD,LIME if selected else LINE,8)
            ui.hit('roster-'+pid,card,('select',pid),'Select '+pid,surface=surface)
            draw_character(surface,p.color,(card.x+22,card.y+38),28,alpha=100 if p.status!='alive' else 255)
            text(surface,pid.upper(),card.x+43,card.y+9,17,TEXT,True)
            role=index.role(pid) if app.prefs['omniscient'] and app.prefs['reveal_roles'] else 'hidden role'
            role_label='IMP' if role=='impostor' else 'CREW' if role=='crewmate' else 'HIDDEN'
            text(surface,role_label+' / '+p.status.upper(),card.x+43,card.y+31,11,RED if role=='impostor' else MUTED,width=card.w-50)
            controller=index.header.get('controller_types',{}).get(pid,'Unknown')
            if index.config.get('training_condition')=='zero' and controller=='ppo':controller='FROZEN'
            text(surface,controller.upper()+' · '+str(app.world.region(p.position)),card.x+12,card.y+51,12,MUTED,width=card.w-24)
            tp=index.task_progress(pid)
            if tp:
                progress(surface,pygame.Rect(card.x+12,card.y+73,card.w-70,5),tp['fraction'],player_color(p.color))
                text(surface,f"{tp['completed']}/{tp['total']}",card.right-49,card.y+63,12,TEXT,mono=True)
            else:text(surface,'Owned tasks: not recorded',card.x+12,card.y+69,11,DIM,width=card.w-22)
            reward=index.reward(pid);value=reward['latest'].get('cumulative_reward') if reward else None
            text(surface,f"{p.intention}"+(f'   R {value:+.1f}' if isinstance(value,(int,float)) else ''),card.x+12,card.y+86,11,MUTED,width=card.w-22)
            if rh>=120:
                stats=index.actor_stats(pid)
                text(surface,f"K {stats['kills']}   REPORT {stats['reports']}   VOTE {stats['ballots']}",card.x+12,card.y+104,10,MUTED,width=card.w-22)
        surface.set_clip(old);maximum=max(0,len(players)*rh-body.h);app.scroll_regions.append((body,'roster',maximum));scrollbar(surface,body,app.scrolls.get('roster',0),maximum)

    def meeting(self,surface,r,m,app,ui):
        box(surface,r,(20,29,40),LINE,8)
        reporter=m['report'].get('reporter_id') if m['report'] else 'caller not recorded'
        heading=f"MEETING {m['id']}  /  reporter {reporter}"
        if app.index.frame().phase not in ('discussion','voting'):heading+=' / HISTORICAL'
        text(surface,heading,r.x+16,r.y+11,14,AMBER,True,width=r.w-180)
        if m['remaining'] is not None:text(surface,f"max {m['remaining']:.1f}s",r.right-125,r.y+11,13,MUTED,mono=True)
        left=r.w*.48;y=r.y+44
        for ballot in m['ballots']:
            state=ballot['state'];label={'missing':'No ballot','hidden':'Submitted · hidden','skip':'SKIP','published_skip':'Published skip · submission unrecorded','not_recorded':'Submitted · target not recorded','vote':'→ '+str(ballot['target'])}[state]
            text(surface,ballot['voter'],r.x+18,y,14,TEXT,True)
            text(surface,label+(' [debug]' if ballot['debug'] else ''),r.x+65,y,13,PURPLE if ballot['debug'] else MUTED,width=int(left)-75)
            y+=27
        text(surface,'CURRENT-MEETING CLAIMS',r.x+left,r.y+43,11,CYAN,True,width=int(r.w-left)-14)
        y=r.y+68
        for e in m['claims'][-3:]:
            ui.button(surface,'meeting-claim-'+str(id(e)),f'{time_of(e):.1f}s · '+event_text(e),pygame.Rect(r.x+left,y,r.w-left-15,29),('seek',time_of(e)),tooltip='Jump to this recorded claim');y+=34
        if not m['claims']:text(surface,'No recorded claims yet',r.x+left,y,13,MUTED,width=int(r.w-left)-14)
        ejection=next((e for e in reversed(m['events']) if e.get('kind')=='ejection' and e.get('channel')=='public'),None)
        if ejection:
            role=ejection.get('confirmed_role',ejection.get('role'))
            summary='Published result: '+str(ejection.get('player_id',''))+' ejected. '+('Confirmed role: '+str(role) if role else 'Confirmed role not recorded.')
        else:summary='Eligibility is evaluated at this meeting’s start. Hidden targets require both debug switches.'
        text(surface,summary,r.x+16,r.bottom-23,11,DIM,width=r.w-30)

    def visible_markers(self,app,include_future=False):
        if not app.index:return []
        when=app.index.frame().sim_time;group=app.prefs['event_filter'];pid=app.prefs['event_player']
        return [m for m in app.index.markers if (include_future or app.prefs['timeline_all'] or m.sim_time<=when+1e-8)
            and (group=='All' or category(m.kind)==group) and (pid=='All' or m.player_id==pid)
            and (group!='All' or m.kind!='reward' or getattr(m,'meaningful',False))
            and (app.prefs['omniscient'] or getattr(m,'channel','public')!='truth')]

    def transport(self,surface,r,app,ui):
        pygame.draw.rect(surface,PANEL,r);pygame.draw.line(surface,LINE,r.topleft,r.topright)
        x=18;y=r.y+10;small=r.w<1280
        controls=[('previous','Prev',('previous',),42),('restart','Start',('restart',),46),('play','Pause' if app.playing else 'Play',('play',),58),
            ('back','< tick' if not small else '<',('step',-1),48 if not small else 30),('forward','tick >' if not small else '>',('step',1),48 if not small else 30),
            ('prev-event','< event',('event',-1),57),('next-event','event >',('event',1),57),('next','Next',('next',),44)]
        for key,label,action,width in controls:
            ui.button(surface,'transport-'+key,label,pygame.Rect(x,y,width,32),action,active=key=='play' and app.playing,tooltip='Space: play/pause; arrows: tick; comma/period: events');x+=width+6
        if small:
            ui.button(surface,'speed-menu',f'{app.speed:g}x',pygame.Rect(x+4,y,74,32),('dropdown','speed',SPEEDS),tooltip='Actual playback speed');x+=86
        else:
            for value in SPEEDS:
                ui.button(surface,'speed-'+str(value),f'{value:g}x',pygame.Rect(x+6,y,42,32),('speed',value),abs(value-app.speed)<1e-6,accent=CYAN);x+=48
        ui.button(surface,'playback-mode',app.mode.replace('_',' '),pygame.Rect(x+8,y,128,32),('dropdown','mode',['manual','auto_next','follow_latest']),tooltip='Only complete recordings are queued');x+=144
        ui.button(surface,'timeline-view','Full timeline' if app.prefs['timeline_all'] else 'Past markers',pygame.Rect(x,y,116,32),('toggle','timeline_all'),app.prefs['timeline_all'],accent=AMBER,tooltip='Full timeline explicitly reveals future event markers')
        if r.w>1500:
            ui.button(surface,'bookmark','Bookmark',pygame.Rect(r.right-246,y,102,32),('bookmark',),tooltip='Save a local replay-time bookmark')
            ui.button(surface,'event-log','Event log',pygame.Rect(r.right-134,y,112,32),('log',),tooltip='Search/filter and jump to recorded events')
        else:
            ui.button(surface,'event-log','Log',pygame.Rect(r.right-62,y,48,32),('log',))
        index=app.index;duration=index.player.duration if index else 0.;current=index.player.current_sim_time if index else 0.
        line=pygame.Rect(r.x+28,r.y+63,r.w-56,37);self.timeline_rect=line
        pygame.draw.rect(surface,(36,49,65),pygame.Rect(line.x,line.y+14,line.w,6),border_radius=3)
        if duration:
            filled=pygame.Rect(line.x,line.y+14,round(line.w*current/duration),6);pygame.draw.rect(surface,LIME,filled,border_radius=3)
        ui.hit('timeline-scrub',line,('scrub',),'Drag to seek. Exact seconds can be entered below.')
        bins={}
        for marker in self.visible_markers(app):
            if duration<=0:continue
            px=line.x+round(marker.sim_time/duration*line.w);lane=['Kills','Reports','Meetings','Votes','Claims','Tasks','Rewards','Result'].index(category(marker.kind)) if category(marker.kind) in ['Kills','Reports','Meetings','Votes','Claims','Tasks','Rewards','Result'] else 0
            bins[(px//4,lane)]=marker
        for (_,lane),marker in bins.items():
            x=line.x+round(marker.sim_time/max(.01,duration)*line.w);yy=line.y+5+(lane%3)*9
            pygame.draw.circle(surface,marker.color,(x,yy),3)
            ui.hit('marker-'+str(marker.tick)+'-'+marker.kind+str(marker.player_id),pygame.Rect(x-4,yy-4,9,9),('seek',marker.sim_time),f'{marker.sim_time:.2f}s · '+marker.label)
        knob=line.x+round(line.w*current/duration) if duration else line.x
        pygame.draw.line(surface,TEXT,(knob,line.y+2),(knob,line.bottom-2),2)
        yy=r.y+108
        ui.button(surface,'seek-seconds',app.seek_text if app.text_focus=='seek_text' else f'{current:.2f} / {duration:.2f}s',pygame.Rect(24,yy,192,30),('focus','seek_text'),active=app.text_focus=='seek_text',tooltip='Enter exact simulation seconds, then press Enter')
        ui.button(surface,'timeline-filter',app.prefs['event_filter'],pygame.Rect(228,yy,102,30),('dropdown','pref:event_filter',EVENTS))
        ids=['All']+list(index.frame().players) if index else ['All']
        ui.button(surface,'timeline-player',app.prefs['event_player'],pygame.Rect(341,yy,96,30),('dropdown','pref:event_player',ids),tooltip='Filter events by recorded player')
        x=454
        for label,color in [('Kill',RED),('Report',LIME),('Meeting',CYAN),('Vote',PURPLE),('Task',AMBER),('Reward',MUTED)]:
            if x+65>r.right-190:break
            pygame.draw.circle(surface,color,(x,yy+15),3);text(surface,label,x+9,yy+7,11,MUTED);x+=78
        elapsed=int(time.monotonic()-app.opened_monotonic)
        text(surface,f'Viewer {elapsed//60:02d}:{elapsed%60:02d} · {app.fps:.0f} fps',r.right-180,yy+8,11,MUTED,width=170)

    def event_log(self,surface,app,ui):
        size=surface.get_size();back=pygame.Rect(0,0,*size);ui.hit('close-log-back',back,('log',))
        r=pygame.Rect(70,130,size[0]-140,size[1]-230);box(surface,r,PANEL,CYAN,12)
        text(surface,'EVENT JOURNAL',r.x+22,r.y+17,23,TEXT,True)
        ui.button(surface,'log-close','Close',pygame.Rect(r.right-95,r.y+16,72,32),('log',))
        ui.button(surface,'log-search',app.event_query or 'Search recorded events…',pygame.Rect(r.x+22,r.y+61,r.w-44,34),('focus','event_query'),active=app.text_focus=='event_query',tooltip='Chronological records only')
        body=pygame.Rect(r.x+20,r.y+110,r.w-40,r.h-130);old=surface.get_clip();surface.set_clip(body)
        rows=app.index.filtered_events(app.prefs['event_filter'],app.prefs['event_player'],app.prefs['omniscient'],app.event_query,app.prefs['reveal_ballots']) if app.index else []
        for i,row in enumerate(reversed(rows)):
            rect=pygame.Rect(body.x,body.y+i*37-app.scrolls.get('log',0),body.w,32)
            if rect.colliderect(body):ui.button(surface,'log-row-'+str(i),f"{row['time']:07.2f}s  /  {row['category'].upper()}  /  {row['text']}",rect,('seek_match',row['time'],None),tooltip='Jump to this actual replay moment')
        if not rows:text(surface,'No matching recorded events by this tick.',body.x+10,body.y+20,16,MUTED)
        surface.set_clip(old);app.scroll_regions.append((body,'log',max(0,len(rows)*37-body.h)))

    def dropdown(self,surface,app,ui):
        menu=app.dropdown;ui.hit('dropdown-dismiss',surface.get_rect(),('close_dropdown',))
        options=menu['options'];height=min(420,len(options)*36+16);w=320
        x=min(surface.get_width()-w-15,max(15,menu['pos'][0]));y=min(surface.get_height()-height-15,max(15,menu['pos'][1]))
        r=pygame.Rect(x,y,w,height);box(surface,r,CARD,CYAN,9)
        for i,(label,action) in enumerate(options):
            item=pygame.Rect(x+8,y+8+i*36,w-16,31)
            if item.bottom>r.bottom:break
            ui.button(surface,'dropdown-choice-'+str(i),label,item,action,tooltip=label)

    def help(self,surface,app,ui):
        ui.hit('help-backdrop',surface.get_rect(),('help',))
        r=pygame.Rect(90,130,surface.get_width()-180,min(570,surface.get_height()-230));box(surface,r,PANEL,CYAN,12)
        text(surface,'CONTROLS & DATA BOUNDARIES',r.x+24,r.y+20,23,TEXT,True)
        ui.button(surface,'help-close','Close',pygame.Rect(r.right-100,r.y+17,74,32),('help',))
        lines=[('Space','Play / pause'),('Left / Right','Step recorded tick'),('Comma / Period','Previous / next visible event'),('N / P / R','Next / previous recording / restart'),('1–9','Select recorded player slot'),('Wheel / Middle-drag','Map zoom / pan'),('F11 / F12','Fullscreen / capture actual GUI'),('B / S / L / V','Bookmark / Settings / Library / meeting'),('Tab / Shift+Tab / Enter','Focus next / previous / activate')]
        y=r.y+75
        for key,label in lines:
            text(surface,key,r.x+28,y,14,CYAN,mono=True,width=310)
            text(surface,label,r.x+350,y,14,TEXT,width=r.w-380);y+=30
        wrap(surface,'All views read existing recordings. Private telemetry stays with the selected actor. Hidden targets require BOTH omniscient mode and ballot revelation. Library/Study and Full timeline are explicitly retrospective. Missing data is never inferred as actor knowledge. The viewer runs no policy or simulator.',pygame.Rect(r.x+28,y+16,r.w-56,r.bottom-y-25),14,AMBER)
