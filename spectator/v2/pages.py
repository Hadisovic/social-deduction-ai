"""Retrospective library/study analysis and real persisted application settings."""
import pygame
from .widgets import *
from .data import MISSING

FILTERS=['All','Favorites','Recent','Bookmarked','Wrong ballots','Confident errors','Report decisions','Impostor reports','Position changes','Task victory','No neural telemetry']


class Pages:
    def draw(self,surface,rect,app,ui):
        old=surface.get_clip();surface.set_clip(rect)
        {'Library':self.library,'Study':self.study,'Settings':self.settings}[app.scene](surface,rect,app,ui)
        surface.set_clip(old)

    def header(self,surface,r,title,subtitle):
        text(surface,title,r.x+24,r.y+16,27,TEXT,True)
        text(surface,subtitle,r.x+24,r.y+54,14,MUTED,width=r.w-48)

    def library(self,surface,r,app,ui):
        self.header(surface,r,'Replay library','RETROSPECTIVE ANALYSIS · Complete recordings only · Results and issue tags use full-replay/offline truth.')
        x,y=r.x+24,r.y+92
        query=app.library_query or 'Search seed, model, result, session…'
        ui.button(surface,'library-search',query,pygame.Rect(x,y,max(230,r.w-670),36),('focus','library_query'),active=app.text_focus=='library_query',accent=CYAN,tooltip='Type to search. Escape ends text entry.')
        x+=max(230,r.w-670)+10
        for key,value,options,width in [('condition',app.condition,['All']+sorted({e['condition'] for e in app.library.entries.values()}),155),('context',app.context,['All','ordinary','body','meeting_crew','meeting_impostor'],142),('tag',app.tag,FILTERS,165),('sort',app.sort,['Newest','Seed','Duration','Model','Outcome'],112)]:
            ui.button(surface,'filter-'+key,str(value),pygame.Rect(x,y,width,36),('dropdown',key,options),tooltip='Filter '+key);x+=width+8
        y+=52
        ui.button(surface,'open-file','Open replay…',pygame.Rect(r.x+24,y,132,34),('open_file',),tooltip='Read a .jsonl recording; no simulation is started.')
        ui.button(surface,'open-folder','Watch folder…',pygame.Rect(r.x+166,y,140,34),('open_folder',))
        status=f'{len(app.library.entries)} indexed recordings'
        if app.library.busy:status+=' · indexing in background'
        if app.library.cap_reached:status+=' · 5,000-file safety cap'
        ui.button(surface,'bookmarks-menu','Bookmarks',pygame.Rect(r.x+318,y,128,34),('bookmarks',),tooltip='Jump to a saved local replay/decision moment')
        text(surface,status,r.x+466,y+8,13,MUTED,width=r.w-490)
        body=pygame.Rect(r.x+20,y+48,r.w-40,max(40,r.bottom-y-60));old=surface.get_clip();surface.set_clip(body)
        rows=app.library.filtered(app.library_query,app.condition,app.context,app.tag,app.sort,app.prefs['favorites'],app.prefs['recent'],app.prefs['bookmarks'])
        rowh=116 if app.prefs['density']=='compact' else 137
        offset=app.scrolls.get('library',0)
        for i,entry in enumerate(rows):
            card=pygame.Rect(body.x+4,body.y+i*rowh-offset,body.w-8,rowh-10)
            if not card.colliderect(body):continue
            active=app.current_path==entry['path'];box(surface,card,HOVER if card.collidepoint(ui.mouse) else CARD,LIME if active else LINE,10)
            ui.hit('replay-'+entry['path'],card,('load',entry['path']),'Open this genuine recording',surface=surface)
            label=f"{entry['condition'].upper()} / {entry['context'].replace('_',' ').upper()}"
            text(surface,label,card.x+18,card.y+12,17,LIME if active else TEXT,True,width=card.w-300)
            text(surface,f"Seed {entry['seed']}  ·  {entry['duration']:.1f}s  ·  {entry['session'] or 'Historical session'}",card.x+18,card.y+41,13,MUTED,width=card.w-265)
            task=f"{entry['tasks_completed']}/{entry['tasks_total']} tasks" if entry['tasks_total'] else 'Tasks not recorded'
            details=f"{entry['kills']} kills   {entry['reports']} reports ({entry['neural_reports']} neural, {entry['prefix_reports']} setup)   {entry['meetings']} meetings   {task}   {entry['neural_count']} neural records"
            text(surface,details,card.x+18,card.y+65,13,MUTED,width=card.w-265)
            text(surface,entry['reason'].replace('_',' '),card.right-245,card.y+17,13,LIME if entry['winner']=='crewmate' else RED,width=175)
            fav=entry['path'] in app.prefs['favorites']
            ui.button(surface,'favorite-'+entry['path'],'*' if fav else '+',pygame.Rect(card.right-58,card.y+14,38,30),('favorite',entry['path']),fav,tooltip='Toggle local favorite')
            if rowh>120:text(surface,' · '.join(entry['tags']) or 'No indexed review flags',card.x+18,card.y+92,12,AMBER,width=card.w-50)
        if not rows:text(surface,'No matching recordings. Open a folder or clear the filters.',body.x+10,body.y+22,17,MUTED)
        surface.set_clip(old);maximum=max(0,len(rows)*rowh-body.h);app.scroll_regions.append((body,'library',maximum));scrollbar(surface,body,offset,maximum)

    def settings(self,surface,r,app,ui):
        self.header(surface,r,'Settings','Preferences are saved locally. Changes affect the viewer only; no simulation or research data is modified.')
        nav=['Appearance','Map','Debug','Playback','Interface'];x=r.x+24;y=r.y+92
        for name in nav:
            ui.button(surface,'settings-'+name,name,pygame.Rect(x,y,132,36),('settings_category',name),app.settings_category==name,accent=CYAN);x+=142
        ui.button(surface,'reset-settings','Reset defaults',pygame.Rect(r.right-172,y,146,36),('reset_settings',),accent=AMBER,tooltip='Restore viewer preferences; replay/checkpoint files are never deleted.')
        body=pygame.Rect(r.x+24,y+54,r.w-48,r.bottom-y-70);old=surface.get_clip();surface.set_clip(body)
        category=app.settings_category
        groups={
            'Appearance':[('choice','theme','Surface theme',['ink','midnight']),('choice','scale','Interface / font scale',[.85,1.,1.1,1.2,1.3]),('choice','density','Data density',['comfortable','compact']),('toggle','reduced_motion','Reduced motion'),('toggle','interpolate','Interpolate compatible roaming snapshots')],
            'Map':[('toggle','labels','Player labels'),('toggle','rooms','Room labels'),('toggle','tasks','Task-console positions'),('toggle','bodies','Unreported bodies'),('toggle','visibility','Selected actor sight-radius overlay'),('toggle','trails','Recorded traveled trail (not a planned route)'),('toggle','collisions','Collision geometry'),('toggle','follow','Follow selected player'),('toggle','reveal_roles','Role markers (requires omniscient mode)')],
            'Debug':[('toggle','omniscient','Omniscient spectator mode'),('toggle','reveal_roles','Reveal actual player roles'),('toggle','reveal_ballots','Reveal hidden ballot targets'),('toggle','neural','Selected actor neural details'),('toggle','rewards','Recorded rewards'),('toggle','evidence','Private suspicion / evidence inspector'),('toggle','unavailable','Show optional unavailable-data rows')],
            'Playback':[('choice','speed','Playback speed',[.25,.5,1.,2.,4.,8.,16.]),('choice','mode','Playback mode',['manual','auto_next','follow_latest']),('toggle','autoplay','Autoplay newly selected replays'),('toggle','timeline_all','Full-replay timeline (reveals future markers)'),('choice','event_filter','Event filter',['All','Kills','Reports','Meetings','Votes','Claims','Tasks','Rewards','Result'])],
            'Interface':[('toggle','roster','Player roster panel'),('toggle','inspector','Agent inspector panel'),('choice','left_width','Roster width',[220,252,290,340]),('choice','right_width','Inspector width',[320,370,430,500])]}
        yy=body.y-app.scrolls.get('settings',0);column=min(850,body.w)
        for row in groups[category]:
            kind,key,label,*rest=row;card=pygame.Rect(body.x,yy,column,54)
            box(surface,card,CARD,LINE,8)
            if kind=='toggle':ui.toggle(surface,'setting-'+key,label,app.prefs[key],card,('toggle',key),self.setting_help(key))
            else:
                text(surface,label,card.x+16,card.y+16,15,TEXT,width=card.w-260)
                value=app.prefs[key];display=f'{value*100:.0f}%' if key=='scale' else str(value)
                ui.button(surface,'setting-'+key,display,pygame.Rect(card.right-230,card.y+9,210,36),('dropdown','pref:'+key,rest[0]),tooltip=self.setting_help(key))
            yy+=65
        if category=='Interface':
            ui.button(surface,'default-player','Use selected player as default: '+str(app.selected or 'none'),pygame.Rect(body.x,yy,column,40),('set_default_player',));yy+=54
            yy=wrap(surface,'Drag the vertical panel separators in Match view to resize. Narrow windows may collapse a panel; the header buttons restore it.',pygame.Rect(body.x,yy,column,100),14,MUTED)+20
        notes={'Map':'Sight circles show a recorded range, not reconstructed line of sight. Planned routes and destinations are unavailable unless explicitly recorded. The trail is past recorded movement.',
               'Debug':'Hidden targets require BOTH omniscient mode and Reveal hidden ballot targets. Ground truth and selected-agent private telemetry never enter a learner. Sabotage remains not implemented.',
               'Playback':'Chronological mode withholds future event markers. Full-replay timeline is an explicit retrospective feature. Auto-next queues complete recordings; follow-latest switches to newly completed recordings.',
               'Appearance':'Typography and layout respond to interface scale. Reduced motion disables interpolation and decorative movement. Camera transforms never change recorded positions.',
               'Interface':'Settings and bookmarks live in the local preferences file. Reset restores UI defaults, not historical data.'}
        yy=wrap(surface,notes[category],pygame.Rect(body.x,yy+8,column,140),14,AMBER)+24
        if app.prefs.error:yy=wrap(surface,'Preference file error: '+app.prefs.error,pygame.Rect(body.x,yy,column,110),14,RED)
        surface.set_clip(old);maximum=max(0,yy-body.y+app.scrolls.get('settings',0)-body.h);app.scroll_regions.append((body,'settings',maximum));scrollbar(surface,body,app.scrolls.get('settings',0),maximum)

    @staticmethod
    def setting_help(key):
        return {'reveal_ballots':'Requires omniscient mode; reveals only targets recorded by the displayed tick.',
                'timeline_all':'Explicit retrospective mode: timeline markers may describe future moments.',
                'visibility':'Geometric range only. Actor-visible entity lists were not recorded.',
                'trails':'Draws the selected player’s recorded past trajectory; no route is fabricated.',
                'scale':'Changes actual typography and layout; all hitboxes are rebuilt.',
                'unavailable':'Hide optional missing rows; essential data-integrity notices remain.'}.get(key,'Saved locally; affects this viewer only.')

    def study(self,surface,r,app,ui):
        self.header(surface,r,'Research desk','RETROSPECTIVE ANALYSIS · Context-separated development results · Acceptance remains pending.')
        y=r.y+92;x=r.x+24
        for name in ('Registered study','Paired replay','Training monitor'):
            ui.button(surface,'study-'+name,name,pygame.Rect(x,y,160,36),('study_tab',name),app.study_tab==name);x+=172
        ui.button(surface,'metrics-open','Load metrics JSON…',pygame.Rect(r.right-218,y,192,36),('open_metrics',))
        body=pygame.Rect(r.x+24,y+55,r.w-48,r.bottom-y-70)
        old=surface.get_clip();surface.set_clip(body)
        offset=app.scrolls.get('study',0);drawbody=body.move(0,-offset)
        if app.study_tab=='Paired replay':
            height=self.paired(surface,drawbody,app,ui) or 180
            surface.set_clip(old);maximum=max(0,height-body.h);app.scroll_regions.append((body,'study',maximum));scrollbar(surface,body,offset,maximum);return
        if app.study_tab=='Training monitor':
            height=self.monitor(surface,drawbody,app,ui)
            surface.set_clip(old);maximum=max(0,height-body.h);app.scroll_regions.append((body,'study',maximum));scrollbar(surface,body,offset,maximum);return
        if not app.metrics:
            wrap(surface,'No aggregate metrics loaded. Select the existing registered study metrics.json; no benchmark or neural model is executed.',body.inflate(-10,-20),17,MUTED);surface.set_clip(old);return
        groups=app.metrics.get('registered_aggregate',app.metrics).get('groups',{})
        if not groups:wrap(surface,'This JSON has no supported aggregate groups.',body,17,RED);surface.set_clip(old);return
        x=body.x;y=drawbody.y
        for context in ('ordinary','controlled','body','meeting'):
            ui.button(surface,'study-context-'+context,context.title(),pygame.Rect(x,y,120,34),('study_context',context),app.study_context==context,accent=CYAN);x+=132
        y+=52
        text(surface,'ONE INITIALIZATION · EIGHT PAIRED SEED CLUSTERS · FOUR PRIMARY BODY CASES',body.x,y,13,AMBER,True,width=body.w)
        y+=34;cw=(body.w-32)//3
        colors=[CYAN,LIME,PURPLE]
        for i,label in enumerate(('original','shaped_v2','zero')):
            g=groups.get(label,{}).get(app.study_context,{})
            card=pygame.Rect(body.x+i*(cw+16),y,cw,155);box(surface,card,CARD,LINE,10)
            text(surface,{'original':'A · ORIGINAL','shaped_v2':'B · SHAPED V2','zero':'FROZEN REFERENCE'}[label],card.x+18,card.y+14,14,colors[i],True,width=cw-30)
            wins,games=g.get('wins',0),g.get('games',0)
            text(surface,f'{wins}/{games}',card.x+18,card.y+43,34,TEXT,True)
            text(surface,'crewmate wins',card.x+122,card.y+59,13,MUTED,width=cw-140)
            progress(surface,pygame.Rect(card.x+18,card.y+99,cw-36,7),wins/games if games else None,colors[i])
            text(surface,f"{g.get('games_with_crew_reports',0)} games with neural reports",card.x+18,card.y+119,13,MUTED,width=cw-30)
        y+=180
        rows=[('Living task events',lambda g:g.get('actor_event_counts',{}).get('alive_tasks',0)),
              ('Accepted neural reports',lambda g:g.get('report_actions',{}).get('accepted',0)),
              ('Legal actor/body pairs',lambda g:g.get('reporting_opportunities',{}).get('choice_legal',0)),
              ('Ledger correct / wrong ballots',lambda g:str(g.get('actor_event_counts',{}).get('correct_votes',0))+' / '+str(g.get('actor_event_counts',{}).get('incorrect_votes',0))),
              ('Skip events',lambda g:g.get('actor_event_counts',{}).get('skips',0)),
              ('Frozen belief top-1',lambda g:f"{g['frozen_belief']['top1']:.1%}" if g.get('frozen_belief',{}).get('top1') is not None else MISSING),
              ('Frozen belief NLL',lambda g:f"{g['frozen_belief']['nll']:.3f}" if g.get('frozen_belief',{}).get('nll') is not None else MISSING)]
        for name,extract in rows:
            text(surface,name,body.x,y,14,MUTED,width=290)
            for i,label in enumerate(('original','shaped_v2','zero')):
                text(surface,extract(groups.get(label,{}).get(app.study_context,{})),body.x+330+i*max(160,(body.w-330)//3),y,15,TEXT,mono=True,width=max(150,(body.w-330)//3)-12)
            pygame.draw.line(surface,LINE,(body.x,y+27),(body.right,y+27));y+=36
        note='Registered ballot metrics are reward-ledger counters; direct replay ballot counts can differ when voters become inactive. Historical results remain unchanged. One initialization, correlated agent/body counts and reporter/script/choreography confounds limit conclusions. The frozen belief predictor is shared.'
        end=wrap(surface,note,pygame.Rect(body.x,y+12,body.w,100),14,AMBER)+20
        surface.set_clip(old);maximum=max(0,end-drawbody.y-body.h);app.scroll_regions.append((body,'study',maximum));scrollbar(surface,body,offset,maximum)

    def paired(self,surface,r,app,ui):
        current=app.library.entries.get(app.current_path)
        if not app.index or not current:
            wrap(surface,'Load a recording and allow the library to index it. Pairing requires the same session, seed, context, environment/source and comparable policy version.',r,17,MUTED);return
        matches=[e for e in app.library.entries.values() if app.index.matching_header(e['header'])]
        if not matches:
            wrap(surface,'No compatible paired counterpart found. Unrelated recordings are not presented as paired.',r,17,AMBER);return
        matches.sort(key=lambda e:(e['condition']!='shaped_v2' if current['condition']=='original' else e['condition']!='original',-e['mtime']))
        other=matches[0];y=r.y
        text(surface,f"Matched seed {current['seed']} · {current['context']} · {current['session']}",r.x,y,17,CYAN,True,width=r.w);y+=42
        half=(r.w-20)//2
        for i,entry in enumerate((current,other)):
            card=pygame.Rect(r.x+i*(half+20),y,half,450);box(surface,card,CARD,LINE,10)
            text(surface,entry['condition'].upper(),card.x+20,card.y+16,22,LIME,True,width=half-40)
            profile=entry['header'].get('config',{}).get('reward_profile',{})
            scoring='Common/original' if isinstance(profile,dict) and profile.get('report')==0 and profile.get('correct_vote')==0 else 'Recorded shaped' if profile else MISSING
            text(surface,'Scoring: '+scoring,card.x+20,card.y+49,12,AMBER,width=half-40)
            yy=card.y+78
            for key in ('reason','duration','kills','reports','neural_reports','meetings','tasks_completed','tasks_total','correct_ballots','wrong_ballots','skips','report_decision_windows','confident_errors'):
                text(surface,key.replace('_',' ').title(),card.x+20,yy,13,MUTED,width=half*.58)
                value=entry.get(key);text(surface,str(value) if value is not None else MISSING,card.x+half*.6,yy,13,TEXT,width=half*.37);yy+=23
            ui.button(surface,'pair-open-'+str(i),'Open this recording',pygame.Rect(card.x+20,card.bottom-48,half-40,32),('load',entry['path']))
        y+=472
        wrap(surface,'Ballots are actual published crewmate votes; they can differ from registered ledger counters. Confidence flags count raw neural records, not independent meeting audits. All reports include setup; neural reports exclude it. Task totals include prefix/ghost work. Reward profiles can differ. Review flags are not confirmed bugs.',pygame.Rect(r.x,y,r.w,110),15,AMBER)
        y+=105
        for moment in current['moments'][:6]:
            ui.button(surface,'review-moment-'+str(y),f"{moment['time']:.1f}s · {moment['label']}",pygame.Rect(r.x,y,r.w,32),('seek_match',moment['time'],moment.get('player')));y+=40
        return y-r.y+20

    def monitor(self,surface,r,app,ui):
        if app.status.get('schema')=='patrick-jane-progress-v1':
            return self.patrick_jane_monitor(surface,r,app,ui)
        status=app.status;y=r.y
        text(surface,'READ-ONLY TRAINING STATUS',r.x,y,21,TEXT,True);y+=42
        text(surface,str(status.get('stage','No status file recorded')).upper(),r.x,y,16,CYAN,True,width=r.w);y+=40
        decisions=status.get('decisions',{});target=status.get('decision_target')
        if not isinstance(decisions,dict):decisions={'recorded decisions':decisions}
        for label,value in decisions.items():
            text(surface,f'{label}: {value}'+(f' / {target}' if target else ''),r.x,y,17,TEXT);y+=32
            progress(surface,pygame.Rect(r.x,y,min(800,r.w),8),value/target if isinstance(value,(int,float)) and isinstance(target,(int,float)) and target>0 else None,CYAN);y+=28
        for label,key in [('Validation games','validation_games'),('Validation target','validation_target'),('Elapsed seconds','elapsed_seconds'),('ETA','eta_seconds'),('Training CPU','cpu_percent'),('Training RAM','ram_bytes')]:
            value=status.get(key,MISSING)
            text(surface,label,r.x,y,14,MUTED);text(surface,str(round(value,2)) if isinstance(value,float) else str(value),r.x+230,y,14,TEXT,mono=True,width=r.w-240);y+=30
        y+=16
        text(surface,'RECORDED TRAINING REPORT CURVES',r.x,y,13,LIME,True);y+=32
        curves=app.metrics.get('training_curves',{})
        chart=pygame.Rect(r.x,y,min(900,r.w),145);box(surface,chart,CARD,LINE,8)
        if curves:
            all_rows=[p for rows in curves.values() for p in rows]
            xmax=max(1,max(p['matched_games'] for p in all_rows));ymax=max(1,max(p['cumulative_accepted_neural_reports'] for p in all_rows))
            for label,color in [('original',CYAN),('shaped_v2',LIME)]:
                rows=curves.get(label,[])
                if rows:
                    points=[(chart.x+14+p['matched_games']/xmax*(chart.w-28),chart.bottom-20-p['cumulative_accepted_neural_reports']/ymax*(chart.h-50)) for p in rows]
                    if len(points)>1:pygame.draw.lines(surface,color,False,points,2)
                    text(surface,f"{label}: {rows[-1]['cumulative_accepted_neural_reports']} reports / {rows[-1]['matched_games']} games",chart.x+18,chart.y+9+(18 if label=='shaped_v2' else 0),12,color)
            text(surface,f'Games 0–{xmax} · reports 0–{ymax} (shared axes)',chart.x+18,chart.bottom-20,11,MUTED)
        else:text(surface,'Historical training curves not recorded in loaded metrics',chart.x+15,chart.y+25,14,MUTED,width=chart.w-30)
        y=chart.bottom+24
        wrap(surface,'Status values come from the supplied atomic status JSON. Curves come from recorded aggregate training batches, not fabricated live samples. CPU/RAM/ETA remain unavailable when absent. This viewer never starts, pauses or changes training.',pygame.Rect(r.x,y,r.w,100),14,AMBER)
        return y-r.y+110

    def patrick_jane_monitor(self,surface,r,app,ui):
        from pathlib import Path
        status=app.status;y=r.y;counters=status.get('counters',{})
        text(surface,'PATRICK JANE · CENTRAL RECURRENT PPO',r.x,y,18,CYAN,True);y+=38
        completion=status.get('completion_status',{})
        if completion:
            label='TRAINING COMPLETE' if completion.get('success') else 'TRAINING '+str(completion.get('state','stopped')).upper()
            text(surface,label,r.x,y,18,LIME if completion.get('success') else AMBER,True);y+=32
        for label,value in [('Stage',status.get('stage','Registered training')),('Update',counters.get('update',0)),
            ('Individual crew decisions',counters.get('individual_decisions',0)),('Worker environments',status.get('workers',0)),
            ('Elapsed seconds',round(status.get('elapsed',0),1))]:
            text(surface,label,r.x,y,14,MUTED);text(surface,str(value),r.x+265,y,14,TEXT);y+=29
        for name,value in status.get('resources',{}).items():
            text(surface,name+' · mean / peak',r.x,y,13,MUTED)
            text(surface,f"{value['mean']:.1f} / {value['peak']:.1f}",r.x+330,y,13,TEXT);y+=26
        counters=status.get('counters',{});active=status.get('environment_progress',{})
        total=status.get('overall_decisions',counters.get('individual_decisions',0)+active.get('decisions',0))
        target=status.get('target_decisions',0)
        if target:
            progress(surface,pygame.Rect(r.x,y,min(850,r.w),10),total/target);y+=24
            text(surface,f"{total:,} / {target:,} decisions · {status.get('throughput',0):.1f}/s · ETA {status.get('eta_seconds',0)/60:.1f} min",r.x,y,14,CYAN);y+=30
        outcomes=status.get('outcomes',{})
        text(surface,f"Complete games {outcomes.get('games',0)} · crew wins {outcomes.get('wins',0)} · losses {outcomes.get('losses',0)} · draws {outcomes.get('draws',0)}",r.x,y,14,TEXT);y+=30
        text(surface,'Latest checkpoint: '+str(status.get('checkpoint','unavailable')),r.x,y,12,MUTED,width=r.w);y+=28
        if str(status.get('stage','')).startswith('validation'):
            text(surface,f"Development validation: {status.get('validation_complete',0)} / {status.get('validation_target',0)} complete",r.x,y,14,CYAN);y+=30
        for name,value in status.get('reward_components',{}).items():
            text(surface,f'{name}: {value:.3f}',r.x,y,12,MUTED);y+=21
        for name,value in status.get('ppo',{}).items():
            if isinstance(value,(int,float)):text(surface,f'PPO {name}: {value:.5g}',r.x,y,12,MUTED);y+=21
        curve=status.get('learning_curve',[])
        if curve:
            chart=pygame.Rect(r.x,y,min(850,r.w),120);box(surface,chart,CARD,LINE,8)
            text(surface,'HELD-OUT CREW WIN RATE · FIXED DEVELOPMENT SCENARIOS',chart.x+12,chart.y+8,11,CYAN)
            xmax=max(1,max(v['decisions'] for v in curve))
            points=[(chart.x+14+v['decisions']/xmax*(chart.w-28),chart.bottom-16-v['validation_wins']/max(1,v['validation_games'])*(chart.h-48)) for v in curve]
            if len(points)>1:pygame.draw.lines(surface,LIME,False,points,2)
            for point in points:pygame.draw.circle(surface,LIME,point,3)
            y=chart.bottom+18
        y+=18;text(surface,'PARALLEL ENVIRONMENTS · SELECT LIVE INSTANCE OR COMPLETE REPLAY',r.x,y,13,LIME,True);y+=30
        for stream in status.get('environment_progress',{}).get('streams',[]):
            label=f"Env {stream['environment']:02d} · seed {stream['seed']} · {stream['simulation_seconds']:.1f}s · {'complete' if stream['complete'] else 'simulating'}"
            path=stream.get('replay')
            if stream.get('snapshot') and status.get('monitor_dir') and not stream['complete']:
                ui.button(surface,'pj-live-'+str(stream['environment']),label+' · LIVE',pygame.Rect(r.x,y,min(650,r.w),32),('live_environment',stream['environment']))
                if path and stream['complete'] and Path(path).is_file():
                    ui.button(surface,'pj-env-'+str(stream['environment']),'Replay',pygame.Rect(r.x+660,y,95,32),('load',str(Path(path).resolve())))
            elif path and stream['complete'] and Path(path).is_file():
                ui.button(surface,'pj-env-'+str(stream['environment']),label+' · inspect',pygame.Rect(r.x,y,min(850,r.w),32),('load',str(Path(path).resolve())))
            else:text(surface,label+(' · recording selected' if path else ' · not recorded'),r.x,y+7,13,MUTED,width=r.w)
            y+=40
        wrap(surface,'Live snapshots show genuine selected-instance activity with bounded history. Complete replays contain actual results and full playback controls. Action-head probabilities are not impostor beliefs. Engineering smoke is not evidence of learning.',pygame.Rect(r.x,y+12,r.w,110),14,AMBER)
        return y-r.y+130
