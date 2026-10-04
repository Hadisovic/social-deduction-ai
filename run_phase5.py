"""Watch a trained strategic crewmate; never silently substitutes scripted play."""
import argparse
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'artifacts/phase5/release/policy.pt')
    parser.add_argument('--seed',type=int,default=620000)
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--display',type=int,default=0)
    parser.add_argument('--window-scale',type=float)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--capture',type=Path)
    args=parser.parse_args(argv)
    if not args.checkpoint.is_file():parser.error('Missing release policy. Restore artifacts/phase5/release/policy.pt or pass --checkpoint explicitly.')
    if not 0<args.speed<=32:parser.error('Speed must be in (0,32]')
    import pygame,torch
    from phase5_runtime import checked_environment
    from phase5_policy import StrategicPolicy
    from phase5_renderer import PolicyPanel
    from phase3_renderer import Phase3Renderer,ViewOptions
    from phase4_avatar_hud import AvatarHUD
    from spectator_display import fitted_window_size
    torch.set_num_threads(1)
    policy,metadata=StrategicPolicy.load(args.checkpoint)
    env=checked_environment(metadata,ablation=metadata.get('ablation',False),render_trace=True);env.reset(seed=args.seed)
    pygame.init()
    try:size=fitted_window_size(pygame.display.get_desktop_sizes(),args.display,args.window_scale)
    except ValueError as error:parser.error(str(error))
    screen=pygame.display.set_mode(size,pygame.RESIZABLE,display=args.display)
    pygame.display.set_caption('Phase 5 | learned focal crew | '+('SMOKE MODEL' if metadata.get('quick') else 'trained policy'))
    renderer=Phase3Renderer(env.planner.map);panel=PolicyPanel();hud=AvatarHUD()
    options=ViewOptions(routes=True);clock=pygame.time.Clock();running=True;paused=False;acc=0.;frames=0;segment_duration=.6
    toggles={pygame.K_TAB:'hud',pygame.K_v:'visibility',pygame.K_t:'tasks',pygame.K_l:'labels',pygame.K_e:'events',pygame.K_b:'bodies'}
    if args.capture:args.capture.mkdir(parents=True,exist_ok=True)
    try:
        while running:
            dt=min(.25,clock.tick(60)/1000);single=False
            for e in pygame.event.get():
                if e.type==pygame.QUIT:running=False
                elif e.type==pygame.VIDEORESIZE:
                    size=(max(640,e.w),max(360,e.h));screen=pygame.display.set_mode(size,pygame.RESIZABLE,display=args.display)
                elif e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_ESCAPE:running=False
                    elif e.key in (pygame.K_p,pygame.K_SPACE):paused=not paused;acc=0.
                    elif e.key==pygame.K_RIGHT:single=paused
                    elif e.key==pygame.K_1:options.roles=not options.roles
                    elif e.key==pygame.K_g:options.routes=not options.routes
                    elif e.key==pygame.K_c:options.collision=not options.collision
                    elif e.key in toggles:
                        name=toggles[e.key];setattr(options,name,not getattr(options,name))
                    elif e.key in (pygame.K_PLUS,pygame.K_EQUALS,pygame.K_KP_PLUS):args.speed=min(32.,args.speed*2)
                    elif e.key in (pygame.K_MINUS,pygame.K_KP_MINUS):args.speed=max(.25,args.speed/2)
                    elif e.key in (pygame.K_r,pygame.K_n):
                        args.seed+=int(e.key==pygame.K_n);env.reset(seed=args.seed);acc=0.;segment_duration=.6
            if not running:break
            if not paused:acc+=dt*args.speed
            if not env.finished and (single or acc>=segment_duration):
                acc=max(0.,acc-segment_duration)
                action,_,_=policy.act(env.packet,True);_,_,_,_,info=env.step(action)
                segment_duration=info['elapsed_seconds']
                if single:acc=segment_duration
            animation_time,visual=animation_frame(env.visual_frames,acc)
            if env.observation.context.phase.value!='roaming':visual=None;animation_time=env.match.game.time
            canvas=pygame.Surface((1800,900));canvas.blit(renderer.draw(env.match.game,options,paused,args.speed,clock.get_fps(),visual,animation_time),(0,0))
            focal=env.match.game.players[env.focal]
            if options.routes and focal.navigator.route and env.observation.context.phase.value=='roaming':
                goal=renderer.map_renderer.point(focal.navigator.route[-1])
                pygame.draw.circle(canvas,(255,235,137),goal,10,2)
                renderer.text(canvas,'FOCAL GOAL',(goal[0]+12,goal[1]-8),(255,235,137),renderer.small)
            canvas.blit(panel.draw_policy(env,policy,metadata),(1440,0));hud.draw(canvas,env.observation)
            # Replace the inherited phase label; all remaining renderer geometry is shared.
            pygame.draw.rect(canvas,(12,20,33),(16,41,327,22))
            canvas.blit(renderer.small.render('LEARNED FOCAL CREW / PHASE 5',True,(139,163,184)),(23,46))
            factor=min(size[0]/1800,size[1]/900)
            scaled=pygame.transform.smoothscale(canvas,(round(1800*factor),round(900*factor)))
            screen.fill((8,13,22));screen.blit(scaled,((size[0]-scaled.get_width())//2,(size[1]-scaled.get_height())//2));pygame.display.flip()
            if args.capture and frames==0:pygame.image.save(canvas,str(args.capture/'phase5-start.png'))
            frames+=1
            if args.max_frames and frames>=args.max_frames:break
    finally:pygame.quit()
    return 0


def animation_frame(trace,elapsed):
    """Interpolate recorded physical steps, without changing engine or actor state."""
    if len(trace)<2:return trace[0]
    now=min(trace[-1][0],trace[0][0]+max(0.,elapsed))
    for (t0,a),(t1,b) in zip(trace,trace[1:]):
        if now<=t1 and t1>t0:
            fraction=(now-t0)/(t1-t0)
            return now,{pid:(tuple(x+(y-x)*fraction for x,y in zip(a[pid][0],b[pid][0])),
                             tuple((y-x)/(t1-t0) for x,y in zip(a[pid][0],b[pid][0]))) for pid in a}
    return trace[-1]


if __name__=='__main__':raise SystemExit(main())
