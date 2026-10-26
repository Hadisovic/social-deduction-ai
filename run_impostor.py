"""Watch the selected learned impostor against frozen full-29 and three crew scripts."""
import argparse
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'artifacts/impostor-v2/release/policy.pt')
    parser.add_argument('--seed',type=int,default=3600000)
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--stochastic',action='store_true',help='Sample learned family/target probabilities; default is greedy family then target')
    args=parser.parse_args(argv)
    if not args.checkpoint.is_file():parser.error('No selected model yet. Finish training or pass --checkpoint explicitly.')
    if not 0<args.speed<=32:parser.error('Speed must be in (0,32]')
    if any(a<=args.seed<=b for a,b in ((1230000,1230499),(1240000,1240499),(3300000,3300499))):parser.error('Reserved final-test seeds are unavailable in the viewer')
    import pygame,torch
    from impostor_research.frozen import FrozenModels
    from impostor_research.learning_v2 import load
    from impostor_research.strategic_v2 import StrategicEnv,FAMILIES
    from crewmate_v2.config import LobbySettings
    from phase3_renderer import Phase3Renderer,ViewOptions
    from run_phase5 import animation_frame
    from spectator_display import fitted_window_size
    torch.set_num_threads(1)
    models=FrozenModels();policy,metadata=load(args.checkpoint,models)
    env=StrategicEnv(models=models,trace=True,lobby=LobbySettings.from_file(Path(__file__).resolve().parent/'configs/crewmate-v2-lobby.json'));env.reset(seed=args.seed)
    generator=torch.Generator().manual_seed(20261027)
    pygame.init();size=fitted_window_size(pygame.display.get_desktop_sizes(),0,None)
    screen=pygame.display.set_mode(size,pygame.RESIZABLE);pygame.display.set_caption('Learned impostor | frozen crewmate opponents')
    renderer=Phase3Renderer(env.planner.map);options=ViewOptions(roles=True,routes=True)
    clock=pygame.time.Clock();running=True;paused=False;acc=.6;duration=.6;frames=0
    try:
        while running:
            dt=min(clock.tick(60)/1000,.25);single=False
            for event in pygame.event.get():
                if event.type==pygame.QUIT:running=False
                elif event.type==pygame.VIDEORESIZE:
                    size=(max(640,event.w),max(360,event.h));screen=pygame.display.set_mode(size,pygame.RESIZABLE)
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:running=False
                    elif event.key in (pygame.K_SPACE,pygame.K_p):paused=not paused
                    elif event.key==pygame.K_RIGHT:single=paused
                    elif event.key==pygame.K_1:options.roles=not options.roles
                    elif event.key==pygame.K_g:options.routes=not options.routes
                    elif event.key==pygame.K_v:options.visibility=not options.visibility
                    elif event.key in (pygame.K_PLUS,pygame.K_EQUALS):args.speed=min(32,args.speed*2)
                    elif event.key==pygame.K_MINUS:args.speed=max(.25,args.speed/2)
                    elif event.key in (pygame.K_r,pygame.K_n):
                        candidate=args.seed+int(event.key==pygame.K_n)
                        if not any(a<=candidate<=b for a,b in ((1230000,1230499),(1240000,1240499),(3300000,3300499))):
                            args.seed=candidate;env.reset(seed=args.seed);acc=.6;duration=.6
            if not running:break
            if not paused:acc+=dt*args.speed
            if not env.finished and (single or acc>=duration):
                acc=max(0.,acc-duration)
                action,_,_=policy.act(env.packet,generator=generator,deterministic=not args.stochastic)
                _,_,_,_,info=env.step(action);duration=info['elapsed_seconds']
                if single:acc=duration
            animation_time,visual=animation_frame(env.visual_frames,acc)
            if env.envelope.actor.context.phase.value!='roaming':visual=None;animation_time=env.match.game.time
            canvas=pygame.Surface((1800,900));canvas.blit(renderer.draw(env.match.game,options,paused,args.speed,clock.get_fps(),visual,animation_time),(0,0))
            pygame.draw.rect(canvas,(12,20,33),(16,41,450,23));renderer.text(canvas,'LEARNED IMPOSTOR / V2 / SPECTATOR VIEW',(23,46),(255,172,130),renderer.small)
            panel=pygame.Surface((360,900));panel.fill((15,26,41))
            def text(value,y,color=(221,233,242),font=None):
                rendered=(font or renderer.font).render(str(value),True,color);panel.blit(rendered,(16,y))
            focal=env.match.game.players[env.match.focal]
            text('LEARNED IMPOSTOR',20,font=renderer.title)
            text(f'{focal.color_name.upper()} / seed {args.seed}',60,(255,172,130))
            text(f'Checkpoint: {metadata["decisions"]} decisions',90)
            text('Frozen full-29 + 3 crew scripts',120)
            text('Crew report/witness safeguards ON',150)
            if metadata.get('development'):text('PREFLIGHT MODEL / NOT AN EVALUATED AGENT',173,(255,172,130),renderer.small)
            text(f'Last choice: {env.last_kind}',195)
            text(f'Cooldown: {env.envelope.actor.kill_cooldown or 0:.1f}s',225)
            text(f'Game: {env.match.game.time:.1f}s / {env.envelope.actor.context.phase.value}',255)
            text(f'Kills: {env.match.game.metrics["eliminations"]}',285)
            text(f'Return: {env.strategic_return:.2f}',315)
            text('ACTION-FAMILY PROBABILITIES',365,(105,229,197))
            with torch.no_grad():
                _,value,distribution,_,_=policy.evaluate({k:torch.as_tensor(v).unsqueeze(0) for k,v in env.packet.items()})
            probs=distribution.probs[0].tolist()
            for row,index in enumerate(sorted(range(len(probs)),key=lambda i:-probs[i])[:9]):
                y=400+row*35;text(f'{FAMILIES[index]:16s} {probs[index]:5.1%}',y)
                pygame.draw.rect(panel,(70,131,142),(16,y+21,round(328*probs[index]),4))
            if env.finished:text('Outcome: '+env.match.game.result.reason,740,(255,172,130))
            text('Space/P pause | Right step',790);text('+/- speed | R reset | N new game',820)
            text('1 roles | G routes | V sight | Esc quit',850)
            text('Spectator truth never enters the policy.',878,(139,163,184),renderer.small)
            canvas.blit(panel,(1440,0))
            # Focal marker follows the drawn physical position, not a hidden target.
            point=visual.get(env.match.focal,(focal.position,None))[0] if visual else focal.position
            pygame.draw.circle(canvas,(255,172,130),renderer.map_renderer.point(point),18,3)
            factor=min(size[0]/1800,size[1]/900);scaled=pygame.transform.smoothscale(canvas,(round(1800*factor),round(900*factor)))
            screen.fill((8,13,22));screen.blit(scaled,((size[0]-scaled.get_width())//2,(size[1]-scaled.get_height())//2));pygame.display.flip()
            if args.capture and frames==0:
                args.capture.mkdir(parents=True,exist_ok=True);pygame.image.save(canvas,str(args.capture/'impostor.png'))
            frames+=1
            if args.max_frames and frames>=args.max_frames:break
    finally:pygame.quit()
    return 0


if __name__=='__main__':raise SystemExit(main())
