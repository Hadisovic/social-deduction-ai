"""Watch live legitimate crew beliefs: python run_phase4.py. No action learning."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=15)
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--exit-on-finish',action='store_true')
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--display',type=int,default=0,help='Monitor index; 0 selects the primary display')
    parser.add_argument('--window-scale',type=float,help='Override automatic window fit (0.25 to 2.0)')
    args = parser.parse_args(argv)
    if not 0 < args.speed <= 32 or args.max_frames is not None and args.max_frames <= 0:
        parser.error('Speed must be in (0,32]; max-frames must be positive')
    import torch
    torch.set_num_threads(1)
    from belief.model import BeliefModel, DEFAULT_CHECKPOINT
    try:
        model = BeliefModel.load(args.checkpoint or DEFAULT_CHECKPOINT)
    except (FileNotFoundError,ValueError) as error:
        parser.error(str(error))
    import pygame
    from phase3_renderer import Phase3Renderer,ViewOptions
    from phase3_runner import ScriptedMatch
    from phase4_observer import MatchObserver
    from phase4_renderer import BeliefPanel
    from phase4_avatar_hud import AvatarHUD
    from spectator_display import fitted_window_size
    pygame.init()
    # Fit only the selected display: a small secondary monitor must not shrink
    # the primary window. This transform affects pixels, never world geometry.
    desktops = pygame.display.get_desktop_sizes()
    try:
        size = fitted_window_size(desktops,args.display,args.window_scale)
    except ValueError as error:
        parser.error(str(error))
    screen = pygame.display.set_mode(size,pygame.RESIZABLE,display=args.display)
    canvas = pygame.Surface((1800,900))
    pygame.display.set_caption('Social Deduction AI | Phase 4 | Loading')
    screen.fill((12,20,33)); pygame.display.flip(); pygame.event.pump()
    match = ScriptedMatch(args.seed)
    observer = MatchObserver(match,model)
    renderer = Phase3Renderer(match.game.map)
    panel = BeliefPanel()
    avatar_hud = AvatarHUD(renderer.map_width)
    options = ViewOptions()
    clock = pygame.time.Clock()
    paused, expanded, running = False, True, True
    speed,accumulator,frames = args.speed,0.,0
    event_offset = 0
    captured = set()
    if args.capture:
        args.capture.mkdir(parents=True,exist_ok=True)
    toggles = {pygame.K_1:'roles',pygame.K_g:'routes',pygame.K_v:'visibility',pygame.K_t:'tasks',
               pygame.K_l:'labels',pygame.K_c:'collision',pygame.K_e:'events',pygame.K_b:'bodies'}
    try:
        while running:
            wall_dt = min(clock.tick(60)/1000,.25)
            one_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    size = (max(640,event.w),max(360,event.h))
                    screen = pygame.display.set_mode(size,pygame.RESIZABLE,display=args.display)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused; accumulator = 0.
                    elif event.key == pygame.K_RIGHT and paused:
                        one_step = True
                    elif event.key in (pygame.K_PLUS,pygame.K_EQUALS,pygame.K_KP_PLUS):
                        speed = min(32.,speed*2)
                    elif event.key in (pygame.K_MINUS,pygame.K_KP_MINUS):
                        speed = max(.25,speed/2)
                    elif event.key in (pygame.K_r,pygame.K_n):
                        match.reset(match.seed+(event.key==pygame.K_n))
                        observer = MatchObserver(match,model)
                        paused,accumulator,event_offset = False,0.,0; captured.clear()
                    elif event.key == pygame.K_f:
                        observer.cycle()
                        event_offset = 0
                    elif event.key == pygame.K_m:
                        expanded = not expanded
                    elif event.key == pygame.K_PAGEUP:
                        event_offset = min(max(0,len(observer.memories[observer.focal].events)-1),event_offset+5)
                    elif event.key == pygame.K_PAGEDOWN:
                        event_offset = max(0,event_offset-5)
                    elif event.key in toggles:
                        key = toggles[event.key]; setattr(options,key,not getattr(options,key))
            if not running:
                break
            if one_step and not match.game.is_terminal:
                observer.step(match)
            elif not paused and not match.game.is_terminal and frames:
                accumulator += wall_dt*speed
                steps = 0
                while accumulator >= match.game.config.dt and steps < 32 and not match.game.is_terminal:
                    observer.step(match); accumulator -= match.game.config.dt; steps += 1
            game = match.game
            canvas.blit(renderer.draw(game,options,paused,speed,clock.get_fps()),(0,0))
            focal = observer.focal
            memory,belief = observer.memories[focal],observer.beliefs[focal]
            canvas.blit(panel.draw(memory,belief,observer.active[focal],expanded,event_offset),(1440,0))
            avatar_hud.draw(canvas,game.observe(focal))
            if size == canvas.get_size():
                screen.blit(canvas,(0,0))
            else:
                # Letterbox manual window resizing, preserving avatar proportions.
                fit = min(size[0]/1800,size[1]/900)
                scaled = pygame.transform.smoothscale(canvas,(round(1800*fit),round(900*fit)))
                screen.fill((8,13,22))
                screen.blit(scaled,((size[0]-scaled.get_width())//2,(size[1]-scaled.get_height())//2))
            pygame.display.set_caption(f'Phase 4 | seed {game.seed} | {game.time:.1f}s | {speed:g}x | F focal, M memory, 1 truth')
            pygame.display.flip()
            if args.capture:
                states = ['spawn'] if not frames else []
                if game.time >= 6:
                    states.append('exploration')
                if game.phase.value in ('discussion','voting','finished'):
                    states.append(game.phase.value)
                if memory.contradictions():
                    states.append('contradiction')
                if belief.entropy < .7:
                    states.append('strong-evidence')
                for state in states:
                    if state not in captured:
                        path = args.capture/f'seed-{game.seed}-{state}.png'
                        pygame.image.save(canvas,str(path))
                        print(f'Captured {path} | belief={belief.by_player}',flush=True)
                        captured.add(state)
            frames += 1
            if args.max_frames and frames >= args.max_frames or args.exit_on_finish and game.is_terminal:
                running = False
    finally:
        pygame.quit()
    return 0


if __name__ == '__main__':
    from phase3_cli import use_project_environment
    use_project_environment()
    raise SystemExit(main())
