"""Separate high-fidelity Skeld sandbox, adapted from skeld_environment.py.

Run: python among_us_map_simulation.py
Export: python among_us_map_simulation.py --screenshots docs/images
Gymnasium: AmongUsMapEnv(render_mode=None), same 22-value observation layout and
2-value screen-oriented movement action as SkeldNavEnv. World info uses native
game units (x right, y up). Existing legacy scripts and checkpoints are untouched.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pygame
import shapely
import gymnasium as gym
from gymnasium import spaces
from among_us_map import get_map, ROOMS
from among_us_renderer import MapRenderer


class AmongUsMapEnv(gym.Env):
    metadata = {'render_modes':['human','rgb_array'], 'render_fps':30}

    def __init__(self, render_mode=None, task_goal_mode=True, goal_mode=None,
                 player_radius=None, speed=2.5, max_episode_time=60., render_size=(1440,900)):
        super().__init__()
        if render_mode not in (None,'human','rgb_array'):
            raise ValueError('Unsupported render_mode')
        if not math.isfinite(speed) or speed<=0 or not math.isfinite(max_episode_time) or max_episode_time<=0:
            raise ValueError('Speed and episode duration must be positive and finite')
        self.map=get_map(player_radius)
        self.render_mode=render_mode
        self.goal_mode=goal_mode or ('task' if task_goal_mode else 'room_to_room')
        self.speed=float(speed); self.max_episode_time=float(max_episode_time)
        self.render_size=render_size; self.dt=1/30
        low=np.zeros(22,dtype=np.float32); low[2:4]=-1
        self.observation_space=spaces.Box(low,np.ones(22,dtype=np.float32),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(2,),dtype=np.float32)
        self._renderer=None; self._screen=None; self._finished=True
        self._player_x=self._player_y=self._goal_x=self._goal_y=0.
        self._room_nodes = {}

    @property
    def position(self): return (self._player_x,self._player_y)

    @property
    def goal(self): return (self._goal_x,self._goal_y)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        options=options or {}
        mode=options.get('goal_mode',self.goal_mode)
        if mode not in ('task','task_to_task','room_to_room'):
            raise ValueError('Unknown goal_mode')
        index=int(self.np_random.integers(len(self.map.destinations)))
        target=self.map.destinations[index]
        self._target_task=target if mode!='room_to_room' else None
        if mode=='task_to_task':
            starts=[d for d in self.map.destinations if d.id!=target.id and math.dist(d.standing,target.standing)>2]
            start=starts[int(self.np_random.integers(len(starts)))].standing
        else:
            start=tuple(self.map.nodes[int(self.np_random.integers(len(self.map.nodes)))])
        if mode=='room_to_room':
            if not self._room_nodes:
                for name, shape in self.map.regions:
                    if name in ROOMS:
                        self._room_nodes[name] = self.map.nodes[shapely.contains_xy(
                            shape, self.map.nodes[:,0], self.map.nodes[:,1])]
            start_room = ROOMS[int(self.np_random.integers(len(ROOMS)))]
            nodes = self._room_nodes[start_room]
            start = tuple(options.get('start', nodes[int(self.np_random.integers(len(nodes)))]))
            other_rooms = [r for r in ROOMS if r != self.map.region(start)]
            target_room = other_rooms[int(self.np_random.integers(len(other_rooms)))]
            goals = self._room_nodes[target_room]
            goal = tuple(goals[int(self.np_random.integers(len(goals)))])
        else: goal=target.standing
        start=tuple(options.get('start',start)); goal=tuple(options.get('goal',goal))
        if not self.map.contains(start) or not self.map.contains(goal):
            raise ValueError('Start and goal must have full player clearance')
        if 'goal' in options:self._target_task=None
        self._player_x,self._player_y=start; self._goal_x,self._goal_y=goal
        self._elapsed=0.; self._step_idx=0; self._total_path_length=0.
        self._best_dist=math.dist(start,goal); self._last_prog_check=0.
        self._blocked_steps=0; self._stuck_flag=0.; self._stag_progress=0.
        self._stag_anchor=start; self._stag_steps=0; self._stag_armed=True
        self._finished=False
        return self._build_observation(),self._info()

    def _info(self):
        return {'player_pos':self.position,'goal_pos':self.goal,'elapsed':self._elapsed,
                'distance':math.dist(self.position,self.goal),'total_path_len':self._total_path_length,
                'room':self.map.region(self.position),'current_region':self.map.region(self.position),
                'target_task':self._target_task.name if self._target_task else None,
                'stuck':self._stuck_flag,'stagnation':self._stag_progress}

    def step(self,action):
        if self._finished: raise RuntimeError('Call reset before step / after episode completion')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all(): raise ValueError('Expected two finite movement values')
        action=np.clip(action,-1,1); mag=float(np.linalg.norm(action))
        direction=action/mag if mag>=.05 else np.zeros(2)
        direction[1]*=-1  # Preserve existing action convention: positive dy moves down.
        old=self.position
        new,collided=self.map.move(old,direction*self.speed*self.dt)
        self._player_x,self._player_y=new
        moved=math.dist(old,new); self._total_path_length+=moved
        self._elapsed+=self.dt; self._step_idx+=1
        blocked=collided and mag>=.05 and moved<.2*self.speed*self.dt
        self._blocked_steps=self._blocked_steps+1 if blocked else max(0,self._blocked_steps-1)
        self._stuck_flag=float(self._blocked_steps>=15)
        if math.dist(self._stag_anchor,new)<.375:self._stag_steps+=1
        else:self._stag_anchor=new; self._stag_steps=0; self._stag_armed=True
        self._stag_progress=min(1.,self._stag_steps/90)
        reward=-.01; dist=math.dist(new,self.goal)
        if self._elapsed-self._last_prog_check>=.5:
            improvement=self._best_dist-dist
            if improvement>=.125:reward+=1.2*improvement; self._best_dist=dist
            self._last_prog_check=self._elapsed
        if self._stag_steps>=90 and self._stag_armed:reward-=25.; self._stag_armed=False
        # Reach a walkable standing destination. Never succeed through a wall.
        terminated=dist<=.3 and self.map.segment_clear(new,self.goal)
        truncated=not terminated and self._elapsed>=self.max_episode_time
        if terminated: reward+=100.
        elif truncated:reward-=50.
        self._finished=bool(terminated or truncated)
        return self._build_observation(),float(reward),bool(terminated),bool(truncated),self._info()

    def _build_observation(self):
        m=self.map; left,bottom,right,top=m.bounds
        obs=np.zeros(22,dtype=np.float32)
        obs[:2]=((self._player_x-left)/m.width,(top-self._player_y)/m.height)
        obs[2:4]=np.clip([(self._goal_x-self._player_x)/m.width,(self._player_y-self._goal_y)/m.height],-1,1)
        for i in range(16):
            d=(math.cos(i*math.tau/16),-math.sin(i*math.tau/16))
            obs[4+i]=self.map.raycast(self.position,d)/5.5
        obs[20:]=self._stuck_flag,self._stag_progress
        return obs

    @property
    def renderer(self):
        if self._renderer is None:self._renderer=MapRenderer(self.map,self.render_size)
        return self._renderer

    def render(self):
        if self.render_mode is None:return None
        surf=self.renderer.draw(self.position,self.goal,self._target_task.name if self._target_task else None)
        if self.render_mode=='rgb_array':return np.transpose(pygame.surfarray.array3d(surf),(1,0,2))
        if self._screen is None:self._screen=pygame.display.set_mode(self.render_size)
        self._screen.blit(surf,(0,0)); pygame.event.pump(); pygame.display.flip()
        return None

    def close(self):
        if self._screen is not None:pygame.display.quit(); self._screen=None
        self._renderer=None


# Explicit import alias for clients migrating only their environment import.
AmongUsNavEnv=AmongUsMapEnv


def screenshots(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    env=AmongUsMapEnv(render_mode='rgb_array',render_size=(1800,1125))
    env.reset(seed=7,options={'start':(-.7,-2.8)})
    target=next(d for d in env.map.destinations if d.name=='CalibrateDistributor')
    route=env.map.astar(env.position,target.standing)
    for name,flags in [('overview',{}),('collision',{'collision':True}),
                       ('interactions',{'interactions':True,'labels':True}),
                       ('blueprint',{'blueprint':True}),('rays',{'rays':True,'collision':True})]:
        surf=env.renderer.draw(env.position,target.standing,target.name,route if name=='blueprint' else (),**flags)
        path=directory/f'among_us_map_{name}.png'; pygame.image.save(surf,str(path)); print(path)
    env.close()


def inspect():
    pygame.init()
    env=AmongUsMapEnv(); env.reset(seed=7,options={'start':(-.7,-2.8)})
    screen=pygame.display.set_mode(env.render_size); pygame.display.set_caption('Among Us Map Simulation - The Skeld')
    clock=pygame.time.Clock(); index=0; route=[]; following=False; show_route=True
    flags={k:False for k in ('collision','interactions','labels','rays','blueprint','fixtures')}
    keys={pygame.K_c:'collision',pygame.K_t:'interactions',pygame.K_l:'labels',pygame.K_r:'rays',pygame.K_b:'blueprint',pygame.K_p:'fixtures'}
    running=True
    while running:
        dt=min(clock.tick(60)/1000,.05)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:running=False
            elif event.type==pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,pygame.K_q):running=False
                elif event.key in keys:flags[keys[event.key]]=not flags[keys[event.key]]
                elif event.key==pygame.K_g:show_route=not show_route
                elif event.key==pygame.K_SPACE:
                    route=env.map.astar(env.position,env.goal); following=bool(route)
                elif event.key in (pygame.K_TAB,pygame.K_n):
                    index=(index+1)%len(env.map.destinations); d=env.map.destinations[index]
                    env._target_task=d; env._goal_x,env._goal_y=d.standing
                    route=env.map.astar(env.position,env.goal); following=False
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                goal=env.renderer.screen_to_world(event.pos)
                if env.map.contains(goal):
                    env._goal_x,env._goal_y=goal; env._target_task=None
                    route=env.map.astar(env.position,goal); following=False
        held=pygame.key.get_pressed()
        delta=np.array([int(held[pygame.K_d] or held[pygame.K_RIGHT])-int(held[pygame.K_a] or held[pygame.K_LEFT]),
                        int(held[pygame.K_w] or held[pygame.K_UP])-int(held[pygame.K_s] or held[pygame.K_DOWN])],dtype=float)
        if np.linalg.norm(delta):
            following=False; route=[]; delta*=env.speed*dt/np.linalg.norm(delta)
        elif following:
            while route and math.dist(env.position,route[0])<.015:route.pop(0)
            if route:
                diff=np.array(route[0])-env.position; length=np.linalg.norm(diff)
                delta=diff*min(1.,env.speed*dt/length)
            else:following=False
        if np.linalg.norm(delta):env._player_x,env._player_y=env.map.move(env.position,delta)[0]
        name=env._target_task.name if env._target_task else 'Custom destination'
        surf=env.renderer.draw(env.position,env.goal,name,route if show_route else (),**flags)
        screen.blit(surf,(0,0));pygame.display.flip()
    env.close();pygame.quit()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshots',metavar='DIRECTORY',help='Export headless renderer evidence and exit')
    args=parser.parse_args()
    if args.screenshots:screenshots(args.screenshots)
    else:inspect()
