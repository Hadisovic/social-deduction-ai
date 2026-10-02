"""Pygame artwork/blueprint renderer; all transforms leave physics unchanged."""
import math
import numpy as np
import pygame
from among_us_map import ASSET_DIR, ROOMS, polygons


class MapRenderer:
    def __init__(self, world, size=(1440,900)):
        self.world=world; self.size=size
        pygame.font.init()
        self.font=pygame.font.SysFont('segoeui',16)
        self.small=pygame.font.SysFont('segoeui',13)
        self.title=pygame.font.SysFont('segoeui',23,bold=True)
        w,h=size
        self.scale=min((w-32)/world.width,(h-145)/world.height)
        self.offset=((w-world.width*self.scale)/2,64+(h-145-world.height*self.scale)/2)
        source=pygame.image.load(str(ASSET_DIR/'reference_map.png'))
        self.art=pygame.transform.smoothscale(source,(round(world.width*self.scale),round(world.height*self.scale)))
        self.background=pygame.Surface(size)
        self.background.fill((8,13,22))
        self.background.blit(self.art,self.offset)

    def world_to_screen(self, point):
        left,_,_,top=self.world.bounds
        return (self.offset[0]+(point[0]-left)*self.scale,self.offset[1]+(top-point[1])*self.scale)

    def screen_to_world(self, point):
        left,_,_,top=self.world.bounds
        return (left+(point[0]-self.offset[0])/self.scale,top-(point[1]-self.offset[1])/self.scale)

    def point(self,p): return tuple(round(v) for v in self.world_to_screen(p))

    def line(self,surf,points,color,width=1,closed=False):
        if len(points)>1: pygame.draw.lines(surf,color,closed,[self.point(p) for p in points],width)

    def polygon(self,surf,shape,color):
        for poly in polygons(shape):
            pygame.draw.polygon(surf,color,[self.point(p) for p in poly.exterior.coords])
            for hole in poly.interiors:
                pygame.draw.polygon(surf,(14,23,36),[self.point(p) for p in hole.coords])

    def text(self,surf,text,pos,color=(226,237,249),font=None,center=False):
        img=(font or self.font).render(text,True,color)
        rect=img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
        surf.blit(img,rect)

    def character(self,surf,position,color):
        # Feet sit on the physical circle; body/visor are original display shapes.
        x,y=self.point(position); s=self.scale
        pygame.draw.ellipse(surf,(4,9,15),(x-.3*s,y-.11*s,.65*s,.28*s))
        body=pygame.Rect(round(x-.25*s),round(y-.69*s),max(8,round(.5*s)),max(14,round(.75*s)))
        pygame.draw.rect(surf,(12,20,28),body.inflate(4,4),border_radius=max(3,round(.15*s)))
        pygame.draw.rect(surf,color,body,border_radius=max(3,round(.15*s)))
        visor=pygame.Rect(round(x-.13*s),round(y-.59*s),max(7,round(.42*s)),max(5,round(.22*s)))
        pygame.draw.rect(surf,(163,225,240),visor,border_radius=4)
        pygame.draw.line(surf,(235,252,255),visor.topleft,(visor.right-3,visor.top),2)

    def draw(self,position,goal=None,target=None,route=(),collision=False,interactions=False,
             labels=False,rays=False,blueprint=False,fixtures=False):
        surf=self.background.copy()
        if blueprint:
            surf.fill((14,23,36))
            palette=[(43,72,84),(57,68,99),(72,64,80),(42,78,76)]
            self.polygon(surf,self.world.floor,(38,49,66))
            for i,(name,region) in enumerate(self.world.regions):
                if name in ROOMS:
                    self.polygon(surf,region.intersection(self.world.floor),palette[i%len(palette)])
        if collision or blueprint:
            for r in self.world.data['walls']:
                self.line(surf,r['points'],(255,166,77),2)
            for r in self.world.data['obstacles']:
                self.line(surf,r['points'],(255,102,120),2,r['closed'])
            for repair in self.world.data['seam_repairs']:
                self.line(surf,repair,(245,241,100),3)
            if collision:
                for poly in polygons(self.world.center_domain):
                    self.line(surf,list(poly.exterior.coords),(72,220,166),1)
                    for ring in poly.interiors:self.line(surf,list(ring.coords),(72,220,166),1)
        if interactions or blueprint:
            for d in self.world.data['doors']:
                color=(255,80,80) if d['id'] in self.world.closed_doors else (90,224,200)
                self.line(surf,d['points'],color,1,True)
            for vid,v in self.world.data['vents'].items():
                p=(v['position']['x'],v['position']['y']); x,y=self.point(p)
                pygame.draw.rect(surf,(74,233,206),(x-5,y-3,10,6),1)
                self.text(surf,vid,(x+5,y-14),font=self.small,color=(125,255,227))
                if blueprint:
                    for key in ('left','center','right'):
                        dest=v[key]
                        if dest is not None and int(vid)<dest:
                            other=self.world.data['vents'][str(dest)]['position']
                            self.line(surf,[p,(other['x'],other['y'])],(51,112,110))
            for d in self.world.destinations:
                pygame.draw.circle(surf,(255,219,111),self.point(d.position),3)
                pygame.draw.circle(surf,(116,185,255),self.point(d.standing),2)
            for c in self.world.data['cameras']:
                x,y=self.point(self.world.art_to_world(c['preview_position']))
                pygame.draw.rect(surf,(169,202,255),(x-5,y-3,10,6))
            # Spawn ring is a location marker; it does not spawn an actor inside a table.
            spawn=self.world.data['spawn']; p=spawn['initialSpawnCenter']
            pygame.draw.circle(surf,(144,164,224),self.point((p['x'],p['y'])),round(spawn['spawnRadius']*self.scale),1)
        if labels or blueprint:
            for name,region in self.world.regions:
                if name not in ROOMS:continue
                p=region.intersection(self.world.center_domain).representative_point()
                if p.is_empty:continue
                x,y=self.point(p.coords[0])
                img=self.small.render(name,True,(241,245,255))
                rect=img.get_rect(center=(x,y))
                pygame.draw.rect(surf,(20,29,43),rect.inflate(10,5),border_radius=3)
                surf.blit(img,rect)
        if route:self.line(surf,route,(87,229,255),2)
        if goal is not None:
            pygame.draw.circle(surf,(90,244,173),self.point(goal),max(5,round(self.scale*.24)),2)
        if fixtures:
            for room,color in zip(('Navigation','Electrical','MedBay','Storage'),
                                  ((229,78,91),(249,199,73),(188,133,239),(233,236,247))):
                p=next(d.standing for d in self.world.destinations if d.room==room)
                self.character(surf,p,color)
        if rays:
            for i in range(16):
                direction=(math.cos(i*math.tau/16),-math.sin(i*math.tau/16))
                dist=self.world.raycast(position,direction)
                end=np.array(position)+np.array(direction)*dist
                self.line(surf,[position,end],(102,205,248))
        self.character(surf,position,(63,206,214))
        if collision:
            pygame.draw.circle(surf,(240,249,255),self.point(position),round(self.world.radius*self.scale),1)
        w,h=self.size
        pygame.draw.rect(surf,(12,20,33),(0,0,w,56))
        self.text(surf,'THE SKELD',(22,12),font=self.title)
        self.text(surf,'MAP SIMULATION  /  NATIVE GAME COORDINATES',(200,20),color=(114,160,180),font=self.small)
        self.text(surf,'BLUEPRINT' if blueprint else 'REFERENCE ARTWORK',(w-185,20),color=(109,225,198),font=self.small)
        pygame.draw.rect(surf,(12,20,33),(0,h-79,w,79))
        self.text(surf,f'{self.world.region(position)}   |   x {position[0]:.3f}   y {position[1]:.3f}',(22,h-71))
        if target:self.text(surf,f'Target: {target}',(w//2,h-71),color=(141,231,192))
        self.text(surf,'WASD / arrows  move     Click  goal     Tab  destination     G  route     Space  follow route',(22,h-45),font=self.small)
        self.text(surf,'B  blueprint     C  collision     T  interactions     L  labels     R  rays     P  preview crew     Esc  quit',(22,h-24),font=self.small)
        return surf
