"""Small immediate-mode UI kit with real hitboxes, clipping, hover and focus."""
from dataclasses import dataclass
from functools import lru_cache
import pygame

BG=(11,15,21);PANEL=(17,24,33);CARD=(24,33,44);HOVER=(33,46,61);LINE=(43,57,73)
TEXT=(231,238,245);MUTED=(151,166,185);DIM=(101,119,141)
LIME=(185,245,106);CYAN=(112,217,245);AMBER=(249,204,116);RED=(249,113,121);PURPLE=(207,171,255)
SCALE=1.


def set_scale(value):
    global SCALE
    if SCALE!=value:SCALE=value;font.cache_clear()


@lru_cache(maxsize=80)
def font(size=15,bold=False,mono=False):return pygame.font.SysFont('consolas' if mono else 'segoeui',max(10,int(size*SCALE)),bold=bold)


def text(surface,value,x,y,size=15,color=TEXT,bold=False,width=None,mono=False):
    f=font(size,bold,mono);original=str(value);value=original
    if width is not None:
        while value and f.size(value)[0]>width:value=value[:-1]
        if value!=original:value=value[:-2]+'…' if len(value)>2 else ''
    image=f.render(value,True,color);surface.blit(image,(int(x),int(y)));return image.get_height()


def wrap(surface,value,rect,size=14,color=MUTED,line_height=None):
    f=font(size);line_height=line_height or size+7;y=rect.y
    for paragraph in str(value).split('\n'):
        words=paragraph.split();line=''
        for word in words:
            if line and f.size(line+' '+word)[0]>rect.w:
                if y+line_height>rect.bottom:return y
                text(surface,line,rect.x,y,size,color);y+=line_height;line=word
            else:line=(line+' '+word).strip()
        if line and y+line_height<=rect.bottom:text(surface,line,rect.x,y,size,color);y+=line_height
    return y


def box(surface,rect,color=PANEL,border=LINE,radius=10):
    pygame.draw.rect(surface,color,rect,border_radius=radius)
    if border:pygame.draw.rect(surface,border,rect,1,border_radius=radius)


def progress(surface,rect,value,color=LIME):
    pygame.draw.rect(surface,(41,53,66),rect,border_radius=max(1,rect.h//2))
    if value is not None and value>0:
        fill=rect.copy();fill.w=max(rect.h,round(rect.w*max(0,min(1,value))))
        pygame.draw.rect(surface,color,fill,border_radius=max(1,rect.h//2))


def scrollbar(surface,rect,offset,maximum):
    if maximum<=0:return
    track=pygame.Rect(rect.right-4,rect.y,3,rect.h)
    pygame.draw.rect(surface,LINE,track,border_radius=1)
    length=max(25,round(rect.h*rect.h/(rect.h+maximum)))
    thumb=pygame.Rect(track.x,track.y+round((track.h-length)*min(1,max(0,offset)/maximum)),3,length)
    pygame.draw.rect(surface,CYAN,thumb,border_radius=1)


@dataclass
class Control:
    key:str
    rect:pygame.Rect
    action:object
    tooltip:str=''
    enabled:bool=True


class Widgets:
    def __init__(self):self.controls=[];self.mouse=(0,0);self.focus=None;self.hover=None;self.last_hover_time=0.

    def begin(self,mouse):self.controls=[];self.mouse=mouse;self.hover=None

    def hit(self,key,rect,action,tooltip='',enabled=True,surface=None):
        area=rect.copy()
        if surface is not None:area=area.clip(surface.get_clip())
        if area.w>0 and area.h>0:
            self.controls.append(Control(key,area,action,tooltip,enabled))
            if area.collidepoint(self.mouse):self.hover=self.controls[-1]

    def button(self,surface,key,label,rect,action,active=False,accent=LIME,tooltip='',enabled=True,size=14):
        hovered=rect.collidepoint(self.mouse);color=HOVER if hovered else CARD
        if active:color=tuple(max(0,int(v*.16)) for v in accent)
        box(surface,rect,color,accent if active or self.focus==key else LINE,7)
        f=font(size,active);label_color=accent if active else TEXT if enabled else DIM
        image=f.render(str(label),True,label_color)
        old=surface.get_clip();surface.set_clip(rect.clip(old));surface.blit(image,image.get_rect(center=rect.center));surface.set_clip(old)
        self.hit(key,rect,action,tooltip,enabled,surface)

    def toggle(self,surface,key,label,value,rect,action,tooltip=''):
        self.hit(key,rect,action,tooltip,surface=surface)
        if rect.collidepoint(self.mouse):pygame.draw.rect(surface,HOVER,rect,border_radius=7)
        text(surface,label,rect.x+8,rect.y+10,15,width=rect.w-78)
        track=pygame.Rect(rect.right-56,rect.y+10,44,24)
        pygame.draw.rect(surface,(66,94,45) if value else (51,63,78),track,border_radius=12)
        pygame.draw.circle(surface,LIME if value else MUTED,(track.right-12 if value else track.left+12,track.centery),8)

    def tooltip(self,surface):
        if self.hover is None or not self.hover.tooltip:return
        content=self.hover.tooltip;w=min(450,font(13).size(content)[0]+22);h=68 if len(content)>60 else 40
        x=min(surface.get_width()-w-10,self.mouse[0]+14);y=min(surface.get_height()-h-8,self.mouse[1]+22)
        r=pygame.Rect(max(5,x),max(5,y),w,h);box(surface,r,(36,47,62),CYAN,7)
        wrap(surface,content,r.inflate(-16,-12),13,TEXT,18)

    def click(self,point):
        for item in reversed(self.controls):
            if item.rect.collidepoint(point):
                self.focus=item.key;return item.action if item.enabled else None
        return None

    def focus_next(self,reverse=False):
        controls=[c for c in self.controls if c.enabled]
        if not controls:return
        keys=[c.key for c in controls];i=keys.index(self.focus) if self.focus in keys else -1
        self.focus=keys[(i+(-1 if reverse else 1))%len(keys)]

    def activate(self):return next((c.action for c in self.controls if c.key==self.focus and c.enabled),None)
