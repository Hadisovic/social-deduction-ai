"""Validated local preferences. Never writes into research/replay directories."""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULTS = dict(version=2, omniscient=False, reveal_roles=False, reveal_ballots=False,
    neural=True, rewards=True, evidence=True, unavailable=True, labels=True,
    tasks=True, bodies=True, visibility=False, trails=False, collisions=False,
    follow=False, interpolate=True, reduced_motion=False, rooms=False,
    autoplay=True, speed=1., mode='manual', scale=1., density='comfortable',
    left_width=252, right_width=370, roster=True, inspector=True, theme='ink',
    default_player='', event_filter='All', event_player='All', timeline_all=False, favorites=[],
    bookmarks=[], recent=[], watch_dir='')


class Preferences:
    def __init__(self, path):
        self.path=Path(path); self.values=deepcopy(DEFAULTS);self.error=''
        try:
            raw=json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(raw,dict):raise ValueError('Expected a settings object')
            for key,value in raw.items():self.set(key,value)
        except FileNotFoundError:pass
        except (ValueError,OSError,TypeError) as exc:self.error=str(exc)

    def __getitem__(self,key):return self.values[key]

    def set(self,key,value):
        if key not in DEFAULTS:return False
        default=DEFAULTS[key]
        if isinstance(default,bool):
            if not isinstance(value,bool):return False
        elif isinstance(default,(int,float)):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):return False
            bounds={'speed':(.25,16),'scale':(.85,1.3),'left_width':(220,400),'right_width':(300,550),'version':(2,2)}
            low,high=bounds[key];value=max(low,min(high,value))
            if isinstance(default,int):value=int(value)
        elif isinstance(default,list):
            if not isinstance(value,list):return False
            if key=='bookmarks':value=[v for v in value[-200:] if isinstance(v,dict) and isinstance(v.get('path'),str) and isinstance(v.get('time'),(int,float)) and math.isfinite(v['time'])]
            else:value=[v for v in value[-1000:] if isinstance(v,str)]
        elif not isinstance(value,str):return False
        if key=='mode' and value not in ('manual','auto_next','follow_latest'):return False
        if key=='density' and value not in ('comfortable','compact'):return False
        if key=='theme' and value not in ('ink','midnight'):return False
        self.values[key]=value;return True

    def save(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(self.values,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            temp.replace(self.path);self.error='';return True
        except OSError as exc:self.error=str(exc);return False

    def reset(self):self.values=deepcopy(DEFAULTS);return self.save()

    def favorite(self,path):
        items=self.values['favorites']
        if path in items:items.remove(path)
        else:items.append(path)
        self.save()

    def bookmark(self,path,when,player=None):
        record=dict(path=str(path),time=round(float(when),3))
        if player:record['player']=str(player)
        self.values['bookmarks'].append(record)
        self.values['bookmarks']=self.values['bookmarks'][-200:];self.save()

    def remember(self,path):
        self.values['recent']=([str(path)]+[p for p in self.values['recent'] if p!=str(path)])[:30]
        self.save()
