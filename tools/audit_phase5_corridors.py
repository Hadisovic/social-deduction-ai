"""Read-only movement audit at the two requested Skeld passages. No training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shapely.geometry import LineString, Point
from among_us_map import get_map, ASSET_DIR
from navigation_service import NavigationPlanner, NavigationService, NavTarget, NavStatus


def audit():
    world=get_map(); planner=NavigationPlanner(world)
    def point(p): return world.nearest_node(p)
    passages=[('upper_engine_south',point((-17.,-.8)),point((-17.,-6.5))),
              ('big_y_o2',point((9.4,.0)),point((9.4,-9.5)))]
    cases=[]
    for name,a,b in passages:
        for reverse in (False,True):
            start,goal=(b,a) if reverse else (a,b)
            for dt in (1/30,.2):
                nav=NavigationService(planner); nav.navigate(start,NavTarget.location(goal,.08))
                pos=start; minimum=Point(pos).distance(world.barriers); valid=True
                for step in range(math.ceil(60/dt)+1):
                    if nav.status is not NavStatus.MOVING: break
                    delta=nav.command(pos,dt)
                    if nav.awaiting_feedback:
                        nxt,collided=world.move(pos,delta)
                        valid=valid and world.segment_clear(pos,nxt) and world.contains(nxt)
                        if math.dist(pos,nxt)>0:
                            minimum=min(minimum,LineString([pos,nxt]).distance(world.barriers))
                        nav.feedback(nxt,collided); pos=nxt
                cases.append({'passage':name,'reverse':reverse,'dt':dt,'start':start,'goal':goal,
                    'status':nav.status.value,'success':nav.status is NavStatus.SUCCESS,
                    'swept_segments_valid':valid,'collisions':nav.collisions,'replans':nav.replans,
                    'steps':nav.steps,'minimum_center_to_barrier':minimum,
                    'exact_circle_clearance':minimum>=world.radius-1e-9,
                    'minimum_margin_over_radius':minimum-world.radius})
    return {'schema':'phase5-corridor-audit-v1','date':'2026-10-03',
        'map_sha256':hashlib.sha256((ASSET_DIR/'among_us_map.json').read_bytes()).hexdigest(),
        'radius':world.radius,'doors':len(world.data['doors']),
        'closed_doors':sorted(world.closed_doors),'cases':cases,
        'simulation_runs_passed':all(c['success'] and c['swept_segments_valid'] and c['collisions']==0 for c in cases),
        'exact_circle_clearance_passed':all(c['exact_circle_clearance'] for c in cases),
        'interpretation':'Route clearance is not corridor width. Polygonal buffer approximation can slightly undercut exact circle clearance; flag separately. No live-game parity claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs'/'phase5_corridor_audit.json')
    args=parser.parse_args(); result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['simulation_runs_passed'] and result['exact_circle_clearance_passed'] else 1)
