"""Rebuild the versioned Skeld blueprint from checked-in research sources.

Run from any directory. No network requests or executable third-party code.
Geometry is retained in native x-right/y-up game coordinates.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets' / 'skeld'
SOURCES = ASSETS / 'sources'
ROOM_NAMES = {'Cockpit': 'Navigation', 'LifeSupport': 'O2', 'RightEngine': 'Lower Engine',
              'LeftEngine': 'Upper Engine', 'Comms': 'Communications', 'Medical': 'MedBay'}


def points(path):
    return [[float(x), float(y)] for x, y in re.findall(r'(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)', path)]


def room(name):
    key = name.split('/')[1]
    return ROOM_NAMES.get(key, key)


def build():
    raw = json.loads((SOURCES / 'skeld.json').read_text())
    layers = raw['colliders']['layers']
    data = {'schema_version': 1, 'name': 'The Skeld', 'coordinates': 'native game units; x right, y up',
            'geometry_revision': 'a1acf39d1751a553141cc7153e9386e37199ee2d',
            'metadata_revision': 'b09c40b7e12d35c6cae996a26a943185b341c907',
            'metadata_game_version': '2026.8.18', 'geometry_game_version': 'unknown; dump committed 2021-09-05',
            'regions': [], 'walls': [], 'obstacles': [], 'visibility': [], 'doors': [], 'consoles': [],
            'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(SOURCES.glob('*.json'))}}
    for layer_id, layer in layers.items():
        for i, c in enumerate(layer['colliders']):
            supported = not re.search(r'[aAcCqQsStThHvV]', c['path'])
            entry = {'id': f'{layer_id}:{i}', 'name': c['name'], 'room': room(c['name']),
                     'points': points(c['path']), 'closed': c['path'].endswith('Z'),
                     'geometry_supported': supported, 'svg_path': c['path'],
                     'provenance': 'SkeldJS runtime dump'}
            name = c['name']
            if layer_id == '2':
                data['regions'].append(entry)
            elif layer_id == '10':
                data['visibility'].append(entry)
            elif layer_id == '9':
                data['obstacles' if entry['closed'] else 'walls'].append(entry)
            elif layer_id == '11':
                data['doors' if 'Door' in name else 'obstacles'].append(entry)
            elif layer_id == '12':
                if '/Table' in name or 'medBay_bed' in name or 'engine_railback' in name:
                    data['obstacles'].append(entry)
                if 'Console' in name or 'Scanner' in name:
                    ps = entry['points']
                    # Trigger bounds locate the object; standing destinations are derived later.
                    entry['position'] = ([(min(p[j] for p in ps) + max(p[j] for p in ps))/2 for j in (0, 1)]
                                         if supported else None)
                    data['consoles'].append(entry)
    # Source postprocessor's explicit repairs. Kept separate from measured chains.
    data['seam_repairs'] = [
        [[-17.709921,-1.654321],[-17.559524,-1.654321]],
        [[-10.317460,-.024691],[-9.920635,-.024691]],
        [[-8.730159,-.616284],[-8.382937,-.617284]],
        [[8.184524,-4.024691],[8.184524,-4.123457]],
        [[14.831349,-4.222222],[14.831349,-4.419753]],
        [[0,-7.925926],[.099206,-7.925926]],
        [[-16.220238,-1.703704],[-16.121032,-1.703704]],
        [[-1.835317,-4.518519],[-1.438492,-4.518519]],
    ]
    for key in ['vents', 'tasks', 'spawn', 'systems']:
        data[key] = json.loads((SOURCES / f'impostor-{key}.json').read_text())
    data['door_metadata'] = json.loads((SOURCES / 'impostor-doors.json').read_text())
    # Measured on the supplied 2048x1120 preview, not native game camera data.
    data['artwork'] = {'file': 'reference_map.png', 'size': [5792,3168],
        'landmarks_preview_size': [2048,1120],
        'vent_landmarks': {'0':[1317,713], '1':[1606,566], '2':[1389,305],
            '3':[796,631], '4':[572,186], '5':[680,585], '6':[760,470],
            '7':[1582,152], '8':[340,600], '9':[571,876], '10':[1611,897],
            '11':[295,435], '12':[1886,428], '13':[1886,565]},
        'confidence': 'artwork calibrated by vent landmarks; source artwork may be distorted'}
    data['cameras'] = [
        {'id':'camera_west','preview_position':[467,485], 'room':'CrossHallway'},
        {'id':'camera_north','preview_position':[907,199], 'room':'NorthHallway'},
        {'id':'camera_admin','preview_position':[1233,549], 'room':'AdminHallway'},
        {'id':'camera_east','preview_position':[1765,448], 'room':'BigYHallway'},
    ]
    for camera in data['cameras']:
        camera['confidence'] = 'image-estimated housing position; view coverage uncalibrated'
    data['utility_artwork_positions'] = {'MapRoomConsole':[1408,657],
        'EmergencyConsole':[1173,249], 'SwitchConsole':[852,708],
        'UpperHandConsole':[322,360], 'LowerHandConsole':[321,651]}
    # Task definitions omit consoles used by multi-stage minigames. Preserve
    # source bounds where available; explicitly label visual estimates elsewhere.
    native_stages = [
        ('GarbageConsole', 'EmptyGarbage', 'Cafeteria'),
        ('DataConsole', 'DownloadData', 'Cafeteria'),
        ('UploadDataConsole', 'DownloadData', 'Navigation'),
        ('UploadDataConsole', 'DownloadData', 'Weapons'),
        ('ChartCourseConsole', 'ChartCourse', 'Navigation'),
        ('gasCanConsole', 'FuelEngines', 'Storage'),
        ('FuelEngineConsole', 'FuelEngines', 'Lower Engine'),
        ('FuelEngineConsole', 'FuelEngines', 'Upper Engine'),
        ('AirlockConsole', 'EmptyGarbage/EmptyChute', 'Storage'),
    ]
    data['supplemental_task_stages'] = []
    for suffix, name, area in native_stages:
        console = next(c for c in data['consoles'] if c['name'].endswith('/'+suffix) and c['room'] == area)
        assert console['position'] is not None
        data['supplemental_task_stages'].append({'id': console['id'], 'name': name,
            'room': area, 'position': console['position'], 'radius': 1.5,
            'provenance': '2021 trigger bounds; interaction radius estimated'})
    visual_stages = [
        ('DownloadData', 'Electrical', [801,598]),
        ('DownloadData', 'Communications', [1371,877]),
        ('EmptyChute', 'O2', [1416,435]),
        ('AcceptDivertedPower', 'Security', [702,421]),
        ('AcceptDivertedPower', 'Lower Engine', [458,694]),
        ('AcceptDivertedPower', 'Upper Engine', [490,161]),
        ('AcceptDivertedPower', 'Weapons', [1693,219]),
        ('AcceptDivertedPower', 'Shields', [1663,716]),
        ('AcceptDivertedPower', 'Navigation', [1885,397]),
        ('AcceptDivertedPower', 'Communications', [1476,887]),
        ('AcceptDivertedPower', 'O2', [1565,397]),
    ]
    for i, (name, area, pixel) in enumerate(visual_stages):
        data['supplemental_task_stages'].append({'id': f'image:{i}', 'name': name,
            'room': area, 'preview_position': pixel, 'radius': 1.5,
            'provenance': 'image-estimated console and interaction radius; SVG circle lacks world transform'})
    # Approach the Comms desk from inside the room, not through its north wall.
    for stage in data['supplemental_task_stages']:
        if stage['room'] == 'Communications':
            stage['standing_preview'] = ([1371,925] if stage['name'] == 'DownloadData' else [1460,930])
    data['simulation_defaults'] = {'player_radius': .22, 'speed': 2.5, 'dt': 1/30,
        'ray_range': 5.5, 'grid_cell': .06, 'area_envelope_padding': .25,
        'confidence': 'player radius/speed are configurable approximations, not measured client physics'}
    path = ASSETS / 'among_us_map.json'
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(path)
    print({key:len(data[key]) for key in ['regions','walls','obstacles','visibility','doors','consoles','vents','tasks']})


if __name__ == '__main__':
    build()
