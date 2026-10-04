"""Optional, pinned local artwork pack. No artwork is bundled or downloaded on import.

Install explicitly with ``python phase3_assets.py --install``. See
docs/phase3_asset_provenance.md for third-party artwork provenance.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / 'assets' / 'phase3_assets.json'
LOCAL = ROOT / 'assets' / 'phase3_local'


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


@lru_cache(maxsize=256)
def load_image(key):
    """Return a checked image surface, or None for the original-art fallback."""
    entry = manifest()['files'].get(key)
    if entry is None:
        return None
    path = LOCAL / entry['path']
    if not path.is_file():
        return None
    data = path.read_bytes()
    if blob_hash(data) != entry['git_blob_sha1']:
        return None
    import io
    import pygame
    source = pygame.image.load(io.BytesIO(data), entry['path'])
    # Some upstream panels are palette PNGs. Normalize without convert_alpha(),
    # which would require an initialized display and break offline unit tests.
    surface = pygame.Surface(source.get_size(), pygame.SRCALPHA, 32)
    surface.blit(source, (0, 0))
    return surface


def blob_hash(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


MATERIAL_PALETTES = {
    'red': ((198,17,17), (122,8,56)),
    'purple': ((107,47,188), (59,23,124)),
    'black': ((63,71,78), (30,31,38)),
    'pink': ((238,84,187), (172,43,174)),
    'white': ((215,225,241), (132,149,192)),
    'brown': ((113,73,30), (94,38,21)),
}


def resolve_red_palette(source, color='red'):
    """Decode the upstream red RGB material-mask into suit/shadow/visor colors.

    Other colors already contain their display palette. Work on a surface copy;
    pinned PNG bytes, alpha, outlines and neutral bone highlights stay intact.
    """
    import numpy as np
    import pygame
    result = source.copy()
    pixels = pygame.surfarray.pixels3d(result)
    original = pixels.astype(np.float32)
    suit, shadow = MATERIAL_PALETTES[color]
    for channel, palette in enumerate((suit, (148,201,219), shadow)):
        others = [i for i in range(3) if i != channel]
        selected = ((original[:,:,channel] > 1.6*original[:,:,others[0]]) &
                    (original[:,:,channel] > 1.6*original[:,:,others[1]]))
        pixels[selected] = np.rint(original[:,:,channel][selected,None]/255 * np.array(palette)).astype(np.uint8)
    del pixels
    return result


@lru_cache(maxsize=512)
def character_image(color, height, direction, frame, body=False):
    color = color.lower()
    key = f'body:{color.lower()}' if body else f'player:{color.lower()}:{direction}:{frame % 4}'
    # Five upstream colors alias every walking frame to the same standing PNG.
    # Reuse the complete material-mask animation with that color's display palette.
    recolor = color == 'red'
    if not body and color in MATERIAL_PALETTES:
        hashes = {manifest()['files'].get(f'player:{color}:{direction}:{i}', {}).get('git_blob_sha1') for i in range(4)}
        if len(hashes) == 1:
            key = f'player:red:{direction}:{frame % 4}'
            recolor = True
    source = load_image(key)
    if source is None:
        return None
    import pygame
    if recolor:
        source = resolve_red_palette(source, color)
    # Preserve source bytes and aspect ratio; feet remain the renderer's anchor.
    bounds = source.get_bounding_rect()
    if bounds.height == 0:
        return None
    cropped = source.subsurface(bounds)
    target_height = max(18, round(height * .65)) if body else height
    return pygame.transform.smoothscale(cropped, (max(1, round(bounds.width * target_height / bounds.height)), target_height))


@lru_cache(maxsize=32)
def task_image(task_name, width=108, height=108):
    source = load_image('task:' + task_name)
    if source is None:
        return None
    import pygame
    scale = min(width / source.get_width(), height / source.get_height())
    return pygame.transform.smoothscale(source, (max(1, round(source.get_width() * scale)),
                                               max(1, round(source.get_height() * scale))))


def install():
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import quote
    from urllib.request import Request, urlopen
    data = manifest()
    def fetch(entry):
        relative = entry['path']
        target = LOCAL / relative
        if target.is_file() and blob_hash(target.read_bytes()) == entry['git_blob_sha1']:
            return False
        url = data['raw_base'] + quote(relative, safe='/')
        with urlopen(Request(url, headers={'User-Agent': 'social-deduction-ai-asset-installer'}), timeout=60) as response:
            image = response.read()
        if not image.startswith(b'\x89PNG\r\n\x1a\n') or blob_hash(image) != entry['git_blob_sha1']:
            raise ValueError(f'Pinned PNG validation failed: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.png.tmp')
        temporary.write_bytes(image)
        temporary.replace(target)
        return True
    # Manifest aliases may share a file; write each target exactly once.
    entries = {entry['path']: entry for entry in data['files'].values()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        count = sum(pool.map(fetch, entries.values()))
    load_image.cache_clear()
    character_image.cache_clear()
    task_image.cache_clear()
    print(f'Local pack verified: {len(entries)} PNG files ({count} downloaded) in {LOCAL}')
    print('Artwork attribution: Innersloth, via AI0702/Among-Us-clone. Local pack is ignored by Git.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', required=True)
    parser.parse_args()
    install()
