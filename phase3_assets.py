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


@lru_cache(maxsize=512)
def character_image(color, height, direction, frame, body=False):
    key = f'body:{color.lower()}' if body else f'player:{color.lower()}:{direction}:{frame % 4}'
    source = load_image(key)
    if source is None:
        return None
    import pygame
    # Preserve source bytes and aspect ratio; feet remain the renderer's anchor.
    bounds = source.get_bounding_rect()
    if bounds.height == 0:
        return None
    cropped = source.subsurface(bounds)
    target_height = max(10, round(height * .50)) if body else height
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
    print(f'Local pack verified: {len(entries)} PNG files ({count} downloaded) in {LOCAL}')
    print('Artwork attribution: Innersloth, via AI0702/Among-Us-clone. Local pack is ignored by Git.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', required=True)
    parser.parse_args()
    install()
