"""Optional artwork must not break an offline clone or affect simulation state."""
import json
from pathlib import Path, PurePosixPath

import pygame
import pytest

import phase3_assets as assets


def test_manifest_is_pinned_and_contains_only_selected_pngs():
    manifest = assets.manifest()
    assert manifest['commit'] in manifest['raw_base']
    assert len(manifest['commit']) == 40
    assert len({entry['path'] for entry in manifest['files'].values()}) == 119
    for key, entry in manifest['files'].items():
        path = PurePosixPath(entry['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert path.suffix == '.png' and len(entry['git_blob_sha1']) == 40
        assert key.startswith(('player:', 'body:', 'task:'))


@pytest.fixture
def local_pack(tmp_path, monkeypatch):
    image = pygame.Surface((20, 30), pygame.SRCALPHA)
    pygame.draw.rect(image, (200, 60, 80), (5, 5, 10, 20))
    target = tmp_path / 'sprite.png'
    pygame.image.save(image, str(target))
    entry = {'path': target.name, 'git_blob_sha1': assets.blob_hash(target.read_bytes())}
    monkeypatch.setattr(assets, 'LOCAL', tmp_path)
    monkeypatch.setattr(assets, 'manifest', lambda: {'files': {'player:red:right:0': entry}})
    assets.load_image.cache_clear()
    assets.character_image.cache_clear()
    yield target
    assets.load_image.cache_clear()
    assets.character_image.cache_clear()


def test_verified_local_sprite_preserves_aspect_and_missing_art_returns_fallback(local_pack):
    sprite = assets.character_image('Red', 40, 'right', 0)
    assert sprite.get_size() == (20, 40)
    assert assets.character_image('Blue', 40, 'right', 0) is None
    assert assets.task_image('UnknownTask') is None


def test_modified_local_bytes_are_not_loaded(local_pack):
    local_pack.write_bytes(b'not a pinned PNG')
    assert assets.character_image('Red', 40, 'right', 0) is None


def test_palette_png_is_normalized_before_smooth_scaling(local_pack):
    palette = pygame.Surface((12, 16), depth=8)
    palette.set_palette([(i, 0, 0) for i in range(256)])
    palette.fill(180)
    pygame.image.save(palette, str(local_pack))
    assets.manifest()['files']['player:red:right:0']['git_blob_sha1'] = assets.blob_hash(local_pack.read_bytes())
    assert assets.load_image('player:red:right:0').get_bitsize() == 32
    assert assets.character_image('Red', 40, 'right', 0).get_height() == 40
