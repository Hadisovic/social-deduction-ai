"""Known upstream bug and proposed in-memory correction; never trains."""
import os
import importlib.util
from pathlib import Path
import pytest


def test_independent_geometry_cache_eviction_reproduces_guard_and_pairing_repairs_it():
    source=os.environ.get('JANE_UPSTREAM_REVIEW_ROOT')
    if not source:pytest.skip('Explicit pinned source required in an isolated invocation')
    path=Path(__file__).resolve().parents[1]/'tools/reproduce_lizbon_geometry_cache.py'
    spec=importlib.util.spec_from_file_location('geometry_diagnostic',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    r=module.reproduce(source)
    assert r['mismatch_reproduced'] and r['equivalent_collision_domains'] and r['proposed_pairing_passed']
    assert not r['upstream_source_modified']
