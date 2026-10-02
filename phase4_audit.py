"""Explicit acceptance audit for registered data and repeatable actor extraction."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from belief.features import FEATURE_NAMES, VIEWS
from phase4_data import generate_match, source_fingerprint


def audit_dataset(directory, replay_seeds=(10000,10017,20000,30000,40000)):
    directory = Path(directory)
    config = json.loads((directory/'config.json').read_text())
    if config['sources'] != source_fingerprint():
        raise ValueError('Generated data source fingerprint differs from current extraction code')
    seen = set(); counts = {}; replays = []
    for split,(start,count) in config['splits'].items():
        paths = list((directory/split).glob('*.npz'))
        expected = set(range(start,start+count))
        assert {int(p.stem) for p in paths} == expected
        assert not expected&seen; seen |= expected
        samples = 0
        for path in paths:
            metadata = json.loads(path.with_suffix('.json').read_text())
            with np.load(path,allow_pickle=False) as source:
                data = {k:source[k] for k in source.files}
            n = len(data['y']); samples += n
            assert metadata['focal_actors']==4
            assert n==metadata['samples'] and all(len(v)==n for v in data.values())
            assert (data['match']==int(path.stem)).all()
            assert set(data['actor']) == set(metadata['colors'])-{metadata['impostor_id']}
            assert np.all(data['candidates'][np.arange(n),data['y']]==metadata['impostor_id'])
            assert 'finished' not in data['stage']
            assert (data['tick']*.2 < metadata['result']['simulation_time']).all()
            assert metadata['metrics']['illegal_actions']==metadata['metrics']['navigation_failures']==0
            assert metadata['result']['reason'] != 'timeout'
            expected_family = {'patient'} if split=='heldout' else {'hunter','self_report'}
            assert metadata['impostor_style'] in expected_family
            for view in VIEWS:
                assert data['x_'+view].shape == (n,4,len(FEATURE_NAMES))
                assert np.isfinite(data['x_'+view]).all()
            if int(path.stem) in replay_seeds:
                replay, record = generate_match(int(path.stem),split=='heldout')
                assert replay.keys()==data.keys()
                assert all(np.array_equal(replay[k],data[k]) for k in data)
                assert record['trajectory_hash']==metadata['trajectory_hash']
                replays.append(int(path.stem))
        counts[split]={'matches':count,'samples':samples}
    return {'status':'passed','split_counts':counts,'exact_match_and_feature_replays':sorted(replays),
            'source_hashes_match':True,'overlapping_matches':0,'invalid_samples':0,
            'terminal_samples':0,'navigation_failures':0,'illegal_actions':0}
