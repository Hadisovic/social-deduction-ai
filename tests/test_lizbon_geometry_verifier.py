"""The offline comparator must detect NN or physics changes, not just equal lengths."""
import gzip
import importlib.util
import json
from pathlib import Path


def verifier():
    path=Path(__file__).resolve().parents[1]/'tools/verify_lizbon_geometry_fix.py'
    spec=importlib.util.spec_from_file_location('geometry_verifier',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def files(root,changed_command=False,changed_physics=False):
    for name,count in (('normal',2),('geometry-evictions',3)):
        with gzip.open(root/(name+'.decisions.jsonl.gz'),'wt') as f:
            for i in range(count):
                f.write(json.dumps(dict(record='decision',tick=i*3,time=i*.6,
                    selected='kill' if changed_command and name!='normal' and i==1 else 'wait'))+'\n')
        with (root/(name+'.jsonl')).open('w') as f:
            for i in range(count):
                f.write(json.dumps(dict(record='neural',tick=i*3,choice='wait'))+'\n')
                f.write(json.dumps(dict(record='tick',tick=i*3,position=99 if changed_physics and name!='normal' else i))+'\n')


def test_equal_common_prefix_with_different_budget_tails(tmp_path):
    files(tmp_path);r=verifier().compare_prefixes(tmp_path)
    assert r['common_decisions']==2 and r['impostor_inputs_actions_equal'] and r['physics_equal']


def test_changed_neural_command_is_rejected(tmp_path):
    files(tmp_path,changed_command=True);r=verifier().compare_prefixes(tmp_path)
    assert not r['impostor_inputs_actions_equal']


def test_changed_physics_is_rejected_even_when_commands_match(tmp_path):
    files(tmp_path,changed_physics=True);r=verifier().compare_prefixes(tmp_path)
    assert r['impostor_inputs_actions_equal'] and not r['physics_equal']
