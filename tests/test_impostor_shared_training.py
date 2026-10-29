"""PPO signal, ownership, terminal credit, and fixed selection/partition regressions."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from impostor_shared.weights import ASSETS, load_transfer, tensor_hash
from impostor_shared.checkpoints import optimizer_for
from impostor_shared.training import transition_batch, ppo_update, check_sources, source_signature
from impostor_shared.study import TRAIN, SELECT, CONFIRM, STREAMS, select_candidate, write_json


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def fixture(self):
        policy,_=load_transfer(trainable=True)
        with np.load(ASSETS/'synthetic-parity.npz',allow_pickle=False) as data:
            packets=[{k:data[f'{n}_{k}'].copy() for k in ('global_','extra','players','actions','targets','mask')} for n in range(3)]
        rows=[]
        for n,packet in enumerate(packets):
            action,logp,value=policy.act(packet)
            rows.append(dict(packet=packet,action=action,logp=logp,value=value,reward=[3.,-.2,-16.][n],
                duration=3. if n<2 else 0.,done=n==2,settled=True,next_value=0.))
        for n in range(2):rows[n]['next_value']=rows[n+1]['value']
        return policy,rows

    def test_real_joint_logp_update_and_reproducibility(self):
        first,rows=self.fixture();second,_=load_transfer(trainable=True)
        before=tensor_hash(first)
        a=ppo_update(first,optimizer_for(first),rows,torch.Generator().manual_seed(101),completed_episode=1)
        b=ppo_update(second,optimizer_for(second),rows,torch.Generator().manual_seed(101),completed_episode=1)
        self.assertNotEqual(before,tensor_hash(first))
        self.assertEqual(tensor_hash(first),tensor_hash(second))
        self.assertEqual(a['minibatch_updates'],4)
        self.assertAlmostEqual(a['family_entropy_coefficient'],.02)
        self.assertAlmostEqual(a['approximate_kl'],b['approximate_kl'])
        self.assertGreater(a['actor_gradient_norm'],0)
        self.assertGreater(a['critic_gradient_norm'],0)
        self.assertTrue(np.isfinite([v for v in a.values() if isinstance(v,(float,int))]).all())

    def test_incomplete_masked_and_off_policy_fail_closed(self):
        policy,rows=self.fixture();before=tensor_hash(policy)
        broken=deepcopy(rows);broken[-1]['done']=False
        with self.assertRaises(ValueError):transition_batch(broken)
        broken=deepcopy(rows);broken[0]['settled']=False
        with self.assertRaises(ValueError):transition_batch(broken)
        broken=deepcopy(rows);broken[0]['packet']['mask'][broken[0]['action']]=0
        with self.assertRaises(ValueError):transition_batch(broken)
        broken=deepcopy(rows);broken[0]['logp']+=.1
        with self.assertRaises(ValueError):ppo_update(policy,optimizer_for(policy),broken,torch.Generator(),completed_episode=1)
        self.assertEqual(tensor_hash(policy),before)

    def test_wrong_optimizer_and_frozen_impostor_are_rejected(self):
        policy,rows=self.fixture();other,_=load_transfer(trainable=True)
        with self.assertRaises(ValueError):ppo_update(policy,optimizer_for(other),rows,torch.Generator(),completed_episode=1)
        policy.requires_grad_(False)
        with self.assertRaises(ValueError):ppo_update(policy,optimizer_for(policy),rows,torch.Generator(),completed_episode=1)

    def test_terminal_zero_time_credit_and_entropy_endpoint(self):
        policy,rows=self.fixture()
        *_,adv,returns=transition_batch(rows)
        self.assertAlmostEqual(float(returns[-1]),-16.,places=5)
        result=ppo_update(policy,optimizer_for(policy),rows,torch.Generator().manual_seed(202),completed_episode=8)
        self.assertAlmostEqual(result['family_entropy_coefficient'],.005)

    def test_partition_and_selection_fixed_before_confirmation(self):
        self.assertEqual(STREAMS,(101,202))
        self.assertEqual(len(TRAIN),8);self.assertEqual(len(SELECT),8);self.assertEqual(len(CONFIRM),16)
        self.assertFalse(set(TRAIN)&set(SELECT) or set(TRAIN)&set(CONFIRM) or set(SELECT)&set(CONFIRM))
        candidates=[dict(stream=s,episode=e,path=f'{s}-{e}',sha256='x') for s in STREAMS for e in (4,8)]
        summaries=[dict(wins=0,kills=2,ejections=1,matches=[dict(seed=s) for s in SELECT],checkpoint=c['path'],crew='B') for c in candidates]
        selected=select_candidate(candidates,summaries)
        self.assertEqual(selected['candidate']['path'],'101-4')
        self.assertTrue(selected['all_zero_selection_wins'])
        summaries[3]['wins']=1
        self.assertEqual(select_candidate(candidates,summaries)['candidate']['path'],'202-8')
        summaries[0]['matches'][0]['seed']=CONFIRM[0]
        with self.assertRaises(ValueError):select_candidate(candidates,summaries)

    def test_source_freeze_and_immutable_lock(self):
        check_sources(source_signature())
        with self.assertRaises(RuntimeError):check_sources({})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lock.json'
            write_json(path,dict(locked=True),exclusive=True)
            with self.assertRaises(FileExistsError):write_json(path,dict(locked=False),exclusive=True)


if __name__=='__main__':unittest.main()
