"""Synthetic optimizer-state checks, not reinforcement-learning matches/studies."""
from pathlib import Path
import tempfile
import unittest
import torch
from impostor_shared.model import FamilyPolicy
from impostor_shared.weights import tensor_hash
from impostor_shared.checkpoints import save,load,optimizer_for

def synthetic_update(policy,optimizer):
    optimizer.zero_grad();sum(p.square().sum() for p in policy.parameters()).backward();optimizer.step()

class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_optimizer_rng_and_next_update_roundtrip(self):
        policy=FamilyPolicy(17);optimizer=optimizer_for(policy);generator=torch.Generator().manual_seed(31)
        synthetic_update(policy,optimizer)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'synthetic-only.pt'
            global_rng=torch.get_rng_state().clone()
            save(path,policy,optimizer,generator,record=dict(development_only=True,rl_updates=0),next_seed=3600000,episode_complete=True)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            restored,opt,rng,payload=load(path)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
            self.assertEqual(tensor_hash(policy),tensor_hash(restored))
            self.assertEqual(payload['next_seed'],3600000)
            self.assertTrue(torch.equal(torch.rand(8,generator=generator),torch.rand(8,generator=rng)))
            synthetic_update(policy,optimizer);synthetic_update(restored,opt)
            self.assertEqual(tensor_hash(policy),tensor_hash(restored))
            with self.assertRaises(FileExistsError):
                save(path,policy,optimizer,generator,record={},next_seed=3600001,episode_complete=True)

    def test_incomplete_episode_and_wrong_optimizer_are_rejected(self):
        policy=FamilyPolicy(17);other=FamilyPolicy(18);rng=torch.Generator()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rejected.pt'
            with self.assertRaises(ValueError):save(path,policy,optimizer_for(policy),rng,record={},next_seed=3600000,episode_complete=False)
            with self.assertRaises(ValueError):save(path,policy,optimizer_for(other),rng,record={},next_seed=3600000,episode_complete=True)
            self.assertFalse(path.exists())

if __name__=='__main__':unittest.main()
