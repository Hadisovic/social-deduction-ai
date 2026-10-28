import unittest
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from patrick_jane.config import Experiment
from patrick_jane.network import PatrickJane
from patrick_jane.collector import Pool,collect
from patrick_jane.ppo import update,prepare
from patrick_jane.checkpoint import save,load,select_champion
from patrick_jane.training import train


class TrainingReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17);self.policy=PatrickJane()
        base=Experiment();self.experiment=replace(base,lobby=replace(base.lobby,timeout=1.8))

    def rollout(self,folder=None,processes=False):
        with Pool(1,processes) as pool:
            return collect(self.policy,pool,self.experiment,[2920099],replay_dir=folder,
                           record_every=1 if folder else 0,wall_limit=60.)

    def test_central_recurrence_actual_update_checkpoint_and_replay_equality(self):
        from spectator.replay_format import parse_replay
        from spectator.v2.data import ReplayIndex
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"local-runs") as folder:
            plain=self.rollout();recorded=self.rollout(folder)
            self.assertEqual(plain['matches'][0]['trace'],recorded['matches'][0]['trace'])
            self.assertEqual(plain['reward_components'],recorded['reward_components'])
            for key,tr in plain['trajectories'].items():
                other=recorded['trajectories'][key]
                self.assertTrue(all(a.choice==b.choice and a.reward==b.reward and a.logp==b.logp and a.duration==b.duration for a,b in zip(tr,other)))
            replay=next(Path(folder).glob('*.jsonl'));data=parse_replay(replay);index=ReplayIndex(data)
            self.assertEqual(len(data['ticks'][-1]['players']),12);self.assertTrue(data['neural']);self.assertFalse(index.issues)
            before={k:v.clone() for k,v in self.policy.state_dict().items()};optimizer=torch.optim.Adam(self.policy.parameters(),lr=.0003)
            metrics=update(self.policy,optimizer,plain['trajectories'],self.experiment,rng=np.random.default_rng(17))
            self.assertEqual(metrics['epochs'],4);self.assertTrue(np.isfinite(metrics['loss']))
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in self.policy.state_dict().items()))
            updated={k:v.clone() for k,v in self.policy.state_dict().items()}
            self.policy.load_state_dict(before);second_optimizer=torch.optim.Adam(self.policy.parameters(),lr=.0003)
            update(self.policy,second_optimizer,recorded['trajectories'],self.experiment,rng=np.random.default_rng(17))
            self.assertTrue(all(torch.equal(updated[k],v) for k,v in self.policy.state_dict().items()))
            path=Path(folder)/'model.pt';rng=np.random.default_rng(17)
            save(path,self.policy,second_optimizer,self.experiment,dict(update=1),rng=rng)
            fresh=PatrickJane();record=load(path,fresh,restore_rng=False)
            self.assertTrue(all(torch.equal(v,fresh.state_dict()[k]) for k,v in self.policy.state_dict().items()))
            with self.assertRaises(FileExistsError):save(path,fresh,second_optimizer,self.experiment,{},rng=rng)
            self.assertTrue(select_champion(path,plain['matches'],Path(folder)/'champions'))
            # Same result is not a new champion and cannot overwrite a file.
            self.assertFalse(select_champion(path,plain['matches'],Path(folder)/'champions'))

    def test_inline_process_reproducibility_and_actor_boundaries(self):
        plain=self.rollout();worker=self.rollout(processes=True)
        self.assertEqual(plain['matches'][0]['trace'],worker['matches'][0]['trace'])
        self.assertEqual(plain['reward_components'],worker['reward_components'])
        self.assertEqual(len(plain['trajectories']),10)
        for key,tr in plain['trajectories'].items():
            self.assertTrue(tr[-1].done);self.assertTrue(all(t.duration>0 for t in tr))
        seq=prepare(plain['trajectories'],self.experiment)
        self.assertTrue(all(len(tr)<=128 for _,tr,_,_ in seq))

    def test_full_run_requires_explicit_approval(self):
        with self.assertRaises(PermissionError):train('never-created',target_decisions=1_000_000)


if __name__=='__main__':unittest.main()

