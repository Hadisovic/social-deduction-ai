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

    def test_partial_evaluation_batch_reuses_existing_workers(self):
        with Pool(2,False) as pool:
            result=collect(self.policy,pool,self.experiment,[2920099],wall_limit=60.)
        self.assertEqual(len(result['matches']),1)
        self.assertEqual(len(result['trajectories']),10)

    def test_registered_runner_checkpoint_validation_and_bounded_completion(self):
        import json
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"local-runs") as folder:
            output=Path(folder)/'registered'
            experiment=replace(self.experiment,evaluation_matches=2)
            result=train(output,target_decisions=30,owner_approved=True,environments=1,device='cpu',
                experiment=experiment,hard_wall_seconds=90,record_every=1)
            self.assertEqual(result['update'],1);self.assertGreater(result['individual_decisions'],1)
            self.assertTrue((output/'completion.json').is_file());self.assertTrue((output/'latest.json').is_file())
            self.assertEqual(len(list((output/'evaluations').glob('*.json'))),2)
            zero=json.loads((output/'evaluations/update-000000-zero.json').read_text())
            final=json.loads((output/'evaluations/update-000001-final.json').read_text())
            self.assertEqual([m['seed'] for m in zero['matches']],[m['seed'] for m in final['matches']])
            status=json.loads((output/'status.json').read_text());self.assertEqual(status['stage'],'complete')
            self.assertEqual(status['overall_decisions'],result['individual_decisions'])
            self.assertTrue((output/'champions/champion.json').is_file())

    def test_expired_ppo_deadline_preserves_weights(self):
        result=self.rollout();before={k:v.clone() for k,v in self.policy.state_dict().items()}
        optimizer=torch.optim.Adam(self.policy.parameters(),lr=.0003)
        with self.assertRaises(TimeoutError):update(self.policy,optimizer,result['trajectories'],self.experiment,deadline=0)
        self.assertTrue(all(torch.equal(v,self.policy.state_dict()[k]) for k,v in before.items()))

    def test_live_monitor_is_observer_only_and_uses_real_replay_schema(self):
        import json
        from spectator.v2.data import ReplayIndex
        plain=self.rollout()
        with tempfile.TemporaryDirectory(dir=Path.cwd()/"local-runs") as folder:
            with Pool(1,False) as pool:
                live=collect(self.policy,pool,self.experiment,[2920099],monitor_dir=folder,wall_limit=60.)
            self.assertEqual(plain['matches'][0]['trace'],live['matches'][0]['trace'])
            self.assertEqual(plain['reward_components'],live['reward_components'])
            data=json.loads((Path(folder)/'env-00.json').read_text());index=ReplayIndex(data)
            self.assertEqual(len(index.frame().players),12);self.assertTrue(data['neural'])
            self.assertTrue(data['complete']);self.assertTrue(data['live_snapshot']);self.assertFalse(index.issues)
            self.assertEqual(data['result']['reason'],live['matches'][0]['reason'])
            for record in data['rewards']:
                expected=sum(live['reward_components'].get('0/'+record['player_id'],{}).values())
                self.assertAlmostEqual(record['cumulative_reward'],expected)


if __name__=='__main__':unittest.main()

