"""Verify published trained weights and one legal actor decision; never trains."""
import argparse,hashlib,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',type=Path,default=Path('artifacts/patrick-jane/task-pursuit-v1-update102'))
    args=parser.parse_args()
    import torch
    from patrick_jane.network import PatrickJane,collate
    from patrick_jane.worker import Match
    from patrick_jane.lights_setup import LightsExperiment
    from patrick_jane.features import model_packet
    from patrick_jane.actions import choose
    torch.set_num_threads(1);torch.manual_seed(17)
    manifest=json.loads((args.model_dir/'artifact-manifest.json').read_text())
    def verified(name):
        path=args.model_dir/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][name]['sha256']:raise ValueError('Model artifact checksum mismatch: '+name)
        return torch.load(path,map_location='cpu',weights_only=True)
    checkpoint=verified('checkpoint.pt');weights=verified('inference.pt')
    policy=PatrickJane()
    if weights['architecture']!=policy.architecture():raise ValueError('Published architecture mismatch')
    policy.load_state_dict(weights['state_dict'],strict=True);policy.eval()
    assert len(policy.state_dict())==52
    assert all(torch.equal(v,checkpoint['state_dict'][k]) for k,v in policy.state_dict().items())
    # Legacy task-pursuit observation semantics match the trained regime. No game
    # is advanced; seed belongs solely to the engineering partition.
    match=Match(2920006,LightsExperiment(task_commitment_mode=True,task_pursuit_mode=True,living_task_reward=2.));match.prepare()
    owner=match.crew[0];packet=model_packet(match.packets[owner]);hidden=torch.zeros(1,256)
    generator=torch.Generator().manual_seed(17)
    with torch.no_grad():
        output,next_hidden=policy(collate([packet],'cpu'),hidden)
        choice,logp,entropy,_=choose({k:v[0] for k,v in output.items()},packet,generator=generator)
    assert next_hidden.shape==(1,256) and torch.isfinite(next_hidden).all()
    print(json.dumps(dict(trained_regime=weights['regime'],a2_trained=False,lifetime_decisions=weights['lifetime_decisions'],
        parameter_count=sum(p.numel() for p in policy.parameters()),exact_tensor_matches=52,
        checkpoint_update=checkpoint['counters']['update'],accepted_legal_choice=vars(choice),
        log_probability=float(logp),entropy=float(entropy),hidden_shape=list(next_hidden.shape),
        game_steps=0,training_updates=0),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
