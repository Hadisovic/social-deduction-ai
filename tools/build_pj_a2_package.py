"""Build a new A2 package without touching the delivered Task Pursuit package."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.build_pj_portable import sources,archive
from portable_support import digest


def refresh(destination):
    from patrick_jane.a2_config import A2Experiment,coefficients
    destination=Path(destination)
    files=sources()|{ROOT/'tools/run_pj_a2.py',ROOT/'tools/evaluate_pj_a2.py',ROOT/'tools/smoke_pj_published.py'}
    for p in files:
        q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for name in ('Start-PatrickJaneA2.ps1','Show-PatrickJaneA2.ps1','Stop-PatrickJaneA2.ps1','Export-PatrickJaneA2Results.ps1','Import-PatrickJaneResults.ps1'):
        shutil.copyfile(ROOT/name,destination/name)
    config=dict(schema='patrick-jane-a2-config-v1',experiment=A2Experiment().to_dict(),workers='auto: min(6,physical cores,RAM/2GiB); frozen on registration',
        default_hours=12,origin='origin',results='results/investigation-a2-v1',reward_coefficients=coefficients(A2Experiment()),
        training_seed_range=[3200000,3299999],development_comparison_seeds=[3300000,3300019],final_partitions='unopened',training_started=False)
    (destination/'portable-config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    controls=dict(schema='patrick-jane-a2-comparison-registration-v1',reference='original-update102',training_seeds=[3200000,3299999],evaluation_seeds=list(range(3300000,3300020)),
        frozen_conditions=['legacy engine/historical opponents','A2 mechanics/historical opponents','A2 mechanics/upgraded opponents'],
        reward_training_controls=['task-pursuit-v1','investigation-a2-v1'],opponent_training_controls=['historical','upgraded-a2-v1'],
        standard_training='50% historical and 50% upgraded, sampled per match with a role-independent seed stream',
        paired_protocol='Same update102 weights/Adam, matched scenario seed index and fixed decision budgets for causal training controls. Frozen evaluations paired by scenario across the same mechanics/opponent conditions. Do not infer reward effects from changing opponent difficulty.',
        baseline_or_final_games_automatically_run=False,reserved_final_partitions='unopened')
    (destination/'comparison-registration.json').write_text(json.dumps(controls,indent=2)+'\n')
    import subprocess
    try:revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except (OSError,subprocess.CalledProcessError):revision='unpublished'
    from patrick_jane.checkpoint import source_lock
    version=dict(repository='https://github.com/Hadisovic/social-deduction-ai',branch='codex/patrick-jane-recurrent-ppo',
        revision=revision,actor_source_hashes=source_lock(),trained_model_regime='task-pursuit-v1',a2_trained=False)
    (destination/'SOURCE_VERSION.json').write_text(json.dumps(version,indent=2)+'\n')
    for name in ('PATRICK_JANE_INTEGRATION.md','requirements-patrick-jane.txt','requirements-patrick-jane-lock.txt'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,destination/name)
    for name in ('A2_IMPLEMENTATION.md','A2_REGISTRATION.md','A2_RULES.md'):
        p=ROOT/'docs/patrick-jane'/name
        if p.exists():q=destination/'docs'/name;q.parent.mkdir(exist_ok=True);shutil.copyfile(p,q)
    manifest=dict(schema='patrick-jane-portable-a2-v1',runtime=dict(python='3.12.10',numpy='2.5.3',torch='2.7.1+cu128'),
        starting_checkpoint_sha256=digest(destination/'origin/update-000102.pt'),lifetime_decisions=2813879,default_hours=12,training_started=False,
        files={p.relative_to(destination).as_posix():dict(size=p.stat().st_size,sha256=digest(p)) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='portable.manifest.json' and p.relative_to(destination).parts[0] not in ('runtime','results','exports') and '__pycache__' not in p.parts})
    (destination/'portable.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return destination


def build(destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'dist/PatrickJane-TaskPursuit-V1'
    payloads=baseline/'payloads' if (baseline/'payloads').exists() else ROOT/'dist/portable-cache'
    if not (payloads/'wheels').exists():raise FileNotFoundError('Offline payloads missing. Supply dist/portable-cache with pinned wheels,Python embed ZIP,tk-support.zip and signed vc_redist.x64.exe, or reuse the previous portable payloads.')
    shutil.copytree(payloads,destination/'payloads');shutil.copytree(ROOT/'assets',destination/'assets')
    shutil.copyfile(ROOT/'requirements-patrick-jane-lock.txt',destination/'runtime-requirements.lock')
    for name in ('LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','docs/ASSET_SOURCES.md'):
        source=ROOT/name if (ROOT/name).exists() else baseline/name
        if source.exists():
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    original=ROOT/'artifacts/patrick-jane/task-pursuit-v1-update102';origin=destination/'origin';origin.mkdir()
    shutil.copyfile(original/'checkpoint.pt',origin/'update-000102.pt')
    shutil.copyfile(original/'source-snapshot.zip',origin/'source-snapshot.zip')
    shutil.copyfile(original/'artifact-manifest.json',origin/'parent-artifact-manifest.json')
    (destination/'README.md').write_text('''# Patrick Jane — Investigation A2 V1

Extract the complete package to a local Windows x64 folder. Open PowerShell here:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\Start-PatrickJaneA2.ps1 -Hours 12
```

This installs the bundled Python/PyTorch runtime offline, checks CUDA and update102, opens the spectator, and starts A2 training immediately. No long training was run while preparing the package. Default: 12 active collection/PPO hours, complete-match waves, normally six workers on the target i7-9750H/16GB laptop. Worker count is selected from available cores/RAM and frozen; explicit `-Workers 4` is supported for the first launch. Never change workers in an existing run. Keep a GTX1650-compatible driver supporting CUDA12.8 installed; the signed C++ runtime installer is bundled. No system Python or network is required.

Full living task +0.5, ghost 0, task victory +5, ejection victory +10, loss -10; accepted non-skip vote 0.2*(correct - ballot-eligible impostor fraction), skips zero. Existing +4/-2 ejection event credits and other coefficients remain. Task pursuit/commitment remain enabled. Only lights plus temporary Skeld doors; graph-constrained impostor vents; a fixed 50/50 per-match mixture of original and upgraded opponents. See docs for precise rule abstractions and limits.

Outputs/resume: `results/investigation-a2-v1/`: registration, active-time/seed journals, reports, curves, status, all checkpoints with Adam/RNG, final weights, logs and selected replays. Restart the same command to resume. Increase -Hours to extend. Optional `-AdditionalDecisions 250000` stops at the first safe update reaching that additional decision target, with -Hours retained as an upper active-duration target. Partial waves never count as trained. Ctrl+C or Stop-PatrickJaneA2.ps1 requests a safe boundary; a second Ctrl+C aborts uncheckpointed work. This A2 version uses the Windows unbiased interrupt clock to exclude suspension; deterministic clock tests passed, but physical sleep/resume has not been tested. Historical Task Pursuit timing uncertainty remains a limitation of that earlier run. `SOURCE_VERSION.json` identifies the published source; `PATRICK_JANE_INTEGRATION.md` describes the collaborator interface.

Visualization only:
```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\Show-PatrickJaneA2.ps1
```
Add `-Replay "path-to-completed.jsonl"` for a recorded game. Omniscient/debug views show door timers, vent occupancy and retrospective reward records; these are never NN inputs.

`-SetupOnly` installs/verifies runtime without training. `-PrepareOnly` registers an origin without collecting experience or updating weights. Neither is required before a normal launch. Returned results are automatically saved to `exports/` as a checksummed ZIP. Import with Import-PatrickJaneResults.ps1 into a fresh primary-machine directory; it never overwrites prior models.

Controlled development evaluations are MANUAL, using tools/evaluate_pj_a2.py and comparison-registration.json. No validation games run on training launch. Use fresh output directories; all final-test partitions remain sealed. Training win rate is not independent generalization. No automatic best-checkpoint selection.
''',encoding='utf-8')
    return refresh(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',default=str(ROOT/'dist/PatrickJane-Investigation-A2-V1'));p.add_argument('--refresh',action='store_true');p.add_argument('--zip',action='store_true');a=p.parse_args()
    result=refresh(a.destination) if a.refresh else build(a.destination)
    print(archive(result) if a.zip else result)
