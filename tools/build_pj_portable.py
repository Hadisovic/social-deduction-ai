"""Build the approved offline Windows package; never executes a learner."""
import argparse,ast,email,hashlib,json,shutil,zipfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from portable_support import digest


def sources():
    # Packages are small and include dynamic spectator/worker imports. Root
    # module dependencies are discovered statically instead of copying research runs.
    files=set()
    for package in ('patrick_jane','spectator','crewmate_v2','social_deduction','belief'):
        files.update(p for p in (ROOT/package).rglob('*.py') if '__pycache__' not in p.parts)
    for name in ('phase3_renderer.py','among_us_renderer.py','phase3_assets.py','portable_support.py','run_spectator_v2.py',
                 'tools/__init__.py','tools/run_pj_duration.py','tools/run_pj_learning.py','tools/report_patrick_jane_pilot.py'):
        p=ROOT/name
        if p.exists():files.add(p)
    pending=list(files);seen=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=([n.name.split('.')[0] for n in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else [])
            for name in names:
                q=ROOT/(name+'.py')
                if q.exists() and q not in files:files.add(q);pending.append(q)
    return files


def build(destination,cache):
    destination=Path(destination).resolve();cache=Path(cache).resolve()
    if destination.exists():raise FileExistsError('Use a new package directory; previous packages are preserved')
    destination.mkdir(parents=True)
    for p in sources():
        q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for p in (ROOT/'assets').rglob('*'):
        if p.is_file() and p.suffix.lower() not in ('.pyc','.tmp'):
            q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for name in ('Start-PatrickJanePortable.ps1','Stop-PatrickJanePortable.ps1','Export-PatrickJaneResults.ps1','Import-PatrickJaneResults.ps1'):
        shutil.copyfile(ROOT/name,destination/name)
    for name in ('LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','docs/ASSET_SOURCES.md'):
        p=ROOT/name
        if p.is_file():q=destination/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    origin=destination/'origin';origin.mkdir()
    old=ROOT/'local-runs/patrick-jane/task-commitment-v1'
    shutil.copyfile(old/'checkpoints/attempt-0002-update-000018.pt',origin/'update-000018.pt')
    shutil.copyfile(old/'source-snapshot.zip',origin/'source-snapshot.zip')
    shutil.copyfile(old/'registration.json',origin/'parent-registration.json')
    payload=destination/'payloads';payload.mkdir()
    for name in ('python-3.12.10-embed-amd64.zip','vc_redist.x64.exe','tk-support.zip'):shutil.copyfile(cache/name,payload/name)
    wheels=payload/'wheels';wheels.mkdir();requirements=[]
    for p in sorted((cache/'wheels').glob('*.whl')):
        shutil.copyfile(p,wheels/p.name)
        with zipfile.ZipFile(p) as z:
            metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        requirements.append(f"{metadata['Name']}=={metadata['Version']} --hash=sha256:{digest(p)}")
    (destination/'runtime-requirements.lock').write_text('\n'.join(requirements)+'\n',encoding='utf-8')
    (destination/'README.md').write_text('''# Patrick Jane — Task Pursuit V1 portable training

Extract the entire ZIP to a local Windows x64 folder. Open PowerShell there and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\\Start-PatrickJanePortable.ps1
```

Python 3.12.10 and all Python dependencies are bundled; setup installs only inside this folder. No Git, existing Python or network is required. A compatible NVIDIA driver for the GTX 1650 is required (CUDA 12.8; a recent GTX1650-supported driver). The signed Microsoft C++ installer is bundled if missing; Windows may require administrator approval for that system component.

The spectator opens and real training begins immediately after setup/integrity checks. There are no baseline matches, evaluation games or worker benchmarks. Six CPU simulations share one GPU policy, with ten independently observed/memoried crewmates per game and two scripted impostors. Keep the laptop on power and awake while training.

Default: 12 cumulative active collection/PPO hours. Setup, report generation, export and downtime do not use this budget. At the target, finish the current wave and PPO checkpoint. `-Hours 16` or `-Hours 24` appends a duration extension; all other settings remain frozen. Different GPU hardware can change numerical trajectories; cross-device bitwise identity is not claimed.

Ctrl+C requests a safe wave/PPO stop. A second Ctrl+C aborts uncheckpointed work. Alternatively run `powershell.exe -File .\\Stop-PatrickJanePortable.ps1` in another terminal. Rerun the original command to resume the latest verified checkpoint. A crash loses the incomplete wave; trained counters never include it. Active time is durably sampled during collection, so a hard crash can leave a small unrecorded interval.

Results: `results/task-pursuit-v1/`. Read `TRAINING_REPORT.md`, `training_results.json`, `learning_curves.png`, `console.log`, `status.json`, `latest.json`, `final-model.json`, `checkpoints/`, `weights/`, and `attempts/*/replays/`. All checkpoints and optimizer/RNG states are retained. Best checkpoint is **not evaluated**; training wins are not held-out competence.

Full living task reward is +2; ghost reward is 0. All other rewards, architecture and PPO hyperparameters are unchanged. Pursuit selects owned tasks using existing heads. The network chooses task interactions; commitment starts only after actual legal work begins. Body reports and lights repairs remain legitimate pauses.

On completion/interruption, `exports/` receives a checksummed results/source/model ZIP. Transfer it back. From the primary Patrick Jane folder run `powershell.exe -File .\\Import-PatrickJaneResults.ps1 -Archive "PATH-TO-EXPORT.zip"`. Import verifies all members into a new `local-runs/imported/` directory and never replaces existing models or source. Manual re-export: `powershell.exe -File .\\Export-PatrickJaneResults.ps1`.

`-SetupOnly` verifies the runtime and parent checkpoint without opening the spectator or starting training. This is for installation troubleshooting, not a required preparation step.
''',encoding='utf-8')
    configuration(destination)
    files={p.relative_to(destination).as_posix():dict(size=p.stat().st_size,sha256=digest(p)) for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest=dict(schema='patrick-jane-portable-v1',files=files,workers=6,default_hours=12,
        starting_checkpoint_sha256=files['origin/update-000018.pt']['sha256'],lifetime_decisions=1518786,
        runtime=dict(python='3.12.10',numpy='2.5.3',torch='2.7.1+cu128'),training_started=False)
    (destination/'portable.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return destination


def archive(destination):
    destination=Path(destination);target=destination.with_suffix('.zip')
    with zipfile.ZipFile(target,'x',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file() and p.relative_to(destination).parts[0] not in ('runtime','results','exports') and '__pycache__' not in p.parts:
                # Wheels are already compressed; avoid spending minutes recompressing 3GB.
                z.write(p,(Path(destination.name)/p.relative_to(destination)).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ('.whl','.zip','.exe') else zipfile.ZIP_DEFLATED)
    target.with_suffix('.zip.sha256').write_text(digest(target)+'  '+target.name+'\n')
    return target


def refresh(destination):
    """Refresh only build-owned sources, then rehash immutable package payloads."""
    destination=Path(destination)
    for p in sources():
        q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for p in ROOT.glob('*-PatrickJane*.ps1'):
        if p.name in ('Start-PatrickJanePortable.ps1','Stop-PatrickJanePortable.ps1','Export-PatrickJaneResults.ps1','Import-PatrickJaneResults.ps1'):shutil.copyfile(p,destination/p.name)
    manifest=json.loads((destination/'portable.manifest.json').read_text())
    configuration(destination)
    tk_payload=destination/'payloads/tk-support.zip'
    shutil.copyfile(ROOT/'dist/portable-cache/tk-support.zip',tk_payload)
    paths=set(manifest['files'])|{p.relative_to(ROOT).as_posix() for p in sources()}|{'portable-config.json','payloads/tk-support.zip'}
    manifest['files']={name:dict(size=(destination/name).stat().st_size,sha256=digest(destination/name)) for name in sorted(paths)}
    (destination/'portable.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def configuration(destination):
    from patrick_jane.duration_run import experiment
    from patrick_jane.rewards import COEFFICIENTS
    profile=dict(schema='patrick-jane-portable-config-v1',workers=6,default_hours=12,device='cuda',
        origin='origin',results='results/task-pursuit-v1',experiment=experiment().to_dict(),
        reward_coefficients={**COEFFICIENTS,'task':2.},training_seed_range=[3100000,3199999],
        evaluations=False,notes='Frozen profile snapshot. Select duration with launcher -Hours; do not edit frozen source/configuration while resuming.')
    (Path(destination)/'portable-config.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',default=str(ROOT/'dist/PatrickJane-TaskPursuit-V1'));p.add_argument('--cache',default=str(ROOT/'dist/portable-cache'));p.add_argument('--zip-only',action='store_true')
    a=p.parse_args();dest=Path(a.destination)
    if not a.zip_only:build(dest,a.cache)
    print(archive(dest))
