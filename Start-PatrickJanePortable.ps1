[CmdletBinding()]
param([double]$Hours=12,[switch]$SetupOnly)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$runtime=Join-Path $root 'runtime\python'
$python=Join-Path $runtime 'python.exe'
$manifest=Get-Content -Raw -LiteralPath (Join-Path $root 'portable.manifest.json') | ConvertFrom-Json
$payload=Join-Path $root 'payloads\python-3.12.10-embed-amd64.zip'
if ((Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.files.'payloads/python-3.12.10-embed-amd64.zip'.sha256) { throw 'Bundled Python archive integrity failure.' }
if (!(Test-Path -LiteralPath $python)) {
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    Expand-Archive -LiteralPath $payload -DestinationPath $runtime -Force
}
$pathFile=Join-Path $runtime 'python312._pth'
@('python312.zip','.','Lib','Lib\site-packages','..\..','import site') | Set-Content -LiteralPath $pathFile -Encoding ASCII
Push-Location -LiteralPath $root
try {
    & $python portable_support.py --verify --root $root
    if ($LASTEXITCODE -ne 0) { throw 'Package checksum verification failed.' }
    if (!(Test-Path -LiteralPath (Join-Path $runtime 'Lib\tkinter\__init__.py')) -or !(Test-Path -LiteralPath (Join-Path $runtime 'zlib1.dll'))) {
        Expand-Archive -LiteralPath (Join-Path $root 'payloads\tk-support.zip') -DestinationPath $runtime -Force
    }
    $env:TCL_LIBRARY=Join-Path $runtime 'tcl\tcl8.6'
    $env:TK_LIBRARY=Join-Path $runtime 'tcl\tk8.6'
    if (!(Test-Path -LiteralPath (Join-Path $env:WINDIR 'System32\msvcp140.dll')) -or !(Test-Path -LiteralPath (Join-Path $env:WINDIR 'System32\vcruntime140.dll'))) {
        $vc=Start-Process -FilePath (Join-Path $root 'payloads\vc_redist.x64.exe') -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru -WindowStyle Hidden
        if ($vc.ExitCode -notin @(0,3010,1638)) { throw 'Microsoft Visual C++ runtime installation failed. Run payloads\vc_redist.x64.exe with Windows administrator approval, then rerun this command.' }
    }
    $lock=Join-Path $root 'runtime-requirements.lock'
    $lockHash=(Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash
    $marker=Join-Path $root 'runtime\dependencies.ready'
    if (!(Test-Path -LiteralPath $marker) -or (Get-Content -Raw -LiteralPath $marker).Trim() -ne $lockHash) {
        Write-Host 'Installing bundled dependencies offline. Training starts immediately afterward.' -ForegroundColor Cyan
        $pipWheel=(Get-ChildItem -LiteralPath (Join-Path $root 'payloads\wheels') -Filter 'pip-*.whl' | Select-Object -First 1).FullName
        $site=Join-Path $runtime 'Lib\site-packages'
        $setupTemp=Join-Path $root 'runtime\setup-temp'
        New-Item -ItemType Directory -Path $setupTemp -Force | Out-Null
        & $python -c "import sys,runpy,tempfile;sys.path.insert(0,sys.argv.pop(1));tempfile.tempdir=sys.argv.pop(1);runpy.run_module('pip',run_name='__main__')" $pipWheel $setupTemp install --no-index --no-cache-dir --no-compile --find-links (Join-Path $root 'payloads\wheels') --require-hashes -r $lock --target $site --upgrade --no-warn-script-location
        if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed. Rerun after resolving the displayed error.' }
        $lockHash | Set-Content -LiteralPath $marker -Encoding ASCII
    }
    & $python -m tools.run_pj_duration --setup-check --origin (Join-Path $root 'origin')
    if ($LASTEXITCODE -ne 0) { throw 'Runtime/checkpoint check failed. No training started; see the error above.' }
    if ($SetupOnly) { Write-Host 'SETUP VERIFIED - training was not started.' -ForegroundColor Green; return }
    $run=Join-Path $root 'results\task-pursuit-v1'
    New-Item -ItemType Directory -Path $run -Force | Out-Null
    $viewerState=Join-Path $run 'viewer-state.json'
    $running=$false
    if (Test-Path -LiteralPath $viewerState) {
        try {
            $oldViewer=Get-Content -Raw -LiteralPath $viewerState | ConvertFrom-Json
            $process=Get-CimInstance Win32_Process -Filter "ProcessId = $($oldViewer.pid)"
            $running=$null -ne $process -and $process.CommandLine -like '*run_spectator_v2.py*' -and $process.CommandLine -like "*$run*"
        } catch { $running=$false }
    }
    if (!$running) {
        try {
            $viewerArgs=@('run_spectator_v2.py','--watch',"`"$run`"",'--status-file',"`"$(Join-Path $run 'status.json')`"",'--preferences',"`"$(Join-Path $run 'viewer-preferences.json')`"",'--state-file',"`"$viewerState`"",'--mode','follow_latest','--live-environment','0','--fps-limit','30')
            $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
            Start-Process -FilePath (Join-Path $runtime 'pythonw.exe') -WorkingDirectory $root -ArgumentList $viewerArgs -RedirectStandardOutput (Join-Path $run "viewer-$stamp.stdout.log") -RedirectStandardError (Join-Path $run "viewer-$stamp.stderr.log") | Out-Null
        } catch { Write-Warning "Spectator could not open; training will continue: $_" }
    }
    Write-Host "TRAINING NOW - 6 workers; $Hours cumulative active hours. Results: $run" -ForegroundColor Cyan
    Write-Host 'Ctrl+C requests a safe wave/PPO stop; rerun this command to resume. Use -Hours 24 to extend.' -ForegroundColor Cyan
    & $python -m tools.run_pj_duration --folder $run --origin (Join-Path $root 'origin') --hours $Hours
    $code=$LASTEXITCODE
    if ($code -eq 0) { Write-Host 'TRAINING COMPLETE - reports and export ZIP saved.' -ForegroundColor Green }
    elseif ($code -eq 130) { Write-Host 'SAFELY INTERRUPTED - rerun the same command to resume.' -ForegroundColor Yellow }
    else { throw "Training failed (exit $code). Check results\task-pursuit-v1\console.log and preserved checkpoints." }
} finally { Pop-Location }
