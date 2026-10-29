[CmdletBinding()]
param(
    [int]$AdditionalDecisions = 1000000,
    [int]$Workers = 32,
    [double]$WallHours = 12,
    [string]$RunDirectory = '',
    [switch]$CheckOnly,
    [switch]$EngineeringCheck,
    [switch]$RewardV2,
    [int]$StopAfterUpdates = 0
)
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$python = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$pythonWindow = Join-Path $projectDirectory '.venv\Scripts\pythonw.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Patrick Jane Python environment is missing.' }
if (!$RunDirectory) { $RunDirectory = Join-Path $projectDirectory 'local-runs\patrick-jane\lights-learning-v1' }
$RunDirectory = [IO.Path]::GetFullPath($RunDirectory)
if ($EngineeringCheck -and $RunDirectory -eq (Join-Path $projectDirectory 'local-runs\patrick-jane\lights-learning-v1')) {
    throw 'Engineering checks require a separate -RunDirectory.'
}
$env:PYTHONIOENCODING = 'utf-8'
$priorConsoleEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$arguments = @('-m','tools.run_pj_learning','--folder',$RunDirectory,'--additional-decisions',"$AdditionalDecisions",'--workers',"$Workers",'--wall-hours',"$WallHours")
if ($EngineeringCheck) { $arguments += '--engineering' }
if ($RewardV2) { $arguments += '--reward-v2' }
if ($StopAfterUpdates -gt 0) { $arguments += @('--stop-after-updates',"$StopAfterUpdates") }
Push-Location -LiteralPath $projectDirectory
try {
    if ($CheckOnly) {
        & $python @arguments --check-only
        if ($LASTEXITCODE -ne 0) { throw 'Preflight failed.' }
        return
    }
    # Verify the historical import before starting either desktop app or workers.
    & $python @arguments --check-only
    if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; training was not started.' }
    New-Item -ItemType Directory -Path $RunDirectory -Force | Out-Null
    $viewerState = Join-Path $RunDirectory 'viewer-state.json'
    $viewerRunning = $false
    if (Test-Path -LiteralPath $viewerState) {
        try {
            $previousViewer = Get-Content -LiteralPath $viewerState -Raw | ConvertFrom-Json
            $viewerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($previousViewer.pid)"
            $viewerRunning = $null -ne $viewerProcess -and $viewerProcess.CommandLine -like '*run_spectator_v2.py*' -and $viewerProcess.CommandLine -like "*$RunDirectory*"
        } catch { $viewerRunning = $false }
    }
    if (!$viewerRunning) {
        $env:SDL_VIDEODRIVER = $null
        $viewerArguments = @('run_spectator_v2.py','--watch',"`"$RunDirectory`"",'--status-file',"`"$(Join-Path $RunDirectory 'status.json')`"",'--preferences',"`"$(Join-Path $RunDirectory 'preferences.json')`"",'--state-file',"`"$viewerState`"",'--mode','follow_latest','--live-environment','0')
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        Start-Process -FilePath $pythonWindow -WorkingDirectory $projectDirectory -ArgumentList $viewerArguments -RedirectStandardOutput (Join-Path $RunDirectory "viewer-$stamp.stdout.log") -RedirectStandardError (Join-Path $RunDirectory "viewer-$stamp.stderr.log") | Out-Null
    }
    Write-Host "Results and logs: $RunDirectory" -ForegroundColor Cyan
    Write-Host 'Ctrl+C interrupts safely; rerun the same command to resume. The spectator remains available.' -ForegroundColor Cyan
    $trainingExit = 130
    try {
        & $python @arguments
        $trainingExit = $LASTEXITCODE
    } finally {
        if ($trainingExit -ne 0) {
            $completionPath = Join-Path $RunDirectory 'completion-status.json'
            $terminalRecorded = $false
            if (Test-Path -LiteralPath $completionPath) {
                try { $terminalRecorded = (Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json).state -in @('complete','interrupted','failed') } catch {}
            }
            if (!$terminalRecorded) {
                $exitState = if ($trainingExit -eq 130 -or $trainingExit -eq -1073741510) { 'interrupted' } else { 'failed' }
                & $python -m tools.run_pj_learning --folder $RunDirectory --finalize $exitState
            }
        }
    }
    if ($trainingExit -ne 0) { throw "Training did not complete (exit $trainingExit). See TRAINING_REPORT.md and console.log in $RunDirectory." }
} finally { [Console]::OutputEncoding = $priorConsoleEncoding; Pop-Location }
