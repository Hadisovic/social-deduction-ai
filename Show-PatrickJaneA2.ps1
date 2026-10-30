[CmdletBinding()]
param([string]$Replay='')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$python=Join-Path $root 'runtime\python\pythonw.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Run Start-PatrickJaneA2.ps1 -SetupOnly to install the bundled runtime first.' }
$run=Join-Path $root 'results\investigation-a2-v1'
$env:TCL_LIBRARY=Join-Path $root 'runtime\python\tcl\tcl8.6'
$env:TK_LIBRARY=Join-Path $root 'runtime\python\tcl\tk8.6'
$viewerArgs=@('run_spectator_v2.py','--watch',"`"$run`"",'--status-file',"`"$(Join-Path $run 'status.json')`"",'--preferences',"`"$(Join-Path $run 'viewer-preferences.json')`"",'--mode','follow_latest','--live-environment','0','--fps-limit','30')
if ($Replay) { $viewerArgs+=@('--replay',"`"$Replay`"") }
Start-Process -FilePath $python -WorkingDirectory $root -ArgumentList $viewerArgs | Out-Null
