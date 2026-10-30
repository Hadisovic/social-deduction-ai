$ErrorActionPreference='Stop'
$run=Join-Path $PSScriptRoot 'results\investigation-a2-v1'
if (!(Test-Path -LiteralPath $run)) { throw 'No portable run exists.' }
'Stop at the next complete PPO checkpoint.' | Set-Content -LiteralPath (Join-Path $run 'STOP_REQUESTED') -Encoding ASCII
Write-Host 'Stop requested. The current full wave and PPO update will finish and checkpoint.'
