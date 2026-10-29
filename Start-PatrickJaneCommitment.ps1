[CmdletBinding()]
param([switch]$CheckOnly,[switch]$EngineeringCheck,[string]$RunDirectory='',
      [int]$AdditionalDecisions=250000,[int]$Workers=32,[double]$WallHours=12,
      [int]$StopAfterUpdates=0)
$ErrorActionPreference='Stop'
if (!$RunDirectory) { $RunDirectory=Join-Path $PSScriptRoot 'local-runs\patrick-jane\task-commitment-v1' }
if ($EngineeringCheck -and $RunDirectory -eq (Join-Path $PSScriptRoot 'local-runs\patrick-jane\task-commitment-v1')) { throw 'Engineering checks require a separate folder.' }
& (Join-Path $PSScriptRoot 'Start-PatrickJaneLights.ps1') -Commitment -RunDirectory $RunDirectory -AdditionalDecisions $AdditionalDecisions -Workers $Workers -WallHours $WallHours -CheckOnly:$CheckOnly -EngineeringCheck:$EngineeringCheck -StopAfterUpdates $StopAfterUpdates
