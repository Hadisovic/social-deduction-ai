[CmdletBinding()]
param([switch]$CheckOnly,[switch]$EngineeringCheck,[string]$RunDirectory='',
      [int]$AdditionalDecisions=490423,[int]$Workers=32,[double]$WallHours=((43200-7904.076789799961)/3600))
$ErrorActionPreference='Stop'
if (!$RunDirectory) { $RunDirectory=Join-Path $PSScriptRoot 'local-runs\patrick-jane\lights-rewards-v2' }
if ($EngineeringCheck -and $RunDirectory -eq (Join-Path $PSScriptRoot 'local-runs\patrick-jane\lights-rewards-v2')) { throw 'Engineering checks require a separate folder.' }
& (Join-Path $PSScriptRoot 'Start-PatrickJaneLights.ps1') -RewardV2 -RunDirectory $RunDirectory -AdditionalDecisions $AdditionalDecisions -Workers $Workers -WallHours $WallHours -CheckOnly:$CheckOnly -EngineeringCheck:$EngineeringCheck
