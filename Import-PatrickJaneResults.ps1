[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$Archive,[string]$Destination='')
$ErrorActionPreference='Stop'
$python=Join-Path $PSScriptRoot 'runtime\python\python.exe'
if (!(Test-Path -LiteralPath $python)) { $python=Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (!$Destination) { $Destination=Join-Path $PSScriptRoot ('local-runs\imported\task-pursuit-v1-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) }
& $python (Join-Path $PSScriptRoot 'portable_support.py') --import-archive $Archive --destination $Destination
if ($LASTEXITCODE -ne 0) { throw 'Import refused. Existing models were not replaced.' }
Write-Host "Verified model, sources and results imported to $Destination"
