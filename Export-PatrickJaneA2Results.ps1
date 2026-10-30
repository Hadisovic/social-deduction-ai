$ErrorActionPreference='Stop'
$python=Join-Path $PSScriptRoot 'runtime\python\python.exe'
& $python (Join-Path $PSScriptRoot 'portable_support.py') --root $PSScriptRoot --export --folder (Join-Path $PSScriptRoot 'results\investigation-a2-v1')
if ($LASTEXITCODE -ne 0) { throw 'Export failed; original results remain preserved.' }
