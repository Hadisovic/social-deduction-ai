param([Parameter(Mandatory=$true)][string]$Study)
$ErrorActionPreference='Stop'
$studyPath=(Resolve-Path -LiteralPath $Study).Path
$registration=Get-Content -LiteralPath (Join-Path $studyPath 'registration.json') -Raw | ConvertFrom-Json
$deadline=[DateTimeOffset]::Parse($registration.deadline_utc)
$guardPath=Join-Path $studyPath 'process-tree-guard.json'
$reason=$null
while ($true) {
    $status=$null
    try {$status=Get-Content -LiteralPath (Join-Path $studyPath 'status.json') -Raw | ConvertFrom-Json} catch {}
    if ([DateTimeOffset]::UtcNow -ge $deadline) {$reason='registered-three-hour-deadline'}
    elseif ($status.phase -eq 'failed') {$reason='registered-fatal-study-stop'}
    elseif ($status.phase -in @('finished','confirmation-incomplete')) {break}
    if ($reason) {
        # Windows virtual-environment redirectors spawn the actual interpreter.
        # Identify only this study's worker command, never other Python projects
        # or the offline reporter/supervisor. Verify both full output-path scope
        # and script name before stopping each owned runtime/redirector.
        $owned=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
            $_.CommandLine -like '*run_lizbon_classic.py*' -and
            $_.CommandLine.IndexOf($studyPath,[StringComparison]::OrdinalIgnoreCase) -ge 0
        })
        $stopped=@()
        foreach ($worker in $owned) {
            try {Stop-Process -Id $worker.ProcessId -Force -ErrorAction Stop;$stopped+=$worker.ProcessId} catch {
                if (Get-Process -Id $worker.ProcessId -ErrorAction SilentlyContinue) {throw}
            }
        }
        @{phase='enforced';reason=$reason;utc=[DateTimeOffset]::UtcNow.ToString('o');stopped_worker_pids=$stopped} |
            ConvertTo-Json | Set-Content -LiteralPath $guardPath -Encoding utf8
        break
    }
    @{phase='watching';deadline_utc=$registration.deadline_utc;updated_utc=[DateTimeOffset]::UtcNow.ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $guardPath -Encoding utf8
    Start-Sleep -Seconds 5
}
