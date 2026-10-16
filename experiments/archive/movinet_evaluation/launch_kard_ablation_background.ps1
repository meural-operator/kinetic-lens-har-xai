$ErrorActionPreference = 'Stop'
$xaiPython = 'C:\Users\Xerox\miniconda3\envs\xai\python.exe'
$experimentRoot = Join-Path $PSScriptRoot 'kard_a0_controlled_ablation_v1'
$preflightPath = Join-Path $experimentRoot 'preflight.json'
if (-not (Test-Path -LiteralPath $preflightPath)) { throw 'Verified preflight is required before launch.' }
$pidFile = Join-Path $experimentRoot 'launch.json'
if (Test-Path -LiteralPath $pidFile) {
    $priorRun = Get-Content -LiteralPath $pidFile -Raw | ConvertFrom-Json
    $priorProcess = Get-Process -Id $priorRun.pid -ErrorAction SilentlyContinue
    if ($priorProcess) {
        # PowerShell may deserialize ISO timestamps as DateTime, not strings.
        $priorStartUtc = ([DateTime]$priorRun.processStartUtc).ToUniversalTime()
        $startDifferenceMs = [Math]::Abs(($priorProcess.StartTime.ToUniversalTime() - $priorStartUtc).TotalMilliseconds)
        if ($startDifferenceMs -le 1) {
            throw "The existing ablation process is still running (PID $($priorRun.pid))."
        }
        throw "Recorded PID $($priorRun.pid) is active with a different start time; inspect identity before launching."
    }
}
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$stdoutPath = Join-Path $experimentRoot "training_$stamp.log"
$stderrPath = Join-Path $experimentRoot "training_$stamp.stderr.log"
$trainingScript = Join-Path $PSScriptRoot 'run_kard_ablation.py'
# Separate hidden host process and persistent redirected file handles survive
# closing the launcher and chat. Runner's named mutex rejects duplicate jobs.
$process = Start-Process -FilePath $xaiPython -ArgumentList @('-u', '-B', ('"' + $trainingScript + '"'), '--resume') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$launch = [ordered]@{ pid=$process.Id; processStartUtc=$process.StartTime.ToUniversalTime().ToString('o'); launchedUtc=[DateTime]::UtcNow.ToString('o'); executable=$xaiPython; script=$trainingScript; stdout=$stdoutPath; stderr=$stderrPath; monitorPolicy='Check only when requested by the user.' }
$launch | ConvertTo-Json | Set-Content -LiteralPath $pidFile -Encoding utf8
$launch | ConvertTo-Json
