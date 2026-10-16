$ErrorActionPreference = 'Stop'
$adapterPython = 'C:\Users\Xerox\miniconda3\envs\xai\python.exe'
$adapterExperiment = Join-Path $PSScriptRoot 'kard_a0_projection_adapter_v1'
if (-not (Test-Path -LiteralPath (Join-Path $adapterExperiment 'preflight.json'))) { throw 'Verified preflight is required.' }
$adapterPidFile = Join-Path $adapterExperiment 'launch.json'
if (Test-Path -LiteralPath $adapterPidFile) {
    $adapterPrior = Get-Content -LiteralPath $adapterPidFile -Raw | ConvertFrom-Json
    $adapterProcess = Get-Process -Id $adapterPrior.pid -ErrorAction SilentlyContinue
    if ($adapterProcess) { throw "Recorded adapter PID $($adapterPrior.pid) is active; inspect before launching." }
}
$adapterStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$adapterStdout = Join-Path $adapterExperiment "training_$adapterStamp.log"
$adapterStderr = Join-Path $adapterExperiment "training_$adapterStamp.stderr.log"
$adapterScript = Join-Path $PSScriptRoot 'run_kard_projection_adapters.py'
$adapterProcess = Start-Process -FilePath $adapterPython -ArgumentList @('-u','-B',('"' + $adapterScript + '"'),'--resume') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput $adapterStdout -RedirectStandardError $adapterStderr -PassThru
$adapterLaunch = [ordered]@{pid=$adapterProcess.Id;processStartUtc=$adapterProcess.StartTime.ToUniversalTime().ToString('o');launchedUtc=[DateTime]::UtcNow.ToString('o');executable=$adapterPython;script=$adapterScript;stdout=$adapterStdout;stderr=$adapterStderr;monitorPolicy='Check only when requested by user.'}
$adapterLaunch | ConvertTo-Json | Set-Content -LiteralPath $adapterPidFile -Encoding utf8
$adapterLaunch | ConvertTo-Json
