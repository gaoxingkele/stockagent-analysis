$ErrorActionPreference = 'Stop'

$taskName = 'StockAgent-DailyPools-2100'
$runner = Join-Path $PSScriptRoot 'run_daily_pools.ps1'
$powershellExe = Join-Path $PSHOME 'powershell.exe'

foreach ($path in @($runner, $powershellExe)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file does not exist: $path"
    }
}

$taskCommand = '"{0}" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{1}"' -f `
    $powershellExe, $runner

& schtasks.exe /Create `
    /TN $taskName `
    /TR $taskCommand `
    /SC DAILY `
    /ST 21:00 `
    /F

if ($LASTEXITCODE -ne 0) {
    throw "Failed to register scheduled task $taskName (exit $LASTEXITCODE)"
}

& schtasks.exe /Query /TN $taskName /V /FO LIST
if ($LASTEXITCODE -ne 0) {
    throw "Task was created but verification failed: $taskName"
}
