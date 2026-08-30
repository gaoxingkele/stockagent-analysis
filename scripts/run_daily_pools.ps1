$ErrorActionPreference = 'Stop'

$projectRoot = 'D:\aicoding\stockagent-analysis'
$pythonExe = 'D:\Python314\python.exe'
$sourceRoot = 'D:\aicoding\stock_benchmark'
$sourcePipeline = 'D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py'
$dailyReview = 'D:\aicoding\stockagent-analysis\daily_review.py'
$logDir = 'D:\aicoding\stockagent-analysis\logs\daily_pools'

foreach ($path in @($projectRoot, $pythonExe, $sourceRoot, $sourcePipeline, $dailyReview)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

if (-not (Test-Path -LiteralPath $logDir -PathType Container)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDir "daily_pools_$stamp.log"

function Invoke-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$Stage
    )

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $Stage" |
        Tee-Object -FilePath $logPath -Append
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "$Stage failed with exit code $exitCode"
    }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE $Stage" |
        Tee-Object -FilePath $logPath -Append
}

try {
    Invoke-Logged -Executable $pythonExe `
        -Arguments @($sourcePipeline) `
        -WorkingDirectory $sourceRoot `
        -Stage 'stock_benchmark pool E/F publication'

    Invoke-Logged -Executable $pythonExe `
        -Arguments @($dailyReview) `
        -WorkingDirectory $projectRoot `
        -Stage 'stockagent A-G data refresh and scoring'

    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SUCCESS all pools updated" |
        Tee-Object -FilePath $logPath -Append
    exit 0
}
catch {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED $($_.Exception.Message)" |
        Tee-Object -FilePath $logPath -Append
    exit 1
}
