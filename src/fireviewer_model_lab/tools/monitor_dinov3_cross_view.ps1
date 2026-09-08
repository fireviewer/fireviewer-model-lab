[CmdletBinding()]
param(
    [string]$Output = "data\training\dinov3-cross-view-retrieval-v1",
    [string]$LogRoot = $env:TEMP
)

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Output))
$pidPath = Join-Path $outputPath "run.pid"
$metricsPath = Join-Path $outputPath "metrics.csv"
$stdoutPath = Join-Path $LogRoot "fireviewer-dinov3-cross-view-v1.out.log"
$stderrPath = Join-Path $LogRoot "fireviewer-dinov3-cross-view-v1.err.log"

$runPid = if (Test-Path -LiteralPath $pidPath) {
    [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
} else {
    $null
}
$process = if ($null -ne $runPid) {
    Get-Process -Id $runPid -ErrorAction SilentlyContinue
} else {
    $null
}

[pscustomobject]@{
    Pid = $runPid
    Running = $null -ne $process
    ProcessName = if ($process) { $process.ProcessName } else { $null }
    StartTime = if ($process) { $process.StartTime } else { $null }
    MetricsRows = if (Test-Path -LiteralPath $metricsPath) {
        @(Import-Csv -LiteralPath $metricsPath).Count
    } else {
        0
    }
}

if (Test-Path -LiteralPath $metricsPath) {
    Import-Csv -LiteralPath $metricsPath | Select-Object -Last 3
}
if (Test-Path -LiteralPath $stdoutPath) {
    Get-Content -LiteralPath $stdoutPath -Tail 8
}
if (Test-Path -LiteralPath $stderrPath) {
    Get-Content -LiteralPath $stderrPath -Tail 12
}
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
