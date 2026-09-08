[CmdletBinding()]
param(
    [string]$Python = $env:FIREVIEWER_RFDETR_PYTHON,
    [string]$DatasetRoot = $env:FIREVIEWER_DETECTION_DATASET_ROOT,
    [string]$RfHome = $env:RF_HOME,
    [string]$Output = "data\training\rfdetr-small-ground-elite-lowram-v1",
    [string]$LogRoot = $env:TEMP,
    [int]$NumWorkers = 6
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Output))
$pidPath = Join-Path $outputPath "run.pid"
$stdoutPath = Join-Path $LogRoot "fireviewer-rfdetr-small-ground-elite-lowram-v1.out.log"
$stderrPath = Join-Path $LogRoot "fireviewer-rfdetr-small-ground-elite-lowram-v1.err.log"

if ([string]::IsNullOrWhiteSpace($Python) -or -not (Test-Path -LiteralPath $Python)) {
    throw "Set FIREVIEWER_RFDETR_PYTHON to the RF-DETR virtualenv Python executable."
}
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Join-Path $repositoryRoot "data\datasets\fire-smoke-ground-elite-rfdetr-small-v1"
}
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
    throw "RF-DETR Small premium dataset is missing: $DatasetRoot"
}
if ([string]::IsNullOrWhiteSpace($RfHome) -or -not (Test-Path -LiteralPath $RfHome)) {
    throw "Set RF_HOME to the directory containing rf-detr-small.pth."
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "RF-DETR Small is already running with PID $existingPid"
    }
}

$arguments = @(
    "-m", "training.train_rfdetr_large", "train",
    "--variant", "small",
    "--dataset-profile", "ground-elite",
    "--dataset-root", $DatasetRoot,
    "--output", $Output,
    "--epochs", "12",
    "--batch-size", "8",
    "--grad-accum-steps", "4",
    "--num-workers", $NumWorkers,
    "--learning-rate", "1e-4",
    "--encoder-learning-rate", "1e-5",
    "--weight-decay", "1e-4",
    "--resolution", "512",
    "--seed", "420",
    "--rf-home", $RfHome
)

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$startParameters = @{
    FilePath = $Python
    ArgumentList = $arguments
    WorkingDirectory = $repositoryRoot
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
    WindowStyle = "Hidden"
    PassThru = $true
}
$process = Start-Process @startParameters
[IO.File]::WriteAllText($pidPath, "$($process.Id)`n", [Text.UTF8Encoding]::new($false))

[pscustomobject]@{
    Pid = $process.Id
    Output = $outputPath
    Stdout = $stdoutPath
    Stderr = $stderrPath
}
