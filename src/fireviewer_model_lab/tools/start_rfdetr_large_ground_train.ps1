[CmdletBinding()]
param(
    [string]$Python = $env:FIREVIEWER_RFDETR_PYTHON,
    [string]$DatasetRoot = 'data\datasets\fire-smoke-ground-only-rfdetr-large-v1',
    [string]$RfHome = $env:RF_HOME,
    [string]$Output = 'data\training\rfdetr-large-ground-fire-smoke-v2',
    [string]$LogRoot = $env:TEMP,
    [int]$NumWorkers = 8,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 32
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$datasetPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $DatasetRoot))
$outputPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Output))
$pidPath = Join-Path $outputPath 'run.pid'
$stdoutPath = Join-Path $LogRoot 'fireviewer-rfdetr-large-ground-v2.out.log'
$stderrPath = Join-Path $LogRoot 'fireviewer-rfdetr-large-ground-v2.err.log'

if (-not (Test-Path -LiteralPath $Python)) { throw "RF-DETR Python missing: $Python" }
if (-not (Test-Path -LiteralPath $datasetPath)) { throw "Ground dataset missing: $datasetPath" }
if (-not (Test-Path -LiteralPath $RfHome)) { throw "RF-DETR weights directory missing: $RfHome" }
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "RF-DETR Large ground-only is already running with PID $existingPid"
    }
}

$arguments = @(
    '-m', 'training.train_rfdetr_large', 'train',
    '--variant', 'large',
    '--dataset-profile', 'ground-only',
    '--dataset-root', $datasetPath,
    '--output', $Output,
    '--epochs', '3',
    '--batch-size', $BatchSize,
    '--grad-accum-steps', $GradAccumSteps,
    '--num-workers', $NumWorkers,
    '--learning-rate', '1e-4',
    '--encoder-learning-rate', '1e-5',
    '--weight-decay', '1e-4',
    '--resolution', '512',
    '--seed', '420',
    '--rf-home', $RfHome
)

New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$process = Start-Process -FilePath $Python -ArgumentList $arguments `
    -WorkingDirectory $repositoryRoot -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
[IO.File]::WriteAllText($pidPath, "$($process.Id)`n", [Text.UTF8Encoding]::new($false))

[pscustomobject]@{
    Pid = $process.Id
    Output = $outputPath
    Stdout = $stdoutPath
    Stderr = $stderrPath
}
