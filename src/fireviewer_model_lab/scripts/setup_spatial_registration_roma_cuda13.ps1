param(
    [string]$DatasetRoot = "D:\fireviewer-data",
    [string]$RomaRoot = ""
)

$ErrorActionPreference = "Stop"
$WorkerRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $WorkerRoot ".venv\Scripts\python.exe"
$ExpectedSourceSha256 = "c95644abd917c62d7bbcad4ff057201aecf61daab282520603c4db606ecac5b4"
$SourceUrl = "https://codeload.github.com/Xecades/AerialExtreMatch/tar.gz/048ab96f84430f3e0f1144f05c94fe1e1f0bca8a"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Worker virtual environment is missing: $Python"
}
if (-not (Test-Path -LiteralPath $DatasetRoot)) {
    throw "Dataset root is missing: $DatasetRoot"
}
if (-not $RomaRoot) {
    $RomaRoot = Join-Path $DatasetRoot "models\aerialextrematch-roma"
}

$SourceArchive = Join-Path $RomaRoot "source\aerialextrematch-roma.tar.gz"
$SourceDirectory = Split-Path -Parent $SourceArchive
New-Item -ItemType Directory -Force -Path $SourceDirectory | Out-Null

Push-Location $WorkerRoot
try {
    & $Python -m pip install `
        --index-url https://download.pytorch.org/whl/cu130 `
        "torch==2.13.0+cu130" `
        "torchvision==0.28.0+cu130"
    & $Python -m pip install -e ".[spatial-training,roma-registration,dev]"

    if (-not (Test-Path -LiteralPath $SourceArchive)) {
        $DownloadCode = @"
from pathlib import Path
from urllib.request import urlopen
url = r'$SourceUrl'
destination = Path(r'$SourceArchive')
partial = destination.with_suffix(destination.suffix + '.partial')
try:
    with urlopen(url, timeout=120) as source, partial.open('wb') as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
    partial.replace(destination)
finally:
    partial.unlink(missing_ok=True)
"@
        & $Python -c $DownloadCode
    }
    $ActualSourceSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $SourceArchive).Hash.ToLowerInvariant()
    if ($ActualSourceSha256 -ne $ExpectedSourceSha256) {
        throw "AerialExtreMatch source archive SHA-256 mismatch"
    }
    & $Python -m pip install --no-deps --force-reinstall $SourceArchive
    Remove-Item -Force -LiteralPath $SourceArchive

    & $Python training\spatial_register_roma.py preflight `
        --dataset-root $DatasetRoot
    & $Python training\spatial_register_roma.py provision `
        --dataset-root $DatasetRoot `
        --roma-root $RomaRoot
    & $Python training\spatial_register_roma.py probe `
        --dataset-root $DatasetRoot `
        --roma-root $RomaRoot
    & $Python training\spatial_register_roma.py launch-plan `
        --dataset-root $DatasetRoot `
        --roma-root $RomaRoot
}
finally {
    Pop-Location
}
