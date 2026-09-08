param(
    [string]$DatasetRoot = "D:\fireviewer-data"
)

$ErrorActionPreference = "Stop"
$WorkerRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $WorkerRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Worker virtual environment is missing: $Python"
}

& $Python -m pip install `
    --index-url https://download.pytorch.org/whl/cu130 `
    "torch==2.13.0+cu130" `
    "torchvision==0.28.0+cu130"

Push-Location $WorkerRoot
try {
    & $Python -m pip install -e ".[spatial-training,spatial-train,dev]"
    & $Python training\spatial_train_qwen.py prepare --dataset-root $DatasetRoot
    & $Python training\spatial_train_qwen.py launch-plan --dataset-root $DatasetRoot
    Write-Output "Qwen fire-pointing training remains locked; no probe or optimizer was started."
}
finally {
    Pop-Location
}
