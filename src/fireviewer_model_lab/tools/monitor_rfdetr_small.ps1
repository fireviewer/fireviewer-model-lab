[CmdletBinding()]
param(
    [string]$Output = "data\training\rfdetr-small-ground-elite-lowram-v1",
    [string]$LogRoot = $env:TEMP,
    [int]$RefreshSeconds = 15
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outputPath = [IO.Path]::GetFullPath((Join-Path $repositoryRoot $Output))
$pidPath = Join-Path $outputPath "run.pid"
$stdoutPath = Join-Path $LogRoot "fireviewer-rfdetr-small-ground-elite-lowram-v1.out.log"
$stderrPath = Join-Path $LogRoot "fireviewer-rfdetr-small-ground-elite-lowram-v1.err.log"

if (-not (Test-Path -LiteralPath $pidPath)) {
    throw "RF-DETR Small run.pid is missing: $pidPath"
}
$launcherPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()

while ($true) {
    Clear-Host
    Write-Host "FireViewer - RF-DETR Small premium ground" -ForegroundColor Cyan
    Write-Host ("Actualise : {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date))
    Write-Host ("PID launcher : {0}" -f $launcherPid)
    Write-Host ""

    & nvidia-smi `
        --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,pstate `
        --format=csv,noheader

    Write-Host ""
    $metrics = Join-Path $outputPath "metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        Write-Host "Dernieres metriques :" -ForegroundColor Yellow
        Get-Content -LiteralPath $metrics -Tail 8
    } else {
        Write-Host "Initialisation en cours avant la premiere metrique."
    }

    Write-Host ""
    $checkpoints = @(
        Get-ChildItem -LiteralPath $outputPath -File |
            Where-Object { $_.Name -match "^checkpoint_|^last\.ckpt$" } |
            Sort-Object LastWriteTime
    )
    if ($checkpoints.Count -gt 0) {
        $latest = $checkpoints[-1]
        Write-Host ("Dernier checkpoint : {0} ({1:yyyy-MM-dd HH:mm:ss})" -f $latest.Name, $latest.LastWriteTime) -ForegroundColor Green
    } else {
        Write-Host "Aucun checkpoint complet pour le moment."
    }

    Write-Host ""
    if (Test-Path -LiteralPath $stderrPath) {
        $fatal = Select-String -LiteralPath $stderrPath `
            -Pattern "Traceback|CUDA out of memory|RuntimeError|loss=nan|loss: nan" `
            -CaseSensitive:$false
        if ($fatal) {
            Write-Host "Erreur potentiellement fatale :" -ForegroundColor Red
            $fatal | Select-Object -Last 5
        } else {
            Write-Host "Aucun motif d'erreur fatale dans stderr." -ForegroundColor Green
        }
    }

    if (-not (Get-Process -Id $launcherPid -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "Le train est termine. Verifier les metriques et checkpoints." -ForegroundColor Magenta
        break
    }

    Write-Host ""
    Write-Host "Ctrl+C ferme uniquement ce monitoring, pas l'entrainement."
    Start-Sleep -Seconds $RefreshSeconds
}
