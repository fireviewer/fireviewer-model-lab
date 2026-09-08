param(
    [Parameter(Mandatory = $true)]
    [int]$TrainPid,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$StdoutLog,

    [Parameter(Mandatory = $true)]
    [string]$StderrLog
)

$ErrorActionPreference = 'SilentlyContinue'

while ($true) {
    Clear-Host
    Write-Host 'FireViewer - RF-DETR Large' -ForegroundColor Cyan
    Write-Host ("Actualise : {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date))
    Write-Host ("PID train : {0}" -f $TrainPid)
    Write-Host ''

    & nvidia-smi `
        --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,pstate `
        --format=csv,noheader

    Write-Host ''
    $metrics = Join-Path $OutputDir 'metrics.csv'
    if (Test-Path -LiteralPath $metrics) {
        Write-Host 'Dernieres metriques :' -ForegroundColor Yellow
        Get-Content -LiteralPath $metrics -Tail 8
    } else {
        Write-Host 'Initialisation en cours avant la premiere metrique.'
    }

    Write-Host ''
    $checkpoints = @(
        Get-ChildItem -LiteralPath $OutputDir -File |
            Where-Object { $_.Name -match '^checkpoint_|^last\.ckpt$' } |
            Sort-Object LastWriteTime
    )
    if ($checkpoints.Count -gt 0) {
        $latest = $checkpoints[-1]
        Write-Host ("Dernier checkpoint : {0} ({1:yyyy-MM-dd HH:mm:ss})" -f $latest.Name, $latest.LastWriteTime) -ForegroundColor Green
    } else {
        Write-Host 'Aucun checkpoint complet pour le moment.'
    }

    Write-Host ''
    if (Test-Path -LiteralPath $StderrLog) {
        Write-Host 'Journal erreurs/avertissements :' -ForegroundColor Yellow
        Get-Content -LiteralPath $StderrLog -Tail 10
    }

    if (-not (Get-Process -Id $TrainPid -ErrorAction SilentlyContinue)) {
        Write-Host ''
        Write-Host 'Le processus de train est termine. Verifier les checkpoints et les metriques.' -ForegroundColor Magenta
        break
    }

    Write-Host ''
    Write-Host "Ctrl+C ferme uniquement ce monitoring, pas l'entrainement."
    Start-Sleep -Seconds 15
}

Read-Host 'Appuyez sur Entree pour fermer cette fenetre'
