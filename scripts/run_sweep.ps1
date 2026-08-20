# Runs every config in configs/ cheapest-first: train -> predict -> report -> cost.
# Skips configs whose metrics JSON already exists, so it is safe to re-run
# after any interruption (train.py itself auto-resumes from last.pt).
#
#   .\scripts\run_sweep.ps1 -Data data\EchoNet-Dynamic
#   .\scripts\run_sweep.ps1 -Data data\EchoNet-Dynamic -Stage 3a

param(
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$Stage = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$configs = Get-ChildItem configs\*.yaml
if ($Stage) { $configs = $configs | Where-Object { $_.Name -like "$Stage*" } }

# Cheapest first: sort by frames (from the fN token in the name), then period.
$configs = $configs | Sort-Object {
    if ($_.BaseName -match "_f(\d+)_p(\d+)") { [int]$Matches[1] * 10 + [int]$Matches[2] } else { 999 }
}

foreach ($cfg in $configs) {
    $name = $cfg.BaseName
    $metrics = "results\metrics\$name.metrics.json"
    if (Test-Path $metrics) {
        Write-Host "== $name already complete, skipping" -ForegroundColor DarkGray
        continue
    }
    Write-Host "== $name : train" -ForegroundColor Cyan
    python -m src.train $cfg.FullName --data $Data
    if ($LASTEXITCODE -ne 0) { throw "train failed for $name" }

    Write-Host "== $name : predict (test, all-clips protocol)" -ForegroundColor Cyan
    python -m src.evaluate predict $cfg.FullName "checkpoints\$name\best.pt" `
        --data $Data --out "results\metrics\$name.pred.csv"
    if ($LASTEXITCODE -ne 0) { throw "predict failed for $name" }

    Write-Host "== $name : report (bootstrap metrics)" -ForegroundColor Cyan
    python -m src.evaluate report "results\metrics\$name.pred.csv" --out $metrics
    if ($LASTEXITCODE -ne 0) { throw "report failed for $name" }

    Write-Host "== $name : cost profile" -ForegroundColor Cyan
    python -m src.cost $cfg.FullName --data $Data --out results\costs.csv
    if ($LASTEXITCODE -ne 0) { throw "cost failed for $name" }
}

Write-Host "Sweep pass complete." -ForegroundColor Green
