param(
    [string]$Repo = "",
    [string]$Python = "D:\Anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Repo)) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_core300_17cls.yml"
$Output = "outputs\rtv4_hgnetv2_m_cotton_core300_17cls"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Checkpoints = @(
    "best_ap50.pth",
    "best_stg2.pth",
    "last.pth"
)

foreach ($Name in $Checkpoints) {
    $Checkpoint = Join-Path $Output $Name
    if (-not (Test-Path $Checkpoint)) {
        Write-Host "skip missing checkpoint: $Checkpoint"
        continue
    }
    $SafeName = [IO.Path]::GetFileNameWithoutExtension($Name)
    $Log = Join-Path $Output "eval_${SafeName}.log"
    Write-Host "Evaluating $Checkpoint"
    & $Python -u train.py `
        -c $Config `
        -d cuda `
        --test-only `
        -r $Checkpoint 2>&1 | Tee-Object -FilePath $Log
}
