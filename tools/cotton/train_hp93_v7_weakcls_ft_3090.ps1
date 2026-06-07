param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$Checkpoint = "outputs\rtv4_hgnetv2_m_cotton_balanced_v6_768\best_ap50.pth"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_v7_weakcls_ft_768.yml"
$Output = "outputs\rtv4_hgnetv2_m_cotton_v7_weakcls_ft_768"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Log = Join-Path $Output "train_stdout.log"
$Args = @(
    "-u", "train.py",
    "-c", $Config,
    "-t", $Checkpoint,
    "-d", "cuda",
    "--seed", "0",
    "--use-amp"
)

Write-Host "Starting v7 weak-class fine-tune from $Checkpoint"
Write-Host "$Python $($Args -join ' ')"
$ErrorActionPreference = "Continue"
& $Python @Args 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE
"python_exit=$ExitCode" | Tee-Object -FilePath $Log -Append
$ErrorActionPreference = "Stop"
exit $ExitCode
