param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit.yml"
$Output = "outputs\rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Log = Join-Path $Output "train_stdout.log"
$Args = @(
    "-u", "train.py",
    "-c", $Config,
    "-d", "cuda",
    "--seed", "0",
    "--use-amp"
)

Write-Host "Starting from-scratch BoxFit v9 896 run"
Write-Host "$Python $($Args -join ' ')"
$ErrorActionPreference = "Continue"
& $Python @Args 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE
"python_exit=$ExitCode" | Tee-Object -FilePath $Log -Append
$ErrorActionPreference = "Stop"
exit $ExitCode

