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

$Log = Join-Path $Output "train_stdout.log"

$Args = @(
    "-u", "train.py",
    "-c", $Config,
    "-d", "cuda",
    "--seed", "0",
    "--use-amp"
)

Write-Host "Starting core300 17-class from-scratch training"
Write-Host "$Python $($Args -join ' ')"

$ErrorActionPreference = "Continue"
& $Python @Args 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE
"python_exit=$ExitCode" | Tee-Object -FilePath $Log -Append
$ErrorActionPreference = "Stop"
exit $ExitCode
