param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_v9_clean_carafe_bifpn_fromscratch_768.yml"
$Output = "outputs\rtv4_hgnetv2_m_cotton_v9_clean_carafe_bifpn_fromscratch_768"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Log = Join-Path $Output "train_stdout.log"
$Err = Join-Path $Output "train_stderr.log"
"START $(Get-Date -Format o)" | Set-Content -Path $Log -Encoding utf8

$Args = @("-u", "train.py", "-c", $Config, "-d", "cuda", "--seed", "0", "--use-amp")
& $Python @Args 2>> $Err | Tee-Object -FilePath $Log -Append
$ExitCode = $LASTEXITCODE
"EXITCODE=$ExitCode $(Get-Date -Format o)" | Tee-Object -FilePath $Log -Append
exit $ExitCode
