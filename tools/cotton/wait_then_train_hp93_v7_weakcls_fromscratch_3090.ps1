param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$WaitMatch = "v7_weakcls_ft_768",
    [int]$PollSeconds = 120
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

function Test-TrainingStillRunning {
    param([string]$Match)
    $process = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*train.py*" -and
            $_.CommandLine -like "*$Match*" -and
            $_.Name -like "python*"
        } |
        Select-Object -First 1
    return $null -ne $process
}

while (Test-TrainingStillRunning -Match $WaitMatch) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$now] Waiting for active training matching '$WaitMatch'..."
    Start-Sleep -Seconds $PollSeconds
}

powershell -ExecutionPolicy Bypass -File tools\cotton\train_hp93_v7_weakcls_fromscratch_3090.ps1 `
    -Repo $Repo `
    -Python $Python `
    -CleanOutput
