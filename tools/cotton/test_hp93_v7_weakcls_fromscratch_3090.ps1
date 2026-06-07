param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$Checkpoint = "outputs\rtv4_hgnetv2_m_cotton_v7_weakcls_fromscratch_768\best_ap50_v7_weakcls_fromscratch.pth"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

& $Python -u train.py `
    -c configs\cotton\rtv4_hgnetv2_m_cotton_v7_weakcls_fromscratch_768.yml `
    -r $Checkpoint `
    -d cuda `
    --test-only `
    --use-amp
