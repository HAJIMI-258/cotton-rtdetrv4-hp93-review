param(
    [string]$Repo = "C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4",
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$Checkpoint = "outputs\rtv4_hgnetv2_m_cotton_v9_clean_carafe_bifpn_fromscratch_768\best_ap50_v9_clean_carafe_bifpn_fromscratch.pth"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

$Config = "configs\cotton\rtv4_hgnetv2_m_cotton_v9_clean_carafe_bifpn_fromscratch_768.yml"
& $Python -u train.py -c $Config -d cuda --seed 0 --use-amp -t $Checkpoint --test-only
exit $LASTEXITCODE
