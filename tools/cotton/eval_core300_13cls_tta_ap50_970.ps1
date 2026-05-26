param(
    [string]$Repo = "D:\cotton_baseline\RT-DETRv4",
    [string]$Python = "D:\miniconda3\envs\cotton\python.exe",
    [string]$Checkpoint = "outputs\_saved_final\baseline_13cls_ap50_0p953041_epoch100\baseline_best_stg2_map.pth",
    [string]$Output = "outputs\tta_stg2_wbf_ms3_vflip"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo

& $Python -u tools\cotton\eval_tta_ap50.py `
  -c configs\cotton\rtv4_hgnetv2_m_cotton_core300img_13cls_train300_purebaseline.yml `
  -r $Checkpoint `
  -d cuda `
  --output $Output `
  --flip `
  --vflip `
  --sizes 640 768 896 `
  --fuse-mode wbf `
  --pre-fuse-topk 700 `
  --score 0.005 `
  --nms 0.45
