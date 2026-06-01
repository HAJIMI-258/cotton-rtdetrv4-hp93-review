param(
    [string]$Repo = "C:\Users\a1366\Desktop\cotton-rtdetrv4-hp93-public",
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$Checkpoint = "outputs\_saved_final\baseline_13cls_ap50_0p953041_epoch100\baseline_best_ap50_epoch100.pth",
    [string]$BaseConfig = "configs/cotton/rtv4_hgnetv2_m_cotton_core300img_13cls_train300_purebaseline.yml",
    [string]$RootOut = "outputs/ap50_sweep_core300img_13cls_nmsvote"
)

$ErrorActionPreference = "Stop"
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
Set-Location $Repo
New-Item -ItemType Directory -Force -Path $RootOut | Out-Null

$variants = @(
    @{name="raw"; score=0.0; nms=$null; vote=$null; power=1.0},
    @{name="nms_s001_iou080"; score=0.001; nms=0.80; vote=$null; power=1.0},
    @{name="nms_s001_iou085"; score=0.001; nms=0.85; vote=$null; power=1.0},
    @{name="vote_s001_nms082_v070"; score=0.001; nms=0.82; vote=0.70; power=1.0},
    @{name="vote_s001_nms085_v070"; score=0.001; nms=0.85; vote=0.70; power=1.0},
    @{name="vote_s001_nms085_v072"; score=0.001; nms=0.85; vote=0.72; power=1.0},
    @{name="vote_s001_nms088_v072"; score=0.001; nms=0.88; vote=0.72; power=1.0},
    @{name="vote_s001_nms085_v075"; score=0.001; nms=0.85; vote=0.75; power=1.0},
    @{name="vote_s002_nms085_v072"; score=0.002; nms=0.85; vote=0.72; power=1.0},
    @{name="vote_s001_nms085_v072_p15"; score=0.001; nms=0.85; vote=0.72; power=1.5}
)

function Write-VariantConfig($v) {
    $cfg = Join-Path $RootOut ("config_" + $v.name + ".yml")
    $nmsLine = if ($null -eq $v.nms) { "  nms_iou_threshold: null`n" } else { "  nms_iou_threshold: $($v.nms)`n" }
    $voteLine = if ($null -eq $v.vote) { "  box_voting_iou_threshold: null`n" } else { "  box_voting_iou_threshold: $($v.vote)`n" }
    $text = @"
__include__: [
  '../../$BaseConfig'
]

output_dir: ./$RootOut/$($v.name)
summary_dir: ./$RootOut/$($v.name)/tensorboard
eval_spatial_size: [768, 768]

PostProcessor:
  num_top_queries: 300
  score_threshold: $($v.score)
$nmsLine$voteLine  box_voting_score_power: $($v.power)

val_dataloader:
  dataset:
    img_folder: ./data/cotton_core300img_13cls_train300/images/val
    ann_file: ./data/cotton_core300img_13cls_train300/annotations/instances_val.json
    transforms:
      ops:
        - {type: Resize, size: [768, 768]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  shuffle: False
  total_batch_size: 8
  num_workers: 4
  drop_last: False
"@
    Set-Content -Path $cfg -Value $text -Encoding ASCII
    return $cfg
}

function Parse-Metrics($logPath) {
    $text = Get-Content $logPath -Raw
    $vals = [regex]::Matches($text, '= ([0-9]+\.[0-9]+)') | ForEach-Object { [double]$_.Groups[1].Value }
    if ($vals.Count -ge 12) {
        return @{
            map=$vals[0]; ap50=$vals[1]; ap75=$vals[2]; aps=$vals[3]; apm=$vals[4]; apl=$vals[5]; ar100=$vals[8]
        }
    }
    return @{map=$null; ap50=$null; ap75=$null; aps=$null; apm=$null; apl=$null; ar100=$null}
}

$rows = @()
foreach ($v in $variants) {
    $cfg = Write-VariantConfig $v
    $out = Join-Path $RootOut $v.name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    $log = Join-Path $out "eval.log"
    Write-Host "=== EVAL $($v.name) ==="
    $ErrorActionPreference = "Continue"
    & $Python -u train.py -c $cfg -d cuda --test-only -r $Checkpoint 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $m = Parse-Metrics $log
    $rows += [PSCustomObject]@{
        name=$v.name; score=$v.score; nms=$v.nms; vote=$v.vote; power=$v.power; exit=$exitCode
        map=$m.map; ap50=$m.ap50; ap75=$m.ap75; aps=$m.aps; apm=$m.apm; apl=$m.apl; ar100=$m.ar100
    }
    $rows | Sort-Object ap50 -Descending | Export-Csv -Path (Join-Path $RootOut "nmsvote_sweep_results.csv") -NoTypeInformation -Encoding UTF8
}

$rows | Sort-Object ap50 -Descending | Format-Table -AutoSize
