# Core300 13-class AP50-Oriented TTA Result

This result is for the simplified 13-class `core300img` task. It is not the original 21-class task.

## Best AP50 Result

Checkpoint:

`outputs/_saved_final/baseline_13cls_ap50_0p953041_epoch100/baseline_best_stg2_map.pth`

Inference strategy:

- multi-scale evaluation: `640 + 768 + 896`
- horizontal flip + vertical flip TTA
- weighted boxes fusion
- pre-fuse top-k: `700`
- WBF score mode: `consensus`, expected views: `12`
- score threshold: `0.005`
- `open_cotton_boll` score threshold: `0.02`
- WBF IoU threshold: `0.45`

Metrics:

| metric | value |
|---|---:|
| AP50 | 0.970207 |
| mAP50:95 | 0.876233 |
| AP75 | 0.890689 |
| APS | 0.155738 |
| APM | 0.640794 |
| APL | 0.888779 |

Compared with the saved plain baseline checkpoint result `AP50=0.953041`, the AP50-oriented inference route improves AP50 by about `+0.0172`.

## Per-class AP50

| class | AP50 |
|---|---:|
| cotton_aphid | 0.950410 |
| leaf_hopper_jassids | 0.920962 |
| american_bollworm | 0.985768 |
| bacterial_leaf_blight | 0.948856 |
| alternaria_leaf_spot | 1.000000 |
| cotton_leaf_curl_virus | 0.981752 |
| fusarium_wilt | 0.975904 |
| verticillium_wilt | 0.971700 |
| leaf_variegation | 1.000000 |
| leaf_reddening | 0.985651 |
| herbicide_growth_damage | 1.000000 |
| open_cotton_boll | 0.898300 |
| healthy | 0.992904 |

## Reproduce

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\eval_core300_13cls_tta_ap50_970.ps1
```

Equivalent command:

```powershell
D:\miniconda3\envs\cotton\python.exe -u tools\cotton\eval_tta_ap50.py `
  -c configs\cotton\rtv4_hgnetv2_m_cotton_core300img_13cls_train300_purebaseline.yml `
  -r outputs\_saved_final\baseline_13cls_ap50_0p953041_epoch100\baseline_best_stg2_map.pth `
  -d cuda `
  --output outputs\tta_stg2_wbf_ms3_vflip `
  --flip `
  --vflip `
  --sizes 640 768 896 `
  --fuse-mode wbf `
  --pre-fuse-topk 700 `
  --wbf-score-mode consensus `
  --wbf-expected-views 12 `
  --score 0.005 `
  --nms 0.45 `
  --class-score "open_cotton_boll=0.02"
```

## Note

This is an AP50-oriented inference optimization. It improves the target AP50 metric, while mAP50:95 stays close to the plain checkpoint and can move slightly depending on fusion thresholds.
