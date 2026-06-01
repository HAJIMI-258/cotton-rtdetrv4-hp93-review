# Core300img 13-Class AP50 v2 Failure Analysis

## Conclusion

The AP50 v2 / BoxFit route underperformed the clean baseline on the same 13-class
core300img dataset. It should not be used as the final improved result.

The strongest current route is the saved clean baseline checkpoint plus
multi-scale flip TTA and weighted boxes fusion.

## Verified Results

| run | epoch/checkpoint | mAP50:95 | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean baseline | epoch 100 | 0.8791 | 0.9530 | 0.8889 | 0.1492 | 0.5379 | 0.8883 |
| AP50 v2 / BoxFit, cloud | epoch 27 | 0.8387 | 0.9307 | 0.8343 | 0.1423 | 0.4531 | 0.8484 |
| AP50 v2 / BoxFit, 3090 | epoch 94 latest | 0.8425 | 0.9223 | 0.8469 | 0.1459 | 0.4991 | 0.8511 |
| baseline + NMS 0.85 | epoch 100 checkpoint | about 0.880 | about 0.957 | about 0.891 | about 0.151 | about 0.542 | about 0.889 |
| baseline + 640/768/896 H/V flip consensus-WBF | stg2 checkpoint | 0.8762 | 0.9702 | 0.8907 | 0.1557 | 0.6408 | 0.8888 |
| train300 improved + NMS/box-voting | best_stg2 checkpoint | 0.881 | 0.960 | 0.891 | 0.150 | 0.607 | 0.891 |
| train300 improved + 640/768/896 H/V flip consensus-WBF | best_stg2 checkpoint | 0.8811 | 0.9676 | 0.8869 | 0.1600 | 0.5744 | 0.8946 |

## Why AP50 v2 Got Worse

AP50 v2 changed too many variables at once:

- resolution changed from 768 to 896;
- evaluation used NMS=0.65, which is too aggressive for dense cotton boll scenes;
- DINO teacher distillation was enabled;
- PF edge, small-lesion, output-refine, CARAFE, BiFPN, CoordAttention and ECA were enabled together;
- extra box losses were added: NWD, MPDIoU, Alpha-IoU and AP50-margin;
- batch changed from total_batch_size 8 + accumulation 2 to total_batch_size 4 + accumulation 4.

The logs show that AP50-margin quickly became very small, so it did not keep
pushing the final AP50 upward. Meanwhile the added losses and modules reduced
AP75, APM and APL substantially, which means the detector learned a worse global
ranking/localization balance than the clean baseline.

## NMS Sweep

Test-only sweep on the saved baseline checkpoint showed that conservative NMS is
useful:

| postprocess | AP50 |
|---|---:|
| raw baseline | 0.953 |
| score=0.001, NMS=0.85 | 0.957 |
| score=0.001, NMS=0.88 | 0.957 |
| score=0.002, NMS=0.90 | 0.956 |

The best AP50-oriented inference command is:

```text
powershell -ExecutionPolicy Bypass -File tools\cotton\eval_core300_13cls_tta_ap50_970.ps1
```

## 2026-06-01 Output-Layer Refinement Check

The code now includes optional class-aware NMS plus same-class box voting inside
`engine/rtv4/postprocessor.py`. Test-only verification on the WYZ 3090 machine
with `outputs\rtv4_hgnetv2_m_cotton_core300img_13cls_train300\best_stg2.pth`
showed:

| variant | mAP50:95 | AP50 | AP75 | APS | APM | APL |
|---|---:|---:|---:|---:|---:|---:|
| raw output | 0.881 | 0.955 | 0.891 | 0.148 | 0.603 | 0.892 |
| score=0.001, NMS=0.80 | 0.882 | 0.958 | 0.893 | 0.151 | 0.607 | 0.893 |
| score=0.001, NMS=0.82, box-vote=0.70 | 0.881 | 0.960 | 0.891 | 0.150 | 0.607 | 0.891 |
| multi-scale H/V flip consensus-WBF | 0.8811 | 0.9676 | 0.8869 | 0.1600 | 0.5744 | 0.8946 |

Conclusion: box voting gives a real but limited AP50 gain. The only route that
currently opens a larger AP50 gap is the multi-scale consistency/WBF inference
module. It should be presented as an output decision refinement module, not as a
new backbone training module.

## Next Step

Do not continue AP50 v2 / BoxFit training. For the paper route, use the clean
baseline as the stable model and report a separate postprocess optimization
experiment. If another training improvement is needed, start from the clean
baseline configuration and change only one factor at a time.
