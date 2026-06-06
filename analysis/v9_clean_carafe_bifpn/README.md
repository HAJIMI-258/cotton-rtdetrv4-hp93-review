# V9 Clean CARAFE + BiFPN From-Scratch Experiment

This branch records the clean single-variable V9 route for the full 21-class cotton detection split. It keeps the v6 training/data/loss baseline and only enables CARAFE upsampling plus BiFPN fusion with safe trainable residual scaling.

## Uploaded Scope

- Config: `configs/cotton/rtv4_hgnetv2_m_cotton_v9_clean_carafe_bifpn_fromscratch_768.yml`
- Training entry: `tools/cotton/train_hp93_v9_clean_carafe_bifpn_fromscratch_3090.ps1`
- Test entry: `tools/cotton/test_hp93_v9_clean_carafe_bifpn_fromscratch_3090.ps1`
- Parsed epoch metrics: `analysis/v9_clean_carafe_bifpn/v9_metrics_by_epoch.csv`
- Best per-class AP50: `analysis/v9_clean_carafe_bifpn/v9_best_per_class_ap50.csv`
- Best metadata JSON: `analysis/v9_clean_carafe_bifpn/best_ap50_v9_clean_carafe_bifpn_fromscratch.json`

Weights (`.pth`) are intentionally not uploaded.

## Key Configuration

- Dataset/classes: full 21-class split, `num_classes: 21`.
- Enabled: `carafe_upsample: True`, `bifpn_fusion: True`.
- Disabled: PF edge, small lesion, output refine, CoordAttention, ECA, CBAM, LSK, FasterNet PConv, RepVGG, Gather-Distribute.
- Safe scale: init `0.005`, trainable `True`, max `0.03`.
- Best selection: `ap50_hard_tie` with empty hard-class list, so this acts as AP50 best selection.

## Current Best Snapshot

- Best epoch: 26
- mAP50:95: 0.7179
- AP50: 0.8656
- AP75: 0.7089
- APS/APM/APL: 0.2195 / 0.6258 / 0.7487

Reference v6 baseline best used for comparison in the project notes: AP50 `0.9304`, mAP50:95 `0.8101`, AP75 `0.8255`.

## Trend Summary

| epoch | mAP50:95 | AP50 | AP75 | APS | APM | APL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4412 | 0.4693 | 0.4382 | 0.0333 | 0.0828 | 0.4733 |
| 1 | 0.5461 | 0.6092 | 0.5362 | 0.0928 | 0.4326 | 0.5822 |
| 2 | 0.6121 | 0.7199 | 0.5954 | 0.1359 | 0.4772 | 0.6519 |
| 3 | 0.6266 | 0.7363 | 0.6233 | 0.1552 | 0.5250 | 0.6647 |
| 4 | 0.6179 | 0.7215 | 0.6114 | 0.1469 | 0.4939 | 0.6560 |
| 5 | 0.6236 | 0.7318 | 0.6175 | 0.1506 | 0.4966 | 0.6615 |
| 10 | 0.6744 | 0.7907 | 0.6717 | 0.1671 | 0.5564 | 0.7115 |
| 15 | 0.7055 | 0.8436 | 0.6955 | 0.1880 | 0.5865 | 0.7417 |
| 20 | 0.7185 | 0.8636 | 0.7044 | 0.2030 | 0.4943 | 0.7530 |
| 24 | 0.7164 | 0.8610 | 0.7109 | 0.2137 | 0.6259 | 0.7487 |
| 25 | 0.7188 | 0.8584 | 0.7094 | 0.2159 | 0.6273 | 0.7499 |
| 26 | 0.7179 | 0.8656 | 0.7089 | 0.2195 | 0.6258 | 0.7487 |
| 27 | 0.7196 | 0.8569 | 0.7101 | 0.2202 | 0.6313 | 0.7503 |
| 28 | 0.7225 | 0.8589 | 0.7116 | 0.2219 | 0.6180 | 0.7525 |
| 29 | 0.7238 | 0.8550 | 0.7108 | 0.2232 | 0.6178 | 0.7542 |

## Strict Read

At the current stage, V9 is not a valid improvement over v6. AP50 rises quickly in the first 15 epochs, then plateaus around `0.856-0.866`, far below the v6 AP50 `0.9304`. mAP50:95 and APS are still increasing slowly, but this does not satisfy the paper requirement of a clear AP50 gain.

Likely interpretation: clean CARAFE + BiFPN is stable and trainable, but by itself does not solve the weak-class/small-object ranking bottleneck on the 21-class split. It should be treated as a negative/diagnostic experiment unless later epochs reverse the AP50 plateau.

## Weak Classes At Best Epoch

Lowest AP50 classes at the best epoch:
- `mealy_bug`: 0.3955
- `american_bollworm`: 0.6423
- `whitefly`: 0.7525
- `army_worm`: 0.7588
- `thrips`: 0.7592
- `powdery_mildew`: 0.7690
- `cotton_aphid`: 0.8050
- `red_cotton_bug`: 0.8158

These values support the existing diagnosis that small/similar insect classes, especially mealy bug and several bollworm/whitefly categories, remain the main bottleneck.
