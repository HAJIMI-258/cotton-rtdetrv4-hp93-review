# Core300img 13-Class Route

This route replaces the earlier instance-balanced `core300_17cls` task with an
image-count-balanced task for cleaner paper tables.

## Definition

- Source pool for train/val: original `balanced_stratified` train + val only.
- Test policy: original test is sampled proportionally; no test image enters train/val.
- Retained classes: classes with at least 300 unique train+val images.
- Removed classes: original low-sample classes plus classes with fewer than 300 unique train+val images.
- Balance target: 300 training images per retained class.
- Validation target: about 45 images per retained class when enough remaining train+val images are available.
- Test target: about 45 images per retained class from the original test split when enough images are available.
- Dataset name: `cotton_core300img_13cls_train300`.

## Retained Classes

1. `cotton_aphid`
2. `leaf_hopper_jassids`
3. `american_bollworm`
4. `bacterial_leaf_blight`
5. `alternaria_leaf_spot`
6. `cotton_leaf_curl_virus`
7. `fusarium_wilt`
8. `verticillium_wilt`
9. `leaf_variegation`
10. `leaf_reddening`
11. `herbicide_growth_damage`
12. `open_cotton_boll`
13. `healthy`

## Removed Classes

- `mealy_bug`
- `red_cotton_bug`
- `army_worm`
- `powdery_mildew`
- `boll_rot`
- `pink_bollworm`
- `thrips`
- `whitefly`

`pink_bollworm` has 288 train+val images, which is close but still below the
strict 300-training-image paper threshold, so it is removed in this clean route.
`mealy_bug` has 315 train+val images, so it can provide 300 training images but
cannot also provide the proportional 45-image validation split without
duplicating images. It is therefore removed from the final balanced paper route.

## Build

```powershell
python tools/cotton/build_core300img_13cls_dataset.py `
  --src-ann-dir data/cotton_coco_balanced_stratified/annotations `
  --src-img-root data/cotton_rtdetr_dataset_balanced_stratified/images `
  --dst-root data/cotton_core300img_13cls_train300 `
  --target-train-images 300 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --seed 0
```

## Train

Improved route:

```powershell
powershell -ExecutionPolicy Bypass -File tools/cotton/train_core300img_13cls.ps1
```

Clean baseline:

```powershell
powershell -ExecutionPolicy Bypass -File tools/cotton/train_core300img_13cls_baseline.ps1
```

## AP50-Oriented Inference Result

The saved clean baseline checkpoint can be evaluated with multi-scale H/V flip
TTA and weighted boxes fusion:

```powershell
powershell -ExecutionPolicy Bypass -File tools/cotton/eval_core300_13cls_tta_ap50_970.ps1
```

Verified result on the 13-class task:

- AP50: `0.970076`
- mAP50:95: `0.876169`
- AP75: `0.890674`

Detailed output is in `analysis/core300img_13cls/tta_ap50_970/`.

## Notes

This is a 13-class simplified task. It must not be described as directly
comparable to the original 21-class task or the previous 17-class simplified
task without stating the class filtering rule.
