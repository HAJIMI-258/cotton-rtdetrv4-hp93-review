# cotton_core300_17cls

`cotton_core300_17cls` is a simplified 17-class cotton detection task. It is not the same task as the original 21-class split, so its AP cannot be directly claimed as a 21-class improvement.

## Removed Classes

These classes are removed because their original train instances are below 300:

- `red_cotton_bug`
- `army_worm`
- `powdery_mildew`
- `boll_rot`

## Retained Classes

- `cotton_aphid`
- `mealy_bug`
- `leaf_hopper_jassids`
- `american_bollworm`
- `pink_bollworm`
- `thrips`
- `whitefly`
- `bacterial_leaf_blight`
- `alternaria_leaf_spot`
- `cotton_leaf_curl_virus`
- `fusarium_wilt`
- `verticillium_wilt`
- `leaf_variegation`
- `leaf_reddening`
- `herbicide_growth_damage`
- `open_cotton_boll`
- `healthy`

## Dataset Build

```powershell
python tools/cotton/build_core300_17cls_dataset.py `
  --src-ann-dir data/cotton_coco_balanced_stratified/annotations `
  --src-img-root data/cotton_rtdetr_dataset_balanced_stratified/images `
  --dst-root data/cotton_core300_17cls `
  --target-train-instances 300 `
  --val-ratio 0.15 `
  --seed 0
```

Generated files:

- `data/cotton_core300_17cls/annotations/instances_train.json`
- `data/cotton_core300_17cls/annotations/instances_val.json`
- `data/cotton_core300_17cls/annotations/instances_test.json`
- `data/cotton_core300_17cls/reports/class_distribution_core300_17cls.csv`
- `data/cotton_core300_17cls/reports/removed_classes.csv`
- `data/cotton_core300_17cls/reports/category_id_mapping.csv`
- `data/cotton_core300_17cls/reports/split_integrity_report.md`

## Generated Split Summary

The checked seed-0 build generated on the 3090 machine produced:

| split | images | annotations |
| --- | ---: | ---: |
| train | 3302 | 5387 |
| val | 383 | 917 |
| test | 2813 | 4986 |

Integrity results:

- Cross-split `file_name` leakage: none
- Missing image references: 0
- Bad category references: 0
- Empty/negative/zero-area bboxes: 0
- Missing image paths on the build machine: 0
- Category IDs are continuous: 0..16
- Source annotations unchanged: True

Per-class distribution is available in:

- `data/cotton_core300_17cls/reports/class_distribution_core300_17cls.csv`
- `data/cotton_core300_17cls/reports/split_integrity_report.md`

One sampling caveat is intentional and recorded in the report: `open_cotton_boll` has dense multi-instance images, so the train split keeps 25 images but reaches 586 instances. This avoids selecting too few images or deleting visible retained-class boxes inside selected images.

## Training

```powershell
powershell -ExecutionPolicy Bypass -File tools/cotton/train_core300_17cls.ps1
```

## Evaluation

```powershell
powershell -ExecutionPolicy Bypass -File tools/cotton/eval_core300_17cls.ps1
```

## Comparison Rule

This route is for a simplified 17-class experiment. Results can be reported as a filtered core-category experiment, but should not be described as a direct 21-class improvement over the v6 0.9304 AP50 model.
