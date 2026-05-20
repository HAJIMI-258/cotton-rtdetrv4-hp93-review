# Cotton RT-DETRv4 Training Modules

This repository is a research working copy of RT-DETRv4 with cotton disease and pest detection modules added for code review and follow-up optimization.

## Main Modified Files

- `engine/rtv4/hybrid_encoder.py`
  - Prewitt-Franklin edge-guided enhancement
  - 7x7 local edge refinement branch and edge consistency loss hook
  - Small lesion cross-scale enhancement
  - CARAFE upsampling
  - BiFPN-style fusion
  - Coordinate Attention, ECA, CBAM, LSK
  - FasterNet partial convolution block
  - RepVGG enhancement block
  - Multi-scale gather-distribute context
  - Safe residual gates with configurable trainable/clamped scales
- `engine/rtv4/rtv4_criterion.py`
  - NWD box loss
  - Foreground-aware DINO feature distillation
  - Optional positive class loss weights for hard classes
  - Auxiliary distillation skip logic
- `engine/data/transforms/copy_paste.py`
  - Conservative hard-class target-level copy-paste augmentation
- `engine/solver/det_solver.py`
  - Per-class AP50 extraction
  - AP50 plus hard-class tie-break best checkpoint saving
- `engine/core/yaml_config.py`
  - Teacher model construction support
- `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml`
  - Current improved v1/v6 21-class training config
- `configs/cotton/rtv4_hgnetv2_m_cotton_improved_v2_1.yml`
  - Hard-class copy-paste plus PF local refinement experiment
- `configs/cotton/rtv4_hgnetv2_m_cotton_improved_v2_2_finetune.yml`
  - Fine-tune recipe from current improved v1 AP50-best checkpoint
- `configs/cotton/rtv4_hgnetv2_m_cotton_hp93_v22.yml`
  - Canonical v2.2 alias used for final training/review
- `tools/cotton/train_hp93_v22_3090.ps1`
  - Canonical Windows 3090 launcher for v2.2 fine-tuning

## Current Training Context

- Active cleaned balanced dataset uses `num_classes: 21`, not core9.
- Current improved v1/v6 AP50-best checkpoint is produced by:
  - `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml`
  - `outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth`
- Current reported improved v1 best:
  - AP50: `0.9304486021`
  - epoch: `44`
- Baseline catch-up tracking is documented in:
  - `reports/strategy_review/baseline_vs_improved_v6_strategy_snapshot.md`
  - `reports/strategy_review/baseline_vs_improved_v6_epoch_compare.csv`

## v2.2 Fine-Tune Intent

`improved_v2_2_finetune` should not be treated as another all-module stack. It narrows the change to:

- fine-tuning from improved v1 `best_ap50.pth`;
- enabling/strengthening small lesion enhancement;
- keeping PF local refinement and edge consistency;
- tightening safe module scale to `0.005 -> max 0.02`;
- using conservative hard-class copy-paste;
- using AP50 with hard-class AP50 tie-break for checkpoint saving.

Canonical v2.2 launch command on the original Windows 3090:

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\train_hp93_v22_3090.ps1
```

Preflight details are documented in:

`reports/strategy_review/v2.2_check_report.md`

## Notes

Dataset, pretrained weights, training outputs, and local machine paths are intentionally not committed.

Before full training, set the dataset and teacher/pretrained paths in the cotton config if they differ from the 3090 machine:

```yaml
train_dataloader:
  dataset:
    img_folder: ./data/cotton_rtdetr_dataset_balanced_stratified/images/train
    ann_file: ./data/cotton_coco_balanced_stratified/annotations/instances_train.json
```

The old S/all-module configs are retained for history, but should not be used as the current final comparison recipe. In the current v1/v6 and v2.2 configs, safe residual gates are trainable and clamped:

```yaml
safe_module_trainable_scale: True
```
