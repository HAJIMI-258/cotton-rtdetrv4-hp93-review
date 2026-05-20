# improved_v2_2_finetune Plan

This is a narrow fine-tune plan on top of the current improved v1 AP50-best checkpoint:

`outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth`

The purpose is not to add more large modules. It is to make the existing useful modules contribute more directly to small lesion and hard-class detection.

## Important Corrections

- The active improved v1/v6 config is `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml`.
- That config uses `num_classes: 21`, not core9.
- The active data paths are the balanced stratified 21-class split:
  - `data/cotton_rtdetr_dataset_balanced_stratified`
  - `data/cotton_coco_balanced_stratified`
- In the current 21-class dataset, `leaf_reddening` exists, but `yellowish_leaf` and `damaged_cotton_boll` are absent.
- Therefore hard-class copy-paste and hard-class AP selection currently affect `leaf_reddening` only unless those missing classes are restored.

## Files Added Or Changed

- `configs/cotton/improved_v1_final_config.yml`
- `configs/cotton/rtv4_hgnetv2_m_cotton_improved_v2_2_finetune.yml`
- `tools/cotton/train_improved_v2_2_finetune_3090.ps1`
- `engine/rtv4/rtv4_criterion.py`
- `engine/solver/det_solver.py`
- `engine/core/_config.py`

## v2.2 Changes

1. Fine-tune from improved v1 best AP50 checkpoint, not from scratch.
2. Keep image size at 768 and keep the stable batch recipe.
3. Enable and strengthen small lesion enhancement:
   - `small_lesion_enhance: True`
   - `small_lesion_alpha: 0.15`
4. Keep Prewitt-Franklin refinement complete:
   - 7x7 local refinement
   - edge gate residual fusion
   - edge consistency loss
5. Tighten safe module range:
   - `safe_module_trainable_scale: True`
   - `safe_module_init_scale: 0.005`
   - `safe_module_max_scale: 0.02`
6. Keep core modules active:
   - Prewitt-Franklin
   - SmallLesion
   - CARAFE
   - BiFPN
   - Gather-Distribute
7. Keep non-core attention conservative:
   - Coordinate attention alpha lowered to 0.04
   - ECA disabled
   - CBAM/LSK/FasterNet/RepVGG disabled
8. Fix foreground-aware distillation config to match code fields:
   - `foreground_focus`
   - `fg_weight`
   - `bg_weight`
   - `box_scale`
   - `min_fg_pixels`
9. Add conservative hard-class copy-paste:
   - target names: `leaf_reddening`, `yellowish_leaf`, `damaged_cotton_boll`
   - probability: 0.25
   - max paste per image: 2
   - no Mosaic
   - no MixUp
10. Add class-positive classification weights for matched labels:
   - current active id: `leaf_reddening=16`, weight 1.4
11. Increase NWD weight to 0.30.
12. Save custom best checkpoints:
   - `best_ap50_v2_2.pth`
   - `best_hardclass_v2_2.pth`

## Launch

On the original 3090 machine:

```powershell
cd C:\Users\WYZ\Desktop\cotton\rtdetrv4_hp93\RT-DETRv4
powershell -ExecutionPolicy Bypass -File tools\cotton\train_improved_v2_2_finetune_3090.ps1
```

Direct equivalent:

```powershell
D:\Anaconda3\python.exe -u train.py -c configs\cotton\rtv4_hgnetv2_m_cotton_improved_v2_2_finetune.yml -t outputs\rtv4_hgnetv2_m_cotton_balanced_v6_768\best_ap50.pth -d cuda --seed 0 --use-amp
```

## Acceptance Rule

Do not replace the current improved v1 best unless at least one of these holds:

- AP50 improves by at least 0.005.
- mAP50:95 improves meaningfully while AP50 remains within 0.003.
- hard-class mean AP50 improves by at least 0.02.

If none of those happen, keep v2.2 as a recorded attempt only.
