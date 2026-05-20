# Review Points For Pro

Use these as the main inspection questions. The code excerpts and source
snapshots are in the same directory.

## 1. PFEdgeRefine / Prewitt-Franklin

Files:

- `source_snapshots/engine_rtv4_hybrid_encoder.py`
- `extracted_modules/engine__rtv4__hybrid_encoder__PrewittFranklinEdgeGuidedEnhance.py`

Check whether the implementation correctly matches the intended paper logic:

- Prewitt rough edge prior and `T = k * Gmax`.
- 7x7 Franklin moment templates.
- `M00`, `M11`, `M20`, `M31`, `M40`.
- `phi`, `l1`, `l2`, `l`, `k`.
- Eq. 23-style edge decision.
- 7x7 local refinement branch.
- edge gate plus residual fusion.
- edge consistency loss path.

Important wording issue: this code uses Prewitt-Franklin as a feature prior/gate
inside a detector. It does not output final industrial sub-pixel edge coordinates
as a standalone measurement algorithm.

## 2. Small Lesion And Multi-Scale Fusion

Files:

- `extracted_modules/engine__rtv4__hybrid_encoder__SmallLesionCrossScaleEnhance.py`
- `extracted_modules/engine__rtv4__hybrid_encoder__CARAFEUpsample.py`
- `extracted_modules/engine__rtv4__hybrid_encoder__BiFPNFusionBlock.py`

Check whether the residual gates and safe scales are likely too weak or too
strong, especially compared with `safe_module_init_scale`,
`safe_module_trainable_scale`, and `safe_module_max_scale` in each config.

## 3. Attention / Extra Blocks

Files:

- `ECALayer`
- `CBAMLayer`
- `CoordinateAttentionExact`
- `LSKBlock`
- `FasterNetBlock`
- `RepVGGEnhanceBlock`
- `GatherDistributeContext`

The completed improved run disabled CBAM, LSK, FasterNet, RepVGG, and
gather-distribute. The all-module config enabled them, but that config should be
treated as a risky stack rather than the clean final run.

## 4. Losses

File:

- `extracted_modules/engine__rtv4__rtv4_criterion__RTv4Criterion.py`

Check:

- NWD formula and weighting.
- Whether NWD is added in the same `loss_boxes` path as bbox/GIoU.
- Foreground-aware distillation fields:
  - `foreground_focus`
  - `fg_weight`
  - `bg_weight`
  - `box_scale`
  - `min_fg_pixels`
- Whether `enabled` is read or ignored.
- Edge consistency loss target/source alignment.
- Whether class loss weights match the actual category ids.

## 5. HardClassCopyPaste

Files:

- `source_snapshots/engine_data_transforms_copy_paste.py`
- `extracted_modules/engine__data__transforms__copy_paste__HardClassCopyPaste.py`

Check:

- It uses annotated boxes as rectangular ROI crops with context.
- It does not use segmentation masks.
- It skips absent target class names.
- Current documented 21-class context lacks `yellowish_leaf` and
  `damaged_cotton_boll`, so copy-paste may only affect `leaf_reddening` unless
  the dataset taxonomy is changed.

## 6. Best Selection

File:

- `extracted_modules/engine__solver__det_solver__DetSolver.py`

Check:

- `_is_better_v2_1` uses AP50 as the primary metric.
- If AP50 difference is within `hard_class_tie_threshold`, it compares
  `hard_class_mean_ap50`.
- Confirm per-class AP50 extraction is correct for the evaluator output.

## 7. Main Risk Summary

The most likely reasons modules may not improve enough are configuration and
task mismatch, not just missing code:

- 21-class taxonomy vs originally discussed hard classes.
- HardClassCopyPaste target names absent in current categories.
- Safe gates can suppress module contribution if set too low.
- Strong color augmentation can hurt color-defined disease classes.
- Rectangular copy-paste may introduce box-context artifacts.
- PF module is a detector feature prior, not a direct sub-pixel measurement
  output.

