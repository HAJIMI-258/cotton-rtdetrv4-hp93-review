# Cotton RT-DETRv4 HP93 Review Notes

This repository is a cleaned code-only snapshot for reviewing the cotton disease and pest detection experiments.
Datasets, checkpoints, pretrained weights, and training outputs are intentionally excluded.

## Main Configs

- Improved run: `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml`
- Matched clean baseline: `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_clean_baseline_768.yml`
- Earlier high-precision candidate: `configs/cotton/rtv4_hgnetv2_m_cotton_hp93_3090.yml`

## Core Code To Review

- Encoder modules: `engine/rtv4/hybrid_encoder.py`
- Decoder changes: `engine/rtv4/dfine_decoder.py`
- Criterion and losses: `engine/rtv4/rtv4_criterion.py`
- Training loop: `engine/solver/det_engine.py`
- Solver/checkpoint behavior: `engine/solver/det_solver.py`
- YAML/config plumbing: `engine/core/yaml_config.py`, `engine/core/_config.py`
- DINOv3 teacher: `engine/rtv4/dinov3_teacher.py`

## Implemented Improvement Ideas

- Prewitt-Franklin edge-guided enhancement.
- Small lesion cross-scale enhancement.
- Output refinement.
- CARAFE upsampling.
- BiFPN-style weighted fusion.
- Coordinate Attention and ECA attention gates.
- Foreground-aware distillation option.
- NWD loss option for small-object localization tolerance.
- AP50 best-checkpoint watcher: `tools/watch_best_ap50.py`
- Read-only training monitor for the baseline cloud run: `tools/cotton/baseline_web_monitor.py`

## Current Experiment Snapshot

- Improved best checkpoint observed on the 3090 run:
  - epoch 44
  - mAP 0.8101
  - AP50 0.93045
  - AP75 0.8255
- Matched clean baseline early curve on the cloud 3090:
  - epoch 8 AP50 0.7522
  - epoch 11 AP50 0.7937

## Review Questions

- Are the added modules correctly wired and disabled/enabled by config without hidden side effects?
- Is the foreground-aware distillation formulation appropriate for box-level cotton disease/pest data?
- Is NWD integrated with the matching/loss scale in a way that is stable for AP50/AP75?
- Are CARAFE/BiFPN/attention gates likely to help small lesions, or are they amplifying leaf texture noise?
- Should the final paper claim emphasize AP50, AP75, or mAP depending on the final baseline plateau?
