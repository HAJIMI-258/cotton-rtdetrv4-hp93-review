# v10 Progressive Refinement Plan

## Why v7/v8/v9 from-scratch failed

The previous from-scratch attempts changed the early optimization problem too much.

- v7 enabled multiple feature modules and weak-class weighting at the same time.
- v8 isolated weak-class weighting, but it does not improve the from-scratch upper bound by itself.
- v9 did not create a stable from-scratch AP50 gain.

These results indicate that the stable detector representation must be learned first; weak-class and lesion-aware refinement is useful after the detector already has good localization and category separation.

## New route

v10 is a progressive training route:

1. Stage 1 trains the stable v6 detector from scratch.
2. Stage 2 fine-tunes the stage-1 best checkpoint using weak-class/lesion-aware refinement.

This is still model-weight optimization. It is not TTA, WBF, NMS, or output-only postprocessing.

## Files

```text
configs/cotton/rtv4_hgnetv2_m_cotton_v10_progressive_refine_768.yml
tools/cotton/train_hp93_v10_progressive_3090.ps1
```

## Run

If the v6 stage-1 checkpoint already exists:

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\train_hp93_v10_progressive_3090.ps1 -SkipStage1
```

If not:

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\train_hp93_v10_progressive_3090.ps1
```

## Success criterion

Use the current-code re-evaluated v6 stage-1 checkpoint as the baseline. Keep v10 only if:

```text
v10 best AP50 - v6 re-evaluated AP50 >= 0.010
mAP50:95 does not drop by more than 0.005
AP75 does not drop by more than 0.010
```

If it passes, the paper claim should be:

```text
A progressive weak-class/lesion-aware refinement strategy improves the trained detector from the stable baseline.
```

Do not claim that every module combination improves from scratch. The failed v7/v8/v9 from-scratch experiments should be retained as negative evidence for why the final method uses progressive refinement.
