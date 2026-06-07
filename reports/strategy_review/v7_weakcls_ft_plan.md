# v7 Weak-Class Fine-Tune Plan

## Purpose

This route is intended to improve the trained detector itself on the full 21-class cotton split. It is not a test-time augmentation, WBF, NMS voting, or batch-size-only change.

The starting checkpoint is:

```text
outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth
```

The baseline metadata for that checkpoint is:

```text
AP50 = 0.930448602140458
mAP50:95 = 0.8101088526024318
AP75 = 0.8255006663957645
epoch = 44
```

## Success Criterion

Strict success requires model-weight improvement, not postprocess-only gain.

Primary pass line:

```text
AP50 >= 0.9405
```

This is at least +1.0 percentage point over the v6 AP50 baseline.

Secondary requirement:

```text
mAP50:95 should not drop below 0.805
AP75 should not drop below 0.815
```

If AP50 improves but mAP50:95 or AP75 collapses, the line should be treated as AP50 overfitting and not used as the main paper result.

## Rationale

Previous failed routes indicate that the main bottleneck is not pure box geometry:

- P2 + scale-balanced query did not improve AP50 and did not fix small-object APS.
- BoxFit/AP50-margin/MPDIoU/Alpha-IoU reduced the stable AP50 line.
- rankcal_dup did not open a meaningful AP50 gap, although it gave a slight mAP50:95 signal.

v7 therefore avoids another large structural jump. It fine-tunes from the strong v6 checkpoint using a narrower module set:

- Prewitt-Franklin edge enhancement, local refine enabled, edge-consistency loss disabled.
- Small-lesion enhancement.
- Output refine.
- CARAFE + BiFPN.
- Light CoordAttention + ECA.
- No P2, no LSK, no CBAM, no RepVGG, no FasterNet, no Gather-Distribute.

The main new supervision change is conservative positive-class reweighting for weak or low-stability classes:

```text
cotton_aphid, mealy_bug, red_cotton_bug, american_bollworm,
pink_bollworm, army_worm, thrips, whitefly, powdery_mildew, boll_rot
```

The objective is to improve AP50 ranking/recall on weak classes without disturbing already strong disease and healthy classes.

## Files

```text
configs/cotton/rtv4_hgnetv2_m_cotton_v7_weakcls_ft_768.yml
tools/cotton/train_hp93_v7_weakcls_ft_3090.ps1
tools/cotton/test_hp93_v7_weakcls_ft_3090.ps1
```

## Run Command

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\train_hp93_v7_weakcls_ft_3090.ps1
```

## Test Command

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\test_hp93_v7_weakcls_ft_3090.ps1
```

## Decision Rule

Keep as the main paper model only if:

```text
best AP50 >= 0.9405
and mAP50:95 >= 0.805
and AP75 >= 0.815
```

If not, record it as a failed model-training optimization and do not claim the detector itself improved. In that case, only the output-decision/TTA route remains a system-level improvement, not a model-weight improvement.
