# Previous Improved Run Module Map

This file maps the code in this audit package to the improved run that was
actually completed before the v2.2 fine-tune discussion.

## Completed Improved Run

Main config:

```text
configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml
```

Snapshot in this audit package:

```text
reports/module_audit/config_snapshots/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml
```

Output/checkpoint path used on the 3090 machine:

```text
outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth
```

Known best metadata:

```text
epoch 44
mAP50:95 0.8101088526024318
AP50 0.930448602140458
AP75 0.8255006663957645
```

## Enabled In The Completed Improved Run

These switches are enabled in `balanced_v6_768`:

- `prewitt_franklin_enhance: True`
- `small_lesion_enhance: True`
- `output_refine: True`
- `carafe_upsample: True`
- `bifpn_fusion: True`
- `coord_attention_exact: True`
- `eca_attention: True`
- `safe_module_trainable_scale: True`
- `loss_nwd` in `RTv4Criterion.weight_dict`
- `loss_distill` in `RTv4Criterion.weight_dict`
- `distill_adaptive_params.foreground_focus: True`

These switches are disabled in `balanced_v6_768`:

- `edge_enhance`
- `cbam_attention`
- `lsk_attention`
- `fasternet_pconv`
- `repvgg_enhance`
- `gather_distribute`

## Relevant Module Code

- PF/edge and feature modules:
  - `extracted_modules/engine__rtv4__hybrid_encoder__PrewittFranklinEdgeGuidedEnhance.py`
  - `extracted_modules/engine__rtv4__hybrid_encoder__SmallLesionCrossScaleEnhance.py`
  - `extracted_modules/engine__rtv4__hybrid_encoder__CARAFEUpsample.py`
  - `extracted_modules/engine__rtv4__hybrid_encoder__BiFPNFusionBlock.py`
  - `extracted_modules/engine__rtv4__hybrid_encoder__CoordinateAttentionExact.py`
  - `extracted_modules/engine__rtv4__hybrid_encoder__ECALayer.py`
- Loss logic:
  - `extracted_modules/engine__rtv4__rtv4_criterion__RTv4Criterion.py`
- Best checkpoint logic:
  - `extracted_modules/engine__solver__det_solver__DetSolver.py`

## v2.1 Context

The requested clean v2.1-style from-scratch config is present as:

```text
configs/cotton/rtv4_hgnetv2_m_cotton_improved_v2_1.yml
```

Snapshot:

```text
reports/module_audit/config_snapshots/rtv4_hgnetv2_m_cotton_improved_v2_1.yml
```

It adds or makes explicit:

- PF local 7x7 refinement
- PF edge consistency loss
- HardClassCopyPaste
- AP50 plus hard-class AP50 tie-break best selection
- no mosaic/mixup in the v2.1 training recipe

## v2.2 Context

`improved_v2_2_finetune` is included only because it was created later. It should
not be treated as the clean from-scratch improved run:

```text
reports/module_audit/config_snapshots/rtv4_hgnetv2_m_cotton_improved_v2_2_finetune.yml
```

It inherits `balanced_v6_768` and was intended to fine-tune from:

```text
outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth
```

## Dataset/Class Caveat

The active configs in this repo use `num_classes: 21`. The hard-class copy-paste
request names:

```text
leaf_reddening
yellowish_leaf
damaged_cotton_boll
```

In the current 21-class dataset context documented in the v2.2 config,
`leaf_reddening` exists, while `yellowish_leaf` and `damaged_cotton_boll` are
absent. Reviewers should check whether the hard-class design matches the final
paper dataset taxonomy before treating it as an effective augmentation.

