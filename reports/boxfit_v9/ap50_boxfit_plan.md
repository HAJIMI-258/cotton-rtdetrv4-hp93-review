# AP50 BoxFit plan

This plan changes the optimization target, not just the feature extractor.

## Why this should affect AP50 more directly

AP50 rewards detections whose predicted box overlaps the annotation by at least 0.5 IoU. The existing loss uses L1 + GIoU + NWD, but the model can still produce many boxes near the threshold. The patch adds:

1. `loss_mpdiou`: corner distance pressure. This directly pulls predicted top-left and bottom-right corners toward the annotation.
2. `loss_alpha_iou`: power IoU term. This gives stronger gradient to improving already-plausible matches.
3. `loss_ap50_margin`: a metric surrogate. Only matched boxes with IoU in a borderline band are pushed above `ap50_iou_target`.
4. optional inference NMS: removes high-score duplicate detections that hurt COCO precision.

## Patch order

1. Apply `boxfit_ap50_patch.diff`.
2. Optional but recommended for new from-scratch only: apply the previous `safe_scale_patch.diff`.
3. Copy `rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit.yml` into `configs/cotton/`.
4. Launch from scratch with no `-t` and no `-r`.

## Early stop checks

- epoch 10 should not be far below v6 rhythm.
- epoch 20 AP50 below 0.86 is a bad sign.
- epoch 29 AP50 below 0.90 is a bad sign.
