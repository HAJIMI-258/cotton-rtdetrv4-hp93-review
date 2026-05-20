# BoxFit v9 AP50 Route

This is the AP50-oriented route supplied as local files and then applied to the
repository manually because `boxfit_ap50_patch.diff` did not contain valid git
hunk line ranges.

## Added Code

- `engine/rtv4/rtv4_criterion.py`
  - `loss_mpdiou`
  - `loss_alpha_iou`
  - `loss_ap50_margin`
  - config fields:
    - `box_fit_alpha`
    - `ap50_iou_target`
    - `ap50_hinge_low`
    - `mpdiou_eps`
- `engine/rtv4/postprocessor.py`
  - optional `score_threshold`
  - optional class-aware `torchvision.ops.batched_nms`
- `engine/solver/det_solver.py`
  - AP50 custom mode refreshes EMA from the custom AP50 checkpoint at
    `stop_epoch` when available.
  - AP50 custom mode disables the later mAP-based rollback to `best_stg1.pth`.

## Added Config

```text
configs/cotton/rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit.yml
```

This config is a from-scratch recipe. It must be launched without `-t` and
without `-r`.

## Launch

Windows 3090:

```powershell
powershell -ExecutionPolicy Bypass -File tools\cotton\train_boxfit_v9_896_3090.ps1
```

Bash:

```bash
bash tools/cotton/train_boxfit_v9_896.sh
```

## Scope

This route does not add more feature modules. It targets box fitting directly
through matcher/loss/post-processing changes.

## Patch Files Kept For Review

- `boxfit_ap50_patch.diff`
- `ap50_solver_refresh_fix.diff`

Both downloaded diff files lacked valid git hunk line ranges, so their intended
changes were applied manually and then checked in.
