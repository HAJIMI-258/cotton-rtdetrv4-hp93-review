# Improved Module Audit Package

This directory is a code-only audit package for the cotton RT-DETRv4 improved runs.
It is intended for an external reviewer to check whether the added modules were
implemented correctly and whether any implementation/configuration issue could
explain weak or unstable gains.

No training code was changed while creating this package. Files here are copied
snapshots or exact source excerpts from the current repository.

## Contents

- `source_snapshots/`
  - Full copied source files that contain the improved modules and training logic.
- `extracted_modules/`
  - Exact AST-extracted class/function excerpts for each improved module.
- `config_snapshots/`
  - Copied config files for baseline, previous improved run, hp93, v2.1, v2.2,
    and the earlier all-module experiment.
- `module_code_inventory.csv`
  - Module name, source path, source line span, and excerpt path.
- `all_improved_module_code_excerpts.py`
  - One combined file containing all extracted module excerpts.
- `config_switch_matrix.csv`
  - Human-readable switch matrix for the main compared configs.
- `01_PREVIOUS_IMPROVED_RUN_MODULE_MAP.md`
  - Which config corresponds to the previously completed improved run and which
    modules were actually enabled there.
- `02_REVIEW_POINTS_FOR_PRO.md`
  - Specific issues worth checking before trusting or rewriting the modules.

## Main Source Snapshots

- `source_snapshots/engine_rtv4_hybrid_encoder.py`
  - Feature modules: PF edge, small lesion, CARAFE, BiFPN, attention blocks,
    FasterNet/PConv, RepVGG, gather-distribute, output refinement.
- `source_snapshots/engine_rtv4_rtv4_criterion.py`
  - NWD, foreground-aware distillation, edge consistency loss, class weights.
- `source_snapshots/engine_data_transforms_copy_paste.py`
  - Hard-class copy-paste augmentation.
- `source_snapshots/engine_solver_det_solver.py`
  - AP50 plus hard-class tie-break best-checkpoint logic.
- `source_snapshots/engine_rtv4_rtv4.py`
  - Model forward path and auxiliary loss propagation.

## Important Context

The previously completed improved run that reached the known AP50-best checkpoint
is mapped to:

```text
configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml
outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth
```

Known result metadata from the remote run:

```text
epoch: 44
mAP50:95: 0.8101088526024318
AP50: 0.930448602140458
AP75: 0.8255006663957645
```

The `improved_v2_2_finetune` config is included only for review context. It is
not a clean from-scratch final run; it was designed as a fine-tune from the v6
best checkpoint.

