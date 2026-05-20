#!/usr/bin/env bash
set -euo pipefail

# Canonical v2.2 launcher.
# Fine-tunes from the current improved v1 AP50-best checkpoint.

python -u train.py \
  -c configs/cotton/rtv4_hgnetv2_m_cotton_hp93_v22.yml \
  -t outputs/rtv4_hgnetv2_m_cotton_balanced_v6_768/best_ap50.pth \
  -d cuda \
  --seed 0 \
  --use-amp
