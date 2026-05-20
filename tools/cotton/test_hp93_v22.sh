#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-outputs/improved_v2_2_finetune/best_ap50_v2_2.pth}"

python -u train.py \
  -c configs/cotton/rtv4_hgnetv2_m_cotton_hp93_v22.yml \
  -r "$CKPT" \
  -d cuda \
  --test-only \
  --use-amp
