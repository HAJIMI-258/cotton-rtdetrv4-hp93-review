#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u train.py \
  -c configs/cotton/rtv4_hgnetv2_m_cotton_fromscratch_v9_896_boxfit.yml \
  -d cuda \
  --seed 0 \
  --use-amp

