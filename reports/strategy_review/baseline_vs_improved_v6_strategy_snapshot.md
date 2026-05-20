# Baseline vs Improved v6 Strategy Snapshot

Generated at: 2026-05-20 05:22:46

This file tracks the clean baseline catching up with the current improved v6 run. It is meant for Pro review before deciding the next optimization strategy.

## Current Status

| run | config | latest epoch | latest mAP | latest AP50 | best AP50 epoch | best AP50 |
|---|---|---:|---:|---:|---:|---:|
| baseline_clean | `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_clean_baseline_768.yml` | 29 | 0.768246 | 0.895745 | 29 | 0.895745 |
| improved_v6 | `configs/cotton/rtv4_hgnetv2_m_cotton_balanced_v6_768.yml` | 118 | 0.805934 | 0.923688 | 44 | 0.930449 |

## Matched Epoch Comparison

Latest matched epoch: 29

At epoch 29, baseline AP50 is 0.895745, improved v6 AP50 is 0.900802. The gap is only 0.005057 AP50.

| epoch | baseline mAP | baseline AP50 | improved v6 mAP | improved v6 AP50 | baseline AP50 - improved AP50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.371383 | 0.392187 | 0.429577 | 0.456843 | -0.064656 |
| 5 | 0.618730 | 0.709397 | 0.636980 | 0.759161 | -0.049764 |
| 10 | 0.673728 | 0.781635 | 0.689009 | 0.822957 | -0.041322 |
| 15 | 0.713176 | 0.825942 | 0.725261 | 0.859817 | -0.033875 |
| 20 | 0.743479 | 0.867527 | 0.745366 | 0.882253 | -0.014726 |
| 22 | 0.749853 | 0.872329 | 0.753908 | 0.892103 | -0.019774 |
| 25 | 0.758774 | 0.881376 | 0.759657 | 0.894248 | -0.012872 |
| 29 | 0.768246 | 0.895745 | 0.770541 | 0.900802 | -0.005057 |
| 30 | - | - | 0.768953 | 0.901358 | - |
| 35 | - | - | 0.780243 | 0.907864 | - |
| 40 | - | - | 0.798773 | 0.922705 | - |
| 44 | - | - | 0.810109 | 0.930449 | - |
| 50 | - | - | 0.796084 | 0.920700 | - |

Full table: `reports/strategy_review/baseline_vs_improved_v6_epoch_compare.csv`

## Interpretation

- Baseline is no longer clearly weak. By epoch 29, it is within roughly half an AP50 point of improved v6 at the same epoch.
- Improved v6 still has the best current checkpoint: AP50 0.930449 at epoch 44.
- Baseline needs another 0.034703 AP50 to match the improved v6 best. It has 15 epochs before the fair epoch-44 comparison point.
- If baseline reaches 0.92+ AP50 near epoch 44, the paper should avoid claiming a large overall AP50 gain from the current module stack.
- The next useful branch is `improved_v2_1`: narrower hard-class/edge-detail tuning, not more large-module stacking.

## Questions For Pro Review

1. Is the current improved v6 gain mostly caused by training recipe and data cleanup rather than modules?
2. Should v2.1 be trained from scratch for fairness, or initialized from improved v6 best for a final performance run?
3. Should the claim move from overall AP50 improvement to hard-class AP, APS/APM, or AP75 improvements?
4. Since the current dataset lacks `yellowish_leaf` and `damaged_cotton_boll`, should those classes be restored before v2.1?
