# BoxFit v9 失败排查与逐类指标报告

生成时间：2026-05-23 15:07:14

## 结论

BoxFit v9 当前不适合作为最终改进线。它的最佳 AP50 是 `0.9135`（epoch 34），低于已固化的 93 改进线 `0.9304`（epoch 44），AP50 低 `-0.0169`，mAP50:95 低 `-0.0255`，AP75 低 `-0.0331`。

当前下载日志中的最新完整验证为 epoch 63：AP50 `0.8977`，mAP50:95 `0.7845`，说明 epoch 34 后没有继续提升，后段训练在 AP50 上回落。

## 总体指标

| run | epoch | mAP50_95 | AP50 | AP75 | note |
| --- | --- | --- | --- | --- | --- |
| v6_balanced_768_best | 44 | 0.8101 | 0.9304 | 0.8255 | previous strongest 93-line aggregate best |
| v2_2_finetune_best | 9 | 0.8033 | 0.9269 | 0.8188 | finetune best with per-class AP50 json |
| v9_boxfit_best_ap50 | 34 | 0.7846 | 0.9135 | 0.7924 | BoxFit v9 best AP50 checkpoint |
| v9_boxfit_latest_logged | 63 | 0.7845 | 0.8977 | 0.8040 | latest complete eval in downloaded log |

## v9 最弱类别（按 best AP50 排序）

| class_name | v9_best_AP50_epoch34 | train_instances | val_instances | val_images | test_instances | test_images |
| --- | --- | --- | --- | --- | --- | --- |
| whitefly | 0.6285 | 499 | 72 | 13 | 53 | 12 |
| cotton_aphid | 0.7009 | 1123 | 133 | 88 | 146 | 86 |
| mealy_bug | 0.7567 | 512 | 67 | 35 | 75 | 35 |
| pink_bollworm | 0.8432 | 518 | 56 | 33 | 51 | 32 |
| red_cotton_bug | 0.8657 | 154 | 19 | 19 | 20 | 19 |
| powdery_mildew | 0.8697 | 143 | 19 | 18 | 19 | 18 |
| american_bollworm | 0.9001 | 523 | 73 | 52 | 75 | 52 |
| leaf_hopper_jassids | 0.9043 | 3370 | 424 | 263 | 376 | 263 |

## v9 相比 v2.2 fine-tune 下降最多的类别

| class_name | v9_best_AP50_epoch34 | v22_finetune_best_AP50_epoch9 | v9_minus_v22_AP50 | train_instances | val_instances | val_images |
| --- | --- | --- | --- | --- | --- | --- |
| cotton_aphid | 0.7009 | 0.7480 | -0.0470 | 1123 | 133 | 88 |
| red_cotton_bug | 0.8657 | 0.9118 | -0.0461 | 154 | 19 | 19 |
| whitefly | 0.6285 | 0.6743 | -0.0458 | 499 | 72 | 13 |
| mealy_bug | 0.7567 | 0.7903 | -0.0336 | 512 | 67 | 35 |
| american_bollworm | 0.9001 | 0.9322 | -0.0321 | 523 | 73 | 52 |
| powdery_mildew | 0.8697 | 0.9017 | -0.0319 | 143 | 19 | 18 |
| thrips | 0.9571 | 0.9738 | -0.0166 | 423 | 38 | 29 |
| cotton_leaf_curl_virus | 0.9668 | 0.9833 | -0.0166 | 2877 | 362 | 333 |

## 逐类指标文件

完整逐类表见 `per_class_metrics_boxfit_v9.csv`，包括：

- 训练、验证、测试实例数与包含该类别的图片数；
- BoxFit v9 best AP50；
- BoxFit v9 最新 checkpoint 的 per-class mAP50:95、AP50、AP75、AR；
- v2.2 fine-tune best AP50；
- v9 与 v2.2 的 AP50 差值。

## 可能原因

1. **BoxFit 目标没有命中主要瓶颈。** v9 增加 MPDIoU、Alpha-IoU、AP50 margin，主要解决 matched box 的几何贴合；但弱项类别集中在 `whitefly`、`cotton_aphid`、`mealy_bug`、`pink_bollworm` 等小虫/纹理相近类，这更像分类置信度排序、类别混淆和小目标可见性问题，不是单纯 IoU 临界框问题。

2. **AP50 margin 后期几乎不再提供有效梯度。** 日志尾部 `loss_ap50_margin` 多数为 `0.0000`，均值约 `0.0004`，说明临界 IoU hinge 在后期参与度很低；它没有持续把 AP50 从 0.91 推到 0.93+。

3. **v9 同时改了太多变量，破坏了 v6 稳定配置。** v9 不只是加 box loss，还改了 896 输入、NMS、matcher 权重、safe scale、loss 权重和真实 batch。和 v6 相比，`safe_module_init_scale/max_scale` 从 `0.015/0.10` 增到 `0.03/0.15`，可能放大特征模块扰动。

4. **896 分辨率提高了定位细节，但也降低了每步稳定性。** v9 使用 `total_batch_size=4, accumulation_steps=4`，有效 batch 仍是 16，但真实单步 batch 比 v6 的 8 更小，训练噪声和 EMA 更新节奏不同；这可能解释 AP50 在 epoch 34 达峰后回落。

5. **NMS 可能压掉密集或重叠目标。** v9 的 `PostProcessor.nms_iou_threshold=0.75` 会减少重复框，但对 `open_cotton_boll` 这类一图多实例或同类贴近目标，可能误杀真阳性。必须用同一 checkpoint 对比 `nms_iou_threshold: null` 和 `0.75`。

6. **数据分布本身对若干细类不友好。** `whitefly` 验证集只有 13 张图、72 个实例；`mealy_bug` 35 张图、67 个实例；`pink_bollworm` 33 张图、56 个实例。小样本验证 AP 对少量误检/漏检非常敏感。

7. **类别相似性导致 AP50 不是纯定位问题。** 粉虱、蚜虫、粉蚧、蓟马、叶蝉都属于小尺度刺吸式害虫，外观和背景叶片纹理接近。BoxFit 强化框回归不能直接解决类别判别。

8. **best 选择口径只优化 AP50，牺牲了整体排序质量。** v9 best AP50 为 0.9135，但 mAP50:95 只有 0.7846；已固化 v6 为 AP50 0.9304、mAP50:95 0.8101。说明 v9 的排序/定位整体质量没有提升。

9. **从零训练成本高且不如从稳定强权重微调。** v9 从零跑，前 34 epoch 达峰后回落。对于已经有 0.9304 的 v6，直接在 v6 best 上做小范围 test-time/NMS/阈值/困难类微调，更可能保住上限。

## 建议下一步

- 不把 BoxFit v9 作为最终线；保留为失败实验和反例分析。
- 用 v6 93 线作为当前展示/比赛最终结果。
- 对 v9 的同一 checkpoint 做一次 NMS on/off test-only，确认 NMS 是否伤害 AP50。
- 若继续优化，只做单变量：先从 v6 best 出发，只加 NMS test-only 或只调 matcher/loss 中一个变量，不再从零大改。
