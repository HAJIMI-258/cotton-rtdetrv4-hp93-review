# cotton_core300_17cls split integrity report

## Task definition

- Simplified task: 17 retained classes.
- Removed classes are original train_instances < 300.
- Train/val are sampled only from original train+val.
- Test is only filtered from original test; no test image enters train/val.

## Split summary

| split | images | annotations | missing image refs | bad category refs | bad bbox/area | missing image paths | category ids continuous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 3302 | 5387 | 0 | 0 | 0 | 0 | True |
| val | 383 | 917 | 0 | 0 | 0 | 0 | True |
| test | 2813 | 4986 | 0 | 0 | 0 | 0 | True |

## Removed classes

| class_name | original_train_instances | removed_reason |
| --- | ---: | --- |
| red_cotton_bug | 154 | original train_instances < 300 |
| army_worm | 218 | original train_instances < 300 |
| powdery_mildew | 143 | original train_instances < 300 |
| boll_rot | 213 | original train_instances < 300 |

## Category remapping

| old_category_id | new_category_id | class_name |
| ---: | ---: | --- |
| 0 | 0 | cotton_aphid |
| 1 | 1 | mealy_bug |
| 3 | 2 | leaf_hopper_jassids |
| 4 | 3 | american_bollworm |
| 5 | 4 | pink_bollworm |
| 7 | 5 | thrips |
| 8 | 6 | whitefly |
| 9 | 7 | bacterial_leaf_blight |
| 10 | 8 | alternaria_leaf_spot |
| 11 | 9 | cotton_leaf_curl_virus |
| 12 | 10 | fusarium_wilt |
| 13 | 11 | verticillium_wilt |
| 15 | 12 | leaf_variegation |
| 16 | 13 | leaf_reddening |
| 17 | 14 | herbicide_growth_damage |
| 19 | 15 | open_cotton_boll |
| 20 | 16 | healthy |

## Class distribution

| split | class_name | instances | images_with_class |
| --- | --- | ---: | ---: |
| train | cotton_aphid | 300 | 70 |
| train | mealy_bug | 300 | 115 |
| train | leaf_hopper_jassids | 300 | 58 |
| train | american_bollworm | 300 | 211 |
| train | pink_bollworm | 300 | 87 |
| train | thrips | 300 | 140 |
| train | whitefly | 301 | 56 |
| train | bacterial_leaf_blight | 300 | 300 |
| train | alternaria_leaf_spot | 300 | 300 |
| train | cotton_leaf_curl_virus | 300 | 150 |
| train | fusarium_wilt | 300 | 300 |
| train | verticillium_wilt | 300 | 300 |
| train | leaf_variegation | 300 | 300 |
| train | leaf_reddening | 300 | 300 |
| train | herbicide_growth_damage | 300 | 300 |
| train | open_cotton_boll | 586 | 25 |
| train | healthy | 300 | 300 |
| val | cotton_aphid | 57 | 12 |
| val | mealy_bug | 54 | 12 |
| val | leaf_hopper_jassids | 54 | 12 |
| val | american_bollworm | 45 | 13 |
| val | pink_bollworm | 55 | 12 |
| val | thrips | 53 | 12 |
| val | whitefly | 56 | 12 |
| val | bacterial_leaf_blight | 45 | 36 |
| val | alternaria_leaf_spot | 45 | 45 |
| val | cotton_leaf_curl_virus | 45 | 15 |
| val | fusarium_wilt | 45 | 43 |
| val | verticillium_wilt | 45 | 45 |
| val | leaf_variegation | 45 | 45 |
| val | leaf_reddening | 45 | 36 |
| val | herbicide_growth_damage | 45 | 45 |
| val | open_cotton_boll | 138 | 8 |
| val | healthy | 45 | 25 |
| test | cotton_aphid | 146 | 86 |
| test | mealy_bug | 75 | 35 |
| test | leaf_hopper_jassids | 376 | 263 |
| test | american_bollworm | 75 | 52 |
| test | pink_bollworm | 51 | 32 |
| test | thrips | 50 | 28 |
| test | whitefly | 53 | 12 |
| test | bacterial_leaf_blight | 591 | 590 |
| test | alternaria_leaf_spot | 116 | 116 |
| test | cotton_leaf_curl_virus | 362 | 333 |
| test | fusarium_wilt | 224 | 223 |
| test | verticillium_wilt | 130 | 130 |
| test | leaf_variegation | 89 | 89 |
| test | leaf_reddening | 174 | 174 |
| test | herbicide_growth_damage | 118 | 118 |
| test | open_cotton_boll | 1885 | 66 |
| test | healthy | 471 | 471 |

## Integrity checks

- Cross-split file_name leakage: none
- Source annotations unchanged: True
- Missing source images copied/linked: 0
- Original data write policy: only writes under `data\cotton_core300_17cls`.

## Sampling notes

- open_cotton_boll: train_instances=586, train_images=25, outside preferred open-boll range.
