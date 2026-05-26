#!/usr/bin/env python3
"""Build a 13-class cotton dataset balanced by training image count.

The previous core300 route balanced training instances. This route is for a
cleaner paper table: retained classes must have at least 300 unique images in
the original train+val pool, then each retained class contributes about 300
training images. Validation images are added from the remaining train+val pool
according to ``val_ratio`` when enough images are available. Test images are
sampled from the original test split according to ``test_ratio`` when enough
images are available.

Original test images are never mixed into train/val.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KEPT_CLASSES = [
    "cotton_aphid",
    "leaf_hopper_jassids",
    "american_bollworm",
    "bacterial_leaf_blight",
    "alternaria_leaf_spot",
    "cotton_leaf_curl_virus",
    "fusarium_wilt",
    "verticillium_wilt",
    "leaf_variegation",
    "leaf_reddening",
    "herbicide_growth_damage",
    "open_cotton_boll",
    "healthy",
]

REMOVED_CLASSES = [
    "mealy_bug",
    "red_cotton_bug",
    "army_worm",
    "powdery_mildew",
    "boll_rot",
    "pink_bollworm",
    "thrips",
    "whitefly",
]


@dataclass
class ImageRecord:
    source_split: str
    source_image_id: int
    source_file_name: str
    image: dict[str, Any]
    annotations: list[dict[str, Any]]
    classes: set[str]
    key: str


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    name_to_id = {name: cid for cid, name in id_to_name.items()}
    missing = [name for name in KEPT_CLASSES + REMOVED_CLASSES if name not in name_to_id]
    if missing:
        raise ValueError(f"Missing expected categories: {missing}")
    return id_to_name, name_to_id


def build_records(coco: dict[str, Any], split: str, keep_ids: set[int], id_to_name: dict[int, str]) -> list[ImageRecord]:
    images = {int(img["id"]): img for img in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann["category_id"]) in keep_ids:
            anns_by_image[int(ann["image_id"])].append(ann)

    records: list[ImageRecord] = []
    for image_id, anns in anns_by_image.items():
        if image_id not in images or not anns:
            continue
        classes = {id_to_name[int(ann["category_id"])] for ann in anns}
        image = images[image_id]
        records.append(
            ImageRecord(
                source_split=split,
                source_image_id=image_id,
                source_file_name=str(image.get("file_name", "")),
                image=image,
                annotations=anns,
                classes=classes,
                key=f"{split}:{image_id}:{image.get('file_name', '')}",
            )
        )
    return records


def image_counts(records: list[ImageRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for name in record.classes:
            counts[name] += 1
    return counts


def annotation_counts(records: list[ImageRecord], id_to_name: dict[int, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for ann in record.annotations:
            counts[id_to_name[int(ann["category_id"])]] += 1
    return counts


def select_by_image_targets(
    candidates: list[ImageRecord],
    targets: dict[str, int],
    *,
    seed: int,
    forbidden_keys: set[str] | None = None,
) -> list[ImageRecord]:
    rng = random.Random(seed)
    selected: list[ImageRecord] = []
    selected_keys = set(forbidden_keys or set())
    counts: Counter[str] = Counter()
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    max_count = max(targets.values())
    soft_caps = {name: target + max(8, round(target * 0.05)) for name, target in targets.items()}

    def unfinished() -> list[str]:
        return [name for name in KEPT_CLASSES if counts[name] < targets[name]]

    while unfinished():
        need_order = sorted(unfinished(), key=lambda n: (counts[n] / max(targets[n], 1), counts[n]))
        best: ImageRecord | None = None
        best_score = float("-inf")

        for focus in need_order:
            for record in shuffled:
                if record.key in selected_keys or focus not in record.classes:
                    continue

                score = 0.0
                hard_overflow = False
                for name in record.classes:
                    if name not in targets:
                        continue
                    deficit = max(0, targets[name] - counts[name])
                    if deficit:
                        score += 3.0
                        score += deficit / max_count
                    if counts[name] >= soft_caps[name]:
                        hard_overflow = True
                    elif counts[name] >= targets[name]:
                        score -= 1.5

                score -= 0.01 * len(record.annotations)
                score += rng.random() * 0.001
                if hard_overflow:
                    score -= 6.0
                if score > best_score:
                    best_score = score
                    best = record
            if best is not None:
                break

        if best is None:
            break
        selected.append(best)
        selected_keys.add(best.key)
        for name in best.classes:
            counts[name] += 1

    return selected


def source_image_path(src_img_root: Path, record: ImageRecord) -> Path:
    name = record.source_file_name.replace("\\", "/")
    candidates = [
        src_img_root / record.source_split / name,
        src_img_root / name,
        src_img_root / record.source_split / Path(name).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def unique_dest_name(record: ImageRecord, used: set[str]) -> str:
    raw = Path(record.source_file_name.replace("\\", "/")).name
    candidate = raw
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem = Path(raw).stem
    suffix = Path(raw).suffix
    candidate = f"{record.source_split}_{record.source_image_id}_{stem}{suffix}"
    i = 1
    while candidate in used:
        candidate = f"{record.source_split}_{record.source_image_id}_{stem}_{i}{suffix}"
        i += 1
    used.add(candidate)
    return candidate


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_coco_split(
    records: list[ImageRecord],
    split: str,
    src_img_root: Path,
    dst_root: Path,
    old_to_new_cat: dict[int, int],
    categories: list[dict[str, Any]],
    link_mode: str,
) -> tuple[dict[str, Any], list[str]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    missing: list[str] = []
    used_names: set[str] = set()
    ann_id = 1

    for new_img_id, record in enumerate(records, start=1):
        dst_name = unique_dest_name(record, used_names)
        src_path = source_image_path(src_img_root, record)
        dst_path = dst_root / "images" / split / dst_name
        if src_path.exists():
            link_or_copy(src_path, dst_path, link_mode)
        else:
            missing.append(str(src_path))

        img = {k: v for k, v in record.image.items() if k not in {"id", "file_name"}}
        img["id"] = new_img_id
        img["file_name"] = dst_name
        images.append(img)

        for ann in record.annotations:
            old_cid = int(ann["category_id"])
            if old_cid not in old_to_new_cat:
                continue
            bbox = list(ann.get("bbox", []))
            if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                continue
            new_ann = {k: v for k, v in ann.items() if k not in {"id", "image_id", "category_id", "area"}}
            new_ann["id"] = ann_id
            new_ann["image_id"] = new_img_id
            new_ann["category_id"] = old_to_new_cat[old_cid]
            new_ann["bbox"] = bbox
            new_ann["area"] = float(bbox[2]) * float(bbox[3])
            new_ann.setdefault("iscrowd", 0)
            annotations.append(new_ann)
            ann_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}, missing


def count_coco(coco: dict[str, Any]) -> list[dict[str, Any]]:
    cat_map = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    ann_counts: Counter[int] = Counter()
    img_counts: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        ann_counts[cid] += 1
        img_counts[cid].add(int(ann["image_id"]))
    return [
        {
            "category_id": cid,
            "class_name": cat_map[cid],
            "instances": ann_counts[cid],
            "images_with_class": len(img_counts[cid]),
        }
        for cid in sorted(cat_map)
    ]


def validate_split(coco: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_ids = {int(img["id"]) for img in coco["images"]}
    cat_ids = {int(cat["id"]) for cat in coco["categories"]}
    ann_ids = set()
    duplicate_ann_ids = 0
    missing_image_ref = 0
    bad_category_ref = 0
    bad_bbox = 0
    missing_paths = sum(1 for img in coco["images"] if not (image_dir / str(img["file_name"])).exists())
    for ann in coco["annotations"]:
        ann_id = int(ann["id"])
        if ann_id in ann_ids:
            duplicate_ann_ids += 1
        ann_ids.add(ann_id)
        if int(ann["image_id"]) not in image_ids:
            missing_image_ref += 1
        if int(ann["category_id"]) not in cat_ids:
            bad_category_ref += 1
        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0 or float(ann.get("area", 0)) <= 0:
            bad_bbox += 1
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "missing_image_ref": missing_image_ref,
        "bad_category_ref": bad_category_ref,
        "duplicate_ann_ids": duplicate_ann_ids,
        "bad_bbox_or_area": bad_bbox,
        "missing_image_paths": missing_paths,
        "category_ids_continuous": sorted(cat_ids) == list(range(len(cat_ids))),
    }


def make_report(
    dst_root: Path,
    split_cocos: dict[str, dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    validation: dict[str, dict[str, Any]],
    missing_images: dict[str, list[str]],
) -> None:
    reports = dst_root / "reports"
    class_rows: list[dict[str, Any]] = []
    for split, coco in split_cocos.items():
        for row in count_coco(coco):
            class_rows.append({"split": split, **row})
    write_csv(reports / "class_distribution_core300img_13cls.csv", class_rows)
    write_csv(reports / "removed_classes.csv", removed_rows)
    write_csv(reports / "category_id_mapping.csv", mapping_rows)
    write_csv(reports / "source_image_availability.csv", source_rows)

    split_names = ["train", "val", "test"]
    file_sets = {s: {str(img["file_name"]) for img in split_cocos[s]["images"]} for s in split_names}
    leaks = []
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = file_sets[a] & file_sets[b]
            if overlap:
                leaks.append(f"{a}-{b}:{len(overlap)}")

    lines = [
        "# cotton_core300img_13cls split integrity report",
        "",
        "## Task definition",
        "",
        "- Simplified task: 13 retained classes.",
        "- Classes with fewer than 300 unique train+val images are removed.",
        "- Each retained class targets 300 training images.",
        "- Validation images are sampled from the remaining train+val pool by ratio when available.",
        "- Test images are sampled from the original test split by ratio when available.",
        "- Original test never enters train/val.",
        "",
        "## Split summary",
        "",
        "| split | images | annotations | missing image refs | bad category refs | bad bbox/area | missing image paths | category ids continuous |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split in split_names:
        v = validation[split]
        lines.append(
            f"| {split} | {v['images']} | {v['annotations']} | {v['missing_image_ref']} | "
            f"{v['bad_category_ref']} | {v['bad_bbox_or_area']} | {v['missing_image_paths']} | "
            f"{v['category_ids_continuous']} |"
        )
    lines += [
        "",
        "## Removed classes",
        "",
        "| class_name | trainval_images | trainval_instances | removed_reason |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in removed_rows:
        lines.append(
            f"| {row['class_name']} | {row['trainval_images']} | {row['trainval_instances']} | {row['removed_reason']} |"
        )
    lines += [
        "",
        "## Class distribution",
        "",
        "| split | class_name | instances | images_with_class |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in class_rows:
        lines.append(f"| {row['split']} | {row['class_name']} | {row['instances']} | {row['images_with_class']} |")
    lines += [
        "",
        "## Integrity checks",
        "",
        f"- Cross-split file_name leakage: {'none' if not leaks else ', '.join(leaks)}",
        f"- Missing source image paths: {sum(len(v) for v in missing_images.values())}",
        f"- Original data write policy: only writes under `{dst_root}`.",
    ]
    (reports / "split_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    src_ann_dir = Path(args.src_ann_dir)
    src_img_root = Path(args.src_img_root)
    dst_root = Path(args.dst_root)

    train_coco = load_coco(src_ann_dir / "instances_train.json")
    val_coco = load_coco(src_ann_dir / "instances_val.json")
    test_coco = load_coco(src_ann_dir / "instances_test.json")
    id_to_name, name_to_id = category_maps(train_coco)

    keep_old_ids = {name_to_id[name] for name in KEPT_CLASSES}
    all_expected = KEPT_CLASSES + REMOVED_CLASSES
    all_ids = {name_to_id[name] for name in all_expected}

    train_records_all = build_records(train_coco, "train", all_ids, id_to_name)
    val_records_all = build_records(val_coco, "val", all_ids, id_to_name)
    availability_records = train_records_all + val_records_all
    avail_images = image_counts(availability_records)
    avail_instances = annotation_counts(availability_records, id_to_name)

    source_rows = [
        {
            "class_name": name,
            "trainval_images": avail_images[name],
            "trainval_instances": avail_instances[name],
            "retained": name in KEPT_CLASSES,
        }
        for name in all_expected
    ]
    removed_rows = [
        {
            "class_name": name,
            "trainval_images": avail_images[name],
            "trainval_instances": avail_instances[name],
            "removed_reason": "train+val unique images < 300 or original low-sample class",
        }
        for name in REMOVED_CLASSES
    ]

    old_to_new_cat = {name_to_id[name]: i for i, name in enumerate(KEPT_CLASSES)}
    categories = [{"id": i, "name": name} for i, name in enumerate(KEPT_CLASSES)]
    mapping_rows = [
        {"old_category_id": name_to_id[name], "new_category_id": i, "class_name": name}
        for i, name in enumerate(KEPT_CLASSES)
    ]

    pool = build_records(train_coco, "train", keep_old_ids, id_to_name) + build_records(val_coco, "val", keep_old_ids, id_to_name)
    test_pool = build_records(test_coco, "test", keep_old_ids, id_to_name)

    train_target = args.target_train_images
    val_target = round(args.target_train_images * args.val_ratio)
    test_target = round(args.target_train_images * args.test_ratio)
    train_targets = {name: train_target for name in KEPT_CLASSES}
    val_targets = {name: val_target for name in KEPT_CLASSES}
    test_targets = {name: test_target for name in KEPT_CLASSES}

    train_records = select_by_image_targets(pool, train_targets, seed=args.seed)
    train_keys = {record.key for record in train_records}
    val_records = select_by_image_targets(pool, val_targets, seed=args.seed + 17, forbidden_keys=train_keys)
    test_records = select_by_image_targets(test_pool, test_targets, seed=args.seed + 31)

    split_cocos: dict[str, dict[str, Any]] = {}
    missing_images: dict[str, list[str]] = {}
    for split, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        coco, missing = build_coco_split(records, split, src_img_root, dst_root, old_to_new_cat, categories, args.link_mode)
        split_cocos[split] = coco
        missing_images[split] = missing
        write_json(dst_root / "annotations" / f"instances_{split}.json", coco)

    validation = {split: validate_split(coco, dst_root / "images" / split) for split, coco in split_cocos.items()}
    make_report(dst_root, split_cocos, removed_rows, mapping_rows, source_rows, validation, missing_images)

    print(f"Wrote dataset to {dst_root}")
    for split in ("train", "val", "test"):
        v = validation[split]
        print(f"{split}: images={v['images']} annotations={v['annotations']} missing_paths={v['missing_image_paths']}")
    print(f"Report: {dst_root / 'reports' / 'split_integrity_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-ann-dir", required=True)
    parser.add_argument("--src-img-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--target-train-images", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
