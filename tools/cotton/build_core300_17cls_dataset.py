#!/usr/bin/env python3
"""Build the cotton_core300_17cls simplified COCO dataset.

This script creates a new 17-class task from the balanced 21-class cotton
dataset. It never writes to the source annotation or image directories.
Training/validation are sampled only from the original train+val pool; the
original test split is only filtered.
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


REMOVED_CLASSES = [
    "red_cotton_bug",
    "army_worm",
    "powdery_mildew",
    "boll_rot",
]

KEPT_CLASSES = [
    "cotton_aphid",
    "mealy_bug",
    "leaf_hopper_jassids",
    "american_bollworm",
    "pink_bollworm",
    "thrips",
    "whitefly",
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

OPEN_BOLL = "open_cotton_boll"


@dataclass
class ImageRecord:
    source_split: str
    source_image_id: int
    source_file_name: str
    image: dict[str, Any]
    annotations: list[dict[str, Any]]
    counts: Counter[str]
    key: str


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def source_snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    snap = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            snap[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            snap[str(path)] = {"missing": True}
    return snap


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, str], dict[str, int]]:
    id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    name_to_id = {name: cid for cid, name in id_to_name.items()}
    missing = [name for name in KEPT_CLASSES + REMOVED_CLASSES if name not in name_to_id]
    if missing:
        raise ValueError(f"Missing expected categories in source annotations: {missing}")
    return id_to_name, name_to_id


def build_records(coco: dict[str, Any], split: str, keep_ids: set[int], id_to_name: dict[int, str]) -> list[ImageRecord]:
    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        cid = int(ann["category_id"])
        if cid in keep_ids:
            anns_by_image[int(ann["image_id"])].append(ann)

    records: list[ImageRecord] = []
    for image_id, anns in anns_by_image.items():
        if image_id not in images or not anns:
            continue
        counts = Counter(id_to_name[int(ann["category_id"])] for ann in anns)
        image = images[image_id]
        key = f"{split}:{image_id}:{image.get('file_name', '')}"
        records.append(
            ImageRecord(
                source_split=split,
                source_image_id=image_id,
                source_file_name=str(image.get("file_name", "")),
                image=image,
                annotations=anns,
                counts=counts,
                key=key,
            )
        )
    return records


def split_counts(records: list[ImageRecord]) -> tuple[Counter[str], Counter[str]]:
    instances: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    for record in records:
        instances.update(record.counts)
        for name in record.counts:
            image_counts[name] += 1
    return instances, image_counts


def make_targets(target_train_instances: int, val_ratio: float) -> tuple[dict[str, int], dict[str, int]]:
    train_targets = {name: int(target_train_instances) for name in KEPT_CLASSES}
    train_targets[OPEN_BOLL] = max(360, int(target_train_instances))
    val_target = max(30, round(target_train_instances * val_ratio))
    val_targets = {name: int(val_target) for name in KEPT_CLASSES}
    return train_targets, val_targets


def pick_best_candidate(
    candidates: list[ImageRecord],
    selected_keys: set[str],
    counts: Counter[str],
    image_counts: Counter[str],
    targets: dict[str, int],
    lower: dict[str, int],
    caps: dict[str, int],
    min_images: dict[str, int],
    rng: random.Random,
    allow_over_cap: bool,
) -> ImageRecord | None:
    best: ImageRecord | None = None
    best_score = float("-inf")

    for record in candidates:
        if record.key in selected_keys:
            continue

        useful = False
        score = 0.0
        penalty = 0.0
        for name, n in record.counts.items():
            target = max(targets[name], 1)
            need = max(0, targets[name] - counts[name])
            low_need = max(0, lower[name] - counts[name])
            img_need = max(0, min_images[name] - image_counts[name])

            if need > 0 or low_need > 0 or img_need > 0:
                useful = True

            score += min(n, need) / target * 4.0
            score += min(n, low_need) / max(lower[name], 1) * 3.0
            if img_need > 0:
                score += 0.45

            overflow = counts[name] + n - caps[name]
            if overflow > 0:
                penalty += (overflow / max(caps[name], 1)) * 7.0

        if not useful:
            continue
        if penalty > 0 and not allow_over_cap:
            continue

        score -= penalty
        score -= 0.015 * sum(record.counts.values())
        score += rng.random() * 0.001
        if score > best_score:
            best_score = score
            best = record

    return best


def needs_more(
    counts: Counter[str],
    image_counts: Counter[str],
    lower: dict[str, int],
    min_images: dict[str, int],
) -> bool:
    for name in KEPT_CLASSES:
        if counts[name] < lower[name]:
            return True
        if image_counts[name] < min_images[name]:
            return True
    return False


def select_records(
    candidates: list[ImageRecord],
    targets: dict[str, int],
    *,
    seed: int,
    mode: str,
    forbidden_keys: set[str] | None = None,
) -> list[ImageRecord]:
    rng = random.Random(seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    if mode == "train":
        lower = {name: 260 for name in KEPT_CLASSES}
        caps = {name: 340 for name in KEPT_CLASSES}
        min_images = {name: 30 for name in KEPT_CLASSES}
        caps[OPEN_BOLL] = 450
        lower[OPEN_BOLL] = 300
        min_images[OPEN_BOLL] = 25
    else:
        lower = {name: min(targets[name], 30) for name in KEPT_CLASSES}
        caps = {name: max(targets[name] + 25, 55) for name in KEPT_CLASSES}
        min_images = {name: min(12, targets[name]) for name in KEPT_CLASSES}
        caps[OPEN_BOLL] = max(90, targets[OPEN_BOLL] + 45)
        min_images[OPEN_BOLL] = 8

    forbidden_keys = forbidden_keys or set()
    selected: list[ImageRecord] = []
    selected_keys: set[str] = set(forbidden_keys)
    counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()

    for allow_over_cap in (False, True):
        safety = 0
        while needs_more(counts, image_counts, lower, min_images):
            safety += 1
            if safety > len(shuffled) + 100:
                break
            record = pick_best_candidate(
                shuffled,
                selected_keys,
                counts,
                image_counts,
                targets,
                lower,
                caps,
                min_images,
                rng,
                allow_over_cap,
            )
            if record is None:
                break
            selected.append(record)
            selected_keys.add(record.key)
            counts.update(record.counts)
            for name in record.counts:
                image_counts[name] += 1

    return selected


def source_image_path(src_img_root: Path, record: ImageRecord) -> Path:
    file_name = record.source_file_name.replace("\\", "/")
    path = src_img_root / record.source_split / file_name
    if path.exists():
        return path
    return src_img_root / record.source_split / Path(file_name).name


def unique_dest_name(record: ImageRecord, used: set[str]) -> str:
    base = Path(record.source_file_name.replace("\\", "/")).name
    if not base:
        base = f"{record.source_image_id}.jpg"
    name = f"{record.source_split}_{record.source_image_id}_{base}"
    stem = Path(name).stem
    suffix = Path(name).suffix or ".jpg"
    i = 1
    while name in used:
        name = f"{stem}_{i}{suffix}"
        i += 1
    used.add(name)
    return name


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
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
    missing_images: list[str] = []
    used_names: set[str] = set()
    ann_id = 1

    for new_img_id, record in enumerate(records, start=1):
        dst_name = unique_dest_name(record, used_names)
        src_path = source_image_path(src_img_root, record)
        dst_path = dst_root / "images" / split / dst_name
        if not src_path.exists():
            missing_images.append(str(src_path))
        else:
            link_or_copy(src_path, dst_path, link_mode)

        new_img = {k: v for k, v in record.image.items() if k not in {"id", "file_name"}}
        new_img["id"] = new_img_id
        new_img["file_name"] = dst_name
        images.append(new_img)

        for ann in record.annotations:
            bbox = list(ann.get("bbox", []))
            if len(bbox) != 4:
                continue
            w = float(bbox[2])
            h = float(bbox[3])
            if w <= 0 or h <= 0:
                continue
            old_cid = int(ann["category_id"])
            new_ann = {k: v for k, v in ann.items() if k not in {"id", "image_id", "category_id", "area"}}
            new_ann["id"] = ann_id
            new_ann["image_id"] = new_img_id
            new_ann["category_id"] = old_to_new_cat[old_cid]
            new_ann["bbox"] = bbox
            new_ann["area"] = w * h
            new_ann.setdefault("iscrowd", 0)
            annotations.append(new_ann)
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }, missing_images


def count_coco(coco: dict[str, Any]) -> list[dict[str, Any]]:
    cat_map = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    ann_counts: Counter[int] = Counter()
    img_counts: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        ann_counts[cid] += 1
        img_counts[cid].add(int(ann["image_id"]))
    rows = []
    for cid in sorted(cat_map):
        rows.append({
            "category_id": cid,
            "class_name": cat_map[cid],
            "instances": ann_counts[cid],
            "images_with_class": len(img_counts[cid]),
        })
    return rows


def validate_split(coco: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    image_ids = {int(img["id"]) for img in coco["images"]}
    cat_ids = {int(cat["id"]) for cat in coco["categories"]}
    ann_ids = set()
    bad_bbox = 0
    missing_image_ref = 0
    bad_category_ref = 0
    duplicate_ann_ids = 0
    missing_paths = 0
    for img in coco["images"]:
        if not (image_dir / str(img["file_name"])).exists():
            missing_paths += 1
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
        area = float(ann.get("area", 0))
        if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0 or area <= 0:
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_report(
    dst_root: Path,
    split_cocos: dict[str, dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    validation: dict[str, dict[str, Any]],
    source_before: dict[str, dict[str, Any]],
    source_after: dict[str, dict[str, Any]],
    missing_images: dict[str, list[str]],
    selection_notes: list[str],
) -> None:
    reports = dst_root / "reports"
    class_rows = []
    for split, coco in split_cocos.items():
        for row in count_coco(coco):
            row = {"split": split, **row}
            class_rows.append(row)
    write_csv(reports / "class_distribution_core300_17cls.csv", class_rows)
    write_csv(reports / "removed_classes.csv", removed_rows)
    write_csv(reports / "category_id_mapping.csv", mapping_rows)

    split_names = ["train", "val", "test"]
    file_name_sets = {
        split: {str(img["file_name"]) for img in split_cocos[split]["images"]}
        for split in split_names
    }
    leak_pairs = []
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlap = sorted(file_name_sets[a] & file_name_sets[b])
            if overlap:
                leak_pairs.append((a, b, overlap[:20], len(overlap)))

    source_unchanged = source_before == source_after
    lines = [
        "# cotton_core300_17cls split integrity report",
        "",
        "## Task definition",
        "",
        "- Simplified task: 17 retained classes.",
        "- Removed classes are original train_instances < 300.",
        "- Train/val are sampled only from original train+val.",
        "- Test is only filtered from original test; no test image enters train/val.",
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
        "| class_name | original_train_instances | removed_reason |",
        "| --- | ---: | --- |",
    ]
    for row in removed_rows:
        lines.append(f"| {row['class_name']} | {row['original_train_instances']} | {row['removed_reason']} |")
    lines += [
        "",
        "## Category remapping",
        "",
        "| old_category_id | new_category_id | class_name |",
        "| ---: | ---: | --- |",
    ]
    for row in mapping_rows:
        lines.append(f"| {row['old_category_id']} | {row['new_category_id']} | {row['class_name']} |")
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
        f"- Cross-split file_name leakage: {'none' if not leak_pairs else leak_pairs}",
        f"- Source annotations unchanged: {source_unchanged}",
        f"- Missing source images copied/linked: {sum(len(v) for v in missing_images.values())}",
        f"- Original data write policy: only writes under `{dst_root}`.",
        "",
        "## Sampling notes",
        "",
    ]
    lines.extend(f"- {note}" for note in selection_notes)
    if any(missing_images.values()):
        lines += ["", "## Missing source image paths", ""]
        for split, paths in missing_images.items():
            for path in paths[:100]:
                lines.append(f"- {split}: `{path}`")
    (reports / "split_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    src_ann_dir = Path(args.src_ann_dir)
    src_img_root = Path(args.src_img_root)
    dst_root = Path(args.dst_root)
    reports = dst_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    source_paths = [
        src_ann_dir / "instances_train.json",
        src_ann_dir / "instances_val.json",
        src_ann_dir / "instances_test.json",
    ]
    source_before = source_snapshot(source_paths)

    train_coco = load_coco(src_ann_dir / "instances_train.json")
    val_coco = load_coco(src_ann_dir / "instances_val.json")
    test_coco = load_coco(src_ann_dir / "instances_test.json")
    id_to_name, name_to_id = category_maps(train_coco)

    keep_old_ids = {name_to_id[name] for name in KEPT_CLASSES}
    old_to_new_cat = {name_to_id[name]: i for i, name in enumerate(KEPT_CLASSES)}
    categories = [{"id": i, "name": name} for i, name in enumerate(KEPT_CLASSES)]

    train_source_records = build_records(train_coco, "train", keep_old_ids, id_to_name)
    val_source_records = build_records(val_coco, "val", keep_old_ids, id_to_name)
    test_records = build_records(test_coco, "test", keep_old_ids, id_to_name)
    pool_records = train_source_records + val_source_records

    original_train_counts, _ = split_counts(build_records(train_coco, "train", set(name_to_id.values()), id_to_name))
    removed_rows = [
        {
            "class_name": name,
            "original_train_instances": original_train_counts[name],
            "removed_reason": "original train_instances < 300",
        }
        for name in REMOVED_CLASSES
    ]
    mapping_rows = [
        {"old_category_id": name_to_id[name], "new_category_id": i, "class_name": name}
        for i, name in enumerate(KEPT_CLASSES)
    ]

    train_targets, val_targets = make_targets(args.target_train_instances, args.val_ratio)
    val_records = select_records(pool_records, val_targets, seed=args.seed + 17, mode="val")
    val_keys = {record.key for record in val_records}
    train_records = select_records(pool_records, train_targets, seed=args.seed, mode="train", forbidden_keys=val_keys)

    split_cocos = {}
    missing_images: dict[str, list[str]] = {}
    for split, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        coco, missing = build_coco_split(
            records,
            split,
            src_img_root,
            dst_root,
            old_to_new_cat,
            categories,
            args.link_mode,
        )
        split_cocos[split] = coco
        missing_images[split] = missing
        write_json(dst_root / "annotations" / f"instances_{split}.json", coco)

    validation = {
        split: validate_split(coco, dst_root / "images" / split)
        for split, coco in split_cocos.items()
    }
    source_after = source_snapshot(source_paths)

    selection_notes = []
    train_counts = {row["class_name"]: row["instances"] for row in count_coco(split_cocos["train"])}
    train_images = {row["class_name"]: row["images_with_class"] for row in count_coco(split_cocos["train"])}
    for name in KEPT_CLASSES:
        inst = train_counts.get(name, 0)
        imgs = train_images.get(name, 0)
        if name == OPEN_BOLL:
            if not (300 <= inst <= 450) or imgs < 25:
                selection_notes.append(f"{name}: train_instances={inst}, train_images={imgs}, outside preferred open-boll range.")
        elif not (260 <= inst <= 380) or imgs < 30:
            selection_notes.append(f"{name}: train_instances={inst}, train_images={imgs}, outside preferred generic range.")
    if not selection_notes:
        selection_notes.append("All retained classes are within the configured preferred training ranges.")

    make_report(
        dst_root,
        split_cocos,
        removed_rows,
        mapping_rows,
        validation,
        source_before,
        source_after,
        missing_images,
        selection_notes,
    )

    print(f"Wrote dataset to {dst_root}")
    for split in ("train", "val", "test"):
        v = validation[split]
        print(f"{split}: images={v['images']} annotations={v['annotations']} missing_paths={v['missing_image_paths']}")
    print(f"Report: {reports / 'split_integrity_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-ann-dir", required=True, help="Source COCO annotation directory.")
    parser.add_argument("--src-img-root", required=True, help="Source image root containing train/val/test folders.")
    parser.add_argument("--dst-root", required=True, help="Destination dataset root.")
    parser.add_argument("--target-train-instances", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
