import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


def flip_boxes_back(boxes, width):
    flipped = boxes.clone()
    x1 = width - boxes[:, 2]
    x2 = width - boxes[:, 0]
    flipped[:, 0] = x1
    flipped[:, 2] = x2
    return flipped


def vflip_boxes_back(boxes, height):
    flipped = boxes.clone()
    y1 = height - boxes[:, 3]
    y2 = height - boxes[:, 1]
    flipped[:, 1] = y1
    flipped[:, 3] = y2
    return flipped


def threshold_mask(scores, labels, score_thr, class_score_thr):
    if not class_score_thr:
        return scores >= score_thr
    thresholds = torch.full_like(scores, float(score_thr))
    for class_id, value in class_score_thr.items():
        thresholds[labels == int(class_id)] = float(value)
    return scores >= thresholds


def filter_and_merge(boxes, scores, labels, score_thr, nms_thr, max_det, class_score_thr=None, class_iou_thr=None):
    if boxes.numel() == 0:
        return dict(boxes=boxes, scores=scores, labels=labels)

    keep = threshold_mask(scores, labels, score_thr, class_score_thr)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return dict(boxes=boxes, scores=scores, labels=labels)

    if class_iou_thr:
        keep_parts = []
        for cls in labels.unique():
            cls_keep = torch.nonzero(labels == cls, as_tuple=False).flatten()
            cls_thr = float(class_iou_thr.get(int(cls.item()), nms_thr))
            cls_nms = torchvision.ops.nms(boxes[cls_keep], scores[cls_keep], cls_thr)
            keep_parts.append(cls_keep[cls_nms])
        keep = torch.cat(keep_parts, dim=0) if keep_parts else scores.new_zeros((0,), dtype=torch.long)
        keep = keep[torch.argsort(scores[keep], descending=True)]
    else:
        keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_thr)
    keep = keep[:max_det]
    return dict(boxes=boxes[keep], scores=scores[keep], labels=labels[keep])


def weighted_boxes_fusion(
    boxes,
    scores,
    labels,
    score_thr,
    iou_thr,
    max_det,
    class_score_thr=None,
    class_iou_thr=None,
    score_mode="max",
    expected_views=1,
):
    if boxes.numel() == 0:
        return dict(boxes=boxes, scores=scores, labels=labels)

    keep = threshold_mask(scores, labels, score_thr, class_score_thr)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.numel() == 0:
        return dict(boxes=boxes, scores=scores, labels=labels)

    fused_boxes = []
    fused_scores = []
    fused_labels = []
    for cls in labels.unique():
        cls_keep = labels == cls
        cls_boxes = boxes[cls_keep]
        cls_scores = scores[cls_keep]
        order = torch.argsort(cls_scores, descending=True)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        cls_iou_thr = float(class_iou_thr.get(int(cls.item()), iou_thr)) if class_iou_thr else iou_thr
        while cls_scores.numel() > 0:
            ref = cls_boxes[:1]
            ious = torchvision.ops.box_iou(ref, cls_boxes).squeeze(0)
            cluster = ious >= cls_iou_thr
            cb = cls_boxes[cluster]
            cs = cls_scores[cluster]
            weights = cs.clamp_min(1e-6).unsqueeze(1)
            fused_boxes.append((cb * weights).sum(dim=0) / weights.sum())
            if score_mode == "mean":
                fused_score = cs.mean()
            elif score_mode == "avgmax":
                fused_score = 0.5 * (cs.max() + cs.mean())
            elif score_mode == "consensus":
                view_factor = min(float(cs.numel()), float(expected_views)) / max(float(expected_views), 1.0)
                fused_score = cs.max() * (0.5 + 0.5 * view_factor)
            else:
                fused_score = cs.max()
            fused_scores.append(fused_score)
            fused_labels.append(cls)
            remain = ~cluster
            cls_boxes = cls_boxes[remain]
            cls_scores = cls_scores[remain]

    if not fused_boxes:
        return dict(boxes=boxes[:0], scores=scores[:0], labels=labels[:0])

    out_boxes = torch.stack(fused_boxes, dim=0)
    out_scores = torch.stack(fused_scores, dim=0)
    out_labels = torch.stack(fused_labels, dim=0).to(labels.dtype)
    order = torch.argsort(out_scores, descending=True)[:max_det]
    return dict(boxes=out_boxes[order], scores=out_scores[order], labels=out_labels[order])


def parse_class_value_map(entries, class_lookup):
    values = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Class override must be NAME=VALUE or ID=VALUE, got: {entry}")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if key not in class_lookup:
            raise ValueError(f"Unknown class override key: {key}. Known keys: {sorted(class_lookup)}")
        values[int(class_lookup[key])] = float(raw_value)
    return values


def category_lookup_from_solver(solver):
    coco = solver.evaluator.coco_eval["bbox"].cocoGt
    lookup = {}
    for cat_id, cat in coco.cats.items():
        lookup[str(cat_id)] = int(cat_id)
        lookup[str(cat.get("name", cat_id))] = int(cat_id)
    return lookup


def enable_dynamic_eval_spatial_size(module):
    for child in module.modules():
        if hasattr(child, "eval_spatial_size"):
            child.eval_spatial_size = None


def per_class_ap50_from_evaluator(evaluator):
    coco_eval = evaluator.coco_eval["bbox"]
    precision = coco_eval.eval["precision"]
    params = coco_eval.params
    iou_idx = int(min(range(len(params.iouThrs)), key=lambda i: abs(float(params.iouThrs[i]) - 0.5)))
    area_idx = list(params.areaRngLbl).index("all") if "all" in params.areaRngLbl else 0
    maxdet_idx = len(params.maxDets) - 1
    per_class = {}
    for k, cat_id in enumerate(params.catIds):
        values = precision[iou_idx, :, k, area_idx, maxdet_idx]
        values = values[values > -1]
        name = coco_eval.cocoGt.cats[int(cat_id)].get("name", str(cat_id))
        per_class[name] = float(values.mean()) if values.size else None
    return per_class


@torch.no_grad()
def evaluate_tta(
    solvers,
    score_thr,
    nms_thr,
    max_det,
    use_flip,
    fuse_mode,
    sizes,
    pre_fuse_topk,
    use_vflip=False,
    wbf_score_mode="max",
    wbf_expected_views=1,
    class_score_thr=None,
    class_iou_thr=None,
):
    runtime = []
    for solver in solvers:
        model = solver.ema.module if solver.ema else solver.model
        model.eval()
        solver.criterion.eval()
        postprocessor = solver.postprocessor
        postprocessor.score_threshold = 0.0
        postprocessor.nms_iou_threshold = None
        postprocessor.num_top_queries = max(postprocessor.num_top_queries, max_det)
        runtime.append((model, postprocessor, solver.device))

    evaluator = solvers[0].evaluator
    evaluator.cleanup()
    device = solvers[0].device

    for samples, targets in solvers[0].val_dataloader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        all_results = []
        for model, postprocessor, model_device in runtime:
            model_samples = samples if model_device == device else samples.to(model_device)
            model_orig_sizes = orig_sizes if model_device == device else orig_sizes.to(model_device)
            for size in sizes:
                if model_samples.shape[-2:] == (size, size):
                    scaled_samples = model_samples
                else:
                    scaled_samples = F.interpolate(
                        model_samples,
                        size=(size, size),
                        mode="bilinear",
                        align_corners=False,
                    )
                outputs = model(scaled_samples)
                all_results.append(postprocessor(outputs, model_orig_sizes))
                if use_flip:
                    flip_outputs = model(torch.flip(scaled_samples, dims=[3]))
                    flip_results = postprocessor(flip_outputs, model_orig_sizes)
                    corrected = []
                    for item, orig_size in zip(flip_results, model_orig_sizes):
                        item = {k: v for k, v in item.items()}
                        item["boxes"] = flip_boxes_back(item["boxes"], orig_size[0].to(item["boxes"].device))
                        corrected.append(item)
                    all_results.append(corrected)
                if use_vflip:
                    vflip_outputs = model(torch.flip(scaled_samples, dims=[2]))
                    vflip_results = postprocessor(vflip_outputs, model_orig_sizes)
                    corrected = []
                    for item, orig_size in zip(vflip_results, model_orig_sizes):
                        item = {k: v for k, v in item.items()}
                        item["boxes"] = vflip_boxes_back(item["boxes"], orig_size[1].to(item["boxes"].device))
                        corrected.append(item)
                    all_results.append(corrected)
                if use_flip and use_vflip:
                    hvflip_outputs = model(torch.flip(scaled_samples, dims=[2, 3]))
                    hvflip_results = postprocessor(hvflip_outputs, model_orig_sizes)
                    corrected = []
                    for item, orig_size in zip(hvflip_results, model_orig_sizes):
                        item = {k: v for k, v in item.items()}
                        item["boxes"] = flip_boxes_back(item["boxes"], orig_size[0].to(item["boxes"].device))
                        item["boxes"] = vflip_boxes_back(item["boxes"], orig_size[1].to(item["boxes"].device))
                        corrected.append(item)
                    all_results.append(corrected)

        merged = {}
        for image_idx, target in enumerate(targets):
            boxes = torch.cat([result[image_idx]["boxes"] for result in all_results], dim=0)
            scores = torch.cat([result[image_idx]["scores"] for result in all_results], dim=0)
            labels = torch.cat([result[image_idx]["labels"] for result in all_results], dim=0)
            if pre_fuse_topk > 0 and scores.numel() > pre_fuse_topk:
                keep = torch.argsort(scores, descending=True)[:pre_fuse_topk]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if fuse_mode == "wbf":
                out = weighted_boxes_fusion(
                    boxes,
                    scores,
                    labels,
                    score_thr,
                    nms_thr,
                    max_det,
                    class_score_thr,
                    class_iou_thr,
                    wbf_score_mode,
                    wbf_expected_views,
                )
            else:
                out = filter_and_merge(
                    boxes, scores, labels, score_thr, nms_thr, max_det, class_score_thr, class_iou_thr
                )
            merged[int(target["image_id"].item())] = out

        evaluator.update(merged)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval["bbox"].stats.tolist()
    return stats, per_class_ap50_from_evaluator(evaluator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--resume", required=True, action="append")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score", type=float, action="append", default=[])
    parser.add_argument("--nms", type=float, action="append", default=[])
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--fuse-mode", choices=["nms", "wbf"], default="nms")
    parser.add_argument("--sizes", type=int, nargs="+", default=[768])
    parser.add_argument("--pre-fuse-topk", type=int, default=900)
    parser.add_argument("--wbf-score-mode", choices=["max", "mean", "avgmax", "consensus"], default="max")
    parser.add_argument("--wbf-expected-views", type=int, default=1)
    parser.add_argument("--class-score", action="append", default=[])
    parser.add_argument("--class-iou", action="append", default=[])
    args = parser.parse_args()

    dist_utils.setup_distributed(print_rank=0, print_method="builtin")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    solvers = []
    for idx, resume_path in enumerate(args.resume):
        cfg = YAMLConfig(
            args.config,
            resume=resume_path,
            device=args.device,
            output_dir=str(out_dir / f"solver_{idx}"),
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        if args.sizes != [768]:
            module = solver.ema.module if solver.ema else solver.model
            enable_dynamic_eval_spatial_size(module)
        solvers.append(solver)

    class_lookup = category_lookup_from_solver(solvers[0])
    class_score_thr = parse_class_value_map(args.class_score, class_lookup)
    class_iou_thr = parse_class_value_map(args.class_iou, class_lookup)

    scores = args.score or [0.001]
    nms_values = args.nms or [0.85]
    rows = []
    for score in scores:
        for nms in nms_values:
            name = f"{args.fuse_mode}_flip{int(args.flip)}_s{score:g}_iou{nms:g}"
            if args.vflip:
                name = f"{args.fuse_mode}_flip{int(args.flip)}v1_s{score:g}_iou{nms:g}"
            print(f"=== TTA EVAL {name} ===")
            stats, per_class_ap50 = evaluate_tta(
                solvers,
                score_thr=score,
                nms_thr=nms,
                max_det=args.max_det,
                use_flip=args.flip,
                fuse_mode=args.fuse_mode,
                sizes=args.sizes,
                pre_fuse_topk=args.pre_fuse_topk,
                use_vflip=args.vflip,
                wbf_score_mode=args.wbf_score_mode,
                wbf_expected_views=args.wbf_expected_views,
                class_score_thr=class_score_thr,
                class_iou_thr=class_iou_thr,
            )
            row = {
                "name": name,
                "fuse_mode": args.fuse_mode,
                "flip": int(args.flip),
                "vflip": int(args.vflip),
                "checkpoints": len(args.resume),
                "sizes": "+".join(str(x) for x in args.sizes),
                "pre_fuse_topk": args.pre_fuse_topk,
                "wbf_score_mode": args.wbf_score_mode,
                "wbf_expected_views": args.wbf_expected_views,
                "score": score,
                "iou": nms,
                "class_score": json.dumps(class_score_thr, sort_keys=True),
                "class_iou": json.dumps(class_iou_thr, sort_keys=True),
                "map": stats[0],
                "ap50": stats[1],
                "ap75": stats[2],
                "aps": stats[3],
                "apm": stats[4],
                "apl": stats[5],
                "ar100": stats[8],
            }
            for class_name, value in per_class_ap50.items():
                row[f"ap50_{class_name}"] = value
            rows.append(row)
            with (out_dir / "tta_ap50_results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with (out_dir / "tta_ap50_results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda x: x["ap50"], reverse=True))

    print("=== BEST ===")
    for row in sorted(rows, key=lambda x: x["ap50"], reverse=True):
        print(row)


if __name__ == "__main__":
    main()
