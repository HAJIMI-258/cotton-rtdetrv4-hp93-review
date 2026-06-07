"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category',
        'nms_iou_threshold',
        'score_threshold',
        'box_voting_iou_threshold',
        'box_voting_score_power',
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        nms_iou_threshold=None,
        score_threshold=0.0,
        box_voting_iou_threshold=None,
        box_voting_score_power=1.0,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = float(score_threshold)
        self.box_voting_iou_threshold = box_voting_iou_threshold
        self.box_voting_score_power = float(box_voting_score_power)
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return (
            f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, '
            f'num_top_queries={self.num_top_queries}, nms_iou_threshold={self.nms_iou_threshold}, '
            f'box_voting_iou_threshold={self.box_voting_iou_threshold}'
        )

    def _box_voting(self, keep_box, keep_lab, keep_sco, all_box, all_lab, all_sco):
        if (
            self.box_voting_iou_threshold is None
            or float(self.box_voting_iou_threshold) <= 0
            or keep_box.numel() == 0
            or all_box.numel() == 0
        ):
            return keep_box

        vote_thr = float(self.box_voting_iou_threshold)
        score_power = max(self.box_voting_score_power, 1e-6)
        refined = []
        for box, label in zip(keep_box, keep_lab):
            cls_mask = all_lab == label
            cls_boxes = all_box[cls_mask]
            cls_scores = all_sco[cls_mask]
            if cls_boxes.numel() == 0:
                refined.append(box)
                continue

            ious = torchvision.ops.box_iou(box.unsqueeze(0), cls_boxes).squeeze(0)
            vote_mask = ious >= vote_thr
            if not torch.any(vote_mask):
                refined.append(box)
                continue

            vote_boxes = cls_boxes[vote_mask]
            vote_scores = cls_scores[vote_mask].clamp_min(1e-6).pow(score_power)
            refined.append((vote_boxes * vote_scores.unsqueeze(1)).sum(dim=0) / vote_scores.sum())

        return torch.stack(refined, dim=0)

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            if self.score_threshold > 0:
                keep = sco >= self.score_threshold
                lab, box, sco = lab[keep], box[keep], sco[keep]
            all_lab, all_box, all_sco = lab, box, sco
            if (
                self.nms_iou_threshold is not None
                and float(self.nms_iou_threshold) > 0
                and box.numel() > 0
            ):
                keep = torchvision.ops.batched_nms(
                    box, sco, lab, float(self.nms_iou_threshold)
                )
                keep = keep[:self.num_top_queries]
                lab, box, sco = lab[keep], box[keep], sco[keep]
            if (
                self.box_voting_iou_threshold is not None
                and float(self.box_voting_iou_threshold) > 0
                and box.numel() > 0
            ):
                box = self._box_voting(box, lab, sco, all_box, all_lab, all_sco)
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
