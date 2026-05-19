"""
Cotton-specific hard-class copy-paste augmentation.

This transform is intentionally conservative: it only samples object crops from
configured hard classes, keeps the donor crop scale close to the original, and
rejects placements that heavily overlap existing boxes.
"""

import random
from typing import Iterable, Optional

import torch
import torchvision.transforms.v2 as T
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class HardClassCopyPaste(T.Transform):
    def __init__(
        self,
        target_category_names: Optional[Iterable[str]] = None,
        target_category_ids: Optional[Iterable[int]] = None,
        probability: float = 0.35,
        max_paste: int = 2,
        max_trials: int = 25,
        context_ratio: float = 0.18,
        scale_range=(0.90, 1.10),
        max_patch_ratio: float = 0.48,
        max_overlap_iou: float = 0.35,
        min_box_size: int = 4,
        max_epoch: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target_category_names = list(target_category_names or [])
        self.target_category_ids = set(int(x) for x in (target_category_ids or []))
        self.probability = float(probability)
        self.max_paste = int(max_paste)
        self.max_trials = int(max_trials)
        self.context_ratio = float(context_ratio)
        self.scale_range = tuple(float(x) for x in scale_range)
        self.max_patch_ratio = float(max_patch_ratio)
        self.max_overlap_iou = float(max_overlap_iou)
        self.min_box_size = int(min_box_size)
        self.max_epoch = max_epoch
        self._cached_dataset_id = None
        self._resolved_ids = set()
        self._warned_missing = False

    def _resolve_target_ids(self, dataset):
        dataset_id = id(dataset)
        if dataset_id == self._cached_dataset_id:
            return self._resolved_ids

        ids = set(self.target_category_ids)
        name_to_id = {}
        for cat in getattr(dataset, 'categories', []) or []:
            name_to_id[str(cat.get('name'))] = int(cat.get('id'))

        missing = []
        for name in self.target_category_names:
            if name in name_to_id:
                ids.add(name_to_id[name])
            else:
                missing.append(name)

        if missing and not self._warned_missing:
            print(f"     ### HardClassCopyPaste missing categories skipped: {missing} ###")
            self._warned_missing = True

        self._cached_dataset_id = dataset_id
        self._resolved_ids = ids
        return ids

    @staticmethod
    def _clone_target(target):
        cloned = {}
        for key, value in target.items():
            cloned[key] = value.clone() if torch.is_tensor(value) else value
        return cloned

    @staticmethod
    def _box_iou_one_to_many(box, boxes):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.float32)
        lt = torch.maximum(box[:2], boxes[:, :2])
        rb = torch.minimum(box[2:], boxes[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        return inter / (area1 + area2 - inter).clamp(min=1e-6)

    def _sample_donor(self, dataset, target_ids):
        for _ in range(self.max_trials):
            idx = random.randrange(len(dataset))
            donor_img, donor_target = dataset.load_item(idx)
            labels = donor_target.get('labels')
            boxes = donor_target.get('boxes')
            if labels is None or boxes is None or len(labels) == 0:
                continue
            mask = torch.tensor([int(label) in target_ids for label in labels], dtype=torch.bool)
            if not mask.any():
                continue
            candidates = mask.nonzero(as_tuple=False).flatten().tolist()
            obj_idx = random.choice(candidates)
            return donor_img, donor_target, obj_idx
        return None

    def _build_patch(self, donor_img, donor_target, obj_idx, base_size):
        if not isinstance(donor_img, Image.Image):
            return None

        donor_w, donor_h = donor_img.size
        boxes = torch.as_tensor(donor_target['boxes'], dtype=torch.float32)
        box = boxes[obj_idx].clone()
        x1, y1, x2, y2 = box.tolist()
        bw, bh = x2 - x1, y2 - y1
        if bw < self.min_box_size or bh < self.min_box_size:
            return None

        context = self.context_ratio * max(bw, bh)
        cx1 = max(0, int(round(x1 - context)))
        cy1 = max(0, int(round(y1 - context)))
        cx2 = min(donor_w, int(round(x2 + context)))
        cy2 = min(donor_h, int(round(y2 + context)))
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = donor_img.crop((cx1, cy1, cx2, cy2)).convert(donor_img.mode)
        rel_box = torch.tensor([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=torch.float32)
        scale = random.uniform(*self.scale_range)

        base_w, base_h = base_size
        max_pw = max(1, int(base_w * self.max_patch_ratio))
        max_ph = max(1, int(base_h * self.max_patch_ratio))
        new_w = max(1, int(round(crop.size[0] * scale)))
        new_h = max(1, int(round(crop.size[1] * scale)))
        fit_scale = min(1.0, max_pw / max(new_w, 1), max_ph / max(new_h, 1))
        new_w = max(1, int(round(new_w * fit_scale)))
        new_h = max(1, int(round(new_h * fit_scale)))
        if new_w >= base_w or new_h >= base_h:
            return None

        sx = new_w / max(crop.size[0], 1)
        sy = new_h / max(crop.size[1], 1)
        crop = crop.resize((new_w, new_h), Image.BILINEAR)
        rel_box = rel_box * torch.tensor([sx, sy, sx, sy], dtype=torch.float32)
        label = donor_target['labels'][obj_idx].view(1).clone()
        return crop, rel_box, label

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        if self.max_epoch is not None and getattr(dataset, 'epoch', 0) >= self.max_epoch:
            return image, target, dataset
        if self.probability <= 0 or random.random() > self.probability:
            return image, target, dataset
        if not isinstance(image, Image.Image):
            return image, target, dataset

        target_ids = self._resolve_target_ids(dataset)
        if not target_ids:
            return image, target, dataset

        out_img = image.copy()
        out_target = self._clone_target(target)
        base_w, base_h = out_img.size
        existing_boxes = torch.as_tensor(out_target.get('boxes', torch.empty((0, 4))), dtype=torch.float32)
        paste_count = random.randint(1, max(1, self.max_paste))

        new_boxes, new_labels, new_areas, new_iscrowd = [], [], [], []
        for _ in range(paste_count):
            donor = self._sample_donor(dataset, target_ids)
            if donor is None:
                break
            patch = self._build_patch(*donor, base_size=(base_w, base_h))
            if patch is None:
                continue
            crop, rel_box, label = patch
            crop_w, crop_h = crop.size

            placed = False
            for _trial in range(20):
                px = random.randint(0, max(0, base_w - crop_w))
                py = random.randint(0, max(0, base_h - crop_h))
                new_box = rel_box + torch.tensor([px, py, px, py], dtype=torch.float32)
                if (new_box[2] - new_box[0]) < self.min_box_size or (new_box[3] - new_box[1]) < self.min_box_size:
                    continue
                if existing_boxes.numel() and self._box_iou_one_to_many(new_box, existing_boxes).max().item() > self.max_overlap_iou:
                    continue
                out_img.paste(crop, (px, py))
                existing_boxes = torch.cat([existing_boxes, new_box.view(1, 4)], dim=0)
                new_boxes.append(new_box.view(1, 4))
                new_labels.append(label)
                new_areas.append(((new_box[2] - new_box[0]) * (new_box[3] - new_box[1])).view(1))
                new_iscrowd.append(torch.zeros(1, dtype=torch.int64))
                placed = True
                break
            if not placed:
                continue

        if new_boxes:
            out_target['boxes'] = convert_to_tv_tensor(
                torch.cat([torch.as_tensor(out_target['boxes'], dtype=torch.float32)] + new_boxes, dim=0),
                key='boxes',
                box_format='xyxy',
                spatial_size=out_img.size[::-1],
            )
            out_target['labels'] = torch.cat([out_target['labels']] + new_labels, dim=0)
            if 'area' in out_target:
                out_target['area'] = torch.cat([out_target['area'].to(torch.float32)] + new_areas, dim=0)
            if 'iscrowd' in out_target:
                out_target['iscrowd'] = torch.cat([out_target['iscrowd'].to(torch.int64)] + new_iscrowd, dim=0)

        return out_img, out_target, dataset
