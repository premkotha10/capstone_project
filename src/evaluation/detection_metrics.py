"""Detection metrics: precision, recall, AP50 and AP50-95 (COCO-style 101-point interpolation).

Evaluation is class-agnostic ("vehicle") by default. COCO-pretrained detectors and UA-DETRAC use different
taxonomies (no 'van' in COCO, UA-DETRAC 'others' vs COCO 'truck'), so a class-aware score would mostly measure the
label mismatch. Per-class results are reported only through an explicit mapping.

Ignored regions (UA-DETRAC `ignored_region`): detections whose area lies mostly inside an ignored region are
discarded before matching, because those areas are deliberately not annotated.

Small objects (NFR-4 / NT-4) are reported as a separate slice: ground truth with area < `small_area` px^2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
SMALL_AREA = 32.0 * 32.0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def fraction_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each box, the largest fraction of its area covered by any single region."""
    if boxes.size == 0:
        return np.zeros(0)
    if regions.size == 0:
        return np.zeros(len(boxes))
    x1 = np.maximum(boxes[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    return (inter / area[:, None]).max(axis=1)


@dataclass
class ImageEval:
    """Ground truth and predictions for one image (xyxy pixel boxes)."""

    gt_boxes: np.ndarray
    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    ignore_regions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def _match(image: ImageEval, iou_threshold: float, ignore_overlap: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy matching by descending score. Returns (scores, tp_flags) of kept predictions and the GT count."""
    keep = fraction_inside(image.pred_boxes, image.ignore_regions) < ignore_overlap
    boxes, scores = image.pred_boxes[keep], image.pred_scores[keep]
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    ious = iou_matrix(boxes, image.gt_boxes)
    matched = np.zeros(len(image.gt_boxes), dtype=bool)
    tp = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if ious.shape[1] == 0:
            break
        candidates = np.where(~matched & (ious[i] >= iou_threshold))[0]
        if candidates.size:
            best = candidates[np.argmax(ious[i, candidates])]
            matched[best] = True
            tp[i] = True
    return scores, tp, len(image.gt_boxes)


def average_precision(images: Sequence[ImageEval], iou_threshold: float, ignore_overlap: float = 0.5) -> float:
    """COCO-style 101-point interpolated AP at a single IoU threshold. NaN when there is no ground truth."""
    all_scores, all_tp, n_gt = [], [], 0
    for image in images:
        scores, tp, gt = _match(image, iou_threshold, ignore_overlap)
        all_scores.append(scores)
        all_tp.append(tp)
        n_gt += gt
    if n_gt == 0:
        return float("nan")
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, dtype=bool)
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(sampled.mean())


@dataclass(frozen=True)
class DetectionSummary:
    images: int
    gt_objects: int
    predictions: int
    precision: float  # at the evaluated confidence threshold, IoU 0.5
    recall: float
    ap50: float
    ap50_95: float
    small_gt_objects: int
    small_recall: float  # NFR-4 slice: recall on GT with area < SMALL_AREA at IoU 0.5


def _above(image: ImageEval, threshold: float) -> ImageEval:
    keep = image.pred_scores >= threshold
    return ImageEval(image.gt_boxes, image.pred_boxes[keep], image.pred_scores[keep], image.ignore_regions)


def summarize(
    images: Sequence[ImageEval], ignore_overlap: float = 0.5, score_threshold: float = 0.0
) -> DetectionSummary:
    """AP uses all predictions; precision/recall/small-recall use predictions with score >= `score_threshold`."""
    tp_total = pred_total = gt_total = 0
    small_total = small_hit = 0
    for image in (_above(i, score_threshold) for i in images):
        scores, tp, gt = _match(image, 0.5, ignore_overlap)
        tp_total += int(tp.sum())
        pred_total += len(scores)
        gt_total += gt
        if len(image.gt_boxes):
            areas = (image.gt_boxes[:, 2] - image.gt_boxes[:, 0]) * (image.gt_boxes[:, 3] - image.gt_boxes[:, 1])
            small = areas < SMALL_AREA
            if small.any():
                small_total += int(small.sum())
                keep = fraction_inside(image.pred_boxes, image.ignore_regions) < ignore_overlap
                ious = iou_matrix(image.gt_boxes[small], image.pred_boxes[keep])
                small_hit += int((ious.max(axis=1) >= 0.5).sum()) if ious.shape[1] else 0
    ap_by_iou = [average_precision(images, float(t), ignore_overlap) for t in IOU_THRESHOLDS]
    return DetectionSummary(
        images=len(images),
        gt_objects=gt_total,
        predictions=pred_total,
        precision=tp_total / pred_total if pred_total else float("nan"),
        recall=tp_total / gt_total if gt_total else float("nan"),
        ap50=ap_by_iou[0],
        ap50_95=float(np.mean(ap_by_iou)),
        small_gt_objects=small_total,
        small_recall=small_hit / small_total if small_total else float("nan"),
    )
