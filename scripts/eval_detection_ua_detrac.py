"""Evaluate a detector on the downloaded UA-DETRAC subset, per sequence and per weather condition.

Usage:  python scripts/eval_detection_ua_detrac.py [--config configs/base.yaml] [--stride 5] [--tag NAME]

Protocol (recorded in the run file):
* class-agnostic 'vehicle' evaluation (COCO vs UA-DETRAC taxonomies differ);
* predictions computed with confidence 0.001 for AP; precision/recall at the config's confidence threshold;
* predictions mostly (>=50 %) inside UA-DETRAC ignored regions are discarded;
* only frames present in the annotation XML are evaluated, every `stride`-th frame;
* small-object slice = GT area < 32x32 px (NFR-4);
* detector latency measured per frame (batch 1) with a monotonic clock after one warm-up call.
This is a development measurement on sequences not yet assigned to partitions; it is NOT acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import PROJECT_ROOT, Settings, load_config  # noqa: E402
from src.common.hardware import select_device  # noqa: E402
from src.common.logging_setup import configure_logging  # noqa: E402
from src.datasets import ua_detrac  # noqa: E402
from src.datasets.manifest import DatasetManifest, verify_items  # noqa: E402
from src.detection.detector import UltralyticsDetector  # noqa: E402
from src.evaluation.detection_metrics import ImageEval, summarize  # noqa: E402
from src.evaluation.latency import latency_stats  # noqa: E402
from src.evaluation.runs import run_id, write_run_record  # noqa: E402

logger = logging.getLogger("eval_detection")
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "ua_detrac.yaml"


def _xyxy(boxes: list) -> np.ndarray:
    return np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float).reshape(-1, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--tag", default="pretrained")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings.log_level, settings.log_dir, "evaluation.jsonl")
    config = load_config(args.config)
    operating_threshold = config.detection.confidence_threshold
    eval_det_config = config.detection.model_copy(update={"confidence_threshold": 0.001})

    manifest = DatasetManifest.load(MANIFEST)
    problems = verify_items(manifest, settings.data_root)
    if problems:
        logger.error("manifest/data mismatch: %s", problems[:5])
        return 2
    sequences = [i for i in manifest.items if i.attributes.get("weather")]
    annotations: dict[str, ua_detrac.SequenceAnnotation] = {}
    for zip_item in (i for i in manifest.items if i.attributes.get("kind") == "annotations"):
        with zipfile.ZipFile(settings.data_root / zip_item.file) as zf:
            for member in zf.namelist():
                name = Path(member).stem
                if member.endswith(".xml") and any(s.id == name for s in sequences):
                    annotations[name] = ua_detrac.parse_sequence_xml(zf.read(member))

    detector = UltralyticsDetector(eval_det_config, settings.models_root, device=select_device(settings.device))
    latencies: list[float] = []
    per_sequence: dict[str, dict] = {}
    by_condition: dict[str, list[ImageEval]] = {}
    all_images: list[ImageEval] = []
    warmed = False
    for item in sequences:
        seq = annotations[item.id]
        ignore = _xyxy(seq.ignored_regions)
        image_dir = settings.data_root / item.file
        images: list[ImageEval] = []
        for gt_frame in seq.frames[:: args.stride]:
            image = cv2.imread(str(image_dir / f"img{gt_frame.frame_number:05d}.jpg"))
            if image is None:
                raise FileNotFoundError(f"{item.id} frame {gt_frame.frame_number} missing")
            dets = detector.detect([image], [gt_frame.frame_number])[0]
            if warmed:
                latencies.append(detector.last_inference_ms)
            warmed = True
            images.append(
                ImageEval(
                    gt_boxes=_xyxy([o.bbox for o in gt_frame.objects]),
                    pred_boxes=_xyxy([d.bbox for d in dets]),
                    pred_scores=np.array([d.confidence for d in dets], dtype=float),
                    ignore_regions=ignore,
                )
            )
        summary = summarize(images, score_threshold=operating_threshold)
        per_sequence[item.id] = {
            "weather": seq.weather,
            "site_proxy": item.attributes["site_proxy"],
            "mean_density": item.attributes["mean_density"],
            "occluded_fraction": item.attributes["occluded_fraction"],
            **summary.__dict__,
        }
        by_condition.setdefault(seq.weather, []).extend(images)
        all_images.extend(images)
        logger.info(
            "%s (%s): AP50 %.3f AP50-95 %.3f P %.3f R %.3f small-R %.3f [%d frames]",
            item.id,
            seq.weather,
            summary.ap50,
            summary.ap50_95,
            summary.precision,
            summary.recall,
            summary.small_recall,
            summary.images,
        )

    results = {
        "protocol": {
            "stride": args.stride,
            "class_agnostic": True,
            "ap_confidence": 0.001,
            "operating_threshold": operating_threshold,
            "ignore_overlap": 0.5,
            "small_area_px2": 1024,
        },
        "overall": summarize(all_images, score_threshold=operating_threshold).__dict__,
        "by_condition": {
            c: summarize(imgs, score_threshold=operating_threshold).__dict__ for c, imgs in sorted(by_condition.items())
        },
        "per_sequence": per_sequence,
        "detector_latency_ms": latency_stats(latencies).as_dict(),
    }
    out_dir = settings.reports_root / "phase5"
    name = run_id(f"detection_ua_detrac_{args.tag}")
    path = write_run_record(
        out_dir,
        name,
        config,
        inputs=[{"manifest": "ua_detrac", "id": i.id, "sha256": i.sha256} for i in sequences],
        model=detector.info.__dict__,
        results=results,
        notes=["Development measurement; sequences not yet partitioned; not acceptance (AC-3) evidence."],
    )
    print(
        json.dumps(
            {
                "overall": results["overall"],
                "by_condition": results["by_condition"],
                "latency": results["detector_latency_ms"],
            },
            indent=2,
        )
    )
    logger.info("run record: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
