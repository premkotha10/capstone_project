import numpy as np
import pytest

from src.evaluation.detection_metrics import (
    ImageEval,
    average_precision,
    fraction_inside,
    iou_matrix,
    summarize,
)

GT = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=float)


def _img(pred: list[list[float]], scores: list[float], gt: np.ndarray = GT, ignore: list | None = None) -> ImageEval:
    return ImageEval(
        gt_boxes=gt,
        pred_boxes=np.array(pred, dtype=float).reshape(-1, 4),
        pred_scores=np.array(scores, dtype=float),
        ignore_regions=np.array(ignore or [], dtype=float).reshape(-1, 4),
    )


def test_iou_known_values() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_perfect_predictions_score_one() -> None:
    s = summarize([_img(GT.tolist(), [0.9, 0.8])])
    assert s.ap50 == pytest.approx(1.0)
    assert s.ap50_95 == pytest.approx(1.0)
    assert (s.precision, s.recall) == (1.0, 1.0)


def test_no_predictions() -> None:
    s = summarize([_img([], [])])
    assert s.ap50 == 0.0
    assert s.recall == 0.0


def test_no_ground_truth_gives_nan_ap() -> None:
    assert np.isnan(average_precision([_img([[0, 0, 5, 5]], [0.9], gt=np.zeros((0, 4)))], 0.5))


def test_high_scoring_false_positive_lowers_ap() -> None:
    # FP ranked first: precision at recall 0.5 is 0.5, at recall 1.0 is 2/3 -> envelope 2/3 everywhere
    img = _img([[100, 100, 110, 110], *GT.tolist()], [0.99, 0.9, 0.8])
    assert average_precision([img], 0.5) == pytest.approx(2 / 3, abs=1e-6)


def test_duplicate_detection_counts_as_false_positive() -> None:
    s = summarize([_img([GT[0].tolist(), GT[0].tolist(), GT[1].tolist()], [0.9, 0.85, 0.8])])
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == 1.0


def test_predictions_in_ignored_region_are_dropped() -> None:
    img = _img([*GT.tolist(), [100, 100, 110, 110]], [0.9, 0.8, 0.95], ignore=[[90, 90, 120, 120]])
    s = summarize([img])
    assert s.predictions == 2
    assert s.precision == 1.0


def test_fraction_inside() -> None:
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    np.testing.assert_allclose(fraction_inside(boxes, np.array([[0, 0, 5, 10]], dtype=float)), [0.5, 0.5])


def test_small_object_slice_reported_separately() -> None:
    gt = np.array([[0, 0, 10, 10], [100, 100, 200, 200]], dtype=float)  # one small (100 px^2), one large
    s = summarize([_img([[100, 100, 200, 200]], [0.9], gt=gt)])
    assert s.small_gt_objects == 1
    assert s.small_recall == 0.0  # aggregate recall 0.5 hides a total miss on small objects (NT-4 pattern)
    assert s.recall == 0.5
