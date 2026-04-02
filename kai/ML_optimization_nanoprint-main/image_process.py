"""Image normalization, segmentation, and dual-view analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEBUG_DIR, DEBUG_VISUALS, SEGMENTATION_CONFIG
from feature_extractors import extract_angle_view_metrics, extract_top_view_metrics
from grading_engine import grade_print
from utils import ensure_directory


def _save_debug_image(image: np.ndarray, output_path: Path | None) -> None:
    if output_path is None:
        return
    ensure_directory(output_path.parent)
    cv2.imwrite(str(output_path), image)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than the configured area threshold."""

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_index] = 255

    return cleaned


def _score_mask_candidate(mask: np.ndarray) -> float:
    """Heuristic score for choosing between threshold polarities."""

    foreground_ratio = float(cv2.countNonZero(mask) / mask.size)
    if foreground_ratio <= 0.0 or foreground_ratio >= 0.90:
        return -1.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0

    largest_area = max(cv2.contourArea(contour) for contour in contours)
    largest_ratio = largest_area / mask.size
    ratio_penalty = abs(foreground_ratio - 0.12)
    return float((2.0 * largest_ratio) - ratio_penalty)


def _segment_image(gray: np.ndarray) -> np.ndarray:
    """Segment the printed region using Otsu thresholding and morphology."""

    kernel_size = SEGMENTATION_CONFIG["morph_kernel_size"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    _, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates: list[np.ndarray] = [otsu_binary, cv2.bitwise_not(otsu_binary)]
    best_mask: np.ndarray | None = None
    best_score = -1.0

    for candidate in candidates:
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = _remove_small_components(cleaned, SEGMENTATION_CONFIG["min_component_area"])
        score = _score_mask_candidate(cleaned)

        if score > best_score:
            best_score = score
            best_mask = cleaned

    return best_mask if best_mask is not None else np.zeros_like(gray)


def _largest_valid_contour(mask: np.ndarray) -> np.ndarray | None:
    """Return the largest contour that satisfies the minimum area threshold."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= SEGMENTATION_CONFIG["min_contour_area"]
    ]
    if not valid_contours:
        return None
    return max(valid_contours, key=cv2.contourArea)


def _rotate_to_long_axis(image: np.ndarray, mask: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image and mask so the dominant contour lies horizontally."""

    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    width, height = rect[1]

    if width < height:
        angle += 90.0

    center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(
        image,
        rotation,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, rotated_mask


def _crop_to_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop an image and mask around the detected print region."""

    points = cv2.findNonZero(mask)
    if points is None:
        return image, mask

    x, y, w, h = cv2.boundingRect(points)
    padding = SEGMENTATION_CONFIG["crop_padding_px"]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + w + padding)
    y1 = min(image.shape[0], y + h + padding)

    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def normalize_view(
    image_path: str | Path,
    view_name: str,
    debug_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load, normalize, segment, rotate, and crop one inspection view."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "view_name": view_name,
            "image_path": str(image_path),
            "missing_print": True,
            "error": f"Cannot load image: {image_path}",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_kernel = SEGMENTATION_CONFIG["gaussian_blur_kernel"]
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    mask = _segment_image(blurred)
    contour = _largest_valid_contour(mask)

    debug_path = Path(debug_dir) if debug_dir is not None else None
    if DEBUG_VISUALS and debug_path is not None:
        _save_debug_image(gray, debug_path / "01_gray.png")
        _save_debug_image(mask, debug_path / "02_mask.png")

    if contour is None:
        return {
            "view_name": view_name,
            "image_path": str(image_path),
            "image": image,
            "gray": gray,
            "mask": mask,
            "missing_print": True,
            "error": "No valid print region found.",
        }

    rotated_image, rotated_mask = _rotate_to_long_axis(image, mask, contour)
    roi_image, roi_mask = _crop_to_mask(rotated_image, rotated_mask)

    if DEBUG_VISUALS and debug_path is not None:
        outlined = image.copy()
        cv2.drawContours(outlined, [contour], -1, (0, 255, 0), 2)
        _save_debug_image(outlined, debug_path / "03_contour.png")
        _save_debug_image(rotated_mask, debug_path / "04_rotated_mask.png")
        _save_debug_image(roi_image, debug_path / "05_roi.png")
        _save_debug_image(roi_mask, debug_path / "06_roi_mask.png")

    return {
        "view_name": view_name,
        "image_path": str(image_path),
        "image": image,
        "gray": gray,
        "mask": mask,
        "roi_image": roi_image,
        "roi_mask": roi_mask,
        "missing_print": cv2.countNonZero(roi_mask) == 0,
    }


def analyze_images(
    top_view_image: str | Path,
    angle_view_image: str | Path,
    debug_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full dual-view grading workflow and return structured results."""

    debug_root_path = Path(debug_root) if debug_root is not None else None
    top_debug_dir = debug_root_path / "top_view" if debug_root_path is not None else None
    angle_debug_dir = debug_root_path / "angle_view" if debug_root_path is not None else None

    top_view = normalize_view(top_view_image, "top_view", top_debug_dir)
    angle_view = normalize_view(angle_view_image, "angle_view", angle_debug_dir)

    raw_metrics: dict[str, Any] = {}
    raw_metrics.update(extract_top_view_metrics(top_view, top_debug_dir if DEBUG_VISUALS else None))
    raw_metrics.update(extract_angle_view_metrics(angle_view, angle_debug_dir if DEBUG_VISUALS else None))
    raw_metrics["_missing_print"] = bool(top_view.get("missing_print", False))
    raw_metrics["_top_view_missing"] = bool(top_view.get("missing_print", False))
    raw_metrics["_angle_view_missing"] = bool(angle_view.get("missing_print", False))

    graded_result = grade_print(raw_metrics)

    return {
        "raw_metrics": {key: value for key, value in raw_metrics.items() if not key.startswith("_")},
        "grades": graded_result["grades"],
        "summary": graded_result["summary"],
    }


def analyze_image(image_path: str | Path) -> dict[str, Any]:
    """Backward-compatible single-image wrapper around the new grading engine."""

    default_debug_root = DEBUG_DIR / "single_view_compat"
    result = analyze_images(image_path, image_path, default_debug_root if DEBUG_VISUALS else None)
    return {
        **result,
        "score": result["summary"]["final_grade"],
    }
