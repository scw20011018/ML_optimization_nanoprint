"""Feature extraction for top-view and angled-view print inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    ANGLE_VIEW_CONFIG,
    EXPECTED_LINE_WIDTH_PX,
    EXPECTED_PROFILE_HEIGHT_PX,
    EXPECTED_SEPARATION_PX,
    TOP_VIEW_CONFIG,
)
from utils import clamp, ensure_directory

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except ImportError:  # pragma: no cover - optional dependency fallback
    skimage_skeletonize = None


def _empty_top_metrics() -> dict[str, float | int | bool]:
    return {
        "continuity_ratio": 0.0,
        "num_breaks": 0,
        "largest_gap_px": 0.0,
        "mean_separation_px": 0.0,
        "std_separation_px": 0.0,
        "min_separation_px": 0.0,
        "merge_count": 0,
        "mean_width_px": 0.0,
        "std_width_px": 0.0,
        "diffusion_ratio": 0.0,
        "width_cv": 0.0,
        "width_p10": 0.0,
        "width_p90": 0.0,
        "edge_roughness_score_raw": 1.0,
    }


def _empty_angle_metrics() -> dict[str, float | int | bool]:
    return {
        "mean_profile_height_px": 0.0,
        "std_profile_height_px": 0.0,
        "profile_consistency_raw": 0.0,
        "sagging_index": 1.0,
        "collapse_flag": True,
        "bulge_count": 0,
        "bulge_severity_raw": 0.0,
    }


def _save_debug_image(image: np.ndarray, output_path: Path | str | None) -> None:
    if output_path is None:
        return
    output = Path(output_path)
    ensure_directory(output.parent)
    cv2.imwrite(str(output), image)


def _find_runs(binary_line: np.ndarray) -> list[tuple[int, int, int]]:
    """Find contiguous runs of non-zero values in a 1D signal."""

    values = binary_line.astype(np.uint8).tolist()
    runs: list[tuple[int, int, int]] = []
    start: int | None = None

    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            end = index - 1
            runs.append((start, end, end - start + 1))
            start = None

    if start is not None:
        end = len(values) - 1
        runs.append((start, end, end - start + 1))

    return runs


def _skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary mask using scikit-image when available."""

    binary = (mask > 0).astype(np.uint8)

    if skimage_skeletonize is not None:
        return (skimage_skeletonize(binary > 0).astype(np.uint8) * 255)

    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    working = (binary * 255).copy()

    while True:
        opened = cv2.morphologyEx(working, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(working, opened)
        eroded = cv2.erode(working, element)
        skeleton = cv2.bitwise_or(skeleton, residue)
        working = eroded
        if cv2.countNonZero(working) == 0:
            break

    return skeleton


def _contour_roughness(mask: np.ndarray) -> float:
    """Estimate edge roughness using contour compactness."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    roughness_values: list[float] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        compactness = (perimeter * perimeter) / (4.0 * np.pi * area)
        roughness_values.append(max(0.0, compactness - 1.0))

    if not roughness_values:
        return 1.0

    return float(np.mean(roughness_values))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average."""

    if values.size == 0:
        return values

    if values.size < 3:
        return values.copy()

    window = max(3, min(window, values.size if values.size % 2 == 1 else values.size - 1))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def extract_top_view_metrics(
    normalized_view: dict[str, Any],
    debug_dir: Path | str | None = None,
) -> dict[str, float | int | bool]:
    """Extract continuity, separation, diffusion, width, and edge metrics."""

    roi_mask = normalized_view.get("roi_mask")
    roi_image = normalized_view.get("roi_image")

    if roi_mask is None or roi_mask.size == 0 or cv2.countNonZero(roi_mask) == 0:
        return _empty_top_metrics()

    binary = (roi_mask > 0).astype(np.uint8)
    skeleton = _skeletonize_mask(roi_mask)
    skeleton_binary = (skeleton > 0).astype(np.uint8)

    occupied_columns = np.where(skeleton_binary.any(axis=0))[0]
    if occupied_columns.size == 0:
        return _empty_top_metrics()

    start_column = int(occupied_columns[0])
    end_column = int(occupied_columns[-1])
    active_span = max(1, end_column - start_column + 1)
    span_signal = skeleton_binary.any(axis=0)[start_column : end_column + 1].astype(np.uint8)

    zero_runs = _find_runs((span_signal == 0).astype(np.uint8))
    continuity_ratio = float(np.mean(span_signal))
    num_breaks = int(sum(1 for _, _, length in zero_runs if length > 1))
    largest_gap_px = float(max((length for _, _, length in zero_runs), default=0))

    scan_count = min(TOP_VIEW_CONFIG["max_scan_columns"], active_span)
    scan_columns = np.unique(np.linspace(start_column, end_column, scan_count, dtype=int))

    width_samples: list[float] = []
    gap_samples: list[float] = []
    line_counts: list[int] = []
    merge_columns = 0

    for column in scan_columns:
        runs = _find_runs(binary[:, column])
        if not runs:
            continue
        width_samples.extend(float(length) for _, _, length in runs)
        line_counts.append(len(runs))

    expected_line_count = int(np.median(line_counts)) if line_counts else 1
    expected_line_count = max(expected_line_count, 1)

    for column in scan_columns:
        runs = _find_runs(binary[:, column])
        if not runs:
            continue

        column_gaps = [float(runs[index + 1][0] - runs[index][1] - 1) for index in range(len(runs) - 1)]
        gap_samples.extend(column_gaps)

        gap_too_small = any(gap < TOP_VIEW_CONFIG["merge_gap_threshold_px"] for gap in column_gaps)
        if len(runs) < expected_line_count or gap_too_small:
            merge_columns += 1

    if not width_samples:
        return _empty_top_metrics()

    width_array = np.asarray(width_samples, dtype=np.float32)
    mean_width_px = float(np.mean(width_array))
    std_width_px = float(np.std(width_array))
    width_cv = float(std_width_px / mean_width_px) if mean_width_px > 0 else 0.0
    diffusion_ratio = float((mean_width_px - EXPECTED_LINE_WIDTH_PX) / EXPECTED_LINE_WIDTH_PX)
    width_p10 = float(np.percentile(width_array, 10))
    width_p90 = float(np.percentile(width_array, 90))

    if gap_samples:
        gap_array = np.asarray(gap_samples, dtype=np.float32)
        mean_separation_px = float(np.mean(gap_array))
        std_separation_px = float(np.std(gap_array))
        min_separation_px = float(np.min(gap_array))
    else:
        mean_separation_px = EXPECTED_SEPARATION_PX
        std_separation_px = 0.0
        min_separation_px = EXPECTED_SEPARATION_PX

    edge_roughness_score_raw = _contour_roughness(roi_mask)

    if debug_dir is not None:
        debug_path = ensure_directory(debug_dir)
        overlay_base = roi_image.copy() if isinstance(roi_image, np.ndarray) else cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        skeleton_overlay = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        skeleton_overlay[skeleton > 0] = (0, 0, 255)
        for column in scan_columns:
            cv2.line(overlay_base, (int(column), 0), (int(column), overlay_base.shape[0] - 1), (255, 0, 0), 1)
        _save_debug_image(skeleton, debug_path / "top_skeleton.png")
        _save_debug_image(skeleton_overlay, debug_path / "top_skeleton_overlay.png")
        _save_debug_image(overlay_base, debug_path / "top_scan_overlay.png")

    return {
        "continuity_ratio": clamp(continuity_ratio, 0.0, 1.0),
        "num_breaks": num_breaks,
        "largest_gap_px": largest_gap_px,
        "mean_separation_px": mean_separation_px,
        "std_separation_px": std_separation_px,
        "min_separation_px": min_separation_px,
        "merge_count": int(merge_columns),
        "mean_width_px": mean_width_px,
        "std_width_px": std_width_px,
        "diffusion_ratio": diffusion_ratio,
        "width_cv": width_cv,
        "width_p10": width_p10,
        "width_p90": width_p90,
        "edge_roughness_score_raw": edge_roughness_score_raw,
    }


def extract_angle_view_metrics(
    normalized_view: dict[str, Any],
    debug_dir: Path | str | None = None,
) -> dict[str, float | int | bool]:
    """Extract profile, sagging, and bulging metrics from the angled view."""

    roi_mask = normalized_view.get("roi_mask")
    roi_image = normalized_view.get("roi_image")

    if roi_mask is None or roi_mask.size == 0 or cv2.countNonZero(roi_mask) == 0:
        return _empty_angle_metrics()

    binary = (roi_mask > 0).astype(np.uint8)
    column_heights = binary.sum(axis=0).astype(np.float32)
    active_columns = np.where(column_heights > 0)[0]
    if active_columns.size == 0:
        return _empty_angle_metrics()

    heights = column_heights[active_columns]
    mean_profile_height_px = float(np.mean(heights))
    std_profile_height_px = float(np.std(heights))

    if mean_profile_height_px > 0:
        profile_consistency_raw = float(clamp(1.0 - (std_profile_height_px / mean_profile_height_px), 0.0, 1.0))
    else:
        profile_consistency_raw = 0.0

    height_deficit = max(0.0, (EXPECTED_PROFILE_HEIGHT_PX - mean_profile_height_px) / EXPECTED_PROFILE_HEIGHT_PX)
    lower_half = binary[binary.shape[0] // 2 :, :]
    lower_mass_ratio = float(np.sum(lower_half) / max(1, np.sum(binary)))
    lower_mass_excess = max(0.0, (lower_mass_ratio - 0.50) / 0.50)
    sagging_index = float(clamp((0.65 * height_deficit) + (0.35 * lower_mass_excess), 0.0, 1.5))

    collapse_flag = bool(
        sagging_index >= 0.55 or mean_profile_height_px <= (EXPECTED_PROFILE_HEIGHT_PX * 0.55)
    )

    baseline = _moving_average(heights, ANGLE_VIEW_CONFIG["profile_smoothing_window"])
    residual = heights - baseline
    bulge_threshold = max(1.0, mean_profile_height_px * ANGLE_VIEW_CONFIG["bulge_height_threshold_ratio"])
    bulge_runs = _find_runs((residual > bulge_threshold).astype(np.uint8))
    bulge_count = int(len(bulge_runs))

    if bulge_count > 0:
        bulge_severity_raw = float(np.mean(np.maximum(0.0, residual)) / max(mean_profile_height_px, 1.0))
    else:
        bulge_severity_raw = 0.0

    if debug_dir is not None:
        debug_path = ensure_directory(debug_dir)
        profile_overlay = roi_image.copy() if isinstance(roi_image, np.ndarray) else cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        for local_index, column in enumerate(active_columns):
            height_value = int(round(heights[local_index]))
            cv2.line(
                profile_overlay,
                (int(column), profile_overlay.shape[0] - 1),
                (int(column), max(0, profile_overlay.shape[0] - 1 - height_value)),
                (0, 255, 255),
                1,
            )
        _save_debug_image(profile_overlay, debug_path / "angle_profile_overlay.png")

    return {
        "mean_profile_height_px": mean_profile_height_px,
        "std_profile_height_px": std_profile_height_px,
        "profile_consistency_raw": profile_consistency_raw,
        "sagging_index": sagging_index,
        "collapse_flag": collapse_flag,
        "bulge_count": bulge_count,
        "bulge_severity_raw": bulge_severity_raw,
    }
