"""Convert raw inspection metrics into interpretable grades and defect labels."""

from __future__ import annotations

from typing import Any, Mapping

from config import (
    EXPECTED_LINE_WIDTH_PX,
    EXPECTED_PROFILE_HEIGHT_PX,
    EXPECTED_SEPARATION_PX,
    GRADE_WEIGHTS,
    THRESHOLDS,
)
from utils import clamp


def _inverse_scale(value: float, slope: float) -> float:
    return clamp(10.0 - (slope * value), 0.0, 10.0)


def _target_scale(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 10.0
    error = abs(value - target) / tolerance
    return clamp(10.0 - (10.0 * error), 0.0, 10.0)


def _grade_continuity(raw_metrics: Mapping[str, Any]) -> float:
    score = 10.0 * float(raw_metrics.get("continuity_ratio", 0.0))
    score -= 0.6 * float(raw_metrics.get("num_breaks", 0))
    score -= 0.04 * float(raw_metrics.get("largest_gap_px", 0.0))
    return clamp(score, 0.0, 10.0)


def _grade_separation(raw_metrics: Mapping[str, Any]) -> float:
    mean_separation = float(raw_metrics.get("mean_separation_px", EXPECTED_SEPARATION_PX))
    std_separation = float(raw_metrics.get("std_separation_px", 0.0))
    min_separation = float(raw_metrics.get("min_separation_px", EXPECTED_SEPARATION_PX))
    merge_count = float(raw_metrics.get("merge_count", 0))

    mean_component = _target_scale(mean_separation, EXPECTED_SEPARATION_PX, EXPECTED_SEPARATION_PX * 0.70)
    std_component = _inverse_scale(std_separation / max(EXPECTED_SEPARATION_PX, 1.0), 12.0)
    min_component = _target_scale(min_separation, EXPECTED_SEPARATION_PX, EXPECTED_SEPARATION_PX * 0.80)
    merge_component = clamp(10.0 - (2.5 * merge_count), 0.0, 10.0)

    return clamp(
        (0.40 * mean_component) + (0.20 * std_component) + (0.20 * min_component) + (0.20 * merge_component),
        0.0,
        10.0,
    )


def _grade_diffusion(raw_metrics: Mapping[str, Any]) -> float:
    diffusion_ratio = abs(float(raw_metrics.get("diffusion_ratio", 0.0)))
    width_error = abs(float(raw_metrics.get("mean_width_px", EXPECTED_LINE_WIDTH_PX)) - EXPECTED_LINE_WIDTH_PX)
    diffusion_component = _inverse_scale(diffusion_ratio, 25.0)
    width_component = _target_scale(float(raw_metrics.get("mean_width_px", EXPECTED_LINE_WIDTH_PX)), EXPECTED_LINE_WIDTH_PX, EXPECTED_LINE_WIDTH_PX)
    return clamp((0.70 * diffusion_component) + (0.30 * width_component) - (0.05 * width_error), 0.0, 10.0)


def _grade_width_uniformity(raw_metrics: Mapping[str, Any]) -> float:
    width_cv = float(raw_metrics.get("width_cv", 0.0))
    width_spread = float(raw_metrics.get("width_p90", 0.0)) - float(raw_metrics.get("width_p10", 0.0))
    cv_component = _inverse_scale(width_cv, 30.0)
    spread_component = _inverse_scale(width_spread / max(EXPECTED_LINE_WIDTH_PX, 1.0), 8.0)
    return clamp((0.70 * cv_component) + (0.30 * spread_component), 0.0, 10.0)


def _grade_edge_quality(raw_metrics: Mapping[str, Any]) -> float:
    return _inverse_scale(float(raw_metrics.get("edge_roughness_score_raw", 1.0)), 12.0)


def _grade_profile(raw_metrics: Mapping[str, Any]) -> float:
    mean_height = float(raw_metrics.get("mean_profile_height_px", 0.0))
    consistency = float(raw_metrics.get("profile_consistency_raw", 0.0))
    height_component = _target_scale(mean_height, EXPECTED_PROFILE_HEIGHT_PX, EXPECTED_PROFILE_HEIGHT_PX)
    consistency_component = clamp(consistency * 10.0, 0.0, 10.0)
    return clamp((0.45 * height_component) + (0.55 * consistency_component), 0.0, 10.0)


def _grade_sagging(raw_metrics: Mapping[str, Any]) -> float:
    sagging_index = float(raw_metrics.get("sagging_index", 0.0))
    collapse_flag = bool(raw_metrics.get("collapse_flag", False))
    score = _inverse_scale(sagging_index, 12.0)
    if collapse_flag:
        score -= 2.5
    return clamp(score, 0.0, 10.0)


def _grade_bulging(raw_metrics: Mapping[str, Any]) -> float:
    bulge_count = float(raw_metrics.get("bulge_count", 0))
    bulge_severity = float(raw_metrics.get("bulge_severity_raw", 0.0))
    score = 10.0 - (1.5 * bulge_count) - (20.0 * bulge_severity)
    return clamp(score, 0.0, 10.0)


def calculate_subgrades(raw_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Generate the eight required 0-10 subgrades."""

    return {
        "continuity_grade": _grade_continuity(raw_metrics),
        "separation_grade": _grade_separation(raw_metrics),
        "diffusion_grade": _grade_diffusion(raw_metrics),
        "width_uniformity_grade": _grade_width_uniformity(raw_metrics),
        "edge_quality_grade": _grade_edge_quality(raw_metrics),
        "profile_grade": _grade_profile(raw_metrics),
        "sagging_grade": _grade_sagging(raw_metrics),
        "bulge_grade": _grade_bulging(raw_metrics),
    }


def _weighted_final_grade(grades: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return clamp(
        (
            weights["continuity"] * grades["continuity_grade"]
            + weights["separation"] * grades["separation_grade"]
            + weights["diffusion"] * grades["diffusion_grade"]
            + weights["width_uniformity"] * grades["width_uniformity_grade"]
            + weights["edge_quality"] * grades["edge_quality_grade"]
            + weights["profile"] * grades["profile_grade"]
            + weights["sagging"] * grades["sagging_grade"]
            + weights["bulge"] * grades["bulge_grade"]
        ),
        0.0,
        10.0,
    )


def _apply_hard_penalties(final_grade: float, raw_metrics: Mapping[str, Any]) -> float:
    """Apply configurable caps and penalties for catastrophic defects."""

    adjusted_grade = float(final_grade)

    if bool(raw_metrics.get("_missing_print", False)):
        return THRESHOLDS["missing_print_grade"]

    if float(raw_metrics.get("continuity_ratio", 1.0)) < THRESHOLDS["continuity_fail"]:
        adjusted_grade = min(adjusted_grade, THRESHOLDS["continuity_cap_grade"])

    merge_count = int(float(raw_metrics.get("merge_count", 0)))
    if merge_count > 0:
        adjusted_grade -= min(
            THRESHOLDS["max_merge_penalty"],
            THRESHOLDS["merge_penalty_per_event"] * merge_count,
        )

    if bool(raw_metrics.get("collapse_flag", False)):
        adjusted_grade = min(adjusted_grade, THRESHOLDS["collapse_grade_cap"])

    return clamp(adjusted_grade, 0.0, 10.0)


def _defect_severity(grades: Mapping[str, float], raw_metrics: Mapping[str, Any]) -> list[tuple[str, float]]:
    severities = {
        "discontinuity": (10.0 - grades["continuity_grade"]) + (0.4 * float(raw_metrics.get("num_breaks", 0))),
        "line_merge": (10.0 - grades["separation_grade"]) + (1.2 * float(raw_metrics.get("merge_count", 0))),
        "diffusion": (10.0 - grades["diffusion_grade"]) + (4.0 * max(0.0, float(raw_metrics.get("diffusion_ratio", 0.0)))),
        "nonuniform_width": 10.0 - grades["width_uniformity_grade"],
        "rough_edges": 10.0 - grades["edge_quality_grade"],
        "sagging": (10.0 - grades["sagging_grade"]) + (1.5 if bool(raw_metrics.get("collapse_flag", False)) else 0.0),
        "bulging": (10.0 - grades["bulge_grade"]) + (0.5 * float(raw_metrics.get("bulge_count", 0))),
    }
    return sorted(severities.items(), key=lambda item: item[1], reverse=True)


def _classify_defects(grades: Mapping[str, float], raw_metrics: Mapping[str, Any], final_grade: float) -> dict[str, str]:
    """Assign primary defect, secondary defect, and quality flag."""

    if bool(raw_metrics.get("_missing_print", False)):
        return {
            "primary_defect": "missing_print",
            "secondary_defect": "",
            "quality_flag": "fail",
        }

    severities = _defect_severity(grades, raw_metrics)
    weak_grade_count = sum(1 for value in grades.values() if value < THRESHOLDS["defect_grade_cutoff"])

    if final_grade >= THRESHOLDS["pass_grade"] and weak_grade_count == 0:
        return {
            "primary_defect": "good_print",
            "secondary_defect": "",
            "quality_flag": "pass",
        }

    top_label, top_severity = severities[0]
    secondary_label = ""
    if len(severities) > 1 and severities[1][1] > 1.0:
        secondary_label = severities[1][0]

    if weak_grade_count >= 3 and top_severity > 2.0:
        primary_defect = "multiple_defects"
        secondary_defect = top_label
    else:
        primary_defect = top_label if top_severity > 0.5 else "good_print"
        secondary_defect = secondary_label if primary_defect != "good_print" else ""

    if final_grade >= THRESHOLDS["pass_grade"]:
        quality_flag = "pass"
    elif final_grade >= THRESHOLDS["borderline_grade"]:
        quality_flag = "borderline"
    else:
        quality_flag = "fail"

    return {
        "primary_defect": primary_defect,
        "secondary_defect": secondary_defect,
        "quality_flag": quality_flag,
    }


def grade_print(
    raw_metrics: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> dict[str, dict[str, float | str]]:
    """Generate subgrades, a weighted final grade, and defect labels."""

    weights = weights or GRADE_WEIGHTS
    grades = calculate_subgrades(raw_metrics)
    final_grade = _weighted_final_grade(grades, weights)
    final_grade = _apply_hard_penalties(final_grade, raw_metrics)
    summary = _classify_defects(grades, raw_metrics, final_grade)
    summary["final_grade"] = round(final_grade, 3)

    rounded_grades = {name: round(value, 3) for name, value in grades.items()}
    return {
        "grades": rounded_grades,
        "summary": summary,
    }


def build_failure_result(reason: str = "analysis_failed") -> dict[str, dict[str, float | str]]:
    """Fallback result for capture or analysis failures."""

    raw_metrics = {
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
        "mean_profile_height_px": 0.0,
        "std_profile_height_px": 0.0,
        "profile_consistency_raw": 0.0,
        "sagging_index": 1.0,
        "collapse_flag": True,
        "bulge_count": 0,
        "bulge_severity_raw": 0.0,
        "_missing_print": True,
    }
    graded = grade_print(raw_metrics)
    graded["summary"]["primary_defect"] = "missing_print"
    graded["summary"]["secondary_defect"] = reason
    graded["summary"]["quality_flag"] = "fail"
    return {
        "raw_metrics": {key: value for key, value in raw_metrics.items() if not key.startswith("_")},
        "grades": graded["grades"],
        "summary": graded["summary"],
    }
