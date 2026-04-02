"""Shared helpers for filesystem, CSV history, and result flattening."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

from config import HISTORY_COLUMNS, THRESHOLDS


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a numeric value to a closed interval."""

    return max(lower, min(upper, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float, falling back to a default on failure."""

    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def history_target_from_row(row: Mapping[str, Any]) -> float:
    """Return the optimization target from a history row."""

    target_value = safe_float(row.get("final_grade", row.get("score", 0.0)))
    return target_value / 10.0 if target_value > 10.0 else target_value


def _coerce_csv_value(value: Any) -> str | float | int:
    """Normalize values before they are written into CSV."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)


def ensure_history_schema(history_file: Path | str) -> Path:
    """Ensure the experiment history exists and matches the expanded schema."""

    history_path = Path(history_file)
    ensure_directory(history_path.parent)

    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()
        return history_path

    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if fieldnames == HISTORY_COLUMNS:
        return history_path

    migrated_rows = [_migrate_history_row(row) for row in rows]

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated_rows)

    return history_path


def _migrate_history_row(row: Mapping[str, Any]) -> dict[str, str | float | int]:
    """Map a legacy row into the expanded CSV schema."""

    migrated = {column: "" for column in HISTORY_COLUMNS}
    migrated["generation"] = row.get("generation", "")
    migrated["mix_ratio"] = row.get("mix_ratio", "")
    migrated["mix_time"] = row.get("mix_time", "")

    final_grade = row.get("final_grade", row.get("score", ""))
    final_grade_value = safe_float(final_grade, 0.0)
    if final_grade not in ("", None) and final_grade_value > 10.0:
        final_grade_value /= 10.0
    migrated["final_grade"] = "" if final_grade in ("", None) else round(final_grade_value, 6)

    if final_grade not in ("", None):
        if final_grade_value >= THRESHOLDS["pass_grade"]:
            migrated["quality_flag"] = "pass"
        elif final_grade_value >= THRESHOLDS["borderline_grade"]:
            migrated["quality_flag"] = "borderline"
        else:
            migrated["quality_flag"] = "fail"

    for column in ("top_image_path", "angle_image_path", "primary_defect", "secondary_defect"):
        if column in row:
            migrated[column] = row[column]

    return migrated


def read_history_rows(history_file: Path | str) -> list[dict[str, str]]:
    """Read experiment history as a list of dictionaries."""

    history_path = ensure_history_schema(history_file)
    with history_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def get_next_generation(history_file: Path | str) -> int:
    """Return the next generation number based on the history file."""

    rows = read_history_rows(history_file)
    if not rows:
        return 1

    generations = [int(safe_float(row.get("generation", 0), 0.0)) for row in rows]
    return max(generations, default=0) + 1


def build_history_row(
    generation: int,
    mix_ratio: float,
    mix_time: float,
    image_paths: Mapping[str, str],
    analysis_result: Mapping[str, Any],
) -> dict[str, str | float | int]:
    """Flatten a structured analysis result into a CSV-ready experiment row."""

    row: dict[str, Any] = {column: "" for column in HISTORY_COLUMNS}
    row["generation"] = generation
    row["mix_ratio"] = mix_ratio
    row["mix_time"] = mix_time
    row["top_image_path"] = image_paths.get("top_view", "")
    row["angle_image_path"] = image_paths.get("angle_view", "")

    for section_name in ("raw_metrics", "grades", "summary"):
        section = analysis_result.get(section_name, {})
        if isinstance(section, Mapping):
            for key, value in section.items():
                if key in row:
                    row[key] = value

    return {column: _coerce_csv_value(row[column]) for column in HISTORY_COLUMNS}


def append_history_row(history_file: Path | str, row: Mapping[str, Any]) -> None:
    """Append one experiment row to the history file."""

    history_path = ensure_history_schema(history_file)
    with history_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writerow({column: _coerce_csv_value(row.get(column, "")) for column in HISTORY_COLUMNS})
