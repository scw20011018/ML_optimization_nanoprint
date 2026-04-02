"""Central configuration for the nanoprinting optimization pipeline."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

HISTORY_FILE = BASE_DIR / "experiment_history.csv"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "outputs"
CAPTURE_DIR = OUTPUT_DIR / "captures"
DEBUG_DIR = OUTPUT_DIR / "debug"

USE_FAKE_CAMERA = True
DEBUG_VISUALS = True

DEFAULT_SAMPLE_IMAGES = {
    "top_view": SAMPLE_IMAGES_DIR / "top_view_sample.jpg",
    "angle_view": SAMPLE_IMAGES_DIR / "angle_view_sample.jpg",
}

FAKE_CAMERA_FALLBACKS = {
    "top_view": [
        DEFAULT_SAMPLE_IMAGES["top_view"],
        BASE_DIR / "sample.jpg",
        BASE_DIR / "sample_break.jpg",
    ],
    "angle_view": [
        DEFAULT_SAMPLE_IMAGES["angle_view"],
        BASE_DIR / "sample_bend.jpg",
        BASE_DIR / "sample.jpg",
    ],
}

PARAMETER_BOUNDS = {
    "mix_ratio": (0.10, 0.90),
    "mix_time": (10.0, 60.0),
}

OPTIMIZER_RANDOM_STATE = 1
MIN_HISTORY_FOR_BO = 3

EXPECTED_LINE_WIDTH_PX = 20.0
EXPECTED_SEPARATION_PX = 30.0
EXPECTED_PROFILE_HEIGHT_PX = 12.0

IDEAL_MIX_RATIO = 0.50
IDEAL_MIX_TIME = 20.0

SYNTHETIC_BASE_SEED = 1729
SYNTHETIC_TOP_VIEW_IMAGE_SIZE = (720, 960)
SYNTHETIC_ANGLE_VIEW_IMAGE_SIZE = (720, 960)
SYNTHETIC_STATE_FILENAME = "synthetic_state.json"
SYNTHETIC_GENERATION_CONFIG = {
    "ratio_error_scale": 0.35,
    "time_error_scale": 25.0,
    "top_line_count": 3,
    "top_x_margin_rel": 0.10,
    "top_center_y_rel": 0.50,
    "top_base_separation_rel": 0.12,
    "top_base_width_rel": 0.018,
    "top_diffusion_width_gain_rel": 0.015,
    "top_min_width_px": 6.0,
    "angle_x_margin_rel": 0.08,
    "angle_baseline_y_rel": 0.78,
    "angle_base_height_rel": 0.18,
    "max_rotation_deg": 4.0,
    "base_background_intensity": 228.0,
    "base_print_intensity": 42.0,
    "illumination_strength": 18.0,
    "sensor_noise_sigma": 4.0,
    "top_render_step_px": 2,
}

SEGMENTATION_CONFIG = {
    "gaussian_blur_kernel": (5, 5),
    "morph_kernel_size": 3,
    "min_component_area": 80,
    "min_contour_area": 120,
    "crop_padding_px": 12,
}

TOP_VIEW_CONFIG = {
    "max_scan_columns": 160,
    "merge_gap_threshold_px": 5.0,
}

ANGLE_VIEW_CONFIG = {
    "profile_smoothing_window": 21,
    "bulge_height_threshold_ratio": 0.18,
}

GRADE_WEIGHTS = {
    "continuity": 0.25,
    "separation": 0.20,
    "diffusion": 0.15,
    "width_uniformity": 0.10,
    "edge_quality": 0.05,
    "profile": 0.10,
    "sagging": 0.10,
    "bulge": 0.05,
}

THRESHOLDS = {
    "continuity_fail": 0.50,
    "continuity_cap_grade": 4.0,
    "merge_penalty_per_event": 0.5,
    "max_merge_penalty": 2.0,
    "missing_print_grade": 0.5,
    "collapse_grade_cap": 4.5,
    "collapse_sagging_index": 0.55,
    "collapse_height_ratio": 0.55,
    "pass_grade": 7.5,
    "borderline_grade": 5.0,
    "defect_grade_cutoff": 6.0,
}

RAW_METRIC_FIELDS = [
    "continuity_ratio",
    "num_breaks",
    "largest_gap_px",
    "mean_separation_px",
    "std_separation_px",
    "min_separation_px",
    "merge_count",
    "mean_width_px",
    "std_width_px",
    "diffusion_ratio",
    "width_cv",
    "width_p10",
    "width_p90",
    "edge_roughness_score_raw",
    "mean_profile_height_px",
    "std_profile_height_px",
    "profile_consistency_raw",
    "sagging_index",
    "collapse_flag",
    "bulge_count",
    "bulge_severity_raw",
]

GRADE_FIELDS = [
    "continuity_grade",
    "separation_grade",
    "diffusion_grade",
    "width_uniformity_grade",
    "edge_quality_grade",
    "profile_grade",
    "sagging_grade",
    "bulge_grade",
]

SUMMARY_FIELDS = [
    "primary_defect",
    "secondary_defect",
    "quality_flag",
    "final_grade",
]

HISTORY_COLUMNS = [
    "generation",
    "mix_ratio",
    "mix_time",
    "top_image_path",
    "angle_image_path",
    *RAW_METRIC_FIELDS,
    *GRADE_FIELDS,
    *SUMMARY_FIELDS,
]
