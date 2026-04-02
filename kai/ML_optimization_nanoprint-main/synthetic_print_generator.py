"""Synthetic dual-camera image generation for nanoprint inspection.

The synthetic renderer converts the current process parameters into a latent
print state, then produces paired top-view and angled-view images that are
easy for the current CV pipeline to segment while still exhibiting realistic
defect trends.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import (
    IDEAL_MIX_RATIO,
    IDEAL_MIX_TIME,
    SYNTHETIC_ANGLE_VIEW_IMAGE_SIZE,
    SYNTHETIC_BASE_SEED,
    SYNTHETIC_GENERATION_CONFIG,
    SYNTHETIC_STATE_FILENAME,
    SYNTHETIC_TOP_VIEW_IMAGE_SIZE,
)
from utils import clamp, ensure_directory


def _derive_seed(mix_ratio: float, mix_time: float, seed: int | None = None) -> int:
    """Build a deterministic seed when one is not provided."""

    if seed is not None:
        return int(seed) & 0xFFFFFFFF

    payload = f"{SYNTHETIC_BASE_SEED}:{mix_ratio:.6f}:{mix_time:.6f}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _smooth_series(
    rng: np.random.Generator,
    count: int,
    amplitude: float,
    knot_count: int = 10,
) -> np.ndarray:
    """Create a smooth 1D noise signal by interpolating coarse random knots."""

    if count <= 0 or amplitude <= 0:
        return np.zeros(max(count, 0), dtype=np.float32)

    knot_count = max(4, min(knot_count, count))
    x_dense = np.linspace(0.0, 1.0, count, dtype=np.float32)
    x_knots = np.linspace(0.0, 1.0, knot_count, dtype=np.float32)
    y_knots = rng.normal(0.0, amplitude, knot_count).astype(np.float32)
    return np.interp(x_dense, x_knots, y_knots).astype(np.float32)


def _gaussian(x: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 1e-3)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _rotate_binary_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a binary mask while preserving a clean background."""

    center = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def _compose_grayscale_image(mask: np.ndarray, state: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Blend a binary print mask into a softly varying grayscale background."""

    height, width = mask.shape
    config = SYNTHETIC_GENERATION_CONFIG
    base_background = float(config["base_background_intensity"])
    base_print = float(config["base_print_intensity"])

    x_axis = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_axis = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)

    illumination_strength = float(state["illumination_strength"])
    illumination_tilt = float(state["illumination_tilt"])
    illumination = illumination_strength * (
        illumination_tilt * grid_x + (1.0 - abs(illumination_tilt)) * grid_y
    )

    image = np.full((height, width), base_background, dtype=np.float32)
    image += illumination
    image += rng.normal(0.0, float(state["sensor_noise_sigma"]), image.shape).astype(np.float32)

    print_texture = rng.normal(0.0, 3.0, image.shape).astype(np.float32)
    image[mask > 0] = base_print + print_texture[mask > 0]

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _json_ready_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert nested NumPy values into plain Python types for JSON output."""

    json_state: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.generic):
            json_state[key] = value.item()
        elif isinstance(value, list):
            json_state[key] = [_json_ready_state(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            json_state[key] = _json_ready_state(value)
        else:
            json_state[key] = value
    return json_state


def simulate_print_state(mix_ratio: float, mix_time: float, seed: int | None = None) -> dict[str, Any]:
    """Compute a latent synthetic print state from process parameters."""

    seed_value = _derive_seed(mix_ratio, mix_time, seed)
    rng = _rng_from_seed(seed_value)
    config = SYNTHETIC_GENERATION_CONFIG

    ratio_error = float(mix_ratio - IDEAL_MIX_RATIO)
    time_error = float(mix_time - IDEAL_MIX_TIME)

    ratio_strength = clamp(abs(ratio_error) / float(config["ratio_error_scale"]), 0.0, 1.2)
    time_strength = clamp(abs(time_error) / float(config["time_error_scale"]), 0.0, 1.2)

    diffusion_strength = clamp(0.15 + (0.95 * ratio_strength), 0.0, 1.4)
    continuity_strength = clamp(0.08 + (0.95 * time_strength), 0.0, 1.4)
    merge_strength = clamp(max(0.0, ratio_strength - 0.20) + 0.15 * diffusion_strength, 0.0, 1.4)
    roughness_strength = clamp(0.10 + (0.90 * time_strength), 0.0, 1.4)
    width_variation_strength = clamp(0.08 + (0.85 * time_strength) + (0.15 * ratio_strength), 0.0, 1.5)
    spacing_variation_strength = clamp(0.10 + (0.55 * ratio_strength), 0.0, 1.2)
    bulge_strength = clamp(0.05 + (0.80 * time_strength) + (0.20 * ratio_strength), 0.0, 1.5)
    profile_inconsistency_strength = clamp(0.08 + (0.90 * time_strength), 0.0, 1.4)
    sagging_strength = clamp(0.10 + (0.85 * ratio_strength) + (0.10 * time_strength), 0.0, 1.5)
    collapse_strength = clamp(max(0.0, sagging_strength - 0.65) + (0.20 * merge_strength), 0.0, 1.4)
    profile_height_scale = clamp(1.05 - (0.45 * ratio_strength) - (0.08 * time_strength), 0.35, 1.10)

    line_count = int(config["top_line_count"])
    line_width_bias = [float(rng.normal(0.0, 0.04 + 0.06 * width_variation_strength)) for _ in range(line_count)]
    line_offset_bias = [float(rng.normal(0.0, 0.20 + 0.40 * spacing_variation_strength)) for _ in range(line_count)]
    line_phase = [float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(line_count)]

    break_budget = int(round(continuity_strength * 4.0))
    breaks: list[dict[str, float | int]] = []
    for _ in range(break_budget):
        line_index = int(rng.integers(0, line_count))
        start_rel = float(rng.uniform(0.10, 0.78))
        width_rel = float(rng.uniform(0.02, 0.06 + 0.03 * continuity_strength))
        breaks.append(
            {
                "line_index": line_index,
                "start_rel": start_rel,
                "end_rel": min(0.95, start_rel + width_rel),
            }
        )

    bulge_budget = int(round(bulge_strength * 3.0))
    bulges: list[dict[str, float | int]] = []
    for _ in range(bulge_budget):
        bulges.append(
            {
                "line_index": int(rng.integers(0, line_count)),
                "x_rel": float(rng.uniform(0.12, 0.88)),
                "width_rel": float(rng.uniform(0.025, 0.08)),
                "amplitude": float(rng.uniform(0.25, 0.70 + 0.20 * bulge_strength)),
            }
        )

    merge_window_count = 0
    if merge_strength > 0.45:
        merge_window_count = 1 + int(merge_strength > 0.90)

    merge_windows: list[dict[str, float | int]] = []
    for _ in range(merge_window_count):
        pair_index = int(rng.integers(0, max(1, line_count - 1)))
        merge_windows.append(
            {
                "pair_index": pair_index,
                "x_rel": float(rng.uniform(0.18, 0.82)),
                "width_rel": float(rng.uniform(0.05, 0.12)),
                "strength": float(rng.uniform(0.30, merge_strength)),
            }
        )

    return {
        "seed": int(seed_value),
        "top_seed": int(rng.integers(0, 2**31 - 1)),
        "angle_seed": int(rng.integers(0, 2**31 - 1)),
        "mix_ratio": float(mix_ratio),
        "mix_time": float(mix_time),
        "ideal_mix_ratio": IDEAL_MIX_RATIO,
        "ideal_mix_time": IDEAL_MIX_TIME,
        "ratio_error": ratio_error,
        "time_error": time_error,
        "ratio_strength": float(ratio_strength),
        "time_strength": float(time_strength),
        "diffusion_strength": float(diffusion_strength),
        "continuity_strength": float(continuity_strength),
        "merge_strength": float(merge_strength),
        "roughness_strength": float(roughness_strength),
        "width_variation_strength": float(width_variation_strength),
        "spacing_variation_strength": float(spacing_variation_strength),
        "bulge_strength": float(bulge_strength),
        "profile_height_scale": float(profile_height_scale),
        "profile_inconsistency_strength": float(profile_inconsistency_strength),
        "sagging_strength": float(sagging_strength),
        "collapse_strength": float(collapse_strength),
        "global_rotation_deg": float(
            rng.uniform(-float(config["max_rotation_deg"]), float(config["max_rotation_deg"]))
        ),
        "illumination_tilt": float(rng.uniform(-0.9, 0.9)),
        "illumination_strength": float(
            config["illumination_strength"] * (0.75 + 0.35 * max(ratio_strength, time_strength))
        ),
        "sensor_noise_sigma": float(
            config["sensor_noise_sigma"] * (0.80 + 0.35 * max(ratio_strength, time_strength))
        ),
        "line_count": line_count,
        "line_width_bias": line_width_bias,
        "line_offset_bias": line_offset_bias,
        "line_phase": line_phase,
        "breaks": breaks,
        "bulges": bulges,
        "merge_windows": merge_windows,
    }


def render_top_view(state: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    """Render a synthetic top-view print image from the latent state."""

    height, width = image_size
    config = SYNTHETIC_GENERATION_CONFIG
    rng = _rng_from_seed(int(state["top_seed"]))

    mask = np.zeros((height, width), dtype=np.uint8)
    line_count = int(state["line_count"])
    x_margin = float(config["top_x_margin_rel"]) * width
    x_start = int(round(x_margin))
    x_end = int(round(width - x_margin))
    x_coords = np.arange(x_start, x_end, int(config["top_render_step_px"]), dtype=np.int32)
    x_rel = (x_coords - x_start) / max(1.0, float(x_end - x_start))

    base_separation = height * float(config["top_base_separation_rel"]) * (1.0 - 0.18 * float(state["merge_strength"]))
    base_center_y = height * float(config["top_center_y_rel"])
    spacing_jitter_px = height * (0.004 + 0.020 * float(state["spacing_variation_strength"]))

    half_index = (line_count - 1) / 2.0
    line_centers = [
        base_center_y + ((line_index - half_index) * base_separation) + (state["line_offset_bias"][line_index] * spacing_jitter_px)
        for line_index in range(line_count)
    ]

    base_width_px = max(
        float(config["top_min_width_px"]),
        height
        * (
            float(config["top_base_width_rel"])
            + float(config["top_diffusion_width_gain_rel"]) * float(state["diffusion_strength"])
        ),
    )

    for line_index in range(line_count):
        centerline_noise = _smooth_series(
            rng,
            len(x_coords),
            amplitude=height * (0.0015 + 0.0050 * float(state["roughness_strength"])),
            knot_count=14,
        )
        thickness_noise = _smooth_series(
            rng,
            len(x_coords),
            amplitude=base_width_px * (0.04 + 0.18 * float(state["width_variation_strength"])),
            knot_count=12,
        )

        sinusoid = np.sin((2.0 * np.pi * (1.5 + 0.3 * line_index) * x_rel) + float(state["line_phase"][line_index]))
        y_profile = line_centers[line_index] + centerline_noise + (0.6 + 1.8 * float(state["roughness_strength"])) * sinusoid
        thickness = (base_width_px * (1.0 + float(state["line_width_bias"][line_index]))) + thickness_noise

        for bulge in state["bulges"]:
            if int(bulge["line_index"]) != line_index:
                continue
            bulge_curve = _gaussian(x_rel, float(bulge["x_rel"]), float(bulge["width_rel"]))
            thickness += base_width_px * float(bulge["amplitude"]) * bulge_curve
            y_profile += height * 0.003 * float(bulge["amplitude"]) * bulge_curve

        for merge_window in state["merge_windows"]:
            pair_index = int(merge_window["pair_index"])
            if line_index not in (pair_index, pair_index + 1):
                continue

            local_weight = float(merge_window["strength"]) * _gaussian(
                x_rel,
                float(merge_window["x_rel"]),
                float(merge_window["width_rel"]),
            )
            midpoint = 0.5 * (line_centers[pair_index] + line_centers[pair_index + 1])
            y_profile = ((1.0 - local_weight) * y_profile) + (local_weight * midpoint)
            thickness += base_width_px * 0.60 * local_weight

        draw_mask = np.ones(len(x_coords), dtype=bool)
        for break_defect in state["breaks"]:
            if int(break_defect["line_index"]) != line_index:
                continue
            draw_mask &= ~(
                (x_rel >= float(break_defect["start_rel"])) & (x_rel <= float(break_defect["end_rel"]))
            )

        for x_value, y_value, thickness_value, should_draw in zip(x_coords, y_profile, thickness, draw_mask):
            if not should_draw:
                continue
            radius = int(round(max(1.0, float(thickness_value) / 2.0)))
            cv2.circle(mask, (int(x_value), int(round(y_value))), radius, 255, thickness=-1)

    diffusion_kernel = 1 + int(round(6.0 * float(state["diffusion_strength"])))
    if diffusion_kernel % 2 == 0:
        diffusion_kernel += 1
    if diffusion_kernel > 1:
        blurred = cv2.GaussianBlur(mask, (diffusion_kernel, diffusion_kernel), 0)
        _, mask = cv2.threshold(blurred, 18, 255, cv2.THRESH_BINARY)

    mask = _rotate_binary_mask(mask, float(state["global_rotation_deg"]))
    return _compose_grayscale_image(mask, state, rng)


def render_angle_view(state: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    """Render a synthetic angled-view profile image from the latent state."""

    height, width = image_size
    config = SYNTHETIC_GENERATION_CONFIG
    rng = _rng_from_seed(int(state["angle_seed"]))

    mask = np.zeros((height, width), dtype=np.uint8)
    x_margin = float(config["angle_x_margin_rel"]) * width
    x_start = int(round(x_margin))
    x_end = int(round(width - x_margin))
    x_coords = np.arange(x_start, x_end + 1, dtype=np.int32)
    x_rel = (x_coords - x_start) / max(1.0, float(x_end - x_start))

    baseline_y = height * float(config["angle_baseline_y_rel"])
    base_height = height * float(config["angle_base_height_rel"]) * float(state["profile_height_scale"])
    base_height *= 1.0 - 0.22 * float(state["sagging_strength"])
    base_height = max(6.0, base_height)

    profile_noise = _smooth_series(
        rng,
        len(x_coords),
        amplitude=base_height * (0.04 + 0.24 * float(state["profile_inconsistency_strength"])),
        knot_count=15,
    )
    sag_curve = _gaussian(x_rel, 0.5, 0.28)
    height_profile = np.full(len(x_coords), base_height, dtype=np.float32)
    height_profile += profile_noise
    height_profile *= 1.0 - (0.22 * float(state["sagging_strength"]) * sag_curve)
    height_profile *= 1.0 - (0.15 * float(state["collapse_strength"]))

    for bulge in state["bulges"]:
        bulge_curve = _gaussian(x_rel, float(bulge["x_rel"]), float(bulge["width_rel"]) * 0.8)
        height_profile += base_height * 0.80 * float(bulge["amplitude"]) * bulge_curve

    height_profile = np.clip(height_profile, 4.0, height * 0.35)

    baseline_noise = _smooth_series(
        rng,
        len(x_coords),
        amplitude=height * (0.002 + 0.004 * float(state["sagging_strength"])),
        knot_count=10,
    )
    baseline_profile = baseline_y + baseline_noise
    top_profile = baseline_profile - height_profile

    top_points = np.column_stack([x_coords, np.round(top_profile).astype(np.int32)])
    bottom_points = np.column_stack([x_coords[::-1], np.round(baseline_profile[::-1]).astype(np.int32)])
    polygon = np.vstack([top_points, bottom_points]).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [polygon], 255)

    skirt_radius = int(round(10 + 18 * float(state["sagging_strength"])))
    if skirt_radius > 0:
        kernel_width = max(3, skirt_radius | 1)
        skirt = cv2.GaussianBlur(mask, (kernel_width, 1), 0)
        _, skirt = cv2.threshold(skirt, 16, 255, cv2.THRESH_BINARY)
        mask = cv2.max(mask, skirt)

    mask = _rotate_binary_mask(mask, float(state["global_rotation_deg"]) * 0.6)
    return _compose_grayscale_image(mask, state, rng)


def generate_synthetic_capture(
    output_dir: str | Path,
    generation: int,
    mix_ratio: float,
    mix_time: float,
    seed: int | None = None,
) -> dict[str, str]:
    """Generate and save paired synthetic top-view and angled-view captures."""

    output_path = ensure_directory(output_dir)
    if seed is None:
        payload = f"{SYNTHETIC_BASE_SEED}:{generation}:{mix_ratio:.6f}:{mix_time:.6f}".encode("utf-8")
        capture_seed = int(hashlib.sha256(payload).hexdigest()[:8], 16)
    else:
        capture_seed = int(seed) & 0xFFFFFFFF
    state = simulate_print_state(mix_ratio, mix_time, seed=capture_seed)

    top_view = render_top_view(state, SYNTHETIC_TOP_VIEW_IMAGE_SIZE)
    angle_view = render_angle_view(state, SYNTHETIC_ANGLE_VIEW_IMAGE_SIZE)

    top_path = output_path / "top_view_synthetic.jpg"
    angle_path = output_path / "angle_view_synthetic.jpg"
    state_path = output_path / SYNTHETIC_STATE_FILENAME

    cv2.imwrite(str(top_path), top_view)
    cv2.imwrite(str(angle_path), angle_view)

    state_payload = _json_ready_state(
        {
            **state,
            "generation": int(generation),
            "top_view_path": str(top_path),
            "angle_view_path": str(angle_path),
        }
    )
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state_payload, handle, indent=2, sort_keys=True)

    return {
        "top_view": str(top_path),
        "angle_view": str(angle_path),
        "state_json": str(state_path),
    }
