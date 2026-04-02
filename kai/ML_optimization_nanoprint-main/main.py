"""Main workflow for autonomous nanoprinting optimization and grading."""

from __future__ import annotations

from pathlib import Path
from pprint import pformat

from ML_Bayesian import get_next_parameters
from camera_capture import capture_images
from config import CAPTURE_DIR, DEBUG_DIR, DEBUG_VISUALS, HISTORY_FILE, USE_FAKE_CAMERA
from grading_engine import build_failure_result
from image_process import analyze_images
from utils import append_history_row, build_history_row, ensure_history_schema, get_next_generation


def simulate_mixing(mix_ratio: float, mix_time: float) -> None:
    """Placeholder for the liquid handling / mixing stage."""

    print("\n[Mixing]")
    print(f"Preparing formulation with mix_ratio={mix_ratio:.4f}, mix_time={mix_time:.2f}s")


def simulate_printing() -> None:
    """Placeholder for robotic transfer and printer execution."""

    print("\n[Printing]")
    print("Simulating robotic transfer and print execution.")


def run_pipeline() -> dict[str, object]:
    """Execute one optimization iteration of the closed-loop workflow."""

    ensure_history_schema(HISTORY_FILE)
    generation = get_next_generation(HISTORY_FILE)

    next_params = get_next_parameters(str(HISTORY_FILE))
    mix_ratio = next_params["mix_ratio"]
    mix_time = next_params["mix_time"]

    print("\n===== Suggested Mixing Parameters =====")
    print(f"mix_ratio = {mix_ratio:.4f}")
    print(f"mix_time = {mix_time:.4f}")

    simulate_mixing(mix_ratio, mix_time)
    simulate_printing()

    capture_dir = Path(CAPTURE_DIR) / f"generation_{generation:04d}"
    debug_dir = (Path(DEBUG_DIR) / f"generation_{generation:04d}") if DEBUG_VISUALS else None

    image_paths: dict[str, str] = {"top_view": "", "angle_view": ""}
    try:
        image_paths = capture_images(
            capture_dir,
            generation=generation,
            mix_ratio=mix_ratio,
            mix_time=mix_time,
            use_fake=USE_FAKE_CAMERA,
            seed=generation,
        )
        analysis_result = analyze_images(
            image_paths["top_view"],
            image_paths["angle_view"],
            debug_root=debug_dir,
        )
    except Exception as exc:
        print(f"Capture or analysis failed: {exc}")
        analysis_result = build_failure_result(reason=type(exc).__name__)

    history_row = build_history_row(
        generation=generation,
        mix_ratio=mix_ratio,
        mix_time=mix_time,
        image_paths=image_paths,
        analysis_result=analysis_result,
    )
    append_history_row(HISTORY_FILE, history_row)

    print("\n===== Experiment Summary =====")
    print(f"Generation {generation}")
    print(f"Predicted params: mix_ratio={mix_ratio:.4f}, mix_time={mix_time:.2f}")
    print(f"Final grade: {analysis_result['summary']['final_grade']}")
    print(f"Primary defect: {analysis_result['summary']['primary_defect']}")
    print(f"Secondary defect: {analysis_result['summary']['secondary_defect']}")
    print(f"Quality flag: {analysis_result['summary']['quality_flag']}")

    print("\nSaved experiment record to", HISTORY_FILE)
    return {
        "generation": generation,
        "mix_ratio": mix_ratio,
        "mix_time": mix_time,
        "image_paths": image_paths,
        "analysis_result": analysis_result,
    }


if __name__ == "__main__":
    pipeline_result = run_pipeline()
    print("\nStructured analysis result:")
    print(pformat(pipeline_result["analysis_result"], sort_dicts=False))
