README - Autonomous Nanoprinting Optimization Pipeline

Overview

This project implements a closed-loop nanoprinting workflow that now grades each print from two camera views:

ML Prediction -> Mixing -> Printing -> Dual-Angle Image Capture -> Image Processing -> Feature Extraction -> Subgrades -> Final Grade -> Update History -> Repeat

The current implementation keeps the original project structure but upgrades the old single score into a modular grading engine that computes:

- top-view metrics: continuity, separation, diffusion, width uniformity, edge quality
- angled-view metrics: profile consistency, sagging/collapse, bulging
- interpretable 0-10 subgrades
- a weighted final grade
- defect labels and pass/borderline/fail flags

Project Files

- `main.py`: orchestrates one optimization iteration
- `ML_Bayesian.py`: suggests the next `mix_ratio` and `mix_time` using `final_grade`
- `camera_capture.py`: fake dual-camera capture now, real IDS hooks later
- `synthetic_print_generator.py`: parameter-driven synthetic dual-camera renderer
- `image_process.py`: normalization, segmentation, ROI extraction, dual-view analysis
- `feature_extractors.py`: raw metric extraction for top and angled views
- `grading_engine.py`: raw metrics -> subgrades -> final grade -> defect labels
- `config.py`: thresholds, paths, expected dimensions, grading weights
- `utils.py`: CSV schema migration, history helpers, flattening utilities
- `experiment_history.csv`: append-only experiment log after one-time schema migration
- `sample_images/`: fake camera assets

Fake Camera Workflow

`camera_capture.capture_images(...)` returns:

```python
{
    "top_view": "outputs/captures/generation_0005/top_view_synthetic.jpg",
    "angle_view": "outputs/captures/generation_0005/angle_view_synthetic.jpg",
}
```

In the current version, fake mode is parameter-driven: `main.py` passes `generation`, `mix_ratio`, and `mix_time` into `capture_images(...)`, which routes to `synthetic_print_generator.generate_synthetic_capture(...)`.

Each fake capture now writes:

- `outputs/captures/generation_0005/top_view_synthetic.jpg`
- `outputs/captures/generation_0005/angle_view_synthetic.jpg`
- `outputs/captures/generation_0005/synthetic_state.json`

Image Processing and Grading

Each view goes through:

1. grayscale conversion
2. Gaussian denoising
3. Otsu thresholding with polarity selection
4. morphology cleanup
5. contour-based print detection
6. rotation normalization
7. ROI cropping

Then the feature extractors compute raw metrics, and `grading_engine.py` converts them into:

- `continuity_grade`
- `separation_grade`
- `diffusion_grade`
- `width_uniformity_grade`
- `edge_quality_grade`
- `profile_grade`
- `sagging_grade`
- `bulge_grade`

The weighted final grade is optimized in the current loop. Hard penalties cap catastrophic failures such as missing print, severe discontinuity, or collapse.

Experiment History

`experiment_history.csv` now stores:

- process parameters
- image paths for both views
- all raw metrics
- all subgrades
- primary defect
- secondary defect
- quality flag
- final grade

Legacy four-column history files are automatically migrated into the richer schema the first time the pipeline runs.

Dependencies

Install the core packages:

```bash
pip install numpy opencv-python pandas scikit-image scipy bayesian-optimization
```

Optional real-camera support requires the IDS peak SDK Python bindings available in your environment.

How To Run

```bash
python main.py
```

Notes and TODOs

- Fake camera mode is enabled by default in `config.py`.
- Debug segmentation and overlay images are saved under `outputs/debug/` when `DEBUG_VISUALS = True`.
- `camera_capture.py` includes TODO hooks for dedicated top-camera and angle-camera hardware routing.
- `ML_Bayesian.py` is structured so the scalar `final_grade` objective can later be replaced with multi-objective or defect-aware optimization.
