"""Camera capture helpers for dual-view nanoprint inspection."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Mapping

import cv2

from config import IDEAL_MIX_RATIO, IDEAL_MIX_TIME, USE_FAKE_CAMERA
from synthetic_print_generator import generate_synthetic_capture
from utils import ensure_directory


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _capture_real_view(save_path: Path) -> str:
    """Capture one image from an IDS camera and save it to disk.

    TODO: Replace the generic first-camera selection with dedicated serial-number
    routing once the top-view and angle-view cameras are permanently assigned.
    """

    try:
        import ids_peak
        import ids_peak_ipl_extension
    except ImportError as exc:  # pragma: no cover - depends on vendor SDK
        raise RuntimeError("IDS camera SDK is not installed; use fake camera mode for now.") from exc

    ids_peak.Library.Initialize()
    try:
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()

        if device_manager.Devices().empty():
            raise RuntimeError("No IDS camera found.")

        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        datastream = device.DataStreams()[0].OpenDataStream()
        nodemap = device.RemoteDevice().NodeMaps()[0]

        payload_size = nodemap.FindNode("PayloadSize").Value()
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)

        nodemap.FindNode("AcquisitionStart").Execute()
        nodemap.FindNode("AcquisitionStart").WaitUntilDone()

        finished_buffer = datastream.WaitForFinishedBuffer(5000)
        image = ids_peak_ipl_extension.BufferToImage(finished_buffer)
        image_np = image.get_numpy_1D().reshape((image.Height(), image.Width()))

        ensure_directory(save_path.parent)
        cv2.imwrite(str(save_path), image_np)
        nodemap.FindNode("AcquisitionStop").Execute()
        return str(save_path)
    finally:  # pragma: no cover - depends on vendor SDK
        ids_peak.Library.Close()


def capture_images(
    output_dir: str | Path,
    generation: int | None = None,
    mix_ratio: float | None = None,
    mix_time: float | None = None,
    use_fake: bool = USE_FAKE_CAMERA,
    seed: int | None = None,
    sample_images: Mapping[str, str | Path] | None = None,
) -> dict[str, str]:
    """Capture or simulate both the top-view and angled-view inspection images."""

    output_path = ensure_directory(output_dir)

    if use_fake:
        if generation is None or mix_ratio is None or mix_time is None:
            raise ValueError("Synthetic fake capture requires generation, mix_ratio, and mix_time.")

        # ``sample_images`` is retained for API compatibility but synthetic mode
        # now generates parameter-driven captures instead of copying static files.
        return generate_synthetic_capture(
            output_dir=output_path,
            generation=int(generation),
            mix_ratio=float(mix_ratio),
            mix_time=float(mix_time),
            seed=seed,
        )

    capture_id = _timestamp_slug()
    top_path = output_path / f"top_view_{capture_id}.jpg"
    angle_path = output_path / f"angle_view_{capture_id}.jpg"

    # TODO: Replace the sequential capture placeholder with synchronized dual-camera capture.
    return {
        "top_view": _capture_real_view(top_path),
        "angle_view": _capture_real_view(angle_path),
    }


def capture_image(save_path: str = "current_capture.jpg", use_fake: bool = USE_FAKE_CAMERA) -> str:
    """Backward-compatible single-image capture helper."""

    target_path = Path(save_path)
    if use_fake:
        capture = generate_synthetic_capture(
            output_dir=target_path.parent,
            generation=0,
            mix_ratio=IDEAL_MIX_RATIO,
            mix_time=IDEAL_MIX_TIME,
            seed=0,
        )
        synthetic_top = Path(capture["top_view"])
        if synthetic_top.resolve() != target_path.resolve():
            ensure_directory(target_path.parent)
            shutil.copy2(synthetic_top, target_path)
        return str(target_path)

    return _capture_real_view(target_path)
