"""Compatibility wrapper around the fake camera path in camera_capture.py."""

from __future__ import annotations

from camera_capture import capture_image, capture_images

__all__ = ["capture_image", "capture_images"]
