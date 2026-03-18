import cv2
import shutil


def capture_image(save_path="current_capture.jpg"):
    source_image = "sample_break.jpg"

    image = cv2.imread(source_image)
    if image is None:
        raise RuntimeError(f"Cannot load test image: {source_image}")

    cv2.imwrite(save_path, image)

    return save_path