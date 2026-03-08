"""OpenCV compatibility guards used across app entrypoints."""

from __future__ import annotations


def install_opencv_circle_guard() -> None:
    """Coerce circle centers to (x, y) so OpenCV accepts mixed point formats."""
    try:
        import cv2
    except Exception:
        return

    if getattr(cv2.circle, "_varbox_guarded", False):
        return

    original_circle = cv2.circle

    def _coerce_center(center):
        if isinstance(center, (tuple, list)) and len(center) >= 2:
            return int(center[0]), int(center[1])
        if hasattr(center, "shape") and hasattr(center, "size"):
            try:
                if int(center.size) >= 2:
                    flat = center.reshape(-1)
                    return int(flat[0]), int(flat[1])
            except Exception:
                return center
        return center

    def guarded_circle(img, center, radius, color, thickness=-1, lineType=None, shift=None):
        c = _coerce_center(center)
        if lineType is None and shift is None:
            return original_circle(img, c, radius, color, thickness)
        if shift is None:
            return original_circle(img, c, radius, color, thickness, lineType)
        return original_circle(img, c, radius, color, thickness, lineType, shift)

    setattr(guarded_circle, "_varbox_guarded", True)
    cv2.circle = guarded_circle
