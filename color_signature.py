# color_signature.py
import cv2
import numpy as np

# HSV ranges (tune per venue lighting)
# Red needs two ranges due to hue wrap-around
RED_RANGES = [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))]
BLUE_RANGE = ((95, 80, 60), (135, 255, 255))
WHITE_RANGE = ((0, 0, 200), (180, 50, 255))

def _mask_range(hsv, low, high):
    low = np.array(low, dtype=np.uint8)
    high = np.array(high, dtype=np.uint8)
    return cv2.inRange(hsv, low, high)

def compute_color_scores(bgr_crop):
    """Return fractional coverage for red, blue, white in [0,1]."""
    if bgr_crop.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    area = max(h * w, 1)

    red_mask = sum((_mask_range(hsv, *rng) for rng in RED_RANGES))
    blue_mask = _mask_range(hsv, *BLUE_RANGE)
    white_mask = _mask_range(hsv, *WHITE_RANGE)

    red = float(cv2.countNonZero(red_mask)) / area
    blue = float(cv2.countNonZero(blue_mask)) / area
    white = float(cv2.countNonZero(white_mask)) / area
    return red, blue, white

def compute_hist_signature(bgr_crop, bins=16):
    """Return normalized flattened HSV (H,S) histogram signature."""
    if bgr_crop.size == 0:
        return np.zeros((bins * bins,), dtype=np.float32)
    patch = cv2.resize(bgr_crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)

def signature_similarity(sig_a, sig_b, method=cv2.HISTCMP_CORREL):
    """Return similarity in [0,1] (clip if metric produces [-1,1])."""
    if sig_a is None or sig_b is None or len(sig_a) != len(sig_b) or len(sig_a) == 0:
        return 0.0
    s = cv2.compareHist(sig_a, sig_b, method)
    # CORREL returns [-1,1]; map to [0,1]
    return float(np.clip((s + 1.0) * 0.5, 0.0, 1.0))
