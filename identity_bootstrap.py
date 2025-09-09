# identity_bootstrap.py
import numpy as np
from collections import defaultdict
from color_signature import compute_color_scores, compute_hist_signature, signature_similarity

class IdentityBootstrap:
    def __init__(self, frames=30, min_samples=5):
        self.frames = frames
        self.min_samples = min_samples
        self.frame_start = None
        self.samples = defaultdict(list)     # id -> [ (red,blue,white) ]
        self.signatures = {}                 # id -> running avg hist
        self.counts = defaultdict(int)
        self.finalized = False
        self.role_map = {}                   # id -> "BLUE"/"RED"/"REF"

    def _update_sig(self, boxer_id, sig, alpha=0.2):
        prev = self.signatures.get(boxer_id)
        if prev is None or prev.shape != sig.shape:
            self.signatures[boxer_id] = sig
        else:
            self.signatures[boxer_id] = (1 - alpha) * prev + alpha * sig

    def add_observation(self, frame_idx, boxer_id, frame_bgr, box_xyxy):
        if self.frame_start is None:
            self.frame_start = frame_idx
        x1, y1, x2, y2 = box_xyxy
        crop = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
        r, b, w = compute_color_scores(crop)
        self.samples[boxer_id].append((r, b, w))
        self._update_sig(boxer_id, compute_hist_signature(crop))
        self.counts[boxer_id] += 1

    def ready(self, frame_idx):
        return (self.frame_start is not None) and (frame_idx - self.frame_start >= self.frames)

    def finalize(self):
        if self.finalized:
            return self.role_map
        # Compute means
        means = {}
        for bid, lst in self.samples.items():
            if len(lst) >= self.min_samples:
                arr = np.array(lst, dtype=np.float32)  # columns (r,b,w)
                means[bid] = arr.mean(axis=0)
        if len(means) == 0:
            self.finalized = True
            return {}

        # Pick BLUE by max blue, RED by max red. REF ≈ max white not already taken.
        byid = list(means.items())
        blue_id = max(byid, key=lambda kv: kv[1][1])[0] if len(byid) else None
        red_id = max([kv for kv in byid if kv[0] != blue_id], key=lambda kv: kv[1][0])[0] if len(byid) > 1 else None
        remaining = [kv for kv in byid if kv[0] not in {blue_id, red_id}]
        ref_id = max(remaining, key=lambda kv: kv[1][2])[0] if remaining else None

        if blue_id is not None: self.role_map[blue_id] = "BLUE"
        if red_id is not None:  self.role_map[red_id]  = "RED"
        if ref_id is not None:  self.role_map[ref_id]  = "REF"

        self.finalized = True
        return self.role_map
