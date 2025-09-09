# participant_manager.py
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from color_signature import compute_color_scores, compute_hist_signature, signature_similarity

ROLE_COLOR = {
    "BLUE": (255, 80, 60),
    "RED":  (0, 0, 255),
}

class ParticipantManager:
    """
    Lock scoring to exactly TWO roles: RED and BLUE.

    - Bootstrap: seed RED/BLUE anchors by color dominance (or tracker roles if present).
    - Freeze anchors after seeding to prevent drift.
    - Each frame, assign transient detections to RED/BLUE by best histogram similarity.
    - If a role is temporarily missing, keep last ID sticky up to `max_missing_frames`.
    - Scoring always maps to canonical roles ("RED"/"BLUE"), never to raw IDs.
    """

    def __init__(self,
                 min_color_fraction: float = 0.03,
                 min_sim_accept: float = 0.30,
                 smooth_alpha: float = 0.15,
                 max_missing_frames: int = 45,  # ~1.5s at 30fps
                 freeze_anchors_after_seed: bool = True):
        self.roles = ("RED", "BLUE")
        self.anchors: Dict[str, Optional[np.ndarray]] = {"RED": None, "BLUE": None}  # HSV hist anchors
        self.anchor_counts: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.smooth_alpha = smooth_alpha
        self.min_color_fraction = min_color_fraction
        self.min_sim_accept = min_sim_accept
        self.freeze_anchors_after_seed = freeze_anchors_after_seed
        self.anchors_frozen = False

        # current assignment
        self.id_to_role: Dict[int, str] = {}
        self.role_to_id: Dict[str, Optional[int]] = {"RED": None, "BLUE": None}

        # visibility / stickiness
        self.role_missing_frames: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.max_missing_frames = max_missing_frames

    # ----------------- internal utils -----------------
    @staticmethod
    def _crop(frame, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        return frame[y1:y2, x1:x2]

    def _update_anchor(self, role: str, sig: np.ndarray):
        if self.anchors_frozen:
            return
        prev = self.anchors.get(role)
        if prev is None or prev.shape != sig.shape:
            self.anchors[role] = sig
        else:
            a = self.smooth_alpha
            self.anchors[role] = (1 - a) * prev + a * sig
        self.anchor_counts[role] += 1

    # ----------------- seeding -----------------
    def _seed_from_colors(self, detections: List[Tuple[int, Tuple[int,int,int,int], np.ndarray, Tuple[float,float,float]]]):
        # RED by red fraction; BLUE by blue fraction
        if self.anchors["RED"] is None:
            best_id, best_sig, best_val = None, None, -1.0
            for bid, box, sig, (r, b, w) in detections:
                if r > best_val and r >= self.min_color_fraction:
                    best_id, best_sig, best_val = bid, sig, r
            if best_sig is not None:
                self._update_anchor("RED", best_sig)

        if self.anchors["BLUE"] is None:
            best_id, best_sig, best_val = None, None, -1.0
            for bid, box, sig, (r, b, w) in detections:
                if b > best_val and b >= self.min_color_fraction:
                    best_id, best_sig, best_val = bid, sig, b
            if best_sig is not None:
                self._update_anchor("BLUE", best_sig)

        # Optionally freeze after first good seed to avoid drift
        if self.freeze_anchors_after_seed and all(self.anchors[r] is not None for r in self.roles):
            self.anchors_frozen = True

    def _seed_from_tracker_roles(self, frame, poses: Dict[int, dict]):
        for bid, data in poses.items():
            role = data.get("role")
            if role in self.roles and self.anchors[role] is None:
                crop = self._crop(frame, data["box"])
                sig = compute_hist_signature(crop)
                if sig is not None and sig.size:
                    self._update_anchor(role, sig)
        if self.freeze_anchors_after_seed and all(self.anchors[r] is not None for r in self.roles):
            self.anchors_frozen = True

    # ----------------- public per-frame update -----------------
    def update(self, frame, poses: Dict[int, dict]):
        # Build candidates
        detections = []
        for bid, data in poses.items():
            box = data["box"]
            crop = self._crop(frame, box)
            sig = compute_hist_signature(crop)
            r, b, w = compute_color_scores(crop)
            detections.append((bid, box, sig, (r, b, w)))

        # Seed if needed
        if self.anchors["RED"] is None or self.anchors["BLUE"] is None:
            self._seed_from_tracker_roles(frame, poses)
            if self.anchors["RED"] is None or self.anchors["BLUE"] is None:
                self._seed_from_colors(detections)

        # compute similarities
        sims = {}  # (bid, role) -> [0..1]
        for bid, box, sig, _ in detections:
            for role in self.roles:
                anchor = self.anchors.get(role)
                sims[(bid, role)] = 0.0 if anchor is None else signature_similarity(sig, anchor)

        # assign roles uniquely by best similarity
        assigned: Dict[str, Optional[int]] = {"RED": None, "BLUE": None}
        used = set()
        # order by which anchor is more stable (more updates)
        for role in sorted(self.roles, key=lambda r: self.anchor_counts[r], reverse=True):
            best_bid, best_sim = None, -1.0
            for bid, _, _, _ in detections:
                if bid in used:
                    continue
                s = sims.get((bid, role), 0.0)
                if s > best_sim:
                    best_bid, best_sim = bid, s
            # accept if similarity is decent
            if best_bid is not None and best_sim >= self.min_sim_accept:
                assigned[role] = best_bid
                used.add(best_bid)

        # stickiness: if none assigned this frame, keep last ID up to tolerance
        for role in self.roles:
            if assigned[role] is None:
                # role temporarily missing?
                self.role_missing_frames[role] += 1
                if self.role_missing_frames[role] <= self.max_missing_frames:
                    # keep previous id if still present in poses; else leave None
                    prev = self.role_to_id.get(role)
                    if prev in poses:
                        assigned[role] = prev
                # else, give up (will wait for reappearance)
            else:
                self.role_missing_frames[role] = 0

        # finalize maps
        self.role_to_id = assigned
        self.id_to_role = {v: k for k, v in assigned.items() if v is not None}

        # update anchors ONLY from assigned roles (avoid drift to others)
        for role, bid in self.role_to_id.items():
            if bid is None:
                continue
            # find sig again for that bid
            for t_bid, box, sig, _ in detections:
                if t_bid == bid and sig is not None and sig.size:
                    self._update_anchor(role, sig)
                    break

    # queries
    def role_for_id(self, boxer_id: int) -> Optional[str]:
        return self.id_to_role.get(boxer_id)

    def id_for_role(self, role: str) -> Optional[int]:
        return self.role_to_id.get(role)

    def anchors_ready(self) -> bool:
        return all(self.anchors[r] is not None for r in self.roles)
