import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from color_signature import compute_color_scores, compute_hist_signature, signature_similarity

ROLE_COLOR = {
    "BLUE": (255, 80, 60),
    "RED": (0, 0, 255),
}


class ParticipantManager:
    """
    Role-locked tracking for exactly two fighters: RED and BLUE.

    Features:
    - Fused assignment score: appearance + motion + corner-color prior.
    - Long-lived identity memory for re-ID after long obstruction.
    - Fingerprint profiles that strengthen over time.
    - Optional manual fingerprint seeding from user-annotated boxes.
    """

    def __init__(
        self,
        min_color_fraction: float = 0.03,
        min_sim_accept: float = 0.30,
        smooth_alpha: float = 0.15,
        max_missing_frames: int = 45,
        freeze_anchors_after_seed: bool = True,
        lock_after_stable_frames: int = 24,
        unlock_after_missing_frames: int = 900,  # ~30s @30fps
        reid_min_score: float = 0.44,
        fingerprint_build_frames: int = 1800,  # ~60s @30fps
    ):
        self.roles = ("RED", "BLUE")
        self.anchors: Dict[str, Optional[np.ndarray]] = {"RED": None, "BLUE": None}
        self.anchor_counts: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.smooth_alpha = smooth_alpha
        self.min_color_fraction = min_color_fraction
        self.min_sim_accept = min_sim_accept
        self.freeze_anchors_after_seed = freeze_anchors_after_seed
        self.anchors_frozen = False

        self.id_to_role: Dict[int, str] = {}
        self.role_to_id: Dict[str, Optional[int]] = {"RED": None, "BLUE": None}

        self.role_missing_frames: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.max_missing_frames = max_missing_frames
        self.unlock_after_missing_frames = max(max_missing_frames, unlock_after_missing_frames)
        self.reid_min_score = reid_min_score
        self._frame_idx = 0

        self.lock_after_stable_frames = max(1, int(lock_after_stable_frames))
        self.locked_role_to_id: Dict[str, Optional[int]] = {"RED": None, "BLUE": None}
        self.role_stable_counts: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.reacquire_counts: Dict[str, int] = {"RED": 0, "BLUE": 0}

        self.role_memory = {
            "RED": {"sig": None, "center": None, "vel": (0.0, 0.0), "diag": None, "last_seen": 0},
            "BLUE": {"sig": None, "center": None, "vel": (0.0, 0.0), "diag": None, "last_seen": 0},
        }

        self.fingerprint_build_frames = max(60, int(fingerprint_build_frames))
        self.fingerprint_sig: Dict[str, Optional[np.ndarray]] = {"RED": None, "BLUE": None}
        self.fingerprint_frames: Dict[str, int] = {"RED": 0, "BLUE": 0}
        self.fingerprint_ready: Dict[str, bool] = {"RED": False, "BLUE": False}
        self.manual_seeded: Dict[str, bool] = {"RED": False, "BLUE": False}

    @staticmethod
    def _crop(frame, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _center(box_xyxy) -> Tuple[float, float]:
        x1, y1, x2, y2 = box_xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _diag(box_xyxy) -> float:
        x1, y1, x2, y2 = box_xyxy
        return math.hypot(max(1, x2 - x1), max(1, y2 - y1))

    def _update_anchor(self, role: str, sig: np.ndarray):
        if sig is None or sig.size == 0 or self.anchors_frozen:
            return
        prev = self.anchors.get(role)
        if prev is None or prev.shape != sig.shape:
            self.anchors[role] = sig
        else:
            a = self.smooth_alpha
            self.anchors[role] = (1 - a) * prev + a * sig
        self.anchor_counts[role] += 1

    def _update_role_memory(self, role: str, sig: np.ndarray, box, frame_idx: int):
        mem = self.role_memory[role]
        center = self._center(box)
        diag = self._diag(box)
        prev_center = mem["center"]
        dt = max(1, frame_idx - int(mem["last_seen"] or frame_idx))

        if prev_center is not None:
            vx = (center[0] - prev_center[0]) / dt
            vy = (center[1] - prev_center[1]) / dt
            pvx, pvy = mem["vel"]
            mem["vel"] = (0.78 * pvx + 0.22 * vx, 0.78 * pvy + 0.22 * vy)
        else:
            mem["vel"] = (0.0, 0.0)

        mem["center"] = center
        mem["diag"] = diag
        mem["last_seen"] = frame_idx

        prev_sig = mem["sig"]
        if prev_sig is None or prev_sig.shape != sig.shape:
            mem["sig"] = sig
        else:
            a = min(0.18, self.smooth_alpha)
            mem["sig"] = (1 - a) * prev_sig + a * sig

    def _update_fingerprint(self, role: str, sig: np.ndarray):
        if sig is None or sig.size == 0:
            return

        # Build RED profile first, then BLUE profile.
        if role == "BLUE" and not (self.fingerprint_ready["RED"] or self.manual_seeded["RED"]):
            return

        prev = self.fingerprint_sig.get(role)
        if prev is None or prev.shape != sig.shape:
            self.fingerprint_sig[role] = sig
        else:
            alpha = 0.012 if self.fingerprint_ready[role] else 0.04
            self.fingerprint_sig[role] = (1 - alpha) * prev + alpha * sig

        if not self.fingerprint_ready[role]:
            self.fingerprint_frames[role] += 1
            if self.fingerprint_frames[role] >= self.fingerprint_build_frames:
                self.fingerprint_ready[role] = True

    def _seed_from_colors(self, detections):
        if self.anchors["RED"] is None:
            best_sig, best_val = None, -1.0
            for d in detections:
                r = d["colors"][0]
                if r > best_val and r >= self.min_color_fraction:
                    best_sig, best_val = d["sig"], r
            if best_sig is not None:
                self._update_anchor("RED", best_sig)

        if self.anchors["BLUE"] is None:
            best_sig, best_val = None, -1.0
            for d in detections:
                b = d["colors"][1]
                if b > best_val and b >= self.min_color_fraction:
                    best_sig, best_val = d["sig"], b
            if best_sig is not None:
                self._update_anchor("BLUE", best_sig)

        if self.freeze_anchors_after_seed and all(self.anchors[r] is not None for r in self.roles):
            self.anchors_frozen = True

    def _seed_from_tracker_roles(self, frame, poses):
        for _, data in poses.items():
            role = data.get("role")
            if role in self.roles and self.anchors[role] is None:
                sig = compute_hist_signature(self._crop(frame, data["box"]))
                self._update_anchor(role, sig)
        if self.freeze_anchors_after_seed and all(self.anchors[r] is not None for r in self.roles):
            self.anchors_frozen = True

    def _appearance_score(self, role: str, sig: np.ndarray) -> float:
        mem_sig = self.role_memory[role]["sig"]
        anchor_sig = self.anchors.get(role)
        fp_sig = self.fingerprint_sig.get(role)

        s_mem = signature_similarity(sig, mem_sig) if mem_sig is not None else 0.0
        s_anchor = signature_similarity(sig, anchor_sig) if anchor_sig is not None else 0.0
        s_fp = signature_similarity(sig, fp_sig) if fp_sig is not None else 0.0

        if self.fingerprint_ready[role] or self.manual_seeded[role]:
            return float(max(0.0, min(1.0, 0.62 * s_fp + 0.25 * s_mem + 0.13 * s_anchor)))
        return float(max(0.0, min(1.0, 0.22 * s_fp + 0.52 * s_mem + 0.26 * s_anchor)))

    def _motion_score(self, role: str, center: Tuple[float, float], diag: float, frame_idx: int) -> float:
        mem = self.role_memory[role]
        if mem["center"] is None:
            return 0.45
        dt = max(1, frame_idx - int(mem["last_seen"] or frame_idx))
        vx, vy = mem["vel"]
        pred = (mem["center"][0] + vx * dt, mem["center"][1] + vy * dt)
        dist = math.hypot(center[0] - pred[0], center[1] - pred[1])
        scale = max(35.0, 1.35 * max(diag, mem["diag"] or diag))
        return float(max(0.0, min(1.0, math.exp(-((dist / scale) ** 2)))))

    @staticmethod
    def _color_prior(role: str, colors) -> float:
        r, b, _ = colors
        raw = r if role == "RED" else b
        return float(max(0.0, min(1.0, raw / 0.25)))

    def _role_score(self, role: str, det, frame_idx: int) -> float:
        app = self._appearance_score(role, det["sig"])
        mot = self._motion_score(role, det["center"], det["diag"], frame_idx)
        clr = self._color_prior(role, det["colors"])
        if role == "BLUE" and not (self.fingerprint_ready["RED"] or self.manual_seeded["RED"]):
            return 0.50 * app + 0.25 * mot + 0.25 * clr
        return 0.60 * app + 0.28 * mot + 0.12 * clr

    def _best_unique_assignment(self, detections: List[Dict], frame_idx: int):
        scores = {
            (d["bid"], role): self._role_score(role, d, frame_idx)
            for d in detections
            for role in self.roles
        }
        assigned = {"RED": None, "BLUE": None}
        if not detections:
            return assigned, scores

        role_order = sorted(
            self.roles,
            key=lambda role: max(scores[(d["bid"], role)] for d in detections),
            reverse=True,
        )
        used = set()
        for role in role_order:
            ranked = sorted(
                ((d["bid"], scores[(d["bid"], role)]) for d in detections if d["bid"] not in used),
                key=lambda t: t[1],
                reverse=True,
            )
            if not ranked:
                continue
            best_bid, best_score = ranked[0]
            loss = self.role_missing_frames.get(role, 0)
            accept = self.min_sim_accept if loss <= self.max_missing_frames else max(0.22, self.min_sim_accept * 0.75)
            if self.fingerprint_ready[role] or self.manual_seeded[role]:
                accept = max(accept, 0.36)
            if best_score >= accept:
                assigned[role] = best_bid
                used.add(best_bid)
        return assigned, scores

    def _try_reid(self, role: str, detections: List[Dict], used: set, frame_idx: int):
        best_bid, best_score = None, -1.0
        for d in detections:
            bid = d["bid"]
            if bid in used:
                continue
            score = self._role_score(role, d, frame_idx)
            if score > best_score:
                best_bid, best_score = bid, score
        if best_bid is not None and best_score >= self.reid_min_score:
            return best_bid, best_score
        return None, None

    def set_manual_fingerprint(self, role: str, frame, box_xyxy, frame_idx: int = 0) -> bool:
        role = str(role).upper()
        if role not in self.roles:
            return False
        crop = self._crop(frame, box_xyxy)
        sig = compute_hist_signature(crop)
        if sig is None or sig.size == 0:
            return False

        self.manual_seeded[role] = True
        self.fingerprint_sig[role] = sig
        self.fingerprint_frames[role] = self.fingerprint_build_frames
        self.fingerprint_ready[role] = True

        self._update_anchor(role, sig)
        self._update_role_memory(role, sig, box_xyxy, max(1, int(frame_idx or 1)))
        self.role_stable_counts[role] = self.lock_after_stable_frames
        return True

    def update(self, frame, poses: Dict[int, dict], frame_idx: Optional[int] = None):
        self._frame_idx = self._frame_idx + 1 if frame_idx is None else int(frame_idx)

        detections = []
        for bid, data in poses.items():
            box = data["box"]
            sig = compute_hist_signature(self._crop(frame, box))
            if sig is None or sig.size == 0:
                continue
            r, b, w = compute_color_scores(self._crop(frame, box))
            detections.append(
                {
                    "bid": bid,
                    "box": box,
                    "sig": sig,
                    "colors": (r, b, w),
                    "center": self._center(box),
                    "diag": self._diag(box),
                }
            )

        if self.anchors["RED"] is None or self.anchors["BLUE"] is None:
            self._seed_from_tracker_roles(frame, poses)
            if self.anchors["RED"] is None or self.anchors["BLUE"] is None:
                self._seed_from_colors(detections)

        assigned, score_matrix = self._best_unique_assignment(detections, self._frame_idx)

        for role in self.roles:
            prev_id = self.role_to_id.get(role)
            if prev_id in poses:
                prev_score = score_matrix.get((prev_id, role), 0.0)
                new_id = assigned.get(role)
                new_score = score_matrix.get((new_id, role), -1.0) if new_id is not None else -1.0
                if prev_score >= max(0.20, new_score - 0.08):
                    assigned[role] = prev_id

        used = {bid for bid in assigned.values() if bid is not None}

        for role in self.roles:
            chosen = assigned.get(role)
            prev = self.role_to_id.get(role)
            if chosen is not None and chosen == prev:
                self.role_stable_counts[role] += 1
            elif chosen is not None:
                self.role_stable_counts[role] = 1
            else:
                self.role_stable_counts[role] = 0

            if self.role_stable_counts[role] >= self.lock_after_stable_frames:
                self.locked_role_to_id[role] = chosen

        for role in self.roles:
            locked_id = self.locked_role_to_id.get(role)
            if locked_id in poses:
                assigned[role] = locked_id
                self.role_missing_frames[role] = 0
                used.add(locked_id)
                continue

            if locked_id is not None:
                reid_bid, _ = self._try_reid(role, detections, used, self._frame_idx)
                if reid_bid is not None:
                    assigned[role] = reid_bid
                    self.locked_role_to_id[role] = reid_bid
                    self.role_missing_frames[role] = 0
                    self.reacquire_counts[role] += 1
                    used.add(reid_bid)
                else:
                    self.role_missing_frames[role] += 1
                    # keep lock indefinitely once a strong fingerprint exists
                    if (
                        self.role_missing_frames[role] > self.unlock_after_missing_frames
                        and not (self.fingerprint_ready[role] or self.manual_seeded[role])
                    ):
                        self.locked_role_to_id[role] = None
                        assigned[role] = None

        for role in self.roles:
            if assigned[role] is None:
                self.role_missing_frames[role] += 1
                prev = self.role_to_id.get(role)
                if self.role_missing_frames[role] <= self.max_missing_frames and prev in poses:
                    assigned[role] = prev
            else:
                self.role_missing_frames[role] = 0

        if assigned["RED"] is not None and assigned["RED"] == assigned["BLUE"]:
            red_score = score_matrix.get((assigned["RED"], "RED"), 0.0)
            blue_score = score_matrix.get((assigned["BLUE"], "BLUE"), 0.0)
            if red_score >= blue_score:
                assigned["BLUE"] = None
            else:
                assigned["RED"] = None

        self.role_to_id = assigned
        self.id_to_role = {v: k for k, v in assigned.items() if v is not None}

        det_by_id = {d["bid"]: d for d in detections}
        for role, bid in self.role_to_id.items():
            if bid is None:
                continue
            det = det_by_id.get(bid)
            if det is None:
                continue
            self._update_role_memory(role, det["sig"], det["box"], self._frame_idx)
            self._update_anchor(role, det["sig"])
            self._update_fingerprint(role, det["sig"])

    def role_for_id(self, boxer_id: int) -> Optional[str]:
        return self.id_to_role.get(boxer_id)

    def id_for_role(self, role: str) -> Optional[int]:
        return self.role_to_id.get(role)

    def anchors_ready(self) -> bool:
        return all(self.anchors[r] is not None for r in self.roles)

    def lock_status(self) -> Dict[str, Optional[int]]:
        return dict(self.locked_role_to_id)

    def all_roles_locked(self) -> bool:
        return all(self.locked_role_to_id[r] is not None for r in self.roles)

    def fingerprint_status(self) -> Dict[str, Dict[str, int]]:
        return {
            role: {
                "frames": int(self.fingerprint_frames.get(role, 0)),
                "ready": int(bool(self.fingerprint_ready.get(role, False))),
                "manual_seeded": int(bool(self.manual_seeded.get(role, False))),
            }
            for role in self.roles
        }

    def tracking_stats(self) -> Dict[str, Dict[str, int]]:
        return {
            role: {
                "missing_frames": int(self.role_missing_frames.get(role, 0)),
                "reacquired": int(self.reacquire_counts.get(role, 0)),
                "fingerprint_frames": int(self.fingerprint_frames.get(role, 0)),
                "fingerprint_ready": int(bool(self.fingerprint_ready.get(role, False))),
            }
            for role in self.roles
        }
