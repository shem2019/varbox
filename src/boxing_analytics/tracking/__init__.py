"""Tracking primitives for fighter identity and referee separation."""

from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)
from boxing_analytics.tracking.identity_manager import IdentityManager
from boxing_analytics.tracking.reid import build_reid_embedder
from boxing_analytics.tracking.sam2_identity import Sam2FighterIdentityTrack
from boxing_analytics.tracking.tracklet_stitcher import (
    TrackletStitcherConfig,
    TwoFighterTrackletStitcher,
)

__all__ = [
    "IdentityHMMConfig",
    "IdentityManager",
    "Sam2FighterIdentityTrack",
    "build_reid_embedder",
    "TrackletStitcherConfig",
    "TrackObservation",
    "TwoFighterIdentityHMM",
    "TwoFighterTrackletStitcher",
]
