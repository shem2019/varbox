"""Tracking primitives for fighter identity and referee separation."""

from boxing_analytics.tracking.identity_hmm import (
    IdentityHMMConfig,
    TrackObservation,
    TwoFighterIdentityHMM,
)
from boxing_analytics.tracking.identity_manager import IdentityManager
from boxing_analytics.tracking.tracklet_stitcher import (
    TrackletStitcherConfig,
    TwoFighterTrackletStitcher,
)

__all__ = [
    "IdentityHMMConfig",
    "IdentityManager",
    "TrackletStitcherConfig",
    "TrackObservation",
    "TwoFighterIdentityHMM",
    "TwoFighterTrackletStitcher",
]
