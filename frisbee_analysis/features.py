"""
Kinematic features for possession detection.

MEASURED FINDING (real UFATrack, see results/pivot_finding.md): the holder is
the single slowest offense player only ~40% of the time, but is in the slowest 3
~67% of the time. So speed is a strong FILTER, weak PINPOINT. The multi-feature
approach below exists because of that finding -- speed alone is not enough.

Features (per frame, per player), all designed so the holder is an outlier
among OFFENSE players:
  rel_speed        : player speed / offense-median speed. Holder < 1. Workhorse,
                     scale-free. (40% pinpoint, 67% top-3 on real data.)
  accel            : |velocity change|. A planted pivot has low accel.
  dist_to_marker   : distance to nearest OPPONENT (offense->nearest defender,
                     defense->nearest offense player -- symmetric, so it's
                     defined for both teams; needed once the possession model
                     can hold a defense player too, e.g. right after a
                     turnover). The holder is usually being actively marked,
                     so this is SMALL for the holder. UFATrack ships a
                     'closest' column you can use as truth while developing.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .schema import Track


@dataclass
class FeatureConfig:
    smooth_window: int = 5


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        c = np.pad(x[:, j], pad, mode="edge")
        out[:, j] = np.convolve(c, np.ones(window) / window, mode="valid")
    return out


def compute_features(track: Track, cfg: FeatureConfig = FeatureConfig()) -> dict:
    speed = _smooth(track.speeds(), cfg.smooth_window)
    T, N = speed.shape
    team = track.team

    # relative speed vs own-team median
    rel = np.ones_like(speed)
    for t in range(T):
        for tm in (0, 1):
            m = team == tm
            med = np.median(speed[t, m])
            rel[t, m] = speed[t, m] / (med if med > 1e-9 else 1e-9)

    # acceleration magnitude
    vel = track.velocities()
    acc = np.zeros_like(speed)
    acc[1:] = np.linalg.norm(vel[1:] - vel[:-1], axis=-1)
    acc = _smooth(acc, cfg.smooth_window)

    # distance to nearest OPPONENT, per player (symmetric: offense vs defense)
    dist_marker = np.full((T, N), np.nan)
    off_idx = np.where(team == 0)[0]
    def_idx = np.where(team == 1)[0]
    pos = track.positions
    for t in range(T):
        for j in off_idx:
            if len(def_idx) == 0:
                continue
            d = np.linalg.norm(pos[t, def_idx] - pos[t, j], axis=1)
            dist_marker[t, j] = d.min()
        for j in def_idx:
            if len(off_idx) == 0:
                continue
            d = np.linalg.norm(pos[t, off_idx] - pos[t, j], axis=1)
            dist_marker[t, j] = d.min()

    return {"rel_speed": rel, "accel": acc, "dist_to_marker": dist_marker,
            "speed": speed}
