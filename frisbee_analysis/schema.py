"""
The internal data contract for the whole project.

Every source of tracking data -- UFATrack, the synthetic generator, and
eventually your own YOLO+homography pipeline -- must produce a Track in THIS
format. Possession logic consumes only this; it never sees a pixel. That
decoupling is the point: you develop and validate the hard part (possession) on
coordinate data with no footage.

Coordinates are FRACTIONAL field position, not metres:
    x in [0,1] along field length, y in [0,1] across width.
UFATrack uses a UFA pitch (109.73 x 48.77 m); you will film WFDF (100 x 37 m).
Fractional coords + a per-source FieldConfig keeps logic field-agnostic. Convert
metres->fractional at the loader boundary ONLY.

GROUND TRUTH NOTE: in UFATrack the disc position is reconstructed from the
holder annotation by interpolation. So disc x,y is derived, not measured.
TRAIN/EVALUATE AGAINST holder_id, never against disc position.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FieldConfig:
    length_m: float
    width_m: float
    endzone_m: float
    name: str = ""

    @property
    def attacking_endzone_frac(self) -> float:
        return 1.0 - self.endzone_m / self.length_m

    @property
    def defending_endzone_frac(self) -> float:
        return self.endzone_m / self.length_m


UFA_FIELD = FieldConfig(109.73, 48.77, 18.29, "UFA")
WFDF_FIELD = FieldConfig(100.0, 37.0, 18.0, "WFDF")


@dataclass
class Track:
    """One possession sequence.

    positions: (T,N,2) fractional coords. T frames, N players, (x,y).
    team:      (N,) int 0/1. In UFATrack: 0=offense, 1=defense (per-possession).
    holder_id: (T,) int, index of disc-holder each frame, -1 if none. GROUND TRUTH.
    fps:       sample rate (UFATrack tracking = 10fps).
    """
    positions: np.ndarray
    team: np.ndarray
    holder_id: np.ndarray
    fps: float = 10.0
    field: FieldConfig = field(default_factory=lambda: UFA_FIELD)

    def __post_init__(self):
        T, N, _ = self.positions.shape
        assert self.team.shape == (N,), "team must be (N,)"
        assert self.holder_id.shape == (T,), "holder_id must be (T,)"

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_players(self) -> int:
        return self.positions.shape[1]

    def velocities(self) -> np.ndarray:
        dt = 1.0 / self.fps
        v = np.zeros_like(self.positions)
        v[:-1] = (self.positions[1:] - self.positions[:-1]) / dt
        v[-1] = v[-2]
        return v

    def speeds(self) -> np.ndarray:
        return np.linalg.norm(self.velocities(), axis=-1)
