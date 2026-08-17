"""
Loader: open-starlab/UFATrack  ->  Track

Verified against the REAL dataset (open-starlab/UFATrack, CC-BY-4.0).

Dataset facts:
  - 20 CSVs, 2_1.csv .. 2_20.csv, ONE FILE PER POSSESSION (2nd quarter,
    Oakland Spiders vs Salt Lake Shred, 2025-06-27). ~9.5 min total.
  - Row = one object at one frame. 14 players (id 1..14) + disc (id 15).
  - Columns: frame,id,x,y,vx,vy,ax,ay,v_mag,a_mag,v_angle,a_angle,
    diff_v_a_angle,diff_v_angle,diff_a_angle,class,holder,closest,
    selected,prev_holder,def_selected
  - class = 'offense'/'defense'/'disc'. offense->team 0, defense->team 1.
  - holder = 'True'/'False'; True on the disc-holder that frame. GROUND TRUTH.
  - x,y metres on 109.73 x 48.77 m. 10fps tracking (video 30fps).

The disc row x,y is the derived/interpolated position -- ignored here.

The dataset ships extra per-row features you may want later:
  closest      : id of nearest opponent (marking relationship)
  def_selected : defensive-selection helper
  prev_holder  : previous holder id
These are NOT loaded into Track (which is deliberately minimal). If your
possession model wants marking features, extend the loader to carry them --
see load_raw_rows() which returns everything.
"""
from __future__ import annotations
import csv
import os
import numpy as np

from .schema import Track, UFA_FIELD, FieldConfig

COL_FRAME, COL_ID, COL_X, COL_Y = "frame", "id", "x", "y"
COL_CLASS, COL_HOLDER = "class", "holder"
COL_CLOSEST = "closest"
CLASS_DISC, CLASS_OFFENSE, CLASS_DEFENSE = "disc", "offense", "defense"
FPS = 10.0


def _build_track(rows, field):
    frames = sorted({int(r[COL_FRAME]) for r in rows})
    frame_index = {f: i for i, f in enumerate(frames)}
    T = len(frames)
    player_ids = sorted({int(r[COL_ID]) for r in rows if r[COL_CLASS] != CLASS_DISC})
    id_index = {oid: j for j, oid in enumerate(player_ids)}
    N = len(player_ids)

    positions = np.full((T, N, 2), np.nan)
    team = np.full(N, -1, dtype=int)
    holder_id = np.full(T, -1, dtype=int)

    for r in rows:
        cls = r[COL_CLASS]
        if cls == CLASS_DISC:
            continue
        f = frame_index[int(r[COL_FRAME])]
        j = id_index[int(r[COL_ID])]
        positions[f, j, 0] = float(r[COL_X])
        positions[f, j, 1] = float(r[COL_Y])
        if team[j] < 0:
            team[j] = 0 if cls == CLASS_OFFENSE else 1
        if str(r[COL_HOLDER]).strip().lower() == "true":
            holder_id[f] = j

    positions = _interp(positions)
    positions[..., 0] /= field.length_m
    positions[..., 1] /= field.width_m
    positions = np.clip(positions, 0.0, 1.0)
    return Track(positions=positions.astype(np.float32), team=team,
                 holder_id=holder_id, fps=FPS, field=field)


def _interp(positions):
    T, N, D = positions.shape
    out = positions.copy()
    for j in range(N):
        for d in range(D):
            col = out[:, j, d]
            nans = np.isnan(col)
            if nans.all():
                col[:] = 0.5
            elif nans.any():
                idx = np.arange(T)
                col[nans] = np.interp(idx[nans], idx[~nans], col[~nans])
    return out


def load_ufatrack(dir_path, field: FieldConfig = UFA_FIELD):
    """Load all UFATrack possession CSVs. One Track per file, ordered 2_1..2_20."""
    files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]

    def pnum(fn):
        try:
            return int(fn.replace(".csv", "").split("_")[1])
        except (IndexError, ValueError):
            return 0

    files.sort(key=pnum)
    tracks = []
    for fn in files:
        with open(os.path.join(dir_path, fn), newline="") as f:
            rows = list(csv.DictReader(f))
        tracks.append(_build_track(rows, field))
    return tracks


def load_raw_rows(dir_path):
    """Return {possession_num: list[dict rows]} with ALL columns intact, for
    when you want the marking/closest features the Track schema drops."""
    out = {}
    for fn in os.listdir(dir_path):
        if not fn.endswith(".csv"):
            continue
        try:
            p = int(fn.replace(".csv", "").split("_")[1])
        except (IndexError, ValueError):
            continue
        with open(os.path.join(dir_path, fn), newline="") as f:
            out[p] = list(csv.DictReader(f))
    return out
