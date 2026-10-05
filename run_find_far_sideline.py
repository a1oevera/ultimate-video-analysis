"""Find a second far-end frame whose FAR SIDELINE is better than t=3625's.

The far end of the bundle rests on one weak input: t=3625's far-sideline stroke is
112px sitting at the horizon (rows 366-376), which is why the (100,37) corner moves
1.4m median under click noise (3.54) and why lengthening the stroke does not help --
what is unpinned is the line's perpendicular offset, not its direction.

The near end got strong by having TWO frames. This looks for a second far-end frame
to do the same, and scores the thing that is actually short rather than the frame
overall: the painted segment of y=37 inside the far endzone, x=82..100m.

Ranked on, in order:
  length    how many image pixels that segment spans -- directly what t=3625 lacks
  rows      how far below the horizon it sits; a stroke at the vanishing point
            cannot pin an offset however long it is
  res       metres of ground per pixel there

Chain drift bounds this to frames near t=3625, where the chain is short and was
measured against the paint at 21cm (3.52). Every candidate still gets clicked.

RESULT: THERE IS NO BETTER FRAME. All three far-end shots in this footage put the
far sideline's endzone segment at 255-338px on rows 365-400, with the horizon at
rows 360-380. It is at the vanishing point in every one of them. Combined with
3.54's measurement that lengthening the stroke from 112px to 1200px barely moves
the answer, the far end is as good as this camera position allows, and more
clicking will not improve it.

Also note the first run of this search was worthless: ranked over t=3300..4050 it
put midfield frames on top, which show the complete centre circle and no endzone at
all. The chain drifts badly that far from its anchor. The numbers above come from
re-anchoring on the t=3625 BUNDLE calibration and hopping only a few chain steps.

Run:  python3 run_find_far_sideline.py
"""
import numpy as np

from frisbee_analysis import UltimatePitchConfiguration

cfg = UltimatePitchConfiguration()
W, H = 1920, 1080
SEG = (8200.0, 10000.0)          # the far endzone, where y=37 is painted
Z = np.load("outputs/pan_extent/chain_field_H.npz")

rows = []
for key in Z.files:
    t = int(key)
    if not (3300 <= t <= 4050):
        continue
    Hf = np.linalg.inv(Z[key])
    ref = Hf @ np.array([cfg.length / 2, cfg.width / 2, 1.0])
    xs = np.linspace(SEG[0], SEG[1], 300)
    P = np.column_stack([xs, np.full(300, cfg.width)])
    Q = Hf @ np.column_stack([P, np.ones(300)]).T
    uv = (Q[:2] / Q[2]).T
    p1 = Hf @ np.column_stack([P + [0.0, 1.0], np.ones(300)]).T
    p1 = (p1[:2] / p1[2]).T
    with np.errstate(invalid="ignore", divide="ignore"):
        mpp = 1.0 / (np.linalg.norm(p1 - uv, axis=1) * 100.0)
    ok = ((np.sign(Q[2]) == np.sign(ref[2])) & np.isfinite(mpp)
          & (uv[:, 0] > 4) & (uv[:, 0] < W - 4) & (uv[:, 1] > 4) & (uv[:, 1] < H - 4))
    if ok.sum() < 20:
        continue
    V = uv[ok]
    length = float(np.hypot(*(V[-1] - V[0])))
    rows.append((t, length, float(np.median(V[:, 1])), float(np.median(mpp[ok])),
                 int(ok.sum())))

rows.sort(key=lambda r: -(r[1] / 1000.0 + r[2] / 1080.0))
print(f"t=3625 for comparison: the clicked stroke is 112px at rows 366-376\n")
print(f"{'t':>6} {'y=37 in endzone':>16} {'median row':>11} {'m/px':>7} {'pts':>5}")
for t, ln, row, mpp, n in rows[:16]:
    print(f"{t:6d} {ln:14.0f}px {row:11.0f} {mpp:7.3f} {n:5d}")
