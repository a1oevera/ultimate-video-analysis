"""
Pivot-rule viability test on real UFATrack ground truth.

Answers the prior question that decides the project: is the disc-holder the
slowest/most-stationary offense player? Ranks offense players by speed each
frame and reports where the true holder falls.

RESULT ON REAL DATA (already run -- see results/pivot_finding.md):
  holder is slowest 40% / top-2 55% / top-3 67%. Moderate signal: speed is a
  strong filter, weak pinpoint. -> multi-feature + HMM needed (run_viability.py).

Run:  python run_pivot_test.py [path-to-ufatrack_data]
"""
import sys
import numpy as np
from frisbee_analysis import load_ufatrack

path = sys.argv[1] if len(sys.argv) > 1 else "ufatrack_data"
tracks = load_ufatrack(path)

ranks = []
rels = []
for t in tracks:
    sm = t.speeds()
    k = 5; pad = k // 2
    for j in range(t.n_players):
        c = np.pad(sm[:, j], pad, mode="edge")
        sm[:, j] = np.convolve(c, np.ones(k) / k, mode="valid")
    offense = t.team == 0
    off_idx = np.where(offense)[0]
    for f in range(t.n_frames):
        h = t.holder_id[f]
        if h < 0 or not offense[h]:
            continue
        os_ = sm[f, off_idx]
        order = off_idx[np.argsort(os_)]
        ranks.append(int(np.where(order == h)[0][0]))
        med = np.median(os_)
        rels.append(sm[f, h] / med if med > 1e-9 else np.nan)

r = np.array(ranks)
print(f"Frames analysed: {len(r)}   (20 possessions, ~9.5 min)\n")
print("Where does the true holder rank by speed? (0 = slowest offense player)")
for i in range(7):
    pct = 100 * np.mean(r == i)
    print(f"  rank {i}: {pct:5.1f}%  {'#' * int(pct/2)}")
print(f"\nslowest:      {100*np.mean(r==0):5.1f}%")
print(f"in slowest 2: {100*np.mean(r<=1):5.1f}%")
print(f"in slowest 3: {100*np.mean(r<=2):5.1f}%")
print(f"holder speed / offense-median: {np.nanmedian(rels):.2f}")
print("\nModerate signal. Speed filters well, pinpoints poorly. See NEXT_STEPS.md.")
