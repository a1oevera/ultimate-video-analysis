"""Which frames are pointed at which part of the field?

The problem this solves: `calibration_t4200_clicked.json` is only valid out to
x=50.7m, so the far endzone (x=82-100m) has never been calibrated, and scrubbing
a 92-minute video by eye to find frames that show it is slow and unreliable.

How it works. The camera is a Veo fixed-centre virtual pan, so frame-to-frame
homographies compose exactly. Sample the video, chain consecutive samples out
from t=4200, push each frame's footprint through that frame's calibration, and
report the field x-range each sample sees.

WHAT THIS IS NOT. The x-range is the clicked calibration EXTRAPOLATED far past
its validity region and then composed through a long registration chain. Near
the far end line it is not just imprecise, it is structurally wrong: the overlay
draws the x=100 end line nearly PARALLEL to the y=37 sideline, which cannot
happen on a real field -- both have been pushed past the calibration's horizon,
where the projection folds. So:
  * the RANKING is trustworthy (which frames look toward the far end)
  * the near sideline is trustworthy -- MEASURED, see below
  * the far-end x positions are NOT, and every candidate still has to be
    clicked and fitted on its own

MEASURED (t=3625): the chained prediction of the near sideline lands on the
actual blue paint at field y = +21cm, sd 1.6cm over 14 independent image rows.
So the chain itself is sound across the pan; it is the along-field
extrapolation that fails. t=3970 and t=3320 gave sd 456cm and 538cm on the same
test, but from contamination -- blue kit and blue cones enter the paint mask --
not from a worse chain.

Two passes because the file is damaged: a sequential read dies at t=3175s on a
corrupt region (ffmpeg "partial file" at byte 1.25GB of 2.07GB). Seeking reads
straight past it, so the tail is sampled by seek.

Run:  python3 run_pan_extent.py scan      # ~12 min, writes thumbnails
      python3 run_pan_extent.py chain     # ~3 min, writes pan_extent.csv
      python3 run_pan_extent.py overlay 3625
"""
import os
import sys
import json
import cv2
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
VIDEO = f"{PROJ}/videos/torovcanyon.mp4"
WORK = f"{PROJ}/outputs/pan_extent"
THUMBS = f"{WORK}/thumbs"
REF_T = 4200
SCALE = 4.0                     # thumbs are 480x270 out of 1920x1080
MASK = (0, 0, 200, 40)          # scoreboard, in thumb coords
CORRUPT_AT = 3175               # where the sequential read dies

L, W, EZ = 10000.0, 3700.0, 1800.0
SEGS = [((0, 0), (0, W), "end line x=0", (0, 160, 0)),
        ((EZ, 0), (EZ, W), "goal x=18", (0, 255, 0)),
        ((L / 2, 0), (L / 2, W), "halfway x=50", (255, 0, 255)),
        ((L - EZ, 0), (L - EZ, W), "goal x=82", (0, 255, 255)),
        ((L, 0), (L, W), "end line x=100", (0, 128, 255)),
        ((0, 0), (L, 0), "sideline y=0", (255, 120, 0)),
        ((0, W), (L, W), "sideline y=37", (255, 200, 0))]


def scan(step_s=5.0):
    os.makedirs(THUMBS, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(round(step_s * fps))
    idx, n = 0, 0
    while idx < total:                      # pass 1: sequential, until it breaks
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(f"{THUMBS}/{int(round(idx / fps)):05d}.jpg",
                    cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        n += 1
        for _ in range(step - 1):
            if not cap.grab():
                idx = total
                break
        idx += step
    cap.release()
    print(f"sequential pass: {n} thumbs, stopped at t={idx / fps:.0f}s")

    cap = cv2.VideoCapture(VIDEO)            # pass 2: seek past the corruption
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    m = 0
    for t in range(int(idx / fps) + int(step_s), dur, int(step_s)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        cv2.imwrite(f"{THUMBS}/{t:05d}.jpg",
                    cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        m += 1
    cap.release()
    print(f"seek pass: {m} thumbs")


def _orb_chain():
    import glob
    files = sorted(glob.glob(THUMBS + "/*.jpg"))
    times = [int(os.path.basename(f)[:-4]) for f in files]
    ref_i = times.index(min(times, key=lambda t: abs(t - REF_T)))

    orb = cv2.ORB_create(nfeatures=2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    cache = {}

    def feat(i):
        if i not in cache:
            g = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
            m = np.full(g.shape, 255, np.uint8)
            m[MASK[1]:MASK[3], MASK[0]:MASK[2]] = 0
            cache[i] = orb.detectAndCompute(g, m)
        return cache[i]

    def pair(i, j):
        ka, da = feat(i)
        kb, db = feat(j)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, 0
        good = [m for m, n in bf.knnMatch(db, da, k=2)
                if n and m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None, 0
        src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H, (int(mask.sum()) if mask is not None else 0)

    # Bridging gaps matters more than it looks. A neighbour-only chain stops dead at
    # the first production cut or corrupt read, and the first version of this stopped
    # at t=2795 and t=4310 -- 304 of 1112 samples, 25 minutes of 92. Trying several
    # recent anchors walks straight over a short cutaway, because the frame after it
    # usually still overlaps the frame before it.
    chain = {ref_i: (np.eye(3), 999)}
    for step in (1, -1):
        recent = [ref_i]
        i = ref_i
        while 0 <= i + step < len(files):
            j = i + step
            got = None
            for anchor in recent[:6] + [ref_i]:
                H, n = pair(anchor, j)
                if H is not None and n >= 12:
                    got = (chain[anchor][0] @ H, n)
                    break
            if got is None:
                # skip this sample and try to reacquire from the next few
                i = j
                continue
            chain[j] = got
            recent.insert(0, j)
            i = j
    return files, times, chain


def chain():
    files, times, ch = _orb_chain()
    cal = json.load(open(f"{PROJ}/outputs/calibration/calibration_t4200_clicked.json"))
    Hi2f = np.array(cal["homography"])                 # image -> field (cm)
    S = np.diag([1 / SCALE, 1 / SCALE, 1.0])           # full-res -> thumb

    rows, mats = [], {}
    for i, (Ht, n) in sorted(ch.items()):
        M = Hi2f @ (np.linalg.inv(S) @ Ht @ S)         # this frame -> field
        mats[str(times[i])] = M
        us, vs = np.meshgrid(np.linspace(0, 1919, 24), np.linspace(400, 1079, 12))
        Q = M @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
        ok = Q[2] < 0                                  # in front of the camera
        if ok.sum() < 20:
            continue
        X, Y = Q[0, ok] / Q[2, ok], Q[1, ok] / Q[2, ok]
        band = (Y > -1500) & (Y < 5200) & (X > -3000) & (X < 13000)
        if band.sum() < 20:
            continue
        rows.append((times[i], n, np.percentile(X[band], 5) / 100,
                     np.percentile(X[band], 95) / 100))

    os.makedirs(WORK, exist_ok=True)
    np.savez(f"{WORK}/chain_field_H.npz", **mats)
    with open(f"{WORK}/pan_extent.csv", "w") as f:
        f.write("t_sec,inliers,x_lo_m,x_hi_m\n")
        for t, n, lo, hi in rows:
            f.write(f"{t},{n},{lo:.1f},{hi:.1f}\n")
    print(f"chained {len(ch)} of {len(files)} samples "
          f"(t={times[min(ch)]}..{times[max(ch)]}), {len(rows)} usable")
    for t, n, lo, hi in sorted(rows, key=lambda r: -r[3])[:12]:
        print(f"  t={t:5}  inliers={n:4}  sees x = {lo:5.1f} .. {hi:5.1f} m")


def overlay(t):
    Z = np.load(f"{WORK}/chain_field_H.npz")
    Hf = np.linalg.inv(Z[str(t)])                      # field -> image
    path = f"{PROJ}/outputs/diagnostics/frame_{t}.jpg"
    if not os.path.exists(path):
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(int(t) * (cap.get(cv2.CAP_PROP_FPS) or 30)))
        ok, fr = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"could not read t={t}")
        cv2.imwrite(path, fr)
    out = cv2.imread(path)
    for a, b, name, col in SEGS:
        P = np.stack([np.linspace(a[0], b[0], 400),
                      np.linspace(a[1], b[1], 400), np.ones(400)])
        Q = Hf @ P
        uv = (Q[:2] / Q[2]).T[Q[2] < 0]                # in front of the camera
        drawn = [p for p in uv if -4000 < p[0] < 6000 and -4000 < p[1] < 5000]
        for i in range(1, len(drawn)):
            cv2.line(out, tuple(np.int32(drawn[i - 1])),
                     tuple(np.int32(drawn[i])), col, 3)
        vis = [p for p in drawn if 0 <= p[0] < 1920 and 0 <= p[1] < 1080]
        if vis:
            p = vis[len(vis) // 2]
            for c, th in (((0, 0, 0), 5), (col, 2)):
                cv2.putText(out, name, (int(p[0]) + 6, int(p[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, c, th)
    p = f"{PROJ}/outputs/diagnostics/chain_overlay_{t}.jpg"
    cv2.imwrite(p, out)
    print(p)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "chain"
    if cmd == "scan":
        scan()
    elif cmd == "chain":
        chain()
    elif cmd == "overlay":
        overlay(sys.argv[2])
    else:
        sys.exit(__doc__)
