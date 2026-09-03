"""Fit a metric ground plane from SOCCER markings: the centre circle (9.15m radius
under the Laws of the Game, every pitch size) plus the halfway line.

Why this exists: the ultimate field is coned and its lines cluster near the
horizon, which is what caps a line-fitted calibration at ~44m of validity. The
soccer pitch beneath is painted with known geometry.

Two things that matter, both learned the hard way:
  * Filter the circle pixels to those actually ON the fitted ellipse. Taking every
    paint pixel in a box around the circle left 53% contamination (player kit, the
    dark seam lines) and the fit converged to 57cm circle rms with a badly wrong
    plane. Iterating fitEllipse -> keep within 6px drops it to 7.4cm.
  * Optimising is fine but the answer is only as good as the feature LABELS. A long
    line detected above the circle was assumed to be the far touchline; the fit
    places it 12.3m from the pitch centre (bootstrap sd 0.02m), where a touchline
    must be 32-37.5m. It is some other marking. Do not feed it in as a touchline.

Run:  python3 run_soccer_fit.py <frame_jpg>
Outputs a 3x3 image->field homography in metres, centred on the pitch centre,
x along the pitch length, y across it.
"""
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares

CIRCLE_R_M = 9.15

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/diagnostics/frame_1700.jpg"
frame = cv2.imread(path)
if frame is None:
    sys.exit(f"cannot read {path}")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)
grass = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 60)).astype(np.uint8)
paint = (white & (cv2.dilate(grass, np.ones((25, 25), np.uint8)) > 0)).astype(np.uint8)


def box(x0, x1, y0, y1, drop=None):
    m = np.zeros_like(paint)
    m[y0:y1, x0:x1] = paint[y0:y1, x0:x1]
    if drop:
        m[:, drop[0]:drop[1]] = 0
    ys, xs = np.nonzero(m)
    return np.stack([xs, ys], 1).astype(float)


def trim_to_line(P, keep=85):
    for _ in range(2):
        c = P.mean(0)
        _, _, V = np.linalg.svd(P - c)
        d = V[0]
        n = np.array([-d[1], d[0]])
        P = P[np.abs((P - c) @ n) < np.percentile(np.abs((P - c) @ n), keep)]
    c = P.mean(0)
    _, _, V = np.linalg.svd(P - c)
    return P, V[0]


def on_ellipse(P, tol=6.0, iters=4):
    """Keep only pixels lying on the fitted ellipse -- see module docstring."""
    for _ in range(iters):
        (ex, ey), (d1, d2), a = cv2.fitEllipse(P.astype(np.float32))
        th = np.radians(a)
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = (P - [ex, ey]) @ R.T
        dev = np.abs(np.hypot(q[:, 0] / (d1 / 2), q[:, 1] / (d2 / 2)) - 1) * ((d1 + d2) / 4)
        P = P[dev < tol]
    return P


def to_field(H, P):
    Q = H @ np.column_stack([P, np.ones(len(P))]).T
    return (Q[:2] / Q[2]).T


def fit(circle_px, half_px, half_dir, lengthwise_px=None):
    """circle + halfway is only 7 constraints (conic 5 + line 2) for 8 DOF, so it
    does NOT determine the plane: with and without an 8th constraint the same
    residual is reached by planes that put a test line at 21.12m and 12.33m from
    the centre. Pass `lengthwise_px` -- pixels of any long marking running along
    the pitch -- to supply the 8th via its DIRECTION (it must map to constant y).
    That assumes only that the marking is parallel to the touchline, not that it
    IS the touchline at a known distance."""
    (ex, ey), (d1, d2), a = cv2.fitEllipse(circle_px.astype(np.float32))
    th = np.radians(a)
    u1 = np.array([np.cos(th), np.sin(th)])
    A0 = np.diag([CIRCLE_R_M / (d1 / 2), CIRCLE_R_M / (d2 / 2)]) @ np.stack([u1, [-u1[1], u1[0]]])
    v = A0 @ half_dir
    v /= np.linalg.norm(v)
    Aff = np.array([[v[1], -v[0]], [v[0], v[1]]]) @ A0
    H0 = np.eye(3)
    H0[:2, :2] = Aff
    H0[:2, 2] = -Aff @ np.array([ex, ey])

    def resid(p):
        H = np.append(p, 1.0).reshape(3, 3)
        fc, fh = to_field(H, circle_px), to_field(H, half_px)
        out = [np.hypot(*fc.T) - CIRCLE_R_M, fh[:, 0]]
        if lengthwise_px is not None:
            ft = to_field(H, lengthwise_px)
            out.append(ft[:, 1] - ft[:, 1].mean())
        return np.concatenate(out)

    s = least_squares(resid, H0.ravel()[:8], method="trf", x_scale="jac", max_nfev=40000)
    return np.append(s.x, 1.0).reshape(3, 3)


if __name__ == "__main__":
    circ = on_ellipse(box(80, 1320, 430, 830, drop=(670, 790)))
    half, dh = trim_to_line(box(640, 820, 800, 1080))
    lengthwise, _ = trim_to_line(box(0, 1900, 340, 410))
    H = fit(circ, half, dh, lengthwise)
    r = np.hypot(*to_field(H, circ).T)
    print(f"circle pixels on ellipse: {len(circ)}")
    print(f"  radius {r.mean():.3f} m (target {CIRCLE_R_M})  rms {np.std(r-CIRCLE_R_M)*100:.1f} cm")
    print(f"  halfway |x| mean {np.abs(to_field(H, half)[:,0]).mean()*100:.1f} cm")
    print(f"  the lengthwise marking lands {abs(to_field(H, lengthwise)[:,1].mean()):.2f} m "
          f"from the pitch centre")
    print("\nNOT VERIFIED. The circle and halfway are both USED, so their residuals are"
          "\nnot independent checks, and the 8th constraint is only a direction. A"
          "\ntouchline sits 32-37.5m from centre; nothing here reaches that, so no"
          "\nlandmark in this frame confirms the scale away from the circle.")
    np.save("outputs/diagnostics/H_soccer.npy", H)
