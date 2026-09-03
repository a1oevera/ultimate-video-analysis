"""Does a calibration put its ground lines on the actual playing surface?

A line on the ground plane MUST project onto turf. If it lands in the trees or
the sky the calibration is wrong there, no matter how well it scores near the
middle of the field. This catches far-field failure that the centre-circle radius
test cannot: the circle only spans 9.15m around the pitch centre, so it certifies
the plane locally and says nothing about where 34m ends up.

Finding the turf edge is harder than it looks, and three attempts failed first:
  * plain green hue -- trees are green too, so the "surface" started in the sky
  * largest green component touching the bottom -- morphological closing bridges
    turf to trees through the background, giving 89% of the frame
  * per-column walk (either direction) -- stops at painted lines, players, and
    shadows, reporting the edge mid-field
What works is a ROW PROFILE: the fraction of each row that is green AND smooth
(low local texture separates turf from foliage). The surface shows as a clean
step. Report the transition band rather than one row, because the painted line
and the row of standing people sit right at the boundary and depress it.

Run:  python3 run_ground_check.py <frame.jpg>
"""
import sys
import numpy as np
import cv2


def turf_profile(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu = cv2.blur(gray, (15, 15))
    tex = np.sqrt(np.maximum(cv2.blur(gray * gray, (15, 15)) - mu * mu, 0))
    turf = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 60)) & (tex < 18)
    return turf.mean(axis=1)


def surface_band(prof, lo=0.35, hi=0.70):
    """Rows where turf coverage first rises through lo and then holds above hi."""
    solid = np.nonzero(prof > hi)[0]
    if not len(solid):
        return None, None
    bot = solid.min()
    above = np.nonzero(prof[:bot] < lo)[0]
    return (above.max() if len(above) else 0), bot


def check(frame_bgr, project_row_range, names):
    prof = turf_profile(frame_bgr)
    top, bot = surface_band(prof)
    print(f"playing surface begins between rows {top} and {bot} "
          f"(band = the painted line and standing row sit here)")
    for nm, (r0, r1) in zip(names, project_row_range):
        if r0 is None:
            print(f"  {nm:24s} outside the frame")
            continue
        verdict = ("ON the surface" if r0 >= bot else
                   "IN THE BAND -- inconclusive" if r1 >= top else
                   "ABOVE THE SURFACE -- impossible for a ground line")
        print(f"  {nm:24s} rows {r0:4.0f}..{r1:4.0f}   {verdict}")


if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1])
    prof = turf_profile(frame)
    top, bot = surface_band(prof)
    print(f"{sys.argv[1]}: surface transition rows {top}..{bot}")
    for y in range(0, frame.shape[0], 40):
        print(f"  row {y:4d}: {prof[y]*100:5.1f}%  {'#'*int(prof[y]*50)}")
