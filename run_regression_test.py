"""
Regression test: pin the answers this project has already paid for.

WHY THIS EXISTS. Over the course of building frisbee_analysis/field_lines.py
and field_model.py, THREE separate geometry gates were written that rejected
the one labelling a human had confirmed correct -- each by judging field
coordinates past the calibration's horizon, each discovered only because
someone happened to look. A ranking term was added that actively rewarded
wrong answers. A "check" was shipped that returned False for every calibration
including the verified-good one. None of that was caught by anything except a
person noticing.

Every assertion below is a fact that was expensive to establish and easy to
break. The fixture holds the human-confirmed labelling in segment-reference
coordinates, so this runs in seconds and needs no video.

Run:  python3 run_regression_test.py
"""
import sys
import json
import numpy as np

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import (Calibration, fit_calibration_with_lines,
                                          check_line_reprojection_error, vertices_within_bounds,
                                          field_covers_points, field_validity_region,
                                          calibration_covers_evidence, is_in_field, is_measurable)
from frisbee_analysis.field_lines import (projected_field_is_sane, no_parallel_crossings,
                                          confusable_groups)
from frisbee_analysis.field_model import (visible_extent, extent_is_plausible,
                                           extent_supports_observation, build_field_model)
from frisbee_analysis.watchers import WatcherRow, watchers_outside_field

FIXTURE = "tests/fixtures/torovcanyon_3945.json"
cfg = UltimatePitchConfiguration()
failures = []


def check(name, condition, detail=""):
    ok = bool(condition)
    print(f"  {'PASS' if ok else 'FAIL':4s}  {name}" + (f"   [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


class _Line:
    """Minimal stand-in with the .points/.p1/.p2/.homogeneous() interface the
    geometry helpers expect, so the fixture doesn't have to carry TrackedLines."""
    def __init__(self, points, colour="white", support=1000, n_frames=60):
        self.points = [tuple(map(float, p)) for p in points]
        arr = np.asarray(self.points)
        self.p1, self.p2 = tuple(arr[0]), tuple(arr[-1])
        self.colour, self.total_support_px, self.n_frames = colour, support, n_frames
        self.rms_fit_px = 0.0

    def homogeneous(self):
        l = np.cross([self.p1[0], self.p1[1], 1.0], [self.p2[0], self.p2[1], 1.0])
        return l / np.hypot(l[0], l[1])


with open(FIXTURE) as f:
    fx = json.load(f)
frame_w, frame_h = fx["frame_w"], fx["frame_h"]
pan = [tuple(p) for p in fx["pan_polygon"]]
edges = [tuple(int(v) for v in k.split(",")) for k in fx["clicked_lines"]]
point_sets = [fx["clicked_lines"][k] for k in fx["clicked_lines"]]
all_points = [p for s in point_sets for p in s]
rows = [WatcherRow(np.asarray(w["line"]), [tuple(p) for p in w["points"]], w["rms_px"],
                    w["n_frames"], w["extent_px"], w["median_box_h"], w["fraction_on_surface"])
        for w in fx["watcher_rows"]]

calib = fit_calibration_with_lines([], [], "regression fixture", point_sets, edges, cfg=cfg)
worst = max(r["max"] for r in check_line_reprojection_error(calib, cfg))
extent = visible_extent(calib, frame_w, frame_h, pan, cfg)
validity = field_validity_region(calib, cfg)

# --------------------------------------------------------------------------
print("\n1. THE HUMAN-CONFIRMED LABELLING MUST SURVIVE EVERY GATE")
print("   (three gates have rejected it historically -- all by reading past the horizon)")
check("fit reproduces the clicks to <= 5px", worst <= 5.0, f"{worst:.2f}px")
check("no_parallel_crossings", no_parallel_crossings(list(zip(edges, [_Line(p) for p in point_sets]))))
check("vertices_within_bounds", vertices_within_bounds(calib, frame_w, frame_h, cfg))
check("projected_field_is_sane", projected_field_is_sane(calib, cfg))
check("calibration_covers_evidence", calibration_covers_evidence(calib, all_points, cfg))
check("field_covers_points", field_covers_points(calib, all_points, cfg))
check("extent_is_plausible", extent_is_plausible(extent, cfg))
check("every fitted edge counts as seen",
      all(extent_supports_observation(e, extent, cfg) for e in edges))
check("watchers_outside_field", watchers_outside_field(calib, rows, cfg, min_points=10))

# --------------------------------------------------------------------------
print("\n2. LOCAL IS NOT BROKEN (Result 3.14)")
check("this calibration is recognised as LOCAL", not validity["whole_field_usable"])
check("usable extent is ~55% of the field, not ~0 or 100%",
      0.45 <= validity["usable_fraction"] <= 0.65,
      f"{validity['usable_fraction'] * 100:.0f}%")
check("it is still served despite being local",
      calibration_covers_evidence(calib, all_points, cfg))

# --------------------------------------------------------------------------
print("\n3. THE VALIDITY BOUND REACHES THE METRICS (Result 3.20)")
accepted = measurable = 0
for x in range(0, frame_w, 15):
    for y in range(0, frame_h, 15):
        if is_in_field(x, y, calib, cfg):
            accepted += 1
            if is_measurable(x, y, calib, cfg):
                measurable += 1
check("is_measurable is strictly stricter than is_in_field", measurable < accepted,
      f"{accepted - measurable} of {accepted} samples are past the horizon")
check("but it does not reject everything", measurable > 0.5 * accepted,
      f"{100 * measurable / max(accepted, 1):.0f}% kept")

# --------------------------------------------------------------------------
print("\n4. THE DECOY MUST BE REJECTED (Results 3.16 / 3.18)")
print("   the near-parallel line the person's watcher measurement places at field")
print("   y ~ 50m -- the NEXT FIELD OVER -- which every strength-based metric prefers")
decoy_sets = [fx["clicked_lines"]["1,7"], fx["decoy_far_sideline"],
              fx["clicked_lines"]["3,4"], fx["clicked_lines"]["1,2"]]
decoy = fit_calibration_with_lines([], [], "decoy", decoy_sets, edges, cfg=cfg)
check("watchers_outside_field rejects the decoy labelling",
      not watchers_outside_field(decoy, rows, cfg, min_points=10))
check("the decoy fits its own lines far worse",
      max(r["max"] for r in check_line_reprojection_error(decoy, cfg)) > 4 * worst,
      f"{max(r['max'] for r in check_line_reprojection_error(decoy, cfg)):.1f}px vs {worst:.2f}px")

# --------------------------------------------------------------------------
print("\n5. THE SIX-LINE MODEL SHAPE (Result 3.8)")
model = build_field_model(calib, [_Line(p) for p in point_sets], [],
                           list(zip(edges, [_Line(p) for p in point_sets])), cfg,
                           frame_w=frame_w, frame_h=frame_h, pan_polygon=pan)
check("exactly six lines, always", len(model.lines) == 6, f"{len(model.lines)}")
check("the four fitted edges report as observed", model.n_observed == 4, f"{model.n_observed}/6")
check("the two far edges are estimated and unseen",
      all(l.source == "estimated" and l.visibility == "unseen"
          for l in model.lines if l.edge in ((7, 8), (5, 6))))

# --------------------------------------------------------------------------
print("\n6. CONFUSABLE PAIRS ARE SURFACED, NOT SILENTLY RESOLVED (Result 3.17)")
groups = confusable_groups([_Line(fx["clicked_lines"]["2,8"]), _Line(fx["decoy_far_sideline"])])
check("the true far sideline and its decoy are flagged as confusable", len(groups) == 1)

# --------------------------------------------------------------------------
print("\n7. THE TEST ITSELF MUST BE ABLE TO FAIL")
print("   Reintroduce each historical bug and confirm the matching assertion flips.")
print("   A regression test that cannot fail is worse than none -- it reports safety.")
import frisbee_analysis.calibration as _CAL
import frisbee_analysis.field_model as _FM
from frisbee_analysis.calibration import pixel_to_field as _p2f

# BUG A: projected_field_is_sane judging corners past the horizon (3 gates did this)
_real_validity = _CAL.field_validity_region
_CAL.field_validity_region = lambda c, cf=None: {
    "usable_x_max": float((cf or cfg).length), "whole_field_usable": True,
    "usable_fraction": 1.0, "horizon_x_at": lambda y: 1e9}
check("catches: projected_field_is_sane reading past the horizon",
      not projected_field_is_sane(calib, cfg))
_CAL.field_validity_region = _real_validity

# BUG B: _bbox_in_field using min/max instead of robust percentiles
def _grid_minmax(points_px, c, cf, samples=24):
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    H = c.H()
    grid = [(gx, gy, H[2, 0] * gx + H[2, 1] * gy + H[2, 2])
            for gx in np.linspace(pts[:, 0].min(), pts[:, 0].max(), samples)
            for gy in np.linspace(pts[:, 1].min(), pts[:, 1].max(), samples)]
    signs = [np.sign(w) for _, _, w in grid if abs(w) > 1e-12]
    dom = 1.0 if sum(1 for sg in signs if sg > 0) >= len(signs) / 2 else -1.0
    xs, ys = [], []
    for gx, gy, w in grid:
        if abs(w) <= 1e-12 or np.sign(w) != dom:
            continue
        fxy = _p2f(gx, gy, c)
        if np.isfinite(fxy[0]) and np.isfinite(fxy[1]):
            xs.append(fxy[0])
            ys.append(fxy[1])
    return (min(xs), max(xs), min(ys), max(ys)) if xs else (float("nan"),) * 4

_real_bbox = _FM._bbox_in_field
_FM._bbox_in_field = _grid_minmax
check("catches: pan extent from min/max instead of percentiles",
      not extent_is_plausible(_FM.visible_extent(calib, frame_w, frame_h, pan, cfg), cfg))
_FM._bbox_in_field = _real_bbox

# BUG C: is_measurable losing the validity bound
_real_measurable = _CAL.is_measurable
_CAL.is_measurable = _CAL.is_in_field
_acc = _meas = 0
for x in range(0, frame_w, 30):
    for y in range(0, frame_h, 30):
        if _CAL.is_in_field(x, y, calib, cfg):
            _acc += 1
            if _CAL.is_measurable(x, y, calib, cfg):
                _meas += 1
check("catches: is_measurable degraded to is_in_field", not (_meas < _acc))
_CAL.is_measurable = _real_measurable

# --------------------------------------------------------------------------
print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} REGRESSION(S):")
    for f_ in failures:
        print(f"   - {f_}")
    sys.exit(1)
print("all regressions pass")
