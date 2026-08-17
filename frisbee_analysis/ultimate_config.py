"""
UltimatePitchConfiguration

WFDF-field analogue of roboflow/sports' SoccerPitchConfiguration. Same shape, so
the sports annotators (draw_pitch, draw_points_on_pitch) and ViewTransformer
work against it unchanged -- that's the whole point of matching the pattern.

Dimensions verified against WFDF Rules of Ultimate 2025-2028, Appendix A1.2.2:
  - total length 100 m (central zone 64 m + two 18 m endzones)
  - width 37 m
  - brick marks 18 m from each goal line, centred between sidelines
  - corners of central zone + endzones marked by 8 cones (A2/2.6)
Units are CENTIMETRES to match the soccer config exactly (it uses cm). If you
mix this with sports code, keep cm; the homography and annotators don't care
about the unit as long as it's consistent.

*** THE KEYPOINTS THAT MATTER ***
Soccer calibrates off a rich painted-line structure (penalty boxes, centre
circle, ~32 keypoints). You have NONE of that. An ultimate field is bare grass
plus 8 cones. So your usable, reliably-detectable keypoints are:
    the 8 cones (4 outer corners + 4 goal-line corners) + 2 brick marks = 10.
Everything else in `vertices` below is geometrically real but NOT visually
detectable on your footage (no paint), so DON'T feed it to a keypoint model as a
detection target. It's there for drawing the minimap and for hand-clicking
during manual calibration.

`CALIBRATION_KEYPOINTS` lists the subset you can actually detect/click. Use that
subset as the `source`/`target` for ViewTransformer.

Coordinate frame: origin (0,0) at one back-of-endzone corner. x runs along the
100 m length, y across the 37 m width. Matches soccer config convention
(length = x, width = y).
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class UltimatePitchConfiguration:
    width: int = 3700          # [cm] sideline-to-sideline
    length: int = 10000        # [cm] full length incl. both endzones
    endzone_depth: int = 1800  # [cm]
    brick_from_goal: int = 1800  # [cm] brick mark distance from goal line

    @property
    def goal_line_near(self) -> int:
        """x of the endzone-1 goal line (near end)."""
        return self.endzone_depth

    @property
    def goal_line_far(self) -> int:
        """x of the endzone-2 goal line (far end)."""
        return self.length - self.endzone_depth

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """All geometrically-meaningful points. NOTE: only a subset is visually
        detectable (see CALIBRATION_KEYPOINTS). Indexed 1..N in comments to
        mirror the soccer config's 1-based labelling used by the annotators."""
        w, L = self.width, self.length
        gn, gf = self.goal_line_near, self.goal_line_far
        bn = self.brick_from_goal                      # near brick x
        bf = L - self.brick_from_goal                  # far brick x
        return [
            # --- back line, near endzone (x=0) ---
            (0, 0),            # 1  cone: near-left back corner   [DETECTABLE]
            (0, w),            # 2  cone: near-right back corner  [DETECTABLE]
            # --- near goal line (x=gn) ---
            (gn, 0),           # 3  cone: near-left goal corner   [DETECTABLE]
            (gn, w),           # 4  cone: near-right goal corner  [DETECTABLE]
            # --- far goal line (x=gf) ---
            (gf, 0),           # 5  cone: far-left goal corner    [DETECTABLE]
            (gf, w),           # 6  cone: far-right goal corner   [DETECTABLE]
            # --- back line, far endzone (x=L) ---
            (L, 0),            # 7  cone: far-left back corner    [DETECTABLE]
            (L, w),            # 8  cone: far-right back corner   [DETECTABLE]
            # --- brick marks (crossed 1 m lines, centre width) ---
            (bn, w // 2),      # 9  near brick mark               [DETECTABLE]
            (bf, w // 2),      # 10 far brick mark                [DETECTABLE]
            # --- midline sideline points (geometric aid, NOT painted) ---
            (L // 2, 0),       # 11 midfield left  [draw-only, not detectable]
            (L // 2, w),       # 12 midfield right [draw-only, not detectable]
        ]

    # Indices (1-based, into `vertices`) of points you can actually detect or
    # click for homography. THESE are what you pass to ViewTransformer.
    CALIBRATION_INDICES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    @property
    def calibration_keypoints(self) -> List[Tuple[int, int]]:
        """The detectable subset, in field coords, ready as ViewTransformer
        `target`. Pair each with its clicked/detected pixel location as
        `source`, in the same order."""
        vs = self.vertices
        return [vs[i - 1] for i in self.CALIBRATION_INDICES]

    # Edges for drawing the minimap. 1-based index pairs, matching soccer config.
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2),   # near back line
        (3, 4),   # near goal line
        (5, 6),   # far goal line
        (7, 8),   # far back line
        (1, 7),   # left sideline (back to back)
        (2, 8),   # right sideline (back to back)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"
    ])

    colors: List[str] = field(default_factory=lambda: [
        # cones pink, bricks blue, draw-only aids grey
        "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#FF1493", "#FF1493", "#FF1493", "#FF1493",
        "#00BFFF", "#00BFFF",
        "#808080", "#808080",
    ])


if __name__ == "__main__":
    cfg = UltimatePitchConfiguration()
    print(f"Field: {cfg.length/100:.0f}m x {cfg.width/100:.0f}m, "
          f"endzones {cfg.endzone_depth/100:.0f}m, "
          f"brick {cfg.brick_from_goal/100:.0f}m from goal")
    print(f"Near goal line at x={cfg.goal_line_near/100:.0f}m, "
          f"far goal line at x={cfg.goal_line_far/100:.0f}m")
    print(f"\n{len(cfg.calibration_keypoints)} detectable calibration keypoints "
          f"(cm):")
    for idx, pt in zip(cfg.CALIBRATION_INDICES, cfg.calibration_keypoints):
        print(f"  vertex {idx:2d}: {pt}")
