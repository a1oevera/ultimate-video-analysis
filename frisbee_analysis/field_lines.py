"""
Scene-aware field line/marker detection and cross-pan line TRACKING.

WHY THIS EXISTS: results/calibration_finding.md measured naive automated
detection as unusable on this footage, twice, for two different reasons --
Hough-on-the-whole-frame drowned in treeline/scoreboard/caption edges, and
HSV cone filtering drowned in crowd skin tone. Both failures share ONE root
cause: the detector was run over the ENTIRE frame, with no idea which pixels
are even part of a playing surface. Everything here follows from fixing that
-- work out where the surface is FIRST, then only look for field markings
inside it.

The old finding called restricting cone search to the field "circular
without already having a calibration". MEASURED: it isn't. A playing surface
is recoverable from colour + texture alone, with no geometry at all (see
`playing_surface_mask`) -- so "field-region-only" detection is available
BEFORE any homography exists, which is exactly what breaks the circle.

THREE INDEPENDENT SCENE CUES, in the order they buy the most:

  1. `playing_surface_mask` -- green + LOW-TEXTURE + one grown component.
     Trees are green too; the discriminator that separates them is texture,
     not colour (MEASURED, torovcanyon.mp4 t=3945s: turf local_std 1-4,
     crowd row 20.6).
  2. `detect_line_segments` -- white paint is a thin BRIGHT RIDGE on a
     darker surface, which is a top-hat response, not a Canny edge. Canny
     fires on every grass-texture and player boundary; the top-hat only
     fires on structures narrower than its kernel.
  3. `detect_cone_hints` -- cones are hints, never anchors. Person boxes are
     subtracted first: skin tone was the binding confound in the earlier
     measurement, and people standing ON the field are not excluded by the
     surface mask (MEASURED, this module: on-surface orange blobs 17 -> 0
     and 15 -> 1 once YOLO person boxes are subtracted).

CAMERA PANS -- the "tracking" half. A single frame's detections are noisy
and partial (a pan shows one end of a line, then the other). `track_lines`
lifts each frame's detections into the segment's SHARED reference
coordinates using the homographies mosaic.register_sequence already
computes, then clusters across frames. A real painted line is static in the
world, so every frame that sees any part of it contributes to the SAME
reference-space cluster; a shadow edge, a player, or a one-frame artifact
does not repeat. Persistence across frames is therefore a filter no
single-frame detector can implement, and it comes for free from machinery
this project already has.

The output is deliberately shaped as `line_image_points` for
`calibration.fit_calibration_with_lines` in its LINES-ONLY mode -- the one
path results/calibration_finding.md Addendum 3 validates as exactly
correct. Nothing here fits a homography behind your back; it proposes
lines, and `rank_line_labelings` proposes what they might BE.
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from itertools import permutations
from .ultimate_config import UltimatePitchConfiguration


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class SurfaceConfig:
    """Playing-surface segmentation. Thresholds are MEASURED on real frames
    from all three source videos (see results/field_lines_finding.md), not
    guessed -- but they are colour thresholds on broadcast footage, so treat
    them as "works on what we have", not universal."""
    hue_lo: int = 25         # OpenCV hue (0-179). Turf and dry grass both land 25-95;
    hue_hi: int = 95         # sky/blue kit sit above, red/orange below.
    sat_min: int = 40
    val_min: int = 30
    texture_win_at_1920: int = 9   # MEASURED: must scale with frame width, not be
    # fixed. At a fixed 9px this window spans 3x as much of the SCENE on the 640x360
    # source as on the 1920x1080 ones, so it straddles the painted yard lines of a
    # gridiron and reads the whole field as "high texture" -- the surface mask
    # collapsed to 8.4% of the frame and the detector found zero lines there.
    texture_std_max: float = 14.0  # MEASURED: lit turf 1.6-3.1, shaded turf 0.8-4.4,
    # crowd/tents 20.6. Trees measure LOW in a blue-sky frame's dark silhouette but
    # high in a lit one -- texture alone doesn't exclude the treeline, the connected-
    # component step below does. Both are needed.
    gap_px_at_1920: int = 25  # how wide a split the surface is grown across, at 1920px
    # frame width (scaled for other widths). MEASURED (torovcanyon.mp4 t=3945s): a
    # shadow seam and a dark painted line each split the turf into separate
    # components -- the surface is ONE physical thing, so neighbouring components are
    # absorbed. Doing this with a bigger morphological CLOSE instead does NOT work:
    # anything >=15px there also bridges the field to the treeline (measured: surface
    # area jumps 48% -> 79% and its top edge leaps from y=353 to y=0).
    min_component_frac: float = 0.005  # ignore specks when growing
    line_close_px_at_1920: int = 15   # re-absorb painted lines, which are NOT green,
    # back into the surface at the end -- otherwise the very markings this module
    # looks for are holes in the mask it searches.


@dataclass
class LineConfig:
    """White-paint ridge detection inside the surface.

    MEASURED, first version REPLACED (torovcanyon.mp4 t=3945s): a single-scale
    top-hat thresholded by Otsu found ONE line in the whole frame. Two
    independent reasons, both fixed here:
      - Otsu is a GLOBAL threshold and this footage is half sunlit, half in
        shadow -- it locked onto the brightest sunlit paint and discarded
        every shaded line. Replaced by a local-contrast ridge test whose
        threshold scales with local brightness.
      - one kernel size can't work: perspective makes the same painted line
        ~2px wide at the far touchline and ~20px wide in the foreground, so
        a ridge test at one scale is blind to most of the frame. Now
        multi-scale (measured: merged lines 3 -> 8 on the same frame)."""
    ridge_scales_at_1920: tuple = (3, 7, 14, 24)  # half-widths probed, in px at 1920
    ridge_tau_abs: float = 12.0   # minimum brightness step over the surrounding surface
    ridge_tau_rel: float = 0.22   # ...or this fraction of local brightness, whichever is larger.
    # The relative term is what makes one setting work in sun AND shade: measured paint
    # sits ~2.7x the surrounding turf's V in both, while the absolute gap does not.
    chroma_tau: float = 5.0       # ...and the same ridge test on COLOUR distance from the
    # local turf colour, which is what finds paint that is not brighter than grass.
    # MEASURED (torovcanyon.mp4 t=3945s, person's own report that "the close sideline is a
    # blue line"): that sideline's V is 56-68 against turf's 52-71 -- no brightness step
    # exists to detect, by construction. Its HUE is 84-101 against turf's 36-42, and its
    # Lab chroma distance from the surrounding turf is 22-26 against turf's own 0.4-1.3.
    # So brightness alone is structurally blind to it; chroma is not. Chroma also survives
    # shadow better than luminance, which is why the two channels are OR'd rather than
    # the luminance one being replaced.
    chroma_window_at_1920: int = 121  # window for "the local turf colour" 
    search_dilate_px_at_1920: int = 15  # MEASURED: the sidelines ARE the surface's own
    # boundary, so a search region clipped exactly to the green mask misses precisely
    # the lines that matter most (merged lines 3 -> 8 with this at 15px).
    min_elongation: float = 5.0   # reject compact blobs (white shorts, kit bags) and the
    # soccer/lacrosse CENTRE CIRCLE these fields are painted with.
    #
    # MEASURED BUG, first version was BROKEN (torovcanyon.mp4 t=3945s): elongation was
    # computed from the component's BOUNDING BOX, which is near-square for any DIAGONAL
    # line however thin it is. The blue near sideline -- a 7,090px component running
    # corner to corner -- scored bbox-elongation 1.17 and was silently discarded, after
    # the detector had correctly found it. Now measured from the component's second
    # moments (sqrt of the eigenvalue ratio of its pixel-coordinate covariance), which
    # is orientation-invariant: that same line scores 32.4. The threshold sits in a
    # clean measured gap -- real lines score 19-36, players and the centre circle score
    # 1.9-4.0.
    min_component_px: int = 40
    flank_tolerance: float = 26.0  # a painted line has the SAME surface on both sides;
    # a scene boundary does not. MEASURED (ojuc.mp4 t=1580s): the row of white spectator
    # tents along the grass edge is a genuinely thin, bright, elongated band -- a real
    # ridge by every local test -- and got reported as a 1,636px "line" on a field that
    # has no paint at all. Grass on one side, tents and parked cars on the other. Tested
    # the obvious alternative first and MEASURED IT WRONG: requiring a detection to lie
    # mostly inside the un-dilated surface mask deletes the sidelines, because a sideline
    # IS the surface's own boundary (torovcanyon.mp4 t=3945s: 9 lines -> 4, losing both
    # long white ones). What separates the two cases is not where the line sits, it is
    # whether the two sides MATCH.
    hough_threshold: int = 55
    min_length_frac: float = 0.045   # of frame width
    max_gap_frac: float = 0.012
    merge_angle_deg: float = 2.5
    merge_offset_px_at_1920: float = 25.0  # MEASURED: 14px was too tight. The ridge test
    # probes half-widths up to 24px, and a foreground painted line really is ~10-20px
    # wide, so two honest detections of the SAME line sit that far apart before any
    # registration drift is added. At 14px the blue near sideline arrived as two separate
    # candidates 19.1px apart (torovcanyon.mp4 t=3945-4005s). Safe because distinct
    # parallel field lines are hundreds of px apart in these framings, not tens.
    merge_extrapolation: bool = True  # allow two fragments of ONE line to merge across a
    # large gap along that line, widening the perpendicular tolerance by the uncertainty
    # cone the angle tolerance implies. MEASURED NEED (person's report): on this field the
    # far sideline "is only painted on the endzone" -- in midfield there is no paint at
    # all. Its two painted stretches are one physical line separated by ~60m of nothing,
    # and a fixed 14px tolerance cannot join them: 2.5 degrees of angle error over that
    # distance is far more than 14px of offset, so the same line arrives as two.


@dataclass
class ConeConfig:
    """Cone hints. Deliberately permissive on shape and strict on context:
    the context filters (surface + people) are what MEASURED as decisive,
    the shape filters are not doing the heavy lifting."""
    hue_max: int = 22
    sat_min: int = 120
    val_min: int = 90
    min_area: int = 6
    max_area: int = 4000
    min_aspect: float = 0.4
    max_aspect: float = 2.2
    min_fill: float = 0.35
    surface_dilate_px: int = 31   # cones sit ON the boundary, so allow a border
    person_pad: float = 0.15      # grow person boxes before subtracting them


# --------------------------------------------------------------------------
# data types
# --------------------------------------------------------------------------

@dataclass
class LineSegment:
    """A detected straight segment. `points` are samples along it in whatever
    coordinate space it was produced in (frame pixels for a single-frame
    detection, segment-reference pixels once lifted by `track_lines`)."""
    p1: tuple
    p2: tuple
    points: list = field(default_factory=list)
    support_px: int = 0
    frame_idx: int = None
    colour: str = "unknown"   # "white" | "blue" | "other" -- see classify_paint_colour

    @property
    def length(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))

    @property
    def angle_deg(self) -> float:
        """0-180, undirected."""
        a = np.degrees(np.arctan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0]))
        return float(a % 180.0)

    def homogeneous(self) -> np.ndarray:
        """[a,b,c] with ax+by+c=0, normalized so hypot(a,b)==1 -- so
        |a*x+b*y+c| is a true perpendicular pixel distance."""
        l = np.cross([self.p1[0], self.p1[1], 1.0], [self.p2[0], self.p2[1], 1.0])
        n = np.hypot(l[0], l[1])
        return l / n if n > 1e-9 else l


@dataclass
class ConeHint:
    x: float
    y: float
    area: int
    frame_idx: int = None


@dataclass
class TrackedLine:
    """One physical line, seen across several frames of a pan, expressed in
    the segment's shared reference coordinates."""
    p1: tuple
    p2: tuple
    points: list                 # every contributing sample, all frames pooled
    frame_indices: list          # which raw frame each entry in `points` came from
    n_frames: int                # DISTINCT frames that saw this line
    total_support_px: int
    rms_fit_px: float            # how well the pooled points actually fit one line
    colour: str = "unknown"      # majority paint colour of the contributing detections

    def to_segment(self) -> LineSegment:
        return LineSegment(self.p1, self.p2, list(self.points), self.total_support_px,
                            colour=self.colour)

    def homogeneous(self) -> np.ndarray:
        return self.to_segment().homogeneous()


# --------------------------------------------------------------------------
# 1. playing surface
# --------------------------------------------------------------------------

def _local_std(gray: np.ndarray, win: int) -> np.ndarray:
    f = gray.astype(np.float32)
    mu = cv2.boxFilter(f, -1, (win, win))
    mu2 = cv2.boxFilter(f * f, -1, (win, win))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))


def _scaled(px_at_1920: int, frame_w: int, minimum: int = 3) -> int:
    return max(minimum, int(round(px_at_1920 * frame_w / 1920.0)))


def _fill_interior_holes(mask: np.ndarray, step: int = 25) -> np.ndarray:
    """Fill regions of background fully enclosed by the mask (players, kit
    bags, and the painted lines themselves). Flood the background inward
    from the border; whatever background is left is enclosed."""
    h, w = mask.shape
    background = cv2.bitwise_not(mask)
    pad = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, step):
        for y in (0, h - 1):
            if background[y, x] == 255:
                cv2.floodFill(background, pad, (x, y), 128)
    for y in range(0, h, step):
        for x in (0, w - 1):
            if background[y, x] == 255:
                cv2.floodFill(background, pad, (x, y), 128)
    out = mask.copy()
    out[background == 255] = 255
    return out


def playing_surface_mask(bgr: np.ndarray, overlay_boxes=(), cfg: SurfaceConfig = None) -> np.ndarray:
    """Which pixels are the playing surface (turf/grass), as a 0/255 mask.
    No geometry, no calibration, no training -- colour + texture + one
    connectivity rule. This is the step that un-circularizes "only look for
    field markings on the field".

    MEASURED (results/field_lines_finding.md): on torovcanyon.mp4 t=3945s
    this keeps 58.5% of the frame with its top edge at y=353-404 -- i.e. it
    ends at the real far touchline and excludes the treeline, powerlines,
    sky, hedge, spectator tents and the scoreboard. That single fact is what
    takes Hough segment counts from "89 lines, nearly all treeline/seam
    artifacts" (the original finding) down to something workable.

    Pass `overlay_boxes` (frisbee_analysis.video_overlays.overlay_boxes_for)
    -- a static broadcast graphic sitting over grass is not grass, and a
    green-ish watermark would otherwise anchor part of the mask."""
    cfg = cfg or SurfaceConfig()
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    green = (H >= cfg.hue_lo) & (H <= cfg.hue_hi) & (S >= cfg.sat_min) & (V >= cfg.val_min)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    texture_win = _scaled(cfg.texture_win_at_1920, w, minimum=3)
    smooth = _local_std(gray, texture_win) < cfg.texture_std_max

    mask = (green & smooth).astype(np.uint8) * 255
    for (x0, y0, x1, y1) in overlay_boxes:
        mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return mask

    # Grow from the largest component across THIN splits only (shadow seams,
    # dark painted lines). Deliberately iterative rather than one big close:
    # a close big enough to bridge these also bridges field -> treeline.
    frame_area = float(h * w)
    keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    candidates = [i for i in range(1, n_labels)
                  if i not in keep and stats[i, cv2.CC_STAT_AREA] >= cfg.min_component_frac * frame_area]
    gap = _scaled(cfg.gap_px_at_1920, w, minimum=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1))
    changed = True
    while changed and candidates:
        changed = False
        current = np.isin(labels, list(keep)).astype(np.uint8) * 255
        reachable = cv2.dilate(current, kernel)
        for i in list(candidates):
            component = labels == i
            if (reachable[component] > 0).mean() > 0.05:
                keep.add(i)
                candidates.remove(i)
                changed = True

    surface = np.isin(labels, list(keep)).astype(np.uint8) * 255
    close_px = _scaled(cfg.line_close_px_at_1920, w, minimum=3)
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE,
                                np.ones((close_px, close_px), np.uint8))
    return _fill_interior_holes(surface)


# --------------------------------------------------------------------------
# 2. line markings inside the surface
# --------------------------------------------------------------------------

def classify_paint_colour(bgr: np.ndarray, seg: LineSegment, samples: int = 40) -> str:
    """What colour is this line painted? Sampled from the pixels under it.

    Worth reporting because it is how the marking actually differs between
    lines on a real multi-sport field, and therefore how a person identifies
    them. On torovcanyon.mp4 (person's own description) the near sideline is
    a BLUE line while the far sideline and goal lines are white -- so the
    colour alone rules out most mislabelings, which is exactly the failure
    that produced the bowtie in results/field_lines_finding.md.

    Measured on that footage: the blue sideline reads hue 84-101 at S 68-135
    and V 56-68 -- the same brightness as the turf around it. White paint
    reads as a clear brightness step at low saturation."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    xs = np.linspace(seg.p1[0], seg.p2[0], samples)
    ys = np.linspace(seg.p1[1], seg.p2[1], samples)
    hues, sats, vals = [], [], []
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        hues.append(int(hsv[yi, xi, 0]))
        sats.append(int(hsv[yi, xi, 1]))
        vals.append(int(hsv[yi, xi, 2]))
    if len(hues) < 3:
        return "unknown"
    hue, sat, val = float(np.median(hues)), float(np.median(sats)), float(np.median(vals))
    if 80 <= hue <= 130 and sat >= 50:
        return "blue"
    if sat < 90 and val >= 110:
        return "white"
    return "other"


def _multiscale_ridge(channel: np.ndarray, scales, tau_abs: float, tau_rel: float,
                       frame_w: int) -> np.ndarray:
    """True where `channel` is higher than the channel a line-width away on
    BOTH sides, at any of the probed half-widths. A step between two large
    regions (shadow edge, field boundary, a player's outline) fails by
    construction, because only one side is lower -- which is exactly the
    difference from a Canny edge."""
    h, w = channel.shape
    found = np.zeros((h, w), dtype=bool)
    for scale in scales:
        d = _scaled(scale, frame_w, minimum=2)
        if d * 2 >= min(h, w):
            continue
        local_mean = cv2.boxFilter(channel, -1, (d * 6 + 1, d * 6 + 1))
        tau = np.maximum(tau_abs, tau_rel * local_mean)
        up = np.roll(channel, d, axis=0)
        down = np.roll(channel, -d, axis=0)
        left = np.roll(channel, d, axis=1)
        right = np.roll(channel, -d, axis=1)
        at_scale = (((channel - up > tau) & (channel - down > tau)) |
                    ((channel - left > tau) & (channel - right > tau)))
        # np.roll wraps around, so the borders of each shifted copy are garbage
        at_scale[:d * 2, :] = False
        at_scale[-d * 2:, :] = False
        at_scale[:, :d * 2] = False
        at_scale[:, -d * 2:] = False
        found |= at_scale
    return found


def _elongation(pixel_ys: np.ndarray, pixel_xs: np.ndarray) -> float:
    """How line-like a pixel set is, independent of its ORIENTATION: the
    square root of the eigenvalue ratio of its coordinate covariance.

    Deliberately not the bounding-box aspect ratio, which was the first
    version and was measurably wrong -- a thin diagonal line has a
    near-square bounding box, so bbox-elongation rejected a 7,090px painted
    sideline at 1.17 while this scores the same pixels 32.4."""
    if len(pixel_xs) < 3:
        return 0.0
    coords = np.stack([pixel_xs.astype(np.float64), pixel_ys.astype(np.float64)])
    covariance = np.cov(coords)
    if not np.all(np.isfinite(covariance)):
        return 0.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor = max(float(eigenvalues[0]), 1e-9)
    major = max(float(eigenvalues[1]), 1e-9)
    return float(np.sqrt(major / minor))


def line_pixel_mask(bgr: np.ndarray, surface: np.ndarray, cfg: LineConfig = None) -> np.ndarray:
    """Pixels that look like painted line markings, restricted to (a slightly
    grown) `surface`.

    NOT Canny. Canny answers "is there an intensity step here", which is true
    at every grass tuft, player outline and shadow boundary -- that is why the
    original attempt in results/calibration_finding.md returned 80 segments
    dominated by the treeline and the caption underline. This asks the
    question that actually defines a painted line: is this pixel different
    from the surface a line-width away on BOTH sides?

    TWO CHANNELS, OR'd, because "different" is not always "brighter":
      - brightness (HSV V), which finds white paint;
      - distance in Lab chroma from the LOCAL turf colour, which finds paint
        that is not brighter than grass at all. The blue near sideline on
        this footage is exactly that case: same V as the turf, completely
        different hue (see LineConfig.chroma_tau for the measurements).
    Probed at several half-widths because perspective makes one real line
    ~2px wide at the far touchline and ~20px wide in the foreground, and
    thresholded relative to local brightness so one setting works in sun and
    shade alike.

    Elongation filtering then removes what survives but isn't a line: white
    shorts, kit bags, and the centre circle of the soccer/lacrosse markings
    these ultimate games are played over."""
    cfg = cfg or LineConfig()
    h, w = bgr.shape[:2]
    scales = cfg.ridge_scales_at_1920

    dilate_px = _scaled(cfg.search_dilate_px_at_1920, w, minimum=3)
    region = cv2.dilate(surface, np.ones((dilate_px, dilate_px), np.uint8))

    value = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    brightness_ridge = _multiscale_ridge(value, scales, cfg.ridge_tau_abs, cfg.ridge_tau_rel, w)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma_win = _scaled(cfg.chroma_window_at_1920, w, minimum=15) | 1
    a_local = cv2.boxFilter(lab[:, :, 1], -1, (chroma_win, chroma_win))
    b_local = cv2.boxFilter(lab[:, :, 2], -1, (chroma_win, chroma_win))
    chroma_distance = np.hypot(lab[:, :, 1] - a_local, lab[:, :, 2] - b_local)
    chroma_ridge = _multiscale_ridge(chroma_distance, scales, cfg.chroma_tau, 0.0, w)

    # Shape-filter each channel SEPARATELY, then union the survivors.
    # MEASURED BUG, first version OR'd the two channels first (torovcanyon.mp4
    # t=3975s): chroma noise touching a clean white line fused them into one
    # component whose elongation then fell below threshold, so adding a channel
    # DESTROYED detections instead of adding them -- the frame's two longest
    # lines (1120px and 792px) vanished the moment chroma was switched on.
    # Filtering per channel keeps one channel's noise from disqualifying
    # another channel's evidence.
    out = np.zeros(value.shape, dtype=np.uint8)
    for channel_ridge in (brightness_ridge, chroma_ridge):
        binary = cv2.bitwise_and((channel_ridge.astype(np.uint8) * 255), region)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < cfg.min_component_px:
                continue
            component = labels == i
            ys, xs = np.nonzero(component)
            if _elongation(ys, xs) < cfg.min_elongation:
                continue
            out[component] = 255
    return out


def flanks_match(bgr: np.ndarray, seg: LineSegment, tolerance: float,
                  offset_px: int = None, samples: int = 25) -> bool:
    """Is the surface on one side of this line the same as on the other?

    A painted marking lies ON a uniform surface, so grass to its left looks
    like grass to its right. A boundary in the scene -- the edge of the
    field, a tent row, a treeline -- has different worlds on each side. This
    is the difference the local ridge test cannot express, because the ridge
    test only compares the line to its surroundings, never the surroundings
    to each other.

    Compared in Lab, which keeps this from firing on a line that merely runs
    through shadow on one side."""
    h, w = bgr.shape[:2]
    if offset_px is None:
        offset_px = max(6, int(round(0.012 * w)))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    direction = np.array([seg.p2[0] - seg.p1[0], seg.p2[1] - seg.p1[1]], dtype=np.float64)
    length = np.hypot(*direction)
    if length < 1e-6:
        return True
    normal = np.array([-direction[1], direction[0]]) / length
    left, right = [], []
    for t in np.linspace(0.05, 0.95, samples):
        base = np.array([seg.p1[0], seg.p1[1]]) + t * direction
        for sign, bucket in ((1.0, left), (-1.0, right)):
            p = base + sign * offset_px * normal
            xi, yi = int(round(p[0])), int(round(p[1]))
            if 0 <= xi < w and 0 <= yi < h:
                bucket.append(lab[yi, xi])
    if len(left) < 5 or len(right) < 5:
        return True   # can't tell (line runs off-frame) -- don't reject on no evidence
    return float(np.linalg.norm(np.median(left, axis=0) - np.median(right, axis=0))) <= tolerance


def detect_line_segments(bgr: np.ndarray, surface: np.ndarray = None,
                          overlay_boxes=(), cfg: LineConfig = None,
                          surface_cfg: SurfaceConfig = None,
                          frame_idx: int = None) -> list:
    """Straight segments of line marking visible in this frame. Computes the
    playing-surface mask itself if you don't pass one."""
    cfg = cfg or LineConfig()
    if surface is None:
        surface = playing_surface_mask(bgr, overlay_boxes, surface_cfg)
    binary = line_pixel_mask(bgr, surface, cfg)
    h, w = bgr.shape[:2]
    found = cv2.HoughLinesP(binary, 1, np.pi / 360, threshold=cfg.hough_threshold,
                             minLineLength=int(cfg.min_length_frac * w),
                             maxLineGap=int(cfg.max_gap_frac * w))
    if found is None:
        return []
    segments = []
    for x1, y1, x2, y2 in found[:, 0, :]:
        seg = LineSegment((float(x1), float(y1)), (float(x2), float(y2)),
                           points=[(float(x1), float(y1)), (float(x2), float(y2))],
                           support_px=0, frame_idx=frame_idx)
        seg.support_px = _count_support(binary, seg)
        seg.colour = classify_paint_colour(bgr, seg)
        segments.append(seg)
    merged = merge_collinear(segments, cfg, bgr.shape[1])
    return [m for m in merged if flanks_match(bgr, m, cfg.flank_tolerance)]


def _count_support(binary: np.ndarray, seg: LineSegment, tol: float = 2.0) -> int:
    """How many line-mask pixels actually lie under this segment. Hough
    reports a segment, not how much evidence it had; two segments of equal
    length are not equally believable."""
    h, w = binary.shape
    n = max(2, int(seg.length))
    xs = np.linspace(seg.p1[0], seg.p2[0], n)
    ys = np.linspace(seg.p1[1], seg.p2[1], n)
    hits = 0
    r = int(tol)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        if x1 > x0 and y1 > y0 and binary[y0:y1, x0:x1].any():
            hits += 1
    return hits


def _fit_line_tls(points) -> np.ndarray:
    """Total-least-squares homogeneous line through >=2 points, normalized so
    |a*x+b*y+c| is a perpendicular distance. Same estimator as
    calibration._fit_image_line, kept local so this module doesn't drag the
    calibration import in for one helper."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 2:
        l = np.cross([pts[0, 0], pts[0, 1], 1.0], [pts[1, 0], pts[1, 1], 1.0])
    else:
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        _, _, Vt = np.linalg.svd(A)
        l = Vt[-1]
    n = np.hypot(l[0], l[1])
    return l / n if n > 1e-9 else l


def _extent_along(points, line: np.ndarray):
    """Endpoints of the segment covered by `points`, projected onto `line`."""
    direction = np.array([-line[1], line[0]])
    pts = np.asarray(points, dtype=np.float64)
    # foot of each point on the line, so endpoints sit ON the fitted line
    d = pts @ np.array([line[0], line[1]]) + line[2]
    feet = pts - np.outer(d, np.array([line[0], line[1]]))
    t = feet @ direction
    return tuple(feet[int(np.argmin(t))]), tuple(feet[int(np.argmax(t))])


def _offset_tolerance(group_points, line: np.ndarray, point, cfg: LineConfig,
                       frame_w: int = 1920) -> float:
    """How far off the group's fitted line a candidate point may sit and
    still be the same physical line.

    Flat `merge_offset_px` inside the span the group already covers, widened
    beyond it by the uncertainty cone the angle tolerance implies. A line
    fitted from one 200px painted stretch is only known to within
    `merge_angle_deg`, so at 2,000px away from that stretch its true position
    is uncertain by ~tan(2.5 deg) x 1800px, not by 14px -- demanding 14px
    there splits one line into two. MEASURED NEED: on this field the far
    sideline is painted only inside the two endzones, so its two stretches
    are one line separated by the whole midfield."""
    base = _scaled(int(round(cfg.merge_offset_px_at_1920)), frame_w, minimum=5)
    if not cfg.merge_extrapolation:
        return float(base)
    pts = np.asarray(group_points, dtype=np.float64)
    direction = np.array([-line[1], line[0]])
    t_group = pts @ direction
    t_point = float(np.asarray(point, dtype=np.float64) @ direction)
    beyond = max(0.0, t_point - t_group.max(), t_group.min() - t_point)
    return float(base) + np.tan(np.radians(cfg.merge_angle_deg)) * beyond


def _dominant_colour(colour_votes) -> str:
    """Support-weighted majority: a long, well-supported detection of a blue
    line should outvote a short fragment that happened to read 'other'."""
    weights = {}
    for colour, support in colour_votes:
        weights[colour] = weights.get(colour, 0) + max(1, support)
    known = {c: v for c, v in weights.items() if c not in ("unknown", "other")}
    pool = known or weights
    return max(pool, key=pool.get)


def merge_collinear(segments: list, cfg: LineConfig = None, frame_w: int = 1920) -> list:
    """Collapse Hough's many fragments of one painted line into one segment.

    Greedy, strongest-first: a segment joins an existing group when its angle
    is within `merge_angle_deg` AND both its endpoints sit within the
    group's current fitted line's tolerance (see `_offset_tolerance` -- flat
    inside the span already covered, widening into the extrapolation beyond
    it). Testing the actual endpoints (not just the line's offset at the
    origin) is what keeps two parallel lines a field apart from merging."""
    cfg = cfg or LineConfig()
    groups = []  # list of (points, support)
    for seg in sorted(segments, key=lambda s: -s.length):
        placed = False
        for g in groups:
            line = _fit_line_tls(g["points"])
            da = abs(seg.angle_deg - g["angle"])
            da = min(da, 180.0 - da)
            if da > cfg.merge_angle_deg:
                continue
            d1 = abs(line[0] * seg.p1[0] + line[1] * seg.p1[1] + line[2])
            d2 = abs(line[0] * seg.p2[0] + line[1] * seg.p2[1] + line[2])
            tol1 = _offset_tolerance(g["points"], line, seg.p1, cfg, frame_w)
            tol2 = _offset_tolerance(g["points"], line, seg.p2, cfg, frame_w)
            if d1 > tol1 or d2 > tol2:
                continue
            g["points"].extend([seg.p1, seg.p2])
            g["support"] += seg.support_px
            g["frames"].add(seg.frame_idx)
            g["colours"].append((seg.colour, seg.support_px))
            placed = True
            break
        if not placed:
            groups.append({"points": [seg.p1, seg.p2], "angle": seg.angle_deg,
                            "support": seg.support_px, "frames": {seg.frame_idx},
                            "colours": [(seg.colour, seg.support_px)]})

    merged = []
    for g in groups:
        line = _fit_line_tls(g["points"])
        p1, p2 = _extent_along(g["points"], line)
        merged.append(LineSegment(p1, p2, points=list(g["points"]),
                                   support_px=g["support"],
                                   frame_idx=next(iter(g["frames"])),
                                   colour=_dominant_colour(g["colours"])))
    return sorted(merged, key=lambda s: -s.length)


# --------------------------------------------------------------------------
# 3. cone hints
# --------------------------------------------------------------------------

def detect_cone_hints(bgr: np.ndarray, surface: np.ndarray, person_boxes=(),
                       cfg: ConeConfig = None, frame_idx: int = None) -> list:
    """Orange cone candidates on or just off the playing surface, with people
    subtracted.

    results/calibration_finding.md Addendum 4 measured plain HSV cone
    filtering as unusable, with crowd skin tone the binding confound (81
    false candidates on one crowd shot). BOTH context filters here are
    needed and neither is sufficient:
      - the surface mask removes crowd, tents, parking, a maroon field logo
        and a background goalpost -- but NOT players, who stand on the field;
      - `person_boxes` (YOLO class 0, which this project already runs)
        removes exactly those. MEASURED on this footage: on-surface orange
        blobs 17 -> 0 (torovcanyon.mp4 t=3945s) and 15 -> 1 (ojuc.mp4
        t=1580s) once person boxes are subtracted.

    HONEST LIMIT, measured at the same time: on the bare-grass footage the
    real corner cones are only ~3-5px across at broadcast distance, below
    what this (or arguably any colour filter) reliably separates from noise.
    Treat cone hints as CORROBORATION for a line labelling, which is how
    `rank_line_labelings` uses them -- not as calibration anchors. That is
    also why nothing here fits a homography from cones."""
    cfg = cfg or ConeConfig()
    h, w = bgr.shape[:2]
    allowed = cv2.dilate(surface, np.ones((cfg.surface_dilate_px, cfg.surface_dilate_px), np.uint8))
    for (x0, y0, x1, y1) in person_boxes:
        bw, bh = x1 - x0, y1 - y0
        px0 = int(max(0, x0 - cfg.person_pad * bw))
        px1 = int(min(w, x1 + cfg.person_pad * bw))
        py0 = int(max(0, y0 - cfg.person_pad * bh))
        py1 = int(min(h, y1 + cfg.person_pad * bh))
        allowed[py0:py1, px0:px1] = 0

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)
    orange = ((H <= cfg.hue_max) & (S >= cfg.sat_min) & (V >= cfg.val_min)).astype(np.uint8) * 255
    orange = cv2.bitwise_and(orange, orange, mask=allowed)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(orange, 8)
    hints = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (cfg.min_area <= area <= cfg.max_area):
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw <= 0 or bh <= 0:
            continue
        aspect = bw / bh
        if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
            continue
        if area / float(bw * bh) < cfg.min_fill:
            continue
        hints.append(ConeHint(float(centroids[i][0]), float(centroids[i][1]), area, frame_idx))
    return hints


def person_boxes_from_yolo(model, bgr: np.ndarray, conf: float = 0.20):
    """Convenience wrapper so callers don't have to remember the imgsz rule.
    MEASURED (results/detection_finding.md): YOLO's own 640px default misses
    the entire field of play on this footage -- run at native resolution."""
    h, w = bgr.shape[:2]
    imgsz = 1920 if w >= 1280 else 1280
    result = model.predict(bgr, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    return [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]


# --------------------------------------------------------------------------
# 4. tracking across a pan
# --------------------------------------------------------------------------

def _transform_points(points, H: np.ndarray):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [tuple(map(float, p)) for p in out]


def track_lines(frames: list, registrations: list, segment_id: int = None,
                 overlay_boxes=(), cfg: LineConfig = None,
                 surface_cfg: SurfaceConfig = None,
                 min_frames: int = 3, max_lines: int = 12,
                 split_merged: bool = True,
                 progress: bool = False) -> list:
    """Detect line markings per frame and fuse them into one set of lines in
    the segment's SHARED reference coordinates -- the camera-pan-aware core
    of this module.

    `registrations` is `mosaic.register_sequence(frames, ...)`'s output. Only
    frames with `ok=True` and the chosen `segment_id` are used, because those
    are exactly the frames that share a coordinate system (see
    mosaic.register_sequence -- homographies are NOT comparable across a
    segment boundary). Defaults to the largest segment.

    Why this beats detecting on one frame, and why it beats detecting on the
    stitched mosaic:
      - one frame during a pan shows a fraction of each line, often the
        low-contrast far end; different frames show different fractions, and
        pooling them recovers the whole line;
      - a real line is STATIC in the world, so every frame that sees it
        contributes to the same reference-space cluster. `n_frames` is
        therefore a confound filter nothing single-frame can express: a
        shadow edge that happens to look like a line in one frame doesn't
        repeat in the same reference position across a pan;
      - detection still happens on RAW frames. results/calibration_finding.md
        (Addendum 1) measured that detecting on the blended mosaic instead
        finds the stitching seams, not the field -- this gets the mosaic's
        pooling benefit without ever detecting on composited pixels.

    Returns TrackedLine objects sorted by (n_frames, support), longest-lived
    first, capped at `max_lines`."""
    cfg = cfg or LineConfig()
    by_segment = {}
    for r in registrations:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r)
    if not by_segment:
        return []
    if segment_id is None:
        segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
    members = by_segment.get(segment_id, [])
    if not members:
        return []

    clusters = []
    for n, reg in enumerate(members):
        frame = frames[reg.frame_idx]
        segments = detect_line_segments(frame, None, overlay_boxes, cfg,
                                         surface_cfg, frame_idx=reg.frame_idx)
        if progress:
            print(f"  frame {reg.frame_idx} ({n + 1}/{len(members)}): "
                  f"{len(segments)} merged segments")
        for seg in segments:
            ref_pts = _transform_points([seg.p1, seg.p2], reg.homography)
            _accumulate(clusters, ref_pts, seg.support_px, reg.frame_idx, cfg, seg.colour,
                        frame.shape[1])

    clusters = _consolidate_clusters(clusters, cfg, frames[members[0].frame_idx].shape[1])

    tracked = []
    for c in clusters:
        if len(c["frames"]) < min_frames:
            continue
        line = _fit_line_tls(c["points"])
        pts = np.asarray(c["points"], dtype=np.float64)
        residuals = np.abs(pts @ np.array([line[0], line[1]]) + line[2])
        p1, p2 = _extent_along(c["points"], line)
        tracked.append(TrackedLine(p1=p1, p2=p2, points=list(c["points"]),
                                    frame_indices=list(c["frame_indices"]),
                                    n_frames=len(c["frames"]),
                                    total_support_px=c["support"],
                                    rms_fit_px=float(np.sqrt((residuals ** 2).mean())),
                                    colour=_dominant_colour(c["colours"])))
    # A cluster whose points do not fit ONE line is not one line. Splitting
    # before ranking is what lets a real marking that got merged with a
    # neighbour be selected at all -- see split_merged_lines and Result 3.27.
    if split_merged:
        tracked = split_merged_lines(tracked)
    tracked.sort(key=lambda t: (-t.n_frames, -t.total_support_px))
    return tracked[:max_lines]


def _accumulate(clusters, ref_pts, support, frame_idx, cfg: LineConfig, colour="unknown",
                 frame_w=1920):
    """Add one frame's segment (already in reference coords) to the matching
    persistent cluster, or start a new one. Tolerances are deliberately the
    SAME ones single-frame merging uses -- a line seen from two pan positions
    should land within click-level distance of itself, and if it doesn't,
    the registration for that frame is the problem, not the tolerance."""
    seg = LineSegment(ref_pts[0], ref_pts[1])
    for c in clusters:
        line = _fit_line_tls(c["points"])
        da = abs(seg.angle_deg - c["angle"])
        da = min(da, 180.0 - da)
        if da > cfg.merge_angle_deg:
            continue
        d1 = abs(line[0] * seg.p1[0] + line[1] * seg.p1[1] + line[2])
        d2 = abs(line[0] * seg.p2[0] + line[1] * seg.p2[1] + line[2])
        if (d1 > _offset_tolerance(c["points"], line, seg.p1, cfg, frame_w) or
                d2 > _offset_tolerance(c["points"], line, seg.p2, cfg, frame_w)):
            continue
        c["points"].extend(ref_pts)
        c["frame_indices"].extend([frame_idx] * len(ref_pts))
        c["support"] += support
        c["frames"].add(frame_idx)
        c["colours"].append((colour, support))
        return
    clusters.append({"points": list(ref_pts), "frame_indices": [frame_idx] * len(ref_pts),
                      "angle": seg.angle_deg, "support": support, "frames": {frame_idx},
                      "colours": [(colour, support)]})


def _consolidate_clusters(clusters, cfg: LineConfig, frame_w: int):
    """Second pass over the finished cross-frame clusters, merging any two
    that are really one line.

    `_accumulate` is greedy and incremental: it tests each new detection
    against a cluster's line as fitted from whatever happened to arrive
    first, early in a pan when that is only a short stub. Two halves of one
    line seen at opposite ends of a pan can therefore both start clusters and
    never meet. MEASURED (torovcanyon.mp4 t=3945-4005s): the blue near
    sideline came out as two candidates 19.1px apart, both labelled blue,
    both obviously the same line to a person looking at the frame -- which is
    exactly the "lots of broken lines" this pass exists to stop producing.

    Here every cluster's FULL pooled point set is available, so the
    comparison is between two properly-fitted lines rather than between a
    finished line and a stub."""
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                line_a = _fit_line_tls(a["points"])
                line_b = _fit_line_tls(b["points"])
                angle_a = np.degrees(np.arctan2(-line_a[0], line_a[1])) % 180.0
                angle_b = np.degrees(np.arctan2(-line_b[0], line_b[1])) % 180.0
                delta = abs(angle_a - angle_b)
                delta = min(delta, 180.0 - delta)
                if delta > cfg.merge_angle_deg:
                    continue
                # every point of each must sit within the other's tolerance --
                # symmetric, so a long line can't swallow a parallel short one
                # that merely happens to start near it
                ok = True
                for line, host, guest in ((line_a, a, b), (line_b, b, a)):
                    for point in guest["points"]:
                        distance = abs(line[0] * point[0] + line[1] * point[1] + line[2])
                        if distance > _offset_tolerance(host["points"], line, point, cfg, frame_w):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                a["points"].extend(b["points"])
                a["frame_indices"].extend(b["frame_indices"])
                a["support"] += b["support"]
                a["frames"] |= b["frames"]
                a["colours"].extend(b["colours"])
                clusters.pop(j)
                merged = True
                break
            if merged:
                break
    return clusters


def track_cone_hints(frames: list, registrations: list, segment_id: int = None,
                      overlay_boxes=(), yolo_model=None, cfg: ConeConfig = None,
                      surface_cfg: SurfaceConfig = None,
                      cluster_px: float = 25.0, min_frames: int = 2) -> list:
    """Same persistence idea as `track_lines`, for cone hints: lift each
    frame's hints into reference coordinates and keep only positions that
    repeat across frames. A cone is static; a moving orange thing (a shirt
    the person filter missed, a disc) is not.

    `yolo_model` is optional but strongly recommended -- without it, people
    are not subtracted and skin tone dominates (see `detect_cone_hints`)."""
    cfg = cfg or ConeConfig()
    by_segment = {}
    for r in registrations:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r)
    if not by_segment:
        return []
    if segment_id is None:
        segment_id = max(by_segment, key=lambda s: len(by_segment[s]))

    clusters = []
    for reg in by_segment.get(segment_id, []):
        frame = frames[reg.frame_idx]
        surface = playing_surface_mask(frame, overlay_boxes, surface_cfg)
        boxes = person_boxes_from_yolo(yolo_model, frame) if yolo_model is not None else ()
        for hint in detect_cone_hints(frame, surface, boxes, cfg, reg.frame_idx):
            (rx, ry), = _transform_points([(hint.x, hint.y)], reg.homography)
            for c in clusters:
                if np.hypot(rx - c["x"], ry - c["y"]) <= cluster_px:
                    c["n"] += 1
                    c["frames"].add(reg.frame_idx)
                    c["x"] += (rx - c["x"]) / c["n"]
                    c["y"] += (ry - c["y"]) / c["n"]
                    c["area"] = max(c["area"], hint.area)
                    break
            else:
                clusters.append({"x": rx, "y": ry, "n": 1, "area": hint.area,
                                  "frames": {reg.frame_idx}})
    return [ConeHint(c["x"], c["y"], c["area"]) for c in clusters
            if len(c["frames"]) >= min_frames]


# --------------------------------------------------------------------------
# 5. what ARE these lines? -- WFDF geometry + cone corroboration
# --------------------------------------------------------------------------

# The six field edges a detected line can plausibly BE, split by direction.
# Both groups come straight from UltimatePitchConfiguration.edges; nothing
# here invents geometry, it only says which of the known edges are mutually
# parallel on the real field (and therefore share a vanishing point in the
# image).
SIDELINE_EDGES = ((1, 7), (2, 8))
CROSS_EDGES = ((1, 2), (3, 4), (5, 6), (7, 8))   # near back, near goal, far goal, far back
GOAL_LINE_EDGES = ((3, 4), (5, 6))


def line_intersection(l1: np.ndarray, l2: np.ndarray):
    """Image-space intersection of two homogeneous lines, or None if they're
    parallel in the image (a genuinely fronto-parallel view)."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def group_by_vanishing_point(lines: list, max_angle_deg: float = 1.5):
    """Split lines into the two real-world directions of a rectangular field.

    Parallel world lines meet at one vanishing point in the image, so the
    pairwise intersections of same-direction lines cluster while
    cross-direction ones scatter. Largest cluster = family A, rest = family B.

    MEASURED, and the reason the test is ANGULAR rather than a pixel
    tolerance (torovcanyon.mp4 t=3945-4005s): a vanishing point is routinely
    thousands of px outside the frame, so any fixed pixel tolerance is either
    meaningless out there or absurdly strict near the frame. A first version
    using `max(50px, 0.15 * extent)` swept 11 of 12 tracked lines into ONE
    family -- including a line ground truth confirms is the right sideline,
    i.e. genuinely perpendicular to the rest. What is scale-free is the ANGLE
    a line would have to be rotated through, about its own midpoint, to pass
    through the candidate vanishing point.

    Returns (family_a, family_b, vp); vp is None when no grouping was found."""
    if len(lines) < 2:
        return list(lines), [], None
    homogeneous = [l.homogeneous() for l in lines]
    midpoints = [((l.p1[0] + l.p2[0]) / 2.0, (l.p1[1] + l.p2[1]) / 2.0) for l in lines]
    sin_tol = np.sin(np.radians(max_angle_deg))

    def members_for(vp):
        out = []
        for k, line in enumerate(homogeneous):
            distance = abs(line[0] * vp[0] + line[1] * vp[1] + line[2])
            reach = np.hypot(vp[0] - midpoints[k][0], vp[1] - midpoints[k][1])
            if reach < 1e-6:
                continue
            if distance / reach <= sin_tol:
                out.append(k)
        return out

    best = (0, None, None)
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = line_intersection(homogeneous[i], homogeneous[j])
            if vp is None:
                continue
            members = members_for(vp)
            if len(members) > best[0]:
                best = (len(members), vp, members)
    if best[1] is None or best[0] < 2:
        return list(lines), [], None
    members = set(best[2])
    family_a = [lines[k] for k in range(len(lines)) if k in members]
    family_b = [lines[k] for k in range(len(lines)) if k not in members]
    return family_a, family_b, best[1]


def projected_field_is_sane(calib, cfg: UltimatePitchConfiguration = None) -> bool:
    """Does this calibration project the FIELD ITSELF to a sensible shape,
    over the part of it the calibration can actually represent?

    A homography fitted from lines that are all (nearly) parallel in the world
    is under-constrained perpendicular to them: it can reproduce every
    contributing line to sub-pixel accuracy -- so reprojection error says
    "excellent" -- while folding the field into a bowtie.

    MEASURED, and the reason this is CLIPPED (torovcanyon.mp4): the first
    version tested the four outer corners at x=0 and x=100m and so **rejected
    the person's own verified-correct labelling**, because that calibration is
    LOCAL (Result 3.14) -- usable to x=54.7m, with vertices 7 and 8 sitting
    past its horizon at x=100m where their projected positions are garbage.
    Judging a shape by coordinates you cannot see is not a sanity check, it is
    a coin flip. The field is now clipped to `field_validity_region` before
    the convexity and ordering tests, so a locally-good calibration is judged
    on the part it actually describes."""
    from .calibration import field_to_pixel, field_validity_region
    cfg = cfg or UltimatePitchConfiguration()
    region = field_validity_region(calib, cfg)
    x_far = min(float(cfg.length), region["usable_x_max"] * 0.95)
    if x_far <= 0.05 * cfg.length:
        return False        # almost nothing is representable -- genuinely useless
    corners = [field_to_pixel(0.0, 0.0, calib), field_to_pixel(0.0, float(cfg.width), calib),
               field_to_pixel(x_far, float(cfg.width), calib), field_to_pixel(x_far, 0.0, calib)]
    if not all(np.isfinite(c).all() for c in corners):
        return False
    crosses = []
    for i in range(4):
        a = np.array(corners[i])
        b = np.array(corners[(i + 1) % 4])
        c = np.array(corners[(i + 2) % 4])
        crosses.append(float(np.cross(b - a, c - b)))
    if any(abs(x) < 1e-6 for x in crosses):
        return False
    if not (all(x > 0 for x in crosses) or all(x < 0 for x in crosses)):
        return False        # self-intersecting or reflex -- not a field

    # goal lines must lie between the back lines, in order, along the length --
    # but only those that fall inside the representable region
    midline_y = cfg.width / 2.0
    xs = [x for x in (0.0, float(cfg.goal_line_near), float(cfg.goal_line_far),
                      float(cfg.length)) if x <= x_far]
    if len(xs) < 2:
        return True
    pts = [np.array(field_to_pixel(x, midline_y, calib)) for x in xs]
    direction = pts[-1] - pts[0]
    norm = np.hypot(*direction)
    if norm < 1e-6:
        return False
    direction = direction / norm
    t = [float((p - pts[0]) @ direction) for p in pts]
    return all(t[i] < t[i + 1] for i in range(len(t) - 1))


def _cone_corroboration(calib, cone_hints, cfg: UltimatePitchConfiguration, tol_px: float):
    """How many of the 8 known boundary cones land on an actual cone hint.

    This is the ONLY place cone detections influence anything, and it is
    deliberately a tie-breaker on an already-fitted, already-plausible
    labelling -- never an input to the fit. Reason (measured, see
    `detect_cone_hints`): cone detections on this footage are sparse and
    small enough that fitting to them would be fitting to noise, while
    *checking* whether an independently-fitted field lands on them is free
    information that costs nothing when they're absent."""
    from .calibration import field_to_pixel
    if not cone_hints:
        return 0, 0
    hits = 0
    for vertex_index in (1, 2, 3, 4, 5, 6, 7, 8):
        fx, fy = cfg.vertices[vertex_index - 1]
        px, py = field_to_pixel(fx, fy, calib)
        if any(np.hypot(px - h.x, py - h.y) <= tol_px for h in cone_hints):
            hits += 1
    return hits, len(cone_hints)


def rank_line_labelings(lines: list, frame_w: int, frame_h: int,
                         cone_hints=(), cfg: UltimatePitchConfiguration = None,
                         image_path: str = "field_lines proposal",
                         max_fits: int = 20000, cone_tol_px: float = 40.0,
                         top_k: int = 5) -> list:
    """Propose what each detected line IS, ranked -- the WFDF-geometry step.

    For every assignment of detected lines to known field edges, fit a
    LINES-ONLY homography (`calibration.fit_calibration_with_lines` with
    `image_points=[]` -- the path Addendum 3 of
    results/calibration_finding.md validates as exactly correct, and the one
    whose point/line MIXING bug this deliberately never touches), then keep
    only assignments that survive two independent checks:

      1. `vertices_within_bounds` -- every field vertex, including ones no
         detected line constrains, must project somewhere sane. This is the
         check that catches the failure mode reprojection error structurally
         cannot see (a low-spread line fit whose far corner lands 500,000px
         off-frame -- measured, torovcanyon.mp4 t=3945s).
      2. reprojection error of the contributing points against their own
         fitted lines.

    Cone hints then break ties (`_cone_corroboration`). WFDF dimensions enter
    through `cfg.vertices` -- an assignment is only self-consistent if the
    detected lines really are 37m/18m apart in the way the labelling claims,
    and a wrong labelling shows up as one of the two failures above.

    Returns up to `top_k` dicts: {"calibration", "assignment", "max_reproj_px",
    "mean_reproj_px", "cone_hits", "n_lines", "over_determined", "score"},
    best first.

    READ `over_determined` BEFORE READING `max_reproj_px`. Four line
    correspondences determine a homography exactly (8 DOF, 2 per line), so a
    4-line proposal reproduces its own lines perfectly whatever it calls them
    -- its reprojection error is a statement about how straight the detected
    lines are, NOT about whether the labelling is correct. Only 5+ lines make
    the fit over-determined and the number discriminating.

    A returned proposal is a PROPOSAL: confirm it against the overlay image
    the way any manual calibration is confirmed, and expect to reject some.
    MEASURED, and the reason that warning is not boilerplate: on
    torovcanyon.mp4 t=3945-4005s this function's top-ranked proposal was
    wrong, because the framing contains only ONE of the two sidelines (the
    other projects entirely off-frame) -- so every candidate had to invent a
    second one, and with 4 lines nothing in the fit could tell them apart."""
    from .calibration import fit_calibration_with_lines, check_line_reprojection_error, vertices_within_bounds
    cfg = cfg or UltimatePitchConfiguration()
    if len(lines) < 4:
        return []

    family_a, family_b, _ = group_by_vanishing_point(lines)
    if len(family_a) < 2 or len(family_b) < 2:
        # REFUSE, don't guess. Lines that all share one vanishing point are
        # all parallel on the real field, and parallel lines cannot constrain
        # a homography's perpendicular direction no matter how many of them
        # there are -- the fit is free to put the rest of the field anywhere.
        # An earlier version fell back to "any line can be any edge" here and
        # duly produced a confident, precise, 720px-wrong answer (see
        # projected_field_is_sane). Returning nothing is the correct output:
        # this framing genuinely does not contain enough to calibrate from,
        # and the caller should widen the window / pan further.
        return []

    # A rectangular field has two directions and no way to know from pixels
    # alone which family is the sidelines, so try BOTH ways round.
    orientations = [(family_a, family_b), (family_b, family_a)]
    results = []
    fits = 0
    seen = set()

    class _BudgetSpent(Exception):
        pass

    try:
      for side_pool, cross_pool in orientations:
          for side_assign in permutations(range(len(side_pool)), 2):
              for n_cross in (4, 3, 2):
                  if n_cross > len(cross_pool):
                      continue
                  for cross_edges in permutations(CROSS_EDGES, n_cross):
                      for cross_assign in permutations(range(len(cross_pool)), n_cross):
                          if fits >= max_fits:
                              raise _BudgetSpent
                          chosen_lines = [side_pool[i] for i in side_assign] + \
                                         [cross_pool[i] for i in cross_assign]
                          edges = list(SIDELINE_EDGES) + list(cross_edges)
                          key = tuple(sorted((id(l), e) for l, e in zip(chosen_lines, edges)))
                          if key in seen:
                              continue
                          seen.add(key)
                          if len({id(l) for l in chosen_lines}) != len(chosen_lines):
                              continue
                          fits += 1
                          try:
                              calib = fit_calibration_with_lines(
                                  image_points=[], keypoint_indices=[], image_path=image_path,
                                  line_image_points=[list(l.points) for l in chosen_lines],
                                  line_field_edges=edges, cfg=cfg)
                          except (ValueError, np.linalg.LinAlgError):
                              continue
                          if not vertices_within_bounds(calib, frame_w, frame_h, cfg):
                              continue
                          if not projected_field_is_sane(calib, cfg):
                              continue
                          try:
                              per_line = check_line_reprojection_error(calib, cfg)
                          except ValueError:
                              continue
                          max_err = max(r["max"] for r in per_line)
                          mean_err = float(np.mean([r["mean"] for r in per_line]))
                          if not np.isfinite(max_err):
                              continue
                          calib.source_max_reprojection_px = float(max_err)
                          hits, _ = _cone_corroboration(calib, cone_hints, cfg, cone_tol_px)
                          results.append({
                              "calibration": calib,
                              "assignment": [(edge, line) for edge, line in zip(edges, chosen_lines)],
                              "max_reproj_px": float(max_err),
                              "mean_reproj_px": mean_err,
                              "cone_hits": hits,
                              "n_lines": len(chosen_lines),
                              # A homography has 8 degrees of freedom and each line
                              # correspondence supplies 2, so FOUR lines determine it
                              # exactly -- the fit can then reproduce all four to
                              # sub-pixel accuracy no matter which edges they were
                              # called, and the reported reprojection error carries
                              # ZERO information about whether the labelling is right.
                              # Only a 5th line makes the system over-determined and
                              # the error meaningful. This flag says which kind of
                              # number you are looking at; don't compare a 4-line
                              # proposal's error against anything.
                              "over_determined": len(chosen_lines) >= 5,
                              # lower is better: accuracy first, then how much of the
                              # field is actually pinned down, then cone agreement
                              "score": float(max_err) - 2.0 * len(chosen_lines) - 5.0 * hits,
                          })
    except _BudgetSpent:
        pass
    results.sort(key=lambda r: r["score"])
    return results[:top_k]


def draw_detections(bgr: np.ndarray, surface: np.ndarray = None, lines=(), cone_hints=(),
                     line_color=(0, 0, 255), cone_color=(0, 255, 255)) -> np.ndarray:
    """Sanity-check overlay: dim everything off-surface, draw what was found.
    Nothing in this module should be trusted without looking at this --
    every real fix in results/calibration_finding.md came from someone
    eyeballing an overlay, not from a metric."""
    vis = bgr.copy()
    if surface is not None:
        off = surface == 0
        vis[off] = (vis[off] * 0.35).astype(np.uint8)
    for line in lines:
        p1 = tuple(int(round(v)) for v in line.p1)
        p2 = tuple(int(round(v)) for v in line.p2)
        cv2.line(vis, p1, p2, line_color, 2)
    for hint in cone_hints:
        cv2.circle(vis, (int(round(hint.x)), int(round(hint.y))), 12, cone_color, 2)
    return vis


def segment_pan_polygon(frames: list, registrations: list, segment_id: int = None) -> list:
    """The region of the segment's shared reference coordinates that the
    camera actually swept, as a convex polygon.

    A broadcast camera almost never frames the whole field -- one frame shows
    a part of it, and the PAN shows a bigger part. Those are two different
    questions ("is this line in shot?" vs "was this line ever seen?") and
    both matter to calibration, so both are computed: each registered frame's
    four corners are mapped into reference coordinates and the convex hull of
    all of them is the pan's footprint.

    Returned in the SAME coordinates as `track_lines`' output, so it can be
    pushed through a fitted homography to answer "which part of the field did
    this shot cover" in metres."""
    by_segment = {}
    for r in registrations:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r)
    if not by_segment:
        return []
    if segment_id is None:
        segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
    # A frame warped through the horizon comes back mirrored and would drag
    # the hull to absurd coordinates -- the same defect that produced a bowtie
    # labelling canvas. mosaic.warp_is_plausible rejects those.
    from .mosaic import warp_is_plausible
    corners = []
    for reg in by_segment.get(segment_id, []):
        h, w = frames[reg.frame_idx].shape[:2]
        box = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
        pts = _transform_points(box, reg.homography)
        if not warp_is_plausible(np.asarray(pts, dtype=np.float64), float(w * h)):
            continue
        corners.extend(pts)
    if len(corners) < 3:
        return corners
    hull = cv2.convexHull(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))
    return [tuple(map(float, p)) for p in hull.reshape(-1, 2)]


# Edges that are PARALLEL on the real field. Two lines assigned to edges in
# the same group can only meet at the horizon -- never inside the region
# where they were actually observed.
PARALLEL_GROUPS = (frozenset(SIDELINE_EDGES), frozenset(CROSS_EDGES))


def no_parallel_crossings(assignment, margin_px: float = 30.0) -> bool:
    """Reject a correspondence set in which two lines assigned to
    PARALLEL field edges cross inside their own observed spans.

    Parallel world lines meet only at a vanishing point, which for a field
    being filmed from its own side is far outside the region the paint
    occupies. If the line you called the left sideline crosses the line you
    called the right sideline right where you clicked them both, one of those
    labels is wrong -- and no fit, however exact, can rescue it.

    IMPORTANT, and the reason this is `parallel` rather than the more obvious
    "edges that don't share a vertex": a sideline and a goal line DO meet on
    the field, because the goal line's endpoint lies on the sideline, even
    though they share no vertex INDEX. MEASURED on bank canvas_002: edges
    (2,8) and (3,4) cross at (958,379), inside both clicked spans, and that
    is entirely correct -- vertex 4 is exactly there. A shared-index
    adjacency test flags that as impossible and would reject correct
    labellings."""
    entries = [(tuple(edge), line) for edge, line in assignment]
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            edge_a, line_a = entries[i]
            edge_b, line_b = entries[j]
            if not any(edge_a in group and edge_b in group for group in PARALLEL_GROUPS):
                continue
            point = np.cross(line_a.homogeneous(), line_b.homogeneous())
            if abs(point[2]) < 1e-9:
                continue   # parallel in the image too -- exactly as it should be
            x, y = point[0] / point[2], point[1] / point[2]
            inside_both = True
            for line in (line_a, line_b):
                pts = np.asarray(line.points, dtype=np.float64)
                if not (pts[:, 0].min() - margin_px <= x <= pts[:, 0].max() + margin_px and
                        pts[:, 1].min() - margin_px <= y <= pts[:, 1].max() + margin_px):
                    inside_both = False
                    break
            if inside_both:
                return False
    return True


def direction_hypotheses(lines: list, max_angle_deg: float = 1.5, min_family: int = 2,
                          max_hypotheses: int = 40) -> list:
    """Every plausible way to split these lines into the field's TWO
    directions, as (family_a, family_b, vp_a, vp_b) tuples.

    MEASURED BUG this replaces (torovcanyon.mp4 t=3945-4005s):
    `group_by_vanishing_point` returns ONE split -- the largest vanishing-point
    cluster as family A, and *everything else* as family B. Family B is then a
    grab-bag, not a direction. On the test window that put the blue near
    sideline into a "family" with seven lines that do not share its vanishing
    point, so every labelling that used blue as a sideline paired it with a
    line pointing somewhere else entirely, and all 672 such candidates failed
    the geometry gates. The data was never the problem: checked directly,
    **11 of 11** candidate partners for the blue line yield a usable
    two-direction structure. The single-split assumption was throwing them
    all away.

    Here each vanishing point is a hypothesis in its own right, and a
    hypothesis PAIR is what a labelling search needs. Families must be
    disjoint -- a line cannot run in both directions at once."""
    if len(lines) < 4:
        return []
    homogeneous = [l.homogeneous() for l in lines]
    midpoints = [((l.p1[0] + l.p2[0]) / 2.0, (l.p1[1] + l.p2[1]) / 2.0) for l in lines]
    sin_tol = np.sin(np.radians(max_angle_deg))

    def members_of(vp):
        out = []
        for k, line in enumerate(homogeneous):
            distance = abs(line[0] * vp[0] + line[1] * vp[1] + line[2])
            reach = np.hypot(vp[0] - midpoints[k][0], vp[1] - midpoints[k][1])
            if reach > 1e-6 and distance / reach <= sin_tol:
                out.append(k)
        return out

    seen, clusters = set(), []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = line_intersection(homogeneous[i], homogeneous[j])
            if vp is None:
                continue
            members = tuple(members_of(vp))
            if len(members) < min_family or members in seen:
                continue
            seen.add(members)
            clusters.append((members, vp))
    # prefer bigger clusters -- more mutually-consistent lines is stronger evidence
    clusters.sort(key=lambda c: -len(c[0]))

    out = []
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            members_a, vp_a = clusters[a]
            members_b, vp_b = clusters[b]
            shared = set(members_a) & set(members_b)
            only_a = [k for k in members_a if k not in shared]
            only_b = [k for k in members_b if k not in shared]
            if len(only_a) < min_family or len(only_b) < min_family:
                continue
            out.append(([lines[k] for k in only_a], [lines[k] for k in only_b], vp_a, vp_b))
            if len(out) >= max_hypotheses:
                return out
    return out


def sideline_family_candidates(lines: list, max_angle_deg: float = 1.5,
                                min_family: int = 2, max_families: int = 16) -> list:
    """Vanishing-point clusters that could be the two SIDELINES, each paired
    with everything else as the cross-line pool.

    Replaces `direction_hypotheses` inside the labelling search, and the
    difference is what it does NOT require. Both directions of a field are
    genuinely concurrent, so demanding that the cross lines share a vanishing
    point too looks principled -- but the two cross lines a real window offers
    are often a long, well-observed one and a short endzone stub, whose fitted
    directions differ by more than a 1.5 degree concurrency test tolerates.

    MEASURED (torovcanyon.mp4 t=3945-4005s): the correct labelling -- blue
    near sideline plus the far sideline recovered from its endzone paint --
    needs cross lines [0] and [5], which never co-occur in any single
    vanishing-point cluster. Requiring it hid all four valid models. Fixing
    only the sideline family finds them, and the seven geometry gates that
    follow are what reject bad cross-line choices. That division of labour is
    the point: use concurrency where it is reliable (the pair that must be
    parallel for the fit to be conditioned at all), and use geometry for the
    rest.

    Returns [(sideline_pool, cross_pool, vp), ...], strongest cluster first."""
    if len(lines) < 4:
        return []
    homogeneous = [l.homogeneous() for l in lines]
    midpoints = [((l.p1[0] + l.p2[0]) / 2.0, (l.p1[1] + l.p2[1]) / 2.0) for l in lines]
    sin_tol = np.sin(np.radians(max_angle_deg))

    seen, out = set(), []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            vp = line_intersection(homogeneous[i], homogeneous[j])
            if vp is None:
                continue
            members = []
            for k, line in enumerate(homogeneous):
                distance = abs(line[0] * vp[0] + line[1] * vp[1] + line[2])
                reach = np.hypot(vp[0] - midpoints[k][0], vp[1] - midpoints[k][1])
                if reach > 1e-6 and distance / reach <= sin_tol:
                    members.append(k)
            key = tuple(members)
            if len(members) < min_family or key in seen:
                continue
            seen.add(key)
            out.append((members, vp))
    # Rank by EVIDENCE, not member count. MEASURED (torovcanyon.mp4
    # t=3945-4005s): sorting by size buried the cluster {blue near sideline,
    # far sideline} at rank 24 of 61 -- past any sane cutoff -- because every
    # 3-member cluster outranked it, even though it carries 431,516 support
    # against the top-ranked cluster's 208,732. The sidelines of a field are
    # its longest, most-persistently-observed lines, so total support is the
    # right proxy for "these two are plausibly the sidelines"; the number of
    # other lines that happen to share their vanishing point is not.
    out.sort(key=lambda c: -sum(lines[k].total_support_px for k in c[0]))
    result = []
    for members, vp in out[:max_families]:
        member_set = set(members)
        result.append(([lines[k] for k in members],
                        [lines[k] for k in range(len(lines)) if k not in member_set],
                        vp))
    return result


def confusable_groups(lines: list, max_angle_deg: float = 4.0,
                       max_offset_px: float = 120.0) -> list:
    """Groups of candidate lines that could be mistaken for each other.

    THE failure this project keeps hitting, and the one no automated score has
    caught. On torovcanyon.mp4 the true far sideline (white paint inside the
    endzone) has a near-parallel neighbour running behind the row of standing
    players -- the turf edge. Everything a ranking can measure prefers the
    wrong one: it is longer, straighter, higher-contrast and carries 3.4x more
    detected paint under it (Result 3.16). They diverge by only 11px where the
    real evidence is and 73px by the far side of the frame.

    MEASURED, and this is why the function reports rather than decides:
      - support / length / persistence all pick the WRONG line (Result 3.16)
      - `flanks_match` does not separate them: there is grass on both sides of
        the turf edge too (Result 3.7's check, tested here and insufficient)
      - vanishing-point structure does not separate them either: the two give
        sideline VPs of (-562,228) and (-493,286), both plausible, with the
        evidence entirely on one side of the horizon in both cases. The
        hypothesis that "the turf edge is not parallel to the sideline so a VP
        test will catch it" was checked directly and is FALSE -- they are
        near-parallel in the world.

    So this does not pick a winner. It finds the pairs a person needs to look
    at, which is the step that has resolved every one of these cases. Pair it
    with `draw_extended` to render them.

    Returns a list of index-lists, largest group first."""
    def angle_of(line):
        # works for LineSegment and TrackedLine alike -- TrackedLine has no
        # angle_deg property, and this function is used with both
        return float(np.degrees(np.arctan2(line.p2[1] - line.p1[1],
                                            line.p2[0] - line.p1[0])) % 180.0)

    groups = []
    used = set()
    for i in range(len(lines)):
        if i in used:
            continue
        group = [i]
        line_i = lines[i].homogeneous()
        pts_i = np.asarray(lines[i].points, dtype=np.float64)
        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            delta = abs(angle_of(lines[i]) - angle_of(lines[j]))
            delta = min(delta, 180.0 - delta)
            if delta > max_angle_deg:
                continue
            pts_j = np.asarray(lines[j].points, dtype=np.float64)
            # near where the evidence actually is, not at the frame edge
            offset = float(np.median(np.abs(pts_j @ line_i[:2] + line_i[2])))
            line_j = lines[j].homogeneous()
            back = float(np.median(np.abs(pts_i @ line_j[:2] + line_j[2])))
            if min(offset, back) <= max_offset_px:
                group.append(j)
        if len(group) > 1:
            groups.append(group)
            used.update(group)
    groups.sort(key=len, reverse=True)
    return groups


def draw_extended(image: np.ndarray, lines, labels=None, colours=None,
                   thickness: int = 2) -> np.ndarray:
    """Draw each line EXTENDED across the whole frame, not just over its
    observed span.

    The person asked for exactly this -- *"generate lines that go past the
    image to see where these lines all meet"* -- and it is what made the
    turf-edge confusion visible: two candidates 11px apart where the paint is
    are 73px apart by the far edge of the frame, which the eye reads instantly
    and no support metric noticed."""
    palette = [(0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 0)]
    vis = image.copy()
    h, w = vis.shape[:2]
    for n, line in enumerate(lines):
        l = line.homogeneous()
        colour = (colours[n] if colours else palette[n % len(palette)])
        pts = []
        if abs(l[1]) > abs(l[0]):
            for x in range(0, w, 2):
                y = -(l[0] * x + l[2]) / l[1]
                if 0 <= y < h:
                    pts.append((x, int(y)))
        else:
            for y in range(0, h, 2):
                x = -(l[1] * y + l[2]) / l[0]
                if 0 <= x < w:
                    pts.append((int(x), y))
        for p in pts:
            cv2.circle(vis, p, thickness, colour, -1)
        for p in line.points:
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(vis, (int(p[0]), int(p[1])), 6, colour, -1)
        if labels and pts:
            anchor = pts[len(pts) // 2]
            for c, th in (((0, 0, 0), 5), (colour, 2)):
                cv2.putText(vis, labels[n], anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, th)
    return vis


def transform_lines(lines, T: np.ndarray) -> list:
    """Re-express TrackedLines in another 2D frame via the 3x3 `T`.

    Used to put candidates found in a segment's reference coordinates onto
    `mosaic.segment_canvas`'s wide pan image, so the person labelling can see
    every candidate at once -- including the ones that were never in the same
    frame as each other."""
    out = []
    for t in lines:
        p1, p2 = _transform_points([t.p1, t.p2], T)
        out.append(TrackedLine(p1=p1, p2=p2,
                                points=_transform_points(t.points, T) if t.points else [],
                                frame_indices=list(t.frame_indices),
                                n_frames=t.n_frames,
                                total_support_px=t.total_support_px,
                                rms_fit_px=t.rms_fit_px, colour=t.colour))
    return out


# --------------------------------------------------------------------------
# curved markings (soccer circles, penalty arcs) masquerading as field lines
# --------------------------------------------------------------------------

def _conic_from_5(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    A = np.vstack([x * x, x * y, y * y, x, y, np.ones_like(x)]).T
    return np.linalg.svd(A)[2][-1]


def _conic_params(c):
    """(centre, sorted semi-axes) for an ellipse, or None if `c` isn't one."""
    A, B, C, D, E, F = c
    M = np.array([[2 * A, B], [B, 2 * C]])
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    centre = np.linalg.solve(M, [-D, -E])
    Fp = F + (D * centre[0] + E * centre[1]) / 2.0
    lam = np.linalg.eigvalsh(np.array([[A, B / 2.0], [B / 2.0, C]]))
    if np.any(lam * (-Fp) <= 0):
        return None
    return centre, np.sort(np.sqrt(-Fp / lam))


def conic_distance(conic: np.ndarray, points) -> np.ndarray:
    """Sampson (first-order geometric) distance from each point to the conic."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = P[:, 0], P[:, 1]
    A, B, C, D, E, F = conic
    v = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    grad = np.hypot(2 * A * x + B * y + D, B * x + 2 * C * y + E)
    return np.abs(v) / np.maximum(grad, 1e-12)


def detect_conics(bgr: np.ndarray, surface: np.ndarray = None, max_conics: int = 1,
                   tol_px: float = 3.0, iterations: int = 40000,
                   min_inliers: int = 80, min_coverage_deg: float = 120.0,
                   max_axis_ratio: float = 4.0, seed: int = 0) -> list:
    """Find CURVED white markings -- soccer centre circles, penalty arcs --
    so straight-line candidates lying on them can be discounted.

    Returns a list of dicts: `conic` (6 coefficients, image pixels), `centre`,
    `semi_axes`, `coverage_deg`, `n_inliers`.

    MEASURED, and the reason this fits PIXELS rather than TrackedLine points
    (results/field_lines_finding.md Result 3.26): the tracker shatters a
    circle into short chords that are each individually straight, so there is
    no curvature left inside any one line to detect -- a per-line bow test
    scores known circle chords at 1.6-7.5px while genuine straight lines
    score up to 33px, i.e. it is worse than useless. Fitting the conic to the
    sparse chord endpoints fails differently: with points drawn from several
    straight lines, RANSAC threads a degenerate sliver ellipse through them
    and "explains" 92% of a real straight line. Dense paint pixels are the
    only input on which the fit locks onto the actual circle.

    `max_conics` defaults to 1 because MEASURED: the first conic on
    torovcanyon t=3600 is the real centre circle (1251 inliers, 250 deg of
    coverage) and flags only genuine chords; asking for a SECOND returns a
    121-deg arc that flags straight lines 4, 15 and 17 at 63-75%. Raise it
    only on footage that actually has two curved markings in shot, and check
    the result.

    The three constraints all earned their place against those failures:
    a genuine ellipse (negative discriminant), a bounded axis ratio so a
    sliver cannot pass, and wide ANGULAR COVERAGE so a short arc-like stretch
    of a straight line cannot pass."""
    rng = np.random.default_rng(seed)
    if surface is None:
        surface = playing_surface_mask(bgr) > 0
    else:
        surface = surface > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    if not surface.any():
        return []
    turf = float(np.median(V[surface]))
    white = ((V > turf + 28) & (S < 110) & surface).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(white > 0)
    if len(xs) < 200:
        return []
    pick = rng.choice(len(xs), min(6000, len(xs)), replace=False)
    P = np.stack([xs[pick], ys[pick]], 1).astype(np.float64)
    scale = float(np.abs(P).max()) or 1.0
    Q = P / scale
    tol = tol_px / scale
    found, live = [], np.ones(len(Q), bool)
    for _ in range(max_conics):
        best = (0, None, None, None)
        idx = np.where(live)[0]
        if len(idx) < 200:
            break
        for _ in range(iterations):
            s = rng.choice(idx, 5, replace=False)
            c = _conic_from_5(Q[s])
            if c[1] ** 2 - 4 * c[0] * c[2] >= 0:      # must be an ellipse
                continue
            pr = _conic_params(c)
            if pr is None:
                continue
            centre, axes = pr
            if axes[1] / axes[0] > max_axis_ratio:     # no slivers
                continue
            if not (80 / scale < axes[0] < 2500 / scale):
                continue
            inl = idx[conic_distance(c, Q[idx]) < tol]
            if len(inl) < min_inliers:
                continue
            ang = np.sort(np.arctan2(Q[inl, 1] - centre[1], Q[inl, 0] - centre[0]))
            gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
            coverage = 2 * np.pi - gaps.max()
            if coverage < np.radians(min_coverage_deg):
                continue
            if len(inl) > best[0]:
                best = (len(inl), c, inl, (centre * scale, axes * scale, np.degrees(coverage)))
        if best[1] is None:
            break
        n, c, inl, (centre, axes, coverage) = best
        # rescale the conic so it applies to raw pixel coordinates
        Sm = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        A, B, C, D, E, F = c
        M = np.array([[A, B / 2, D / 2], [B / 2, C, E / 2], [D / 2, E / 2, F]])
        M = Sm.T @ M @ Sm
        found.append({"conic": np.array([M[0, 0], 2 * M[0, 1], M[1, 1],
                                          2 * M[0, 2], 2 * M[1, 2], M[2, 2]]),
                       "centre": tuple(map(float, centre)),
                       "semi_axes": tuple(map(float, axes)),
                       "coverage_deg": float(coverage), "n_inliers": int(n)})
        live[inl] = False
    return found


def arc_fractions(lines, conics, tol_px: float = 3.0) -> list:
    """For each TrackedLine, the largest fraction of its points lying on any
    detected conic. High = the "line" is a chord of a curved marking.

    FLAGS, never drops. On torovcanyon t=3600 this reaches 35% for the
    best-supported circle chord and 0% for every genuine straight line, but
    chords carrying only 4-6 points score 0% too -- so a low value is NOT
    evidence a line is straight, and using it to delete candidates would
    silently discard real lines. See Result 3.26."""
    out = []
    for t in lines:
        best = 0.0
        pts = np.asarray(t.points, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            for k in conics:
                best = max(best, float((conic_distance(k["conic"], pts) < tol_px).mean()))
        out.append(best)
    return out


def split_merged_lines(lines, rms_threshold: float = 4.0, inlier_px: float = 3.0,
                        min_points: int = 5, max_parts: int = 4, seed: int = 0) -> list:
    """Split any TrackedLine whose points do NOT fit one line into the
    several lines they actually are.

    `_consolidate_clusters` merges collinear-ish evidence, and on a busy
    multi-sport pitch it sometimes merges genuinely different markings that
    happen to lie near one another -- a real field line and the turf edge
    running a few pixels behind it. The merged fit then sits between them,
    matching neither, and its high `rms_fit_px` is the tell.

    MEASURED (results/field_lines_finding.md Result 3.27), and the reason
    this exists: on torovcanyon t=3600 the person said the far sideline was
    "almost detected as number 6". Line 6 had the worst rms of all 29
    candidates (9.63px over 26 points). RANSAC-splitting it yields three
    sub-lines at 1.02, 0.26 and 0.20px rms -- and the last passes 0.8px from
    the endzone corner the person had independently pointed at. The sideline
    WAS detected; it was averaged together with two neighbours and so could
    never be selected.

    A line already fitting under `rms_threshold` is returned untouched, so
    this only ever acts where the evidence itself says one line is wrong."""
    rng = np.random.default_rng(seed)
    out = []
    for t in lines:
        P = np.asarray(t.points, dtype=np.float64).reshape(-1, 2)
        if t.rms_fit_px <= rms_threshold or len(P) < 2 * min_points:
            out.append(t)
            continue
        remaining = np.ones(len(P), bool)
        parts = []
        for _ in range(max_parts):
            idx = np.where(remaining)[0]
            if len(idx) < min_points:
                break
            best = (0, None)
            for _ in range(4000):
                a, b = rng.choice(idx, 2, replace=False)
                d = P[b] - P[a]
                L = np.linalg.norm(d)
                if L < 1e-6:
                    continue
                n = np.array([-d[1], d[0]]) / L
                inl = idx[np.abs((P[idx] - P[a]) @ n) < inlier_px]
                if len(inl) > best[0]:
                    best = (len(inl), inl)
            if best[1] is None or len(best[1]) < min_points:
                break
            inl = best[1]
            line = _fit_line_tls([tuple(q) for q in P[inl]])
            resid = np.abs(P[inl] @ np.array([line[0], line[1]]) + line[2])
            p1, p2 = _extent_along([tuple(q) for q in P[inl]], line)
            frames = {t.frame_indices[k] for k in inl} if t.frame_indices else set()
            parts.append(TrackedLine(
                p1=p1, p2=p2, points=[tuple(q) for q in P[inl]],
                frame_indices=[t.frame_indices[k] for k in inl] if t.frame_indices else [],
                n_frames=len(frames) or t.n_frames,
                total_support_px=int(round(t.total_support_px * len(inl) / len(P))),
                rms_fit_px=float(np.sqrt((resid ** 2).mean())), colour=t.colour))
            remaining[inl] = False
        # only accept the split if it actually explains the points better
        if len(parts) >= 2 and max(p.rms_fit_px for p in parts) < t.rms_fit_px:
            out.extend(parts)
        else:
            out.append(t)
    return out


# Cross-field markings exist ONLY at these distances along the field (metres):
# back line, goal line, halfway, goal line, back line. There is nothing between
# them -- which is what makes a detected "line" at 30m proof of an error.
LEGAL_CROSS_POSITIONS_M = (0.0, 18.0, 50.0, 82.0, 100.0)


def cross_field_candidates(lines, goal_idx: int, surface: np.ndarray = None,
                            offset=(0.0, 0.0), max_sep_deg: float = 25.0,
                            min_gap_px: float = 120.0, min_crossing_px: float = 1500.0,
                            surface_tol_px: float = 40.0, region=None) -> dict:
    """Which candidates could be a SECOND cross-field line (end, halfway, or
    the far goal line), given one already identified as the goal line.

    Four rules, all of them stated by the person after watching automated
    scores pass one wrong answer after another. Each is returned separately so
    a rejection can be explained rather than just applied:

      on_field       -- distance to the playing surface, NOT membership in it.
                        MEASURED (Result 3.29): a membership test rejected the
                        person's own confirmed far sideline at 50%, because a
                        sideline IS the boundary of the surface. Distance keeps
                        it (60%) while still excluding treeline and crowd.
      cross_direction-- within `max_sep_deg` of the goal line.
      parallel       -- two lines PARALLEL on the field meet only at the
                        vanishing point. *"they shouldn't be angled enough to
                        be touching the goal line if its an end line"*. A
                        candidate crossing the goal line near the field is
                        therefore not parallel to it, whatever its angle.
      distinct       -- and it must not BE the goal line. MEASURED: the three
                        candidates that passed every other rule sat 5px, 30px
                        and 37px from the goal line; assigning one of them to
                        an edge 32m away collapsed the whole field to a sliver
                        while scoring 3/3 gates and 100% validity.

    `region` is a polygon (segment reference px) used for the parallel test;
    defaults to the convex hull of every line's points."""
    goal = lines[goal_idx]
    hg = goal.homogeneous()
    gmid = (np.asarray(goal.p1) + np.asarray(goal.p2)) / 2.0
    gang = np.degrees(np.arctan2(goal.p2[1] - goal.p1[1], goal.p2[0] - goal.p1[0])) % 180

    if region is None:
        allp = np.asarray([p for t in lines for p in t.points], dtype=np.float32)
        region = cv2.convexHull(allp.reshape(-1, 1, 2)).reshape(-1, 2) if len(allp) >= 3 else None
    region = np.asarray(region, dtype=np.float32) if region is not None else None

    dist = None
    if surface is not None:
        dist = cv2.distanceTransform((surface == 0).astype(np.uint8), cv2.DIST_L2, 5)

    out = {}
    for i, t in enumerate(lines):
        if i == goal_idx:
            continue
        why = []
        if dist is not None and t.points:
            pts = np.asarray(t.points) + np.asarray(offset)
            v = [dist[int(y), int(x)] for x, y in pts
                 if 0 <= int(y) < dist.shape[0] and 0 <= int(x) < dist.shape[1]]
            if v and float(np.mean(np.asarray(v) <= surface_tol_px)) < 0.6:
                why.append("not on the field")
        ang = np.degrees(np.arctan2(t.p2[1] - t.p1[1], t.p2[0] - t.p1[0])) % 180
        sep = min(abs(ang - gang), 180 - abs(ang - gang))
        if sep > max_sep_deg:
            why.append(f"runs along the field, not across it ({sep:.0f} deg)")
        h = t.homogeneous()
        gap = abs(h[0] * gmid[0] + h[1] * gmid[1] + h[2])
        if gap < min_gap_px:
            why.append(f"only {gap:.0f}px from the goal line -- the same line")
        X = line_intersection(hg, h)
        if X is None:
            why.append("degenerate intersection with the goal line")
        elif region is not None:
            d = cv2.pointPolygonTest(region, (float(X[0]), float(X[1])), True)
            if d >= 0 or abs(d) < min_crossing_px:
                why.append("crosses the goal line at/near the field, so not parallel to it")
        out[i] = why
    return out
