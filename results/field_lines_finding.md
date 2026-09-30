# Scene-Aware Field Line Tracking — Finding (measured on real footage)

Prompted by the milestone "fix scene aware field line tracking / use cones
and other lines as hints / be aware of camera pans / goal line (endzone)".

`results/calibration_finding.md` had twice measured automated detection as
unusable on this footage, and named what a working version would need:
"overlay + non-grass masking before any edge detection, orientation
filtering, and probably a dedicated cone detector". `frisbee_analysis/
field_lines.py` is that, plus a piece that note didn't call for and which
turned out to matter more than any of them: **tracking the detections across
the pan** in the shared coordinate system `mosaic.register_sequence` already
provides.

## What was built

1. `playing_surface_mask` — green **and low-texture**, largest component
   grown across thin splits. The texture term is what separates turf from
   the treeline (both are green); the growth step is what survives shadows
   and dark painted seams cutting the surface into pieces.
2. `line_pixel_mask` — a **multi-scale local-contrast ridge test**, not
   Canny and not a global threshold.
3. `detect_cone_hints` — orange blobs on the surface with **YOLO person
   boxes subtracted**.
4. `track_lines` / `track_cone_hints` — lift every frame's detections into
   the segment's shared reference coordinates and keep what repeats.
5. `projected_field_is_sane` — a geometric validity check on the *fitted
   field*, which found a real bug (below).
6. `rank_line_labelings` — enumerate what the lines could BE, gated by WFDF
   geometry, tie-broken by cone hints.

`run_field_lines_test.py` measures all of it against ground truth;
`run_field_lines_propose.py` is the usable tool built on it.

## Result 1: scene-awareness works, and the earlier failures are explained

Same Hough settings, one frame (`torovcanyon.mp4` t=3945s):

| | segments |
|---|---|
| naive Canny + Hough, whole frame | **888** |
| scene-aware (surface mask + ridge test), merged | **6** |

The playing surface comes out at 59.8% of the frame with its top edge at
y=353–404 — it ends at the real far touchline and excludes the treeline,
powerlines, sky, hedge, spectator tents, adjacent fields and the scoreboard.
That single restriction is the whole difference; the original finding's "89
lines, nearly all treeline/seam artifacts" was a detector with no idea which
pixels were even field.

**The "circular without a calibration" objection was wrong.** The earlier
note dismissed field-region-only detection because it seemed to need a
homography first. It doesn't: colour + texture + connectivity recover the
surface with no geometry at all, which is exactly what breaks the circle.

Two measured details that each cost a rewrite:

- **Trees are green.** Texture separates them (turf local_std 1.6–4.4, crowd
  row 20.6), but a *silhouetted* treeline measures smooth too — connectivity
  is what actually excludes it, and only if the morphology stays small. A
  close kernel ≥15px at 1920 bridges field→treeline (surface 48%→79%, top
  edge y=353→0). Bridging shadow seams therefore has to be done by growing
  the component across small gaps, not by a bigger close.
- **A global threshold cannot work on half-sunlit footage.** The first
  version (single-scale top-hat + Otsu) found **one** line in the whole
  frame: Otsu locked onto the brightest sunlit paint and discarded every
  shaded line, and one kernel size is blind to a line that is 2px wide at
  the far touchline and 20px wide in the foreground. Multi-scale, with a
  threshold that scales with local brightness: 1 → 8 lines, same frame.

## Result 2: tracking across the pan recovers the real lines

`torovcanyon.mp4` t=3945–4005s, 60 frames at 1fps, one registered segment.
Per-frame the detector finds 5–12 lines (median 7); pooled across the pan and
filtered to those seen in ≥3 frames, **12 tracked lines**.

Ground truth is the human's own clicked points from bank `canvas_002`,
mapped into the same coordinates — deliberately **not** routed through that
calibration's homography (see Result 3 for why that matters):

| human line | nearest tracked line | distance | angle error |
|---|---|---|---|
| `(2,8)` right sideline | [3] | 33.0px | 3.0° |
| `(3,4)` **NEAR GOAL LINE** | [0] | **4.9px** | 0.5° |
| `(1,2)` near back line | [9] | **7.4px** | 1.5° |
| `(1,7)` left sideline (BLUE paint) | [2] | **6.9px** | 0.5° |

All four recovered to within a few pixels and under 1.5°, with **no
clicking**. The `(1,7)` row is the blue near sideline, and it only appears
here because of the chroma channel added in Result 3.5 — with a
brightness-only detector it read 56.8px / 43.8° away, i.e. not found at all.

## Result 3: a real bug — three of four bank calibrations fold the field into a bowtie

`projected_field_is_sane` checks something neither existing check can: the
four outer corners must form a convex quadrilateral traversed consistently,
and the goal lines must fall between the back lines along the field's
length. Validated on synthetic cameras (four plausible sideline/corner
viewpoints → True; the same field with two corner labels swapped → False),
then run over the bank:

| calibration | source error | `vertices_within_bounds` | `projected_field_is_sane` |
|---|---|---|---|
| `canvas_000` (videoplayback) | 2.06px | True | **True** |
| `canvas_002` (torovcanyon) | 2.74px | True | **False** |
| `canvas_003` (torovcanyon, same fit) | 2.74px | True | **False** |
| `canvas_004` (torovcanyon) | 54.98px | True | **False** |
| `outputs/calibration.json` | 54.98px | True | **False** |

`canvas_002` projects vertex 7 to (-1017,-157) while vertex 8, its partner
on the same back line, lands at (8050,1076); the field self-intersects.
Along the left sideline the projection reverses direction (v1 → v11 → v7
goes down, then back up past where it started).

**Both existing checks pass it.** Worst-line reprojection error is 2.74px,
because reprojection error only compares clicked points to the line fitted
through those same points. `vertices_within_bounds` is True, because a
bowtie's vertices are all in bounds — just in an impossible order.

**Three of the four bank canvases are serving a broken calibration**,
including the one `results/calibration_finding.md` Addendum 4.6 records as
the verified-good redo. Anything downstream that used it — `is_in_field`
filtering and the speed numbers in `results/metrics_finding.md` — is
unreliable for `torovcanyon.mp4`.

**ACTION NEEDED — PARTLY RETRACTED, see Result 3.14**: canvas_002/003 turn
out to be valid LOCAL calibrations (usable to x=54.7m) rather than broken;
only canvas_004 needs redoing. The check is wired into `run_calibrate.py` (refuses to bank a bowtie),
`calibration_bank.add_entry` (a bowtie is never "the plausible one" in a
merge comparison), and `run_calibrate_auto.py` (warns that the bank canvas
itself is at fault, since every match against it inherits the same fold).

### CORRECTION: the cause is NOT "a line clicked on bare turf"

The first version of this document claimed the `(1,7)` clicks sat on plain
unmarked turf. **That was wrong, and the person caught it** by pointing out
what the field is actually marked with: "the close sideline is a blue line
and the far sideline can be detected by a white line but the white line is
only painted on the endzone."

Those clicks are on the BLUE near sideline. The detector at the time only
looked for brightness ridges, so it was structurally incapable of seeing a
line that is the same brightness as the turf (Result 3.5) — it reported
nothing there, and that absence was misread as evidence the human had
clicked on grass. Once the chroma channel exists, the same clicks match a
tracked line at **6.9px / 0.5°**. All four of the human's clicked lines are
on real paint.

The bowtie is still real. Re-fitting the SAME four correspondences from the
tracker's own, much longer-baseline versions of those lines still produces
`projected_field_is_sane = False` and `vertices_within_bounds = False`, so
it isn't merely a short-baseline precision problem either. At least one of
the four labels is geometrically inconsistent with the others — **which one
is not yet identified**, and this document is not going to guess a second
time.

What the vanishing-point grouping says, for whoever picks this up: the blue
line and the white line clicked as `(2,8)` land in the SAME family (parallel
on the real field, consistent with them being the two sidelines), while the
lines clicked as `(3,4)` and `(1,2)` land in the other. That much is
self-consistent, which is why the inconsistency is not obvious.

**Method note worth keeping.** The original wrong diagnosis came from
trusting a detector's silence as evidence of absence. A detector that cannot
represent a thing will always report that the thing is not there, and that
is not the same as it not being there. The person's domain knowledge, not
any measurement, is what corrected it.

## Result 3.5: paint is not always brighter than grass

Prompted directly by the person's description of the field. Measured on
`torovcanyon.mp4` t=3945s, walking along the near sideline:

| | line pixels | turf 30px away |
|---|---|---|
| HSV V (brightness) | 56–68 | 52–71 |
| HSV hue | 84–101 | 36–42 |
| Lab chroma distance from local turf colour | 22–26 | 0.4–1.3 |

**There is no brightness step to detect.** A luminance ridge test is
structurally blind to this line, no matter how it is tuned. Adding a second
ridge channel on Lab chroma distance from the *local* turf colour finds it
immediately, and chroma is also less shadow-sensitive than luminance, so
the two channels are OR'd rather than one replacing the other.

Two further bugs surfaced while wiring that up, both mine, both measured:

1. **Elongation was computed from the bounding box**, which is near-square
   for any diagonal line however thin. The blue sideline — a 7,090px
   component — scored bbox-elongation **1.17** and was discarded *after*
   being correctly detected. Second-moment (eigenvalue-ratio) elongation
   scores the same pixels **32.4**. Real lines score 19–36 by that measure,
   players and the centre circle 1.9–4.0, so the threshold now sits in a
   measured gap rather than on a guess.
2. **OR-ing the two channels before component filtering made adding a
   channel DESTROY detections.** On t=3975s, chroma noise touching a clean
   white line fused them into one component that then failed the elongation
   test, and the frame's two longest lines (1,120px and 792px) vanished the
   moment chroma was enabled. Each channel is now shape-filtered separately
   and the survivors unioned. Per-frame line counts on the test window went
   from median 7 to median 10 after both fixes.

Lines now carry their own paint colour (`classify_paint_colour`), reported
in the candidates table, because on a multi-sport field that is how a person
tells them apart — and mislabelling is the failure mode that produced the
bowtie above.

## Result 3.6: a line painted only in the endzones, and foreground drift

The person: "the white line is only painted on the endzone... in the middle
of the field the far sideline does not exist." Two consequences:

- **Merging has to tolerate a gap.** Two painted stretches of one line
  separated by the whole midfield cannot be joined under a flat 14px
  perpendicular tolerance: 2.5° of angle uncertainty over that distance is
  far more than 14px. `_offset_tolerance` is now flat inside the span a
  group already covers and widens into the extrapolation beyond it by the
  cone the angle tolerance implies.
- **Foreground lines drift between frames.** The blue sideline still comes
  out as two candidates, 28–46px apart perpendicular at 1.8° to each other.
  That is not a merging failure — it is registration drift. The line is in
  the foreground, where `mosaic.py`'s frame-to-frame chaining is weakest
  (its features are mostly background scenery). Deliberately NOT merged
  away: a tolerance wide enough to join them would start joining genuinely
  different lines. `run_field_lines_propose.py` flags same-colour
  near-parallel candidate pairs instead, so the person picks one.

A consolidation pass (`_consolidate_clusters`) does re-test finished
clusters against each other with their full pooled points, which fixes the
cases where greedy incremental accumulation compared a finished line against
a stub.

## Result 3.7: a painted line has the same surface on both sides

Adding the chroma channel introduced a false positive worth recording,
because the fix is the sharper idea in this whole module. On `ojuc.mp4`
t=1580s -- a bare grass field with **no paint at all** -- the detector
started reporting a confident 1,636px "line". It was the row of white
spectator tents along the grass edge: genuinely thin, genuinely bright,
genuinely elongated, a real ridge by every local test.

The obvious fix was tried first and MEASURED WRONG: requiring a detection to
lie mostly inside the un-dilated surface mask deleted the sidelines, because
a sideline **is** the surface's own boundary (torovcanyon.mp4 t=3945s: 9
lines → 4, losing both long white ones).

What separates the two cases is not where the line sits but whether its two
sides MATCH. A painted marking lies ON a uniform surface, so grass to its
left looks like grass to its right; a scene boundary has different worlds on
each side. `flanks_match` samples Lab colour a line-width out on both sides
and compares the medians. Lab rather than RGB so a line running through
shadow on one side isn't rejected.

| | before | after |
|---|---|---|
| `ojuc.mp4` t=1580s (no paint exists) | 1 line | **0 lines** |
| `ojuc.mp4` t=1700s | 2 lines | 1 line |
| `torovcanyon.mp4` t=3945s | 9 lines | 6 lines |

and all four of the human's clicked lines are still recovered on the tracked
window afterwards (6.9 / 33.0 / 4.9 / 7.4 px), so the removed lines were the
confounds, not the signal. One false positive survives on `ojuc.mp4`
t=1700s -- this narrows the problem, it does not close it.

## Result 3.8: the output is six connected lines, not a list of fragments

Person's requirement: *"the only field lines that should be generated are
sidelines, end lines, and goal lines, six total, and they should all be able
to connect to create the field, make estimations on where they should be
based on existing lines, cones, perceived depth and knowledge of how big
field should be."*

`frisbee_analysis/field_model.py` is that. A `FieldModel` always carries
exactly six `ModelLine`s -- two sidelines, two end lines, two goal lines --
each drawn corner to corner and tagged:

- `observed`  — a tracked detection supports this line
- `estimated` — no paint was detectable, so its position is computed from
  the fitted homography plus WFDF Appendix A1.2.2 dimensions

They connect by construction: all six come from one homography applied to
`ultimate_config.py`'s vertex list, so there is no way for them to disagree.
On this field that matters — half its ultimate boundary isn't painted at
all, so the far sideline across midfield can only ever be `estimated`.

**Ranking uses evidence the fit did not consume.** With four lines the fit
is exact and reprojection error cannot rank labellings, so ranking is by:
how much of the detected paint the resulting six lines account for
(support-weighted), how many of the eight boundary cones land on a cone
hint, and how many of the six are directly observed. Reprojection error is
a reject gate (>30px, only meaningful when the fit is over-determined),
not a ranking term.

**Order preservation is what made the search work.** A homography preserves
the order in which parallel lines cross a transversal, so the sequence in
which the near end line, near goal line, far goal line and far end line
cross a sideline is the same in the image as on the field. Enumerating only
order-consistent labellings is not a heuristic — the alternatives are
geometrically impossible — and it collapsed the search from ~40,000
permutations per sideline pair to ~140. Effect measured on
`torovcanyon.mp4` t=3945-4005s:

| | before ordering constraint | after |
|---|---|---|
| blue line labelled | "near end line" | **left sideline** |
| best model's worst residual | 215.3px | **29.4px** |
| detected paint explained | 79% | **87%** |
| lines observed | 5/6 (inflated, see below) | 5/6 |

Two supporting fixes, both measured:

- **One detection per edge.** Without exclusivity a single 21,047-support
  detection was counted as evidence for BOTH the far end line and the far
  goal line — lines 18m apart — inflating "observed" counts for a labelling
  that had really seen four. Matching is now greedy by residual, and neither
  an edge nor a detection can be used twice.
- **An optional `colour_prior`.** The caller can say what colour an edge is
  on their field (`{(1,7): "blue"}`); agreement adds to the score. Never
  geometric, and never required — it only lets ranking use what the person
  already knows.

**Still not solved: the top-ranked model is not reliably correct.** On the
test window it puts the blue line on a sideline (right, residual 1.5px) and
fits five of six lines at 1.5-8.8px, but the resulting field is visibly
misplaced when you look at it — it is fitting the soccer/lacrosse markings
the ultimate field is laid over. The `--assign` path, where the person names
the lines and this module supplies the estimation, is the trustworthy one.

**On depth**: monocular depth (DepthAnything) is deliberately not used --
held for a later milestone, and redundant here anyway. Two vanishing points
of a rectangular field fix the ground plane's orientation and WFDF's
dimensions fix its scale; depth becomes interesting for things OFF the
ground plane (player height, disc altitude), not for where a line on the
grass is.

## Result 3.9: the camera almost never sees the whole field

Person's observation on the model image: *"field line calibration needs to be
aware that camera will almost never be looking at the whole field, only part
of it... only both sidelines and one endzone including back line and goal
line is visible. reference to where the camera shot is in the mosaic to know
how much of the field is visible."*

Implemented as three things:

- `field_lines.segment_pan_polygon` — every registered frame's four corners
  mapped into the segment's shared reference coordinates, convex-hulled. That
  is the footprint the camera actually swept, in the same coordinates the
  tracked lines live in.
- `field_model.visible_extent` — pushes the frame corners AND that pan
  footprint through the fitted homography to answer, in metres, which part of
  the field is in view. Two different questions are kept separate: "is this
  line in shot?" and "was this line ever seen during the pan?"
- every `ModelLine` now carries `visibility`: `in_frame`, `in_pan`, or
  `unseen`.

On the test window it now reports, unprompted:

```
pan covers x=-62..56m of 0..100m (56% of the length), y=-26..73m of 0..37m;
both sidelines in view; endzones in view: near endzone
  left sideline    observed  in_frame
  right sideline   observed  in_frame
  near end line    observed  in_frame
  far end line     estimated unseen
  near goal line   observed  in_frame
  far goal line    estimated unseen
```

which is exactly the situation the person described.

**Two new rejection gates, both grounded in the field's known size:**

- `extent_supports_observation` — a line the camera was never pointed at
  cannot have been observed, however well a detection appears to sit on it.
  Without this, nothing stops a fit from shrinking the field until an
  unrelated painted line lands where a far boundary "should" be.
- `extent_is_plausible` — the implied coverage has to be physically sensible.
  MEASURED: a wrong labelling claimed its pan swept y = -110m..54m across a
  field that is 37m wide (164m of width) while covering only 14% of the
  length. A camera filming a 37m-wide field does see past the sidelines, but
  not four field-widths past. With the gate in, the surviving models report
  56% of the length and 99m of width — sane for a broadcast shot.

**And a scoring result worth recording, because it is counter-intuitive:**
ranking by "how much detected paint do these six lines account for" actively
rewards the WRONG answer. A wrongly-shrunken field explains *more* paint, not
less, because more of the soccer/lacrosse markings the ultimate field is laid
over fall inside it — the implausible models scored 80–87% while the
geometrically sensible one scored 15%. Its weight is now a quarter of what it
was, and the load is carried by evidence a wrong scale cannot fake (cone
hits, how many of the six were actually visible, paint colour).

### Open, with a concrete lead

With these gates in place, **no surviving model puts the blue line on a
sideline** — which contradicts what the person says their field is. Counted
directly: of every candidate that pairs the blue line with another
same-family line as the two sidelines, 76 fail `vertices_within_bounds`, 192
fail `projected_field_is_sane`, 154 fail `extent_is_plausible`, 82 claim to
observe a line the pan never covered, and **0 survive**.

So one of these is true, and this is where to pick it up:
1. the candidate set is missing the real far sideline — plausible, since on
   this field it is painted only inside the endzones, and the near endzone's
   short white stretch may be one of the weak, low-persistence candidates
   rather than a tracked line; or
2. one of the gates is too strict for a fit that is right but imprecise; or
3. the blue line's tracked geometry is degraded by the foreground
   registration drift measured in Result 3.6 (it arrived as two clusters
   28–46px apart).

Nothing here should be tuned until it is known which. The `--assign` path,
where the person names the lines and the model supplies estimation, extent
and the six-line output, remains the trustworthy one.

## Result 3.10: the actual defect in canvas_002 — the horizon cuts through the field

The person went and measured the open item above rather than accepting it,
and was right that it was mis-framed: "which single label is wrong" is not
identifiable and was never going to terminate. Holding all four clicked
geometries fixed and enumerating every valid relabelling (288 fits), **80
pass `projected_field_is_sane`, 20 of them keeping the sidelines exactly as
clicked.** Geometry alone cannot pin down the bad label.

Their second observation is the important one: every sideline-preserving sane
fit puts the field somewhere the clicks are not, and `vertices_within_bounds`
at `margin=8.0` cannot see it — it only catches a field that EXPLODES, never
one that COLLAPSES.

### Their duplicate-line hypothesis: MEASURED, and refuted

Proposed cause: `(1,2)` and `(2,8)` are the same far touchline clicked in two
halves — they meet at 10.5° exactly at their own clicking seam (x=725, 14px
from where one span ends and the other begins), drawn from two different
frames of a pan chained through the mosaic. The signature is genuinely
suggestive and the spans check out exactly ((2,8) from frame 0, x 709→1269;
(1,2) from frame 12, x −305→677).

It does not survive measurement. Registering frames 0 and 12 **directly**
and comparing against the chained homography:

| frame-12 pixel | chained → frame 0 | direct → frame 0 | divergence |
|---|---|---|---|
| (200,400) | (−195.3, 424.5) | (−198.2, 424.9) | **2.9px** |
| (950,380) | (705.7, 382.3) | (704.6, 383.1) | **1.3px** |
| (1550,420) | (1274.7, 407.3) | (1273.9, 408.7) | **1.6px** |
| (900,900) | (681.3, 914.2) | (676.5, 920.7) | 8.1px |

Not ~100px. Re-deriving `(1,2)`'s line through the direct registration
instead of the chain moves its angle by **0.02°** (175.14 → 175.16), and the
`(2,8)` clicks' distances from it are unchanged (1.9/44.5/100.4 chained vs
2.5/43.8/99.6 direct). The 10.5° is real geometry, not drift. Twelve frames
into this segment's chain, registration is accurate to a couple of pixels.

(One methodological note, because it cost a wrong test first: "are these two
click sets collinear?" cannot distinguish the two hypotheses at all.
Collinearity is projectively invariant, so applying any homography to both
sets preserves the answer. The chained-vs-direct comparison is the test that
actually separates them.)

### What IS wrong: part of the field is behind the camera

`calibration.field_in_front_of_camera`. Under field→image each vertex has a
homogeneous w. A sign change between vertices means the image of the line at
infinity cuts through the playing surface — the fitted camera puts part of
the field behind itself. No real camera filming a field can do that.

| calibration | w signs across the 8 boundary vertices | verdict |
|---|---|---|
| `canvas_000` (verified good) | `+ + + + + + + +` | **True** |
| `canvas_002` / `canvas_003` | `- - - - + - + -` | **False** |
| `canvas_004` (known bad, 55px) | `+ - + - - - - -` | **False** |
| five synthetic sideline cameras | all one sign | **True** |

canvas_002 puts vertices 5 and 7 — the far end of the field at y=0 — on the
far side of the horizon from everything else. That is why its clicked
stretches fit to 2.7px while its far half projects nonsensically, and it is
the *cause* of which `projected_field_is_sane`'s bowtie was the symptom.

### Four checks added, and one correction to the person's plan

- **`field_in_front_of_camera`** (above). The strongest of the set, and the
  only one that identifies canvas_002's defect rather than its symptom.
- **`field_covers_points`** — the person's check (a): a fit must contain the
  pixels it was fitted from, or it has explained nothing. It does NOT fire on
  canvas_002 (whose clicks do map inside the field, 6/6, 3/3, 2/2, 6/6) but it
  is the right check for the collapse mode they found in the relabellings: in
  the census below it rejects 168 candidates, more than any gate except
  shape sanity.
- **`no_parallel_crossings`** — the person's check (b), **corrected**. As
  stated ("two lines assigned to different edges cross inside their clicked
  spans") it would reject correct labellings: a sideline and a goal line DO
  meet on the field, because the goal line's endpoint lies on the sideline,
  even though they share no vertex index. MEASURED on canvas_002: `(2,8)` and
  `(3,4)` cross at (958,379) inside both clicked spans, and vertex 4 is
  exactly there — correct, not impossible. The rule that holds is about
  PARALLEL edges: two lines called the two sidelines, or two called different
  cross lines, can only meet at a vanishing point, never inside the paint.
- **One frame per calibration** — enforced in `run_calibrate.py`, which now
  refuses to bank an entry whose `line_details` span multiple
  `frame_indices`. Recorded honestly: the drift measurement above does NOT
  support this being canvas_002's problem. It stays as policy, because a
  correspondence set should not depend on registration quality when it
  doesn't have to, and Addendum 4.6 already measured a multi-frame redo
  getting worse at 48 frames deep.

### Where the blue-sideline question now stands

Re-running the census with all gates, over every candidate pairing the blue
line with another same-family line as the two sidelines:

| gate | rejected |
|---|---|
| `no_parallel_crossings` (fit-free) | 96 |
| `vertices_within_bounds` | 104 |
| `projected_field_is_sane` | 188 |
| `field_in_front_of_camera` | 0 (shape sanity catches these first) |
| `field_covers_points` | 168 |
| `extent_is_plausible` | 72 |
| claims to observe an unseen line | 44 |
| **survive** | **0** |

Still zero. The rejections are now well-characterised rather than mysterious,
and the leading explanation is unchanged: the real far sideline, painted only
inside the endzones, is probably not in the tracked candidate set at all.

## Result 3.11: the "no blue sideline" result was my bug, and fixing it did not change the answer

Chasing the open lead from Result 3.10 found a real defect in the search, not
in the data.

**`group_by_vanishing_point` returns ONE split**: the largest vanishing-point
cluster as family A, and *everything else* as family B. Family B is therefore
a grab-bag, not a direction. On the test window that put the blue near
sideline in a "family" with seven lines that do not share its vanishing point,
so every labelling using blue as a sideline paired it with a line pointing
somewhere else entirely.

Checked directly, the data never had that problem: for the blue line, **11 of
11 candidate partners** yield a usable two-direction structure (a vanishing
point with >=2 members, and >=2 of the remainder sharing a second one).

`field_lines.direction_hypotheses` replaces the single split with every
plausible pair of vanishing points — 40 hypotheses on this window, 10 of them
containing the blue line. `fit_field_model` now iterates over hypotheses.
Effect:

| | before | after |
|---|---|---|
| surviving field models | 0 | **3** |
| blue-as-sideline candidates ever enumerated | 672 | **1,684** |
| runtime | 2m30s | **46s** |

**And the answer is unchanged: still 0 blue-as-sideline survivors.** Now with
the candidates actually being tried, and rejected by four independent gates:

| gate | rejected |
|---|---|
| `no_parallel_crossings` | 324 |
| `vertices_within_bounds` | 168 |
| `projected_field_is_sane` | 306 |
| `field_in_front_of_camera` | 0 (shape sanity catches these first) |
| `field_covers_points` | **630** |
| reprojection (over-determined only) | 80 |
| `extent_is_plausible` | 108 |
| observes an unseen line | 68 |
| **survive** | **0** |

`field_covers_points` was checked for over-strictness, since a soccer
touchline legitimately extends past the ends of an inset ultimate field. It is
not too strict — measured across all 886 fits that reached it, the worst
fitted point lands outside the field by:

| percentile | distance outside a 100m x 37m field |
|---|---|
| p10 | 32.3 m |
| p50 | **204.5 m** |
| p75 | 846.8 m |
| p90 | 15,064 m |

Only 1.8% fall within the current 8m margin; loosening it to 50m would admit
13.5%, all still 10-50m out; **50.3% are beyond 200m**, i.e. outright
collapse. These fits are not near-misses.

### Where that leaves the open question

Four independent geometric criteria reject every labelling that makes the
blue line an ultimate sideline, using a candidate set that has been confirmed
complete for this frame (every single-frame detection on the reference frame
maps to a tracked cluster within 2-9px, so nothing is lost at the persistence
stage). Two explanations remain, and they are distinguishable by eye rather
than by more search:

1. **The far sideline is not detected.** It is painted only inside the
   endzones, so its evidence is a short stretch that may be one of the weak
   candidates or absent entirely. Without it there is no second sideline for
   blue to pair with.
2. **The blue line is not the ultimate sideline.** These games are played
   over soccer/lacrosse markings; if the ultimate boundary is coned rather
   than painted, the blue line may run parallel to it but not on it.

Do not tune the gates further until one of these is settled — every gate that
rejects blue does so for an independently-derived reason, and loosening them
to force the expected answer would discard exactly the checks that caught
canvas_002.

## Result 3.12: cones cannot extend the far sideline here — the teams stand on it

Person's correction, which resolves the Result 3.11 ambiguity: the blue line
IS definitely the close sideline, and the far sideline is the white line they
clicked as `(2,8)` — painted only inside the endzone, and **marked by cones
through midfield**. None of the 12 tracked candidates is that line.

That makes cone-based extension the obvious move: fit the far sideline
through the endzone paint PLUS the cones continuing it downfield, turning a
563px stub into a full-length baseline. Measured, it does not work on this
window, for a reason worth recording.

**The cone hints that align with the far sideline are all false positives.**
Three of the six sit 33-71px off the line and extend it rightward past the
stub, which looked promising:

| hint | perpendicular | position |
|---|---|---|
| (1305.8, 379.0) area=31 | 32.8px | 34px beyond the stub |
| (1372.7, 369.9) area=12 | 48.5px | 99px beyond |
| (1861.2, 395.3) area=12 | 70.9px | 588px beyond |

Checked by eye at high magnification: **A** is bare turf beside a player,
**B** is a red bag in the sideline kit pile, **C** is bare turf beside a
player. Not one is a cone.

**Why this band specifically defeats the detector.** `detect_cone_hints`
searches for orange (hue <= 22). Enumerating every compact non-green blob in
the far-sideline band returns 15, and the six largest all sit at hue 16-22
with areas 152-325 — the players' red/maroon kit and the gear pile, in
exactly the hue range a cone occupies. This is the skin/kit confound from
Addendum 4 again, concentrated: the band contains a pile of bags and a row of
roughly fifteen standing players.

**The generalisable point**: on this footage the far sideline is the worst
place in the frame to look for field markers, because it is where the teams
stand. Its paint is occluded by people for most of its length, and anything
cone-shaped near it is more likely to be equipment than a cone. Person
boxes remove the players themselves but not their kit.

So the far sideline is not recoverable from this window by any amount of
detector tuning. Two ways forward that don't involve tuning:
1. **Pick a window where that sideline is clear** — during a pull all
   fourteen players are on the field rather than on the line. `run_shot_scan.py`
   already finds field-view windows; this needs an additional "sideline not
   occluded" criterion.
2. **Re-click once on a single frame** via `run_field_lines_propose.py
   --assign`, which now enforces the one-frame rule and runs all four
   geometry checks. This is what Result 3's ACTION NEEDED asks for anyway.

## Result 3.13: RESOLVED — extend the endzone paint; the search was over-constrained

Person's question, which cut through the previous two results: *"couldn't you
just extend the sideline from the white paint shown on the sideline inside the
endzone?"*

Yes. That is what a line correspondence already does — `fit_calibration_with_lines`
takes >=2 points on a known field line and constrains the whole infinite line,
so a painted stub inside the endzone defines the sideline across midfield
whether or not anything is painted there. Result 3.12's hunt for cones to
"extend" it was solving a problem that does not exist. Cones were never
needed; the search was broken.

**First, the human's stub is not the best estimate of that line.** *(RETRACTED
-- see Result 3.16. `tracked[3]` follows the turf edge behind the players, not
the sideline. The person's clicks were right and this table measures the wrong
thing.)* Walking
each candidate across the frame and counting how much detected paint lies
under it:

| candidate | paint under it, x 700-1280 | x 1280-1900 | angle |
|---|---|---|---|
| human `(2,8)` clicks (3 pts, 563px) | 33/290 (**11%**) | 6/310 (2%) | 5.61 deg |
| **tracked[3]** (38/60 frames, sup 21,047) | 108/290 (**37%**) | 56/310 (**18%**) | 2.65 deg |

The tracker's version sits on 3.4x more paint over the endzone stretch and
spans 1,200px rather than 563px. So the "33px / 3.0deg mismatch" reported in
Result 2 was mostly the three-click stub being imprecise, not the detector
missing the line. (Those two numbers were never independent either: 3deg over
a 563px baseline *is* ~29px at the ends.)

**Two over-constraints in my own search were hiding every valid model.**

1. *Requiring the CROSS lines to share a vanishing point.* Both directions of
   a field really are concurrent, so demanding it of both families looks
   principled. But the two cross lines this window offers are a long,
   well-observed line and a short endzone stub, whose fitted directions differ
   by more than a 1.5deg concurrency test tolerates — lines [0] and [5] never
   co-occur in any single cluster. Fixed by
   `field_lines.sideline_family_candidates`: only the SIDELINE pair must be
   concurrent (that is what conditions the fit), and the seven geometry gates
   reject bad cross-line choices. Use concurrency where it is reliable, and
   geometry for the rest.
2. *Ranking sideline families by member count.* There are 61 distinct
   vanishing-point clusters here. Sorting by size buried the cluster
   {blue near sideline, far sideline} at **rank 24**, past any sane cutoff,
   even though it carries 431,516 support against the top-ranked cluster's
   208,732. Sidelines are a field's longest, most-persistently-observed lines,
   so total support is the right proxy; how many other lines happen to share
   their vanishing point is not. Re-ranked, that cluster is rank 5 and every
   one of the top eleven contains the blue line.

**Result.** The top-ranked model is now the one the person described from the
start:

```
MODEL #1  4/6 observed, cones 1/8, explains 72% of detected paint, reproj 29.4px
  view: pan covers x=-123..52m of 0..100m (52% of the length); both sidelines
        in view; endzones in view: near endzone
    left sideline    observed  in_frame  blue   residual= 1.5px
    right sideline   observed  in_frame  white  residual= 6.7px
    near end line    observed  in_frame  white  residual= 2.0px
    far end line     estimated unseen           --
    near goal line   observed  in_frame  white  residual= 1.8px
    far goal line    estimated unseen           --
```

Blue is a sideline at 1.5px; the far sideline comes from its endzone paint and
is extended across midfield by the homography; both far-end lines are correctly
marked `estimated unseen`. Drawn in `outputs/field_model_final_0.jpg`.

**Two ambiguities remain, and both are genuine rather than defects.** Model #2
is the same field with the camera at the other end (near vs far endzone) —
geometry cannot tell which end of a symmetric field you are looking at. Model
#3 swaps which sideline is called "left" — a naming convention, not an error.
Cone hits (1/8) are too weak to break the first; the person, a scoreboard, or
play direction would.

Runtime is 5m08s for the labelling search over 12 tracked lines. Worth
optimising if this becomes routine, but not before it is trusted.

## Result 3.14: CORRECTION — canvas_002 is not broken, it is LOCAL

Person, after seeing the "fixed" model: *"definitely not right, the bowtie one
is STILL the most accurate."* They are correct, and Result 3.10 drew the wrong
conclusion from a true observation.

The horizon really does cut through the field in that fit. But that is not a
defect, and the calibration is not degenerate. Round-tripping a grid of field
points through it (project to image, perturb 1px, project back):

| field x | 1 image pixel = |
|---|---|
| 0 m | 0.04 m on the ground |
| 20 m | 0.03 m |
| 40 m | 0.02 m |
| 60 m | 0.01 m |

Sharp everywhere the camera can see. What is actually happening is that the
horizon is **tilted across the field**:

| along | the plane vanishes at |
|---|---|
| y = 0 m (blue sideline) | x = **64.4 m** |
| y = 18.5 m | x = 90.9 m |
| y = 37 m (far sideline) | x = **117.5 m** |

So v7 = (100m, 0m) is past its horizon and v8 = (100m, 37m) is not. That sign
split IS the bowtie. Everything within the shot is fine; only the unobserved
far end is meaningless.

**The wrong shape of answer was pass/fail.** A calibration fitted from
evidence at one end of a field is not wrong, it is LOCAL — and rejecting it
threw away a good measurement of the near half, which is the half this footage
actually shows.

`calibration.field_validity_region` reports how far a calibration can be
trusted, and `calibration_covers_evidence` replaces the veto with the
disqualifier that actually matters — being invalid where your own evidence
lives:

| calibration | usable to | % of field | covers own evidence |
|---|---|---|---|
| `canvas_000` | 209.8 m | 100% | yes |
| **`canvas_002` / `canvas_003`** | **54.7 m** | **54.7%** | **yes** |
| `canvas_004` | 27.9 m | 27.9% | **no** |

That separates "local but good" from "actually broken" — and note 54.7% agrees
closely with the independently-measured 52-56% of the field length the camera
sweeps in this window (Result 3.9). The calibration is valid over almost
exactly the region the camera can see, which is the expected result and a
useful cross-check on both measurements.

`field_in_front_of_camera` is kept, because the statement it makes is true and
worth reporting. It is no longer a veto in `run_calibrate.py`,
`calibration_bank.add_entry`, `run_calibrate_auto.py` or `fit_field_model`;
`calibration_covers_evidence` is. **canvas_002 would now be banked**, with a
note that it is usable to x=54.7m.

### Retraction

Result 3's "ACTION NEEDED: re-do the `torovcanyon.mp4` calibrations" was
wrong for canvas_002 and canvas_003. They are usable for the near half of the
field, which is the half the footage shows. canvas_004 (27.9m, fails on its
own evidence) does still need redoing. The downstream caveat on
`results/metrics_finding.md` narrows correspondingly: speed and `is_in_field`
numbers are trustworthy for play inside x < 54.7m and not beyond.

## Result 3.15: the correct labelling, and the bug that was hiding it

The person named the lines directly, from the bowtie overlay: **yellow = the
back (end) line, blue = the near sideline, green/white = the far sideline**,
magenta = the near goal line. That is exactly canvas_002's original labelling
— `(1,7)` blue, `(2,8)` far sideline, `(1,2)` end line, `(3,4)` goal line — so
the human labelling was right all along, and the search has been wrong at
every step.

Their reading of the auto model was equally specific: only the near goal line
was correct, the right sideline ran too far, the near sideline had vanished,
the second endzone was on the wrong side, and — the useful one — *"the near
end line runs parallel to near goal line no?? why is it diagonal across the
field."*

### A real bug: the model was discarding its own evidence

Fitting the person's labelling onto the tracked lines and inspecting each
stage found `build_field_model` reporting **5 of 6 lines as `estimated`**
despite four of them having just been fitted, with matches at 1.5, 6.7, 2.4
and 1.8px sitting right there in `_match_detections_to_edges`.

Root cause: `_bbox_in_field` projected only the pan polygon's four CORNERS and
took a naive min/max. Through a LOCAL calibration — one whose horizon crosses
the frame, which Result 3.14 establishes is normal — corners near or past the
vanishing line project to enormous or sign-flipped field coordinates. The pan
extent came out as **y = -163.9m .. 6.6m on a 37m-wide field**.
`extent_supports_observation` then declared `(2,8)`, `(1,2)` and `(3,4)`
"unseen", and `build_field_model` demoted every one of them to `estimated`.

Fixed by sampling the region rather than its corners and keeping only samples
genuinely in front of the camera (consistent sign of the homogeneous w).
Points behind the horizon are not part of what the camera saw and must not
contribute to a bounding box. With that fixed, the person's labelling builds
correctly:

| line | source | visibility | residual |
|---|---|---|---|
| left sideline (blue) | observed | in_frame | **1.5px** |
| right sideline (far, from endzone paint) | observed | in_frame | **6.7px** |
| near end line | observed | in_frame | **2.4px** |
| near goal line | observed | in_frame | **1.8px** |
| far end line | estimated | unseen | — |
| far goal line | estimated | unseen | — |

usable to x=68.8m. Drawn in `outputs/field_model_human_labelling.jpg`, and the
end line and goal line now converge correctly instead of crossing diagonally.

### A failed experiment, kept as a warning

The person's parallelism observation suggested a gate: lines parallel on the
field meet only at a vanishing point, so reject a fit whose same-group edges
intersect inside the playing surface. Implemented as
`parallel_edges_meet_outside_field` and MEASURED WRONG — it returns False for
**every** calibration tested, including canvas_000, the verified-good one.

The reason is fundamental, not tuning: the intersection of two correctly
parallel field lines is a point at infinity, so `pixel_to_field` on it is
ill-conditioned and returns an arbitrary finite coordinate that lands inside
the field as often as not. The check cannot distinguish the case it was built
for. It is left in the module raising `NotImplementedError` with this
explanation rather than deleted, because the idea is tempting enough to be
worth re-deriving correctly (in homogeneous coordinates, where a correct pair
meets at w ~ 0) rather than re-inventing badly.

### Still open

The auto-ranking still does not select the person's labelling. Three separate
scoring/search fixes across Results 3.11-3.13 each changed which wrong answer
came first. The honest read is that ranking is not close, and the `--assign`
path — where the person names the lines and the model supplies estimation,
extent and the six-line output — is the one that works.

## Result 3.16: RETRACTION — paint support is not evidence of being a FIELD line

Result 3.13 claimed the tracker's `tracked[3]` was a better estimate of the far
sideline than the person's three clicks, on the grounds that it had 3.4x more
detected paint under it (37% vs 11%). **That was wrong, and the conclusion
drawn from it sent the next three rounds down the wrong path.**

The person, from the rendered model: *"the right sideline is too far out, can
you see how it's behind the line of players? and how in the endzone there is a
white line parallel to the wrong sideline? that's supposed to be the
sideline."*

Drawing both lines extended across the frame settles it. They diverge steadily:

| x | your clicked line | tracked[3] | gap |
|---|---|---|---|
| 700 | 352.4 | 341.4 | 11px |
| 1100 | 391.8 | 360.0 | 32px |
| 1500 | 431.1 | 378.5 | 53px |
| 1900 | 470.4 | 397.0 | **73px** |

`tracked[3]` runs along and behind the row of standing players — it is the
turf edge / boundary structure behind the field, not the field's own line.
The person's line runs in front of them and sits on the endzone white paint.

**Why the metric lied.** "How much detected paint lies under this line" cannot
distinguish a field marking from a high-contrast scene boundary running
roughly parallel behind it. On this footage the far sideline is precisely
where such a boundary exists (Result 3.12: it is where the teams stand, their
kit is piled, and the turf ends). The competing structure is longer, straighter
and higher-contrast than the short painted stub, so it wins on every
support-based measure while being the wrong line. `flanks_match` (Result 3.7)
is meant to catch exactly this and did not, because at its sampling offset
there is grass on both sides of the turf-edge line too.

**Fitting the person's four clicked lines directly**, and drawing each edge cut
at the horizon (Result 3.15):

| line | source | residual |
|---|---|---|
| left sideline (blue) | observed | 1.8px |
| right sideline (endzone paint, extended) | observed | **0.7px** |
| near end line | observed | 0.6px |
| near goal line | observed | **0.0px** |
| far end line / far goal line | estimated | — |

max reprojection **2.74px** (against 29.4px for every fit that used
`tracked[3]`), usable to x=54.7m. `outputs/field_model_your_lines.jpg` — the
right sideline now runs in front of the players and continues across the field,
which is what "just extend the endzone paint" means in practice.

**The lasting lesson**: support, length and persistence measure how strong a
line's evidence is, not whether it is the line you want. Every ranking in this
module leans on them. That is a standing weakness, not a fixed bug.

## Result 3.17: what actually helps — surface the confusable pairs, and two more negatives

Acting on Results 3.15 and 3.16 rather than adding another ranking term.

### Negative result: vanishing points do NOT separate the confusable pair

Last round's proposal was that the true sideline and the turf edge behind it
cannot both pass through the sideline vanishing point, so a VP test would
catch what support-based ranking missed. **Measured directly, and it is
false** — they are near-parallel in the world:

| labelling | sideline VP | cross VP | evidence vs horizon |
|---|---|---|---|
| correct (your 4 clicked lines) | (-562, 228) | (1972, 249) | all one side |
| wrong (far sideline = the turf edge) | (-493, 286) | (1972, 249) | all one side |

Both plausible, both with the evidence entirely on one side of the horizon.
The VP structure cannot tell them apart, so it was not built as a gate.

### What the failure actually looks like, and what does help

Every automated signal in this module prefers the wrong line:

| signal | prefers |
|---|---|
| support / length / persistence | **wrong** (3.4x more paint, Result 3.16) |
| `flanks_match` | no separation (grass both sides of the turf edge too) |
| vanishing-point structure | no separation (above) |
| reprojection error | reflects a line's internal scatter, not its identity |

What does resolve it is the picture: 11px apart where the paint is, **73px
apart by the far edge of the frame**. So `field_lines.confusable_groups`
finds the near-parallel candidate sets, and `draw_extended` renders them
extended across the whole frame -- the person's own suggestion, *"generate
lines that go past the image to see where these lines all meet."*

On the test window it finds two groups, correctly:

```
group 0:  [0] sup=112661   [1] sup=14111   [9] sup=14357
          [0] vs [1]:  22px apart at x=0,  92px at x=1919
          [1] vs [9]:  35px apart at x=0, 135px at x=1919
group 1:  [3] sup= 21047  [11] sup= 4382
          [3] vs [11]: 128px apart at x=0, 106px at x=1919
```

`run_field_lines_propose.py` now prints these and writes one image per group
before any labelling is proposed. It does NOT pick a winner -- the measured
point of this result is that nothing in the module can, and pretending
otherwise is what produced four rounds of confident wrong answers.

## Result 3.18: people mark the sideline that has no paint — and it is the first signal that discriminates

**Contributed and measured by the person**, reproduced here and turned into
`frisbee_analysis/watchers.py`.

YOLO11n over the 60 frames of canvas_002's own window, foot points mapped into
reference coordinates through the existing registration: 2,710 person
detections, ~45s of CPU. The discriminator is free -- a watcher's foot point
sits in nearly the same registered spot all window, a running player's does
not. Counting distinct frames with a foot within 12px:

| persistence | person's count | reproduced |
|---|---|---|
| >= 5 frames | 1,929 / 2,710 | 2,109 |
| >= 20 frames | 1,228 | 1,435 |
| >= 30 frames | 898 | 924 |

RANSAC over the >=20 set finds one dominant row, and the module reproduces it
closely:

| | person | module |
|---|---|---|
| dominant row | 1,060 pts, 4.5px RMS, 1,330px extent | 29 standing positions (1,435 detections), **4.20px**, **1,210px** |
| median person box height | 47px | **46.6px** |
| field y through canvas_002 | 41.1m, IQR 40.3-41.9 | **41.0m**, IQR 39.7-41.7 |
| standoff on a 37m field | ~4m outside | **+4.0m** |

Fed to `group_by_vanishing_point` alongside the 12 tracked paint lines it lands
in family B with the blue sideline, no special handling.

### It is a DIRECTION cue, not a POSITION one

The row stands ~4m outside the boundary and the standoff DRIFTS along the
field (person: y = 37.33 + 0.107x, 0.78m residual; module over a wider x
range: y = 32.97 + 0.211x, 1.55m residual). Used as the sideline directly it
is off by 4-6m. Its direction disagrees with canvas_002's far sideline by
2.28 degrees -- unresolved, and either people not standing parallel or
canvas_002 being slightly rotated.

`standoff_from_paint` exists for the person's route out of this: the far
sideline IS painted inside the endzones, so measure the offset where paint and
people coexist and carry it across midfield where none exists.

### The failure mode is real and immediate

The person's second static cluster -- 167 points, straight to **1.0px, i.e.
straighter than the players** -- is the fence/path beyond the field, 10.2
degrees off the sideline. **A straight row of standing people is not by itself
a sideline, and straightness is actively misleading as a criterion.** Two
separators worked: those people are smaller (median box 36px vs 47px), and
they sit outside the playing-surface mask this project already computes. Both
are carried on `WatcherRow`. Honest caveat from reproduction: at cluster level
the watcher row itself is only 59% on-surface, so that separator is weaker
here than at detection level -- use box height first.

### THE RESULT THAT MATTERS: it discriminates where nothing else did

`watchers_outside_field` is a labelling gate. Watchers stand outside the
boundary, so any labelling that puts a dense static collinear row of people
INSIDE the field is wrong. This is evidence the homography fit never consumed,
so unlike reprojection error it can rank labellings -- the same role as cone
hints, with 1,435 detections behind it instead of 1 cone in 8.

Tested on the exact pair that defeated every other signal (Result 3.16 -- the
person's true far sideline versus `tracked[3]`, which their measurement
identifies as **the next field over**, at y ~ 50m, a better explanation than
this document's earlier "turf edge" guess):

| labelling | `watchers_outside_field` |
|---|---|
| correct (person's 4 clicked lines) | **True** |
| wrong (far sideline = `tracked[3]`) | **False** |

Set against the scoreboard from Result 3.17:

| signal | separates the confusable pair? |
|---|---|
| support / length / persistence | no -- prefers the WRONG line |
| `flanks_match` | no |
| vanishing-point structure | no |
| reprojection error | no (measures scatter, not identity) |
| **watchers_outside_field** | **YES** |

Wired into `fit_field_model` as an optional `watcher_rows` gate. It is the
first automated signal in this investigation that agrees with the person's
own reading of their footage.

## Result 3.19: my gate stack was rejecting the ground truth

Wiring the watcher gate in (Result 3.18) surfaced something worse than a
ranking problem. Running the full search returned **0 models from ~81,000
candidates**. Testing the person's verified-correct labelling against every
gate individually:

| gate | verdict |
|---|---|
| `no_parallel_crossings` | PASS |
| `vertices_within_bounds` | PASS |
| **`projected_field_is_sane`** | **FAIL** |
| `calibration_covers_evidence` | PASS |
| `field_covers_points` | PASS |
| reprojection <= 30px (2.74px) | PASS |
| **`extent_is_plausible`** | **FAIL** |
| all fitted edges "seen" | PASS |
| `watchers_outside_field` | PASS |

**Two gates I built were rejecting the known-correct answer**, and both for
the same reason `field_in_front_of_camera` did in Result 3.14: they judged
field coordinates PAST THE HORIZON, where the projection is meaningless.

- `projected_field_is_sane` tested the outer corners at x=0 and x=100m. This
  calibration is LOCAL -- usable to x=54.7m -- so vertices 7 and 8 project
  from beyond its own horizon and the convexity test was reading noise. Now
  the field is clipped to `field_validity_region` before the convexity and
  ordering tests. Regression: still True for canvas_000 and canvas_002/003,
  still **False** for canvas_004, which fails on its own evidence.
- `extent_is_plausible` read a pan bbox of y = -4.4..184.2m on a 37m-wide
  field. Two fixes, both measured: exclude samples beyond the validity limit
  (not just those behind the camera -- points merely NEAR the horizon carry
  the right sign while projecting absurdly), and take **robust percentile
  bounds rather than min/max**, because a handful of near-horizon pixels set
  the extremes. The bbox becomes x -36.0..54.4m, y -4.3..64.0m.

After both fixes **all nine gates pass the person's labelling**, and the
search returns models again. The top one now has the blue line correctly as a
sideline, with a plausible view description (87% of the field's length, near
endzone in view).

**Judging geometry you cannot see is not a sanity check.** Three separate
gates made this same mistake independently. Any future check that reads a
projected field coordinate must first ask whether that coordinate is inside
`field_validity_region`.

### What is still blocking, and what it is NOT

The search still selects `tracked[3]` as the far sideline -- the line the
person's watcher measurement places at field y ~ 50m, i.e. **the next field
over**. The true far sideline is simply not in the tracked candidate set.

Tested and eliminated as the cause: extrapolation merging absorbing the short
endzone stub into its stronger neighbour. Re-tracking the window with
`merge_extrapolation` off gives the same answer:

| | lines | best match to the person's (2,8) |
|---|---|---|
| `merge_extrapolation=True` | 12 | [3] at 32.3px / 2.97 deg |
| `merge_extrapolation=False` | 12 | [2] at 30.3px / 3.22 deg |

So the endzone paint is not being lost at the merge stage -- it is not
surviving detection as its own line at all. It is short, adjacent to a
stronger parallel line, and partly occluded by the gear pile and standing
players documented in Result 3.12.

Also honest about the watcher gate: it discriminates cleanly in the isolated
test (Result 3.18) but does not fire during the full search, because a wrong
labelling is free to place the watcher row outside ITS OWN field. It rejects a
bad answer when you hand it one; it does not by itself steer the search.

## Result 3.20: making the validity bound reach the metrics

Acting on Result 3.14 downstream instead of only inside the calibration code.

`is_in_field` asks whether a point PROJECTS inside the boundary. It never asks
whether the calibration can be trusted there, and for a LOCAL calibration that
is the difference between a measurement and noise: past the horizon
`pixel_to_field` returns ordinary finite coordinates, and sampling the whole
canvas_002 canvas shows **14% of what `is_in_field` accepts lies beyond the
54.7m validity limit**. Every such position was being counted, and every speed
computed from a pair of them was noise times a frame rate.

`calibration.is_measurable` adds the bound. `run_metrics_test.py` now filters
with it and reports the drop. Re-run on torovcanyon t=3945s+30s:

```
CALIBRATION VALIDITY: usable to x=54.7m of the 100m field (55%).
  This calibration is LOCAL -- past that x the field crosses the horizon.
  Detections dropped as past the horizon: 7
On-field detections per frame -- team A: mean=2.9   team B: mean=9.6
Consecutive-frame track matches usable for speed: 72
Team A: n=14 mean=1.16 m/s median=0.67  Team B: n=58 mean=1.07 m/s median=0.51
```

72 usable speed samples against the 8 that `results/metrics_finding.md`
recorded, and now with an explicit statement of which half of the field the
numbers cover.

**The bank no longer serves a broken canvas.** `try_auto_calibrate` skips any
canvas whose own calibration fails `calibration_covers_evidence`, because
composition only multiplies homographies -- matching against a broken canvas
inherits its defect silently. A LOCAL canvas is deliberately still served:

| canvas | usable to | verdict |
|---|---|---|
| `canvas_000` | 209.8m (100%) | served, full field |
| `canvas_002` / `canvas_003` | 54.7m (55%) | **served, local** |
| `canvas_004` | 27.9m (28%) | **skipped -- broken** |

canvas_004 is the one still needing a manual redo.

## Result 3.21: a regression test, because a person was the only thing catching these

Building this module produced, in order: three separate geometry gates that
rejected the one labelling a human had confirmed correct (each by judging
field coordinates past the horizon); a ranking term that actively rewarded
wrong answers; and a "check" that returned False for every calibration
including the verified-good one. **Every one of those was caught by the person
noticing, not by anything in the code.**

`run_regression_test.py` pins the answers that were expensive to establish.
It reads `tests/fixtures/torovcanyon_3945.json` -- the human-confirmed
labelling in segment-reference coordinates, plus the decoy line and the
watcher row -- so it runs in seconds and needs no video.

22 assertions across six areas:

1. **The human-confirmed labelling survives every gate** (2.74px fit,
   `no_parallel_crossings`, `vertices_within_bounds`,
   `projected_field_is_sane`, `calibration_covers_evidence`,
   `field_covers_points`, `extent_is_plausible`, edges-seen,
   `watchers_outside_field`)
2. **Local is not broken** -- recognised as LOCAL, ~55% usable, still served
3. **The validity bound reaches the metrics** -- `is_measurable` strictly
   stricter than `is_in_field` (684 of 5,341 samples past the horizon) but
   keeps 87%
4. **The decoy is rejected** -- the next-field-over line that every
   strength-based metric prefers: the watcher gate rejects it, and it fits its
   own lines 7x worse (18.8px vs 2.74px)
5. **The six-line model shape** -- always six, four observed, the two far
   edges `estimated`/`unseen`
6. **Confusable pairs are surfaced, not silently resolved**

### And a seventh section, which is the point

A regression test that cannot fail is worse than none, because it reports
safety. Section 7 **reintroduces each historical bug and asserts the matching
check flips**:

| injected bug | caught |
|---|---|
| `projected_field_is_sane` reading past the horizon | yes |
| pan extent from min/max instead of robust percentiles | yes |
| `is_measurable` degraded to `is_in_field` | yes |

Worth recording how that section was itself validated: the first attempt at
injecting the extent bug reported MISSED, and the test looked inadequate. It
was the INJECTION that was wrong -- it used only the polygon corners, while
the real bug grid-sampled and then took min/max. Faithfully reproduced, the
bug produces a pan extent 23.5x the field width and is caught. A bug injection
is itself code, and can be wrong in the direction that makes your test look
worse than it is.

## Result 3.22: registration drift is what degrades the tracked lines

The human-clicked lines fit to 0.5-2.7px. The tracked lines built on the same
footage carry rms 4.0-8.4px, and the blue sideline arrives as two clusters
28-46px apart. That gap has been attributed to detection quality throughout
this document. It is mostly drift.

`register_sequence` registers each frame against the last good one, so error
compounds along the chain. Measured on torovcanyon.mp4 t=3945-4005s by
comparing each frame's chained homography against an independent DIRECT
registration to the segment anchor:

| chain depth | divergence from direct |
|---|---|
| frames <= 20 | median **2.2px** |
| frames > 40 | median **7.7px** (max 8.9px) |

Those magnitudes are the tracked-line rms almost exactly. It is also why a
159-frame segment produced a pan extent 2,354 frame-widths wide (Result on
whole-field feasibility).

**Direct-to-anchor is not a drop-in replacement.** It succeeded on only 28 of
59 frames, the failures coverage-limited (7-14% against the 15% floor),
because a wide pan leaves distant frames sharing too little background with
the anchor. That is the same constraint `results/mosaic_finding.md` recorded
when incremental chaining was chosen in the first place.

`mosaic.refine_to_anchor` keeps chaining but makes the chain SHORT: every
frame that registers directly to the anchor becomes a hub, and the rest reach
it through one of those in a single extra hop. Chain depth goes from "up to N"
to 1 or 2. On a 30-frame window, 15 of 29 frames now reach the anchor
directly, in 20s.

**Honest about what is verified.** For the frames that register directly the
measured divergence is 0.00px, but that is tautological -- they now ARE the
direct registration. The benefit for the remaining 2-hop frames follows from
depth (2 instead of up to 29) but is NOT independently measured, because
there is no ground truth for exactly the frames that fail direct registration.
The claim is "chain depth is provably shorter", not "drift is provably
smaller everywhere".

Wired into `run_field_lines_propose.py`. Non-destructive: any frame it cannot
improve keeps its original chained homography.

## Result 3.23: the watcher row as a SEARCH PRIOR — measured, and it does not work

The plan from Result 3.18: use the watcher row's direction and standoff to hunt
for the endzone paint where we now know it should be, turning the row from a
gate into a selector. Integrating paint ALONG a known direction should rescue a
line too weak to detect on its own.

**It does not.** Sweeping perpendicular offsets from the row and measuring
paint density (the true far sideline sits at -11.2px from the row at its
midpoint, the decoy at -29.6px):

| offset | paint density | |
|---|---|---|
| +32px | **37.9%** | the profile's peak |
| +16px | 14.6% | |
| -16px | 2.6% | nearest sample to the TRUE sideline |
| -32px | 7.0% | the decoy |

The peak is 20x stronger somewhere else. Restricting the integration to windows
along the row -- since the paint exists only over part of the span -- does not
help either: **every** window peaks at +28..+44px, and density at the true
offset is 0-7.7% throughout, including the endzone region.

### Why: this line is at the detector's contrast floor

Sampling nine points along the person's clicked (2,8) line and asking what each
stage of `line_pixel_mask` sees:

| x | V | in surface? | in search region? | V-ridge | chroma-ridge | final mask |
|---|---|---|---|---|---|---|
| 709 | 99 | **no** | yes | yes | no | yes |
| 779 | 73 | **no** | **no** | yes | no | no |
| 849 | 85 | **no** | **no** | yes | no | no |
| 919 | 88 | **no** | yes | no | no | no |
| 989 | 86 | yes | yes | no | no | no |
| 1059 | 86 | yes | yes | no | yes | no |
| 1129-1269 | 83-85 | yes | yes | no | no | no |

Two independent problems, neither fixable by better searching:

1. **There is barely a brightness step.** V along the line reads 73-99 against
   turf measured at 52-78. The ridge test fires on 3 of 9 samples, the chroma
   test on 1. The paint is visible to a person but sits at the edge of what
   this detector can separate from grass at that distance.
2. **The surface mask excludes x=709-919** -- the endzone end of the line, the
   part that actually has paint. Part of it is not even inside the search
   region, so no threshold change could recover it.

Component analysis agrees: of five V-ridge components touching the clicked
line, two are kept (elongation 16.7 and 51.0) and three dropped, and the kept
ones cover only the far-left stretch.

### What this rules out, and what it points to

Not a labelling problem, not a merging problem, not an integration problem --
all three were tested and eliminated (Results 3.16, 3.19, and this one). The
far sideline is near-invisible to the current detector on this footage.

The actionable sub-finding is (2): **the playing-surface mask excludes part of
a line that is definitionally at the surface's boundary.** The search region is
the surface dilated by 15px at 1920, which is evidently not enough where the
boundary is ragged. That is a concrete, bounded thing to fix, and unlike
everything else tried here it is upstream of the problem rather than
downstream of it.

## Result 4: cone hints are usable now, but only as hints

`results/calibration_finding.md` Addendum 4 measured HSV cone filtering as
unusable, with crowd skin tone the binding confound (81 false candidates on
one crowd shot). Two context filters fix that, and both are needed:

| | torovcanyon t=3945s | ojuc t=1580s |
|---|---|---|
| orange blobs, on/near the surface | 17 | 15 |
| ...after subtracting YOLO person boxes | **0** | **1** |

The surface mask removes crowd, tents, parking, a maroon field logo and a
background goalpost; it does **not** remove players, who stand on the field —
person boxes do. The one survivor on `ojuc` was checked by eye: a red bag
under a sideline tent, not a cone.

**Honest limit, measured at the same time:** on the bare-grass footage the
real corner cones are ~3–5px across at broadcast distance. One visible in
`ojuc.mp4` t=1580s at roughly (20,450) was missed. Cones are corroboration
for a labelling, never anchors for a fit — which is exactly how
`rank_line_labelings` uses them, and why nothing here fits a homography from
cone positions.

## Result 5: automatic LABELLING does not work, and here is the reason

Detection works; deciding which detected line is which does not. Two
independent reasons, both structural rather than tuning problems:

1. **Four line correspondences determine a homography exactly** (8 DOF, 2
   per line). Every candidate labelling therefore reproduces its own four
   lines to sub-pixel accuracy, so reprojection error carries *zero*
   information about whether the labelling is right. Measured directly: a
   deliberately scrambled labelling of the same four lines scored
   max 7.11px and passed all three checks, just as the plausible one did.
   Only a 5th line makes the system over-determined. The returned proposals
   now carry an `over_determined` flag so this isn't mistaken for accuracy.
2. **A broadcast framing often shows only ONE of the two sidelines.** On the
   window above the left sideline projects entirely off-frame
   ((-308,442)→(-1018,-158)), so every candidate labelling has to invent it.
   Lines that are all parallel on the real field share a vanishing point and
   cannot constrain the perpendicular direction, however many there are.

`rank_line_labelings` now **refuses** rather than guesses when the
vanishing-point grouping doesn't yield ≥2 lines in each direction. An
earlier version fell back to "any line can be any edge" and produced a
confident, precise, wrong answer. Its vanishing-point grouping also had to
become **angular** rather than pixel-based: a vanishing point sits thousands
of px off-frame, and a `max(50px, 0.15·extent)` tolerance swept 11 of 12
tracked lines into one family, including a line ground truth confirms is
genuinely perpendicular to the rest.

## Result 6: what this does and doesn't work on

| footage | surface | lines found | verdict |
|---|---|---|---|
| `torovcanyon.mp4` (1080p turf, painted markings) | 59.8% | 12 tracked | **works** — 3 of 4 human lines to 1.4–6.3px |
| `ojuc.mp4` (1080p bare grass, cones only) | 32.7% | **0–1** | near-correct refusal — there is no paint to find; one false positive remains at t=1700s |
| `videoplayback.mp4` (640×360 gridiron) | 8.6% | 0 | **fails**, see below |

`ojuc.mp4` returning zero is the right answer, not a failure: that field is
marked with cones alone, and the detector declines to invent lines out of
grass texture. Calibrating that footage still needs `run_calibrate.py`.

`videoplayback.mp4` fails for a measured reason. At 640×360 the painted
markings are a large fraction of the surface and split it into diagonal
bands that are further apart than the morphology can bridge — the largest
component is 9.2% of the frame, and joining the bands needs a close kernel
of ~13px at 640 (≈39px at 1920), which is far past the ~9–14px at which the
1080p footage merges the field into the treeline. Lowering the saturation
floor recovers only 9.2% → 11.4%. Not pursued further: that is a UFA field,
which is a different size from the WFDF geometry `ultimate_config.py`
encodes, so it was never the target of this milestone.

## What to use

- `run_field_lines_propose.py` — detect and track lines over a window, look
  at `outputs/field_lines_candidates.jpg`, then `--assign 0=3,4 5=1,2 ...`
  to name them and fit. Replaces clicking points along lines you have to
  find by eye (which is how the bowtie happened) with picking from lines the
  detector found in real paint. Caches detection so the assign step is
  instant.
- `run_field_lines_test.py` — the measurement above, re-runnable on any
  window.
- `run_calibrate.py` — still the tool for footage with no painted lines
  (`ojuc.mp4`), now with the bowtie check wired in.

## Open

- **Re-do the `torovcanyon.mp4` calibrations** (Result 3). Everything
  downstream of them is suspect until then, and the new check will now catch
  a repeat.
- A 5th tracked line would make labelling discriminable; the window above
  has 12 candidates but only ~4 that are real field boundaries. Worth
  re-testing on a window whose pan covers both sidelines.
- Depth (DepthAnything) for distance is explicitly held for a later
  milestone and is not touched here.

---

## Result 3.24: the labelling image was hiding most of the best line

**Prompted by the person, stating a constraint:** *"i will never have a single
frame with four lines."*

That is a fact about their footage, and it sounded like it invalidated the
whole `--assign` workflow. It does not — but it exposed a real defect one
layer up.

**What was already fine.** `track_lines` never worked per-frame. It detects on
each raw frame, pushes every segment through that frame's homography, and
fuses them into the segment's *shared reference coordinates*. Four lines the
camera saw at four different moments of one pan are all in the same candidate
list, and `--assign` labels them together. The requirement is one uncut
**segment**, not one frame. (Across a cut, homographies are not comparable and
this genuinely fails — that limit is real.)

**What was broken.** Step 2 drew the numbered candidates on the *anchor frame*.
Anything outside it was clipped away. Measured on
torovcanyon.mp4 t=3945+60s, per line, as the fraction of pooled sample points
landing inside the 1920x1080 anchor:

| idx | paint | support | inside anchor frame |
|----:|:------|--------:|--------------------:|
| 1 | blue | 466356 | **26%** |
| 11 | other | 5403 | **39%** |
| 0 | white | 112661 | 70% |
| 7 | white | 72140 | 74% |
| others | | | 98-100% |

Overall **17% of all tracked line evidence was never visible on the anchor
frame** — but it is not spread evenly. Line 1 is the BLUE CLOSE SIDELINE: the
highest-support line in the window by a factor of four, the one the person
identified unprompted as *"DEFINETLY the close sideline"*, and the one whose
detection took a dedicated chroma channel to recover at all (Result 3.10). It
was **74% off-frame in the labelling image**. Line 11 is the near goal line in
the auto-ranker's top model, 61% off-frame.

So the person was being asked to label lines from a picture that showed a
quarter of the most important one. Their statement was not a limitation of
their footage; it was an accurate description of the picture we handed them.

**Fix.** `mosaic.segment_canvas` composites every registered frame of the
segment into one image spanning the whole pan, returning the 3x3 that maps
reference coordinates onto it; `field_lines.transform_lines` moves the
candidates there. Step 2 now draws candidates, cones and the confusable-pair
images on that canvas. On this window it is 3770x1778 against a 1920x1080
frame — 3.2x the area.

Composited as a **collage**, not a blend, and the anchor frame paints first.
Same reason as `calibration_bank._grow_and_composite`: averaging overlapping
frames smears exactly the thin low-contrast markings this tool exists to show.

**Explicitly unchanged:** detection still runs on raw frames only, never on
the composite (Addendum 1 of the mosaic finding measured that detecting on
stitched pixels finds the seams), and `add_entry` still banks the raw anchor
frame, because bank matching registers against real frames. The canvas is a
*display* surface. Nothing is fitted on it.

**Caveat.** This does not make the four lines any more correct — it makes them
visible. The confusable-pair failure of Result 3.17 is unaffected and still
needs the eye.

---

## Result 3.25: the tool could not express the answer

**The person, on t=3600+60s:** *"line 1 is the goal line and line 2 is nothing.
both sidelines are visible from this frame"* — and then, of my proposed far
corner, *"corner b is wrong"*.

**What the cones appeared to give.** One cone hit sits 1.0px from line 0 and
5.9px from line 1 — the corner where the near sideline meets the goal line.
That one is real. Two more sit ~36px from the goal line near the row of
standing players; I took their midpoint as the far corner, and the person
rejected it. They are kit or clothing, not boundary cones. **A cone hint near
a line is not a corner.** Nothing in the pipeline distinguished the two.

**The automated pick, and why it was wrong.** Of all 12 candidates, line 6
passed closest to that (wrong) corner — 9px — and its vanishing point with
line 0 landed on the horizon at (-126, 297). Fitting
`0=1,7 6=2,8 1=3,4 10=1,2` gave:

- `vertices_within_bounds`, `projected_field_is_sane`,
  `field_in_front_of_camera`, `field_covers_points` — **all four True**
- **84% of detected paint explained** — the highest this project has produced,
  above the 72% of the t=3945 window that ground truth confirms
- a cone landing on a boundary corner

And it is visibly wrong: the overlay draws the right sideline along the
treeline and collapses the field to a wedge across the top, leaving the lower
two thirds of the frame — obviously playing surface — outside the model. The
blue sideline's own paint also continues past the fitted end line, which a
sideline cannot do.

**This is the fifth time every gate has passed a visibly wrong labelling.**
Combined with Result 3.17 the conclusion is no longer tentative: the gate
stack can reject some wrong answers, but it cannot select the right one, and
*explained paint fraction is not a ranking signal* — the wrong answer scored
higher than confirmed ground truth. Stop treating a high score as evidence.

**The actual blocker, which is new.** The far sideline is visible to the
person and has **no candidate on it at all**. Measured: the entire pan
contains 2918px of blue on the playing surface, and every one of those pixels
belongs to line 0. So `--assign`, which can only name lines the detector
found, had no move — the window was unusable no matter how good the labelling
judgement was.

**Fix: `--line vi,vj=x1,y1,x2,y2`.** The person reads two points off a new
100px grid (`outputs/field_lines_grid.jpg`, the pan canvas of Result 3.24)
and states the line directly. Canvas coordinates in, reference coordinates
out. A manual line carries `support_px=0` and `n_frames=0` on purpose, so it
can never inflate the explained-paint score that just proved untrustworthy.

This is the honest division of labour the finding doc has been circling: the
detector contributes lines it can actually see, the person contributes the
ones only a person can, and neither is asked to do the other's job.

**Still open:** which line the far sideline actually is. That is the person's
to say; nothing measured here identifies it.

---

## Result 3.26: ignoring circles — what works, and three things that don't

**The person:** *"something to ignore circles and non straight lines?"*

Warranted: on t=3600+60s, **six of the twelve candidates crossing the goal
line were chords of one soccer centre circle** (3, 8, 13, 19, 20, 21).

### Failure 1 — per-line curvature is not there to measure

The obvious test is to fit a parabola to each line's points and measure the
bow. It does not merely underperform, it **inverts**:

| line | bow | truth |
|---|---:|---|
| 3 | 1.6px | circle chord |
| 13 | 2.0px | circle chord |
| 8 | 7.5px | circle chord |
| 19 | 11.3px | circle chord |
| 18 | 20.9px | **straight** |
| 16 | 33.1px | **straight** |

The cause is structural: `_consolidate_clusters` already splits a curve into
short chords, each of which genuinely *is* straight to within a pixel or two.
There is no curvature left inside a single tracked line. Any per-line
straightness test is measuring the tracker's segmentation, not the world.

### Failure 2 — fitting the conic to tracked-line points

The signal is that *several* lines share one conic, so fit a conic to their
pooled points. With points drawn from many straight lines, RANSAC threads a
degenerate sliver ellipse through them: the best fit claimed **92% of line 2**
and 46% of line 1 — both straight — while catching only 30% of a real chord.

### Failure 3 — allowing a second conic

With `max_conics=2` the first conic is the real circle (1251 inliers, 250° of
coverage) but the second is a 121° arc that flags straight lines 4, 15 and 17
at **63-75%**. Default is now 1.

### What works

Fit the conic to **dense white paint pixels**, not to tracker output, under
three constraints that each answer one of the failures above: a genuine
ellipse (negative discriminant), bounded axis ratio (no slivers, vs failure
2), and wide **angular coverage** ≥120° (a straight line cannot wrap, vs
failure 2). `field_lines.detect_conics` + `arc_fractions`.

Result on this window — one conic, centre (2019,698), semi-axes 169/663px,
250° of arc, 1251 pixels:

| threshold | flags | false positives | missed |
|---|---|---|---|
| ≥25% of points | 3, 8, 17, 20 | 17 (2 of 8 points) | 13, 19, 21 |
| ≥25% **and** ≥5 points | 3, 8 | **none** | 13, 19, 20, 21 |

`--drop-arcs` uses the second, precision-first rule. The default is to
**flag only**, shown as an `arc` column in the candidate table.

**The limit, stated plainly:** chords carrying only 4-6 sample points score
0% because too few of their points land within tolerance. *A low arc fraction
is not evidence a line is straight.* Dropping on this signal is safe only
because the rule requires positive evidence; it must never be inverted into
"low arc% therefore keep".

### Consequence for Result 3.25

Line 19 — the candidate whose geometry pointed at the person's stated endzone
corner (0px at corner A, 34px at corner B) — **is a circle chord**. It carries
6 points, below the flagging threshold, so the filter does not catch it, but
its paint lies on the circle. That corner agreement is most likely
coincidence, and the far sideline at t=3600 remains undetected.

---

## Result 3.27: the sideline WAS detected — it was averaged with two neighbours

**The person:** *"sideline was almost detected in field lines candidates as
number 6"*.

Line 6 had **the worst rms of all 29 candidates** — 9.63px over 26 points,
against 0.2-3.2px for everything trusted. That number was visible the whole
time and treated as a quality score. It is not: it is a *statement that the
points do not fit one line*.

RANSAC-splitting line 6's own point cloud:

| subset | pts | rms | extent (ref) | distance to the person's endzone corner |
|---|---:|---:|---|---:|
| 0 | 11 | 1.02px | (1784,398)→(890,353) | 10.3px |
| 1 | 8 | 0.26px | (1082,346)→(1717,390) | 26.2px |
| 2 | 6 | **0.20px** | (1127,375)→(1764,412) | **0.8px** |

Three genuinely straight markings, merged by `_consolidate_clusters` into one
line that matched none of them. The merged fit is the *average* of three
things a few pixels apart, so it lies on none — and since only whole
candidates can be labelled, the real sideline could never be selected no
matter how good the labelling judgement was.

This is the same shape of bug as Result 3.24 (the candidate was there but
could not be seen) and 3.25 (the line was there but could not be named): the
evidence existed and the tool could not express it.

**Fix: `field_lines.split_merged_lines`,** on by default in `track_lines` and
also applied to cached candidates so old caches benefit. A line already
fitting under 4.0px rms is returned untouched, and a split is accepted only
if every part fits better than the merged line did — so it acts only where
the evidence itself says the merge was wrong. On this window: 29 -> 35
candidates, five of which pass within 15px of the person's corner at under
2px rms, where before there were none.

**Not yet resolved.** Fitting with subset 2 as the far sideline still fails:
it sits only 9.2 degrees off the goal line, so their intersection is
ill-conditioned and the resulting field stretches to 114m across a 37m width.
Whether that means subset 2 is the wrong sub-line, or the 4th line is wrong,
is not yet measured.

**Retraction.** Result 3.26 closed by saying "the far sideline at t=3600
remains undetected". That was wrong, and wrong in the direction this project
keeps failing in: it treated absence of a usable candidate as absence of
evidence. The sideline was in the data the whole time, inside line 6.

---

## Result 3.28: a shot cannot tell a halfway line from a back line

**The person, on the drawn field:** *"everything looks right except the far
goal line and the far end line are not correct: this camera view only shows
around half of the field and the far end line detected is actually the
halfway point."*

Four of the six lines were right. The two wrong ones were both **estimated**,
not observed — projected from the homography — and they were wrong because
the fourth *fitted* line was mislabelled.

**The structural cause.** `EDGE_NAMES` offered only sideline, goal line and
back line. A cross-field line at 50m therefore had no correct label available,
and the search assigned it (5,6) or (7,8). Labelling a 50m line as the 100m
back line halves the length scale of the entire calibration — and *every gate
still passes*, because a halved field is a perfectly valid rectangle.

Nothing inside one 60-second shot can distinguish these. A goal line, a
halfway line and a back line are all straight cross-field markings; they
differ only in **where they sit along a 100m field**, which is exactly the
information a single shot does not contain.

**Fix 1 — vocabulary.** `(11, 12)` is now a labellable edge. The vertices
already existed in `ultimate_config` (5000,0) and (5000,3700); nothing could
name them. On a shared soccer/ultimate pitch the halfway line is usually the
most clearly painted cross-field marking present, so this is not an edge case.
Re-fitting with the 4th line as `(11,12)` moved the pan from a full-length
claim to x=24..44m — about half the field, matching what the person reported
seeing.

**Fix 2 — `run_camera_map.py`,** which answers the person's actual request:
*"i want to be able to already have known this based on running the entire
video through."* It registers the whole video, links segments that a cut
separated into ONE global frame, and reports each sample's footprint there.
The union of a 92-minute sweep approaches the whole field, so any single shot
becomes a known fraction of it — which is the missing information above.

Validated on the first 8 minutes: 96 samples, 3 segments, all 3 linked
(309 and 512 ORB inliers), 89% of samples placed.

**Measured while building it:** chaining a homography through a segment link
sometimes yields a wildly distorted warp. 90% of footprints came out at
0.99-1.33x the frame's own area; the rest ranged to **579x**, and those alone
inflated the reported sweep from 2.8 frame-widths to 93. A camera cannot
change its footprint area 500-fold, so footprint area is a cheap
assumption-light rejection test. With it, the 8-minute sweep measures
5464 x 2744px = 2.8 frame-widths.

**Still open:** even with the halfway label, the fit puts the pan at
y=-9.7..13.0m across a 37m width, which contradicts both sidelines being
visible. The width scale is still wrong, and the camera map is the intended
way to settle it rather than another round of guessing.

---

## Result 3.29: the window was never calibratable, and the fourth line was never the problem

Four rules the person stated, each after watching an automated score pass a
wrong answer, and each of which turned out to be load-bearing:

1. *"detect the lines that are on the field"*
2. *"if you know that the halfway line is on one side, why would the fourth
   line be in between the goal line"* — cross-field markings exist ONLY at
   x = 0, 18, 50, 82, 100m; a detected line at 30m is proof of an error.
3. *"the near end line would be behind the near goal line"* — side matters.
4. *"they shouldn't be angled enough to be touching the goal line if its an
   end line"* — lines parallel on the field meet only at the vanishing point.

Together (`field_lines.cross_field_candidates`) these cut 34 candidates to 9,
with a stated reason per rejection: 12 ran along the field rather than across
it, 9 were within 120px of the goal line, 3 crossed it inside the field.

**Rule 1 as first written was wrong, and the person's own line caught it.**
Testing membership in the playing-surface mask rejected their confirmed far
sideline at 50% -- because a sideline IS the boundary of the surface. Replaced
with distance to the surface (tolerance 40px at 1920): the far sideline
survives at 60% and nothing real is lost.

**The 120px separation threshold is also wrong** and is left documented rather
than trusted: at this foreshortening two lines 18m apart are only 30-60px
apart, so a pixel threshold cannot express "distinct line". It should be
measured in field metres after a trial fit.

**The actual blocker, measured at last.** Image angles in this framing:

| line | image angle |
|---|---:|
| near sideline (blue) | 45.5 deg |
| goal line | -5.9 deg |
| far sideline | **5.1 deg** |
| any cross-field candidate | 3-6 deg |

Three of the four constraints lie inside an **11 degree fan**. The far sideline
is perpendicular to the goal line on the field but nearly parallel to it in
this image, because its corner sits near the horizon. A homography needs two
well-separated pencils; this framing offers one direction at 45 deg and
everything else bunched near 0.

**Confirmed by the fit's own output.** Given the person's point on the end
line (canvas 700,400) the width finally came out right -- pan y = -1.4..39.4m
of 37m, the first correct width in the window. But:

| line | should land | actually lands |
|---|---|---|
| near sideline | y=0 | y=-0.0..0.0m |
| goal line | x=18 | x=16.7..19.2m |
| far sideline | y=37 | **y=10.9..15.4m, x=40.8..45.6m** |

The calibration's validity limit is 37.4m, so its own input line sits past its
own horizon. Per-line reprojection reports 0.59px and passes, because that is
an image-space test; the field-space placement is meaningless. The same
horizon blindness documented in 3.14, now inside the fit rather than a gate.

**Conclusion:** t=3600+60s cannot be calibrated from its own lines, whatever
the fourth line is. The limit is the camera angle. Every line the person
identified was correct; six rounds of searching for a fourth were spent on a
constraint that did not exist.

**Consequence.** `run_camera_map.py` links all 87 segments of the 92 minutes
into one frame (0 isolated). Clustering the 734 placed footprints gives six
distinct framings spanning 4157px -- 2.2 frame widths of camera travel.
t=3600 is framing 0. The most distant framing with a run long enough to
register is framing 5 at t=3235-3300s, 1085px away, where the sideline that is
edge-on here will not be.

### 3.27 addendum — measured effect of splitting on the ground-truth window

Checked before committing, on t=3945+60s (the only window with human-clicked
ground truth), comparing the SAME physical lines located by geometry rather
than by index:

| | gates | worst reprojection | validity |
|---|---|---:|---:|
| before `split_merged_lines` | 4/4 | 21.8px | 100% |
| after | 4/4 | **11.2px** | 100% |

Splitting halves the worst per-line reprojection by removing contaminating
points from merged clusters, and takes the window under the 20px bank-ready
threshold it previously failed. Note the candidate INDICES change (1/4/7/11
-> 1/4/6/14), so any assignment recorded before this change must be re-read
off the table rather than reused.

---

## Result 3.30: labels transfer between framings; geometry does not

**The person:** *"is there not a way to frame by frame reference the camera
movement and then map on the known lines to the new frame"*.

Yes, and it is the first thing in this whole investigation that generalises.

**Single hop, validated against independent evidence.** Registering t=3600
against t=3235 (335 ORB inliers) and pushing the four human-confirmed lines
through, then checking each against lines the t=3235 window had detected *by
itself*:

| transferred line | matched | angle | offset |
|---|---|---:|---:|
| goal line | cand 0 (136 pts) | 0.6 deg | 10px |
| far sideline | cand 2 (62 pts) | 1.6 deg | 3px |
| end line | cand 8 (35 pts) | 1.8 deg | 9px |
| near sideline | -- | 10.0 deg | 707px |

This is not self-consistency: the destination window found those lines on its
own, and the transfer says which is which. It also CORRECTS the auto-ranker,
which had proposed cand 0 as a sideline and cand 2 as the goal line -- the
labelling the person rejected on sight.

Fitting t=3235 from those three plus the transferred near sideline puts every
line where it belongs: near sideline y=-0.0..0.0, far sideline y=35.3..38.2
(want 37), goal line x=16.8..20.0 (want 18), end line x=-1.4..1.5 (want 0).
The equivalent check at t=3600 had the far sideline landing at y=10.9..15.4
instead of 37.

**Full walk (`run_line_transfer.py`).** 368 steps of 5s from t=3600, reaching
t=2795..4635s -- 1840 seconds, 31 minutes, from ONE labelling. Only 2
registration failures in 368 steps.

| line | median offset | within 30px | no match at all |
|---|---:|---:|---:|
| goal line | 16-43px | 47-57% | 2-5 steps |
| far sideline | 23-34px | 46-61% | 2 steps |
| end line | 62-79px | 23-25% | 2-5 steps |
| near sideline | **208-636px** | 13-29% | **16-68 steps** |

**The near sideline cannot be carried**, exactly as the person predicted:
*"perfectly possible for the near sideline to be off screen or for the angle
to change very drastically considering how close it is to the camera"*. It
swings hardest and leaves frame most often.

**Degradation is not drift.** Offsets hold at 1px for 40 steps, jump to a
median 136px over steps 41-50 where inliers collapse to 302, then RECOVER to
12px. A bad patch is a per-frame registration problem and can be routed
around; accumulated drift could not recover.

**Therefore: transfer for IDENTITY, local detection for GEOMETRY.** A single
hop disagrees with the frame-by-frame chain by 40px mean / 72px max -- ample
to say "that is the goal line", far too loose to fit a calibration from.

**A measurement retracted along the way.** I scored "paint support" under the
transferred lines, got 0-8% against a 4% random baseline, and nearly concluded
that ORB was aligning the treeline rather than the ground plane. The control
then showed the SAME lines scoring 0-8% in their own window -- including the
goal line, a confirmed bright marking with 102 detected points on it. The
metric was broken; the conclusion is withdrawn. Always run the instrument on a
case whose answer is known before believing it on one that is not.

---

## Result 3.32: the far sideline was fitted onto a row of standing people

**The person:** *"i think that youre detecting people in a stack as a sideline
and basing the far sideline before those players instead of behind"* -- and
when I got the direction backwards, *"no the field is too small actually"*.

**MEASURED.** In the t=3235 frame YOLO finds 51 people; the densest collinear
row is **33 of them spanning 1686px**. Under the calibration in use their FEET
read field **y = 33.4m**. People standing out of play are OUTSIDE the field, so
they must read y > 37. Reading 33.4 means the field was drawn about **3.6m
(10%) too narrow**, with the far sideline short of the row rather than past it.

The error also GREW across the frame -- the fitted sideline and the row of feet
cross over at x~400, and by x=1789 the sideline sits 28px nearer the camera
than the people. So the line was at the wrong ANGLE, not merely offset.
Unsurprising: it came from a 2-point transferred line, the weakest constraint
in that fit, contributing zero degrees of freedom of its own.

**Fix.** Refit with the far sideline constrained to the 33-person row, keeping
the 136-point goal line and 35-point end line. The row then reads y=38.6m --
1.6m outside the sideline, a plausible standoff for watchers -- and validity
improves from 40.1m to 44.4m. Confirmed by eye: *"the sideline looks good"*.

Speeds rise 5-25%, the right order for a 10% width correction that affects
only the cross-field component (3fps: team A mean 1.19 -> 1.26 m/s, median
0.79 -> 1.02).

**Why the detector keeps doing this.** A dense row of standing people is long,
straight and high-contrast against turf, so it beats the true sideline on
length, persistence and support -- and the true one is thin and often painted
only in the endzone. `watchers.py` already encoded that a row INDICATES a
sideline. What was missing is that they stand OUTSIDE it, so the line belongs
at or beyond their feet, never short of them.

**Two process failures worth recording, both mine.**

The person reported this same defect at t=3600 much earlier (*"the right
sideline is too far out ... behind the line of players"*). I took their
corrected line there and never checked whether the same error had propagated
into the other calibration. It had, and everything built on it -- the
whole-video overlay, the coverage plan, the speed metrics -- inherited a 10%
cross-field bias.

And when they first raised it here I reasoned from "the row is inside the
field, so the sideline is too far out" and reported the correction backwards.
The measurement was right; the inference from it was not. They corrected it in
four words.

**The sample-count drop, traced and explained.** Speed samples fell (n=72 -> 38
at 1fps) and total on-field detections with them (377 -> 248), which looked
backwards for a "wider" field. Two measurements settle it:

  - the NEAR sideline did not move: blue paint from t=3600 lands at y=-0.02m
    under both the old and new calibration
  - the FAR sideline moved TOWARD the camera, down onto the players' feet

So the field's footprint in the IMAGE shrank, even though the field is now
correctly placed -- fewer detections fall inside it. Meanwhile far fewer are
thrown away past the horizon (342 -> 134) because validity improved from
40.1m to 44.4m. Both effects are real and point opposite ways; neither is a
defect. Not a bug: a direct consequence of the correction.

Re-running the whole-video overlay on the corrected calibration: still 80/93
frames, but median edges per frame rises 3 -> 4, and the blue near sideline now
appears in frame where it previously fell outside the validity region.

## 3.33 The clicked t=2700 calibration was broken, and why

The rectified check rendered black because the fit put the horizon on the field:
`usable_x_max` = 5.7m of 100m. Cause, from the saved `line_details`: four of the
five cross-field lines were traced as near-HORIZONTAL strokes in a pan canvas
where the field runs left-right, so end/goal lines must run steeply down the
image. Only halfway (82 deg) was drawn correctly.

| line | drawn at | should be |
|---|---|---|
| near end | 4.4 deg | across |
| near goal | 10.3 deg | across |
| far goal | 8.7 deg | across |
| far end | 5.9 deg | across |

Leave-one-out confirms the labels are mutually contradictory rather than merely
noisy: dropping the far sideline moves usable 5.7% -> 64%, dropping the near
sideline -> 55.9%, but both leave a >2500px or NaN worst-case error.

Root cause is upstream: the t=2700 canvas is 58% painted and its right-hand 40%
(the far end of the field) only ~37%, so the lines had to be guessed.

## 3.34 Both sidelines dominate calibration conditioning

Synthetic test on `calibration_t1700` (the only calibration with all 7 lines
inside its validity region), 2px Gaussian click noise, 3 points per line, all 30
four-line subsets, scored by predicting the 3 lines NOT clicked:

- all 10 subsets containing BOTH sidelines: median 7.1 - 47.1px
- all 20 subsets containing ONE sideline: median 284 - 1769px

The separation is complete -- no one-sideline subset beat any both-sideline
subset. Scaling the number of lines (2px noise, worst case over random subsets):
4 lines median 304-1312px, 5 lines 59-105px, 6 lines 21-34px, 7 lines 8.7-14.3px.
Extra points per line help far less than extra lines.

Caveat: with noiseless clicks 4 lines reproduce the held-out lines to 0.000px --
4 lines determine H exactly. All the error above is noise amplification, so these
numbers describe conditioning, not bias.

Acted on in `run_click_corners.py`: the fit is refused when either sideline is
missing (shift-F overrides). An angle-based orientation guard was also added here
and is RETRACTED in 3.35 below -- do not reinstate it. A broken fit (usable < 20%) saves to
`calibration_clicked_REJECTED.json` so it cannot be picked up by mistake, and
prints a leave-one-out table. Region past the validity limit now renders dark
grey with a caption instead of black.

## 3.35 RETRACTION: the 25-degree orientation guard was wrong

3.33 concluded the bad t=2700 lines could be recognised by being drawn within
25 deg of the sideline family. That test is invalid, and it is the same
angle-based mistake already recorded against line grouping: image angle is not
preserved under perspective, and near the horizon a cross-field line genuinely
flattens toward the sideline direction.

Checked against four real calibrations, sampling each line only inside its own
validity region and comparing with that calibration's own sideline family:

| calibration | sidelines | cross lines within 25 deg (i.e. wrongly refused) |
|---|---|---|
| t1700 (valid 100m) | 26.7 deg | near goal 12.6, far goal 9.3, far end 10.9, halfway 4.0 |
| t3235 rowfit (44m) | 15.5 deg | near goal 6.8, near end 10.1 |
| t3600 (37m) | 25.3 deg | near goal 19.4, near end 23.4 |
| from_lines (89m) | 46.2 deg | none |

9 of the 13 measurable valid lines would have been refused. The guard now always
returns [] and is documented in place so it is not reintroduced.

Vanishing-point concurrency was tried as a principled replacement and also fails:
the five bad t=2700 cross lines are concurrent to 1.56 deg, which is inside
ordinary click noise (calibration-generated lines score 0.000 deg only because
they come from a homography by construction). There is no reliable geometric
signature of this mistake in the clicks themselves.

What does work is outcome-based and was already in place: a mislabelled set
produces a degenerate homography, and the validity region measures it directly.
Across every calibration in the project the separation is clean -- 27.9%, 37.4%,
44.4%, 49.3%, 89.1%, 100% for fits believed sound, against 5.7% for the bad
clicked one. The <20% rejection rule keeps that test and drops the heuristics.

## 3.36 First independently verified calibration (t=3235)

The ultimate field is coned; the soccer pitch beneath it is painted, and its
centre circle is 9.15m radius under the Laws of the Game for every pitch size.
Those pixels are never used to fit an ultimate-field calibration, so measuring
them through one is genuine verification rather than a circular test.

Method (`run_circle_check.py`): threshold white paint on grass, map image->field
with H, RANSAC a circle in FIELD coordinates -- where a correct homography leaves
the circle circular and the straight lines straight, so the two separate.

`calibration_t3235_rowfit`, anchor frame t=3235s:

- radius **9.09 m vs regulation 9.15 m (-0.6%, 6 cm)**, 2290 inlier px
- circle-fit residual RMS 7.6 cm
- radius by 30-degree sector: 9.04, 9.16, 9.04, 9.08, 9.17, 9.02 m -- spread 1.6%
- fitted centre (47.1, 17.2) m, against an ultimate-field centre of (50.0, 18.5),
  consistent with the ultimate field being laid out on the soccer pitch
- inliers visually confirmed to trace the painted arc

This is the first independent metric verification in the project.

### Two traps, both hit

`cv2.fitEllipse` and an algebraic conic fit BOTH reported axis ratio ~1.18 on
these points, which reads as an 18% distorted plane. Both are wrong: the visible
paint covers only the left half of the circle, and on a half arc an ellipse fit
is unconstrained. Per-sector radius is the honest shape test -- it varies 1.6%
where a 1.18 ellipse would vary 18%. Residual does NOT discriminate: simulation
shows a true 1.18 ellipse over this arc leaves 6.9 cm circle residual against the
7.6 cm observed.

The test could not be applied to `calibration_t1700` or `calibration_t3600`. Runs
there returned 25.45m and 8.34m "circles" with excellent sector spreads (0.4%,
1.1%) and plausible footprints, and were on inspection a few specks near the
halfway line and a row of orange cones. No automatic check separated these from
the real detection -- footprint, sector spread, residual and radius band all
passed. The inlier overlay is therefore written every run and must be inspected
before the radius is quoted. t=1700 and t=3600 remain UNVERIFIED, not disproven.

## 3.37 Soccer geometry measured through the verified calibration

Using `calibration_t3235_rowfit` (verified in 3.36), all white paint in the anchor
frame was mapped to field coordinates and straight lines extracted by RANSAC,
scoring each by continuity (fraction of 1m bins along it that contain points)
rather than by raw min-max extent, which stray inliers inflate.

The strongest through-centre line: 2470 px, direction 87.6 deg, and it passes
**0.21 m from the fitted centre-circle centre**. The soccer halfway line is a
diameter of the centre circle, so this is a real corroboration of the
calibration: the circle centre and the halfway line are independent image
features, and nothing in the fit (ultimate sidelines, near goal, near end) forced
them to be collinear.

Derived soccer-pitch pose, in ultimate-field coordinates:

- centre (47.1, 17.2) m, against an ultimate-field centre of (50.0, 18.5)
- width 66.8 m (regulation 64-75, commonly 68) -- approximate: the halfway line
  had only 30% bin coverage, so treat this as +/- a metre, not surveyed
- long axis rotated -2.4 deg from the ultimate long axis

So the two fields are close to co-aligned. This is the transform needed to use
soccer landmarks as constraints for an ultimate-field calibration.

### The paint-hit metric is broken -- do not use it (second occurrence)

The predicted soccer touchlines (never used in any fit) were scored by the
fraction of sampled points landing within 6px of paint: 58%. The control kills
it. Sliding the SAME line sideways:

| offset | -8m | -4m | -2m | 0 | +2m | +4m | +8m |
|---|---|---|---|---|---|---|---|
| on paint | 42% | 40% | 55% | 58% | 70% | 87% | 94% |

The score rises monotonically the further the line is displaced outward, because
outward is toward the horizon, where white clutter (the standing row of people,
fencing, tents) saturates the mask. There is no peak at the correct position, so
the metric carries no localisation information near the horizon.

This is the same metric withdrawn earlier in this project for the same reason. It
has now failed twice. Any future "fraction of predicted line landing on paint"
claim must be accompanied by this offset sweep, and is void without a peak at
zero offset.

The predicted touchlines therefore remain UNVALIDATED. The soccer-pitch pose
above rests only on features inside the 44m validity region.

## 3.38 Fitting from soccer geometry: works, but one frame does not determine the plane

`run_soccer_fit.py` fits an image->field homography by optimising 8 DOF so the
centre-circle paint maps to radius 9.15m and the halfway line maps to x=0.

Two things had to be right:

- **Circle pixels must be filtered to the ellipse.** Taking every paint pixel in a
  box around the circle left 53% contamination (player kit, turf seam lines) and
  the optimiser settled at 57cm circle rms on a badly wrong plane. Iterating
  `fitEllipse` and keeping pixels within 6px drops that to 3.8-7.4cm.
- **Conditioning was NOT the problem.** Normalising pixel coordinates and switching
  to trf/x_scale='jac' reproduced the bad fit to three decimals; the contamination
  was the whole story.

**The blocking result:** circle + halfway is 7 constraints (conic 5, line 2) for 8
DOF, so it does not determine the plane. A long marking above the circle supplies
an 8th via its direction, and the two variants reach the SAME circle residual
while disagreeing badly about the plane:

| constraints | circle rms | that marking lands at |
|---|---|---|
| circle + halfway | 3.8 cm | 21.12 m from centre |
| + its direction | 7.4 cm | 12.33 m from centre |

With the direction constraint the answer is stable (bootstrap sd 0.02m; perturbing
the perspective term enough to double the circle residual moves it only
11.94..12.54m) -- but stability is not correctness, and the two constraint sets
differ by 9m.

That marking is also **not the far touchline**: a touchline sits 32-37.5m from the
pitch centre, and no rescaling reconciles 12.3m (it would need a 25m centre
circle). It runs parallel to the pitch length 12.3m off centre -- most likely one
of the dashed multi-sport markings; its straightness residual (24.9cm) is worse
than the halfway line's (12cm), consistent with a dashed line.

Conclusion: soccer-based fitting is sound in method but needs a landmark with
known geometry away from the centre -- a touchline, penalty box or goal line. The
centre circle alone fixes scale locally and leaves the far field free. This is
precisely why the published pipelines (PnLCalib, nbjw_calib) regress many pitch
keypoints rather than fitting a circle.

## 3.39 PnLCalib: verified on wide views, fails on tight ones

Installed `third_party/PnLCalib` (gitignored; clone + weight URLs are in
`.gitignore`). `run_pnlcalib.py` runs it and returns an image->field homography in
pitch-centred metres, taking columns 0,1,3 of the 3x4 projection.

**Extraction trap:** rebuilding the rotation from pan/tilt/roll via
`pan_tilt_roll_to_orientation` gives a different orientation from the solver's own
`rotation_matrix`, and put the pitch centre 2342m away. `inference.py` reads
`rotation_matrix`; so must we.

Scored the same independent way as everything else -- measure the PAINTED centre
circle through the calibration. The network regresses keypoints and never sees
that constraint, so this is a real check:

| frame | view | radius (truth 9.15m) | rms | circle centre (pitch centre 0,0) |
|---|---|---|---|---|
| t=1700 | wide, full circle | **9.111 m** (-4cm) | 5.0 cm | (-0.08, +0.06) m |
| t=3235 | tight, partial | 11.238 m (+209cm) | 46.4 cm | (+2.55, +1.12) m |

So PnLCalib is excellent on wide views and wrong on tight partial ones, matching
the overlays exactly (at t=3235 its halfway line sits ~150px off the paint). Its
operating envelope must be checked per frame, not assumed.

### Mutual verification, and a correction to 3.38

3.38 measured a long marking above the circle at 12.33m from the pitch centre and
called that well-determined (bootstrap sd 0.02m). PnLCalib puts the same line at
**19.17 m**. The two disagree by 7m, which confirms 3.38's own caveat: that fit
was one constraint short and its stability was not correctness. Prefer 19.17.

19.17m also identifies the line. From the verified `calibration_t3235_rowfit`, the
soccer centre sits at ultimate coordinates (47.1, 17.2), so the ultimate field's
FAR SIDELINE at y=37 lies 19.8m from the soccer centre. PnLCalib, on a different
frame by a completely different method, measures that marking at 19.17m -- a 0.6m
agreement between two independent calibrations, and it identifies the line as the
ultimate far sideline (the one the standing row stands behind).

Caveat: that marking's straightness through PnLCalib is 255cm rms, so its pixel
set is contaminated by the standing row; 19.17m carries real uncertainty. The
agreement is corroborating, not surveyed.

## 3.40 A ground line must land on turf -- PnLCalib fails this in the far field

Observation from the user, checking the PnLCalib overlay: the far sideline looks
like it is in the sky. It is.

Measuring the turf coverage of each image row at t=1700 (green AND smooth; low
local texture separates turf from foliage) gives a clean step -- rows 400+ are
53-98% turf, rows above ~360 are 0-35% (track, benches, trees). Against that:

| PnLCalib ground line | image rows | on the playing surface? |
|---|---|---|
| centre (0m) | 505-614 | yes |
| circle far edge (9.2m) | 400-500 | just barely |
| the tested marking (19m) | 319-411 | mostly above it |
| **far touchline (34m)** | **236-320** | **no -- in the trees** |

**This is the check the circle test cannot do.** The centre circle spans 9.15m
around the pitch centre, so its radius certifies the plane LOCALLY. PnLCalib
scored 9.111m (-4cm, 5.0cm rms) at t=1700 and is still projecting the touchline
into the foliage. Local metric accuracy does not imply far-field accuracy, and
every verification in 3.36-3.39 was local.

**This weakens 3.39's "mutual verification".** That result had PnLCalib measuring
a marking at 19.17m and the verified t=3235 calibration predicting the ultimate
far sideline at 19.8m, agreeing to 0.6m. But PnLCalib compresses depth -- it puts
34m at row 280 while the turf does not even start until ~380 -- so if that marking
is really the soccer far touchline at 34m, then 19.17m is an underestimate and the
agreement is two methods sharing a bias, not confirming each other. Treat 3.39's
agreement as unconfirmed.

Finding the turf edge took four attempts; the three failures are recorded in
`run_ground_check.py` so they are not repeated (green hue alone puts the surface
in the sky, since trees are green; largest-green-component-touching-the-bottom
bridges turf to trees under closing and claims 89% of the frame; per-column walks
in either direction stop at painted lines, players and shadows).

## 3.41 The verified calibration passes the ground check; PnLCalib is the outlier

Applying 3.40's test to `calibration_t3235_rowfit` on its anchor frame.

The automated row profile put the surface band at rows 404-436 and flagged the
soccer 34m touchline (rows 348-377) and the ultimate near end (370-400) as above
the surface. **Visual inspection overrules that**: in this frame the turf clearly
reaches row ~345 before meeting the hedge, and all three projected lines land on
grass -- the soccer touchline along the top edge of the turf, the ultimate far
sideline on the standing row the user confirmed, the near end line on grass.

The row profile fails here because a dense row of standing people, two goals and
the hedge sit exactly at the boundary and depress turf coverage across rows
345-405. So `surface_band` is only trustworthy where the boundary is clear; where
people crowd it, it reads 40-60 rows low. Use it to rank candidates, confirm by
eye -- the same rule that has applied to every check in this project.

Scale of the two cases, which is what separates them:

| case | projected line | surface starts | margin |
|---|---|---|---|
| PnLCalib t=1700 | far touchline rows 236-320 | ~390 | 70-150px above, in the trees |
| t=3235 rowfit | soccer touchline rows 348-377 | ~345 | on the turf edge |

So the far-field failure is PnLCalib's, not a property shared by every homography
here, and the lens-distortion hypothesis is no longer needed to explain the
t=3235 calibration. `calibration_t3235_rowfit` now passes both a local metric
check (3.36, circle radius 9.09m) and a far-field check, and remains the anchor.

3.39's 19.17m vs 19.8m agreement stays UNCONFIRMED but for a narrower reason: one
side of it (PnLCalib) is demonstrably wrong in the far field, so the agreement is
more likely coincidence than mutual verification.

## 3.42 Why the end line and touchline are right and the far sideline is not

User, reading the ground-check overlay: the touchline and end line are correct,
everything else is not. The constraint inventory of `calibration_t3235_rowfit`
explains this exactly.

| fitted line | points | direction it constrains |
|---|---|---|
| near goal | 136 | along-field |
| near end | 35 | along-field |
| far sideline | 33 | across-field |
| near sideline | **2** | across-field |

The along-field constraints are strong (136 and 35 points), which is why the near
END line projects correctly.

The across-field constraints are both unusable:

- the near sideline is a 2-point transferred line. Two points define a line
  exactly, so it contributes no redundancy and cannot correct anything.
- the far sideline's 33 points are the standing row's feet (3.32). Mapped through
  the calibration they span ultimate y = **29.70 to 47.91 m**, median 45.28 -- an
  18m spread for something being treated as a straight line, with 8 of 17 sampled
  points reading INSIDE the 37m field, which is impossible for spectators.

So the ultimate field's placement ACROSS the pitch rests on one phantom
constraint and one 18m-wide smear. That is the whole of the far sideline's
support, and it is why it lands wrong.

The soccer touchline projects correctly because it does not come from those
constraints at all: it derives from the soccer pose (3.37), which was built from
the centre circle and halfway line -- both independent of the ultimate sidelines.

**Status correction.** 3.36 and 3.41 stand as far as they go: the calibration is
locally metric (circle radius 9.09m) and its lines land on turf. Neither test
constrains the ultimate field's across-pitch offset, and 3.42 shows that offset is
unsupported. `calibration_t3235_rowfit` should be described as a verified SOCCER
plane carrying an unverified ultimate-field placement, not as a verified ultimate
calibration.

Next: stop fitting the ultimate sidelines from cones and people. Place the
ultimate field inside the soccer frame instead, where the geometry is reliable,
using a trustworthy observation of where one ultimate sideline actually is.

## 3.43 The blue NEAR sideline is painted and detectable -- and never coincides with PnLCalib

Correction to labelling used earlier in this file: the FAR sideline is marked with
blue CONES; the NEAR sideline is a painted BLUE LINE. Several earlier notes have
these the wrong way round.

That matters because a painted line is densely detectable by colour, which is
exactly the across-field constraint 3.42 found missing. Detector: blue hue
95-130, saturation >80, restricted to pixels near turf (green AND low texture),
then the largest elongated component.

19 of 54 frames sampled across the video carry a line-like blue segment. Quality
where it appears is excellent:

| t | pixels | image span | elongation | straightness |
|---|---|---|---|---|
| 4200s | 3040 | 464 px | 49 | **2.5 px** |
| 2900s | 2332 | 341 px | 41 | 2.3 px |
| 600s | 2744 | 297 px | 26 | 3.1 px |
| 4700s | 1656 | 210 px | 20 | 2.8 px |

Straightness of 2-3px over hundreds of pixels is a far stronger constraint than
either of 3.42's across-field inputs (a 2-point phantom and an 18m-wide smear of
people). Every detection runs off the bottom edge of the frame, so only a
fragment of the sideline near the camera is ever visible.

**Two blocking structural facts:**

1. **Blue line and PnLCalib never co-occur.** Of 10 frames tested, none had both:
   where the blue line is visible (600, 800, 1400, 3600, 4000, 4200, 4400, 5200)
   PnLCalib returned nothing, and where PnLCalib worked (400, 5000) no blue line
   was found. The line is visible when the camera is tight on the near side,
   which is exactly when too little of the soccer pitch is in frame for the
   keypoint model. So the soccer plane cannot be carried to the blue line through
   PnLCalib in a single frame.

2. **The anchor window cannot be repaired in place.** In t=3235-3300 the blue line
   appears in only 1 of 14 sampled frames (t=3240), as a 99x65px corner fragment;
   registered into the anchor frame (SIFT, 653/838 inliers) it lands at
   x=-178..-61, off the frame entirely. So `calibration_t3235_rowfit`'s 2-point
   near sideline cannot be replaced with blue-line evidence from its own window.

Consequence: fixing the across-field geometry means building a NEW calibration in
a window where the blue line is long -- t=4200 is the best (3040px, span 464px,
straightness 2.5px) -- not repairing the existing anchor.

## 3.44 t=4200 fit attempt: correct formulation, features too dirty

User identified the t=4200 candidates: #1 is the soccer centre circle, #3 and #4
sit on the goal line, #0 is the blue painted near sideline.

**The constraint count works out.** In ultimate coordinates, near sideline (y=0)
gives 2, goal line (x=Xg) gives 2, and the centre circle with known 9.15m radius
but unknown centre gives 5-2=3, totalling 7 for 8 DOF -- one short, the same
shortfall as 3.38. It closes by fixing the soccer centre's ALONG-field position
at x=47.13m, which comes from the anchor's well-constrained direction (136+35
points), and solving for its ACROSS-field position, which is the value 3.42 showed
is unsupported. That makes it exactly determined and turns the across-field offset
into an output that tests the anchor's suspect 17.21m.

**The features are not clean enough.** Results:

| goal line assumed | sideline \|y\| | goal \|dx\| | circle rms | recovered offset |
|---|---|---|---|---|
| x=18m | 5.7 cm | 46.9 cm | 29.3 cm | +30.63 m |
| x=82m | 3.9 cm | 63.8 cm | 41.5 cm | +32.07 m |

Circle rms of 29-42cm against the 3.8-7.4cm that clean fits reach (3.38, 3.36)
means this is not trustworthy, and the two goal-line hypotheses are too close in
quality to tell which endzone is in view. The recovered offset (~31m vs the
anchor's 17.21m) is a large disagreement but the fit is too poor to call it a
correction.

**Why the features are dirty:**

- The goal-line region holds three fragments. One runs at -64 deg, 75 deg off the
  other two -- not part of the line, and including it initially drove circle rms
  to 68cm and the offset to +5.50m. Filtering by angle removed it.
- The two real fragments still differ by 6.83 deg in direction, and one's centroid
  sits 29.5px off the other's line. Combined they fit a line at 8.5px rms, far too
  loose for thin paint. Growing the line by absorbing nearby white pixels made it
  worse (a 2058px blob over 412px of span, 13.1px rms).
- The arc is roughly 90 degrees, and 3.36 established that partial arcs admit
  badly wrong conic fits.

Hough merging was also tried and failed: at a loose tolerance it fused unrelated
segments into 1900px lines running through the sky (the bug was comparing
unsigned offsets); with signed offsets and a tight tolerance no line over 200px
survives at all, confirming the fragments are genuinely not collinear.

Conclusion: the formulation is right and the window is wrong. What this needs is
either a frame where the goal line is cleanly visible over a long span, or hand
clicking along the goal line and sideline, where a human beats this detector.

## 3.45 t=4200 with clean features: the across-field offset is unidentifiable

User corrected 3.44: the goal line is candidate **#3 only**, not #3 and #4. That
explains the 6.83 deg discrepancy between them -- they are different lines.

With #3 alone, trimmed to inliers, the features are clean:

- goal line: 405px, span 176px, straightness **1.50px**
- blue near sideline: 2151px, straightness **1.60px**
- centre-circle arc: ~90 deg

Residuals improve a lot, and the goal line is identified as x=18m (the near
endzone) because x=82 fits far worse:

| goal line assumed | sideline \|y\| | goal \|dx\| | circle rms |
|---|---|---|---|
| **x=18 m** | 2.19 cm | 18.99 cm | **9.75 cm** |
| x=82 m | 2.59 cm | 53.89 cm | 28.48 cm |

**But the recovered across-field offset is worthless.** Bootstrapping the same
pixels: mean +17.47m, **sd 14.97m**, range -8.54 to +29.10m. A single fit returned
+28.85m, which looked like an 11.6m disagreement with the anchor's 17.21m; it is
not a disagreement, it is noise. The bootstrap mean landing on 17.47m is
coincidence at that spread, not confirmation.

Projecting the single fit shows the same thing qualitatively: the far soccer
touchline lands at rows 227-329 against a surface starting at 384-448 -- above the
turf, the exact failure mode 3.40 caught in PnLCalib.

**Why counting arguments were not enough.** 3.44 established 8 constraints for 8
DOF by fixing the soccer centre's along-field position and solving for its
across-field one. That is exactly determined on paper, but numerically the offset
is near-unidentifiable: a ~90 deg arc pins the circle's radius and its position
along the arc's own axis, and barely constrains the centre in the perpendicular
direction -- which is precisely the direction being solved for. Exact
determination in the DOF count says nothing about conditioning.

Consequence: the across-field placement of the ultimate field still cannot be
recovered from any single frame tried so far. It needs a frame carrying BOTH
sidelines, or a much larger arc, or hand-clicked points on the far sideline.

## 3.46 A second sideline makes the offset identifiable -- but P is not it

Context the user supplied, which reframes the scene: the camera sits at the MIDDLE
OF A SIDELINE and rotates. So looking toward an endzone, the near sideline recedes
steeply up-frame while the far sideline ascends gently toward the same vanishing
point, and a cross-field line (goal line) descends. Line families are told apart
by which way they tilt relative to the sideline's vanishing point, NOT by image
angle.

I dismissed candidate P (565px at 168.8deg) as "across-field, not along" because
its image angle was far from the blue sideline's 128deg. **That is exactly the
error retracted in 3.35** -- parallel field lines converge, so their image angles
differ by construction. Repeated within the same session.

**The structural result is real.** Adding any second along-field line collapses
the across-field offset from unidentifiable to pinned:

| constraints | bootstrap sd of the offset |
|---|---|
| near sideline + goal line + arc | **14.97 m** |
| + a second sideline | **0.12 m** |

So one sideline never determines the width, and two always do. This is the same
conclusion as 3.34 (both sidelines dominate conditioning), now confirmed on real
data rather than synthetic.

**But P is not the far sideline.** Forcing P to y=37 wrecked every residual (far
sideline 432cm, near sideline 79cm vs 2.69cm, circle rms 39cm). Letting its
across-field position float instead, with everything else well fitted (near
sideline 2.69cm, goal line 17cm, circle rms 11.7cm), P lands at **y = -0.34 m** --
on the near sideline itself. P is part of the near sideline, not a second line.
Its field-plane straightness is 85cm and it survives trimming down to 89 of 608
pixels, so it is a poor feature regardless.

The far sideline at t=4200 is painted in the endzone only and has not been located
by any detector run so far.

## 3.47 Clicked calibration at t=4200: excellent fit, 9% scale gap against the circle

`run_click_lines.py` was built so the user could mark lines directly. Five lines
clicked at t=4200, and the far sideline turned out to be at the upper LEFT
(544-780, 360-386) -- every detector search had been on the right.

Fit from the five ultimate lines only, with the soccer circle held back:

| line | max reprojection |
|---|---|
| near sideline | 0.88 px |
| far sideline | 0.28 px |
| goal line | 0.97 px |
| end line | 0.60 px |
| halfway | 0.33 px |

Sub-pixel on all five. Validity 50.7m (51%). Saved as
`calibration_t4200_clicked.json`.

**The labels are right.** Testing every monotonic assignment of legal x values to
the three cross-field lines, the user's (halfway 50, goal 18, end 0) wins at
0.97px against 12.83px for the next best. Sideline swaps and end-for-end
reflections give identical scale, as they must.

**But the independent check disagrees by 9%.** The clicked centre-circle arc maps
to radius **8.318 m** against a regulation 9.15 m (-9.1%), with only 4.2cm
residual -- the arc is a clean circle, just the wrong size. Click noise does not
explain it: perturbing the six points by 1px moves the radius by sd 0.14m, so the
0.83m gap needs ~6px of error.

This contradicts the t=3235 anchor, which measured the same physical circle at
9.09m from 2290 real paint pixels. Both fits use the same 100x37x18 model, so the
two homographies disagree about the world by 9%.

Two explanations remain, and reprojection cannot separate them: a uniform scale
error is invisible to reprojection, since scaling the field model just rescales H.

1. the field is ~10% larger than 100x37 with 18m endzones
2. this pitch's centre circle is not 9.15m

Attempts to resolve it by solving for the dimensions failed: with both width and
endzone depth free the optimiser collapsed the field to zero width (degenerate,
all residuals 0); bounded, the width ran to its 20m limit with circle rms blowing
from 4.2cm to 80cm, and forcing width=37 gave rms 121cm. The clicked lines and a
9.15m circle are genuinely inconsistent.

Resolvable by clicking one more object of known size -- the soccer goal in frame
is 7.32m between posts. If the fit reads it ~9% small the scale is wrong; if it
reads 7.32m the circle is anomalous.
