# Team Classification Finding (measured on real footage)

Prompted by a scope simplification: skip per-player identity (measured hard —
`results/tracking_finding.md`, ~90% track fragmentation) and classify each
detection into one of the two teams instead. Team is a per-frame,
per-detection property — it doesn't need cross-frame identity continuity at
all, so this sidesteps the fragmentation problem entirely, in principle.

Deliberately not using `roboflow/sports`' `TeamClassifier` (SigLIP-based) —
needs `transformers`' torch backend, measured unavailable on this platform
(torch capped at 2.2.2 on Intel macOS; `transformers` needs ≥2.5; see
`requirements.txt`). Testing plain color clustering instead.

## Method

Real frame (t=320s, 69 YOLO detections, `imgsz=1920`). Four variants, each a
reaction to what the previous one got wrong — k-means (k=2, unsupervised, no
pre-specified team colors) on:

1. **Mean BGR** of a torso crop (middle-third height, full width) per box.
2. **Mean HSV hue+saturation** (drop brightness/V — lighting-sensitive)
   instead of BGR.
3. Same as (2), restricted to the larger half of detections by box area (a
   crude proxy for "closer/more confident crop, less likely a tiny distant
   bystander") with a tighter crop (middle 50% width too, not just height).
4. **Saturation-weighted dominant hue**: histogram-mode hue among the
   higher-saturation pixels in the crop, instead of a plain mean — a plain
   mean gets dragged by any contamination (skin, shadow, background) inside
   the box; a saturation-weighted mode is more robust to that.

## Result

| variant | separation ratio (other/own distance) | visual quality |
|---|---|---|
| 1. mean BGR | 3.0 | poor — same-team players split across clusters |
| 2. mean HSV hue+sat | 2.8 | poor — same issue persists |
| 3. filtered + tighter crop | 2.8 | poor — no real improvement |
| 4. saturation-weighted dominant hue | 2.7 | **better for the maroon team, still weak for white** |

The separation-ratio number barely moved across variants (2.7–3.0) — a
misleading signal on its own. **The real difference showed up visually**:
variant 4 correctly grouped a clearly-visible cluster of maroon-jersey
players together for the first time; variants 1–3 split same-team players
inconsistently, confirmed by eye each time (`outputs/team_classify_*.jpg`,
local only — real people, not committed).

## Root cause

Two compounding problems, not one:
1. **White jerseys have near-zero saturation.** Saturation-weighting (which
   is what fixed the maroon team) actively down-weights white/grey pixels —
   it's built to suppress noise, but for a genuinely low-saturation team,
   that suppresses the signal along with the noise. A technique tuned to
   find a *colored* team's hue is structurally weak at representing "no
   color" as its own class.
2. **The detection set includes non-players.** Sideline coaches, bench
   players, and spectators (grey shirts, black shirts, backpacks) get
   clustered right alongside the two actual team colors, adding noise
   neither team should have to compete with.

## Fifth variant: split saturation first, THEN cluster hue — this one works

Tested the root-cause fix directly instead of just recommending it: bucket
detections into low-saturation (white/grey candidate) vs high-saturation
(colored-jersey candidate) by a simple **median-saturation threshold** first
— a binary split, not a joint hue+saturation clustering — then (for a
2-team, one-white/one-colored game) that threshold split IS the team split.

**Result: qualitatively correct.** The maroon TORO group (right side of
frame) is now consistently classified together, AND the white WEST trio
(middle of frame) is now consistently classified together — both groups that
every previous variant split inconsistently. Confirmed by eye
(`outputs/team_classify_sat_split.jpg`, local only).

This makes sense given the root cause diagnosis above: treating "is this
white or colored" as its own binary decision, separate from "what hue is
the colored one," stops the near-zero-saturation white team from being
outvoted or distorted by a joint clustering step that's implicitly built
around finding hue variation.

**Caveat**: this specific threshold rule (low-sat = white team) is a
reasonable default for a white-vs-colored-jersey matchup like this game, but
won't generalize as-is to a colored-vs-colored matchup (e.g. two different
saturated jersey colors) — that case still needs hue clustering, just
applied AFTER separating out any genuinely low-saturation (white/grey)
outliers like bystanders first, not instead of it.

## Verdict and recommendation

**The team-classification simplification works, with the saturation-first
approach — not the naive joint color clustering.** Shipped in
`run_team_classify_test.py`. Remaining known contamination: sideline/bench
people still get misclassified into one team or the other (visible in the
bottom of the annotated frame) — that's the same fix already queued up:
**filter to on-field players first** via `is_in_field()`
(`frisbee_analysis/calibration.py`), waiting on the user running
`run_calibrate.py`.

Don't revisit `TeamClassifier`/SigLIP on this machine — the platform
constraint isn't going away without different hardware.
