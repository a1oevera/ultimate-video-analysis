# Conversation Log (condensed)

The design thread that produced this package, in order, so Claude Code has the
full reasoning chain. This is narrative context; the authoritative technical
state is in README / NEXT_STEPS / CONTEXT.

1. **Goal set.** AI analysis of ultimate footage: points, passes, completion
   rates, turnovers, speeds, play time. Priority (person's ranking):
   turnovers > passes/completion > play time > speed.

2. **Difficulty triage.** Metrics split into three tiers: player tracking/speed
   (tractable), play time (needs identity — hard), passes/turnovers (needs
   possession — research-grade, because it hinges on the disc).

3. **Footage Qs.** Person: existing footage only; fixed high mount that
   pans/tilts/zooms; often only part of field visible; some lined, some coned;
   bird's-eye-ish.

4. **Full-automation pushback.** Person initially wanted zero manual input.
   Argued against: research teams chose manual holder annotation over automatic
   disc detection; a little manual input (pull tagging) makes identity tractable
   and yields ground truth. Person revised to "tag once, track through game."

5. **"Tag once" pushback.** Trackers don't preserve identity through occlusion;
   ultimate stacks are worst-case. Fix: re-anchor at each pull + per-tracklet
   global assignment + 7-per-team constraint + physics gating in field coords.

6. **How soccer does it.** Researched: multi-camera rigs + human operators +
   per-tracklet jersey-number voting. Transferable bit = per-tracklet identity,
   not per-frame. TrackLab/sn-gamestate exist. Ideas raised: second camera, GPS
   watches as identity/ground-truth, height-from-homography feature.

7. **Pro footage as baseline?** Good for detector pretraining, wrong for
   validation (different camera/kit/resolution). UFATrack pro data IS fine for
   Track A (coordinate-space). Pointed to UFATrack/UltimateTrack.

8. **DeepStream?** Rejected — streaming/causal, conflicts with offline global
   decode, solves a throughput problem the project doesn't have, needs NVIDIA
   GPU.

9. **roboflow/sports.** Read the source. Lift ViewTransformer + TeamClassifier +
   pitch-config pattern. Do NOT use their soccer weights or naive BallTracker.
   It collapses Phases 1/4/8 to "configure"; does nothing for possession/
   identity (their README lists those as unsolved).

10. **UltimatePitchConfiguration built.** WFDF geometry (verified vs WFDF
    2025-28 rules: 100x37m, 18m endzones, brick 18m from goal, 8 cones) in the
    sports config pattern. Only ~10 detectable keypoints (cones+bricks) vs
    soccer's ~32 — a real constraint.

11. **Moving/zoomed/partial-field camera.** Fixed mount = pure rotation = clean
    homography, BUT partial-field framing means per-frame calibration often lacks
    points. Solution: build a mosaic once, register each frame to it via
    background features. Feasibility depends on background richness (test it).

12. **UFATrack loader.** Pulled the real repo (open-starlab/UFATrack, CC-BY-4.0):
    20 CSVs, one per possession, 14 players + disc, columns incl. class
    (offense/defense/disc) and holder (bool). Wrote + verified the loader.

13. **THE VIABILITY TEST (this package's payoff).** Ran the pivot-rule test on
    real ground truth: holder is slowest offense player only 40% of the time
    (top-3 67%). Moderate signal. First-pass HMM detector: transition F1 ~0.36,
    0/5 turnovers. Verdict: OPEN, leaning cautious — needs tuning + maybe disc
    detection before committing to the CV build.

Key habit demonstrated throughout: measure, don't assume. Two of my own
predictions (pivot signal strong; marking feature helpful) were corrected by the
data. Keep doing that.
