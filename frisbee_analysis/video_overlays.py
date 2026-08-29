"""Static broadcast-overlay masks, one entry per known source video.

MEASURED BUG (see results/calibration_finding.md): overlay boxes used to be
selected by frame resolution alone (e.g. "1920x1080 -> ojuc.mp4's boxes").
That broke the moment a second video shared a resolution with a different
overlay layout -- `videos/torovcanyon.mp4` is also 1920x1080 but has a
top-left livestream scoreboard + bottom-right "veo" watermark, nothing like
ojuc.mp4's top scoreboard band + bottom captions band. Keying by resolution
would silently mask the wrong regions (losing real field content where
nothing needs hiding) while leaving the real overlay unmasked (injecting
spurious "always aligned" ORB matches that don't reflect real camera
motion) -- same failure class as the earlier ojuc/videoplayback resolution
collision, just not yet triggered because only one video existed per
resolution at the time.

Fix: key by the video's basename ONLY, never by resolution -- a resolution
"fallback" that guesses another video's boxes for an unrecognized file
reintroduces this exact bug for the next resolution collision (e.g. a
future video that also happens to be 1920x1080), just deferred instead of
fixed. An unrecognized video gets NO mask and a loud warning instead --
worse than a correct mask, but not actively misleading the way a wrong
guessed mask would be (masking real field content while leaving the actual
overlay unmasked).
"""
import os

# (x0, y0, x1, y1) boxes, in that video's own native frame coordinates.
_BY_BASENAME = {
    # Full-width top band, not a tighter corner box around the crisp watermark
    # logo -- MEASURED (see run_mosaic_visualize.py history): a second, larger
    # semi-transparent "ROGERS tv" watermark shows up intermittently higher up
    # and wider than the crisp corner logo, and a small box missed it. y=230
    # stays above the treeline in this footage's wide shots.
    "ojuc.mp4": [
        (0, 0, 1920, 230),     # top scoreboard band + both watermark variants
        (0, 860, 1920, 950),   # closed-captions band
    ],
    "videoplayback.mp4": [
        (0, 295, 640, 360),    # UFA score bug + ticker, bottom band
    ],
    "torovcanyon.mp4": [
        (0, 0, 750, 150),        # top-left "LIVE <clock> TOR n-n CAN" scoreboard
        (1600, 950, 1920, 1080),  # bottom-right "veo" watermark
    ],
}


def overlay_boxes_for(video_path: str, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
    name = os.path.basename(video_path)
    if name in _BY_BASENAME:
        return _BY_BASENAME[name]
    print(f"WARNING: no known overlay mask for '{name}' ({frame_w}x{frame_h}) -- add one to "
          f"frisbee_analysis/video_overlays.py if this video has a static broadcast graphic, "
          f"or registration may be corrupted by it.")
    return []
