import logging
import time
from enum import Enum

import cv2
import numpy as np

from donkeycar.utils import clamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable defaults. Every value here can be overridden by setting the same
# CENTER_LINE_<NAME> constant in myconfig.py -- this Part reads
# `cfg.CENTER_LINE_<NAME>` if present and falls back to the DEFAULT_<NAME>
# value below otherwise, so it runs out of the box with zero config changes.
# See donkeycar/templates/cfg_cv_control.py for the commented reference
# block and center_line_follower.md for what each one does and how to tune
# it moving from DonkeySim to the real car.
# ---------------------------------------------------------------------------

# Region of interest: a tall horizontal band (not a thin slice), so a dash
# fragment is likely to intersect it even across gaps in the tape.
DEFAULT_ROI_Y_TOP = 130
DEFAULT_ROI_Y_BOTTOM = 230

# HSV threshold for the greenish-blue/teal tape (opencv hue is 0..179).
# Widened from the original 75-105 to 75-130: measured directly off this
# same track's tape (a separate session, same lighting) gave hue~99-100,
# sat~119-149, val~125-152 -- comfortably inside 75-105 already, but the
# tape reads as light blue (not teal) under some lighting, which skews
# slightly higher in hue, hence the wider ceiling.
DEFAULT_COLOR_LOW = (75, 80, 40)
DEFAULT_COLOR_HIGH = (130, 255, 255)

DEFAULT_MORPH_KERNEL = 5
DEFAULT_MIN_AREA_FRACTION = 0.005

# Reject a contour whose area fills more than this fraction of its own
# axis-aligned bounding box. A real tape stripe is thin and elongated
# relative to its bounding box (measured ~0.43-0.59 on real dash/edge
# blobs); a false-positive match on the plain floor -- e.g. daylight
# washing the floor's saturation down low enough to pass the white-edge
# gate above -- comes back as a big, roughly-square blob instead (measured
# ~0.84 on exactly that failure case). Applies to both the center dash and
# the edge search.
DEFAULT_MAX_FILL_RATIO = 0.65

# Reject a contour whose solidity (contour area / convex-hull area) is
# below this. Real tape is a smooth, nearly fully-convex stripe (measured
# ~0.90 on a synthetic stripe matched to our real dimensions); foliage/
# shrubs threshold into small, irregular, spread-out highlight clusters
# with real gaps between them, which stay far less convex even after the
# same morphological CLOSE this pipeline already applies (measured ~0.76
# on a synthetic sparse-foliage pattern built specifically to survive that
# CLOSE step). fill_ratio (above) catches compact BLOBS; this catches
# irregular, non-blob SCATTER that fill_ratio's bounding-box test doesn't
# distinguish from a real stripe -- foliage can pass one without the
# other, so both run.
DEFAULT_MIN_SOLIDITY = 0.85

# Local-contrast (top-hat) detection for the solid white lane-boundary edge
# tape. Only used in LEFT/RIGHT (half-lane) mode.
#
# A fixed brightness/saturation threshold can't reliably separate the tape
# from the floor here: both are similarly low-saturation gray, and plain
# floor brightness swings a lot across the scene (measured V~92-103 in one
# shaded patch, ~183-190 in a sunlit one elsewhere in the same frame) --
# wide enough to overlap with the tape's own measured brightness (V~159-165
# in that same shaded patch). A global threshold that's high enough to
# exclude the sunlit floor would also miss the tape in shade, and vice
# versa. What *did* separate cleanly in real measurements: the tape is
# reliably ~55-60 V brighter than the floor immediately next to it, in
# every patch checked, regardless of that patch's absolute brightness.
# That's a local-contrast property, not an absolute one, so detection uses
# a morphological white top-hat (original minus its own grayscale-opening)
# instead of cv2.inRange: top-hat keeps only features smaller than
# EDGE_TOPHAT_KERNEL_SIZE that are brighter than their own immediate
# surroundings -- exactly a thin bright ridge against whatever the local
# floor happens to be -- and zeroes out any region of uniform brightness,
# however bright, the same way it zeroes plain floor.
DEFAULT_EDGE_TOPHAT_KERNEL_SIZE = 21   # must be notably wider than the ~1-4px tape line, but smaller than the scale lighting varies over
DEFAULT_EDGE_CONTRAST_THRESHOLD = 30   # margin below the measured ~55-60 real contrast, above the ~0 seen on plain floor
DEFAULT_EDGE_MIN_AREA_FRACTION = 0.003
# Minimum share of the ROI's HEIGHT a blob must span to count as the solid
# lane-boundary edge. The boundary recedes from the camera so it crosses the
# whole ROI top-to-bottom; shadows lie across the path and cover only a band.
# Measured here: real edge 0.97 of ROI height, dappled shadows 0.27-0.46.
DEFAULT_EDGE_MIN_HEIGHT_FRACTION = 0.60

# Solidity floor for the boundary edge, kept separate from the dash's
# DEFAULT_MIN_SOLIDITY. 0.0 disables it: a curved boundary measured 0.63,
# well under the dash's 0.85, and shadows measured 0.50-0.80 -- so solidity
# rejects real tape without excluding the thing it was meant to exclude.
# Shadow rejection is handled by DEFAULT_EDGE_MIN_HEIGHT_FRACTION instead.
DEFAULT_EDGE_MIN_SOLIDITY = 0.0

# Minimum elongation (long side / short side of the min-area rect) for the
# boundary edge. Rejects gravel/pebble beds, which pass the height gate
# because they fill the whole ROI but are blobby rather than line-like.
# Measured: real edge 8.6-13.4, pebble bed 2.3-3.2.
DEFAULT_EDGE_MIN_ELONGATION = 5.0

# Tracking continuity: real tape moves smoothly frame-to-frame as the car
# drives; a spurious background match (a rock, a glint, a piece of
# architecture that happens to pass the color/contrast+shape gates) shows
# up at an inconsistent, unrelated position instead. So once something has
# been tracked, prefer whichever candidate is *closest to the last known
# position* over just picking the largest blob -- this holds even when a
# background false positive is larger than the real tape (verified against
# a simulated sequence where a spurious blob 8x the real tape's area still
# didn't hijack tracking). MAX_TRACK_JUMP_PX guards the case where the real
# tape is genuinely gone and a background blob is the *only* candidate: if
# even the closest candidate is further than this from the last known
# position, treat it as not-found (letting the existing gap-tolerance
# state machine handle it) rather than snapping onto the wrong feature.
DEFAULT_MAX_TRACK_JUMP_PX = 60

# Frames of *consistent* position required before trusting a brand-new lock
# (no prior confirmed position to gate against yet -- startup, or right
# after being fully lost). This is specifically the gap MAX_TRACK_JUMP_PX
# doesn't cover: with no prior position, _find_blob falls back to "largest
# qualifying blob", and a small background speckle (pavement texture, a
# pebble) can have a similar confidence to real tape in a single frame --
# measured directly on real track footage, a noise speck and a genuine
# detection both landed around 0.02-0.03 confidence, so area/confidence
# alone can't tell them apart. What does distinguish them: real tape holds
# roughly the same position across consecutive frames as the car drives;
# random noise doesn't recur in the same spot. So a fresh lock only gets
# trusted (and reported as found) once the same position (within
# MAX_TRACK_JUMP_PX) has appeared for this many consecutive frames in a
# row -- any gap or jump to a different position resets the count. Once
# genuinely confirmed, MAX_TRACK_JUMP_PX's own proximity gating takes over
# and every subsequent frame is trusted immediately.
DEFAULT_CONFIRM_FRAMES = 3

# Initial guess at the pixel distance between the center dash and a side
# edge when in half-lane mode, refined automatically once both are seen
# together in the same frame. Only matters for the first few frames or
# while one side is temporarily occluded -- see _resolve_track_point().
DEFAULT_HALF_LANE_WIDTH_PX = 150
DEFAULT_HALF_LANE_WIDTH_SMOOTHING = 0.2

# Where in the dash->edge span to aim, in half-lane (LEFT/RIGHT) mode:
#   0.0 = ride directly on the center dash
#   0.5 = midway between dash and edge (the original half-lane behavior)
#   1.0 = ride on the outer white edge
# Lower values "hug" the dash. That matters on corners: the dash is what
# tracking depends on, and the further the car sits from it the sooner it
# swings out of the ROI mid-turn -- at which point steering falls back to
# the width estimate and drifts. Hugging trades some lane centering for
# keeping the thing being tracked inside the frame.
DEFAULT_DASH_HUG = 0.5

DEFAULT_TARGET_PIXEL = None  # None => geometric center of the frame

DEFAULT_STEER_KP = 0.8
DEFAULT_STEER_KD = 0.0
DEFAULT_ERROR_SMOOTHING_ALPHA = 0.5

DEFAULT_THROTTLE = 0.2
DEFAULT_THROTTLE_LOST_MIN = 0.0

DEFAULT_HOLD_TIME_SEC = 0.5
DEFAULT_LOST_TIME_SEC = 2.0

DEFAULT_OVERLAY_IMAGE = True


class LaneMode(str, Enum):
    """
    Which half of the track (if any) to lane-keep within, using the solid
    white boundary tape on that side plus the center dash, instead of
    tracking the center dash alone.

    CENTER: original behavior, unchanged -- track the center dash only,
    steer it toward the frame's horizontal center.

    LEFT/RIGHT: track the center dash AND the white edge on the given
    side; steer their midpoint toward the frame's horizontal center, so
    the car sits centered within that half of the track rather than
    straddling the whole track width. See
    CenterLineFollower._resolve_track_point() for the fusion/fallback
    logic (what happens when only one of the two is visible).
    """
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class CenterLineFollower:
    """
    Classical-CV line follower for intermittent greenish-blue/light-blue
    center tape, with an optional half-lane mode (see LaneMode) that also
    tracks the solid white edge on one side so the car can lane-keep
    within half the track instead of needing to straddle its full width.

    Pipeline: crop to a region of interest -> HSV color threshold -> clean
    up the mask -> find the tape's (and, in half-lane mode, the edge's)
    contour and centroid -> steer proportionally to how far the tracked
    point is from the target pixel, at a constant throttle. Degrades
    gracefully when nothing is found: holds the last steering/throttle
    briefly (dashed-tape gaps), then ramps throttle down, then stops if
    it's been missing too long. See center_line_follower.md for the full
    pipeline write-up and tuning guide.

    Constructor takes (pid, cfg) so this is a drop-in replacement for
    donkeycar.parts.line_follower.LineFollower in
    donkeycar/templates/cv_control.py's add_cv_controller(): point
    CV_CONTROLLER_MODULE / CV_CONTROLLER_CLASS at this module/class in
    myconfig.py and nothing else needs to change. `pid` is accepted only
    for constructor-signature compatibility -- this class implements its
    own proportional(+derivative) control below instead.
    """

    # Gap-tolerance state machine states (see _apply_gap_tolerance()).
    TRACKING = "TRACKING"
    HOLD = "HOLD"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"

    def __init__(self, pid, cfg, lane_mode=None):
        self._pid = pid  # unused; kept only for add_cv_controller()'s fixed (pid, cfg) call signature

        # --- Region of interest ---
        self.roi_y_top = getattr(cfg, 'CENTER_LINE_ROI_Y_TOP', DEFAULT_ROI_Y_TOP)
        self.roi_y_bottom = getattr(cfg, 'CENTER_LINE_ROI_Y_BOTTOM', DEFAULT_ROI_Y_BOTTOM)

        # --- Detection (center dash) ---
        self.color_low = np.asarray(getattr(cfg, 'CENTER_LINE_COLOR_LOW', DEFAULT_COLOR_LOW))
        self.color_high = np.asarray(getattr(cfg, 'CENTER_LINE_COLOR_HIGH', DEFAULT_COLOR_HIGH))

        # --- Morphology ---
        k = getattr(cfg, 'CENTER_LINE_MORPH_KERNEL', DEFAULT_MORPH_KERNEL)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # --- Contour filtering (center dash and, in half-lane mode, edge) ---
        self.min_area_fraction = getattr(cfg, 'CENTER_LINE_MIN_AREA_FRACTION', DEFAULT_MIN_AREA_FRACTION)
        self.max_fill_ratio = getattr(cfg, 'CENTER_LINE_MAX_FILL_RATIO', DEFAULT_MAX_FILL_RATIO)
        self.min_solidity = getattr(cfg, 'CENTER_LINE_MIN_SOLIDITY', DEFAULT_MIN_SOLIDITY)

        # --- Detection + contour filtering (half-lane white edge) ---
        k_edge = getattr(cfg, 'CENTER_LINE_EDGE_TOPHAT_KERNEL_SIZE', DEFAULT_EDGE_TOPHAT_KERNEL_SIZE)
        self.edge_tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_edge, k_edge))
        self.edge_contrast_threshold = getattr(cfg, 'CENTER_LINE_EDGE_CONTRAST_THRESHOLD', DEFAULT_EDGE_CONTRAST_THRESHOLD)
        self.edge_min_area_fraction = getattr(cfg, 'CENTER_LINE_EDGE_MIN_AREA_FRACTION', DEFAULT_EDGE_MIN_AREA_FRACTION)
        self.edge_min_elongation = max(0.0, float(getattr(
            cfg, 'CENTER_LINE_EDGE_MIN_ELONGATION', DEFAULT_EDGE_MIN_ELONGATION)))
        self.edge_min_solidity = clamp(float(getattr(
            cfg, 'CENTER_LINE_EDGE_MIN_SOLIDITY', DEFAULT_EDGE_MIN_SOLIDITY)), 0.0, 1.0)
        self.edge_min_height_fraction = clamp(float(getattr(
            cfg, 'CENTER_LINE_EDGE_MIN_HEIGHT_FRACTION',
            DEFAULT_EDGE_MIN_HEIGHT_FRACTION)), 0.0, 1.0)

        # --- Half-lane width estimate (dash-to-edge pixel distance) ---
        self.half_lane_width_px = getattr(cfg, 'CENTER_LINE_HALF_LANE_WIDTH_PX', DEFAULT_HALF_LANE_WIDTH_PX)
        self.half_lane_width_smoothing = getattr(
            cfg, 'CENTER_LINE_HALF_LANE_WIDTH_SMOOTHING', DEFAULT_HALF_LANE_WIDTH_SMOOTHING)
        # How closely to hug the center dash vs. sit mid-lane (see
        # DEFAULT_DASH_HUG and _resolve_track_point()). Clamped so a bad
        # config value can't push the target outside the lane entirely.
        self.dash_hug = clamp(
            float(getattr(cfg, 'CENTER_LINE_DASH_HUG', DEFAULT_DASH_HUG)), 0.0, 1.0)

        # --- Tracking continuity (see _find_blob()) ---
        self.max_track_jump_px = getattr(cfg, 'CENTER_LINE_MAX_TRACK_JUMP_PX', DEFAULT_MAX_TRACK_JUMP_PX)
        self.last_dash_cx = None
        self.last_edge_cx = None

        # --- Fresh-lock confirmation (see _confirm() and DEFAULT_CONFIRM_FRAMES) ---
        self.confirm_frames = getattr(cfg, 'CENTER_LINE_CONFIRM_FRAMES', DEFAULT_CONFIRM_FRAMES)
        self.pending_dash_cx = None
        self.pending_dash_count = 0
        self.pending_edge_cx = None
        self.pending_edge_count = 0

        # --- Control ---
        self.target_pixel = getattr(cfg, 'CENTER_LINE_TARGET_PIXEL', DEFAULT_TARGET_PIXEL)
        self.steer_kp = getattr(cfg, 'CENTER_LINE_STEER_KP', DEFAULT_STEER_KP)
        self.steer_kd = getattr(cfg, 'CENTER_LINE_STEER_KD', DEFAULT_STEER_KD)
        self.error_smoothing_alpha = getattr(cfg, 'CENTER_LINE_ERROR_SMOOTHING_ALPHA', DEFAULT_ERROR_SMOOTHING_ALPHA)

        # --- Throttle ---
        self.throttle_constant = getattr(cfg, 'CENTER_LINE_THROTTLE', DEFAULT_THROTTLE)
        self.throttle_lost_min = getattr(cfg, 'CENTER_LINE_THROTTLE_LOST_MIN', DEFAULT_THROTTLE_LOST_MIN)

        # --- Gap tolerance ---
        self.hold_time_sec = getattr(cfg, 'CENTER_LINE_HOLD_TIME_SEC', DEFAULT_HOLD_TIME_SEC)
        self.lost_time_sec = getattr(cfg, 'CENTER_LINE_LOST_TIME_SEC', DEFAULT_LOST_TIME_SEC)

        # --- Debug --- (unprefixed/shared: cv_control.py itself also reads
        # cfg.OVERLAY_IMAGE directly to decide what to show in the web UI)
        self.overlay_image = getattr(cfg, 'OVERLAY_IMAGE', DEFAULT_OVERLAY_IMAGE)

        # --- Lane mode (see LaneMode). An explicit constructor kwarg wins
        # (useful for tests); otherwise myconfig.py's CENTER_LINE_LANE_MODE
        # controls it directly, since add_cv_controller() in cv_control.py
        # always calls this with a fixed (pid, cfg) signature and never
        # passes lane_mode itself. set_lane_mode() remains available for a
        # future runtime toggle (button/webpage) on top of this default. ---
        self.lane_mode = LaneMode(lane_mode or getattr(cfg, 'CENTER_LINE_LANE_MODE', LaneMode.CENTER))

        # --- Runtime state ---
        self.state = self.TRACKING
        self.steering = 0.0
        self.throttle = 0.0
        self.last_seen_time = time.time()
        self.smoothed_error = 0.0
        self.prev_smoothed_error = 0.0

    def set_lane_mode(self, lane_mode):
        """Public hook for a future lane-select button/webpage toggle."""
        self.lane_mode = LaneMode(lane_mode)

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
    def _make_mask(self, roi_rgb):
        """
        HSV color threshold. Saturation -- not hue -- is what separates the
        teal tape from the white boundary tape/track floor: white/gray
        surfaces are low-saturation regardless of lighting, while the teal
        tape is a deliberately vivid, high-saturation material. See
        center_line_follower.md for the full explanation.
        """
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)  # cam_img is RGB, not BGR
        return cv2.inRange(hsv, self.color_low, self.color_high)

    def _make_edge_mask(self, roi_rgb):
        """
        Local-contrast (top-hat) detection for the solid white lane-
        boundary tape -- see DEFAULT_EDGE_TOPHAT_KERNEL_SIZE's comment for
        why this isn't a plain HSV threshold. Only called in LEFT/RIGHT
        (half-lane) mode -- see run().
        """
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)  # cam_img is RGB, not BGR
        v = hsv[:, :, 2]
        ridge = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, self.edge_tophat_kernel)
        _, mask = cv2.threshold(ridge, self.edge_contrast_threshold, 255, cv2.THRESH_BINARY)
        return mask

    def _clean_mask(self, mask):
        # OPEN (erode-then-dilate) removes small noise specks; CLOSE
        # (dilate-then-erode) fills small holes, e.g. a glare streak.
        # Order matters: OPEN first, so a stray noise speck can't fuse to
        # the real blob before it gets a chance to be removed.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return mask

    def _find_blob(self, mask, min_area_fraction, x_range=None, last_known_cx=None,
                   min_height_fraction=0.0, min_solidity=None, min_elongation=0.0):
        """
        Returns (cx, cy, confidence) in ROI-local coordinates (cx is also
        full-frame-x, since the ROI spans the full image width) for
        whichever contour passing the area and shape filters is closest to
        last_known_cx (tracking continuity -- see DEFAULT_MAX_TRACK_JUMP_PX
        for why), or the largest one if last_known_cx is None (no prior
        track to continue, e.g. startup or after an extended loss). Returns
        None if nothing qualifies, or if even the closest candidate is
        further than max_track_jump_px from last_known_cx. Restricted to
        columns x_range=(x0, x1) if given (used to search only one half of
        the frame for a side edge). confidence is a genuine fraction of the
        searched area matched, not a raw pixel count.
        """
        search_mask = mask
        if x_range is not None:
            search_mask = np.zeros_like(mask)
            x0, x1 = x_range
            search_mask[:, x0:x1] = mask[:, x0:x1]

        contours, _ = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        search_area = search_mask.shape[0] * (x_range[1] - x_range[0] if x_range else search_mask.shape[1])
        min_area_px = min_area_fraction * search_area

        candidates = []  # (cx, cy, confidence)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_px:
                continue
            _, _, bw, bh = cv2.boundingRect(c)
            # Vertical-span gate. A lane boundary runs AWAY from the camera,
            # so in the ROI it stretches from bottom to top; a shadow cast
            # ACROSS the path is a horizontal band that covers only a slice
            # of the ROI's height. Measured on real frames from this track:
            #     real white edge : bbox height = 0.97 of the ROI height
            #     dappled shadows : 0.46, 0.41, 0.30, 0.27
            # This is the only reliable separator found -- top-hat contrast
            # cannot do it (shadows measured HIGHER contrast than tape, max
            # 143 vs ~102, and are brighter: V 201 vs 171), and elongation
            # cannot either (tape 7.7 vs shadows 5.8-11.1, fully overlapping).
            # Only applied where a caller asks for it (the edge search); the
            # dashed center line is short by design and must not be gated.
            if min_height_fraction > 0.0 and bh < min_height_fraction * mask.shape[0]:
                continue
            # Elongation gate (edge only). Boundary tape is one long thin
            # stripe; a gravel/pebble bed shatters into a field of roughly
            # round clumps that the CLOSE step fuses into fat blobs. Measured
            # on real tub frames:
            #     real white edge : elongation 8.6 and 13.4  (1 blob,  ~600px)
            #     pebble bed      : elongation 2.3 and  3.2  (19-23 blobs, 4000-5900px)
            # minAreaRect is used rather than the axis-aligned box so a
            # diagonal line still measures as elongated. This is complementary
            # to min_height_fraction: height rejects shadows (horizontal
            # bands), elongation rejects gravel (fills the ROI but is blobby).
            if min_elongation > 0.0:
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                shorter = max(min(rw, rh), 1.0)
                if (max(rw, rh) / shorter) < min_elongation:
                    continue
            if bw * bh > 0 and area / (bw * bh) > self.max_fill_ratio:
                continue  # too blob-like to be a thin tape stripe
            # Solidity floor, overridable per caller. The default (tuned for
            # the center dash) is deliberately strict to reject scattered
            # foliage. It must NOT be applied as-is to the boundary edge: a
            # long CURVED line has low solidity by construction, since its
            # convex hull is far larger than the curve itself. Measured on a
            # real daytime frame the white edge scored 0.63 and was being
            # thrown away by the dash's 0.85 floor at every threshold, which
            # is why daytime edge detection failed completely. Solidity also
            # cannot separate shadows here (they measured 0.50-0.80, straddling
            # the edge's 0.63) -- the vertical-span gate does that instead.
            solidity_floor = self.min_solidity if min_solidity is None else min_solidity
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if solidity_floor > 0.0 and hull_area > 0 and area / hull_area < solidity_floor:
                continue  # too irregular/scattered to be a smooth tape stripe (likely foliage)
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            candidates.append((M['m10'] / M['m00'], M['m01'] / M['m00'], area / search_area))
        if not candidates:
            return None

        if last_known_cx is None:
            # No prior track to continue (startup, or been lost too long) --
            # largest surviving blob is the best available guess.
            return max(candidates, key=lambda t: t[2] * search_area)

        cx, cy, confidence = min(candidates, key=lambda t: abs(t[0] - last_known_cx))
        if abs(cx - last_known_cx) > self.max_track_jump_px:
            return None  # even the closest candidate is implausibly far -- treat as not-found
        return cx, cy, confidence

    def _find_line(self, mask):
        return self._find_blob(mask, self.min_area_fraction, last_known_cx=self.last_dash_cx)

    def _find_edge(self, mask, side, width):
        """side: 'left' or 'right' -- restricts the search to that half of
        the frame, so the opposite edge (or the center dash, which can also
        pass the saturation gate at its brightest) can't be mistaken for
        the near edge."""
        mid = width // 2
        x_range = (0, mid) if side == 'left' else (mid, width)
        return self._find_blob(mask, self.edge_min_area_fraction, x_range=x_range,
                               last_known_cx=self.last_edge_cx,
                               min_height_fraction=self.edge_min_height_fraction,
                               min_solidity=self.edge_min_solidity,
                               min_elongation=self.edge_min_elongation)

    def _confirm(self, found, confirmed_cx_attr, pending_cx_attr, pending_count_attr):
        """
        Require CONFIRM_FRAMES consecutive, mutually-consistent detections
        before trusting a *fresh* lock (confirmed_cx_attr currently None)
        -- see DEFAULT_CONFIRM_FRAMES for why. Once genuinely confirmed (or
        already tracking), _find_blob's own proximity gating is already
        protecting against background hijacking, so every subsequent
        frame's candidate is trusted immediately -- this only guards the
        moment tracking starts from scratch.

        Returns `found` unchanged if it should be trusted this frame, or
        None if it's still pending confirmation (or nothing was found).

        A miss during the pending phase does NOT reset progress -- e.g. a
        normal gap in the dashed center line would otherwise make
        confirmation nearly impossible to ever complete (every gap wipes
        out whatever consecutive streak had built up, so a genuinely real,
        stationary tape could get stuck re-starting its confirmation count
        forever and never actually lock on). Only a *different* position
        (inconsistent with the pending one) resets it -- that's still the
        actual noise-rejection signal; an absence of detection isn't.
        """
        already_confirmed = getattr(self, confirmed_cx_attr) is not None
        if found is None:
            return None

        cx = found[0]
        if already_confirmed:
            return found

        pending_cx = getattr(self, pending_cx_attr)
        pending_count = getattr(self, pending_count_attr)
        if pending_cx is not None and abs(cx - pending_cx) <= self.max_track_jump_px:
            pending_count += 1
        else:
            pending_count = 1
        setattr(self, pending_cx_attr, cx)
        setattr(self, pending_count_attr, pending_count)

        if pending_count >= self.confirm_frames:
            return found
        return None  # still building confidence -- don't trust yet

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _get_target_x(self, width):
        """
        The physical steering target is always the frame's horizontal
        center in every lane mode -- that's where the car's forward axis
        is, regardless of which half of the track it's centering within.
        What varies by mode is which detected feature(s) estimate *where
        that center currently is*; see _resolve_track_point().
        """
        if self.target_pixel is not None:
            return float(self.target_pixel)
        return width / 2.0

    def _resolve_track_point(self, dash_found, edge_found):
        """
        Combine whichever of (center dash, side edge) were found this
        frame into a single tracked x position, updating the running
        half-lane-width estimate whenever both are seen together. Returns
        (track_x, cx_for_overlay, cy_for_overlay).

        CENTER mode never calls this with edge_found set (run() skips edge
        detection entirely in that mode), so it always reduces to plain
        dash-centroid tracking -- identical to the original
        center-line-only behavior.

        LEFT/RIGHT mode:
          - both found: track_x is their midpoint (centers the car within
            the half-lane); also refines half_lane_width_px.
          - only the dash found (edge occluded, e.g. mid-curve): estimate
            the half-lane center from the dash plus the running width
            estimate, on the side the missing edge should be.
          - only the edge found (dash in a dashed-tape gap): symmetric
            estimate from the edge instead. This is the main practical
            payoff of half-lane mode over plain center-following: the
            solid edge is continuous, so it can carry tracking through
            every gap in the dashed center line, not just short ones.
        """
        # dash_hug is where in the dash->edge span the car aims:
        #   0.0 = sit right on the dash, 0.5 = lane midpoint (original
        #   behavior), 1.0 = sit on the outer edge.
        # Hugging the dash keeps it nearer the middle of the frame, so it
        # stays in view further into a turn instead of sliding out of the
        # ROI and dropping tracking mid-corner.
        hug = self.dash_hug

        if dash_found is not None and edge_found is not None:
            dash_cx, dash_cy, _ = dash_found
            edge_cx, edge_cy, _ = edge_found
            width_px = abs(edge_cx - dash_cx)
            self.half_lane_width_px += self.half_lane_width_smoothing * (width_px - self.half_lane_width_px)
            # Interpolate from the dash toward the edge; signed difference
            # handles the edge being on either side without a sign term.
            track_x = dash_cx + hug * (edge_cx - dash_cx)
            return track_x, track_x, (dash_cy + edge_cy) / 2.0

        if dash_found is not None:
            dash_cx, dash_cy, _ = dash_found
            if self.lane_mode == LaneMode.CENTER:
                return dash_cx, dash_cx, dash_cy
            sign = 1.0 if self.lane_mode == LaneMode.RIGHT else -1.0
            track_x = dash_cx + sign * hug * self.half_lane_width_px
            return track_x, dash_cx, dash_cy

        edge_cx, edge_cy, _ = edge_found
        # Coming from the edge instead, the target sits (1 - hug) of the
        # span back toward where the dash should be.
        sign = -1.0 if self.lane_mode == LaneMode.RIGHT else 1.0
        track_x = edge_cx + sign * (1.0 - hug) * self.half_lane_width_px
        return track_x, edge_cx, edge_cy

    def _steer_towards(self, cx, width):
        target_x = self._get_target_x(width)
        # Normalized so Kp doesn't need retuning if resolution changes
        # between sim and the real camera. Positive error/steering = tape
        # (and correction) to the right.
        error = clamp((cx - target_x) / (width / 2.0), -1.0, 1.0)

        alpha = self.error_smoothing_alpha
        self.smoothed_error = alpha * error + (1 - alpha) * self.smoothed_error
        derivative = self.smoothed_error - self.prev_smoothed_error
        self.prev_smoothed_error = self.smoothed_error

        return clamp(self.steer_kp * self.smoothed_error + self.steer_kd * derivative, -1.0, 1.0)

    def _apply_gap_tolerance(self):
        """
        Don't overwrite steering/throttle unless the tape is confidently
        detected -- a brief gap holds the last command exactly (dashed tape
        is expected, not a fault); only once a gap has run on far longer
        than any single dash-gap plausibly should does throttle ramp down
        and eventually stop. Steering is always just held, never reset.
        """
        elapsed = time.time() - self.last_seen_time
        if elapsed <= self.hold_time_sec:
            self.state = self.HOLD
        elif elapsed <= self.lost_time_sec:
            self.state = self.DEGRADED
            t = (elapsed - self.hold_time_sec) / (self.lost_time_sec - self.hold_time_sec)
            self.throttle = self.throttle_constant * (1.0 - t) + self.throttle_lost_min * t
        else:
            # Only reset state on the actual transition INTO stopped, not
            # on every subsequent tick this method gets called while
            # already stopped -- this method runs again on every future
            # miss once elapsed > lost_time_sec, and re-wiping the pending-
            # confirmation counters every single one of those ticks would
            # erase _confirm()'s progress before it could ever reach
            # CONFIRM_FRAMES, permanently deadlocking recovery the instant
            # STOPPED was first reached (found in the field: pending_count
            # stuck at 1 forever, never incrementing, even with highly
            # consistent real detections).
            was_already_stopped = (self.state == self.STOPPED)
            self.state = self.STOPPED
            self.throttle = self.throttle_lost_min
            if not was_already_stopped:
                # Fully lost -- forget the last known position so tracking
                # resumes fresh (largest-blob fallback) once something is
                # found again, rather than gating against a now-stale position.
                self.last_dash_cx = None
                self.last_edge_cx = None
                self.pending_dash_cx = None
                self.pending_dash_count = 0
                self.pending_edge_cx = None
                self.pending_edge_count = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, cam_img, lane_mode=None):
        """
        input: cam_img, an RGB numpy array; lane_mode, an optional LaneMode
        (or its string value) -- the seam a future lane-select button will
        drive via CV_CONTROLLER_INPUTS, unused this stage.
        output: steering, throttle, and the (possibly annotated) image.
        """
        if cam_img is None:
            return self.steering, self.throttle, cam_img

        if lane_mode is not None:
            self.set_lane_mode(lane_mode)

        width = cam_img.shape[1]
        roi = cam_img[self.roi_y_top:self.roi_y_bottom, :, :]

        dash_mask = self._clean_mask(self._make_mask(roi))
        dash_found_raw = self._find_line(dash_mask)
        dash_found = self._confirm(dash_found_raw, 'last_dash_cx', 'pending_dash_cx', 'pending_dash_count')
        if dash_found is not None:
            self.last_dash_cx = dash_found[0]

        edge_mask = None
        edge_found = None
        if self.lane_mode != LaneMode.CENTER:
            edge_mask = self._clean_mask(self._make_edge_mask(roi))
            side = 'left' if self.lane_mode == LaneMode.LEFT else 'right'
            # Blank the half we are not tracking. _find_edge() already limits
            # its SEARCH to `side` via x_range, but the mask itself stayed
            # full-width and the debug overlay ORs it in -- so bright
            # pavement on the far side still showed up tinted and looked
            # like the edge detector was chasing it. Zeroing it keeps the
            # overlay honest about what can actually influence steering, and
            # guarantees the far side contributes nothing even if the search
            # window is ever widened.
            mid = width // 2
            if side == 'right':
                edge_mask[:, :mid] = 0
            else:
                edge_mask[:, mid:] = 0
            edge_found_raw = self._find_edge(edge_mask, side, width)
            edge_found = self._confirm(edge_found_raw, 'last_edge_cx', 'pending_edge_cx', 'pending_edge_count')
            if edge_found is not None:
                self.last_edge_cx = edge_found[0]

        cx = cy = None
        if dash_found is not None or edge_found is not None:
            track_x, cx, cy = self._resolve_track_point(dash_found, edge_found)
            self.state = self.TRACKING
            self.last_seen_time = time.time()
            self.steering = self._steer_towards(track_x, width)
            self.throttle = self.throttle_constant
        else:
            # Reset the derivative baseline while not tracking, so the D
            # term doesn't spike from comparing against a stale error once
            # the line is reacquired.
            self.prev_smoothed_error = self.smoothed_error
            self._apply_gap_tolerance()

        cv_img = cam_img
        if self.overlay_image:
            cv_img = self._draw_overlay(cam_img, dash_mask, edge_mask, cx, cy)

        return self.steering, self.throttle, cv_img

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, cam_img, dash_mask, edge_mask, cx, cy):
        """Composite the mask(s) onto the ROI and burn in state/steering/throttle."""
        img = np.copy(cam_img)

        combined_mask = dash_mask if edge_mask is None else cv2.bitwise_or(dash_mask, edge_mask)
        mask_rgb = np.zeros_like(img[self.roi_y_top:self.roi_y_bottom, :, :])
        mask_rgb[:, :, 1] = combined_mask  # green tint
        img[self.roi_y_top:self.roi_y_bottom, :, :] = cv2.addWeighted(
            img[self.roi_y_top:self.roi_y_bottom, :, :], 0.6, mask_rgb, 0.4, 0)
        cv2.rectangle(img, (0, self.roi_y_top), (img.shape[1] - 1, self.roi_y_bottom - 1), (255, 0, 0), 1)

        if cx is not None:
            cv2.drawMarker(img, (int(cx), int(self.roi_y_top + cy)), (0, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        display_str = [
            f"STATE:{self.state}",
            f"LANE:{self.lane_mode.value}",
            f"STEERING:{self.steering:.2f}",
            f"THROTTLE:{self.throttle:.2f}",
        ]
        y = 10
        for s in display_str:
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y += 10

        return img
