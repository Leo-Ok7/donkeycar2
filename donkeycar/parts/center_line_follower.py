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
DEFAULT_COLOR_LOW = (75, 80, 40)
DEFAULT_COLOR_HIGH = (105, 255, 255)

DEFAULT_MORPH_KERNEL = 5
DEFAULT_MIN_AREA_FRACTION = 0.005

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
    Lane-select seam for the future lane-following stage (white boundary
    tape, manually toggled from a web UI button). Not implemented yet --
    LEFT/RIGHT currently behave identically to CENTER, see
    CenterLineFollower._get_target_x().
    """
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class CenterLineFollower:
    """
    Classical-CV line follower for intermittent greenish-blue center tape.

    Pipeline: crop to a region of interest -> HSV color threshold -> clean
    up the mask -> find the tape's contour and its centroid -> steer
    proportionally to how far that centroid is from the target pixel, at a
    constant throttle. Degrades gracefully when the tape isn't found: holds
    the last steering/throttle briefly (dashed-tape gaps), then ramps
    throttle down, then stops if it's been missing too long. See
    center_line_follower.md for the full pipeline write-up and tuning guide.

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

    def __init__(self, pid, cfg, lane_mode=LaneMode.CENTER):
        self._pid = pid  # unused; kept only for add_cv_controller()'s fixed (pid, cfg) call signature

        # --- Region of interest ---
        self.roi_y_top = getattr(cfg, 'CENTER_LINE_ROI_Y_TOP', DEFAULT_ROI_Y_TOP)
        self.roi_y_bottom = getattr(cfg, 'CENTER_LINE_ROI_Y_BOTTOM', DEFAULT_ROI_Y_BOTTOM)

        # --- Detection ---
        self.color_low = np.asarray(getattr(cfg, 'CENTER_LINE_COLOR_LOW', DEFAULT_COLOR_LOW))
        self.color_high = np.asarray(getattr(cfg, 'CENTER_LINE_COLOR_HIGH', DEFAULT_COLOR_HIGH))

        # --- Morphology ---
        k = getattr(cfg, 'CENTER_LINE_MORPH_KERNEL', DEFAULT_MORPH_KERNEL)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # --- Contour filtering ---
        self.min_area_fraction = getattr(cfg, 'CENTER_LINE_MIN_AREA_FRACTION', DEFAULT_MIN_AREA_FRACTION)

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

        # --- Lane-select seam (see LaneMode) ---
        self.lane_mode = LaneMode(lane_mode)

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

    def _clean_mask(self, mask):
        # OPEN (erode-then-dilate) removes small noise specks; CLOSE
        # (dilate-then-erode) fills small holes, e.g. a glare streak.
        # Order matters: OPEN first, so a stray noise speck can't fuse to
        # the real blob before it gets a chance to be removed.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return mask

    def _find_line(self, mask):
        """
        Returns (cx, cy, confidence) in ROI-local coordinates (cx is also
        full-frame-x, since the ROI spans the full image width), or None if
        nothing passed the area filter. confidence is a genuine fraction of
        the ROI area matched, not a raw pixel count.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = mask.shape[0] * mask.shape[1]
        min_area_px = self.min_area_fraction * roi_area

        good = [c for c in contours if cv2.contourArea(c) >= min_area_px]
        if not good:
            return None

        # Largest surviving blob wins -- simpler and more predictable than
        # picking by proximity to a prior frame; gap tolerance (below) is
        # what handles dash-to-dash continuity, not centroid selection.
        best = max(good, key=cv2.contourArea)
        M = cv2.moments(best)
        if M['m00'] == 0:
            return None

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        confidence = cv2.contourArea(best) / roi_area
        return cx, cy, confidence

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _get_target_x(self, width):
        if self.target_pixel is not None:
            return float(self.target_pixel)
        # Lane-select seam: LEFT/RIGHT branches exist for the future
        # lane-following stage to fill in with real white-boundary-relative
        # math. For center-line following, all three modes currently
        # resolve to the same frame-center target.
        if self.lane_mode == LaneMode.LEFT:
            return width / 2.0
        elif self.lane_mode == LaneMode.RIGHT:
            return width / 2.0
        else:
            return width / 2.0

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
            self.state = self.STOPPED
            self.throttle = self.throttle_lost_min

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

        mask = self._make_mask(roi)
        mask = self._clean_mask(mask)
        found = self._find_line(mask)

        cx = cy = None
        if found is not None:
            cx, cy, _confidence = found
            self.state = self.TRACKING
            self.last_seen_time = time.time()
            self.steering = self._steer_towards(cx, width)
            self.throttle = self.throttle_constant
        else:
            # Reset the derivative baseline while not tracking, so the D
            # term doesn't spike from comparing against a stale error once
            # the line is reacquired.
            self.prev_smoothed_error = self.smoothed_error
            self._apply_gap_tolerance()

        cv_img = cam_img
        if self.overlay_image:
            cv_img = self._draw_overlay(cam_img, mask, cx, cy)

        return self.steering, self.throttle, cv_img

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, cam_img, mask, cx, cy):
        """Composite the mask onto the ROI and burn in state/steering/throttle."""
        img = np.copy(cam_img)

        mask_rgb = np.zeros_like(img[self.roi_y_top:self.roi_y_bottom, :, :])
        mask_rgb[:, :, 1] = mask  # green tint
        img[self.roi_y_top:self.roi_y_bottom, :, :] = cv2.addWeighted(
            img[self.roi_y_top:self.roi_y_bottom, :, :], 0.6, mask_rgb, 0.4, 0)
        cv2.rectangle(img, (0, self.roi_y_top), (img.shape[1] - 1, self.roi_y_bottom - 1), (255, 0, 0), 1)

        if cx is not None:
            cv2.drawMarker(img, (int(cx), int(self.roi_y_top + cy)), (0, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        display_str = [
            f"STATE:{self.state}",
            f"STEERING:{self.steering:.2f}",
            f"THROTTLE:{self.throttle:.2f}",
        ]
        y = 10
        for s in display_str:
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            y += 10

        return img
