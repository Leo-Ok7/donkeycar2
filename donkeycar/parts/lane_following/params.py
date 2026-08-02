"""
Every tunable value for line following and lane following, in one place.

HOW TUNING WORKS
----------------
The constants below are the defaults. To change one at the track, do NOT edit
this file -- add it to your car's myconfig.py using the exact same name:

    YELLOW_HSV_HIGH = (30, 255, 255)   # tighter yellow, rejects more foliage

Params.from_config() reads each name out of the config with getattr(), falling
back to the default here. That means a myconfig.py written for an older version
can never crash the car with a missing key -- it just uses defaults for anything
it does not mention.

Nothing in the vision code contains a bare number. If you find one, it belongs
here.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ==  TUNABLE PARAMETERS                                                    ==
# ============================================================================

# --- Camera -----------------------------------------------------------------
# Frames come off the OAK-D at this size (see the camera patch in
# docs/lane_following.md). Every ROI below is a FRACTION of these, so changing
# the resolution does not invalidate the tuning.
IMAGE_WIDTH = 426
IMAGE_HEIGHT = 240

# "BGR" or "RGB" -- the channel order of incoming frames. cv2 assumes BGR, so
# getting this wrong makes the yellow mask silently empty forever. Determine it
# with scripts/oakd_color_check.py; do not guess.
CAMERA_COLOR_ORDER = "BGR"


# --- Yellow color separation (FOLIAGE DEFENSE #1) ---------------------------
# Yellow tape sits around hue 20-35. Pale foliage green starts around hue 40.
# The hue CEILING is what keeps them apart, and the saturation FLOOR is what
# rejects washed-out foliage -- shrubs are pale (low saturation) where tape is
# vividly saturated.
#
# These two are the first things to touch if green leaks into the mask:
#   too much foliage -> LOWER the hue ceiling, then RAISE the saturation floor.
# OpenCV ranges: hue 0-179, saturation 0-255, value 0-255.
YELLOW_HSV_LOW = (20, 110, 90)    # (hue_min, sat_min, val_min)
YELLOW_HSV_HIGH = (33, 255, 255)  # (hue_max, sat_max, val_max)


# --- Region of interest (FOLIAGE DEFENSE #2) --------------------------------
# A trapezoid over the lower part of the frame. Foliage lives at and above the
# horizon, so cropping it away removes most false positives for free.
# All values are fractions of the frame: 0.0 = top/left, 1.0 = bottom/right.
ROI_TOP_FRAC = 0.55        # raise toward 1.0 to crop more sky/horizon away
ROI_BOTTOM_FRAC = 1.0      # 1.0 = all the way to the bottom edge
ROI_TOP_WIDTH_FRAC = 0.70  # width of the trapezoid's FAR edge (1.0 = rectangle)


# --- Blob size and shape filtering (FOLIAGE DEFENSE #3) ---------------------
# Areas are fractions of the ROI area, so they keep working if the ROI changes.
MIN_CONTOUR_AREA_FRAC = 0.0015  # ignore blobs smaller than this (speckle, leaves)
MIN_TOTAL_AREA_FRAC = 0.0030    # below this much yellow in total, the line is LOST

# Shape gates. Tape is a smooth solid strip; foliage is ragged and full of
# concavities. Both measures below are ROTATION-INVARIANT, which matters: an
# upright bounding box is mostly empty for a diagonal strip, so a plain
# area/bbox "extent" test would wrongly reject real tape on curves.
#
#   rect fill = contour area / minAreaRect area  (how well it fills a tilted box)
#   solidity  = contour area / convex hull area  (how ragged the outline is)
MIN_RECT_FILL = 0.55
MIN_SOLIDITY = 0.80
MAX_ASPECT_RATIO = 20.0  # reject hairline slivers (branch/shadow edges)

# Morphology, in pixels. Open removes speckle; close bridges gaps in the tape.
BLUR_KERNEL = 5
MORPH_OPEN_KERNEL = 3
MORPH_CLOSE_KERNEL = 5


# --- Plausibility gate (FOLIAGE DEFENSE #4) ---------------------------------
# The line cannot teleport. Reject any detection that jumps further than this
# (as a fraction of frame width) from where the line was last frame. A foliage
# lock-on is exactly such a jump.
MAX_CENTROID_JUMP_FRAC = 0.25

# The gate has to be able to let go, or the car could never re-acquire the line
# after a genuine loss. After this many consecutive rejected/lost frames the
# gate opens to the whole frame so a fresh lock can form.
PLAUSIBILITY_RESET_FRAMES = 8

# ...and a fresh lock must hold still for this many frames before it is trusted.
# This is what stops a one-frame flash of green from becoming the new "line".
REACQUIRE_CONFIRM_FRAMES = 3


# --- Lost-line behavior (FOLIAGE DEFENSE #5, the final backstop) ------------
# When the line is lost the car HOLDS ITS LAST STEERING and coasts. It never
# steers toward whatever else is in frame.
COAST_FRAMES = 12           # frames to hold heading at reduced throttle
SLOW_FRAMES = 12            # frames after that, ramping throttle down to zero
COAST_THROTTLE_SCALE = 0.6  # throttle multiplier while coasting


# --- Steering and throttle --------------------------------------------------
# steering = STEERING_KP * offset, where offset is -1.0 (line hard left) to
# +1.0 (line hard right). Raise KP for sharper response; too high oscillates.
STEERING_KP = 0.85
STEERING_MAX = 1.0        # clamp; donkeycar steering is -1.0 to 1.0
STEERING_SMOOTHING = 0.35  # 0.0 = no smoothing, 0.9 = very smooth but laggy

THROTTLE_FORWARD = 0.18   # constant cruise throttle. START LOW.
THROTTLE_TURN_SCALE = 0.7  # throttle multiplier at full steering lock


# --- Lane following (stage 2 only) ------------------------------------------
# The ROI is sampled in horizontal bands from near to far. More bands = more
# robust to dashed lines, but more CPU.
LANE_ROW_BANDS = 4

# Max distance (fraction of width) for matching a blob to the line it was last
# frame. Too small and tracks drop on curves; too large and lines swap identity.
LANE_ASSOC_MAX_DIST_FRAC = 0.20

# Half the width of one lane, as a fraction of frame width. Used only when a
# single line is visible. Measure this from a debug frame on a straight section.
# It is also learned live whenever two lines are visible, so this is the seed.
HALF_LANE_WIDTH_FRAC = 0.18

# If the live lane-width measurement is older than this many frames, stop
# trusting it and hold heading instead.
LANE_WIDTH_STALE_FRAMES = 20

# What to do when only one boundary is visible:
#   "offset"  -> place the target one measured half-lane-width from that line
#   "divider" -> steer relative to the divider only
#   "hold"    -> hold heading immediately
SINGLE_BOUNDARY_MODE = "offset"

# How the divider is told apart from a boundary. The divider is DISCONTINUOUS,
# so once its dashes are grouped into one line, only part of that line's vertical
# span is actually painted. A continuous boundary is close to 1.0.
# Lower this if dashes are being mistaken for boundaries; raise it if a scuffed
# or partly occluded boundary is being mistaken for the divider.
DIVIDER_MAX_FILL_RATIO = 0.75

# How far apart in x two blobs can be and still count as the same line. Must be
# well below one lane width, and above the sideways wander of a single line
# across the ROI.
LINE_CLUSTER_TOLERANCE_FRAC = 0.06


# --- Debug overlay and web page ---------------------------------------------
DEBUG_OVERLAY = False   # draw masks + centroid on the output image
LANE_WEB_ENABLE = False  # start the mode/lane toggle web server
LANE_WEB_PORT = 8891     # the donkeycar web controller already uses 8887

# Which mode/lane the car starts in. "line" or "lane"; "left" or "right".
START_MODE = "line"
START_LANE = "left"

# ============================================================================
# ==  END OF TUNABLE PARAMETERS                                             ==
# ============================================================================


def _tunable_names():
    """
    Every UPPER_CASE name defined in the block above.

    Collected automatically so that adding a parameter means adding it in ONE
    place -- the block -- rather than also having to list it here.
    """
    return sorted(
        name for name, value in globals().items()
        if name.isupper() and not name.startswith("_")
        and isinstance(value, (int, float, str, bool, tuple))
    )


class Params:
    """
    A snapshot of the tunables, with any myconfig.py overrides applied.

    Read as attributes with the same names as the constants above, e.g.
    params.STEERING_KP, so the vision code reads like the config file.
    """

    def __init__(self, cfg=None):
        overrides = []
        for name in _tunable_names():
            default = globals()[name]
            value = getattr(cfg, name, default) if cfg is not None else default
            if value != default:
                overrides.append(f"{name}={value!r}")
            setattr(self, name, value)

        self._array_cache = {}
        self._validate()

        if overrides:
            logger.info("lane_following config overrides: " + ", ".join(overrides))
        else:
            logger.info("lane_following using all default parameters")

    def _cached_array(self, value):
        """
        The uint8 array cv2.inRange() wants, built once per distinct value.

        Deliberately keyed on the value rather than computed in __init__: that
        way assigning params.YELLOW_HSV_HIGH = (...) at runtime -- from a tuning
        script or a test -- actually takes effect. Pre-computing it once made
        such an assignment silently do nothing to the mask.
        """
        key = tuple(value)
        array = self._array_cache.get(key)
        if array is None:
            array = np.asarray(key, dtype=np.uint8)
            self._array_cache[key] = array
        return array

    @property
    def yellow_low(self):
        return self._cached_array(self.YELLOW_HSV_LOW)

    @property
    def yellow_high(self):
        return self._cached_array(self.YELLOW_HSV_HIGH)

    def _validate(self):
        """
        Catch the mistakes that would otherwise show up as a car that silently
        does nothing. Raised at startup, not mid-drive.
        """
        problems = []

        if self.CAMERA_COLOR_ORDER not in ("BGR", "RGB"):
            problems.append(
                f'CAMERA_COLOR_ORDER must be "BGR" or "RGB", got '
                f'{self.CAMERA_COLOR_ORDER!r}'
            )

        low, high = self.YELLOW_HSV_LOW, self.YELLOW_HSV_HIGH
        if len(low) != 3 or len(high) != 3:
            problems.append("YELLOW_HSV_LOW/HIGH must each be 3 values (h, s, v)")
        elif any(l >= h for l, h in zip(low, high)):
            problems.append(
                f"YELLOW_HSV_LOW {low} must be below YELLOW_HSV_HIGH {high} "
                "in every channel, or the mask is always empty"
            )
        elif high[0] >= 40:
            # Not fatal -- some tape really is that orange-green -- but it is
            # almost always the cause of foliage leaking into the mask.
            logger.warning(
                f"YELLOW_HSV_HIGH hue is {high[0]}, which overlaps foliage green "
                "(hue 40+). Expect shrubs in the mask."
            )

        if not 0.0 <= self.ROI_TOP_FRAC < self.ROI_BOTTOM_FRAC <= 1.0:
            problems.append(
                f"need 0 <= ROI_TOP_FRAC ({self.ROI_TOP_FRAC}) < "
                f"ROI_BOTTOM_FRAC ({self.ROI_BOTTOM_FRAC}) <= 1"
            )

        if not 0.0 < self.ROI_TOP_WIDTH_FRAC <= 1.0:
            problems.append(
                f"ROI_TOP_WIDTH_FRAC must be in (0, 1], got {self.ROI_TOP_WIDTH_FRAC}"
            )

        if not 0.0 <= self.STEERING_SMOOTHING < 1.0:
            problems.append(
                f"STEERING_SMOOTHING must be in [0, 1), got {self.STEERING_SMOOTHING}"
            )

        if self.SINGLE_BOUNDARY_MODE not in ("offset", "divider", "hold"):
            problems.append(
                f'SINGLE_BOUNDARY_MODE must be "offset", "divider" or "hold", '
                f"got {self.SINGLE_BOUNDARY_MODE!r}"
            )

        if self.MIN_TOTAL_AREA_FRAC < self.MIN_CONTOUR_AREA_FRAC:
            logger.warning(
                "MIN_TOTAL_AREA_FRAC is below MIN_CONTOUR_AREA_FRAC, so the "
                "total-area gate can never reject anything a single blob passed."
            )

        if problems:
            raise ValueError(
                "Invalid lane_following configuration:\n  - "
                + "\n  - ".join(problems)
            )
