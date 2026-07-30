"""
The two driving strategies, behind one interface.

    strategy.process(frame_bgr) -> Result
    strategy.reset()

Both share the ROI crop, the yellow masking, the blob filtering, the
plausibility gate and the proportional controller (vision.py and control.py).
They differ only in how they turn yellow blobs into ONE target x:

    LineFollowStrategy  the area-weighted centroid of all the tape it can see
    LaneFollowStrategy  the center of the selected lane, from the boundary pair
                        that brackets it

Everything after "here is the target x" is identical, which is why switching
modes cannot change how the car steers for a given target.
"""

import logging
from typing import Dict, List, NamedTuple, Optional

from donkeycar.parts.lane_following import vision
from donkeycar.parts.lane_following.control import (
    LostLineState,
    PlausibilityGate,
    SteeringController,
    offset_from_x,
)
from donkeycar.parts.lane_following.state import Lane
from donkeycar.parts.lane_following.vision import (
    Detection,
    RoiGeometry,
    YellowDetector,
)

logger = logging.getLogger(__name__)


class Result(NamedTuple):
    """One frame's output. Identical for both strategies."""
    steering: float
    throttle: float
    state: LostLineState
    offset: float                    # -1.0 (target far left) .. +1.0 (far right)
    detection: Detection
    line_x: Optional[float]          # the trusted target position, if any
    reject_reason: Optional[str]     # why a candidate was not accepted
    # Strategy-specific debug detail. Defaults to None rather than {} because a
    # mutable default on a NamedTuple is shared by every instance.
    info: Optional[Dict] = None


class FollowStrategy:
    """
    Base class holding everything the two strategies have in common.

    Subclasses implement find_target(), which is the only real difference
    between them.
    """

    name = "follow"

    def __init__(self, params):
        self.params = params
        self.roi = RoiGeometry(params)
        self.detector = YellowDetector(params)
        self.controller = SteeringController(params)
        self.gate = None  # built on the first frame, since it needs the width

    def reset(self):
        """
        Forget everything learned from previous frames.

        Called when the mode or lane changes, so a switch can never act on state
        that belonged to the other strategy.
        """
        if self.gate is not None:
            self.gate.reset()
        self.controller.reset()

    # -- the part each strategy implements ---------------------------------

    def find_target(self, scan, lane):
        """
        Reduce a YellowScan to a single steering target.

        :return: (target_x or None, target_y or None, info dict)
        """
        raise NotImplementedError

    # -- the shared pipeline ------------------------------------------------

    def roi_bounds(self, height):
        """Exposed for the debug overlay."""
        return self.roi.bounds(height)

    def process(self, frame_bgr, lane=Lane.CENTER) -> Result:
        """
        One frame in, steering and throttle out.

        :param frame_bgr: HxWx3 uint8 image in BGR order
        :param lane: which lane to aim for; ignored by line following
        """
        height, width = frame_bgr.shape[:2]

        if self.gate is None:
            self.gate = PlausibilityGate(self.params, width)

        scan = vision.scan_yellow(frame_bgr, self.roi, self.detector)

        # Not enough yellow anywhere to be a line. Checked before find_target so
        # both strategies get the same "is there anything at all" test.
        if scan.area_frac < self.params.MIN_TOTAL_AREA_FRAC:
            target_x, target_y, info = None, None, {}
        else:
            target_x, target_y, info = self.find_target(scan, lane)

        detection = Detection(
            x=target_x,
            y=target_y,
            area_frac=scan.area_frac,
            blob_count=len(scan.contours),
            raw_mask=scan.raw_mask,
            clean_mask=scan.clean_mask,
            roi_offset_y=scan.roi_top,
        )

        # FOLIAGE DEFENSE #4, then #5. Both strategies pass through both.
        line_x, reject_reason = self.gate.update(target_x)
        offset = offset_from_x(line_x, width) if line_x is not None else None
        steering, throttle, state = self.controller.update(
            offset, self.gate.lost_count)

        return Result(
            steering=steering,
            throttle=throttle,
            state=state,
            offset=offset if offset is not None else 0.0,
            detection=detection,
            line_x=self.gate.line_x,
            reject_reason=reject_reason,
            info=info,
        )


class LineFollowStrategy(FollowStrategy):
    """
    STAGE 1 -- follow a single strip of yellow tape.

    The target is the area-weighted centroid of every blob that survived the
    filters. Lane selection is ignored: there is only one line to follow.
    """

    name = "line"

    def find_target(self, scan, lane):
        centroid_x, centroid_y = vision.area_weighted_centroid(scan.contours)
        if centroid_x is None:
            return None, None, {}
        # Centroid y is measured inside the ROI; shift it back to full-frame
        # coordinates so overlays land in the right place. x needs no shift
        # because the ROI spans the full width.
        return centroid_x, centroid_y + scan.roi_top, {}


class LaneModel:
    """
    Tracks the three lines that define the two lanes, frame to frame.

        left boundary  |  divider  |  right boundary
              lane LEFT     ^     lane RIGHT

    Two jobs:

    1. IDENTITY. Which blob is which line? Solved by matching each blob to
       whichever line it was nearest to last frame (within
       LANE_ASSOC_MAX_DIST_FRAC). Per-line matching means the divider dropping
       out -- which it does constantly, being discontinuous -- does not disturb
       the boundaries.

    2. LANE WIDTH. Whenever two lines bracketing a lane are both visible, the
       half-lane width is measured and smoothed into an average. That measured
       value is what makes the single-boundary fallback trustworthy: it uses a
       width seen moments ago rather than a number typed into a config file.
    """

    KEYS = ("left", "divider", "right")

    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.tracks = {key: None for key in self.KEYS}
        # Seeded from config, then continuously re-measured from the track.
        self.half_width = None
        self.width_age = 10 ** 6  # frames since last real measurement

    def _seed_half_width(self, frame_width):
        if self.half_width is None:
            self.half_width = self.params.HALF_LANE_WIDTH_FRAC * frame_width

    def half_width_is_fresh(self):
        return self.width_age <= self.params.LANE_WIDTH_STALE_FRAMES

    def _observe_half_width(self, measured):
        """Smooth a new measurement into the running average."""
        if measured <= 0:
            return
        if self.half_width is None:
            self.half_width = measured
        else:
            # Same smoothing constant as the steering, for one less knob.
            alpha = self.params.STEERING_SMOOTHING
            self.half_width = alpha * self.half_width + (1.0 - alpha) * measured
        self.width_age = 0

    def classify(self, lines, frame_width):
        """
        Sort observed lines into left / divider / right.

        :param lines: vision.LineObservation list -- already clustered, so a
                      dashed divider is ONE entry rather than one per dash
        :return: dict of line name -> x position (only for lines seen now)
        """
        self._seed_half_width(frame_width)
        max_dist = self.params.LANE_ASSOC_MAX_DIST_FRAC * frame_width

        found = {}

        # 1. Match to last frame's lines, closest pairs first. This is what keeps
        #    identities stable through curves and dropped dashes.
        candidates = [
            (abs(line.x - self.tracks[key]), key, index)
            for key in self.KEYS if self.tracks[key] is not None
            for index, line in enumerate(lines)
        ]
        candidates.sort()
        used_indices = set()
        for distance, key, index in candidates:
            if distance > max_dist or key in found or index in used_indices:
                continue
            found[key] = lines[index].x
            used_indices.add(index)

        leftovers = [line for index, line in enumerate(lines)
                     if index not in used_indices]

        # 2. Anything left has no history, so fall back on geometry.
        if leftovers:
            self._assign_by_geometry(found, leftovers)

        return found

    def _assign_by_geometry(self, found, leftovers):
        """
        Place unmatched lines using how continuous they are, plus left-to-right
        order.

        The divider is discontinuous, so its fill ratio is low; the outer
        boundaries are continuous, so theirs is near 1.0. That distinction does
        not depend on where the car is or which way the track curves, which is
        why it is the primary cue when there is no history to match against.
        """
        max_fill = self.params.DIVIDER_MAX_FILL_RATIO
        leftovers = sorted(leftovers, key=lambda line: line.x)

        dashed = [line for line in leftovers if line.fill_ratio <= max_fill]
        solid = [line for line in leftovers if line.fill_ratio > max_fill]

        # Exactly one dashed line and the divider slot free: that is the divider.
        if "divider" not in found and len(dashed) == 1:
            found["divider"] = dashed[0].x

        # Two solid lines with both boundary slots free are the outer pair.
        available = [key for key in self.KEYS if key not in found]
        if len(solid) == 2 and "left" in available and "right" in available:
            found["left"], found["right"] = solid[0].x, solid[1].x
            return

        # Otherwise place each remaining line in the slot it fits without
        # breaking left-to-right order against what is already identified.
        available = [key for key in self.KEYS if key not in found]
        placed = {found[key] for key in found}
        for line in leftovers:
            if line.x in placed:
                continue
            key = self._slot_for(line, found, available)
            if key is not None:
                found[key] = line.x
                available.remove(key)

    def _slot_for(self, line, found, available):
        """
        Which free slot can this line occupy without breaking left-to-right order
        against the lines already identified?
        """
        for key in available:
            index = self.KEYS.index(key)
            left_of = [found[k] for k in self.KEYS[:index] if k in found]
            right_of = [found[k] for k in self.KEYS[index + 1:] if k in found]
            if left_of and line.x <= max(left_of):
                continue
            if right_of and line.x >= min(right_of):
                continue
            return key
        return None

    def update_tracks(self, found):
        """Remember this frame's positions, and age the width measurement."""
        for key in self.KEYS:
            if key in found:
                self.tracks[key] = found[key]
        self.width_age += 1

    def lane_center(self, found, lane, frame_width):
        """
        Work out the lane center, and say which method produced it.

        A three-tier ladder, most accurate first:

        TIER 1 -- both lines bracketing the chosen lane are visible.
            Take their midpoint. Exact, and this is where LEFT vs RIGHT lane
            selection happens: it is literally which pair gets used. If both
            outer boundaries are visible but the divider dash is missing, the
            divider is inferred as their midpoint, which is still tier 1.

        TIER 2 -- only one usable line is visible (a sharp curve, or the ROI
            clipping one side). Place the target one measured half-lane-width to
            the correct side of it. Trustworthy because the width was measured
            from a real boundary pair moments ago, not typed into a config file.
            The assumption is that pixel lane width is locally constant -- true
            for a fixed camera on flat ground, and degrading on tight curves
            where perspective compresses the lane. That degradation is bounded
            by tier 3.

        TIER 3 -- nothing identifiable, or the width measurement has gone stale.
            Return None, which sends the car into hold-heading. Losing the lane
            beats inventing a target.

        :return: (target_x or None, method name)
        """
        # --- tier 1: a bracketing pair ---
        left, divider, right = (found.get("left"), found.get("divider"),
                                found.get("right"))

        # Both outer boundaries but no divider dash: the divider is between them.
        if divider is None and left is not None and right is not None:
            divider = 0.5 * (left + right)
            self._observe_half_width(abs(right - left) / 4.0)

        if lane is Lane.RIGHT:
            near, far = divider, right
        else:
            # LEFT is also the sensible default for CENTER, which lane
            # following never receives but which costs nothing to handle.
            near, far = left, divider

        if near is not None and far is not None:
            self._observe_half_width(abs(far - near) / 2.0)
            return 0.5 * (near + far), "pair"

        # --- tier 2: a single line plus a measured lane width ---
        mode = self.params.SINGLE_BOUNDARY_MODE
        if mode == "hold":
            return None, "hold (single-boundary mode)"

        if not self.half_width_is_fresh():
            return None, f"width stale ({self.width_age} frames)"

        half = self.half_width

        if mode == "divider":
            # Steer relative to the divider only. Robust while the divider is
            # visible, but it drops out often -- hence not the default.
            if divider is None:
                return None, "no divider (divider mode)"
            sign = 1.0 if lane is Lane.RIGHT else -1.0
            return divider + sign * half, "divider only"

        # mode == "offset": use whichever single line we have, on the correct
        # side. The sign is what makes lane selection still work here.
        if divider is not None:
            sign = 1.0 if lane is Lane.RIGHT else -1.0
            return divider + sign * half, "divider offset"
        if lane is Lane.RIGHT and right is not None:
            return right - half, "right boundary offset"
        if lane is not Lane.RIGHT and left is not None:
            return left + half, "left boundary offset"

        # A boundary for the other lane is not usable for this one: the divider
        # position is unknown, so the offset would be a guess on top of a guess.
        return None, "no usable line"


class LaneFollowStrategy(FollowStrategy):
    """
    STAGE 2 -- drive a chosen lane between two yellow boundaries and a
    discontinuous yellow center divider.

    Reuses the entire stage 1 pipeline; the only new work is deciding which blob
    is which line and where the middle of the chosen lane therefore is. See
    LaneModel for both.
    """

    name = "lane"

    def __init__(self, params):
        super().__init__(params)
        self.model = LaneModel(params)

    def reset(self):
        super().reset()
        self.model.reset()

    def find_target(self, scan, lane):
        blobs = vision.describe_blobs(scan)
        if not blobs:
            return None, None, {}

        # Group the blobs into lines FIRST. The divider is dashed, so without
        # this a single divider would present as several competing "lines".
        tolerance = self.params.LINE_CLUSTER_TOLERANCE_FRAC * scan.width
        lines = vision.cluster_lines(blobs, tolerance, scan.roi_height)

        found = self.model.classify(lines, scan.width)
        self.model.update_tracks(found)

        target_x, method = self.model.lane_center(found, lane, scan.width)

        info = {
            "method": method,
            "lines": {key: round(value, 1) for key, value in found.items()},
            "half_width": (None if self.model.half_width is None
                           else round(self.model.half_width, 1)),
            "width_age": self.model.width_age,
            "line_count": len(lines),
        }

        if target_x is None:
            return None, None, info

        # Aim at the near end of the ROI, where the geometry is least distorted
        # by perspective and the steering response is most direct.
        target_y = scan.roi_top + scan.roi_height * 0.75
        return target_x, target_y, info


def build_strategy(mode, params) -> FollowStrategy:
    """Make the strategy for a Mode. Imported here to avoid a circular import."""
    from donkeycar.parts.lane_following.state import Mode

    if mode is Mode.LANE:
        return LaneFollowStrategy(params)
    return LineFollowStrategy(params)
