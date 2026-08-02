"""
Shared computer vision: everything both strategies use to find yellow tape.

Extracted from the step-2 line-following pipeline so lane following can reuse it
unchanged. Nothing here knows anything about steering or about lines versus
lanes -- it turns a camera frame into "here are the yellow blobs worth looking
at", and the strategies decide what to do with them.

This module carries FOLIAGE DEFENSES 1-3:

  1. COLOR    YellowDetector.raw_mask -- hue ceiling below foliage green, plus a
              saturation floor that rejects pale washed-out colors
  2. GEOMETRY RoiGeometry -- a trapezoid over the lower frame, excluding the
              horizon where foliage lives
  3. SHAPE    YellowDetector.find_blobs -- minimum area plus rotation-invariant
              fill and solidity tests that reject ragged, leaf-like blobs

Defenses 4 (plausibility) and 5 (hold heading) live in control.py.
"""

import logging
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Detection(NamedTuple):
    """The single steering target found in a frame, plus debug material."""
    x: Optional[float]        # target x in full-frame pixels, None if nothing
    y: Optional[float]        # target y in full-frame pixels
    area_frac: float          # accepted blob area as a fraction of the ROI
    blob_count: int
    raw_mask: Optional[np.ndarray]    # after color threshold, before cleanup
    clean_mask: Optional[np.ndarray]  # after cleanup and shape rejection
    roi_offset_y: int         # ROI top row, so overlays can be drawn in place


class YellowScan(NamedTuple):
    """
    Everything the color stage found, before any decision about what it means.

    Both strategies start from this. Line following collapses it to one
    centroid; lane following sorts the blobs into boundaries and a divider.
    """
    contours: List[np.ndarray]        # blobs that passed every filter
    total_area: float                 # their combined area, in pixels
    area_frac: float                  # ...as a fraction of the ROI area
    raw_mask: np.ndarray
    clean_mask: np.ndarray            # only accepted blobs, redrawn
    roi_top: int
    roi_height: int
    width: int


class RoiGeometry:
    """
    FOLIAGE DEFENSE #2 -- where in the frame we are willing to look.

    A trapezoid across the lower part of the frame: full width at the bottom,
    narrower at the top. Foliage sits at and above the horizon, so simply not
    looking up there removes most false positives at no cost. Narrowing the far
    edge trims the left and right margins near the horizon as well.

    All bounds come from FRACTIONS of the frame size, so the tuning survives a
    change of camera resolution.
    """

    def __init__(self, params):
        self.params = params
        self._cached_shape = None
        self._cached_mask = None

    def bounds(self, height):
        """(top_row, bottom_row) of the ROI."""
        top = int(self.params.ROI_TOP_FRAC * height)
        bottom = int(self.params.ROI_BOTTOM_FRAC * height)
        # Always leave at least one row, however odd the fractions are.
        return top, max(top + 1, bottom)

    def corners(self, height, width):
        """The trapezoid's four corners in full-frame coordinates, for overlays."""
        top, bottom = self.bounds(height)
        half = 0.5 * self.params.ROI_TOP_WIDTH_FRAC * width
        center = width / 2.0
        return np.array([
            (int(center - half), top),
            (int(center + half), top),
            (width - 1, bottom - 1),
            (0, bottom - 1),
        ], dtype=np.int32)

    def trapezoid(self, roi_height, width):
        """
        The trapezoid as a mask covering the ROI rows only.

        Cached, since it only changes if the frame size does.
        """
        shape = (roi_height, width)
        if self._cached_shape == shape and self._cached_mask is not None:
            return self._cached_mask

        mask = np.zeros(shape, dtype=np.uint8)
        top_half_width = 0.5 * self.params.ROI_TOP_WIDTH_FRAC * width
        center = width / 2.0
        polygon = np.array([[
            (int(center - top_half_width), 0),
            (int(center + top_half_width), 0),
            (width, roi_height),
            (0, roi_height),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)

        self._cached_shape = shape
        self._cached_mask = mask
        return mask


class YellowDetector:
    """
    FOLIAGE DEFENSES #1 and #3 -- color thresholding and shape filtering.
    """

    def __init__(self, params):
        self.params = params
        self._open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.MORPH_OPEN_KERNEL, params.MORPH_OPEN_KERNEL))
        self._close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.MORPH_CLOSE_KERNEL, params.MORPH_CLOSE_KERNEL))

    def raw_mask(self, roi_bgr):
        """
        White where the pixel looks like yellow tape, black elsewhere.

        cv2's HSV conversion expects BGR input, so the caller must already have
        the frame in BGR order. Feed it RGB and yellow lands nowhere near the
        hue window, so the mask is empty on every frame -- with no error to
        explain why. See CAMERA_COLOR_ORDER.
        """
        # Blur first. Thresholding turns single noisy pixels into blobs, and
        # blurring removes them before that can happen.
        blurred = cv2.GaussianBlur(
            roi_bgr, (self.params.BLUR_KERNEL, self.params.BLUR_KERNEL), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # inRange keeps pixels inside a box in HSV space. The HUE CEILING is
        # what excludes foliage green (hue 40+); the SATURATION FLOOR is what
        # excludes pale, washed-out colors such as distant shrubs.
        return cv2.inRange(hsv, self.params.yellow_low, self.params.yellow_high)

    def clean(self, mask):
        """
        Opening removes speckle (erode then dilate: small dots vanish).
        Closing fills gaps (dilate then erode: nearby pieces join), which is
        what lets dashed or scuffed tape read as one continuous line.
        """
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._close_kernel)

    def find_blobs(self, mask, roi_area):
        """
        Keep only blobs shaped like tape.

        Both shape tests are ROTATION-INVARIANT on purpose. An upright bounding
        box is mostly empty for a diagonal strip, so a plain area/bbox test
        would reject real tape angled across the frame -- exactly what happens
        on a curve, which is the worst possible time to lose the line.

        :return: (accepted_contours, total_area_in_pixels)
        """
        params = self.params
        min_area = params.MIN_CONTOUR_AREA_FRAC * roi_area

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        accepted = []
        total_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue  # speckle, or too far away to steer by

            # minAreaRect is a bounding box allowed to tilt, so these ratios do
            # not change with the tape's angle.
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(contour)
            long_side = max(rect_w, rect_h)
            short_side = max(min(rect_w, rect_h), 1e-6)
            if long_side <= 0:
                continue

            # How fully does the blob fill its tilted box? Tape does; a ragged
            # clump of leaves does not.
            if area / (long_side * short_side) < params.MIN_RECT_FILL:
                continue

            # How smooth is the outline? Solidity compares the blob against its
            # own convex hull, so deep notches between leaves push it down.
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area > 0 and area / hull_area < params.MIN_SOLIDITY:
                continue

            # Hairline slivers are usually a shadow edge or a branch, not tape.
            if long_side / short_side > params.MAX_ASPECT_RATIO:
                continue

            accepted.append(contour)
            total_area += area

        return accepted, total_area


def scan_yellow(frame_bgr, roi: RoiGeometry, detector: YellowDetector) -> YellowScan:
    """
    Run the color and shape stages over one frame.

    Crops to the ROI first and does the expensive work only there. That is both
    the foliage defense and a real saving on the Pi, since the HSV conversion is
    the most costly step in the pipeline.
    """
    height, width = frame_bgr.shape[:2]
    top, bottom = roi.bounds(height)

    roi_bgr = frame_bgr[top:bottom, :]
    roi_height = roi_bgr.shape[0]

    raw_mask = detector.raw_mask(roi_bgr)
    raw_mask = cv2.bitwise_and(raw_mask, roi.trapezoid(roi_height, width))

    clean_mask = detector.clean(raw_mask)

    roi_area = float(roi_height * width)
    contours, total_area = detector.find_blobs(clean_mask, roi_area)

    # Redraw the mask from only the blobs that survived, so the debug view shows
    # exactly what the steering is based on -- not what the color threshold
    # happened to catch.
    accepted_mask = np.zeros_like(clean_mask)
    if contours:
        cv2.drawContours(accepted_mask, contours, -1, 255, cv2.FILLED)

    return YellowScan(
        contours=contours,
        total_area=total_area,
        area_frac=total_area / roi_area if roi_area > 0 else 0.0,
        raw_mask=raw_mask,
        clean_mask=accepted_mask,
        roi_top=top,
        roi_height=roi_height,
        width=width,
    )


def area_weighted_centroid(contours):
    """
    The area-weighted center of several blobs, as (x, y) or (None, None).

    Weighting by area means a large nearby piece of tape counts for more than a
    small distant one, and a dashed line still yields one steady target instead
    of flicking between dashes.

    Note this averages blobs that have ALREADY passed every filter. It is not
    "pick the biggest thing in frame" -- that is the failure mode the whole
    pipeline exists to prevent.
    """
    total_area = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        sum_x += moments["m10"]
        sum_y += moments["m01"]
        total_area += moments["m00"]

    if total_area <= 0:
        return None, None
    return sum_x / total_area, sum_y / total_area


class BlobShape(NamedTuple):
    """One accepted blob, described in the terms lane following needs."""
    x: float          # centroid x, full-frame pixels
    y: float          # centroid y, full-frame pixels
    area: float
    top_y: float      # highest (furthest) row, full-frame pixels
    bottom_y: float   # lowest (nearest-to-car) row, full-frame pixels
    height_px: float


class LineObservation(NamedTuple):
    """
    One painted line, after the pieces of it have been gathered together.

    A dashed line is many blobs but one line, so this is the unit lane following
    reasons about.

    `fill_ratio` is what tells a boundary from the divider: it is how much of the
    line's vertical span is actually painted. A continuous boundary is near 1.0;
    a dashed divider is well below that, because of the gaps.
    """
    x: float           # area-weighted centroid x, full-frame pixels
    y: float           # area-weighted centroid y, full-frame pixels
    area: float
    span_frac: float   # vertical extent as a fraction of ROI height
    fill_ratio: float  # painted fraction of that extent (1.0 = continuous)
    blob_count: int


def describe_blobs(scan: YellowScan) -> List[BlobShape]:
    """Measure each accepted blob's position and vertical extent."""
    shapes = []
    for contour in scan.contours:
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        _, box_y, _, box_h = cv2.boundingRect(contour)
        shapes.append(BlobShape(
            x=moments["m10"] / moments["m00"],
            y=moments["m01"] / moments["m00"] + scan.roi_top,
            area=moments["m00"],
            top_y=box_y + scan.roi_top,
            bottom_y=box_y + box_h + scan.roi_top,
            height_px=float(box_h),
        ))
    return shapes


def cluster_lines(blobs: List[BlobShape], tolerance_px: float,
                  roi_height: int) -> List[LineObservation]:
    """
    Group blobs that are stacked vertically at about the same x into one line.

    This exists because of the discontinuous center divider. Each dash is its own
    contour, so without clustering a single dashed divider would look like three
    or four separate lines and swamp the three slots lane following has to fill.

    Grouping by x is enough: the lines are roughly vertical in the ROI, and
    genuinely different lines are separated by most of a lane width, which is far
    more than the tolerance.

    :param tolerance_px: how far apart in x two blobs can be and still be the
                         same line
    """
    if not blobs:
        return []

    ordered = sorted(blobs, key=lambda blob: blob.x)
    groups = [[ordered[0]]]
    for blob in ordered[1:]:
        # Compare against the running mean of the current group, so a gentle
        # lean does not accumulate into a split.
        group_x = sum(b.x for b in groups[-1]) / len(groups[-1])
        if abs(blob.x - group_x) <= tolerance_px:
            groups[-1].append(blob)
        else:
            groups.append([blob])

    observations = []
    for group in groups:
        total_area = sum(blob.area for blob in group)
        if total_area <= 0:
            continue
        top = min(blob.top_y for blob in group)
        bottom = max(blob.bottom_y for blob in group)
        span = max(bottom - top, 1.0)
        painted = sum(blob.height_px for blob in group)
        observations.append(LineObservation(
            x=sum(blob.x * blob.area for blob in group) / total_area,
            y=sum(blob.y * blob.area for blob in group) / total_area,
            area=total_area,
            span_frac=span / max(roi_height, 1),
            # Capped at 1.0: overlapping blobs could otherwise exceed the span.
            fill_ratio=min(painted / span, 1.0),
            blob_count=len(group),
        ))
    return observations
