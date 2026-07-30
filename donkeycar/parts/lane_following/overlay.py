"""
Debug overlay -- what the pipeline sees, drawn on the camera feed.

This is the main tuning instrument. With it on you can see whether foliage is
leaking into the mask, whether the ROI is cropped tightly enough, and where the
steering target actually is.

Layout, left to right along the bottom:
  * green tint   pixels the RAW color threshold accepted (before cleanup)
  * white        pixels that survived cleanup AND the shape filters -- these are
                 the only pixels the steering is based on
  * yellow box   the ROI trapezoid
  * red line     frame center (steering neutral)
  * cyan circle  the detected centroid / target point
  * text         mode, state, steering, throttle and detection stats

Comparing the green tint against the white is the fastest way to tell whether a
filter is doing its job: lots of green with no white means the color threshold is
too loose and the shape filters are cleaning up after it.

IMPORTANT: every function here draws onto a COPY. The array handed in is the one
the web server may be encoding to JPEG on another thread, and mutating it in
place would tear the image mid-frame.
"""

import cv2
import numpy as np

# Drawing constants (BGR).
RAW_MASK_TINT = (0, 255, 0)      # green: raw color threshold
ACCEPTED_COLOR = (255, 255, 255)  # white: what steering actually uses
ROI_COLOR = (0, 200, 255)        # yellow-orange: the ROI trapezoid
CENTER_COLOR = (0, 0, 255)       # red: frame center
TARGET_COLOR = (255, 255, 0)     # cyan: the steering target
BOUNDARY_COLOR = (0, 165, 255)   # orange: an identified lane boundary
DIVIDER_COLOR = (255, 0, 255)    # magenta: the identified center divider
TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)

RAW_TINT_WEIGHT = 0.45           # how strongly to tint raw-mask pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.34
LINE_HEIGHT = 11
PANEL_ALPHA = 0.55               # opacity of the dark strip behind the text


def _put_text(image, lines, x=4, y=11):
    """
    Draw the readout over a dark translucent panel.

    The panel matters more than it sounds: this text sits over whatever the
    camera happens to see, and white-on-pale-foliage at this size is genuinely
    unreadable in sunlight. Since this overlay is the instrument used to tune the
    car at the track, it has to stay legible over any background.
    """
    if not lines:
        return

    # A full-width strip, rather than one measured to fit the text. The readout
    # lines change length constantly (the reject reason and the lane info come
    # and go), so a fitted panel would jitter in width frame to frame, which is
    # distracting to watch. The frame is only a few hundred pixels wide, so the
    # extra darkened area costs nothing.
    panel_bottom = min(image.shape[0], y + LINE_HEIGHT * len(lines))
    panel = image[0:panel_bottom, :]
    panel[:] = (panel * (1.0 - PANEL_ALPHA)).astype(np.uint8)

    for line in lines:
        cv2.putText(image, line, (x, y), FONT, FONT_SCALE, TEXT_SHADOW, 2,
                    cv2.LINE_AA)
        cv2.putText(image, line, (x, y), FONT, FONT_SCALE, TEXT_COLOR, 1,
                    cv2.LINE_AA)
        y += LINE_HEIGHT


def _blend_mask(image, mask, roi_top, color, weight):
    """Tint the image where `mask` is set. `mask` covers the ROI rows only."""
    if mask is None:
        return
    roi = image[roi_top:roi_top + mask.shape[0], :]
    if roi.shape[:2] != mask.shape[:2]:
        return
    tint = np.zeros_like(roi)
    tint[:] = color
    where = mask > 0
    if not where.any():
        return
    roi[where] = (roi[where] * (1.0 - weight) + tint[where] * weight).astype(np.uint8)


def draw_roi(image, strategy):
    """Outline the ROI trapezoid, so you can see what is being ignored."""
    height, width = image.shape[:2]
    top, bottom = strategy.roi_bounds(height)
    half = 0.5 * strategy.params.ROI_TOP_WIDTH_FRAC * width
    center = width / 2.0
    corners = np.array([
        (int(center - half), top),
        (int(center + half), top),
        (width - 1, bottom - 1),
        (0, bottom - 1),
    ], dtype=np.int32)
    cv2.polylines(image, [corners], True, ROI_COLOR, 1)


def lane_info_lines(info):
    """
    Text lines describing the lane-following decision, for the overlay.

    Empty for line following, which puts nothing in `info`. The "method" line is
    the important one on the track: it says which tier of the single-boundary
    ladder produced the target, so you can see when the car is running on a
    measured lane width rather than a real boundary pair.
    """
    if not info:
        return []
    lines = [f"lane: {info.get('method', '?')}"]
    found = info.get("lines") or {}
    if found:
        lines.append("  " + " ".join(f"{key[0].upper()}={value:.0f}"
                                    for key, value in sorted(found.items())))
    half_width = info.get("half_width")
    if half_width is not None:
        lines.append(f"  half_w={half_width:.0f} age={info.get('width_age', '?')}")
    return lines


def draw_lane_lines(image, info, strategy):
    """Mark the identified boundary and divider positions as vertical ticks."""
    if not info:
        return
    found = info.get("lines") or {}
    height = image.shape[0]
    top, bottom = strategy.roi_bounds(height)
    for key, x in found.items():
        x = int(x)
        color = DIVIDER_COLOR if key == "divider" else BOUNDARY_COLOR
        cv2.line(image, (x, top), (x, bottom - 1), color, 1)
        cv2.putText(image, key[0].upper(), (x - 3, top - 2), FONT, 0.3,
                    color, 1, cv2.LINE_AA)


def draw(frame_bgr, result, strategy, mode_label="LINE", lane_label="-",
         extra_lines=()):
    """
    Build the debug view for one frame.

    :param frame_bgr: the camera frame, in BGR order. NOT modified.
    :param result: the strategy Result for this frame
    :param strategy: the strategy, for its ROI geometry and params
    :param mode_label: "LINE" or "LANE"
    :param lane_label: which lane is selected, or "-" in line mode
    :param extra_lines: extra text lines (lane following adds its own)
    :return: a new annotated BGR image
    """
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    detection = result.detection

    # Raw threshold first, then the accepted pixels painted solid on top, so
    # white always wins where the two overlap.
    _blend_mask(image, detection.raw_mask, detection.roi_offset_y,
                RAW_MASK_TINT, RAW_TINT_WEIGHT)
    if detection.clean_mask is not None:
        _blend_mask(image, detection.clean_mask, detection.roi_offset_y,
                    ACCEPTED_COLOR, 1.0)

    draw_roi(image, strategy)
    draw_lane_lines(image, result.info, strategy)

    # Frame center: steering is zero when the target sits on this line.
    center_x = width // 2
    cv2.line(image, (center_x, height - 18), (center_x, height - 1), CENTER_COLOR, 1)

    # The steering target.
    if detection.x is not None:
        target = (int(detection.x), int(detection.y))
        cv2.circle(image, target, 5, TARGET_COLOR, -1)
        cv2.circle(image, target, 5, TEXT_SHADOW, 1)
        # A line from center to target makes the sign of the error obvious.
        cv2.line(image, (center_x, target[1]), target, TARGET_COLOR, 1)

    lines = [
        f"{mode_label} lane={lane_label} {result.state.value.upper()}",
        f"steer={result.steering:+.2f} throttle={result.throttle:.2f} "
        f"offset={result.offset:+.2f}",
        f"blobs={detection.blob_count} area={detection.area_frac * 100:.2f}%",
    ]
    if result.reject_reason:
        lines.append(f"rejected: {result.reject_reason}")
    lines.extend(extra_lines)
    _put_text(image, lines)

    return image
