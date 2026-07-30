"""
Tests for line/lane following.

These run on synthetic frames, so they can check the logic that matters most --
foliage rejection, the plausibility gate and the lost-line state machine --
without a car, a camera or a track.

Run with:  pytest tests/test_lane_following.py -v
"""

import numpy as np
import pytest

from donkeycar.parts.lane_following import vision
from donkeycar.parts.lane_following.control import LostLineState
from donkeycar.parts.lane_following.params import Params
from donkeycar.parts.lane_following.state import (
    Lane,
    Mode,
    PipelineState,
    get_pipeline_state,
    reset_pipeline_state,
)
from donkeycar.parts.lane_following.strategies import (
    LaneFollowStrategy,
    LaneModel,
    LineFollowStrategy,
)

WIDTH = 426
HEIGHT = 240

# BGR colors chosen to match the real track.
YELLOW_TAPE = (30, 200, 220)    # vivid yellow: high saturation, hue ~25
PALE_FOLIAGE = (150, 200, 160)  # pale green: LOW saturation, hue ~55
ASPHALT = (60, 60, 60)


def blank_frame(color=ASPHALT):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def add_stripe(frame, center_x, color=YELLOW_TAPE, width=26, top_frac=0.55,
               bottom_frac=1.0):
    """Draw a vertical strip, like tape running away from the car."""
    top = int(top_frac * HEIGHT)
    bottom = int(bottom_frac * HEIGHT)
    left = max(0, int(center_x - width // 2))
    right = min(WIDTH, int(center_x + width // 2))
    frame[top:bottom, left:right] = color
    return frame


def add_foliage_band(frame, top_frac=0.0, bottom_frac=0.55):
    """Fill a horizon-level band with pale foliage, as at a real outdoor track."""
    top = int(top_frac * HEIGHT)
    bottom = int(bottom_frac * HEIGHT)
    frame[top:bottom, :] = PALE_FOLIAGE
    # Ragged texture, so it is not a suspiciously perfect rectangle.
    rng = np.random.default_rng(0)
    noise = rng.integers(-25, 25, size=frame[top:bottom, :].shape, dtype=np.int16)
    frame[top:bottom, :] = np.clip(
        frame[top:bottom, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


@pytest.fixture
def params():
    return Params(None)


@pytest.fixture
def pipeline(params):
    return LineFollowStrategy(params)


def settle(pipeline, frame, frames=5):
    """Run enough frames for the tracker to confirm a lock."""
    result = None
    for _ in range(frames):
        result = pipeline.process(frame)
    return result


# ---------------------------------------------------------------------------
# Basic detection and steering
# ---------------------------------------------------------------------------

def test_centered_line_drives_straight(pipeline):
    frame = add_stripe(blank_frame(), WIDTH // 2)
    result = settle(pipeline, frame)

    assert result.state is LostLineState.TRACKING
    assert abs(result.offset) < 0.05, "centered line should give ~zero offset"
    assert abs(result.steering) < 0.1, "centered line should barely steer"
    assert result.throttle > 0


def test_line_left_steers_left(pipeline):
    frame = add_stripe(blank_frame(), int(WIDTH * 0.25))
    result = settle(pipeline, frame)

    assert result.state is LostLineState.TRACKING
    assert result.offset < -0.3, "line left of center should give negative offset"
    assert result.steering < 0, "negative steering is left in donkeycar"


def test_line_right_steers_right(pipeline):
    frame = add_stripe(blank_frame(), int(WIDTH * 0.75))
    result = settle(pipeline, frame)

    assert result.offset > 0.3
    assert result.steering > 0


def test_steering_is_clamped(params):
    params.STEERING_KP = 10.0  # absurdly high on purpose
    params.STEERING_SMOOTHING = 0.0
    pipeline = LineFollowStrategy(params)
    frame = add_stripe(blank_frame(), WIDTH - 20)
    result = settle(pipeline, frame, frames=10)

    assert abs(result.steering) <= params.STEERING_MAX


def test_diagonal_tape_is_not_rejected(pipeline):
    """
    A rotation-invariant shape gate must accept tape angled across the frame.
    An upright-bounding-box test would reject this, losing the line on curves.
    """
    frame = blank_frame()
    top = int(0.6 * HEIGHT)
    for i, y in enumerate(range(top, HEIGHT)):
        x = int(WIDTH * 0.5 + i * 1.2)
        frame[y, max(0, x - 13):min(WIDTH, x + 13)] = YELLOW_TAPE
    result = settle(pipeline, frame)

    assert result.state is LostLineState.TRACKING, "diagonal tape should be found"
    assert result.detection.blob_count >= 1


# ---------------------------------------------------------------------------
# FOLIAGE REJECTION -- the failure this project exists to prevent
# ---------------------------------------------------------------------------

def test_foliage_alone_finds_nothing(pipeline):
    """
    Pale foliage filling the horizon, no tape anywhere. The mask must be empty.
    This is the exact situation in which a previous version locked onto shrubs.
    """
    frame = add_foliage_band(blank_frame())
    result = settle(pipeline, frame)

    assert result.detection.x is None, "foliage must not produce a centroid"
    assert result.detection.area_frac < pipeline.params.MIN_TOTAL_AREA_FRAC
    assert result.state is not LostLineState.TRACKING


def test_foliage_never_becomes_the_steering_target(pipeline):
    """
    Tape visible, then it disappears leaving only foliage. The car must hold its
    heading rather than re-aim at the foliage.
    """
    with_tape = add_stripe(add_foliage_band(blank_frame()), int(WIDTH * 0.3))
    tracking = settle(pipeline, with_tape, frames=6)
    assert tracking.state is LostLineState.TRACKING
    held_steering = tracking.steering

    # Tape gone; shrubs remain.
    foliage_only = add_foliage_band(blank_frame())
    lost = pipeline.process(foliage_only)

    assert lost.state is LostLineState.COASTING
    assert lost.steering == pytest.approx(held_steering), \
        "steering must FREEZE, not re-aim at the foliage"
    assert lost.detection.x is None


def test_foliage_beside_the_line_does_not_pull_the_centroid(pipeline):
    """Tape on the left, foliage on the right: the target must stay on the tape."""
    frame = add_foliage_band(blank_frame(), 0.0, 0.55)
    # Foliage intruding into the ROI on the right, as an overhanging shrub would.
    frame[int(0.55 * HEIGHT):, int(WIDTH * 0.7):] = PALE_FOLIAGE
    add_stripe(frame, int(WIDTH * 0.3))

    result = settle(pipeline, frame)
    assert result.state is LostLineState.TRACKING
    assert result.detection.x < WIDTH * 0.45, \
        "centroid should sit on the tape, not be dragged right by foliage"


def test_loose_hue_ceiling_is_what_lets_foliage_in(params):
    """
    Documents the tuning advice: the hue ceiling is the knob that matters.
    With it raised past foliage green, shrubs DO enter the mask -- which is why
    lowering it is the first thing to try.
    """
    params.YELLOW_HSV_HIGH = (80, 255, 255)  # deliberately far too loose
    params.YELLOW_HSV_LOW = (20, 30, 60)     # and a very low saturation floor
    leaky = LineFollowStrategy(params)
    frame = add_foliage_band(blank_frame(), 0.55, 1.0)  # foliage inside the ROI

    scan = vision.scan_yellow(frame, leaky.roi, leaky.detector)
    assert scan.raw_mask.any(), \
        "with a loose hue ceiling foliage reaches the mask (hence the tuning hint)"


# ---------------------------------------------------------------------------
# FOLIAGE DEFENSE #4 -- the plausibility gate
# ---------------------------------------------------------------------------

def test_implausible_jump_is_rejected(pipeline):
    left = add_stripe(blank_frame(), int(WIDTH * 0.25))
    settle(pipeline, left, frames=6)

    # The "line" teleports to the far right: physically impossible.
    right = add_stripe(blank_frame(), int(WIDTH * 0.9))
    result = pipeline.process(right)

    assert result.reject_reason is not None
    assert "jumped" in result.reject_reason
    assert result.state is LostLineState.COASTING


def test_gate_reopens_so_the_line_can_be_reacquired(pipeline):
    """
    A permanently shut gate would mean one bad moment ends the run. After
    PLAUSIBILITY_RESET_FRAMES the gate must open and allow a new lock.
    """
    left = add_stripe(blank_frame(), int(WIDTH * 0.25))
    settle(pipeline, left, frames=6)

    right = add_stripe(blank_frame(), int(WIDTH * 0.9))
    frames_needed = (pipeline.params.PLAUSIBILITY_RESET_FRAMES
                     + pipeline.params.REACQUIRE_CONFIRM_FRAMES + 2)
    result = None
    for _ in range(frames_needed):
        result = pipeline.process(right)

    assert result.state is LostLineState.TRACKING, "should re-acquire eventually"
    assert result.offset > 0.5, "and now be tracking the right-hand line"


def test_new_lock_requires_confirmation_frames(params):
    """A single frame of something yellow must not become the line."""
    params.REACQUIRE_CONFIRM_FRAMES = 4
    pipeline = LineFollowStrategy(params)
    frame = add_stripe(blank_frame(), WIDTH // 2)

    for i in range(1, params.REACQUIRE_CONFIRM_FRAMES):
        result = pipeline.process(frame)
        assert result.state is not LostLineState.TRACKING, \
            f"frame {i} should not yet be trusted"

    result = pipeline.process(frame)
    assert result.state is LostLineState.TRACKING


# ---------------------------------------------------------------------------
# FOLIAGE DEFENSE #5 -- lost-line hold heading
# ---------------------------------------------------------------------------

def test_lost_line_progresses_coast_slow_stop(params):
    params.COAST_FRAMES = 3
    params.SLOW_FRAMES = 3
    pipeline = LineFollowStrategy(params)

    frame = add_stripe(blank_frame(), int(WIDTH * 0.35))
    tracking = settle(pipeline, frame, frames=6)
    assert tracking.state is LostLineState.TRACKING
    held = tracking.steering

    empty = blank_frame()
    states = [pipeline.process(empty).state for _ in range(9)]

    assert states[0] is LostLineState.COASTING
    assert LostLineState.SLOWING in states
    assert states[-1] is LostLineState.STOPPED

    # Steering held the whole way down.
    final = pipeline.process(empty)
    assert final.steering == pytest.approx(held)
    assert final.throttle == 0.0


def test_throttle_ramps_down_monotonically(params):
    params.COAST_FRAMES = 2
    params.SLOW_FRAMES = 6
    pipeline = LineFollowStrategy(params)

    settle(pipeline, add_stripe(blank_frame(), WIDTH // 2), frames=6)
    empty = blank_frame()
    throttles = [pipeline.process(empty).throttle for _ in range(12)]

    assert all(b <= a + 1e-9 for a, b in zip(throttles, throttles[1:])), \
        f"throttle must never increase while lost: {throttles}"
    assert throttles[-1] == 0.0


def test_reset_clears_state(pipeline):
    settle(pipeline, add_stripe(blank_frame(), int(WIDTH * 0.8)), frames=6)
    assert pipeline.gate.line_x is not None

    pipeline.reset()
    assert pipeline.gate.line_x is None
    assert pipeline.gate.lost_count == 0
    assert pipeline.controller.steering == 0.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_uniform_frames_never_crash(pipeline):
    """Black, white and mid-gray frames must all be handled without raising."""
    for color in [(0, 0, 0), (255, 255, 255), (128, 128, 128)]:
        result = pipeline.process(blank_frame(color))
        assert result.throttle >= 0.0
        assert -1.0 <= result.steering <= 1.0


def test_odd_frame_sizes_work(params):
    """ROI bounds come from fractions, so any frame size should work."""
    pipeline = LineFollowStrategy(params)
    for height, width in [(120, 160), (240, 426), (480, 640)]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = ASPHALT
        frame[int(0.6 * height):, width // 2 - 8:width // 2 + 8] = YELLOW_TAPE
        result = pipeline.process(frame)
        assert -1.0 <= result.steering <= 1.0
        pipeline.reset()


# ---------------------------------------------------------------------------
# Mode / lane state
# ---------------------------------------------------------------------------

def test_state_parsing_and_updates():
    state = PipelineState()
    assert state.snapshot().mode is Mode.LINE

    assert state.set_mode("lane") is True
    assert state.snapshot().mode is Mode.LANE
    assert state.set_lane("RIGHT") is True
    assert state.snapshot().lane is Lane.RIGHT

    # Invalid input is rejected without changing anything.
    assert state.set_mode("banana") is False
    assert state.snapshot().mode is Mode.LANE
    assert state.set_lane(None) is False
    assert state.snapshot().lane is Lane.RIGHT


def test_singleton_is_shared():
    reset_pipeline_state()
    try:
        first = get_pipeline_state()
        second = get_pipeline_state()
        assert first is second, "web server and CV part must share one state"
        first.set_mode("lane")
        assert second.snapshot().mode is Mode.LANE
    finally:
        reset_pipeline_state()


def test_start_mode_comes_from_config():
    reset_pipeline_state()
    try:
        class Cfg:
            START_MODE = "lane"
            START_LANE = "right"
            DEBUG_OVERLAY = True

        state = get_pipeline_state(Params(Cfg()))
        snapshot = state.snapshot()
        assert snapshot.mode is Mode.LANE
        assert snapshot.lane is Lane.RIGHT
        assert snapshot.debug is True
    finally:
        reset_pipeline_state()


# ---------------------------------------------------------------------------
# STAGE 2 -- lane following
# ---------------------------------------------------------------------------

LEFT_BOUNDARY_X = int(WIDTH * 0.15)
DIVIDER_X = int(WIDTH * 0.50)
RIGHT_BOUNDARY_X = int(WIDTH * 0.85)


def lane_frame(left=True, divider=True, right=True, foliage=True,
               left_x=LEFT_BOUNDARY_X, divider_x=DIVIDER_X,
               right_x=RIGHT_BOUNDARY_X):
    """
    A two-lane track: solid yellow boundaries and a DASHED yellow center divider.

    The divider is drawn with real gaps, because that is what makes stage 2 hard
    -- each dash is its own blob and they have to be recognized as one line.
    """
    frame = blank_frame()
    if foliage:
        add_foliage_band(frame)
    if left:
        add_stripe(frame, left_x, width=18)
    if right:
        add_stripe(frame, right_x, width=18)
    if divider:
        top = int(0.55 * HEIGHT)
        for dash_start in range(top, HEIGHT, 26):
            dash_end = min(dash_start + 13, HEIGHT)
            frame[dash_start:dash_end,
                  divider_x - 9:divider_x + 9] = YELLOW_TAPE
    return frame


@pytest.fixture
def lane_pipeline(params):
    return LaneFollowStrategy(params)


def test_dashed_divider_is_one_line_not_many(lane_pipeline):
    """
    The divider is drawn as several dashes. Clustering must present it as ONE
    line, or it would swamp the three slots lane following has to fill.
    """
    frame = lane_frame()
    scan = vision.scan_yellow(frame, lane_pipeline.roi, lane_pipeline.detector)
    blobs = vision.describe_blobs(scan)
    assert len(blobs) > 3, "expected several dash blobs before clustering"

    tolerance = lane_pipeline.params.LINE_CLUSTER_TOLERANCE_FRAC * scan.width
    lines = vision.cluster_lines(blobs, tolerance, scan.roi_height)
    assert len(lines) == 3, f"should cluster into 3 lines, got {len(lines)}"

    # And the dashed one must be identifiable by its fill ratio.
    by_x = sorted(lines, key=lambda line: line.x)
    assert by_x[1].fill_ratio < by_x[0].fill_ratio, \
        "the dashed divider should be less filled than a solid boundary"
    assert by_x[1].fill_ratio <= lane_pipeline.params.DIVIDER_MAX_FILL_RATIO


def test_all_three_lines_are_classified(lane_pipeline):
    frame = lane_frame()
    result = settle(lane_pipeline, frame, frames=6)

    found = result.info["lines"]
    assert set(found) == {"left", "divider", "right"}, f"got {found}"
    assert found["left"] < found["divider"] < found["right"]


def test_left_lane_targets_between_left_and_divider(lane_pipeline):
    frame = lane_frame()
    result = None
    for _ in range(6):
        result = lane_pipeline.process(frame, Lane.LEFT)

    assert result.info["method"] == "pair"
    expected = (LEFT_BOUNDARY_X + DIVIDER_X) / 2
    assert result.detection.x == pytest.approx(expected, abs=12)


def test_right_lane_targets_between_divider_and_right(lane_pipeline):
    frame = lane_frame()
    result = None
    for _ in range(6):
        result = lane_pipeline.process(frame, Lane.RIGHT)

    assert result.info["method"] == "pair"
    expected = (DIVIDER_X + RIGHT_BOUNDARY_X) / 2
    assert result.detection.x == pytest.approx(expected, abs=12)


def test_lane_toggle_shifts_the_target_right(params):
    """Switching LEFT -> RIGHT must move the target toward the right lane."""
    frame = lane_frame()

    left_pipeline = LaneFollowStrategy(params)
    for _ in range(6):
        left_result = left_pipeline.process(frame, Lane.LEFT)

    right_pipeline = LaneFollowStrategy(params)
    for _ in range(6):
        right_result = right_pipeline.process(frame, Lane.RIGHT)

    assert right_result.detection.x > left_result.detection.x
    assert right_result.offset > left_result.offset
    assert left_result.offset < 0 < right_result.offset, \
        "left lane is left of frame center, right lane is right of it"


def test_missing_divider_still_uses_a_pair(lane_pipeline):
    """
    Both boundaries visible but the divider dash absent: the divider is inferred
    as their midpoint, so this stays the accurate tier-1 case.
    """
    frame = lane_frame(divider=False)
    result = None
    for _ in range(6):
        result = lane_pipeline.process(frame, Lane.LEFT)

    assert result.info["method"] == "pair"
    inferred_divider = (LEFT_BOUNDARY_X + RIGHT_BOUNDARY_X) / 2
    expected = (LEFT_BOUNDARY_X + inferred_divider) / 2
    assert result.detection.x == pytest.approx(expected, abs=15)


def test_single_boundary_uses_measured_lane_width(lane_pipeline):
    """
    TIER 2 of the single-boundary ladder. After seeing a full lane, the right
    boundary disappears; the target must be placed a measured half-lane-width
    from what is left, not abandoned.
    """
    full = lane_frame()
    for _ in range(6):
        lane_pipeline.process(full, Lane.RIGHT)
    learned_half_width = lane_pipeline.model.half_width
    assert learned_half_width is not None

    # Only the divider remains -- as when a curve carries the outer boundary
    # out of the ROI.
    clipped = lane_frame(left=False, right=False)
    result = lane_pipeline.process(clipped, Lane.RIGHT)

    assert "offset" in result.info["method"], \
        f"expected an offset method, got {result.info['method']!r}"
    assert result.detection.x is not None
    assert result.detection.x == pytest.approx(
        DIVIDER_X + learned_half_width, abs=15)


def test_measured_width_beats_the_config_seed(lane_pipeline):
    """
    The point of learning the width live: after seeing a real boundary pair, the
    estimate should reflect the track, not HALF_LANE_WIDTH_FRAC.
    """
    seed = lane_pipeline.params.HALF_LANE_WIDTH_FRAC * WIDTH
    frame = lane_frame()
    for _ in range(8):
        lane_pipeline.process(frame, Lane.LEFT)

    true_half_width = (DIVIDER_X - LEFT_BOUNDARY_X) / 2
    assert lane_pipeline.model.half_width == pytest.approx(true_half_width, abs=20)
    assert lane_pipeline.model.half_width != pytest.approx(seed, abs=1e-6)


def test_stale_lane_width_falls_back_to_hold_heading(params):
    """
    TIER 3. Once the width measurement is too old to trust, the car must hold
    heading rather than keep extrapolating from a stale number.
    """
    params.LANE_WIDTH_STALE_FRAMES = 3
    pipeline = LaneFollowStrategy(params)

    full = lane_frame()
    for _ in range(6):
        pipeline.process(full, Lane.RIGHT)

    # Only the divider from here on, for longer than the width stays fresh.
    clipped = lane_frame(left=False, right=False)
    result = None
    for _ in range(params.LANE_WIDTH_STALE_FRAMES + 4):
        result = pipeline.process(clipped, Lane.RIGHT)

    assert "stale" in result.info["method"]
    assert result.detection.x is None
    assert result.state is not LostLineState.TRACKING


def test_single_boundary_mode_hold_gives_up_immediately(params):
    params.SINGLE_BOUNDARY_MODE = "hold"
    pipeline = LaneFollowStrategy(params)

    for _ in range(6):
        pipeline.process(lane_frame(), Lane.RIGHT)
    result = pipeline.process(lane_frame(left=False, right=False), Lane.RIGHT)

    assert result.detection.x is None
    assert "single-boundary mode" in result.info["method"]


def test_lane_following_holds_heading_when_everything_is_lost(lane_pipeline):
    """The stage 1 lost-line behavior must apply unchanged in lane mode."""
    frame = lane_frame()
    tracking = None
    for _ in range(6):
        tracking = lane_pipeline.process(frame, Lane.LEFT)
    assert tracking.state is LostLineState.TRACKING
    held = tracking.steering

    foliage_only = add_foliage_band(blank_frame())
    lost = lane_pipeline.process(foliage_only, Lane.LEFT)

    assert lost.state is LostLineState.COASTING
    assert lost.steering == pytest.approx(held), "steering must freeze in lane mode too"


def test_lane_following_rejects_foliage(lane_pipeline):
    frame = add_foliage_band(blank_frame())
    result = settle(lane_pipeline, frame)
    assert result.detection.x is None
    assert result.state is not LostLineState.TRACKING


def test_lane_reset_clears_the_model(lane_pipeline):
    for _ in range(6):
        lane_pipeline.process(lane_frame(), Lane.LEFT)
    assert lane_pipeline.model.half_width is not None
    assert any(v is not None for v in lane_pipeline.model.tracks.values())

    lane_pipeline.reset()
    assert lane_pipeline.model.half_width is None
    assert all(v is None for v in lane_pipeline.model.tracks.values())
    assert lane_pipeline.controller.steering == 0.0


def test_both_strategies_share_the_controller_behavior(params):
    """
    A given offset must produce the same steering in either mode. That is the
    point of both strategies feeding the same controller.
    """
    line = LineFollowStrategy(params)
    lane = LaneFollowStrategy(params)

    for _ in range(30):
        line_out = line.controller.update(0.4, 0)
        lane_out = lane.controller.update(0.4, 0)
    assert line_out == lane_out


def test_curved_lane_tracks_through_the_curve(lane_pipeline):
    """Lines drifting sideways frame to frame must keep their identities."""
    result = None
    for step in range(10):
        shift = step * 6
        frame = lane_frame(left_x=LEFT_BOUNDARY_X + shift,
                           divider_x=DIVIDER_X + shift,
                           right_x=min(WIDTH - 12, RIGHT_BOUNDARY_X + shift))
        result = lane_pipeline.process(frame, Lane.LEFT)

    assert result.state is LostLineState.TRACKING
    found = result.info["lines"]
    assert "left" in found and "divider" in found
    assert found["left"] < found["divider"], "identities must not swap on a curve"


# ---------------------------------------------------------------------------
# The donkeycar Part: mode switching mid-drive
# ---------------------------------------------------------------------------

class FakeCfg:
    """A minimal stand-in for the car's config object."""
    DEBUG_OVERLAY = False
    CAMERA_COLOR_ORDER = "BGR"


def make_controller():
    from donkeycar.parts.lane_following.controller import LaneFollowingController
    reset_pipeline_state()
    return LaneFollowingController(pid=None, cfg=FakeCfg())


def test_controller_returns_three_outputs():
    controller = make_controller()
    try:
        out = controller.run(add_stripe(blank_frame(), WIDTH // 2))
        assert len(out) == 3
        steering, throttle, image = out
        assert -1.0 <= steering <= 1.0
        assert 0.0 <= throttle <= 1.0
        assert image is not None
    finally:
        reset_pipeline_state()


def test_controller_handles_no_frame():
    """The camera part is threaded, so run() can be called before any frame."""
    controller = make_controller()
    try:
        assert controller.run(None) == (0.0, 0.0, None)
    finally:
        reset_pipeline_state()


def test_switching_mode_mid_drive_resets_and_does_not_swerve():
    controller = make_controller()
    try:
        # Establish a hard-left line lock in LINE mode.
        frame = add_stripe(blank_frame(), int(WIDTH * 0.15))
        for _ in range(8):
            steering, _, _ = controller.run(frame)
        assert steering < -0.2, "should be steering hard left before the switch"

        # Switch to LANE mode. The lane strategy starts with no state at all, so
        # its first frames must not inherit the line strategy's steering.
        controller.state.set_mode("lane")
        lane_frame_image = lane_frame()
        first_steering, first_throttle, _ = controller.run(lane_frame_image)

        assert first_steering == 0.0, \
            "a freshly reset strategy must start from zero steering, not inherit"
        assert first_throttle >= 0.0

        # And it must go on to work rather than crash or sit dead.
        for _ in range(8):
            result = controller.run(lane_frame_image)
        assert result[1] > 0, "lane following should be driving after the switch"
    finally:
        reset_pipeline_state()


def test_switching_lane_mid_drive_moves_the_target():
    controller = make_controller()
    try:
        controller.state.set_mode("lane")
        controller.state.set_lane("left")
        frame = lane_frame()
        for _ in range(8):
            left_steering, _, _ = controller.run(frame)

        controller.state.set_lane("right")
        for _ in range(8):
            right_steering, _, _ = controller.run(frame)

        assert right_steering > left_steering, \
            "selecting the right lane should steer further right"
    finally:
        reset_pipeline_state()


def test_repeated_mode_switching_never_crashes():
    controller = make_controller()
    try:
        frame = lane_frame()
        for index in range(24):
            controller.state.set_mode("lane" if index % 2 else "line")
            controller.state.set_lane("right" if index % 3 else "left")
            steering, throttle, image = controller.run(frame)
            assert -1.0 <= steering <= 1.0
            assert 0.0 <= throttle <= 1.0
    finally:
        reset_pipeline_state()


def test_debug_overlay_does_not_modify_the_input_frame():
    """
    The web server may be JPEG-encoding the frame on another thread, so the
    overlay must never draw in place.
    """
    from donkeycar.parts.lane_following.controller import LaneFollowingController

    class DebugCfg(FakeCfg):
        DEBUG_OVERLAY = True

    reset_pipeline_state()
    try:
        controller = LaneFollowingController(pid=None, cfg=DebugCfg())
        frame = add_stripe(blank_frame(), WIDTH // 2)
        original = frame.copy()
        for _ in range(4):
            _, _, image = controller.run(frame)
        assert np.array_equal(frame, original), "input frame was mutated"
        assert not np.array_equal(image, original), "overlay should have drawn"
    finally:
        reset_pipeline_state()
