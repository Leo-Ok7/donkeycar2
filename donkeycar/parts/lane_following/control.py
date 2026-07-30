"""
Shared control: turning a target position into steering and throttle.

Extracted from the step-2 line-following pipeline so lane following drives
through exactly the same controller. Both strategies produce a single target x
in the frame, and everything from there on is common.

This module carries FOLIAGE DEFENSES 4 and 5:

  4. PLAUSIBILITY  PlausibilityGate -- a target that jumps too far from where the
                   line was last frame is rejected
  5. HOLD HEADING  SteeringController -- losing the line freezes steering and
                   winds throttle down to zero. It never re-aims at anything.

Defenses 1-3 (color, ROI geometry, blob shape) live in vision.py.
"""

import logging
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class LostLineState(Enum):
    """
    What the car does when it cannot see the line.

    Note what is absent: there is no state meaning "steer toward whatever else
    is in frame". Losing the line can only ever lead to holding the current
    heading and slowing down. That is the property that keeps the car out of the
    shrubs.
    """
    TRACKING = "tracking"  # target is visible and trusted
    COASTING = "coasting"  # just lost it: hold heading, keep rolling
    SLOWING = "slowing"    # still lost: hold heading, ramp throttle down
    STOPPED = "stopped"    # gone: hold heading, throttle zero


class PlausibilityGate:
    """
    FOLIAGE DEFENSE #4 -- decides whether a measured target is really the line.

    The rule is that the line cannot teleport. A target more than
    MAX_CENTROID_JUMP_FRAC of the frame width from last frame's position is
    rejected, and a shrub appearing off to one side is exactly that kind of jump.

    The subtle part is letting go. A gate that never opens means one bad moment
    ends the run -- it would go on rejecting the real line forever. So after
    PLAUSIBILITY_RESET_FRAMES frames with nothing accepted, the gate opens to
    the whole frame so a new lock can form. To stop that opening from letting one
    flash of green become the new "line", a fresh lock must repeat in roughly the
    same place for REACQUIRE_CONFIRM_FRAMES frames before it is trusted.
    """

    def __init__(self, params, frame_width):
        self.params = params
        self.max_jump_px = params.MAX_CENTROID_JUMP_FRAC * frame_width
        self.reset()

    def reset(self):
        self.line_x = None          # trusted position, None until acquired
        self.lost_count = 0         # consecutive frames with nothing accepted
        self._candidate_x = None    # a possible new line, not yet trusted
        self._candidate_count = 0

    def update(self, measured_x):
        """
        :param measured_x: candidate target x, or None if nothing was found
        :return: (trusted_x or None, reject_reason or None)
        """
        if measured_x is None:
            self.lost_count += 1
            self._candidate_x = None
            self._candidate_count = 0
            return None, None

        if self.line_x is not None:
            gate_is_open = self.lost_count >= self.params.PLAUSIBILITY_RESET_FRAMES
            jump = abs(measured_x - self.line_x)

            if not gate_is_open:
                if jump > self.max_jump_px:
                    # Implausible. This is the foliage lock-on signature.
                    self.lost_count += 1
                    return None, f"jumped {jump:.0f}px (max {self.max_jump_px:.0f})"
                self.line_x = measured_x
                self.lost_count = 0
                return measured_x, None
            # Gate open after a long loss: fall through and re-confirm below
            # rather than trusting the first thing that appears.

        # Establishing a new track: require the same position several frames
        # running. This is what stops one frame of green becoming the line.
        if (self._candidate_x is None
                or abs(measured_x - self._candidate_x) > self.max_jump_px):
            self._candidate_x = measured_x
            self._candidate_count = 1
        else:
            self._candidate_x = measured_x
            self._candidate_count += 1

        if self._candidate_count >= self.params.REACQUIRE_CONFIRM_FRAMES:
            was_lost = self.line_x is not None
            self.line_x = measured_x
            self.lost_count = 0
            self._candidate_x = None
            self._candidate_count = 0
            if was_lost:
                logger.info(f"target re-acquired at x={measured_x:.0f}")
            return measured_x, None

        self.lost_count += 1
        needed = self.params.REACQUIRE_CONFIRM_FRAMES
        return None, f"confirming {self._candidate_count}/{needed}"


class SteeringController:
    """
    FOLIAGE DEFENSE #5 -- proportional steering, and hold-heading when lost.

    While the target is visible this is a plain proportional controller with
    exponential smoothing. Once the target is lost, steering FREEZES at its last
    value and the throttle winds down: coast briefly in case the line comes
    straight back, then slow, then stop.

    A PID would be the obvious choice here but the integral term actively fights
    the freeze -- it keeps winding up while the car coasts blind, then snaps when
    the line returns. Proportional-plus-smoothing has no such state.
    """

    def __init__(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.steering = 0.0
        self.throttle = 0.0

    def state_for(self, lost_count):
        if lost_count == 0:
            return LostLineState.TRACKING
        if lost_count <= self.params.COAST_FRAMES:
            return LostLineState.COASTING
        if lost_count <= self.params.COAST_FRAMES + self.params.SLOW_FRAMES:
            return LostLineState.SLOWING
        return LostLineState.STOPPED

    def update(self, offset, lost_count):
        """
        :param offset: target position as -1.0 (hard left) .. +1.0 (hard right),
                       or None when there is nothing to steer to
        :param lost_count: consecutive frames without an accepted target
        :return: (steering, throttle, state)
        """
        params = self.params
        state = self.state_for(lost_count)

        if state is LostLineState.TRACKING and offset is not None:
            # Proportional control: the further off-center the target, the
            # harder we steer toward it.
            target = params.STEERING_KP * offset

            # Exponential smoothing takes the twitch out of frame-to-frame
            # noise. 0.0 is no smoothing; higher is smoother but laggier.
            smoothing = params.STEERING_SMOOTHING
            self.steering = smoothing * self.steering + (1.0 - smoothing) * target
            self.steering = float(np.clip(self.steering,
                                          -params.STEERING_MAX, params.STEERING_MAX))

            # Ease off the throttle in proportion to how hard we are turning.
            turn = abs(self.steering) / max(params.STEERING_MAX, 1e-6)
            scale = 1.0 - (1.0 - params.THROTTLE_TURN_SCALE) * turn
            self.throttle = params.THROTTLE_FORWARD * scale

        elif state is LostLineState.COASTING:
            # HOLD the last steering. Do not re-aim at anything.
            self.throttle = params.THROTTLE_FORWARD * params.COAST_THROTTLE_SCALE

        elif state is LostLineState.SLOWING:
            # Still holding heading; ramp the throttle down to zero.
            elapsed = lost_count - params.COAST_FRAMES
            remaining = 1.0 - (elapsed / max(params.SLOW_FRAMES, 1))
            self.throttle = (params.THROTTLE_FORWARD
                             * params.COAST_THROTTLE_SCALE
                             * max(0.0, remaining))

        else:  # STOPPED
            self.throttle = 0.0

        return self.steering, self.throttle, state


def offset_from_x(target_x, width):
    """
    Convert a pixel position into a steering error.

    0.0 means dead center, -1.0 the far left edge, +1.0 the far right edge.
    """
    half_width = width / 2.0
    return (target_x - half_width) / half_width
