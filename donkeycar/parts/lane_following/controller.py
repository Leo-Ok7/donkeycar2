"""
The donkeycar Part that the cv_control template loads as the autopilot.

The template picks this class by name from config (CV_CONTROLLER_MODULE /
CV_CONTROLLER_CLASS) and adds it with run_condition="run_pilot", which means it
does not execute at all in manual ("user") mode. That is what keeps the stock
manual-drive path safe: even a part that raised on every call could not stop you
driving the car by hand.

The contract is fixed by the template, see donkeycar/templates/cv_control.py:

    part = TheClass(pid, cfg)
    steering, throttle, image = part.run(cam_img)
"""

import logging

import cv2

from donkeycar.parts.lane_following import overlay
from donkeycar.parts.lane_following.params import Params
from donkeycar.parts.lane_following.state import Lane, Mode, get_pipeline_state
from donkeycar.parts.lane_following.strategies import build_strategy

logger = logging.getLogger(__name__)


class PassThroughController:
    """
    Step 1 only: a CV part that steers nothing and drives nothing.

    Its entire job is to prove the wiring is right -- that the part loads, has
    the signature the vehicle loop expects, and returns the three outputs
    manage.py maps to pilot/steering, pilot/throttle and cv/image_array --
    before any autonomous logic exists to confuse the picture.

    Switching to autopilot with this part loaded should make the car sit
    perfectly still with no traceback. If that works, the integration is sound.
    """

    def __init__(self, pid, cfg):
        # `pid` is supplied by the template's add_cv_controller(). This part has
        # no controller to tune, so it is accepted and ignored.
        self.params = Params(cfg)
        self.state = get_pipeline_state(self.params)
        self.frame_count = 0
        logger.info(
            "PassThroughController active: autopilot will output "
            "steering=0.0, throttle=0.0 (step 1 wiring check)"
        )

    def run(self, cam_img):
        """
        :param cam_img: frame from the camera, or None before the first frame
        :return: (steering, throttle, image) -- zeros, and the frame untouched
        """
        self.frame_count += 1

        # The camera part is threaded, so the first few loop ticks can run
        # before any frame has arrived. Returning None for the image is fine;
        # the web UI shows its placeholder.
        if cam_img is None:
            return 0.0, 0.0, None

        if self.frame_count == 1:
            snapshot = self.state.snapshot()
            logger.info(
                f"first frame: shape={cam_img.shape}, dtype={cam_img.dtype}, "
                f"mode={snapshot.mode.value}, lane={snapshot.lane.value}"
            )

        return 0.0, 0.0, cam_img

    def shutdown(self):
        pass


class LaneFollowingController:
    """
    The real autopilot: yellow line following, with lane following added in
    step 3 behind the mode selector.

    Also the same donkeycar Part contract as above -- constructed with
    (pid, cfg), called as run(cam_img) -> (steering, throttle, image), and only
    executed when the car is in an autopilot mode.
    """

    def __init__(self, pid, cfg):
        # `pid` comes from the template. This controller uses its own
        # proportional controller (see pipeline.LostLineController) because the
        # lost-line behavior needs to freeze the output, which a PID's integral
        # term fights against. It is accepted and ignored.
        self.params = Params(cfg)
        self.state = get_pipeline_state(self.params)
        self.frame_count = 0
        self._last_snapshot = None
        self._logged_state = None

        # One strategy instance per mode, built up front and kept for the run.
        # Building both here rather than on demand means switching modes mid-drive
        # never pauses the vehicle loop to construct objects, and each keeps its
        # own state so neither can corrupt the other.
        self.strategies = {mode: build_strategy(mode, self.params)
                           for mode in Mode}

        logger.info(
            f"LaneFollowingController ready: throttle={self.params.THROTTLE_FORWARD}, "
            f"steering_kp={self.params.STEERING_KP}, "
            f"yellow={self.params.YELLOW_HSV_LOW}..{self.params.YELLOW_HSV_HIGH}, "
            f"color_order={self.params.CAMERA_COLOR_ORDER}"
        )

    def _to_bgr(self, frame):
        """
        Put the frame in BGR order, which is what cv2's HSV conversion assumes.

        This is the ONE place color order is handled. If CAMERA_COLOR_ORDER is
        wrong the yellow mask is empty on every frame with nothing to explain
        why, so confirm it with scripts/oakd_color_check.py rather than guessing.
        """
        if self.params.CAMERA_COLOR_ORDER == "RGB":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def _from_bgr(self, frame):
        """Convert a BGR debug image back to whatever the rest of the car uses."""
        if self.params.CAMERA_COLOR_ORDER == "RGB":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def run(self, cam_img):
        """
        :param cam_img: camera frame, or None before the first one arrives
        :return: (steering, throttle, image)
        """
        # The camera part is threaded, so the loop can tick before any frame
        # exists. Do not move the car on a guess.
        if cam_img is None:
            return 0.0, 0.0, None

        self.frame_count += 1
        snapshot = self.state.snapshot()

        strategy = self.strategies[snapshot.mode]

        # A mode or lane change resets the strategy about to run: last offset,
        # lost counters, plausibility history and lane-width estimates all go.
        # Without this, switching mid-drive would act on a target that belonged
        # to a different interpretation of the scene, and the car would swerve.
        if self._last_snapshot is None or (
                snapshot.mode is not self._last_snapshot.mode
                or snapshot.lane is not self._last_snapshot.lane):
            if self._last_snapshot is not None:
                logger.info(
                    f"mode/lane changed to {snapshot.mode.value}/"
                    f"{snapshot.lane.value}, resetting {strategy.name} strategy"
                )
            strategy.reset()
            self._logged_state = None
        self._last_snapshot = snapshot

        frame_bgr = self._to_bgr(cam_img)
        # Line following ignores the lane argument; it takes the same signature
        # so the controller does not need to know which strategy it is calling.
        result = strategy.process(frame_bgr, snapshot.lane)

        # Log state transitions once, not every frame -- this is the line that
        # tells you the car noticed it lost the line.
        if result.state is not self._logged_state:
            logger.info(
                f"{result.state.value}: steer={result.steering:+.2f} "
                f"throttle={result.throttle:.2f} "
                f"area={result.detection.area_frac * 100:.2f}%"
                + (f" ({result.reject_reason})" if result.reject_reason else "")
            )
            self._logged_state = result.state

        if snapshot.debug:
            lane_label = "-" if snapshot.mode is Mode.LINE else snapshot.lane.value
            debug_image = overlay.draw(
                frame_bgr, result, strategy,
                mode_label=snapshot.mode.value.upper(),
                lane_label=lane_label,
                extra_lines=overlay.lane_info_lines(result.info),
            )
            return result.steering, result.throttle, self._from_bgr(debug_image)

        return result.steering, result.throttle, cam_img

    def shutdown(self):
        pass
