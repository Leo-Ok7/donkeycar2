"""
Thread-safe mode and lane selection shared between the CV part and the web page.

Two threads touch this:
  - the vehicle loop thread, which READS it once per frame
  - the web server thread, which WRITES it when a button is clicked

Every access goes through a lock, and readers take a single immutable snapshot
rather than reading fields one at a time -- otherwise a click landing mid-read
could hand the pipeline a half-updated combination (e.g. new mode, old lane).
"""

import logging
import threading
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Mode(Enum):
    """Which driving strategy is active."""
    LINE = "line"   # stage 1: follow a single yellow line
    LANE = "lane"   # stage 2: drive a lane between yellow boundaries

    @classmethod
    def parse(cls, value, default=None):
        """Accept a Mode, or a string like "line"/"LINE". None on bad input."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return default


class Lane(Enum):
    """
    Which lane to drive in.

    CENTER means "no lane offset" and is what line following uses -- stage 1
    ignores lane selection entirely, but reads the same state object so that
    stage 2 plugs in without a rewrite.
    """
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"

    @classmethod
    def parse(cls, value, default=None):
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except (ValueError, AttributeError):
            return default


class StateSnapshot(NamedTuple):
    """An immutable, self-consistent view of the state at one instant."""
    mode: Mode
    lane: Lane
    debug: bool


class PipelineState:
    """
    Mode / lane / debug selection, safe to read and write from any thread.
    """

    def __init__(self, mode=Mode.LINE, lane=Lane.LEFT, debug=False):
        self._lock = threading.Lock()
        self._mode = mode
        self._lane = lane
        self._debug = bool(debug)

    def snapshot(self) -> StateSnapshot:
        """Read all three fields as one consistent set. Call once per frame."""
        with self._lock:
            return StateSnapshot(self._mode, self._lane, self._debug)

    def set_mode(self, mode) -> bool:
        """Returns True if the value was valid and applied."""
        parsed = Mode.parse(mode)
        if parsed is None:
            logger.warning(f"ignoring invalid mode: {mode!r}")
            return False
        with self._lock:
            changed = parsed is not self._mode
            self._mode = parsed
        if changed:
            logger.info(f"mode -> {parsed.value}")
        return True

    def set_lane(self, lane) -> bool:
        parsed = Lane.parse(lane)
        if parsed is None:
            logger.warning(f"ignoring invalid lane: {lane!r}")
            return False
        with self._lock:
            changed = parsed is not self._lane
            self._lane = parsed
        if changed:
            logger.info(f"lane -> {parsed.value}")
        return True

    def set_debug(self, debug) -> bool:
        with self._lock:
            self._debug = bool(debug)
        return True


# ---------------------------------------------------------------------------
# Process-wide singleton.
#
# The CV part is constructed by donkeycar from a config-named module path, so
# manage.py never holds a reference to it and cannot hand the same state object
# to the web part. A singleton is how both find each other. There is exactly one
# vehicle pipeline per process, so one shared instance is the correct scope.
# ---------------------------------------------------------------------------

_state = None
_state_lock = threading.Lock()


def get_pipeline_state(params=None) -> PipelineState:
    """
    Get (creating on first call) the shared PipelineState.

    `params` is only used the first time, to pick the starting mode/lane from
    config. Later calls return the existing instance and ignore it.
    """
    global _state
    with _state_lock:
        if _state is None:
            mode = Mode.LINE
            lane = Lane.LEFT
            debug = False
            if params is not None:
                mode = Mode.parse(params.START_MODE, Mode.LINE)
                lane = Lane.parse(params.START_LANE, Lane.LEFT)
                debug = params.DEBUG_OVERLAY
            _state = PipelineState(mode=mode, lane=lane, debug=debug)
            logger.info(
                f"pipeline state created: mode={mode.value}, lane={lane.value}, "
                f"debug={debug}"
            )
        return _state


def reset_pipeline_state():
    """Drop the singleton. Only used by tests."""
    global _state
    with _state_lock:
        _state = None
