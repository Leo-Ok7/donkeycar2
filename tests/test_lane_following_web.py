"""
Tests for the mode/lane toggle web server.

These start a real tornado server on a free port and talk to it over HTTP, so
they exercise the actual request handling, the MJPEG stream and the thread-safe
frame buffer rather than mocks.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from donkeycar.parts.lane_following.params import Params
from donkeycar.parts.lane_following.state import Mode, reset_pipeline_state
from donkeycar.parts.lane_following.web import LaneFollowingWebServer


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server():
    reset_pipeline_state()
    web = LaneFollowingWebServer(port=free_port(), params=Params(None))
    thread = threading.Thread(target=web.update, daemon=True)
    thread.start()
    assert web.ready.wait(timeout=10), "server never started listening"
    try:
        yield web
    finally:
        web.shutdown()
        thread.join(timeout=5)
        reset_pipeline_state()


def url(server, path):
    return f"http://127.0.0.1:{server.port}{path}"


def get_json(server, path):
    with urllib.request.urlopen(url(server, path), timeout=5) as response:
        return json.loads(response.read())


def post_json(server, path, payload):
    request = urllib.request.Request(
        url(server, path), data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def a_frame(value=90):
    frame = np.zeros((240, 426, 3), dtype=np.uint8)
    frame[:] = value
    return frame


# ---------------------------------------------------------------------------
# Startup behavior
# ---------------------------------------------------------------------------

def test_port_conflict_fails_loudly_at_construction():
    """
    A port clash must raise immediately with a helpful message, not leave a dead
    background thread and a page that never loads.
    """
    reset_pipeline_state()
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        with pytest.raises(RuntimeError) as info:
            LaneFollowingWebServer(port=port, params=Params(None))
        message = str(info.value)
        assert str(port) in message, "the error should name the port"
        assert "LANE_WEB_PORT" in message, "the error should say how to fix it"
    finally:
        blocker.close()
        reset_pipeline_state()


def test_index_page_loads(server):
    with urllib.request.urlopen(url(server, "/"), timeout=5) as response:
        body = response.read().decode()
    assert response.status == 200
    for expected in ["Line following", "Lane following", "/video", "/api/mode"]:
        assert expected in body


# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------

def test_state_endpoint_reports_defaults(server):
    state = get_json(server, "/api/state")
    assert state["mode"] == "line"
    assert state["lane"] in ("left", "right")
    assert state["debug"] is False
    assert state["lane_enabled"] is False, "lane choice is inert in line mode"


def test_mode_toggle_updates_shared_state(server):
    result = post_json(server, "/api/mode", {"mode": "lane"})
    assert result["ok"] is True
    assert result["mode"] == "lane"

    # The CV part reads the same singleton, so it must see the change.
    assert server.state.snapshot().mode is Mode.LANE
    assert get_json(server, "/api/state")["lane_enabled"] is True

    post_json(server, "/api/mode", {"mode": "line"})
    assert server.state.snapshot().mode is Mode.LINE


def test_lane_toggle_updates_shared_state(server):
    assert post_json(server, "/api/lane", {"lane": "right"})["lane"] == "right"
    assert server.state.snapshot().lane.value == "right"
    assert post_json(server, "/api/lane", {"lane": "left"})["lane"] == "left"


def test_debug_toggle_updates_shared_state(server):
    assert post_json(server, "/api/debug", {"debug": True})["debug"] is True
    assert server.state.snapshot().debug is True
    assert post_json(server, "/api/debug", {"debug": False})["debug"] is False


def test_invalid_values_are_rejected_without_changing_state(server):
    before = server.state.snapshot()
    for path, payload in [("/api/mode", {"mode": "banana"}),
                          ("/api/lane", {"lane": "sideways"}),
                          ("/api/mode", {"nope": "line"})]:
        with pytest.raises(urllib.error.HTTPError) as info:
            post_json(server, path, payload)
        assert info.value.code == 400
    assert server.state.snapshot() == before


def test_malformed_json_is_rejected(server):
    request = urllib.request.Request(
        url(server, "/api/mode"), data=b"not json",
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(request, timeout=5)
    assert info.value.code == 400


# ---------------------------------------------------------------------------
# The frame buffer and MJPEG stream
# ---------------------------------------------------------------------------

def test_no_frame_yet_does_not_error(server):
    """Before the camera delivers anything, latest_jpeg is simply None."""
    assert server.latest_jpeg() is None


def test_frame_is_encoded_and_cached(server):
    frame = a_frame()
    server.run_threaded(frame)

    first = server.latest_jpeg()
    assert first is not None and first.startswith(b"\xff\xd8"), "should be JPEG"

    # Same frame object -> reuse the encoding rather than redoing the work for
    # every connected browser.
    assert server.latest_jpeg() is first

    # A new frame must produce new bytes.
    server.run_threaded(a_frame(200))
    assert server.latest_jpeg() is not first


def test_run_threaded_ignores_none_and_keeps_last_frame(server):
    server.run_threaded(a_frame())
    encoded = server.latest_jpeg()
    server.run_threaded(None)
    assert server.latest_jpeg() is encoded, "None must not clear the last frame"


def test_mjpeg_stream_delivers_frames(server):
    server.run_threaded(a_frame())
    with urllib.request.urlopen(url(server, "/video"), timeout=10) as response:
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        chunk = response.read(2048)
    assert b"Content-type: image/jpeg" in chunk
    assert b"\xff\xd8" in chunk, "a JPEG should have been written"


def test_concurrent_frame_writes_and_reads_are_safe(server):
    """
    Hammer the buffer from both sides. Any JPEG returned must be complete and
    decodable -- never a torn or partial image.
    """
    import cv2

    stop = threading.Event()

    def writer():
        value = 0
        while not stop.is_set():
            server.run_threaded(a_frame(value % 256))
            value += 7

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 2.0
        reads = 0
        while time.time() < deadline:
            jpeg = server.latest_jpeg()
            if jpeg is None:
                continue
            decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                   cv2.IMREAD_COLOR)
            assert decoded is not None, "returned bytes were not a valid JPEG"
            assert decoded.shape == (240, 426, 3)
            reads += 1
        assert reads > 20, f"expected many reads, got {reads}"
    finally:
        stop.set()
        thread.join(timeout=5)


def test_rgb_camera_order_is_converted_for_display():
    """
    With an RGB camera the server must swap channels before encoding, or the
    browser shows red and blue reversed.
    """
    import cv2

    reset_pipeline_state()

    class RgbCfg:
        CAMERA_COLOR_ORDER = "RGB"

    web = LaneFollowingWebServer(port=free_port(), params=Params(RgbCfg()))
    try:
        # Pure red expressed in RGB order.
        frame = np.zeros((240, 426, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        web.run_threaded(frame)

        decoded = cv2.imdecode(np.frombuffer(web.latest_jpeg(), np.uint8),
                               cv2.IMREAD_COLOR)
        # cv2 decodes to BGR, so red must be in the LAST channel.
        assert decoded[:, :, 2].mean() > 200
        assert decoded[:, :, 0].mean() < 60
    finally:
        reset_pipeline_state()
