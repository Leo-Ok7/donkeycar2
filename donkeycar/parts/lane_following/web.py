"""
The mode/lane toggle page: a small tornado server with an MJPEG camera feed.

CAMERA STREAM OWNERSHIP
-----------------------
This server does NOT touch the camera. The OAK-D part remains the sole reader of
the DepthAI output queue; there is no second device handle and no second queue
consumer. Frames reach this server the ordinary donkeycar way -- as a part input
from vehicle memory (`cv/image_array`, which the CV part already produces).

NOT BLOCKING THE VEHICLE LOOP
-----------------------------
Added with threaded=True, so the tornado IOLoop runs in its own thread and the
vehicle loop only ever calls run_threaded(), which stores one reference and
returns. The loop never enters tornado code, so a slow or stuck browser cannot
slow the car down.

FAILING LOUDLY INSTEAD OF HANGING
---------------------------------
donkeycar's own WebFpv calls listen() inside the background thread, so a port
conflict kills that thread silently and the page just never appears. Here the
port is test-bound on the MAIN thread in __init__, which turns a conflict into an
immediate startup error naming the port. A `ready` event confirms the real listen
succeeded, so even an unexpected failure gets a loud warning rather than silence.

THREAD SAFETY
-------------
Two things cross threads:

  * mode/lane/debug -- owned by PipelineState, which is lock-guarded and read by
    the CV part as one atomic snapshot per frame.
  * the latest frame -- stored under a lock here. Only the reference is ever
    swapped; the CV part builds each debug overlay into a fresh array and never
    draws into one it has already handed over. So a request encoding a frame to
    JPEG holds an image nobody will modify underneath it, and cannot see a
    half-drawn one.
"""

import asyncio
import errno
import json
import logging
import socket
import threading

import cv2
from tornado.ioloop import IOLoop
from tornado.web import Application, RequestHandler
import tornado.gen
import tornado.iostream

from donkeycar.parts.lane_following.state import Lane, Mode, get_pipeline_state

logger = logging.getLogger(__name__)

# MJPEG stream settings.
JPEG_QUALITY = 70        # lower eases the load on the Pi and the network
STREAM_INTERVAL = 0.05   # seconds between frames pushed to the browser (~20 fps)


class IndexHandler(RequestHandler):
    """Serves the single-page UI. Inline, so there are no template files."""

    def get(self):
        self.set_header("Content-Type", "text/html")
        self.write(PAGE_HTML)


class VideoHandler(RequestHandler):
    """
    The MJPEG feed: an endless multipart response, one JPEG per part.

    Same approach as donkeycar's own VideoAPI. Encoding uses cv2 rather than PIL
    because PIL treats an array as RGB, which would swap red and blue on a BGR
    camera frame.
    """

    async def get(self):
        self.set_header("Content-Type",
                        "multipart/x-mixed-replace;boundary=--boundarydonotcross")
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate")

        while True:
            jpeg = self.application.latest_jpeg()
            if jpeg is None:
                # No frame yet (the camera part is threaded and may still be
                # starting). Wait rather than busy-loop or error out.
                await tornado.gen.sleep(STREAM_INTERVAL)
                continue

            try:
                self.write("--boundarydonotcross\n")
                self.write("Content-type: image/jpeg\r\n")
                self.write(f"Content-length: {len(jpeg)}\r\n\r\n")
                self.write(jpeg)
                await self.flush()
            except tornado.iostream.StreamClosedError:
                # The browser navigated away or refreshed. Perfectly normal.
                return
            except Exception as error:  # noqa: BLE001 - never kill the loop
                logger.debug(f"video stream ended: {error}")
                return

            await tornado.gen.sleep(STREAM_INTERVAL)


class StateHandler(RequestHandler):
    """GET the current mode/lane/debug, for the status display."""

    def get(self):
        snapshot = self.application.state.snapshot()
        self.write({
            "mode": snapshot.mode.value,
            "lane": snapshot.lane.value,
            "debug": snapshot.debug,
            # The lane buttons are meaningless in line mode; the page uses this
            # to grey them out.
            "lane_enabled": snapshot.mode is Mode.LANE,
        })


class _SetterHandler(RequestHandler):
    """Shared POST handling: read one JSON field and apply it to the state."""

    field = None

    def post(self):
        try:
            body = json.loads(self.request.body or b"{}")
        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"ok": False, "error": "body must be JSON"})
            return

        value = body.get(self.field)
        if value is None:
            self.set_status(400)
            self.write({"ok": False, "error": f"missing {self.field!r}"})
            return

        if not self.apply(value):
            self.set_status(400)
            self.write({"ok": False, "error": f"invalid {self.field}: {value!r}"})
            return

        snapshot = self.application.state.snapshot()
        self.write({"ok": True, "mode": snapshot.mode.value,
                    "lane": snapshot.lane.value, "debug": snapshot.debug})

    def apply(self, value):
        raise NotImplementedError


class ModeHandler(_SetterHandler):
    field = "mode"

    def apply(self, value):
        return self.application.state.set_mode(value)


class LaneHandler(_SetterHandler):
    field = "lane"

    def apply(self, value):
        # Accepted in either mode -- the pipeline simply ignores lane selection
        # while line following, so there is nothing unsafe about setting it
        # early. The page greys the buttons out to make that clear.
        return self.application.state.set_lane(value)


class DebugHandler(_SetterHandler):
    field = "debug"

    def apply(self, value):
        return self.application.state.set_debug(bool(value))


class LaneFollowingWebServer(Application):
    """
    A donkeycar part. Add it with threaded=True and one input:

        V.add(LaneFollowingWebServer(port=cfg.LANE_WEB_PORT),
              inputs=['cv/image_array'], threaded=True)
    """

    def __init__(self, port=8891, params=None):
        self.port = port
        self.state = get_pipeline_state(params)

        self._frame_lock = threading.Lock()
        self._frame = None
        self._encoded = None       # cache: the JPEG for _encoded_from
        self._encoded_from = None  # the frame object that JPEG came from
        self._bgr_input = (params is None
                           or getattr(params, "CAMERA_COLOR_ORDER", "BGR") == "BGR")

        self.ready = threading.Event()
        self._loop = None
        self.running = True

        # Fail fast, on the main thread, with a message that names the port.
        # Doing this here rather than in the thread is what turns a port clash
        # into a startup error instead of a page that silently never loads.
        self._check_port_available()

        handlers = [
            (r"/", IndexHandler),
            (r"/video", VideoHandler),
            (r"/api/state", StateHandler),
            (r"/api/mode", ModeHandler),
            (r"/api/lane", LaneHandler),
            (r"/api/debug", DebugHandler),
        ]
        super().__init__(handlers)

        logger.info(f"lane following toggle page will be at "
                    f"http://<your-hostname>.local:{self.port}")

    def _check_port_available(self):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", self.port))
        except OSError as error:
            hint = ""
            if error.errno == errno.EADDRINUSE:
                hint = (f" Another process is already using port {self.port}"
                        " (the donkeycar web controller uses 8887). Set"
                        " LANE_WEB_PORT in myconfig.py to a free port.")
            raise RuntimeError(
                f"Cannot start the lane following web page on port "
                f"{self.port}: {error}.{hint}"
            ) from error
        finally:
            probe.close()

    # -- the donkeycar part interface --------------------------------------

    def update(self):
        """Runs on the vehicle's background thread: serve until shutdown."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            self.listen(self.port)
        except Exception as error:  # noqa: BLE001
            # Should not happen, since __init__ test-bound the port -- but if it
            # does, say so loudly instead of leaving a dead thread behind.
            logger.error(f"lane following web page failed to start: {error}")
            return
        self._loop = IOLoop.current()
        self.ready.set()
        logger.info(f"lane following toggle page serving on port {self.port}")
        self._loop.start()

    def run_threaded(self, img_arr=None):
        """
        Called once per vehicle loop tick. Stores one reference and returns.

        Cheap on purpose: JPEG encoding happens in the web thread, per connected
        client, so an attached browser costs the driving loop nothing.
        """
        if img_arr is not None:
            with self._frame_lock:
                self._frame = img_arr
        return None

    def run(self, img_arr=None):
        return self.run_threaded(img_arr)

    def shutdown(self):
        self.running = False
        if self._loop is not None:
            # The IOLoop belongs to the other thread, so ask it to stop itself.
            self._loop.add_callback(self._loop.stop)

    # -- frame access from the web thread ----------------------------------

    def latest_jpeg(self):
        """
        The latest frame as JPEG bytes, or None if none has arrived.

        Encoding is cached per frame, so several browsers watching at once do not
        each re-encode the same image.
        """
        with self._frame_lock:
            frame = self._frame
            if frame is not None and frame is self._encoded_from:
                return self._encoded

        if frame is None:
            return None

        # cv2.imencode expects BGR. Frames arrive in the camera's native order.
        to_encode = frame if self._bgr_input else cv2.cvtColor(
            frame, cv2.COLOR_RGB2BGR)

        ok, buffer = cv2.imencode(
            ".jpg", to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        jpeg = buffer.tobytes()

        with self._frame_lock:
            self._encoded = jpeg
            self._encoded_from = frame
        return jpeg


PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DonkeyCar line / lane following</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 1rem; background: #14171a; color: #e8eaed;
         font: 15px/1.4 system-ui, -apple-system, sans-serif; }
  h1 { font-size: 1.1rem; margin: 0 0 .75rem; font-weight: 600; }
  #feed { width: 100%; max-width: 852px; image-rendering: pixelated;
          background: #000; border: 1px solid #2c3136; border-radius: 6px;
          display: block; }
  .panel { max-width: 852px; }
  .row { display: flex; align-items: center; gap: .5rem; margin: .75rem 0;
         flex-wrap: wrap; }
  .label { min-width: 4.5rem; color: #9aa0a6; font-size: .8rem;
           text-transform: uppercase; letter-spacing: .05em; }
  button { font: inherit; padding: .55rem 1.1rem; border-radius: 6px;
           border: 1px solid #3c4043; background: #22262a; color: #e8eaed;
           cursor: pointer; min-width: 7rem; }
  button:hover:not(:disabled) { background: #2c3136; }
  button.on { background: #1a73e8; border-color: #1a73e8; color: #fff;
              font-weight: 600; }
  button:disabled { opacity: .35; cursor: not-allowed; }
  #status { margin: .75rem 0 0; padding: .6rem .75rem; background: #1b1f23;
            border-radius: 6px; font-family: ui-monospace, monospace;
            font-size: .85rem; color: #9aa0a6; }
  #status b { color: #e8eaed; }
  .note { color: #9aa0a6; font-size: .8rem; margin-top: .5rem; }
</style>
</head>
<body>
<div class="panel">
  <h1>DonkeyCar &mdash; line / lane following</h1>
  <img id="feed" src="/video" alt="camera feed">

  <div class="row">
    <span class="label">Mode</span>
    <button id="mode-line" onclick="setMode('line')">Line following</button>
    <button id="mode-lane" onclick="setMode('lane')">Lane following</button>
  </div>

  <div class="row">
    <span class="label">Lane</span>
    <button id="lane-left" onclick="setLane('left')">Left</button>
    <button id="lane-right" onclick="setLane('right')">Right</button>
  </div>

  <div class="row">
    <span class="label">Debug</span>
    <button id="debug-toggle" onclick="toggleDebug()">Overlay</button>
  </div>

  <p id="status">connecting&hellip;</p>
  <p class="note">
    Autonomous control only runs while the car is in an autopilot mode. Set that
    on the donkeycar driving page (port 8887); this page selects which CV
    strategy the autopilot uses.
  </p>
</div>

<script>
let current = { mode: null, lane: null, debug: false };

async function post(path, body) {
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    render(await response.json());
  } catch (error) {
    document.getElementById('status').textContent = 'request failed: ' + error;
  }
}

const setMode = (mode) => post('/api/mode', { mode });
const setLane = (lane) => post('/api/lane', { lane });
const toggleDebug = () => post('/api/debug', { debug: !current.debug });

function render(state) {
  if (!state || state.mode === undefined) return;
  current = state;
  const laneMode = state.mode === 'lane';

  document.getElementById('mode-line').classList.toggle('on', !laneMode);
  document.getElementById('mode-lane').classList.toggle('on', laneMode);

  for (const side of ['left', 'right']) {
    const button = document.getElementById('lane-' + side);
    button.classList.toggle('on', laneMode && state.lane === side);
    // Lane choice does nothing in line mode, so show it as unavailable.
    button.disabled = !laneMode;
  }

  document.getElementById('debug-toggle').classList.toggle('on', !!state.debug);

  const laneText = laneMode ? state.lane.toUpperCase() : 'n/a (line mode)';
  document.getElementById('status').innerHTML =
    'mode <b>' + state.mode.toUpperCase() + '</b> &nbsp; ' +
    'lane <b>' + laneText + '</b> &nbsp; ' +
    'debug <b>' + (state.debug ? 'ON' : 'OFF') + '</b>';
}

async function poll() {
  try {
    render(await (await fetch('/api/state')).json());
  } catch (error) {
    document.getElementById('status').textContent = 'lost contact with the car';
  }
}
poll();
setInterval(poll, 500);
</script>
</body>
</html>
"""
