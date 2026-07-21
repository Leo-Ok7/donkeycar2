# CenterLineFollower

A classical computer-vision line follower for DonkeyCar: no neural net, just
OpenCV color thresholding + contour detection + a proportional steering
controller. Built for a track where the **edges are white tape** and the
**center is marked with intermittent (dashed) greenish-blue/teal tape** —
this Part tracks the teal center markers at a constant throttle.

Code: [`center_line_follower.py`](center_line_follower.py), class
`CenterLineFollower`.

**What this is not (yet):** lane-following against the white boundary tape.
This Part has a `LaneMode` (`LEFT`/`CENTER`/`RIGHT`) seam built in so a
future stage can toggle which lane the car should track from a web UI
button, but today all three modes behave identically — see
[The lane-select seam](#the-lane-select-seam) at the bottom.

## Pipeline

```
frame (RGB, e.g. 426x240 from the OAK-D)
  -> crop to a tall ROI band (ROI_Y_TOP..ROI_Y_BOTTOM)   ("ground ahead of the car")
  -> HSV convert                                          (cv2.COLOR_RGB2HSV)
  -> cv2.inRange(COLOR_LOW, COLOR_HIGH)                    (teal mask)
  -> morphological OPEN then CLOSE                         (denoise, bridge small gaps)
  -> cv2.findContours, filter by area
  -> pick the largest surviving contour
  -> centroid (cv2.moments) -> normalized horizontal error
  -> steering = clamp(Kp*smoothed_error + Kd*d(error)/step)
  -> throttle = constant while tracking
```

### Why a tall ROI band, not a thin slice

The center tape is **intermittent** — a thin, single-row scan can land
exactly in a gap between dashes and see nothing even though tape is clearly
visible a few rows above or below. A tall band (`ROI_Y_TOP` to
`ROI_Y_BOTTOM`, ~100px by default) makes it far more likely that *some*
dash fragment intersects it on any given frame. This is the main
gap-tolerance lever at the vision-pipeline level; the state machine further
down handles the temporal side of gap tolerance (holding heading across the
frames where nothing is found at all).

### Why HSV, and specifically why saturation, not hue

The track has **two** tape colors in frame: white edges, teal center.
Hue alone is not a reliable way to tell them apart: white (or any light
gray) surface reflects all wavelengths roughly equally, which means it has
**near-zero saturation** — and when saturation is near zero, the *hue
reading itself* becomes numerically unstable (hue is essentially undefined
for a colorless pixel, so sensor noise can make it read as anything,
including a value that happens to fall inside the teal hue band by chance).

**Saturation is what actually separates them.** Teal tape is a deliberately
vivid, high-saturation material — its `S` channel reliably reads well above
the `COLOR_LOW` floor (default `80`/255). White tape and the gray/silver
track floor are low-saturation regardless of lighting. So if the mask
starts picking up white tape, the fix is to **raise the saturation floor**
in `CENTER_LINE_COLOR_LOW`, not to narrow the hue range.

`V` (brightness) is deliberately kept wide (`40`-`255` by default) — it
only excludes near-black shadow/gaps, and does not help distinguish teal
from white (a bright teal pixel and a bright white pixel can share the same
`V`).

One inherent limit worth knowing about: a specular highlight/glare hotspot
directly on glossy tape can locally wash a pixel toward low-saturation/
high-value (it looks white at the highlight's peak) even though it's
genuinely teal tape. No amount of HSV tuning fully removes this — it's
absorbed downstream by the morphological CLOSE step (bridges the resulting
small hole in the mask) and by the gap-tolerance state machine if a whole
dash gets washed out.

### Tuning the HSV thresholds: use `scripts/hsv_picker.py`

Don't hand-tune blind — this repo already has an interactive tool for this
at `scripts/hsv_picker.py`: it shows trackbars for H/S/V low and high,
lets you click-drag a rectangle over a region to auto-sample its HSV range,
and shows a live masked preview. Press `p` to print the current values, `q`
to print-and-quit, Escape to reset.

Two practical ways to use it against this pipeline:

- **A standard USB/V4L webcam**, live: `python scripts/hsv_picker.py
  --camera 0`. Note this does *not* work directly against the OAK-D — it's
  a generic `cv2.VideoCapture`, and the OAK-D isn't a normal V4L device.
- **A saved sample frame** (the realistic OAK-D workflow): capture one
  frame first — e.g. record a short tub with `CV_CONTROLLER_MODULE`/
  `CLASS` pointed at this Part and `OVERLAY_IMAGE=True`, or a quick one-off
  script that constructs `donkeycar.parts.oak_d.OakD` and calls
  `cv2.imwrite()` on a frame — then run `python scripts/hsv_picker.py
  --file <path-to-frame>` and click-drag over the teal tape in the image.
  Copy the printed low/high values into `CENTER_LINE_COLOR_LOW`/
  `CENTER_LINE_COLOR_HIGH` in `myconfig.py`.

## Control law

`steering = clamp(Kp * smoothed_error + Kd * d(smoothed_error))`, where
`smoothed_error` is an exponential moving average of the normalized
horizontal offset between the tape centroid and the target pixel (roughly
`-1..1`, positive when the tape is right of target).

- Error is normalized to the image half-width, so `Kp` doesn't need to be
  re-derived if resolution changes — only the color thresholds do.
- `Kp` (`CENTER_LINE_STEER_KP`, default `0.8`): "tape at the edge of the
  ROI → ~80% steering" is the tuning anchor.
- `Kd` (`CENTER_LINE_STEER_KD`, default `0`, off): only turn this on if
  `Kp` alone starts to overshoot/oscillate. The derivative is computed only
  between frames that both had a confident detection — during a gap, the
  error is simply held, so no artificial spike gets computed across the gap
  boundary once tape reappears.
- No integral term, deliberately: an I-term is exactly what's prone to
  windup on a track that guarantees periodic loss of signal (the dash
  gaps). A persistent one-sided drift (e.g. an off-center camera mount) is
  fixed directly via `CENTER_LINE_TARGET_PIXEL` instead.
- Throttle is **strictly constant** (`CENTER_LINE_THROTTLE`) while tracking
  — only steering reacts to the tape. It only drops below that during an
  extended loss of the line; see [Gap tolerance](#gap-tolerance-and-failure-modes) below.

### Tuning procedure

1. Start with `CENTER_LINE_STEER_KD = 0.0`, `CENTER_LINE_STEER_KP = 0.8`.
   Tune in DonkeySim first if possible (see caveat below).
2. Car drifts off-track on curves, or reacts sluggishly → raise `Kp` in
   steps of about `0.15`-`0.2`.
3. Car visibly zig-zags/oscillates even on straights → lower `Kp`.
4. Target feel: smooth tracking on straights, at most one small settling
   correction per curve.
5. Only if a slow, repeating overshoot persists after `Kp` is otherwise
   good, add a small `Kd` (start around `Kp / 10`). Twitchy/jittery
   steering — especially right around gap boundaries — usually means `Kd`
   is too high (or the mask is noisy; try lowering
   `CENTER_LINE_ERROR_SMOOTHING_ALPHA` instead).
6. Re-tune `Kp`/`Kd` again on the physical car starting from your sim-tuned
   values, not from scratch — real servo response and tire grip differ from
   sim physics, so some retuning is expected, not a sign of a bug.

Note: the PID increment/decrement buttons already wired up in
`cv_control.py` (`INC_PID_P_BTN` etc.) tune the *stock* `LineFollower`'s
`PID_P`/`PID_D` — they are **not** connected to `CENTER_LINE_STEER_KP/KD`.
Tune this Part by editing `myconfig.py` and restarting `manage.py drive`.

## Gap tolerance and failure modes

One rule, applied consistently: **a tracked value (steering, throttle, the
smoothed error) is only updated on a frame where the tape is confidently
detected; otherwise it's left exactly as it was.** A four-state machine
built on that rule, keyed off wall-clock time since the tape was last seen:

| State | Condition | Steering | Throttle |
|---|---|---|---|
| `TRACKING` | detected this frame | fresh | `CENTER_LINE_THROTTLE` (constant) |
| `HOLD` | not detected, `elapsed <= HOLD_TIME_SEC` | held | **unchanged** — still constant |
| `DEGRADED` | not detected, `HOLD_TIME_SEC < elapsed <= LOST_TIME_SEC` | held | linear ramp down to `THROTTLE_LOST_MIN` |
| `STOPPED` | not detected, `elapsed > LOST_TIME_SEC` | held | `THROTTLE_LOST_MIN` (default `0.0`) |

`TRACKING` and `HOLD` both keep throttle strictly constant — a brief
dash-gap is expected, not a fault. Only once a gap has run on for far
longer than any single dash-gap plausibly should does the car slow down and
eventually stop. To tune `HOLD_TIME_SEC`/`LOST_TIME_SEC`: measure your
track's longest dash gap, estimate how long the car takes to cross it at
`CENTER_LINE_THROTTLE`, and set `HOLD_TIME_SEC` to comfortably (1.5-2x)
more than that; set `LOST_TIME_SEC` to something clearly beyond any
plausible gap (3-5x `HOLD_TIME_SEC`).

There is **no active search/recovery maneuver** once `STOPPED` — this is a
deliberate scope boundary. The car just stays stopped (steering held,
throttle at floor) until the line reappears in frame (which resets the
"last seen" clock automatically, no special-casing needed) or a human takes
over via the normal user/autopilot mode toggle.

| Failure mode | Mitigation |
|---|---|
| Tape lost across a long gap | The `HOLD -> DEGRADED -> STOPPED` timeout tiers above |
| Glare / specular highlight on glossy tape | Wide `V` band (doesn't discriminate on brightness) + morphological CLOSE (fills the resulting mask hole) + area-fraction tolerance |
| Sharp curves | Tall, near-field ROI (more likely to catch a curving dash); if the tape exits the ROI/frame entirely, "hold last steering" is a reasonable fallback mid-curve — better than snapping back to straight |
| White/teal color confusion | Saturation floor is the actual discriminator, not hue — see above |

## Tunable parameters

Every constant below has a `DEFAULT_*` at the top of
`center_line_follower.py`, and can be overridden by setting the same
`CENTER_LINE_<NAME>` in `myconfig.py` (a commented reference copy of all of
them also lives in `donkeycar/templates/cfg_cv_control.py`, right after the
stock `LineFollower` block). None are required — the Part reads
`getattr(cfg, 'CENTER_LINE_<NAME>', DEFAULT_<NAME>)`, so it runs out of the
box and every knob is still reachable without touching the Part's code.

| Group | Parameters | Sim -> real notes |
|---|---|---|
| ROI | `ROI_Y_TOP`, `ROI_Y_BOTTOM` | Needs retuning — camera mount height/angle differs |
| Color | `COLOR_LOW`, `COLOR_HIGH` | **Retune first.** Lighting and the exact tape tint will differ the most between sim and the real camera |
| Morphology | `MORPH_KERNEL` | Usually transfers; raise it if the real camera's mask looks grainier |
| Contour | `MIN_AREA_FRACTION` | Only minor retuning, mostly if ROI size changes |
| Target | `TARGET_PIXEL` | Transfers as `None` (frame center); override only for an off-center camera mount |
| Steering | `STEER_KP`, `STEER_KD`, `ERROR_SMOOTHING_ALPHA` | `Kp`/`Kd` need retuning (servo response and grip differ); `ALPHA` usually transfers |
| Throttle | `THROTTLE`, `THROTTLE_LOST_MIN` | `THROTTLE` always needs per-vehicle retuning; start much lower on the real car than in sim |
| Gap timing | `HOLD_TIME_SEC`, `LOST_TIME_SEC` | Concept transfers if dash spacing is similar; values are coupled to `THROTTLE`, retune together |
| Debug | `OVERLAY_IMAGE` (unprefixed, shared with `LineFollower`) | Draws the ROI box, mask overlay (green tint), centroid crosshair, and current state onto the output image — view it live via the web UI's pilot-image pane while tuning |

## Running it

This Part is a drop-in replacement for
`donkeycar.parts.line_follower.LineFollower` in the existing
`donkeycar/templates/cv_control.py` template — that template's
`add_cv_controller()` already dynamically imports whatever module/class is
named in config, so **no template or `manage.py` edits are needed**. In
your `myconfig.py`:

```python
CV_CONTROLLER_MODULE = "donkeycar.parts.center_line_follower"
CV_CONTROLLER_CLASS = "CenterLineFollower"

# PID_P / PID_I / PID_D must stay defined even though this Part ignores
# them -- cv_control.py constructs a PID from them unconditionally before
# the CV controller is even selected. The stock commented-out values are
# fine as-is.

# Then override any tunables from the table above, e.g.:
CENTER_LINE_COLOR_LOW = (75, 80, 40)
CENTER_LINE_COLOR_HIGH = (105, 255, 255)
CENTER_LINE_THROTTLE = 0.2
OVERLAY_IMAGE = True
```

### In DonkeySim

1. Set `DONKEY_GYM = True` (and the gym env fields) in `myconfig.py` as
   usual.
2. `python manage.py drive`, open the web UI, switch to autopilot mode.
3. **Caveat:** no stock DonkeySim track is confirmed to have this track's
   specific white-boundary / teal-dashed-center visual design. If the
   track you have doesn't, you can still validate the wiring, control law,
   and gap-timing logic against whatever line color it does have (point
   `CENTER_LINE_COLOR_LOW`/`HIGH` at that color temporarily) — just don't
   treat those HSV numbers as valid for the real track. Watch the
   pilot-image pane (`OVERLAY_IMAGE = True`) and adjust thresholds until
   the green mask overlay cleanly covers the line and nothing else.

### On the physical car (OAK-D)

1. In `myconfig.py`:
   ```python
   CAMERA_TYPE = "OAKD"
   IMAGE_W = 426
   IMAGE_H = 240
   OAKD_RGB = True
   OAKD_DEPTH = True   # see note below
   ```
   `IMAGE_W`/`IMAGE_H` control the on-device DepthAI preview size (no
   native-resolution frames are processed on the Pi). **Note:**
   `OAKD_DEPTH = False` looks like a reasonable way to save USB bandwidth
   since this Part doesn't use depth, but there's a separate, pre-existing
   bug in `oak_d.py`'s `_poll()` that makes `OAKD_DEPTH = False` crash at
   runtime (it unconditionally fetches the depth queue whenever either
   stream is enabled) — leave it `True` for now.
2. First-run sanity check: point the camera at a known red or blue object
   and inspect a captured pixel before trusting the color thresholds — the
   OAK-D pipeline was patched to explicitly request RGB output
   (`setColorOrder(...ColorOrder.RGB)`), which should make
   `cam/image_array` genuinely RGB like every other camera in this
   codebase, but this is new, unverified-on-hardware code.
3. Re-tune `CENTER_LINE_COLOR_LOW`/`HIGH` with `scripts/hsv_picker.py`
   against a captured real frame (see the tuning section above) — real
   lighting and tape will differ from sim.
4. Set `CENTER_LINE_THROTTLE` much lower than your sim value for the first
   test.
5. Drive test: watch the overlay, confirm tracking looks right at very low
   throttle, then raise it gradually.

## The lane-select seam

`CenterLineFollower` accepts a `LaneMode` (`LEFT`/`CENTER`/`RIGHT`), both as
a constructor default and via `run(cam_img, lane_mode=...)` /
`set_lane_mode(...)`. Today, all three modes resolve to the same
frame-center target (`_get_target_x()` has an explicit `if/elif/else` per
mode, but all three branches currently return the same value) — this is a
seam for the future lane-following stage, not a working feature yet. A real
implementation will need its own white-boundary-tape HSV mask and
boundary-relative target math.

`run()`'s `lane_mode` parameter is already wired to be a no-op today:
`CV_CONTROLLER_INPUTS = ['cam/image_array']` (one entry) means
`Vehicle.update_parts()` calls `run(cam_img)` and `lane_mode` takes its
default. When lane-following is built, the plan is:

1. Add a small Part (shaped like `complete.py`'s existing
   `ToggleRecording`) that owns and toggles a `'lane/mode'` memory value,
   wired to a `web/wN` button using the same `Lambda` + `run_condition`
   pattern already used for `TOGGLE_RECORDING_BTN`/`INC_PID_P_BTN` in
   `cv_control.py`.
2. Extend `CV_CONTROLLER_INPUTS = ['cam/image_array', 'lane/mode']` in
   `myconfig.py`.

No edits to `cv_control.py` or this Part's core pipeline are needed for
that — only new config and one small new toggle Part.
