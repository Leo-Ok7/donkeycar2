# Line Following and Lane Following (classical CV)

Two autonomous behaviors for a DonkeyCar with a Luxonis OAK-D camera, built with
OpenCV only — no neural nets, no training, no simulator.

- **Line following** — follow a single strip of yellow tape.
- **Lane following** — drive a chosen lane between two yellow boundaries and a
  discontinuous yellow center divider.

Both run on the Raspberry Pi at 426x240 and are tuned entirely through named
config constants, so everything can be adjusted at the track.

---

## Step 0 — The OAK-D full-FOV camera patch

### The problem it fixes

The stock `OakD` part streams the camera's **`video`** output. That output is
hard-cropped to 16:9 no matter what `setResolution()` asks for, and the crop comes
off the sensor's native ~4:3 frame. What gets thrown away is the **bottom** of the
frame — the near-ground strip directly in front of the car. On turns the line
leaves the narrowed view early and the car loses it.

The fix is to stream the **`isp`** output instead, which preserves the true field
of view, and then squash the result down to 426x240.

> **Squash, don't crop.** The ISP frame is ~4:3 and the target is 16:9, so the
> saved image looks horizontally compressed. That is correct and intended.
> Re-cropping back toward 16:9 would throw away exactly the near-ground FOV the
> patch exists to recover.

### The edits

Both are in the `OakD` class in `donkeycar/parts/oak_d.py`.

**1. `setup_rgb_camera()` — capture the full sensor via the ISP path**

```python
# BEFORE (stock)                          # AFTER (patched)
res = ...SensorResolution.THE_1080_P      cam_rgb.setResolution(resolution)
cam_rgb.setResolution(res)                cam_rgb.setIspScale(scale_num, scale_den)
cam_rgb.setPreviewSize(w, h)              cam_rgb.setFps(CAMERA_FPS)
cam_rgb.setInterleaved(False)
...                                       ...
cam_rgb.video.link(xout_rgb.input)        cam_rgb.isp.link(xout_rgb.input)
```

**2. `_poll()` — pin the frame to the exact output size**

`setIspScale()` only hits approximate fractions (a 12MP sensor at 1/6 lands around
676x506), so the frame is resized to exactly `self.width` x `self.height`:

```python
if frame.shape[1] != self.width or frame.shape[0] != self.height:
    frame = cv2.resize(
        frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
    )
```

### Three crash fixes that came with it

The patch as originally drafted left `_poll()` inconsistent with `__init__`. All
three of these are fixed in the version in this repo:

| Bug | Effect |
|---|---|
| `self.resize` read in `_poll()` but never assigned in `__init__` | `AttributeError` on the **first frame** |
| `_poll()` rebuilt both queues every frame and read the depth queue unconditionally | Crash/hang whenever `OAKD_DEPTH = False` |
| `cv2.resize(img, (w, h), cv2.INTER_NEAREST)` | Interpolation flag passed as the positional `fx` argument |

Also: `shutdown()` now tolerates a device that never opened, and `add_camera()` in
`donkeycar/templates/complete.py` passes `cfg.IMAGE_W` / `cfg.IMAGE_H` to `OakD`
so config is the single source of truth (it previously ignored them).

### Sensor resolution is auto-detected

Which sensor resolution a board accepts is board-specific, and asking for an
unsupported one makes depthai **throw when the device is opened** — a launch
crash. So `_open_device()` tries candidates in order and uses the first that
works, logging the winner:

| Candidate | Board | Full res | at 1/6 |
|---|---|---|---|
| `THE_12_MP` | OAK-D (IMX378) | 4056x3040 | ~676x506 |
| `THE_13_MP` | OAK-D-Lite (IMX214) | 4208x3120 | ~701x520 |
| the same two at 1/4 | fallback for firmware that rejects uneven scale fractions | | |

All candidates are 4:3 modes. **Do not add a 16:9 mode such as `THE_4_K`** — that
silently reintroduces the crop this patch removes. If every candidate fails, the
part raises with the full list of what was tried.

### Re-applying after an environment rebuild

This patch edits the **installed** donkeycar package, so it does not travel with
the repo unless the Pi runs an editable install. After any fresh install or
environment rebuild:

```bash
~/env/bin/python scripts/patch_oak_d.py
```

Use the **same interpreter that runs `manage.py`**, or the script will find the
wrong donkeycar. It locates the installed `oak_d.py` via `importlib.util.find_spec`,
backs it up to `oak_d.py.bak`, and copies the patched file over it. It is
idempotent — running it twice is a no-op.

```bash
python scripts/patch_oak_d.py --status    # report state, exit 0 if already patched
python scripts/patch_oak_d.py --dry-run   # show what would change
python scripts/patch_oak_d.py --revert    # restore the .bak backup
```

It recognizes five situations and reports each explicitly: `editable` (the repo
*is* the install — nothing to do), `current` (already patched), `stock` (unpatched
— will patch), `patched-different` (older patch — will update), `missing`
(donkeycar not importable by this interpreter).

The installed file is typically at:

```
~/env/lib/python3.11/site-packages/donkeycar/parts/oak_d.py
```

### Verifying the patch

```bash
python scripts/oakd_color_check.py
```

This must print a frame shape of exactly `(240, 426, 3)`. It also writes two PNGs,
which settle the **color-order** question — see below.

Then look at the image and confirm you can see the ground **close to the car**, and
that the scene looks horizontally squashed. Squashed is correct. If it looks
normally proportioned but the near ground is missing, the patch did not take
effect.

### BGR vs RGB — settled empirically, not assumed

`ImgFrame.getCvFrame()` returns **BGR**, while donkeycar treats `cam/image_array`
as RGB. Getting this backwards is uniquely nasty: `cv2.cvtColor(..., COLOR_BGR2HSV)`
assumes BGR, so with the wrong order yellow tape lands nowhere near hue 20–35, the
mask comes back **empty forever**, and nothing raises an error.

So the pipeline does not guess. `scripts/oakd_color_check.py` writes the *same*
frame under both interpretations:

- `oakd_as_bgr.png` — if the colors look right here, set `CAMERA_COLOR_ORDER = "BGR"`
- `oakd_as_rgb.png` — if the colors look right here, set `CAMERA_COLOR_ORDER = "RGB"`

Point the camera at something strongly yellow or red first — those flip most
obviously. `CAMERA_COLOR_ORDER` is applied at exactly one place in the code (the
HSV conversion), so it is the only thing to change.

The correct-looking PNG then feeds straight into the existing HSV tuning tool:

```bash
python scripts/hsv_picker.py -f oakd_as_bgr.png
```

Drag a box over the yellow tape and it reports the HSV range to put in
`YELLOW_HSV_LOW` / `YELLOW_HSV_HIGH`.

---

## What each step added

The work was built and verified in stages, each proven before the next started.

| Step | Added |
|---|---|
| **0** | The OAK-D full-FOV camera patch (above), three crash fixes that came with it, sensor-resolution auto-detection, `scripts/patch_oak_d.py`, `scripts/oakd_color_check.py`. |
| **1** | `params.py` (all tunables), `state.py` (thread-safe mode/lane), `PassThroughController` (a CV part that outputs zeros), the `OAKD_*` config keys, `scripts/preflight_lane_following.py`. **No autonomous logic** — this step exists only to prove the car still launches and drives by hand. |
| **2** | Line following: `pipeline.py` with the yellow-centroid pipeline, proportional steering, the lost-line state machine, the plausibility gate and all five foliage defenses. `overlay.py` for the debug view. |
| **3** | **Refactor**, then lane following. See below for exactly what moved. |
| **4** | `web.py`: the tornado mode/lane/debug toggle page with an MJPEG feed. |

### What step 3 refactored out of step 2

Step 2's single `pipeline.py` was split so lane following could reuse it rather
than reimplement it. `pipeline.py` no longer exists; its contents moved:

| From `pipeline.py` | To | What it is now |
|---|---|---|
| ROI bounds + trapezoid cache | `vision.py` | `RoiGeometry` |
| HSV masking, morphology, blob filtering | `vision.py` | `YellowDetector` |
| the crop-mask-filter sequence | `vision.py` | `scan_yellow()` → a `YellowScan` |
| the area-weighted centroid | `vision.py` | `area_weighted_centroid()` |
| `LineTracker` | `control.py` | `PlausibilityGate` (renamed) |
| `LostLineController` | `control.py` | `SteeringController` (renamed) |
| `LostLineState` | `control.py` | unchanged |
| `LinePipeline` | `strategies.py` | `LineFollowStrategy`, over the shared parts |
| — | `strategies.py` | `FollowStrategy` base, `LaneFollowStrategy`, `LaneModel` |

Also added in step 3: `vision.cluster_lines()` and `vision.describe_blobs()`,
which exist only for lane following.

**Line following did not change behavior.** That is checked rather than assumed:
`scripts/replay_frames.py` recorded the exact per-frame steering and throttle
sequence *before* the refactor, and the same 78 frames still produce byte-identical
output afterwards.

```bash
python scripts/replay_frames.py --check tests/data/line_golden.json
# PASS: all 78 frames match the baseline exactly.
```

---

## How the two pipelines work

### Shared skeleton

Both strategies run the same sequence and differ in one step only:

```
frame (426x240, BGR)
  -> crop to the trapezoid ROI          vision.RoiGeometry
  -> blur, HSV threshold for yellow     vision.YellowDetector.raw_mask
  -> morphological open then close      vision.YellowDetector.clean
  -> reject blobs by size and shape     vision.YellowDetector.find_blobs
  ---------------- the only difference ----------------
  -> ONE target x                       LineFollowStrategy / LaneFollowStrategy
  -----------------------------------------------------
  -> reject implausible jumps           control.PlausibilityGate
  -> offset = (target - center) / half  control.offset_from_x
  -> proportional steering + throttle   control.SteeringController
```

Because everything after "one target x" is shared, a given target produces the
same steering in either mode. Switching modes changes *what the car aims at*,
never *how it steers*.

### Line following

The target is the **area-weighted centroid of every blob that passed the
filters**. Weighting by area means nearby tape counts for more than distant tape,
and a dashed or scuffed line still yields one steady target rather than flicking
between pieces.

This is deliberately *not* "the largest blob in frame". Only blobs that already
passed color, ROI, size and shape filtering are averaged. There is no code path
anywhere that picks a blob when the line has been lost.

### Lane following

Three yellow lines define two lanes:

```
    left boundary   |   divider (dashed)   |   right boundary
         lane LEFT           ^                    lane RIGHT
```

Two problems to solve, both in `LaneModel`:

**1. Which blob is which line?** The divider is discontinuous, so each dash is a
separate blob. `vision.cluster_lines()` first groups blobs that are stacked
vertically at about the same x into one line — so a dashed divider becomes one
line, not four. Each resulting line then gets matched to whichever line it was
nearest to last frame, which keeps identities stable through curves. With no
history to match against, the **fill ratio** decides: how much of a line's
vertical span is actually painted. A continuous boundary is near 1.0; the dashed
divider is well below `DIVIDER_MAX_FILL_RATIO`.

**2. Where is the lane center?** A three-tier ladder, most accurate first.

**TIER 1 — a bracketing pair.** Both lines around the chosen lane are visible, so
take their midpoint. `LEFT` uses (left boundary, divider); `RIGHT` uses
(divider, right boundary). **The LEFT/RIGHT toggle is literally which pair gets
used**, so the shift is exact rather than a fudge factor. If both outer
boundaries are visible but the divider dash is missing, the divider is inferred as
their midpoint — still tier 1.

**TIER 2 — single visible boundary.** On a sharp curve, or when the ROI clips one
side, only one line is usable. The target is placed one **measured** half-lane
width to the correct side of it, and the sign of that offset is what keeps lane
selection working. Crucially the width is *learned*: whenever a real boundary pair
is visible, the half-lane width is measured and smoothed into a running average.
So the fallback extrapolates from a width seen moments ago, not from a number
typed into a config file.

> *Tradeoff:* this assumes pixel lane width is locally constant — true for a fixed
> camera on flat ground, and degrading on tight curves where perspective
> compresses the lane. The degradation is bounded by tier 3.

**TIER 3 — hold heading.** If no line can be identified, or the width measurement
is older than `LANE_WIDTH_STALE_FRAMES`, the target is `None` and the car falls
into the same lost-line behavior as stage 1. Losing the lane beats inventing a
target.

`SINGLE_BOUNDARY_MODE` selects the tier-2 behavior: `"offset"` (default, as
above), `"divider"` (steer relative to the divider only — robust while the divider
is visible, but it is dashed and drops out often, which is why it is not the
default), or `"hold"` (skip tier 2 entirely; the most conservative choice).

---

## Foliage rejection

Pale-green shrubs at horizon level are the main thing that can fool a yellow
detector, and a previous version of this project drove into them: it tracked the
line fine until shrubs entered view, then at a spot where the tape briefly left
frame it locked onto the shrubs. Five layers now stand in the way, and a false
positive has to beat **all** of them.

**1. Color separation** — `vision.YellowDetector.raw_mask`
Yellow tape sits around hue 20–35; pale foliage green starts around 40. The hue
**ceiling** (`YELLOW_HSV_HIGH`, default 33) keeps them apart. Foliage is also
*pale* — low saturation — where tape is vividly saturated, so the saturation
**floor** (`YELLOW_HSV_LOW`, default 110) rejects it independently of hue. Two
different properties have to line up for foliage to get through.

**2. ROI geometry** — `vision.RoiGeometry`
A trapezoid over the lower frame. Foliage is above the ground plane, so simply
not looking at the horizon removes most of it at no computational cost. Narrowing
the far edge (`ROI_TOP_WIDTH_FRAC`) trims the left and right margins near the
horizon too.

**3. Size and shape** — `vision.YellowDetector.find_blobs`
Blobs below `MIN_CONTOUR_AREA_FRAC` are dropped. Survivors face two
**rotation-invariant** tests: *rect fill* (how fully the blob fills a tilted
bounding box) and *solidity* (how ragged its outline is versus its own convex
hull). Tape is smooth and solid; leaf clumps are notched and ragged.

> Rotation invariance is not incidental. An upright bounding box is mostly empty
> for a diagonal strip, so a plain area/bbox test would reject real tape angled
> across the frame — which happens on every curve, the worst possible moment to
> lose the line.

**4. Plausibility gate** — `control.PlausibilityGate`
The line cannot teleport. A target more than `MAX_CENTROID_JUMP_FRAC` of the frame
width from last frame's position is rejected, and a shrub appearing off to one
side is exactly that kind of jump.

The subtle part is letting go. A gate that never opens means one bad moment ends
the run. So after `PLAUSIBILITY_RESET_FRAMES` frames with nothing accepted the gate
opens to the whole frame — and to stop that opening from letting one flash of
green become the new "line", a fresh lock must repeat in roughly the same place
for `REACQUIRE_CONFIRM_FRAMES` frames before it is trusted.

**5. Hold heading, never chase** — `control.SteeringController`
The final backstop. When the target is lost, steering **freezes** at its last
value and throttle winds down: `COASTING` (hold heading, reduced throttle) →
`SLOWING` (hold heading, throttle ramping to zero) → `STOPPED`.

Note what the state machine does *not* contain: any state meaning "steer toward
whatever else is in frame". Losing the line can only lead to holding heading and
slowing down. Even if foliage beat defenses 1–4, coasting on the last heading
beats chasing a green blob.

This is covered by tests, including the exact past failure —
`test_foliage_never_becomes_the_steering_target` establishes a lock, removes the
tape leaving only shrubs, and asserts the steering **freezes** rather than
re-aiming.

---

## Running it

All commands run from your car directory (the one with `manage.py`), which should
be a copy of the `cv_control` template.

### Line following (stage 1)

In `myconfig.py`:

```python
CV_CONTROLLER_MODULE = "donkeycar.parts.lane_following.controller"
CV_CONTROLLER_CLASS = "LaneFollowingController"
START_MODE = "line"
```

```bash
python manage.py drive
```

Then open `http://<your-hostname>.local:8887`, the normal donkeycar driving page.
The car starts in **user** (manual) mode and does nothing autonomous until you
switch the mode selector there to **local**.

What to expect: on `local`, the car drives forward at `THROTTLE_FORWARD` and
steers to keep the yellow line centered. When the line leaves view it holds its
heading, coasts, slows and stops — it does not hunt around. The console logs each
state change (`tracking`, `coasting`, `slowing`, `stopped`).

### Lane following (stage 2) with the toggle page

In `myconfig.py`:

```python
CV_CONTROLLER_MODULE = "donkeycar.parts.lane_following.controller"
CV_CONTROLLER_CLASS = "LaneFollowingController"
LANE_WEB_ENABLE = True      # start the toggle page
LANE_WEB_PORT = 8891        # 8887 is taken by the donkeycar driving page
START_MODE = "lane"
START_LANE = "left"
```

```bash
python manage.py drive
```

Two pages are now served, and you need both:

| URL | What it does |
|---|---|
| `http://<hostname>.local:8887` | the stock donkeycar page — **engage autopilot here** (mode `local`) |
| `http://<hostname>.local:8891` | the toggle page — camera feed, MODE, LANE and DEBUG |

On the toggle page: **MODE** switches LINE ↔ LANE, **LANE** switches LEFT ↔ RIGHT
(greyed out in line mode, where it has no meaning), and the current mode and lane
are shown beneath the buttons. Both can be changed while driving; the strategy
resets on the change, so the car will not swerve.

### Debug mode

Either flip **DEBUG** on the toggle page, or set `DEBUG_OVERLAY = True` in
`myconfig.py` to start with it on.

The overlay is drawn on the camera feed:

| Marker | Meaning |
|---|---|
| green tint | pixels the **raw** color threshold accepted |
| white | pixels that survived cleanup **and** the shape filters — what steering actually uses |
| orange outline | the ROI trapezoid |
| red tick | frame center (steering neutral) |
| cyan dot + line | the steering target and its offset from center |
| orange verticals | identified lane boundaries (lane mode) |
| magenta vertical | the identified center divider (lane mode) |
| text | mode, lane, state, steering, throttle, blob count, mask area, and in lane mode which tier produced the target |

**Comparing the green tint against the white is the fastest diagnosis available.**
Lots of green with little white means the color threshold is too loose and the
shape filters are cleaning up after it — tighten the color instead.

Debug costs a few milliseconds per frame, so leave it off for fast laps.

### Verifying without the car

```bash
# the wiring: runs the real vehicle loop with MOCK camera/drivetrain, twice
python scripts/preflight_lane_following.py --controller LaneFollowingController --web

# the CV logic: foliage rejection, plausibility gate, lost-line states
pytest tests/test_lane_following.py tests/test_lane_following_web.py -v

# line following has not regressed
python scripts/replay_frames.py --check tests/data/line_golden.json
```

`preflight_lane_following.py` is the one to run after any change to the parts or
config. It asserts the CV part **does not execute** in user mode (so manual
driving stays safe) and **does execute** in local mode.

---

## Tuning at the track

Everything is in `donkeycar/parts/lane_following/params.py`, and any name there
can be overridden in `myconfig.py`. Anything you do not mention keeps its
default, so `myconfig.py` cannot go stale in a way that breaks launch.

### The parameters most likely to need changing

**When foliage leaks into the mask, in this order:**

1. **`YELLOW_HSV_HIGH`** — the hue ceiling (default `33`). **Lower it first.**
   Foliage green starts around hue 40, so anything above ~35 invites shrubs in.
   This is the single highest-leverage knob.
2. **`YELLOW_HSV_LOW`** — the saturation floor (default `110`, the middle value).
   **Raise it second.** Foliage is pale; tape is vivid. This rejects washed-out
   greens even when their hue overlaps.
3. **`ROI_TOP_FRAC`** — raise it (default `0.55`) to crop more horizon away. Free
   foliage rejection, at the cost of seeing less far ahead.

**For how the car drives:**

4. **`STEERING_KP`** — default `0.85`. Raise for sharper cornering; too high and
   the car weaves. Tune this before anything else in the controller.
5. **`THROTTLE_FORWARD`** — default `0.18`. Start low and raise it only once the
   steering is behaving.
6. **`HALF_LANE_WIDTH_FRAC`** (lane mode) — default `0.18`. Only the seed value,
   since it is re-measured live, but a bad seed hurts the first few seconds.
   Read it off a debug frame on a straight section.

### The tuning loop

```bash
# 1. grab a real frame from the camera, on the car
python scripts/oakd_color_check.py

# 2. pick HSV bounds interactively (drag a box over the tape)
python scripts/hsv_picker.py -f oakd_as_bgr.png

# 3. put the result in myconfig.py as YELLOW_HSV_LOW / YELLOW_HSV_HIGH
# 4. capture a sequence and check the effect off the car
python scripts/replay_frames.py --capture 40 --frames tests/data/line_frames
python scripts/replay_frames.py --frames tests/data/line_frames
```

Lighting changes everything about HSV thresholds, so expect to re-check them if
you move from overcast to bright sun.

---

## How mode and lane state reaches the steering output

Clicking **LANE** on the toggle page sends `POST /api/mode {"mode": "lane"}` to
the tornado server, running in its own thread. The handler calls
`PipelineState.set_mode()`, which takes a lock and writes one field. Nothing else
happens on that thread — it does not touch the car.

On the next vehicle-loop tick, `LaneFollowingController.run()` calls
`PipelineState.snapshot()` **once**, getting mode, lane and debug as one atomic
set (read one at a time, a click landing mid-read could hand the pipeline a new
mode with the old lane). Seeing that the snapshot differs from last frame's, the
controller calls `reset()` on the strategy it is about to run, clearing its last
offset, lost-frame counters, plausibility history and lane-width estimates — so
the new strategy starts clean instead of acting on a target that belonged to a
different reading of the scene. It then calls `strategy.process(frame, lane)`,
which returns steering and throttle. Those are written to `pilot/steering` and
`pilot/throttle`, and `DriveMode` forwards them to the drivetrain when the car is
in an autopilot mode. Total latency is one loop tick, about 50 ms at 20 Hz.

The state object is a process-wide singleton (`get_pipeline_state()`) because
donkeycar constructs the CV part from a config-named module path, so `manage.py`
never holds a reference it could pass to the web part. One vehicle pipeline per
process makes a singleton the right scope.

---

## Design notes

**Why not a PID for steering?** The lost-line behavior has to *freeze* the
output, and a PID's integral term fights that: it keeps winding up while the car
coasts blind, then snaps when the line returns. Proportional control with
exponential smoothing has no such state, so freezing it is exactly that. The
template still constructs a PID and hands it to the part; it is accepted and
ignored.

**Why is the camera part still the only DepthAI reader?** The toggle page gets its
frames as an ordinary part input from vehicle memory (`cv/image_array`, which the
CV part already outputs). There is no second device handle and no second queue
consumer, so there is nothing to contend over.

**How is frame tearing avoided?** The CV part builds every debug overlay into a
*fresh* array and never draws into one it has already handed over; the web server
only ever swaps a reference, under a lock. So a request encoding a frame to JPEG
holds an image nobody will modify underneath it. There is a test for this
(`test_concurrent_frame_writes_and_reads_are_safe`) that hammers the buffer from
both sides and decodes every result.

**Why does the web server test-bind its port on the main thread?** donkeycar's own
`WebFpv` calls `listen()` inside its background thread, so a port conflict kills
that thread silently and the page simply never appears. Test-binding in
`__init__` turns the same conflict into an immediate startup error that names the
port and says which config key to change.

---

## Failure modes

| Symptom | Cause | Behavior / fix |
|---|---|---|
| `RuntimeError: No DepthAI device` | camera unplugged, or udev rule missing | Loud failure at startup. See the udev rule at the top of `oak_d.py`. |
| `Could not open the OAK-D with any known full-FOV capture mode` | no candidate resolution worked | The error lists every combination tried and the depthai error for each. |
| Mask always empty, car coasts then stops | `CAMERA_COLOR_ORDER` is wrong, or the hue window misses the tape | Run `scripts/oakd_color_check.py`; then `hsv_picker.py`. |
| Shrubs in the mask | hue ceiling too high, saturation floor too low | Lower `YELLOW_HSV_HIGH`, raise `YELLOW_HSV_LOW`, raise `ROI_TOP_FRAC`. |
| Car weaves down straights | `STEERING_KP` too high, or too little smoothing | Lower `STEERING_KP`; raise `STEERING_SMOOTHING`. |
| Car cuts corners | `STEERING_KP` too low, or ROI looking too far ahead | Raise `STEERING_KP`; raise `ROI_TOP_FRAC`. |
| Loses the line on curves | ROI too tall/narrow, or shape gates too strict | Lower `ROI_TOP_FRAC`; raise `ROI_TOP_WIDTH_FRAC`; lower `MIN_RECT_FILL`. |
| Lane mode reports `width stale` a lot | rarely sees a full boundary pair | Raise `LANE_WIDTH_STALE_FRAMES`, or lower `ROI_TOP_FRAC` to see more track. |
| Divider read as a boundary | dashes merging after the close morphology | Lower `DIVIDER_MAX_FILL_RATIO` or `MORPH_CLOSE_KERNEL`. |
| Lane identities swap on curves | association window too wide | Lower `LANE_ASSOC_MAX_DIST_FRAC`. |
| Toggle page will not start | port already in use | The startup error names the port; change `LANE_WEB_PORT`. |
| `AttributeError: cfg has no attribute OAKD_RGB` | `OAKD_*` keys missing from config | Add them (see the camera section of `cfg_cv_control.py`). |
| Car will not move in `local` mode | `PassThroughController` still selected | Set `CV_CONTROLLER_CLASS = "LaneFollowingController"`. |

Throughout, the CV part runs with `run_condition="run_pilot"`, so **none of these
can stop you driving the car manually.**
