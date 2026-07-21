# Running CenterLineFollower on the Physical Car

Step-by-step guide to getting `CenterLineFollower` actually driving the
physical car. For the pipeline design, HSV/control-law theory, and the
full tunable-parameter reference, see `center_line_follower.md` in this
same directory — this file is the hands-on companion to it and is
self-contained (every command below is given in full).

## 0. One-time hardware and dependency setup

1. Plug the OAK-D into the Pi via USB-C.
2. Add the udev rule so the Pi can access the device without root (from
   `oak_d.py`'s own setup notes):
   ```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
   Skipping this shows up later as `RuntimeError: No DepthAI (Oak-D-Lite)
   device (camera) found!` even with the camera physically connected.
3. Install the DepthAI Python library (not installed automatically with
   donkeycar):
   ```bash
   pip install depthai
   ```

## 1. Create the car project

If you don't already have a car project using this Part:
```bash
donkey createcar --template cv_control --path ~/mycar
cd ~/mycar
```
(If you already have a `~/mycar`, just `cd` into it and skip to step 2.)

## 2. Configure myconfig.py

Open `~/mycar/myconfig.py` and add:
```python
# --- Camera ---
CAMERA_TYPE = "OAKD"
IMAGE_W = 426
IMAGE_H = 240
OAKD_RGB = True
OAKD_DEPTH = True   # leave True -- see note below

# --- CV controller selection ---
CV_CONTROLLER_MODULE = "donkeycar.parts.center_line_follower"
CV_CONTROLLER_CLASS = "CenterLineFollower"

# PID_P / PID_I / PID_D must stay defined even though CenterLineFollower
# ignores them -- cv_control.py constructs a PID from them unconditionally
# before the CV controller is even selected. The stock commented-out
# values already in myconfig.py are fine as-is; just don't delete them.

# --- Starter tunables (re-tune once you've captured a real frame, see step 6) ---
CENTER_LINE_COLOR_LOW = (75, 80, 40)
CENTER_LINE_COLOR_HIGH = (105, 255, 255)
CENTER_LINE_THROTTLE = 0.15   # start low; see step 7
OVERLAY_IMAGE = True
```
`OAKD_DEPTH = False` would save USB bandwidth since this Part doesn't use
depth, but a separate pre-existing bug in `oak_d.py`'s `_poll()` makes that
crash at runtime today (it fetches the depth queue unconditionally
whenever either stream is enabled) — leave it `True` for now.

## 3. First hardware smoke test (before a full drive test)

Confirm the camera itself works and produces genuinely RGB frames before
wiring it into a full drive test.

**If the Pi has a display attached (or you're on VNC/X11 forwarding):**
run `oak_d.py`'s own built-in self-test, which opens a live preview window:
```bash
python "$(python -c 'import donkeycar.parts.oak_d as m; print(m.__file__)')" --rgb
```
Press `q` or Escape to close it. If it opens and shows a live, correctly
colored picture, the camera and the preview-size fix are both working.

**If the Pi is headless (no display — the common case):** use this short
script instead. Save it as e.g. `~/mycar/oak_test.py`:
```python
import cv2
from donkeycar.parts.oak_d import OakD

cam = OakD(width=426, height=240, enable_rgb=True, enable_depth=True)
color_image, _depth_image = cam.run()
print("shape:", color_image.shape, "dtype:", color_image.dtype)

# color_image should now be genuinely RGB. cv2.imwrite always assumes its
# input is BGR, so convert before saving -- that way, if you view the
# saved file and the colors look CORRECT, the fix is confirmed working
# (genuinely RGB). If colors look swapped (red looks blue, etc.), it's
# still BGR and worth re-checking the oak_d.py setColorOrder(...) call.
cv2.imwrite("/tmp/oak_test.jpg", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
print("saved /tmp/oak_test.jpg -- copy it off the Pi and check the colors")
cam.shutdown()
```
```bash
python ~/mycar/oak_test.py
```
Then copy `/tmp/oak_test.jpg` off the Pi (`scp pi@<host>:/tmp/oak_test.jpg .`)
and open it. Point the camera at something with an obvious color (a red or
blue object) before running this so the check is easy to read.

## 4. Start the car

```bash
cd ~/mycar
python manage.py drive
```
Watch for a line like `You can now go to <your hostname>.local:8887 to
drive your car.` (the web UI listens on port 8887 by default). If you have
a physical joystick configured instead of using the web UI, use
`python manage.py drive --js` instead — joystick button mappings aren't
covered here since they depend on your specific controller.

## 5. Activate CenterLineFollower in the web UI

1. On another device on the same network, open
   `http://<hostname-or-ip>.local:8887/drive` in a browser (use the Pi's
   IP address if `.local` mDNS doesn't resolve).
2. Find the **Mode** dropdown. It has three options, and which one you
   pick matters a lot here:

   | Mode | Dropdown label | Steering | Throttle |
   |---|---|---|---|
   | `user` | (U)ser | you | you |
   | `local_angle` | Auto (S)teer | **CenterLineFollower** | you |
   | `local` | Full (A)uto | **CenterLineFollower** | **CenterLineFollower** |

   **`CenterLineFollower`'s constant throttle only actually drives the car
   in "Full Auto."** "Auto Steer" runs the same vision/control pipeline and
   computes steering from it, but throttle stays under your manual control
   — useful as a safer first check (see step 7), but it will NOT drive
   itself forward on its own.
3. You can also use keyboard shortcuts instead of the dropdown (click the
   page first so it has focus): `u` = User, `s` = Auto Steer, `a` = Full
   Auto, `m` = cycle through modes, `r` = start/stop recording.
4. While in Auto Steer or Full Auto, the live camera pane shows
   `CenterLineFollower`'s debug overlay (since `OVERLAY_IMAGE = True`): a
   blue box around the region of interest, a green tint wherever the color
   mask matched, a red crosshair on the tracked centroid, and
   `STATE`/`STEERING`/`THROTTLE` text in the corner. Use this to confirm
   the mask is actually tracking the tape — not just trusting the wheels —
   before relying on it.

## 6. Re-tune the color thresholds against a real frame

The starter `CENTER_LINE_COLOR_LOW`/`HIGH` values in step 2 were tuned for
description, not your actual tape/lighting — expect to redo this on the
real car.

1. Get a real frame to tune against: either the `/tmp/oak_test.jpg` from
   step 3, or record a few seconds live (press `r` in the web UI to start
   recording, drive/hold the car over the tape, press `r` again to stop)
   and grab the newest file from `~/mycar/data/<tub_name>/images/`.
2. Run the interactive picker against it:
   ```bash
   python scripts/hsv_picker.py --file /tmp/oak_test.jpg
   ```
   Click-drag a rectangle over the teal tape in the image to auto-sample
   its HSV range, or adjust the H/S/V trackbars directly. Press `p` to
   print the current low/high values to the terminal, `q` to print and
   quit, Escape to reset.
3. Copy the printed values into `myconfig.py`:
   ```python
   CENTER_LINE_COLOR_LOW = (h_low, s_low, v_low)
   CENTER_LINE_COLOR_HIGH = (h_high, s_high, v_high)
   ```
4. Restart `manage.py drive` and check the overlay (step 5.4) again —
   repeat until the green mask cleanly covers the tape and nothing else.

## 7. Recommended first-drive sequence

Putting steps 5 and 6 together into a safe order for the very first real
run:

1. `CENTER_LINE_THROTTLE` low (e.g. `0.15` or lower) in `myconfig.py`.
2. `python manage.py drive`, open the web UI.
3. Switch to **Auto Steer** (`s`). Manually drive forward slowly over the
   tape and confirm the overlay tracks it and steering visually looks
   correct — you're still in control of throttle here, so this is safe to
   try repeatedly.
4. Once steering looks right, switch to **Full Auto** (`a`) at the same
   low throttle. Be ready to switch back to `u` (User) instantly if
   anything looks wrong.
5. If it tracks well, raise `CENTER_LINE_THROTTLE` a little in
   `myconfig.py`, restart `manage.py drive`, and repeat from step 3. Don't
   jump straight to a high throttle value.
