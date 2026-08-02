# Donkeycar: a python self driving library


![Build Status](https://github.com/autorope/donkeycar/actions/workflows/python-package-conda.yml/badge.svg?branch=main)
![Lint Status](https://github.com/autorope/donkeycar/actions/workflows/superlinter.yml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/autorope/donkeycar)


[![All Contributors](https://img.shields.io/github/contributors/autorope/donkeycar)](#contributors-)
![Issues](https://img.shields.io/github/issues/autorope/donkeycar)
![Pull Requests](https://img.shields.io/github/issues-pr/autorope/donkeycar?)
![Forks](https://img.shields.io/github/forks/autorope/donkeycar)
![Stars](https://img.shields.io/github/stars/autorope/donkeycar)
![License](https://img.shields.io/github/license/autorope/donkeycar)

![Discord](https://img.shields.io/discord/662098530411741184.svg?logo=discord&colorB=7289DA)

Donkeycar is a minimalist and modular self driving library for Python. It is developed for hobbyists and students with a focus on allowing fast experimentation and easy community contributions.  It is being actively used at the high school and university level for learning and research.  It offers a [rich graphical interface](https://docs.donkeycar.com/utility/ui/) and includes a [simulator](https://docs.donkeycar.com/guide/deep_learning/simulator/) so you can experiment with self-driving even before you build a robot.

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Build instructions and Software documentation](http://docs.donkeycar.com)
* [Discord / Chat](https://discord.gg/PN6kFeA)

![donkeycar](https://github.com/autorope/donkeydocs/blob/master/docs/assets/build_hardware/donkey2.png)

### Use Donkeycar if you want to:
* Build a robot and teach it to drive itself.
* Experiment with [autopilots](https://docs.donkeycar.com/guide/train_autopilot/), gps, computer vision and neural networks.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com), including [online simulator races](https://docs.donkeycar.com/guide/deep_learning/virtual_race_league/) against competitors from around the world.
* Participate in a vibrant online community learning cutting edge techology and having fun doing it.

### What do you need to know before starting? (TL;DR nothing)
Donkeycar is designed to be the 'Hello World' of automomous driving; it is simple yet flexible and powerful.  No specific prequisite knowledge is required, but it helps if you have some knowledge of:
- [Python](https://docs.python.org/3.11/) programming.  You do not have to do any programming to use Donkeycar.  The file that you edit to configure your car, `myconfig.py`, is a Python file.  You mostly just uncomment the sections you want to change and edit them; you can avoid common mistakes if you know how Python [comments](https://www.w3schools.com/python/python_comments.asp) and [indentation](https://www.w3schools.com/python/python_syntax.asp) works.
- Raspberry Pi.  The Raspberry Pi is the preferred on-board computer for a Donkeycar.  It is helpful to have setup and used a Raspberry Pi, but it is not necessary.  The Donkeycar documentation describes how to install the software on a RaspberryPi OS, but the specifics of how to install the RaspberryPi OS using [Raspberry Pi Imager](https://www.raspberrypi.com/software/) and how to configure the Raspberry Pi using [raspi-config](https://www.raspberrypi.com/documentation/computers/configuration.html) is left to the Raspberry Pi documentation, which is extensive and quite good. I would recommend setting up your Raspberry Pi using the Raspberry Pi documentation and then play with it a little; use the browser to visit websites and watch YouTube videos, like this one taken at the [very first outdoor race](https://youtu.be/tjWmrCIKgnE) for a Donkeycar.  Use a text editor to write and save a file.  Open a terminal and learn how to navigate the file system (see below). If you are comfortable with the Raspberry Pi then you won't have to learn it and Donkeycar at the same time.
- The Linux [command line shell](https://magpi.raspberrypi.com/articles/terminal-help).  The command line shell is also often called the terminal.  You will type commands into the terminal to install and start the Donkeycar software.  The Donkeycar documentation describes how this works.  It is also helpful to know how navigate the file system and how to list, copy and delete files and directories/folders. You may also access your car [remotely](https://www.raspberrypi.com/documentation/computers/remote-access.html); so you will want to know how to enable and connect WIFI and how to enable and start an [SSH](https://www.raspberrypi.com/documentation/computers/remote-access.html#ssh) terminal or [VNC](https://www.raspberrypi.com/documentation/computers/remote-access.html#vnc) session from your host computer to get a command line on your car.

## Get driving.
After [building a Donkeycar](https://docs.donkeycar.com/guide/build_hardware/) and [installing](https://docs.donkeycar.com/guide/install_software/) the Donkeycar software you can choose your autopilot [template](https://docs.donkeycar.com/guide/create_application/) and [calibrate](https://docs.donkeycar.com/guide/calibrate/) your car and [get driving](https://docs.donkeycar.com/guide/get_driving/)!

## Modify your car's behavior.
Donkeycar includes a number of pre-built [templates](https://docs.donkeycar.com/guide/create_application/) that make it easy to get started by just changing configuration. The pre-built templates are all you may ever need, but if you want to go farther you can change a template or make your own. A Donkeycar template is organized as a pipeline of software [parts](https://docs.donkeycar.com/parts/about/) that run in order on each pass through the vehicle loop, reading inputs and writing outputs to the vehicle's software memory as they run.  A typical car has a parts that:
- Get images from a camera. Donkeycar supports lots of different kinds of [cameras](https://docs.donkeycar.com/parts/cameras/), including 3D cameras and [lidar](https://docs.donkeycar.com/parts/lidar/).
- Get position readings from a GPS receiver.
- Get steering and throttle inputs from a [game controller](https://docs.donkeycar.com/parts/controllers/) or RC controller.  Donkeycar support PS3, PS4, XBox, WiiU, Nimbus and Logitech Bluetooth game controllers and any game controller that works with RaspberryPi.  Donkeycar also implements a WebUI that allows any browser compatible game controller to be connected and also offers an onscreen touch controller that works with phones.
- Control the car's drivetrain [motors](https://docs.donkeycar.com/parts/actuators/) for acceleration and steering. Donkeycar supports various drivetrains including the ESC/Steering-servo configuration that is common to most RC cars and Differential Drive configurations.
- Save telemetry [data](https://docs.donkeycar.com/parts/stores/) such as camera images, steering and throttle inputs, lidar data, etc.
- Drive the car on autopilot.  Donkey supports three kinds of [autopilots](https://docs.donkeycar.com/guide/train_autopilot/); a [deep-learning](https://docs.donkeycar.com/guide/deep_learning/train_autopilot/) autopilot, a [gps autopilot](https://docs.donkeycar.com/guide/path_follow/path_follow/) and a [computer vision](https://docs.donkeycar.com/guide/computer_vision/computer_vision/) autopilot.  The Deep Learning autopilot supports Tensorflow, Tensorflow Lite, and Pytorch and many model [architectures](https://docs.donkeycar.com/parts/keras/).

If there isn't a Donkeycar part that does what you want then write your own [part](https://docs.donkeycar.com/parts/about/#parts) and add it to a vehicle [template](https://docs.donkeycar.com/parts/about/).

```python
#Define a vehicle to take and record pictures 10 times per second.

import time
from donkeycar import Vehicle
from donkeycar.parts.cv import CvCam
from donkeycar.parts.tub_v2 import TubWriter
V = Vehicle()

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

#Add a camera part
cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
V.add(cam, outputs=['image'], threaded=True)

#warmup camera
while cam.run() is None:
    time.sleep(1)

#add tub part to record images
tub = TubWriter(path='./dat', inputs=['image'], types=['image_array'])
V.add(tub, inputs=['image'], outputs=['num_records'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```

See [home page](http://donkeycar.com), [docs](http://docs.donkeycar.com)
or join the [Discord server](http://www.donkeycar.com/community.html) to learn more.

## Line following and lane following (this branch)

Classical-CV line following and lane following for a DonkeyCar with a Luxonis
OAK-D camera. No neural nets, no training, no simulator — everything is tuned
through named config constants at the track.

**Start here: [docs/lane_following.md](docs/lane_following.md)** for the concepts,
the camera patch, run instructions and the tuning guide.

### Quick reference

```bash
# Re-apply the OAK-D full-FOV camera patch (needed after any environment rebuild)
~/env/bin/python scripts/patch_oak_d.py

# Confirm the camera streams 426x240 with the recovered field of view,
# and settle whether frames are BGR or RGB
python scripts/oakd_color_check.py

# Prove manage.py still launches and stays manually drivable, without hardware
python scripts/preflight_lane_following.py --controller LaneFollowingController --web

# Check the CV logic (foliage rejection, plausibility gate, lost-line states)
pytest tests/test_lane_following.py tests/test_lane_following_web.py

# Confirm line following has not regressed
python scripts/replay_frames.py --check tests/data/line_golden.json

# Drive
python manage.py drive
#   http://<hostname>.local:8887  donkeycar page - engage autopilot ("local") here
#   http://<hostname>.local:8891  toggle page - MODE, LANE, DEBUG  (LANE_WEB_ENABLE=True)
```

### If foliage leaks into the yellow mask

Tune in this order (all in `donkeycar/parts/lane_following/params.py`, overridable
in `myconfig.py`):

1. **lower** `YELLOW_HSV_HIGH`'s hue ceiling (default `33` — foliage green starts ~40)
2. **raise** `YELLOW_HSV_LOW`'s saturation floor (default `110` — foliage is pale)
3. **raise** `ROI_TOP_FRAC` (default `0.55`) to crop more horizon away

### Layout

| Path | Purpose |
|---|---|
| `donkeycar/parts/lane_following/params.py` | every tunable, in one labeled block |
| `donkeycar/parts/lane_following/vision.py` | ROI, HSV masking, blob filtering (foliage defenses 1–3) |
| `donkeycar/parts/lane_following/control.py` | plausibility gate, steering, hold-heading (defenses 4–5) |
| `donkeycar/parts/lane_following/strategies.py` | line following, lane following, lane model |
| `donkeycar/parts/lane_following/controller.py` | the donkeycar Part and mode switching |
| `donkeycar/parts/lane_following/web.py` | the mode/lane toggle page |
| `donkeycar/parts/lane_following/overlay.py` | the debug overlay |
| `donkeycar/parts/line_follower.py` | centroid-tracking line follower (steering-loop rewrite, see Development process below) |
| `donkeycar/parts/oak_d.py` | patched OAK-D camera part (full-FOV `isp` stream, auto-exposure) |

---

# DSC190 Final Project — Team 10

**UCSD DSC190, Summer Session I 2026 — Track 2: Agentic Development, New DonkeyCar Capabilities**

**Team:** Leo Okdemir (HDSI) · Talal Jeddawi (CSE)

This is our team's final project for Track 2 of the DonkeyCar course: using an
AI coding agent (Claude) to iteratively build new autonomous behaviors on a
physical DonkeyCar, following a `mission → agent writes code → test on the
track → feedback → agent revises` loop. Everything above this line is the
line-following / lane-following project itself, built on top of the
[autorope/donkeycar](https://github.com/autorope/donkeycar) framework
(original project README further up).

## Mission progress

| Mission stage | Status | Notes |
|---|---|---|
| 1. Line following | ✅ Working | Follows a single strip of yellow tape; centroid tracking with a "hold last heading" fallback when the line is briefly lost. |
| 2. Lane following | ✅ Working | Upgrade of stage 1 — drives a chosen lane between two yellow boundaries and a discontinuous center divider; switch modes live from a web toggle page. |
| 3. Two-way road navigation | ⏳ Not reached | We ran out of runway before starting oncoming-robot detection/avoidance. Our `half-lane-following` branch (drive-in-your-half-of-the-road logic, described below) was exploratory groundwork toward this stage but was superseded by the lane-following approach documented above. |

## Development process (the agentic loop)

Per the assignment, we used Claude as a coding agent throughout: give it a
mission, let it write the CV/control code, test on the car, and feed back what
broke. We restarted the approach more than once before landing on what's in
this branch. Rough timeline, in order:

1. **`center_line_follower.py`, first pass** — initial classical-CV line
   follower. Worked in principle but wasn't robust to real track lighting.
2. **Half-lane following (`half-lane-following` branch)** — pivoted to having
   the car hug one half of the road (LEFT/RIGHT of a boundary edge), aimed at
   eventually supporting two-way traffic. Each track test surfaced a new false
   positive the agent then fixed: BGR/RGB channel mixup, shadows read as fake
   edges, gravel/pebbles read as fake edges, and daytime lighting breaking
   detection entirely. This branch got the detection robust but we decided the
   boundary-hugging strategy wasn't the right foundation for lane following.
3. **Restart on `LL-Following`** — rebuilt line following from scratch with a
   cleaner architecture (`donkeycar/parts/lane_following/`). First real-track
   test found the OAK-D camera part was hard-cropping away the near-ground
   strip the line follower needed on turns ("Oakd fix") — the agent patched
   `oak_d.py` to stream the full-FOV `isp` output instead of the cropped
   `video` output.
4. **`LL-Following-2` (this branch)** — feedback-driven hardening of the
   above: an OAK-D validation step that opens the camera *and* pulls a real
   frame (an earlier check only confirmed the device opened, which had been
   masking failures), optional auto-exposure compensation, the lane-following
   mode plus its web toggle page, and finally a steering-loop rewrite
   (centroid tracking, centered target) after track testing showed the car
   drifting off-center.

## Acknowledgments

Built on top of [autorope/donkeycar](https://github.com/autorope/donkeycar),
an open-source self-driving library for Python — see the original project
README above, or [docs.donkeycar.com](http://docs.donkeycar.com), for
background on the Vehicle/Part/template architecture we built these parts
against.
