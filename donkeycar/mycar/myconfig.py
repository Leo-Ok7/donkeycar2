# Simple centre-line follower: one horizontal scan slice, centroid of the
# yellow pixels, PID to keep that centroid at the middle of the frame.
CV_CONTROLLER_MODULE = "donkeycar.parts.line_follower"
CV_CONTROLLER_CLASS = "LineFollower"

# --- LineFollower tuning, measured on real recorded frames from this track ---
# Scan slice. The default SCAN_Y=100 samples rows 100-120, which on this
# camera is the far field / horizon, not the ground the car is about to drive
# over. Measured average yellow pixels per frame over 60 real tape frames:
#     rows 100-120 : 108.8   <- old default
#     rows 140-160 : 446.7   <- 4x more tape
#     rows 160-180 :  50.8
# BOTTOM HALF of the frame: rows 120-240 of a 240px image. A tall slice (not
# a 20px sliver) means the dashed line is intersected somewhere in the band
# almost all the time, instead of only when a dash happens to line up with one
# thin row.
# Look further ahead. Rows 120-240 is the near ground, effectively the
# bumper: at 20Hz each correction is held 50ms, so at speed the car has
# already driven past what it measured and every correction lands late,
# overshoots, and gets reversed on the next frame. Rows 110-215 keeps most of
# the near ground but adds ~10 rows of look-ahead. Push SCAN_Y lower (back
# toward 120) if it starts losing the line on tight corners, where the tape
# leaves the upper part of the frame first.
SCAN_Y = 180
SCAN_HEIGHT = 60

# Ignore this many pixels down EACH side of the scan band. The gravel bed,
# kerb and dry grass live at the frame edges and match the tape's hue closely
# enough that no colour threshold separates them, while the line being
# followed stays near the middle. 60px each side leaves the central 306 of
# 426 columns (x 60..366). Reduce if the line is genuinely leaving that window
# on sharp corners; raise if edge clutter still gets in.

# Yellow band. The default (0,50,50)..(50,255,255) spans hue 0-50 -- red
# through yellow-green -- at only S>=50, so sunlit concrete and the tan gravel
# bed both fall inside it. Measured on the scan slice across 234 tape frames
# and 2333 no-tape frames:
#     S>=50  -> false-positive 30.9%
#     S>=70  -> false-positive  8.9%
#     S>=90  -> false-positive  0.0%   <- chosen
#     S>=110 -> tape starts dropping out
# Saturation floor 60 -> 40. Some of the tape on this track is PALE yellow,
# and 60 clipped it: on a real pale-dash frame the mask caught only a 152px
# sliver of the dash while the rest sat unmatched, versus 830px at a lower
# floor. Measured over 1500 recent frames, detection 55.5% -> 61.0%, with no
# change on the wall/no-tape reference frames. Kept at 40 rather than 30
# because bare grey pavement piles up below ~20 and rises as the floor drops.
# WORN TAPE. Some dashes on this track are faded/washed out, and their colour
# is far weaker than fresh tape -- measured directly on a live frame, a worn
# dash had saturation p50=19 (p90=33), so an S>=40 floor kept only 4.9% of it.
# That is the tape that was being missed.
#
# Saturation alone cannot go that low safely: adjacent pavement runs S p90=11
# but p99=32, which overlaps the worn tape. What separates them is BRIGHTNESS
# -- the worn tape is still markedly lighter than the asphalt:
#     worn dash : V p10=129  p50=176  p90=203
#     pavement  : V p10=126  p50=142  p90=155
# On that frame, S>=18 with V>=130 yields 54x more dash pixels than pavement
# pixels, where S>=18 alone yields only 7.6x.
# Measured over 1500 recent frames: detection 60.6% -> 68.3%, and the no-tape
# reference frames stay clean.
# V floor set to 155, and that floor is doing MORE than rejecting pavement --
# it is what separates the tape from the WHITE BOUNDARY LINE. Saturation
# cannot: measured on a live frame the white line and the worn dash have the
# SAME saturation (both S p50=26), because below about S=30 hue is mostly
# noise and grey/white pixels scatter randomly into the yellow hue range.
# Brightness does separate them here -- the sunlit dash is markedly lighter
# than the shaded boundary line:
#     worn dash  : V p50=193
#     white line : V p50=144
#     V>=130 -> dash 169px vs white 78px  (2.2x -- white line leaks in and
#               drags the centroid toward the track edge)
#     V>=155 -> dash ~154px vs white ~5px (~30x)
# Lower toward 130 if tape in deep shade is missed, but expect the white line
# to start competing again; raise if it does.
# REVERTED to S>=40, V>=80. I had dropped this to (15,18,155) to catch worn
# tape, choosing the V floor to separate the white boundary line on ONE live
# frame. Across 1200 recorded frames that was clearly worse -- it rejects
# genuine tape wherever the tape is not brightly lit:
#     S90 V80  -> detection 35.9%
#     S60 V80  -> 56.9%
#     S40 V80  -> 65.4%   <- best, and what is set
#     S18 V155 -> 47.2%   <- the overfit value, worse than S40
# Worn tape is still a real problem, but the fix cannot be a hard brightness
# floor tuned on a single frame.
COLOR_THRESHOLD_LOW = (15, 40, 80)
COLOR_THRESHOLD_HIGH = (40, 255, 255)

# Fraction of sampled pixels that must be yellow before steering at all.
# This now genuinely gates (it used to be compared against a raw 0-255 sum, so
# a single yellow pixel passed it); 0.010 = 1% of the 426x20 slice.
# 0.002 = 0.2% of the sampled pixels. This MUST scale with SCAN_HEIGHT: the
# same dash is a far smaller fraction of a 120-row slice than of a 20-row one.
# Measured live on the bottom half with the line in view, only the nearest
# dash matches and that is 0.40% of the band -- so a 1.0% gate suppressed
# steering entirely even though the detection was correct.
CONFIDENCE_THRESHOLD = 0.002

# None => the image centre (213 of 426), so "centred" means centred in the
# camera frame. Set a pixel value to deliberately ride offset from the line.
SCAN_X_MARGIN = 60

# --- Steering damping / speed compensation -------------------------------
# Kd was -0.0001, only 1% of Kp -- essentially no damping, so the loop was
# near-pure-proportional and oscillated once (gain x speed) got high enough.
# The D term reacts to how fast the error is CHANGING, so it starts easing off
# before the car reaches centre instead of driving into an overshoot.
# Raise further if it still weaves; lower if it twitches on mask noise.
# Kp lowered -0.01 -> -0.005. Measured over 1500 real frames with a correct
# 50ms timestep, the old gain left steering PINNED AT FULL LOCK 53.5% of the
# time -- that is the "wavy" feel: not gentle oscillation but slamming between
# left and right lock. At -0.005 saturation drops to 10.6% while mean |steer|
# stays 0.569, so it still has real authority:
#     Kp        saturated   mean|steer|   reversals
#     -0.010      53.5%        0.784         2.7%
#     -0.006      35.5%        0.636         3.4%
#     -0.005      10.6%        0.569         4.3%   <- chosen
#     -0.004       8.9%        0.497         4.4%
# (Reversals rise slightly because at high gain the signal sits pinned at one
# extreme rather than crossing zero -- fewer sign changes, far worse driving.)
PID_P = -0.005
# Kd stays small. Raising it to -0.006 made things WORSE (reversals 2.7 -> 4.7%,
# jump size doubled) because the derivative amplifies the frame-to-frame
# centroid hops from dashes entering and leaving the scan band.
PID_D = -0.002
# Hold the LATERAL response constant across the speed range. A steering value
# moves the car sideways by (steering x speed x time), so gains tuned at
# THROTTLE_INITIAL=0.15 are ~2x too strong at THROTTLE_MAX=0.3. This divides
# the PID output by the throttle ratio, and only ever softens (never
# amplifies), so one set of gains works at every speed.
STEERING_SPEED_SCALING = True
STEERING_BASELINE_THROTTLE = 0.15

# Exponential smoothing of the measured line position (1.0 = raw, no filter).
# The centre line is DASHED and the scan band holds more than one dash, so
# dashes entering/leaving the band shift the centroid even when the car is
# going straight. Measured on real consecutive frames: median jump 4.8px but
# p90 25px and worst 146px -- and a car cannot move 146px sideways in one
# 50ms frame, so those are artefacts, each of which drove a real steering
# correction.
# Effect on steering activity over 3000 consecutive frames:
#     alpha 1.00 (raw) -> |delta| p90 14.31, direction reversals 8.9%
#     alpha 0.50       -> 6.63, 5.8%
#     alpha 0.35       -> 4.41, 4.8%   <- chosen
#     alpha 0.25       -> 2.87, 3.7%   (smoother, but ~200ms lag)
# Lower = calmer but slower to react on corners; raise toward 1.0 if it starts
# cutting corners or reacting late.
LINE_SMOOTHING_ALPHA = 0.35

TARGET_PIXEL = None
# Deadband: no steering correction while the line is within this many pixels
# of centre, so the car does not saw back and forth once it is lined up.
# Widened 10 -> 20px. Halves the residual twitch (reversals 4.3% -> 3.3%) at
# no cost to tracking, since 20px is under 5% of the frame width.
TARGET_THRESHOLD = 20
START_MODE = "line"

# --- Seeing the CV masks in the web UI -------------------------------------
# All three are needed; each fixes a different link in the chain.
#
DEBUG_OVERLAY = True         # 1. DRAW it. Without this the CV part returns the
                             #    plain frame and there is nothing to show.
                             #    Toggleable live from the page on 8891.
OVERLAY_IMAGE = True         # 2. SHOW it on 8887 in AUTOPILOT mode. False makes
                             #    UserPilotCondition show the raw camera instead.
# OFF: this preview comes from the lane_following package and draws with
# lane_following's OWN roi/colour params -- but the active controller is
# LineFollower, which has its own SCAN_Y and COLOR_THRESHOLD. So in manual it
# was reporting "blobs=0 area=0.00%" about a pipeline that is not driving,
# which is worse than showing nothing. Switch to `local` to see the real
# LineFollower overlay. Set back to True only if LaneFollowingController is
# the active controller again.
CV_PREVIEW_IN_MANUAL = False # 3. SHOW it on 8887 in MANUAL mode too, which is
                             #    the mode the car starts in. The autopilot part
                             #    is gated on run_pilot and does not run by hand,
                             #    so a display-only part fills in; see
                             #    donkeycar/parts/lane_following/preview.py.
                             #    It has no steering output and cannot move the car.

#
# CAMERA: Luxonis OAK-D (USB / DepthAI), not the Pi CSI camera.
#
# config.py defaults to CAMERA_TYPE = "PICAM", which sends add_camera() down
# get_camera() -> Picamera2(). With no CSI camera attached,
# Picamera2.global_camera_info() returns [] and indexing it raises IndexError.
# "OAKD" is handled directly in add_camera() (templates/complete.py), so it
# never reaches get_camera() at all.
#
CAMERA_TYPE = "OAKD"

# Frame size handed to OakD(width=..., height=...). Must match what the
# lane-following ROI fractions assume: 426x240.
IMAGE_W = 426
IMAGE_H = 240
# Channel count (3 = colour). Unrelated to OAKD_DEPTH below, which is stereo
# depth sensing.
IMAGE_DEPTH = 3
# Colour order is NOT configurable and does not need to be. depthai delivers
# BGR, and OakD._to_rgb() converts it once in the camera part, so everything
# from 'cam/image_array' onward -- the tub, both web feeds, training, and the
# lane-following CV -- is RGB. Do not add a BGR2RGB part here; that would
# convert it a second time and swap the colours right back.
# Verify on the car with: python scripts/oakd_color_check.py

# add_camera() reads all three of these when CAMERA_TYPE == "OAKD"; config.py
# does not define them, so they must be set here or cfg lookup raises
# AttributeError.
OAKD_RGB = True     # colour stream -> 'cam/image_array'
OAKD_DEPTH = False  # stereo depth -> 'cam/depth_array'; unused by line
                    # following, and skipping it saves USB bandwidth and CPU
                    # on the Pi. OakD._poll() only reads a queue when its
                    # feature is enabled, so this is safe to turn off.
OAKD_ID = None      # serial number, or None to auto-detect the only device

# ---------------------------------------------------------------------------
# THIS CAR'S HARDWARE (ucsdrobocar-DSC-T10)
# Without these, config.py's defaults win -- DRIVE_TRAIN_TYPE would be
# "PWM_STEERING_THROTTLE" (no VESC at all) and CONTROLLER_TYPE would be 'xbox',
# so neither the drivetrain nor the gamepad would work.
# ---------------------------------------------------------------------------
DRIVE_TRAIN_TYPE = "VESC"
# Stable by-id path: /dev/ttyACM* renumbers depending on whether the VESC or
# the OAK-D enumerates first, so the raw device name is not reliable here.
VESC_SERIAL_PORT = "/dev/serial/by-id/usb-STMicroelectronics_ChibiOS_RT_Virtual_COM_Port_304-if00"
VESC_MAX_SPEED_PERCENT = .35
VESC_HAS_SENSOR = True
VESC_START_HEARTBEAT = True
VESC_BAUDRATE = 115200
VESC_TIMEOUT = 0.05
VESC_STEERING_SCALE = 0.5    # maps angle -1..1 -> servo 0..1
VESC_STEERING_OFFSET = 0.5   # (angle * scale) + offset

# Logitech F710 gamepad. CONTROLLER_TYPE must be 'F710' so the
# LogitechJoystick axis map is used (left_stick_horz -> steering,
# right_stick_vert -> throttle); the default 'xbox' map does not match.
USE_JOYSTICK_AS_DEFAULT = True
CONTROLLER_TYPE = 'F710'
JOYSTICK_DEVICE_FILE = "/dev/input/js0"
JOYSTICK_MAX_THROTTLE = 0.35
JOYSTICK_STEERING_SCALE = 0.75
JOYSTICK_THROTTLE_DIR = -1.0   # -1.0 so pushing the stick up drives forward
JOYSTICK_DEADZONE = 0.01
AUTO_RECORD_ON_THROTTLE = True
USE_NETWORKED_JS = False
NETWORK_JS_SERVER_IP = None
# """
# My CAR CONFIG

# This file is read by your car application's manage.py script to change the car
# performance

# If desired, all config overrides can be specified here. 
# The update operation will not touch this file.
# """

# import os
# 
# 
# import os
# 
# #
# # FILE PATHS
# #
# CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = os.path.join(CAR_PATH, 'data')
# 
# 
# #
# # VEHICLE loop
# #
# DRIVE_LOOP_HZ = 20      # the vehicle loop will pause if faster than this speed.
# MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.
# 
# 
# #
# # CAMERA configuration
# #
# CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|D435|OAKD|MOCK|IMAGE_LIST)
# IMAGE_W = 320
# IMAGE_H = 240
# IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
# CAMERA_FRAMERATE = DRIVE_LOOP_HZ
# CAMERA_VFLIP = False
# CAMERA_HFLIP = False
# CAMERA_INDEX = 0  # used for 'WEBCAM' and 'CVCAM' when there is more than one camera connected
# # For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
# CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)
# BGR2RGB = False  # true to convert from BRG format to RGB format; requires opencv
# 
# # For IMAGE_LIST camera
# PATH_MASK = "~/mycar/data/tub_1_20-03-12/*.jpg"
# 
# 
# #
# # PCA9685, over rides only if needed, ie. TX2..
# #
# PCA9685_I2C_ADDR = 0x40     #I2C address, use i2cdetect to validate this number
# PCA9685_I2C_BUSNUM = None   #None will auto detect, which is fine on the pi. But other platforms should specify the bus num.
# 
# 
# #
# # SSD1306_128_32
# #
# USE_SSD1306_128_32 = False    # Enable the SSD_1306 OLED Display
# SSD1306_128_32_I2C_ROTATION = 0 # 0 = text is right-side up, 1 = rotated 90 degrees clockwise, 2 = 180 degrees (flipped), 3 = 270 degrees
# SSD1306_RESOLUTION = 1 # 1 = 128x32; 2 = 128x64
# 
# 
# #
# # MEASURED ROBOT PROPERTIES
# #
# AXLE_LENGTH = 0.03     # length of axle; distance between left and right wheels in meters
# WHEEL_BASE = 0.1       # distance between front and back wheels in meters
# WHEEL_RADIUS = 0.0315  # radius of wheel in meters
# MIN_SPEED = 0.1        # minimum speed in meters per second; speed below which car stalls
# MAX_SPEED = 3.0        # maximum speed in meters per second; speed at maximum throttle (1.0)
# MIN_THROTTLE = 0.1     # throttle (0 to 1.0) that corresponds to MIN_SPEED, throttle below which car stalls
# MAX_STEERING_ANGLE = 3.141592653589793 / 4  # for car-like robot; maximum steering angle in radians (corresponding to tire angle at steering == -1)
# 
# 
# #
# # DRIVE_TRAIN_TYPE
# # These options specify which chasis and motor setup you are using.
# # See Actuators documentation https://docs.donkeycar.com/parts/actuators/
# # for a detailed explanation of each drive train type and it's configuration.
# # Choose one of the following and then update the related configuration section:
# #
# # "PWM_STEERING_THROTTLE" uses two PWM output pins to control a steering servo and an ESC, as in a standard RC car.
# # "MM1" Robo HAT MM1 board
# # "SERVO_HBRIDGE_2PIN" Servo for steering and HBridge motor driver in 2pin mode for motor
# # "SERVO_HBRIDGE_3PIN" Servo for steering and HBridge motor driver in 3pin mode for motor
# # "DC_STEER_THROTTLE" uses HBridge pwm to control one steering dc motor, and one drive wheel motor
# # "DC_TWO_WHEEL" uses HBridge in 2-pin mode to control two drive motors, one on the left, and one on the right.
# # "DC_TWO_WHEEL_L298N" using HBridge in 3-pin mode to control two drive motors, one of the left and one on the right.
# # "MOCK" no drive train.  This can be used to test other features in a test rig.
# # (deprecated) "SERVO_HBRIDGE_PWM" use ServoBlaster to output pwm control from the PiZero directly to control steering,
# #                                  and HBridge for a drive motor.
# # (deprecated) "PIGPIO_PWM" uses Raspberrys internal PWM
# # (deprecated) "I2C_SERVO" uses PCA9685 servo controller to control a steering servo and an ESC, as in a standard RC car
# #
# DRIVE_TRAIN_TYPE = "PWM_STEERING_THROTTLE"
# 
# #
# # PWM_STEERING_THROTTLE drivetrain configuration
# #
# # Drive train for RC car with a steering servo and ESC.
# # Uses a PwmPin for steering (servo) and a second PwmPin for throttle (ESC)
# # Base PWM Frequence is presumed to be 60hz; use PWM_xxxx_SCALE to adjust pulse with for non-standard PWM frequencies
# #
# PWM_STEERING_THROTTLE = {
#     "PWM_STEERING_PIN": "PCA9685.1:40.1",   # PWM output pin for steering servo
#     "PWM_STEERING_SCALE": 1.0,              # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,         # True if hardware requires an inverted PWM pulse
#     "PWM_THROTTLE_PIN": "PCA9685.1:40.0",   # PWM output pin for ESC
#     "PWM_THROTTLE_SCALE": 1.0,              # used to compensate for PWM frequence differences from 60hz; NOT for increasing/limiting speed
#     "PWM_THROTTLE_INVERTED": False,         # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,               #pwm value for full left steering
#     "STEERING_RIGHT_PWM": 290,              #pwm value for full right steering
#     "THROTTLE_FORWARD_PWM": 500,            #pwm value for max forward throttle
#     "THROTTLE_STOPPED_PWM": 370,            #pwm value for no movement
#     "THROTTLE_REVERSE_PWM": 220,            #pwm value for max reverse throttle
# }
# 
# #
# # I2C_SERVO (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_CHANNEL = 1            #(deprecated) channel on the 9685 pwm board 0-15
# STEERING_LEFT_PWM = 460         #pwm value for full left steering
# STEERING_RIGHT_PWM = 290        #pwm value for full right steering
# THROTTLE_CHANNEL = 0            #(deprecated) channel on the 9685 pwm board 0-15
# THROTTLE_FORWARD_PWM = 500      #pwm value for max forward throttle
# THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
# THROTTLE_REVERSE_PWM = 220      #pwm value for max reverse throttle
# 
# #
# # PIGPIO_PWM (deprecated in favor of PWM_STEERING_THROTTLE)
# #
# STEERING_PWM_PIN = 13           #(deprecated) Pin numbering according to Broadcom numbers
# STEERING_PWM_FREQ = 50          #Frequency for PWM
# STEERING_PWM_INVERTED = False   #If PWM needs to be inverted
# THROTTLE_PWM_PIN = 18           #(deprecated) Pin numbering according to Broadcom numbers
# THROTTLE_PWM_FREQ = 50          #Frequency for PWM
# THROTTLE_PWM_INVERTED = False   #If PWM needs to be inverted
# 
# #
# # SERVO_HBRIDGE_2PIN drivetrain configuration
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_2PIN = {
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # provides forward duty cycle to motor
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # provides reverse duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",       # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
# 
# #
# # SERVO_HBRIDGE_3PIN drivetrain configuration
# # - configures a steering servo and an HBridge in 3pin mode (2 ttl pins, 1 pwm pin)
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by three pins,
# #   one ttl output for forward, one ttl output
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# SERVO_HBRIDGE_3PIN = {
#     "FWD_PIN": "RPI_GPIO.BOARD.18",   # ttl pin, high enables motor forward
#     "BWD_PIN": "RPI_GPIO.BOARD.16",   # ttl pin, high enables motor reverse
#     "DUTY_PIN": "RPI_GPIO.BOARD.35",  # provides duty cycle to motor
#     "PWM_STEERING_PIN": "RPI_GPIO.BOARD.33",   # provides servo pulse to steering servo
#     "PWM_STEERING_SCALE": 1.0,        # used to compensate for PWM frequency differents from 60hz; NOT for adjusting steering range
#     "PWM_STEERING_INVERTED": False,   # True if hardware requires an inverted PWM pulse
#     "STEERING_LEFT_PWM": 460,         # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
#     "STEERING_RIGHT_PWM": 290,        # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# }
# 
# #
# # DRIVETRAIN_TYPE == "SERVO_HBRIDGE_PWM" (deprecated in favor of SERVO_HBRIDGE_2PIN)
# # - configures a steering servo and an HBridge in 2pin mode (2 pwm pins)
# # - Uses ServoBlaster library, which is NOT installed by default, so
# #   you will need to install it to make this work.
# # - Servo takes a standard servo PWM pulse between 1 millisecond (fully reverse)
# #   and 2 milliseconds (full forward) with 1.5ms being neutral.
# # - the motor is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - the pwm pins produce a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# HBRIDGE_PIN_FWD = 18       # provides forward duty cycle to motor
# HBRIDGE_PIN_BWD = 16       # provides reverse duty cycle to motor
# STEERING_CHANNEL = 0       # PCA 9685 channel for steering control
# STEERING_LEFT_PWM = 460    # pwm value for full left steering (use `donkey calibrate` to measure value for your car)
# STEERING_RIGHT_PWM = 290   # pwm value for full right steering (use `donkey calibrate` to measure value for your car)
# 
# #
# # DC_STEER_THROTTLE drivetrain with one motor as steering, one as drive
# # - uses L298N type motor controller in two pin wiring
# #   scheme utilizing two pwm pins per motor; one for
# #   forward(or right) and one for reverse (or left)
# #
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_STEER_THROTTLE
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_STEER_THROTTLE = {
#     "LEFT_DUTY_PIN": "RPI_GPIO.BOARD.18",   # pwm pin produces duty cycle for steering left
#     "RIGHT_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for steering right
#     "FWD_DUTY_PIN": "RPI_GPIO.BOARD.15",    # pwm pin produces duty cycle for forward drive
#     "BWD_DUTY_PIN": "RPI_GPIO.BOARD.13",    # pwm pin produces duty cycle for reverse drive
# }
# 
# #
# # DC_TWO_WHEEL drivetrain pin configuration
# # - configures L298N_HBridge_2pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by two pwm pins,
# #   one for forward and one for backward (reverse).
# # - each pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the reverse pwm is 0 duty_cycle,
# #   in backward mode, the forward pwm is 0 duty cycle.
# # - both pwms are 0 duty cycle (LOW) to 'detach' motor and
# #   and glide to a stop.
# # - both pwms are full duty cycle (100% HIGH) to brake
# #
# # Pin specifier string format:
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL = {
#     "LEFT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.18",  # pwm pin produces duty cycle for left wheel forward
#     "LEFT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.16",  # pwm pin produces duty cycle for left wheel reverse
#     "RIGHT_FWD_DUTY_PIN": "RPI_GPIO.BOARD.15", # pwm pin produces duty cycle for right wheel forward
#     "RIGHT_BWD_DUTY_PIN": "RPI_GPIO.BOARD.13", # pwm pin produces duty cycle for right wheel reverse
# }
# 
# #
# # DC_TWO_WHEEL_L298N drivetrain pin configuration
# # - configures L298N_HBridge_3pin driver
# # - two wheels as differential drive, left and right.
# # - each wheel is controlled by three pins,
# #   one ttl output for forward, one ttl output
# #   for backward (reverse) enable and one pwm pin
# #   for motor power.
# # - the pwm pin produces a duty cycle from 0 (completely LOW)
# #   to 1 (100% completely high), which is proportional to the
# #   amount of power delivered to the motor.
# # - in forward mode, the forward pin  is HIGH and the
# #   backward pin is LOW,
# # - in backward mode, the forward pin is LOW and the
# #   backward pin is HIGH.
# # - both forward and backward pins are LOW to 'detach' motor
# #   and glide to a stop.
# # - both forward and backward pins are HIGH to brake
# #
# # GPIO pin configuration for the DRIVE_TRAIN_TYPE=DC_TWO_WHEEL_L298N
# # - use RPI_GPIO for RPi/Nano header pin output
# #   - use BOARD for board pin numbering
# #   - use BCM for Broadcom GPIO numbering
# #   - for example "RPI_GPIO.BOARD.18"
# # - use PIPGIO for RPi header pin output using pigpio server
# #   - must use BCM (broadcom) pin numbering scheme
# #   - for example, "PIGPIO.BCM.13"
# # - use PCA9685 for PCA9685 pin output
# #   - include colon separated I2C channel and address
# #   - for example "PCA9685.1:40.13"
# # - RPI_GPIO, PIGPIO and PCA9685 can be mixed arbitrarily,
# #   although it is discouraged to mix RPI_GPIO and PIGPIO.
# #
# DC_TWO_WHEEL_L298N = {
#     "LEFT_FWD_PIN": "RPI_GPIO.BOARD.16",        # TTL output pin enables left wheel forward
#     "LEFT_BWD_PIN": "RPI_GPIO.BOARD.18",        # TTL output pin enables left wheel reverse
#     "LEFT_EN_DUTY_PIN": "RPI_GPIO.BOARD.22",    # PWM pin generates duty cycle for left motor speed
# 
#     "RIGHT_FWD_PIN": "RPI_GPIO.BOARD.15",       # TTL output pin enables right wheel forward
#     "RIGHT_BWD_PIN": "RPI_GPIO.BOARD.13",       # TTL output pin enables right wheel reverse
#     "RIGHT_EN_DUTY_PIN": "RPI_GPIO.BOARD.11",   # PWM pin generates duty cycle for right wheel speed
# }
# 
# 
# 
# #
# # Input controllers
# #
# #WEB CONTROL
# WEB_CONTROL_PORT = int(os.getenv("WEB_CONTROL_PORT", 8887))  # which port to listen on when making a web controller
# WEB_INIT_MODE = "user"              # which control mode to start in. one of user|local_angle|local. Setting local will start in ai mode.
# 
# #JOYSTICK
# USE_JOYSTICK_AS_DEFAULT = False      #when starting the manage.py, when True, will not require a --js option to use the joystick
# JOYSTICK_MAX_THROTTLE = 0.5         #this scalar is multiplied with the -1 to 1 throttle value to limit the maximum throttle. This can help if you drop the controller or just don't need the full speed available.
# JOYSTICK_STEERING_SCALE = 1.0       #some people want a steering that is less sensitve. This scalar is multiplied with the steering -1 to 1. It can be negative to reverse dir.
# AUTO_RECORD_ON_THROTTLE = False     #if true, we will record whenever throttle is not zero. if false, you must manually toggle recording with some other trigger. Usually circle button on joystick.
# CONTROLLER_TYPE = 'xbox'            #(ps3|ps4|xbox|pigpio_rc|nimbus|wiiu|F710|rc3|MM1|custom) custom will run the my_joystick.py controller written by the `donkey createjs` command
# USE_NETWORKED_JS = False            #should we listen for remote joystick control over the network?
# NETWORK_JS_SERVER_IP = None         #when listening for network joystick control, which ip is serving this information
# JOYSTICK_DEADZONE = 0.01            # when non zero, this is the smallest throttle before recording triggered.
# JOYSTICK_THROTTLE_DIR = -1.0         # use -1.0 to flip forward/backward, use 1.0 to use joystick's natural forward/backward
# USE_FPV = False                     # send camera data to FPV webserver
# JOYSTICK_DEVICE_FILE = "/dev/input/js0" # this is the unix file use to access the joystick.
# 
# 
# #SOMBRERO
# HAVE_SOMBRERO = False           #set to true when using the sombrero hat from the Donkeycar store. This will enable pwm on the hat.
# 
# #PIGPIO RC control
# STEERING_RC_GPIO = 26
# THROTTLE_RC_GPIO = 20
# DATA_WIPER_RC_GPIO = 19
# PIGPIO_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
# PIGPIO_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
# PIGPIO_STOPPED_PWM = 1500
# PIGPIO_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
# PIGPIO_SHOW_STEERING_VALUE = False
# PIGPIO_INVERT = False
# PIGPIO_JITTER = 0.025   # threshold below which no signal is reported
# 
# 
# # ROBOHAT MM1 controller
# MM1_STEERING_MID = 1500         # Adjust this value if your car cannot run in a straight line
# MM1_MAX_FORWARD = 2000          # Max throttle to go fowrward. The bigger the faster
# MM1_STOPPED_PWM = 1500
# MM1_MAX_REVERSE = 1000          # Max throttle to go reverse. The smaller the faster
# MM1_SHOW_STEERING_VALUE = False
# # Serial port
# # -- Default Pi: '/dev/ttyS0'
# # -- Jetson Nano: '/dev/ttyTHS1'
# # -- Google coral: '/dev/ttymxc0'
# # -- Windows: 'COM3', Arduino: '/dev/ttyACM0'
# # -- MacOS/Linux:please use 'ls /dev/tty.*' to find the correct serial port for mm1
# #  eg.'/dev/tty.usbmodemXXXXXX' and replace the port accordingly
# MM1_SERIAL_PORT = '/dev/ttyS0'  # Serial Port for reading and sending MM1 data.
# 
# 
# #
# # LOGGING
# #
# HAVE_CONSOLE_LOGGING = True
# LOGGING_LEVEL = 'INFO'          # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
# 
# 
# #
# # MQTT TELEMETRY
# #
# HAVE_MQTT_TELEMETRY = False
# TELEMETRY_DONKEY_NAME = 'my_robot1234'
# TELEMETRY_MQTT_TOPIC_TEMPLATE = 'donkey/%s/telemetry'
# TELEMETRY_MQTT_JSON_ENABLE = False
# TELEMETRY_MQTT_BROKER_HOST = 'broker.hivemq.com'
# TELEMETRY_MQTT_BROKER_PORT = 1883
# TELEMETRY_PUBLISH_PERIOD = 1
# TELEMETRY_LOGGING_ENABLE = True
# TELEMETRY_LOGGING_LEVEL = 'INFO' # (Python logging level) 'NOTSET' / 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR' / 'FATAL' / 'CRITICAL'
# TELEMETRY_LOGGING_FORMAT = '%(message)s'  # (Python logging format - https://docs.python.org/3/library/logging.html#formatter-objects
# TELEMETRY_DEFAULT_INPUTS = 'pilot/angle,pilot/throttle,recording'
# TELEMETRY_DEFAULT_TYPES = 'float,float'
# 
# 
# #
# # PERFORMANCE MONITOR
# #
# HAVE_PERFMON = False
# 
# 
# #
# # RECORD OPTIONS
# #
# RECORD_DURING_AI = False        #normally we do not record during ai mode. Set this to true to get image and steering records for your Ai. Be careful not to use them to train.
# AUTO_CREATE_NEW_TUB = False     #create a new tub (tub_YY_MM_DD) directory when recording or append records to data directory directly
# 
# 
# #
# # LED
# #
# HAVE_RGB_LED = False            #do you have an RGB LED like https://www.amazon.com/dp/B07BNRZWNF
# LED_INVERT = False              #COMMON ANODE? Some RGB LED use common anode. like https://www.amazon.com/Xia-Fly-Tri-Color-Emitting-Diffused/dp/B07MYJQP8B
# 
# #LED board pin number for pwm outputs
# #These are physical pinouts. See: https://www.raspberrypi-spy.co.uk/2012/06/simple-guide-to-the-rpi-gpio-header-and-pins/
# LED_PIN_R = 12
# LED_PIN_G = 10
# LED_PIN_B = 16
# 
# #LED status color, 0-100
# LED_R = 0
# LED_G = 0
# LED_B = 1
# 
# #LED Color for record count indicator
# REC_COUNT_ALERT = 1000          #how many records before blinking alert
# REC_COUNT_ALERT_CYC = 15        #how many cycles of 1/20 of a second to blink per REC_COUNT_ALERT records
# REC_COUNT_ALERT_BLINK_RATE = 0.4 #how fast to blink the led in seconds on/off
# 
# #first number is record count, second tuple is color ( r, g, b) (0-100)
# #when record count exceeds that number, the color will be used
# RECORD_ALERT_COLOR_ARR = [ (0, (1, 1, 1)),
#             (3000, (5, 5, 5)),
#             (5000, (5, 2, 0)),
#             (10000, (0, 5, 0)),
#             (15000, (0, 5, 5)),
#             (20000, (0, 0, 5)), ]
# 
# #LED status color, 0-100, for model reloaded alert
# MODEL_RELOADED_LED_R = 100
# MODEL_RELOADED_LED_G = 0
# MODEL_RELOADED_LED_B = 0
# 
# 
# #
# # DonkeyGym
# #
# # Only on Ubuntu linux, you can use the simulator as a virtual donkey and
# # issue the same python manage.py drive command as usual, but have them control a virtual car.
# # This enables that, and sets the path to the simualator and the environment.
# # You will want to download the simulator binary from: https://github.com/tawnkramer/donkey_gym/releases/download/v18.9/DonkeySimLinux.zip
# # then extract that and modify DONKEY_SIM_PATH.
# DONKEY_GYM = False
# DONKEY_SIM_PATH = "path to sim" #"/home/tkramer/projects/sdsandbox/sdsim/build/DonkeySimLinux/donkey_sim.x86_64" when racing on virtual-race-league use "remote", or user "remote" when you want to start the sim manually first.
# DONKEY_GYM_ENV_NAME = "donkey-generated-track-v0" # ("donkey-generated-track-v0"|"donkey-generated-roads-v0"|"donkey-warehouse-v0"|"donkey-avc-sparkfun-v0")
# GYM_CONF = { "body_style" : "donkey", "body_rgb" : (128, 128, 128), "car_name" : "car", "font_size" : 100} # body style(donkey|bare|car01) body rgb 0-255
# GYM_CONF["racer_name"] = "Your Name"
# GYM_CONF["country"] = "Place"
# GYM_CONF["bio"] = "I race robots."
# 
# SIM_HOST = "127.0.0.1"              # when racing on virtual-race-league use host "trainmydonkey.com"
# SIM_ARTIFICIAL_LATENCY = 0          # this is the millisecond latency in controls. Can use useful in emulating the delay when useing a remote server. values of 100 to 400 probably reasonable.
# 
# # Save info from Simulator (pln)
# SIM_RECORD_LOCATION = False
# SIM_RECORD_GYROACCEL= False
# SIM_RECORD_VELOCITY = False
# SIM_RECORD_LIDAR = False
# 
# # publish camera over network on TCP socket
# # This is used to create a tcp service to publish the camera feed
# PUB_CAMERA_IMAGES = False
# 
# 
# #
# # AI Overrides
# #
# # Launch mode: override AI at launch time (transition from user to Auto pilot).
# AI_LAUNCH_DURATION = 0.0            # the ai will output throttle for this many seconds
# AI_LAUNCH_THROTTLE = 0.0            # the ai will output this throttle value
# AI_LAUNCH_ENABLE_BUTTON = 'R2'      # this keypress will enable this boost. It must be enabled before each use to prevent accidental trigger.
# AI_LAUNCH_KEEP_ENABLED = False      # when False ( default) you will need to hit the AI_LAUNCH_ENABLE_BUTTON for each use. This is safest. When this True, is active on each trip into "local" ai mode.
# 
# # throttle scaling: scale the output of the throttle of the ai pilot for all model types.
# AI_THROTTLE_MULT = 1.0              # this multiplier will scale every throttle value for all output from NN models
# 
# 
# #
# # Intel Realsense D435 and D435i depth sensing camera
# #
# REALSENSE_D435_RGB = True       # True to capture RGB image
# REALSENSE_D435_DEPTH = False    # True to capture depth as image array
# REALSENSE_D435_IMU = False      # True to capture IMU data (D435i only)
# REALSENSE_D435_ID = None        # serial number of camera or None if you only have one camera (it will autodetect)
# 
# 
# #
# # Stop Sign Detector
# #
# STOP_SIGN_DETECTOR = False
# STOP_SIGN_MIN_SCORE = 0.2
# STOP_SIGN_SHOW_BOUNDING_BOX = True
# STOP_SIGN_MAX_REVERSE_COUNT = 10    # How many times should the car reverse when detected a stop sign, set to 0 to disable reversing
# STOP_SIGN_REVERSE_THROTTLE = -0.5     # Throttle during reversing when detected a stop sign
# 
# #
# # Frames/Second counter
# #
# SHOW_FPS = False
# FPS_DEBUG_INTERVAL = 10    # the interval in seconds for printing the frequency info into the shell
# 
# #
# # computer vision template
# #
# # configure which part is used as the autopilot - change to use your own autopilot
# CV_CONTROLLER_MODULE = "donkeycar.parts.line_follower"
# CV_CONTROLLER_CLASS = "LineFollower"
# CV_CONTROLLER_INPUTS = ['cam/image_array']
# CV_CONTROLLER_OUTPUTS = ['pilot/steering', 'pilot/throttle', 'cv/image_array']
# CV_CONTROLLER_CONDITION = "run_pilot"
# 
# # LineFollower - line color and detection area
# SCAN_Y = 100          # num pixels from the top to start horiz scan
# SCAN_HEIGHT = 20      # num pixels high to grab from horiz scan
# COLOR_THRESHOLD_LOW  = (0, 50, 50)    # HSV dark yellow (opencv HSV hue value is 0..179, saturation and value are both 0..255)
# COLOR_THRESHOLD_HIGH = (50, 255, 255) # HSV light yellow (opencv HSV hue value is 0..179, saturation and value are both 0..255)
# 
# # LineFollower - target (expected) line position and detection thresholds
# TARGET_PIXEL = None   # In not None, then this is the expected horizontal position in pixels of the yellow line.
#                       # If None, then detect the position yellow line at startup;
#                       # so this assumes you have positioned the car prior to starting.
#                       # Alternatively set this to IMAGE_W / 2 to follow middle line
# # Widened 10 -> 20px. Halves the residual twitch (reversals 4.3% -> 3.3%) at
# no cost to tracking, since 20px is under 5% of the frame width.
TARGET_THRESHOLD = 20 # number of pixels from TARGET_PIXEL that vehicle must be pointing
#                       # before a steering change will be made; this prevents algorithm
#                       # from being too twitchy when it is on or near the line.
# CONFIDENCE_THRESHOLD = 0.0015   # The fraction of total sampled pixels that must be yellow in the sample slice.
#                                 # The sample slice will have SCAN_HEIGHT pixels and the total number
#                                 # of sampled pixels is IMAGE_W x SCAN_HEIGHT, so if you want to make sure
#                                 # that all the pixels in the sample slice are yellow, then the confidence
#                                 # threshold should be SCAN_HEIGHT / (IMAGE_W x SCAN_HEIGHT) or (1 / IMAGE_W).
#                                 # if you want half of the pixels in the slice to match hten (1 / IMAGE_W) / 2.
#                                 # If you keep getting `No line detected` logs in the console then you
#                                 # may want to lower the threshold.
# 
# # LineFollower - throttle step controller; increase throttle on straights, descrease on turns
# THROTTLE_MAX = 0.3    # maximum throttle value the controller will produce
# THROTTLE_MIN = 0.15   # minimum throttle value the controller will produce
# THROTTLE_INITIAL = THROTTLE_MIN  # initial throttle value
# THROTTLE_STEP = 0.05  # how much to change throttle when off the line
# 
# # These three PID constants are crucial to the way the car drives. If you are tuning them
# # start by setting the others zero and focus on first Kp, then Kd, and then Ki.
# PID_P = -0.01         # proportional mult for PID path follower
# PID_I = 0.000         # integral mult for PID path follower
# PID_D = -0.0001       # differential mult for PID path follower
# 
# PID_P_DELTA = 0.005   # amount the inc/dec function will change the P value
# PID_D_DELTA = 0.00005 # amount the inc/dec function will change the D value
# 
# OVERLAY_IMAGE = True  # True to draw computer vision overlay on camera image in web ui
#                       # NOTE: this does not affect what is saved to the data
# 
# 
# #
# # Assign path follow functions to buttons.
# # You can use game pad buttons OR web ui buttons ('web/w1' to 'web/w5')
# # Use None use the game controller default
# # NOTE: the cross button is already reserved for the emergency stop
# #
# TOGGLE_RECORDING_BTN = "option" # button to toggle recording mode
# INC_PID_D_BTN = None            # button to change PID 'D' constant by PID_D_DELTA
# DEC_PID_D_BTN = None            # button to change PID 'D' constant by -PID_D_DELTA
# INC_PID_P_BTN = "R2"            # button to change PID 'P' constant by PID_P_DELTA
# DEC_PID_P_BTN = "L2"            # button to change PID 'P' constant by -PID_P_DELTA
# 
# #
# # CenterLineFollower - classical-CV follower for intermittent greenish-blue/
# # light-blue center tape. Also supports half-lane mode (CENTER_LINE_LANE_MODE
# # = "left"/"right"), which additionally tracks the solid white boundary tape
# # on that side so the car lane-keeps within half the track instead of
# # needing to straddle its full width -- see the CENTER_LINE_EDGE_*/
# # CENTER_LINE_HALF_LANE_* group further below. See
# # donkeycar/parts/center_line_follower.py and the companion
# # center_line_follower.md for the full pipeline write-up.
# # None of these are required here -- CenterLineFollower falls back to the
# # DEFAULT_* constants at the top of its own file if a name below isn't set.
# # To actually drive with this Part instead of the stock LineFollower, also
# # set in myconfig.py:
# #   CV_CONTROLLER_MODULE = "donkeycar.parts.center_line_follower"
# #   CV_CONTROLLER_CLASS = "CenterLineFollower"
# # NOTE: PID_P/PID_I/PID_D above must stay defined even when using
# # CenterLineFollower -- cv_control.py constructs a PID from them
# # unconditionally before the CV controller is even selected.  CenterLineFollower
# # itself ignores that PID object and implements its own steering control below.
# #
# 
# # CenterLineFollower - region of interest (a tall horizontal band, not a thin
# # slice, so a dash fragment is likely to intersect it even with gaps in the tape)
# CENTER_LINE_ROI_Y_TOP = 130      # top row (pixels from image top) of the scan band
# CENTER_LINE_ROI_Y_BOTTOM = 230   # bottom row; kept short of IMAGE_H to avoid the chassis/bumper
# 
# # CenterLineFollower - HSV color threshold for the greenish-blue/teal tape
# # (opencv HSV hue value is 0..179, saturation and value are both 0..255).
# # Saturation (the 2nd number) is what actually separates the tape from the
# # low-saturation white boundary tape/track floor -- raise it first if the
# # mask is picking up white. Retune both of these before anything else when
# # moving from sim to the real camera.
# CENTER_LINE_COLOR_LOW = (75, 80, 40)
# CENTER_LINE_COLOR_HIGH = (130, 255, 255)
# 
# # CenterLineFollower - morphological cleanup (open then close) kernel size, in px
# CENTER_LINE_MORPH_KERNEL = 5
# 
# # CenterLineFollower - reject contours smaller than this fraction of the ROI area
# CENTER_LINE_MIN_AREA_FRACTION = 0.005
# 
# # CenterLineFollower - shape filters, applied to both the center dash and
# # (in half-lane mode) the edge search. MAX_FILL_RATIO rejects a contour
# # whose area fills more than this fraction of its own bounding box -- real
# # tape is thin/elongated (measured ~0.43-0.59), a solid background blob is
# # not (measured ~0.84 on a real false-positive). MIN_SOLIDITY rejects a
# # contour whose area/convex-hull-area is below this -- real tape is a
# # smooth, nearly-convex stripe; real shrub/foliage contours measured
# # ~0.56-0.71 on this track (comfortably below the 0.85 floor) since they
# # survive this pipeline's morphological cleanup as one larger, genuinely
# # irregular shape, not a compact blob fill_ratio alone would catch.
# CENTER_LINE_MAX_FILL_RATIO = 0.65
# CENTER_LINE_MIN_SOLIDITY = 0.85
# 
# # CenterLineFollower - target horizontal pixel for the tape centroid.
# # None means "use the geometric center of the frame" (width // 2) -- the
# # correct default for a *center*-line follower. Override only if the camera
# # isn't physically centered on the car.
# CENTER_LINE_TARGET_PIXEL = None
# 
# # CenterLineFollower - steering control (proportional + optional derivative
# # on normalized error, roughly -1..1). Start by tuning STEER_KP alone with
# # STEER_KD at 0; see center_line_follower.md for the full tuning procedure.
# CENTER_LINE_STEER_KP = 0.8
# CENTER_LINE_STEER_KD = 0.0
# CENTER_LINE_ERROR_SMOOTHING_ALPHA = 0.5   # lower = smoother but more lag, higher = more jagged but faster
# 
# # CenterLineFollower - throttle. Strictly constant while tracking; only
# # drops during an extended tape loss (see the gap-timing group below).
# CENTER_LINE_THROTTLE = 0.2          # always re-tune per-vehicle; start low on the real car
# CENTER_LINE_THROTTLE_LOST_MIN = 0.0 # throttle floor once LOST_TIME_SEC has elapsed
# 
# # CenterLineFollower - how long to tolerate the tape being undetected before
# # reacting. A brief HOLD (tape still expected, e.g. a dash gap) keeps
# # steering and throttle unchanged; past LOST_TIME_SEC the car ramps throttle
# # down to CENTER_LINE_THROTTLE_LOST_MIN and stops steering-blind. Measure
# # your track's longest dash gap and set these relative to how long it takes
# # to cross it at CENTER_LINE_THROTTLE.
# CENTER_LINE_HOLD_TIME_SEC = 0.5
# CENTER_LINE_LOST_TIME_SEC = 2.0
# 
# # CenterLineFollower - half-lane mode. "center" (default) is the original
# # center-tape-only behavior. "left"/"right" additionally tracks the solid
# # white boundary tape on that side and steers the *midpoint* of the dash
# # and that edge toward the frame center, so the car stays within that half
# # of the track. There's no runtime toggle for this yet (see LaneMode's
# # docstring) -- set it here before starting the car.
# CENTER_LINE_LANE_MODE = "center"     # "left" | "center" | "right"
# 
# # CenterLineFollower - the solid white edge tape (only used in "left"/
# # "right" mode) is detected by local contrast (a morphological top-hat),
# # not a fixed brightness threshold -- plain floor brightness swings too
# # widely across the scene (measured ~92-190 depending on shade/sun) to
# # separate from the tape with one global number, but the tape is
# # consistently ~55-60 V brighter than whatever floor is immediately next
# # to it. KERNEL_SIZE must be wider than the tape line itself (~1-4px) but
# # smaller than the scale lighting varies over; CONTRAST_THRESHOLD is how
# # much local brightness advantage counts as "found". See
# # center_line_follower.py's DEFAULT_EDGE_TOPHAT_KERNEL_SIZE comment for the
# # full reasoning and measured numbers.
# CENTER_LINE_EDGE_TOPHAT_KERNEL_SIZE = 21
# CENTER_LINE_EDGE_CONTRAST_THRESHOLD = 30
# 
# # CenterLineFollower - reject edge contours smaller than this fraction of
# # the *half*-ROI area searched (left/right half, not the whole ROI).
# CENTER_LINE_EDGE_MIN_AREA_FRACTION = 0.003
# 
# # CenterLineFollower - half-lane width tracking. HALF_LANE_WIDTH_PX is only
# # an initial guess (pixel distance between the dash and a side edge) used
# # before both have been seen together in the same frame, or during an
# # extended one-sided occlusion; it self-corrects via HALF_LANE_WIDTH_
# # SMOOTHING (0-1: higher adapts to a widening/narrowing track faster but
# # noisier) every frame both are visible.
# CENTER_LINE_HALF_LANE_WIDTH_PX = 150
# CENTER_LINE_HALF_LANE_WIDTH_SMOOTHING = 0.2
# 
# # CenterLineFollower - tracking continuity for both the dash and edge
# # search. Real tape moves smoothly frame-to-frame; a spurious background
# # match (a rock, a glint, architecture that happens to pass the color/
# # contrast+shape gates) shows up at an inconsistent, unrelated position
# # instead -- so once something's been tracked, the closest candidate to
# # the last known position wins over just the largest blob, even if a
# # background false positive is larger. MAX_TRACK_JUMP_PX guards the case
# # where the real tape is genuinely gone and a background blob is the only
# # candidate: if even the closest one is further than this, it's treated as
# # not-found (gap tolerance handles it) instead of snapping onto it.
# CENTER_LINE_MAX_TRACK_JUMP_PX = 60
# 
# # CenterLineFollower - frames of *consistent* position required before
# # trusting a brand-new lock (startup, or right after being fully lost --
# # see DEFAULT_CONFIRM_FRAMES's comment in center_line_follower.py). This is
# # what actually rejects background noise/pebbles/pavement texture that
# # passes the area+shape gates with similar confidence to real tape in a
# # single frame (measured ~0.02-0.03 for both on real track footage) --
# # MAX_TRACK_JUMP_PX alone can't help here since there's no prior position
# # yet to gate against. Lower = locks on faster but more exposed to a
# # transient false positive; higher = takes longer to start tracking but
# # more resistant to noise.
# CENTER_LINE_CONFIRM_FRAMES = 3

# --- Behaviour when the line is not visible -------------------------------
# The centre line is DASHED and some stretches of the track have no tape at
# all. Measured over 1200 real recorded frames: 34 separate gaps, median 0.58s
# but p90 2.5s and the worst 5.7 SECONDS. The controller used to hold its last
# steering value for the whole gap, so a line lost mid-turn meant the car kept
# turning for seconds -- which is exactly how it wandered off at a few points.
#
# Hold for LINE_LOST_HOLD_FRAMES (10 = 0.5s at 20Hz) so an ordinary dash gap
# is ridden through committed, then decay toward straight at LINE_LOST_DECAY
# per frame (0.90 => ~1s to unwind). Straight is the safe default when blind.
# Raise HOLD if it straightens during normal dash gaps; lower DECAY (e.g.
# 0.80) to straighten faster.
LINE_LOST_HOLD_FRAMES = 10
LINE_LOST_DECAY = 0.90

# Progressive steering. One linear gain cannot cover both jobs: measured over
# 1500 real frames the line sits a median 78px off centre and 40% of frames
# exceed 80px, so the gain that stops the weave on straights (Kp=-0.005) tops
# out at 0.70 steering even at the largest error seen (148px) -- never enough
# to get round a corner, which is where it kept losing the line.
# This stretches the error by how far off it already is: near-linear for small
# deviations (straights stay calm), full lock reachable on a corner.
#   0.0 = off (plain linear)   1.0 = a 124px error acts like ~196px
STEERING_PROGRESSIVE = 1.0

