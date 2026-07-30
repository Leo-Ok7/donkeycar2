"""
Author: Brian Henry & Manav Gagvani
File: oak_d.py
Date: February 13 2022, revised July 10, 2025
Notes:
    Based on realsense435i.py by Ed Murphy: https://github.com/autorope/donkeycar/blob/454be3068ea5dfbac226c3be4d84b0a61d1cec84/donkeycar/parts/realsense435i.py
    Based on https://github.com/luxonis/depthai-tutorials/blob/d571473911f876b0d4ac52b7ffdc0fb2beae1641/1-hello-world/hello_world.py

    https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#first-steps-with-depthai
    > If you are using a Linux system, in most cases you have to add a new udev rule for our script to be able to access the device correctly. You can add and apply new rules by running
    $ echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
    $ sudo udevadm control --reload-rules && sudo udevadm trigger
    (or: "RuntimeError: No DepthAI (Oak-D-Lite) device (camera) found!")

    `sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow`
"""

import argparse
import logging
import string
import time
import sys

import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2 as cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
from depthai import Pipeline, DataOutputQueue, ImgFrame, ImgDetections, ImgDetection
from numpy import ndarray
from typing import List

logger = logging.getLogger(__name__)

WIDTH = 426
HEIGHT = 240

# Steady capture rate. Match this to cfg.DRIVE_LOOP_HZ so the vehicle loop gets
# roughly one fresh frame per tick. Full-sensor ISP readout limits the ceiling,
# and depthai will clamp (with a warning) if the sensor cannot sustain it.
CAMERA_FPS = 20

# --- Full-FOV capture ---------------------------------------------------------
# The stock part used the `video` output at THE_1080_P, which is hard-cropped to
# 16:9 no matter what resolution is requested. That crop throws away the bottom
# of the sensor's native 4:3 field of view -- exactly the near-ground strip a
# line follower needs -- so the car loses the line on turns.
#
# Instead we capture the whole sensor and let the on-device ISP scale it down,
# then squash (never re-crop) to WIDTH x HEIGHT in _poll(). Squashing keeps the
# recovered FOV; re-cropping to 16:9 would silently undo the whole fix.
#
# Sensor resolution is board-specific and a mismatch makes depthai throw when the
# device is created, so we try candidates in order and use the first that works:
#   THE_12_MP -> OAK-D      / IMX378, 4056x3040; 1/6 -> ~676x506
#   THE_13_MP -> OAK-D-Lite / IMX214, 4208x3120; 1/6 -> ~701x520
# The 1/4 entries are lower-risk fallbacks: some firmware rejects ISP scale
# fractions that do not divide the sensor evenly. All candidates are 4:3 modes --
# do not add a 16:9 mode (e.g. THE_4_K) here, as that reintroduces the crop.
ISP_CAPTURE_CANDIDATES = (
    (depthai.ColorCameraProperties.SensorResolution.THE_12_MP, 1, 6),
    (depthai.ColorCameraProperties.SensorResolution.THE_13_MP, 1, 6),
    (depthai.ColorCameraProperties.SensorResolution.THE_12_MP, 1, 4),
    (depthai.ColorCameraProperties.SensorResolution.THE_13_MP, 1, 4),
)


class OakD(object):
    """
    Donkeycar part for the Oak-D camera
    Intel Movidius based depth sensing camera
    https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html
    https://www.kickstarter.com/projects/opencv/opencv-ai-kit-oak-depth-camera-4k-cv-edge-object-detection
    https://shop.luxonis.com/
    """

    def __init__(
        self,
        width=WIDTH,
        height=HEIGHT,
        enable_rgb=True,
        enable_depth=True,
        device_id=None,
    ):
        self.device_id = device_id  # "18443010C1E4681200" # serial number of device to use|None to use default|"list" to list devices and exit
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        # target output size; frames are resized to this in _poll()
        self.width = width
        self.height = height

        self.rgb_queue = None
        self.depth_queue = None
        self.oak_d_device = None
        self.pipeline = None

        if self.enable_depth or self.enable_rgb:
            device_info = self.get_depthai_device_info(device_id)
            self._open_device(device_info)

            # create the output queues once, not on every frame
            if self.enable_rgb:
                self.rgb_queue = self.oak_d_device.getOutputQueue(
                    "rgb", maxSize=1, blocking=False
                )
            if self.enable_depth:
                self.depth_queue = self.oak_d_device.getOutputQueue(
                    "depth", maxSize=1, blocking=False
                )

        # initialize frame state
        self.color_image = None
        self.depth_image = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_time = self.start_time

        self.running = True

    # Taken from the demo application.
    def get_depthai_device_info(self, device_id: string):
        device_infos = depthai.Device.getAllAvailableDevices()
        if len(device_infos) == 0:
            raise RuntimeError("No DepthAI (Oak-D-Lite) device (camera) found!")
        else:
            print("Available devices:")
            for i, deviceInfo in enumerate(device_infos):
                print(f"[{i}] {deviceInfo.getMxId()} [{deviceInfo.state.name}]")

            # Set the deviceId to "list" in order to list the connected devices' ids.
            if device_id == "list":
                raise SystemExit(0)
            elif device_id is not None:
                matching_device = next(
                    filter(lambda info: info.getMxId() == device_id, device_infos), None
                )
                if matching_device is None:
                    raise RuntimeError(
                        f"No DepthAI device found with id matching {device_id} !"
                    )
                return matching_device
            elif len(device_infos) == 1:
                return device_infos[0]
            else:
                val = input("Which DepthAI Device you want to use: ")
                try:
                    return device_infos[int(val)]
                except:
                    raise ValueError(f"Incorrect value supplied: {val}")

    def _open_device(self, device_info):
        """
        Build the pipeline and open the camera.

        The sensor resolution a board accepts is board-specific, and asking for
        an unsupported one makes depthai raise when the device is created -- not
        when it is configured. So the whole build-and-open has to be retried per
        candidate rather than just the setResolution() call.

        On success self.pipeline and self.oak_d_device are set and the winning
        combination is logged. If every candidate fails we raise with the full
        list of what was tried, so a failure is loud and diagnosable instead of
        a bare depthai error.
        """
        if not self.enable_rgb:
            # No color camera means no sensor-resolution choice to make.
            self.pipeline = depthai.Pipeline()
            self.setup_depth_camera()
            self.oak_d_device = depthai.Device(self.pipeline, device_info)
            return

        failures = []
        for resolution, scale_num, scale_den in ISP_CAPTURE_CANDIDATES:
            try:
                self.pipeline = depthai.Pipeline()
                if self.enable_depth:
                    self.setup_depth_camera()
                self.setup_rgb_camera(resolution, scale_num, scale_den)
                self.oak_d_device = depthai.Device(self.pipeline, device_info)
            except Exception as error:
                failures.append(f"  {resolution} @ {scale_num}/{scale_den}: {error}")
                self.pipeline = None
                continue

            # Opening the device is NOT enough to prove the mode is usable: the
            # ISP size setIspScale() lands on can be one getCvFrame() cannot
            # convert. THE_12_MP @ 1/6 on this hardware yields 676x507, and the
            # odd height fails NV12->BGR:
            #   (-215:Assertion failed) sz.width % 2 == 0 && sz.height % 3 == 0
            # That threw in the camera THREAD on the first frame, long after
            # this function had already reported success -- so the candidate has
            # to be validated by actually converting a frame here, while we can
            # still fall through to the next one.
            usable, why = self._validate_rgb_capture()
            if not usable:
                failures.append(
                    f"  {resolution} @ {scale_num}/{scale_den}: opened but "
                    f"unusable ({why})"
                )
                try:
                    self.oak_d_device.close()
                except Exception:
                    pass
                self.oak_d_device = None
                self.pipeline = None
                continue

            logger.info(
                f"OAK-D capturing at {resolution} with ISP scale "
                f"{scale_num}/{scale_den}, delivered as {self.width}x{self.height}"
            )
            return

        raise RuntimeError(
            "Could not open the OAK-D with any known full-FOV capture mode.\n"
            "Tried:\n" + "\n".join(failures)
        )

    def _validate_rgb_capture(self, timeout_s=8.0):
        """
        Pull one real RGB frame and convert it, to prove the chosen capture mode
        actually works end to end.

        Returns (True, "") if a frame arrived and getCvFrame() succeeded, else
        (False, reason). Never raises -- a candidate that fails here is just
        rejected so _open_device() can try the next one.
        """
        try:
            queue = self.oak_d_device.getOutputQueue(
                "rgb", maxSize=1, blocking=False
            )
        except Exception as error:
            return False, f"no rgb queue: {error}"

        deadline = time.time() + timeout_s
        frame = None
        while time.time() < deadline:
            try:
                frame = queue.tryGet()
            except Exception as error:
                return False, f"queue read failed: {error}"
            if frame is not None:
                break
            time.sleep(0.02)

        if frame is None:
            return False, f"no frame within {timeout_s:.0f}s"

        size = f"{frame.getWidth()}x{frame.getHeight()}"
        try:
            frame.getCvFrame()
        except Exception as error:
            # Most likely an ISP size NV12->BGR cannot handle (odd height).
            return False, f"ISP {size}, getCvFrame failed: {str(error)[:80]}"
        return True, ""

    def setup_depth_camera(self):
        # Set up left and right cameras
        mono_left = self.get_mono_camera(self.pipeline, True)
        mono_right = self.get_mono_camera(self.pipeline, False)

        # Combine left and right cameras to form a stereo pair
        stereo: depthai.node.StereoDepth = self.get_stereo_pair(
            self.pipeline, mono_left, mono_right
        )

        # Define and name output depth map
        xout_depth = self.pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")

        stereo.depth.link(xout_depth.input)

    def setup_rgb_camera(self, resolution, scale_numerator, scale_denominator):
        cam_rgb = self.pipeline.create(depthai.node.ColorCamera)

        # Capture the whole sensor and let the ISP scale it down, rather than
        # using a 16:9 sensor mode which crops away the near-ground FOV.
        cam_rgb.setResolution(resolution)
        cam_rgb.setIspScale(scale_numerator, scale_denominator)
        cam_rgb.setFps(CAMERA_FPS)

        xout_rgb = self.pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        # isp (not video) is the full-FOV output.
        cam_rgb.isp.link(xout_rgb.input)

    def get_mono_camera(self, pipeline: Pipeline, is_left: bool):
        # Configure mono camera
        mono = pipeline.createMonoCamera()

        # Set camera resolution
        mono.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_480_P)

        if is_left:
            # Get left camera
            mono.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        else:
            # Get right camera
            mono.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

        return mono

    def get_stereo_pair(self, pipeline: Pipeline, mono_left, mono_right):
        # Configure the stereo pair for depth estimation
        new_stereo = pipeline.createStereoDepth()
        # Checks occluded pixels and marks them as invalid
        new_stereo.setLeftRightCheck(True)

        # Configure left and right cameras to work as a stereo pair
        mono_left.out.link(new_stereo.left)
        mono_right.out.link(new_stereo.right)

        return new_stereo

    def get_frame(self, queue: DataOutputQueue):
        # Get frame from queue
        new_frame: ImgFrame = queue.get()
        # Convert to OpenCV format
        return new_frame.getCvFrame()

    def _fit_to_output_size(self, frame):
        """
        Squash a frame to exactly (self.width, self.height).

        setIspScale() only hits approximate fractions -- a 12MP sensor at 1/6 is
        about 676x506 -- so this pins the frame to the size manage.py expects.

        This deliberately RESIZES rather than crops. The ISP frame is ~4:3 and
        the target is 16:9, so it gets horizontally squashed. That is the point:
        cropping back toward 16:9 would discard the near-ground field of view
        this whole capture path exists to recover.
        """
        if frame is None:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST
            )
        return frame

    def _poll(self):
        self.frame_time = time.time() - self.start_time
        self.frame_count += 1

        #
        # convert camera frames to images
        # NOTE: only read a queue if its feature is enabled -- reading the depth
        # queue when depth is off blocks forever / throws.
        #
        if self.enable_rgb:
            self.color_image = self._fit_to_output_size(self.get_frame(self.rgb_queue))

        if self.enable_depth:
            self.depth_image = self._fit_to_output_size(self.get_frame(self.depth_queue))

    def update(self):
        """
        When running threaded, update() is called from the background thread
        to update the state.  run_threaded() is called to return the latest state.
        """
        while self.running:
            self._poll()

    def run_threaded(self):
        """
        Return the latest state read by update().  This will not block.
        All 4 states are returned, but may be None if the feature is not enabled when the camera part is constructed.
        For gyroscope, x is pitch, y is yaw and z is roll.
        :return: (rbg_image: nparray, depth_image: nparray, acceleration: (x:float, y:float, z:float), gyroscope: (x:float, y:float, z:float))
        """
        return self.color_image, self.depth_image

    def run(self):
        """
        Read and return frame from camera.  This will block while reading the frame.
        see run_threaded() for return types.
        """
        self._poll()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(2)  # give thread enough time to shutdown

        # done running
        if self.oak_d_device is not None:
            self.oak_d_device.close()
            self.oak_d_device = None


#
# self test
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rgb", default=False, action="store_true", help="Stream RGB camera"
    )
    parser.add_argument(
        "--depth", default=False, action="store_true", help="Stream depth camera"
    )
    parser.add_argument(
        "--device_id",
        help='Camera id (if more than one camera connected), or "list" to print the connected device ids',
    )
    args = parser.parse_args()

    if not (args.rgb or args.depth):
        print("Must specify one or more of --rgb, --depth")
        parser.print_help()
        sys.exit(0)

    show_opencv_window = (
        args.rgb or args.depth
    )  # True to show images in opencv window: note that default donkeycar environment is not configured for this.
    if show_opencv_window:
        import cv2

    enable_rgb = args.rgb
    enable_depth = args.depth

    devices = depthai.Device.getAllAvailableDevices()

    device_id = args.device_id  # getMxId

    width = WIDTH
    height = HEIGHT
    channels = 3

    profile_frames = 0  # set to non-zero to calculate the max frame rate using given number of frames

    camera = None
    try:
        camera = OakD(
            width=width,
            height=height,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            device_id=device_id,
        )

        frame_count = 0
        start_time = time.time()
        frame_time = start_time
        while True:
            #
            # read data from camera
            #
            color_image, depth_image = camera.run()

            # maintain frame timing
            frame_count += 1
            last_time = frame_time
            frame_time = time.time()

            # Show images
            if show_opencv_window and not profile_frames:
                cv2.namedWindow("Oak-D", cv2.WINDOW_AUTOSIZE)
                if enable_rgb or enable_depth:
                    # make sure depth and color images have same number of channels so we can show them together in the window
                    if 3 == channels:
                        depth_colormap = (
                            cv2.applyColorMap(
                                cv2.convertScaleAbs(depth_image, alpha=0.03),
                                cv2.COLORMAP_JET,
                            )
                            if enable_depth
                            else None
                        )
                    else:
                        depth_colormap = (
                            cv2.cvtColor(
                                cv2.applyColorMap(
                                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                                    cv2.COLORMAP_JET,
                                ),
                                cv2.COLOR_RGB2GRAY,
                            )
                            if enable_depth
                            else None
                        )

                    # Stack both images horizontally (i.e. side by side).
                    images = None
                    if enable_rgb:
                        images = (
                            np.hstack((color_image, depth_colormap))
                            if enable_depth
                            else color_image
                        )
                    elif enable_depth:
                        images = depth_colormap

                    if images is not None:
                        cv2.imshow("Oak-D", images)

                # Press esc or 'q' to close the image window
                key = cv2.waitKey(1)
                if key & 0xFF == ord("q") or key == 27:
                    cv2.destroyAllWindows()
                    break
            if profile_frames > 0:
                if frame_count == profile_frames:
                    print(
                        f"Acquired {frame_count} frames in {frame_time - start_time} seconds for {frame_count / (frame_time - start_time)} fps"
                    )

                    break
            else:
                time.sleep(0.05)
    finally:
        if camera is not None:
            camera.shutdown()
