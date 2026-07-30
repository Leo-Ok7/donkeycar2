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
import string
import time
import sys

import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2 as cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
from depthai import Pipeline, DataOutputQueue, ImgFrame, ImgDetections, ImgDetection
from numpy import ndarray
from typing import List

WIDTH = 640
HEIGHT = 480


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

        self.width = width
        self.height = height

        # TODO: Accommodate using device native resolutions to avoid resizing.
        self.resize = (width != WIDTH) or (height != HEIGHT)
        if self.resize:
            print(
                f"The output images will be resized from {(WIDTH, HEIGHT)} to {(self.width, self.height)} using OpenCV. Device resolution in use is 640x480."
            )

        self.pipeline = None
        if self.enable_depth or self.enable_rgb:
            self.pipeline = depthai.Pipeline()

            device_info = self.get_depthai_device_info(device_id)

            if self.enable_depth:
                self.setup_depth_camera(WIDTH, HEIGHT)

            if self.enable_rgb:
                self.setup_rgb_camera(self.width, self.height)

            self.oak_d_device = depthai.Device(self.pipeline, device_info)

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

    def setup_depth_camera(self, width, height):
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

    def setup_rgb_camera(self, width, height):
        cam_rgb = self.pipeline.create(depthai.node.ColorCamera)

        res = depthai.ColorCameraProperties.SensorResolution.THE_1080_P

        cam_rgb.setResolution(res)
        # Preview size is what makes the DEVICE do the downscale -- the Pi
        # never touches full-resolution frames.
        cam_rgb.setPreviewSize(width, height)
        cam_rgb.setInterleaved(False)
        # ColorCamera's default color order is BGR. Every other camera Part
        # in this codebase hands cam/image_array to the rest of the pipeline
        # as RGB (see camera.py CSICamera's explicit COLOR_BGR2RGB), so make
        # the device itself emit RGB instead of relying on a conversion here.
        cam_rgb.setColorOrder(depthai.ColorCameraProperties.ColorOrder.RGB)

        xout_rgb = self.pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        # `.preview` (not `.video`) is the stream that actually respects
        # setPreviewSize(); `.video` ignores it and outputs at full
        # sensor/ISP resolution.
        cam_rgb.preview.link(xout_rgb.input)

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
        # getCvFrame() ALWAYS returns BGR interleaved, regardless of the
        # setColorOrder(...RGB) call in setup_rgb_camera() -- confirmed in
        # Luxonis's own ImgFrame docs, and empirically here: real orange
        # objects came through blue and magenta came through purple, while
        # green (unaffected by an R/B swap) looked correct. The comment up
        # there claiming the device emits RGB "instead of relying on a
        # conversion here" is simply wrong, so downstream code that assumes
        # cam/image_array is RGB (every COLOR_RGB2HSV in this codebase) was
        # silently working on red/blue-swapped pixels.
        if new_frame is None:
            return None
        frame = new_frame.getCvFrame()
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame          # depth/mono frames pass through untouched
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _poll(self):
        last_time = self.frame_time
        self.frame_time = time.time() - self.start_time
        self.frame_count += 1

        #
        # convert camera frames to images
        #
        # Fetch each stream's queue ONLY if that stream was actually created in
        # the pipeline. The old code requested BOTH queues whenever either was
        # enabled, so running with OAKD_DEPTH=False (depth stream never built)
        # killed this thread on the first poll with:
        #   RuntimeError: Queue for stream name 'depth' doesn't exist
        # which froze the camera silently -- the vehicle loop kept running on a
        # stale/None frame. Guarding them separately lets RGB-only work, so
        # OAKD_DEPTH can stay False and the tub schema stays RGB-only.
        if self.enable_depth:
            self.depth_queue: DataOutputQueue = self.oak_d_device.getOutputQueue(
                name="depth", maxSize=1, blocking=False
            )
            self.depth_image = self.get_frame(self.depth_queue)

        if self.enable_rgb:
            self.rgb_queue: DataOutputQueue = self.oak_d_device.getOutputQueue(
                "rgb", maxSize=1, blocking=False
            )
            self.color_image = self.get_frame(self.rgb_queue)

        if self.resize:
            if self.width != WIDTH or self.height != HEIGHT:
                import cv2

                self.color_image = (
                    cv2.resize(
                        self.color_image, (self.width, self.height), cv2.INTER_NEAREST
                    )
                    if self.enable_rgb
                    else None
                )
                self.depth_image = (
                    cv2.resize(
                        self.depth_image, (self.width, self.height), cv2.INTER_NEAREST
                    )
                    if self.enable_depth
                    else None
                )

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
        self.oak_d_device.close()


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

    width = 640
    height = 480
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
