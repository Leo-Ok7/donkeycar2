#!/usr/bin/env python3
"""
OAK-D capture sanity check for the line/lane following pipeline.

Answers the three questions that must be settled before any CV is trusted:

  1. Is the frame exactly 426x240?  (what manage.py and the ROI fractions assume)
  2. Did the ISP patch actually recover the near-ground field of view?
  3. Do frames arrive as BGR or RGB?

Question 3 matters more than it looks. cv2.cvtColor(..., COLOR_BGR2HSV) assumes
BGR. If the real order is RGB, yellow tape lands nowhere near hue 20-35 and the
mask comes back empty forever -- with no error message. Rather than guess, this
script writes the SAME frame twice under both interpretations and you pick the
one that looks right.

Usage:
    python scripts/oakd_color_check.py [--out-dir DIR] [--warmup N]

Then:
  - Compare oakd_as_bgr.png and oakd_as_rgb.png. Point the camera at something
    strongly YELLOW or RED first -- those are the colors that flip most
    obviously. Whichever file shows correct colors names the true order.
  - Set CAMERA_COLOR_ORDER in myconfig.py to that value ("BGR" or "RGB").
  - Feed the correct-looking PNG to scripts/hsv_picker.py to pick the yellow
    HSV bounds:  python scripts/hsv_picker.py -f oakd_as_bgr.png
"""

import argparse
import os
import sys

import cv2
import numpy as np

from donkeycar.parts.oak_d import OakD, WIDTH, HEIGHT


def main(out_dir, warmup):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Opening OAK-D, expecting {WIDTH}x{HEIGHT} frames ...")
    # Depth is not needed here and costs bandwidth; RGB only.
    camera = OakD(width=WIDTH, height=HEIGHT, enable_rgb=True, enable_depth=False)

    try:
        # The first frames can be dark or malformed while auto-exposure settles.
        frame = None
        for _ in range(max(1, warmup)):
            frame, _ = camera.run()

        if frame is None:
            print("FAIL: camera returned no frame.")
            return 1

        print()
        print(f"  frame shape : {frame.shape}   (want ({HEIGHT}, {WIDTH}, 3))")
        print(f"  dtype       : {frame.dtype}")

        size_ok = frame.shape[0] == HEIGHT and frame.shape[1] == WIDTH
        print(f"  size check  : {'PASS' if size_ok else 'FAIL'}")

        # Channel means. On a yellow/red target the first and last channel means
        # differ a lot, which is what makes the two PNGs below easy to tell apart.
        means = frame.reshape(-1, frame.shape[2]).mean(axis=0)
        print(f"  channel means (ch0, ch1, ch2): "
              f"({means[0]:.1f}, {means[1]:.1f}, {means[2]:.1f})")
        print()

        # cv2.imwrite() always writes its input as if it were BGR.
        # So: writing the raw frame shows what it looks like IF it is BGR;
        #     writing the channel-swapped frame shows what it looks like IF it is RGB.
        as_bgr_path = os.path.join(out_dir, "oakd_as_bgr.png")
        as_rgb_path = os.path.join(out_dir, "oakd_as_rgb.png")
        cv2.imwrite(as_bgr_path, frame)
        cv2.imwrite(as_rgb_path, frame[:, :, ::-1])

        print("Wrote two interpretations of the same frame:")
        print(f"  {as_bgr_path}   <- correct colors here means CAMERA_COLOR_ORDER = \"BGR\"")
        print(f"  {as_rgb_path}   <- correct colors here means CAMERA_COLOR_ORDER = \"RGB\"")
        print()
        print("Also check in that image that you can see the ground CLOSE to the car.")
        print("The scene should look horizontally squashed (a ~4:3 view squashed into")
        print("16:9). Squashed is correct. If it looks normally proportioned but the")
        print("near ground is missing, the ISP patch did not take effect.")
        return 0 if size_ok else 1

    finally:
        camera.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=".", help="where to write the PNGs")
    parser.add_argument("--warmup", type=int, default=15,
                        help="frames to discard while auto-exposure settles")
    args = parser.parse_args()
    sys.exit(main(args.out_dir, args.warmup))
