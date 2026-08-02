#!/usr/bin/env python3
"""
Replay frames through the line-following pipeline and compare the output to a
recorded baseline.

WHY
---
Step 3 of this project refactors the line-following code into shared utilities so
lane following can reuse it. "Line following must behave identically afterwards"
is easy to say and hard to know. So: record the exact steering and throttle
sequence BEFORE the refactor, then check it after. Any behavior change shows up
as a numeric difference on a specific frame instead of a vague feeling that the
car drives differently.

It is also a fast way to sanity-check a tuning change off the car: record a
baseline, edit params, and see which frames change.

FRAMES
------
Real frames are best. Capture them on the car with:

    python scripts/replay_frames.py --capture 40 --frames tests/data/line_frames

If no frame directory is given, a deterministic synthetic sequence is generated
that walks the pipeline through every state: acquiring, tracking, drifting,
losing the line to foliage, coasting, slowing, stopping and re-acquiring.

USAGE
-----
    # record a baseline
    python scripts/replay_frames.py --record tests/data/line_golden.json

    # check the current code against it
    python scripts/replay_frames.py --check tests/data/line_golden.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from donkeycar.parts.lane_following.params import Params
from donkeycar.parts.lane_following.strategies import LineFollowStrategy

WIDTH = 426
HEIGHT = 240

# Same colors as the tests, chosen to match real track materials.
YELLOW_TAPE = (30, 200, 220)
PALE_FOLIAGE = (150, 200, 160)
ASPHALT = (60, 60, 60)


def synthetic_sequence():
    """
    A deterministic 60-frame drive that exercises every pipeline state.

    Seeded, so it produces byte-identical frames on every machine and run --
    which is what makes the golden comparison meaningful.
    """
    rng = np.random.default_rng(1234)
    frames = []

    def make_frame(line_x, foliage, dashed_gap=False):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:] = ASPHALT
        # A little sensor noise, so the pipeline is not fed impossibly clean input.
        noise = rng.integers(-6, 6, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if foliage:
            band = frame[: int(0.55 * HEIGHT), :]
            band[:] = PALE_FOLIAGE
            leaf_noise = rng.integers(-25, 25, size=band.shape, dtype=np.int16)
            frame[: int(0.55 * HEIGHT), :] = np.clip(
                band.astype(np.int16) + leaf_noise, 0, 255).astype(np.uint8)

        if line_x is not None:
            top = int(0.55 * HEIGHT)
            for y in range(top, HEIGHT):
                if dashed_gap and ((y // 14) % 2 == 0):
                    continue  # a gap, as in discontinuous tape
                # The line converges toward the vanishing point with distance.
                shrink = (y - top) / max(HEIGHT - top, 1)
                half = int(6 + 8 * shrink)
                x = int(line_x)
                frame[y, max(0, x - half):min(WIDTH, x + half)] = YELLOW_TAPE
        return frame

    # 1-10: acquire a centered line, no foliage
    for _ in range(10):
        frames.append(make_frame(WIDTH * 0.5, foliage=False))
    # 11-20: foliage appears at the horizon; line still centered
    for _ in range(10):
        frames.append(make_frame(WIDTH * 0.5, foliage=True))
    # 21-32: line drifts right (a curve)
    for i in range(12):
        frames.append(make_frame(WIDTH * (0.5 + 0.03 * i), foliage=True))
    # 33-40: dashed tape, to test gap bridging
    for _ in range(8):
        frames.append(make_frame(WIDTH * 0.85, foliage=True, dashed_gap=True))
    # 41-70: line gone, only foliage. Long enough to walk all the way through
    # coast -> slow -> stop (COAST_FRAMES + SLOW_FRAMES + margin).
    for _ in range(30):
        frames.append(make_frame(None, foliage=True))
    # 71-78: line reappears centered -- re-acquisition after a long loss
    for _ in range(8):
        frames.append(make_frame(WIDTH * 0.5, foliage=True))

    return frames


def load_frames(frame_dir):
    """Load PNG frames from a directory, sorted by name."""
    import cv2
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not paths:
        return None
    frames = [cv2.imread(path) for path in paths]
    if any(frame is None for frame in frames):
        raise SystemExit(f"could not read every PNG in {frame_dir}")
    print(f"loaded {len(frames)} frames from {frame_dir}")
    return frames


def capture_frames(frame_dir, count):
    """Grab frames from the OAK-D and save them as PNGs. Run this on the car."""
    import cv2
    from donkeycar.parts.oak_d import OakD

    os.makedirs(frame_dir, exist_ok=True)
    camera = OakD(width=WIDTH, height=HEIGHT, enable_rgb=True, enable_depth=False)
    try:
        for _ in range(10):  # let auto-exposure settle
            camera.run()
        for index in range(count):
            frame, _ = camera.run()
            if frame is None:
                continue
            path = os.path.join(frame_dir, f"frame_{index:04d}.png")
            cv2.imwrite(path, frame)
        print(f"captured {count} frames into {frame_dir}")
    finally:
        camera.shutdown()


def replay(frames):
    """Run every frame through a fresh pipeline; return the output per frame."""
    pipeline = LineFollowStrategy(Params(None))
    records = []
    for index, frame in enumerate(frames):
        result = pipeline.process(frame)
        records.append({
            "frame": index,
            "steering": round(float(result.steering), 6),
            "throttle": round(float(result.throttle), 6),
            "state": result.state.value,
            "offset": round(float(result.offset), 6),
            "line_x": (None if result.line_x is None
                       else round(float(result.line_x), 3)),
            "blob_count": int(result.detection.blob_count),
            "area_frac": round(float(result.detection.area_frac), 6),
        })
    return records


def summarize(records):
    states = {}
    for record in records:
        states[record["state"]] = states.get(record["state"], 0) + 1
    print("states visited: " + ", ".join(
        f"{name}={count}" for name, count in sorted(states.items())))
    return states


def compare(current, golden):
    """Report every differing frame. Returns True when identical."""
    if len(current) != len(golden):
        print(f"FAIL: frame count changed: {len(golden)} -> {len(current)}")
        return False

    keys = ["steering", "throttle", "state", "offset", "line_x",
            "blob_count", "area_frac"]
    differences = []
    for now, before in zip(current, golden):
        for key in keys:
            if now[key] != before[key]:
                differences.append(
                    f"  frame {now['frame']:3d} {key}: {before[key]} -> {now[key]}")

    if differences:
        print(f"FAIL: {len(differences)} difference(s) from the baseline:")
        for line in differences[:40]:
            print(line)
        if len(differences) > 40:
            print(f"  ... and {len(differences) - 40} more")
        return False

    print(f"PASS: all {len(current)} frames match the baseline exactly.")
    return True


def main(args):
    if args.capture:
        capture_frames(args.frames or "tests/data/line_frames", args.capture)
        return 0

    frames = None
    if args.frames:
        frames = load_frames(args.frames)
        if frames is None:
            print(f"no PNGs in {args.frames}, using the synthetic sequence")
    if frames is None:
        frames = synthetic_sequence()
        print(f"using the deterministic synthetic sequence ({len(frames)} frames)")

    records = replay(frames)
    states = summarize(records)

    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        with open(args.record, "w", encoding="utf-8") as handle:
            json.dump({"frame_count": len(records), "records": records},
                      handle, indent=1)
        print(f"baseline written to {args.record}")

        missing = {"tracking", "coasting", "stopped"} - set(states)
        if missing:
            print(f"NOTE: these states were never reached: {sorted(missing)}")
            print("A baseline that never loses the line cannot detect a")
            print("regression in the lost-line behavior.")
        return 0

    if args.check:
        with open(args.check, encoding="utf-8") as handle:
            golden = json.load(handle)["records"]
        return 0 if compare(records, golden) else 1

    # No mode given: just print what happened.
    for record in records:
        print(f"  {record['frame']:3d} {record['state']:<9} "
              f"steer={record['steering']:+.3f} throttle={record['throttle']:.3f} "
              f"blobs={record['blob_count']}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", help="directory of PNG frames to replay")
    parser.add_argument("--record", help="write a baseline JSON to this path")
    parser.add_argument("--check", help="compare against this baseline JSON")
    parser.add_argument("--capture", type=int, metavar="N",
                        help="capture N frames from the OAK-D into --frames")
    sys.exit(main(parser.parse_args()))
