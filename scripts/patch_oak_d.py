#!/usr/bin/env python3
"""
Apply the OAK-D full-FOV camera patch to the INSTALLED donkeycar package.

Why this exists
---------------
The stock OakD part streams the camera's `video` output, which is hard-cropped to
16:9 regardless of setResolution(). That crop throws away the bottom of the
sensor's native 4:3 field of view -- the near-ground strip a line follower needs
-- so the car loses the line on turns. The fix is to stream the `isp` output
instead and squash it to 426x240.

The patched file lives in this repo at donkeycar/parts/oak_d.py. If the Pi runs
an editable install of this repo, that IS the installed file and there is nothing
to do. If the Pi has its own site-packages copy, the patch does not travel with
the repo and must be re-applied after every fresh install or environment rebuild.
This script does that, idempotently.

Usage:
    python scripts/patch_oak_d.py              # apply (no-op if already current)
    python scripts/patch_oak_d.py --dry-run    # report what would happen
    python scripts/patch_oak_d.py --revert     # restore the .bak backup
    python scripts/patch_oak_d.py --status     # just report the current state

Run it with the SAME python that runs manage.py, e.g.
    ~/env/bin/python scripts/patch_oak_d.py
otherwise it will find the wrong donkeycar.
"""

import argparse
import filecmp
import importlib.util
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SOURCE = os.path.join(REPO_ROOT, "donkeycar", "parts", "oak_d.py")

# Markers that only exist in the patched file. Used to report status when the
# installed copy is neither identical to ours nor obviously stock.
PATCH_MARKERS = ("cam_rgb.isp.link", "ISP_CAPTURE_CANDIDATES", "_fit_to_output_size")


def find_installed_oak_d():
    """
    Locate oak_d.py in whichever donkeycar this interpreter would import.

    Deliberately does not `import donkeycar`, which is slow and can fail on a
    machine without the hardware deps; find_spec only reads metadata.
    """
    spec = importlib.util.find_spec("donkeycar")
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = list(spec.submodule_search_locations)[0]
    return os.path.join(package_dir, "parts", "oak_d.py")


def describe(installed):
    """Return (state, message) for the installed file."""
    if installed is None:
        return "missing", "donkeycar is not importable by this interpreter."
    if not os.path.exists(installed):
        return "missing", f"No oak_d.py at {installed}"
    if os.path.realpath(installed) == os.path.realpath(SOURCE):
        return "editable", "Installed donkeycar IS this repo (editable install)."
    if filecmp.cmp(SOURCE, installed, shallow=False):
        return "current", "Installed copy already matches this repo."

    with open(installed, "r", encoding="utf-8") as handle:
        text = handle.read()
    if all(marker in text for marker in PATCH_MARKERS):
        return "patched-different", (
            "Installed copy looks patched but differs from this repo "
            "(probably an older version of the patch)."
        )
    return "stock", "Installed copy is unpatched (stock donkeycar)."


def main(dry_run, revert, status_only):
    if not os.path.exists(SOURCE):
        print(f"ERROR: patched source not found at {SOURCE}")
        return 1

    installed = find_installed_oak_d()
    state, message = describe(installed)

    print(f"python    : {sys.executable}")
    print(f"repo copy : {SOURCE}")
    print(f"installed : {installed or '(not found)'}")
    print(f"state     : {state} -- {message}")
    print()

    if status_only:
        return 0 if state in ("editable", "current") else 1

    if state == "missing":
        print("Nothing to do. Install donkeycar first, or run this with the")
        print("interpreter that runs manage.py (e.g. ~/env/bin/python).")
        return 1

    backup = installed + ".bak"

    if revert:
        if not os.path.exists(backup):
            print(f"ERROR: no backup at {backup}, cannot revert.")
            return 1
        if dry_run:
            print(f"[dry-run] would restore {backup} -> {installed}")
            return 0
        shutil.copy2(backup, installed)
        print(f"Reverted: restored {backup} -> {installed}")
        return 0

    if state == "editable":
        print("Already patched -- the installed package is this repo, so editing")
        print("the repo file is the patch. Nothing to do.")
        return 0

    if state == "current":
        print("Already patched and up to date. Nothing to do.")
        return 0

    # state is "stock" or "patched-different": copy ours over it.
    if dry_run:
        print(f"[dry-run] would back up {installed} -> {backup}")
        print(f"[dry-run] would copy    {SOURCE} -> {installed}")
        return 0

    if not os.access(installed, os.W_OK):
        print(f"ERROR: no write permission on {installed}")
        print("Re-run with the interpreter that owns that environment, or with sudo.")
        return 1

    if not os.path.exists(backup):
        shutil.copy2(installed, backup)
        print(f"Backed up original -> {backup}")
    else:
        print(f"Keeping existing backup at {backup}")

    shutil.copy2(SOURCE, installed)
    print(f"Patched: {SOURCE} -> {installed}")
    print()
    print("Verify with:  python scripts/oakd_color_check.py")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without changing it")
    parser.add_argument("--revert", action="store_true",
                        help="restore the .bak backup")
    parser.add_argument("--status", action="store_true",
                        help="report state only; exit 0 if already patched")
    args = parser.parse_args()
    sys.exit(main(args.dry_run, args.revert, args.status))
