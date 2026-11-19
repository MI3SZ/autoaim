#!/usr/bin/env python3
"""Record frames to disk with timestamps.

Two jobs:

  1. Calibration input. Run this while the trainer's `calib` mode shows its nine
     white dots, then feed the directory to calibrate.py and it pairs camera
     observations with known screen positions by wall clock.

  2. Debugging. Tuning a colour threshold against a live camera means chasing a
     moving target while looking at two windows; tuning against a recording is
     reproducible, which is the difference between ten minutes and a whole night.

    python3 vision/record.py --source http --url http://192.168.42.1:8080 \
        --hz 20 --duration 15 --out rec-calib
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import source as vsource  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="grava frames com timestamp")
    parser.add_argument("--source", choices=("x11", "http", "mjpeg", "v4l2", "test"), default="http")
    parser.add_argument("--url", default="http://192.168.42.1:8080")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--out", default="rec")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    try:
        src = vsource.make_source(
            args.source,
            url=args.url,
            monitor=args.monitor,
            scale=args.scale,
            device=args.device,
        )
    except RuntimeError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    # The region is what makes a screen recording replayable without --rect: the
    # frames already know which screen area they cover, so persist it.
    meta = {
        "source": args.source,
        "region": list(src.region) if src.region else None,
        "hz": args.hz,
        "url": args.url if args.source in ("http", "mjpeg") else None,
        "device": args.device if args.source == "v4l2" else None,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    period = 1.0 / args.hz
    start = time.monotonic()
    deadline = start + args.duration
    next_tick = start
    saved = 0
    failed = 0

    index_path = os.path.join(args.out, "index.csv")
    with open(index_path, "w", encoding="utf-8", newline="") as index:
        writer = csv.writer(index)
        writer.writerow(["epoch", "mono", "file"])

        print(f"gravando {args.duration:.0f}s a {args.hz:.0f} Hz em {args.out}/", file=sys.stderr)
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            try:
                grab = src.grab()
            except RuntimeError as exc:
                failed += 1
                print(f"  frame perdido: {exc}", file=sys.stderr)
                # Still wait out the period: without this the loop spins as fast
                # as the failing read allows and floods the terminal.
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
                continue

            name = f"f{saved:05d}.jpg"
            # epoch is the bridge to the browser's Date.now(); mono is for local
            # ordering. The trainer exports both for the same reason.
            epoch = time.time()
            path = os.path.join(args.out, name)
            if not cv2.imwrite(path, grab.image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                failed += 1
                print(f"  nao consegui gravar {path} (disco cheio?)", file=sys.stderr)
                continue
            writer.writerow([f"{epoch:.6f}", f"{grab.ts:.6f}", name])
            saved += 1
            if saved % 20 == 0:
                print(f"  {saved} frames", file=sys.stderr)

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    src.close()
    print(f"pronto: {saved} frames, {failed} perdidos, indice em {index_path}", file=sys.stderr)
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())