#!/usr/bin/env python3
"""j5-hidmouse: the actuator. Runs on the phone, owns /dev/hidg0.

Takes target coordinates from the vision side over UDP and turns them into real
USB HID mouse reports, so the host sees a physical device and needs no software.

What lives here, and why it is not on the PC side:

  * latest-wins      the socket is drained every tick and only the newest frame
                     is used, so a link hiccup cannot build a queue of stale aims
  * one-euro filter  follows fast movement, stops trembling when the target is
                     still; a plain EMA is either laggy or shaky, never both good
  * step clamp       a detection glitch cannot teleport the cursor 1000px
  * deadband         no USB traffic when nothing moved
  * watchdog         no packet for --watchdog ms and we stop writing entirely,
                     so a crashed PC leaves the cursor parked instead of drifting

Usage (on the phone):
    ./hidmouse.py --dry-run            # print instead of writing to /dev/hidg0
    ./hidmouse.py --hz 125
    ./hidmouse.py --mode rel           # if the host refuses absolute pointing
"""

from __future__ import annotations

import argparse
import errno
import math
import os
import socket
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol  # noqa: E402

GADGET_DIR = os.environ.get("J5_GADGET_DIR", "/sys/kernel/config/usb_gadget")


def detect_report_mode(gadget: str = "g1", func: str = "hid.usb0") -> str:
    """Which descriptor is actually bound: 5 bytes absolute, 3 bytes relative.

    Reading it from configfs means the mode cannot disagree with the descriptor,
    which is the kind of mismatch that produces a cursor that twitches once and
    then does nothing at all.
    """
    path = os.path.join(GADGET_DIR, gadget, "functions", func, "report_length")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            length = int(handle.read().strip())
    except (OSError, ValueError):
        return "abs"
    return "rel" if length == protocol.HID_REL_SIZE else "abs"


class OneEuro:
    """1-euro filter: adaptive low-pass that tracks speed.

    min_cutoff low  -> smooth when standing still
    beta high       -> less smoothing when moving fast
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.012, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[float] = None
        self._dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0

    def __call__(self, x: float, dt: float) -> float:
        if self._x is None or dt <= 0:
            self._x = x
            self._dx = 0.0
            return x
        raw_dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx += a_d * (raw_dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x += a * (x - self._x)
        return self._x


class Actuator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mode = args.mode
        self.deadband = args.deadband
        self.max_step = args.max_step
        self.watchdog = args.watchdog_ms / 1000.0
        self.max_age = args.max_age_ms

        self.fx = OneEuro(args.min_cutoff, args.beta)
        self.fy = OneEuro(args.min_cutoff, args.beta)

        self.target: Optional[tuple[float, float]] = None
        self.sent = (protocol.MAX_COORD // 2, protocol.MAX_COORD // 2)
        self.primed = False
        self.last_frame_at = 0.0
        self.last_tick = 0.0
        self.last_seq: Optional[int] = None

        self.frames = 0
        self.writes = 0
        self.dropped_seq = 0
        self.dropped_age = 0
        self.last_print = 0.0

        self.fd: Optional[int] = None
        self._fd_announced = False
        if not args.dry_run:
            self._open_hidg()
            if self.fd is None:
                sys.exit(f"nao consegui abrir {args.hidg}\nrode com --dry-run pra testar sem o gadget")

    def _open_hidg(self) -> None:
        """(Re)open /dev/hidg0.

        Recreating the HID function (switching absolute <-> relative) destroys
        and recreates this node, and a process holding the old descriptor writes
        into nothing at all: no error, no movement. Reopening keeps the actuator
        alive across a mode switch.
        """
        try:
            self.fd = os.open(self.args.hidg, os.O_WRONLY | os.O_NONBLOCK)
        except OSError:
            self.fd = None
            return
        if not self._fd_announced:
            print(f"hidg {self.args.hidg} aberto", file=sys.stderr)
            self._fd_announced = True

    # --- input -----------------------------------------------------------

    def feed(self, frame: protocol.Frame, now: float) -> None:
        self.frames += 1
        # A restarted sender begins again at seq 1. Without this resync the
        # modular comparison rejects everything for up to 127 packets (~1s at
        # 125Hz) and the watchdog fires in the middle of a healthy session.
        if now - self.last_frame_at > self.watchdog:
            self.last_seq = None
        if not protocol.seq_is_newer(frame.seq, self.last_seq):
            self.dropped_seq += 1
            return
        if frame.age_ms > self.max_age:
            self.dropped_age += 1
            return
        self.last_seq = frame.seq

        if not frame.valid:
            self.target = None
            return

        if frame.delta:
            # Movimento relativo direto: sem modelo, sem saturacao. O PC mede o
            # cursor de verdade e manda so a correcao que falta.
            self.target = None
            self.last_frame_at = now
            self._apply_delta(frame.x, frame.y)
            return

        self.target = (float(frame.x), float(frame.y))
        self.last_frame_at = now

    # --- output ----------------------------------------------------------

    def _write(self, report: bytes) -> bool:
        if self.args.dry_run:
            self.writes += 1
            return True
        if self.fd is None:
            self._open_hidg()
            if self.fd is None:
                return False
        try:
            os.write(self.fd, report)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EPIPE):
                # Endpoint queue full or host gone; dropping is correct, a
                # backlog of old aims is worse than a missed frame.
                return False
            if exc.errno in (errno.ENODEV, errno.ESHUTDOWN, errno.EBADF):
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
                print("hidg caiu, reabrindo no proximo report", file=sys.stderr)
                return False
            raise
        self.writes += 1
        return True

    def _send_abs(self, x: int, y: int) -> None:
        self._write(protocol.hid_abs(x, y))
        self.sent = (x, y)
        self.primed = True

    def _apply_delta(self, dx: int, dy: int) -> None:
        """Escreve o deslocamento pedido, fatiado no tamanho do report."""
        step_x, step_y = int(dx), int(dy)
        for _ in range(64):
            if not step_x and not step_y:
                return
            chunk_x = max(-127, min(127, step_x))
            chunk_y = max(-127, min(127, step_y))
            if not self._write(protocol.hid_rel(chunk_x, chunk_y)):
                return
            step_x -= chunk_x
            step_y -= chunk_y
            cx = max(0, min(protocol.MAX_COORD, self.sent[0] + chunk_x))
            cy = max(0, min(protocol.MAX_COORD, self.sent[1] + chunk_y))
            self.sent = (cx, cy)

    def _send_rel(self, tx: float, ty: float) -> None:
        cx, cy = self.sent
        step_x = max(-self.max_step, min(self.max_step, round(tx - cx)))
        step_y = max(-self.max_step, min(self.max_step, round(ty - cy)))
        if step_x == 0 and step_y == 0:
            return
        # The relative report carries one signed byte per axis, so a large move
        # has to be split across several reports. Cap the burst so one wild
        # detection cannot monopolise the endpoint.
        for _ in range(64):
            if not step_x and not step_y:
                return
            dx = max(-127, min(127, step_x))
            dy = max(-127, min(127, step_y))
            if not self._write(protocol.hid_rel(dx, dy)):
                return
            cx += dx
            cy += dy
            self.sent = (cx, cy)
            step_x -= dx
            step_y -= dy

    def home(self) -> None:
        """Anchor the position model by walking the pointer into the top-left.

        A relative device cannot know where the cursor is, and starting from a
        guess leaves a constant offset for the whole session. Writing a large
        negative delta until the OS clamps at (0,0) is the only way to find out.
        Costs about a second, once.
        """
        written = 0
        for _ in range(self.args.rel_home):
            if not self._write(protocol.hid_rel(-127, -127)):
                break
            written += 1
        self.sent = (0, 0)
        print(f"ancorado no canto superior esquerdo ({written} reports)", file=sys.stderr)

    def tick(self, now: float) -> None:
        dt = now - self.last_tick if self.last_tick else 1.0 / max(1, self.args.hz)
        self.last_tick = now

        if self.target is None or (now - self.last_frame_at) > self.watchdog:
            self.fx.reset()
            self.fy.reset()
            # Re-acquisition must always emit one report: the deadband compares
            # against the position we believe we last sent, and after a gap that
            # belief is a guess. Without this, aiming at the exact centre of the
            # screen right after start-up produces no movement at all.
            self.primed = False
            return

        fx = self.fx(self.target[0], dt)
        fy = self.fy(self.target[1], dt)
        tx, ty = int(round(fx)), int(round(fy))

        if self.mode == "rel":
            if abs(tx - self.sent[0]) <= self.deadband and abs(ty - self.sent[1]) <= self.deadband:
                return
            self._send_rel(tx, ty)
            return

        if self.primed and abs(tx - self.sent[0]) <= self.deadband and abs(ty - self.sent[1]) <= self.deadband:
            return
        # Clamp per-report movement but keep going: the next tick continues the
        # approach, so this rate-limits the cursor instead of freezing it.
        cx, cy = self.sent
        dx = max(-self.max_step, min(self.max_step, tx - cx))
        dy = max(-self.max_step, min(self.max_step, ty - cy))
        self._send_abs(cx + dx, cy + dy)

    def report(self, now: float) -> None:
        if not self.args.stats or now - self.last_print < 2.0:
            return
        self.last_print = now
        print(
            f"frames {self.frames:6d}  writes {self.writes:6d}  "
            f"stale {self.dropped_age:4d}  reorder {self.dropped_seq:4d}  "
            f"pos {self.sent[0]:5d},{self.sent[1]:5d}",
            file=sys.stderr,
            flush=True,
        )


def drain(sock: socket.socket) -> Optional[protocol.Frame]:
    newest: Optional[protocol.Frame] = None
    while True:
        try:
            data, _ = sock.recvfrom(2048)
        except BlockingIOError:
            return newest
        except OSError:
            return newest
        frame = protocol.unpack(data)
        if frame is not None:
            newest = frame


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="J5 USB HID actuator")
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hidg", default="/dev/hidg0")
    p.add_argument("--hz", type=float, default=125.0, help="report loop rate")
    p.add_argument("--mode", choices=("auto", "abs", "rel"), default="auto",
                   help="auto le o report_length do gadget")
    p.add_argument("--deadband", type=int, default=3, help="HID units (32767 = full screen)")
    p.add_argument("--max-step", type=int, default=1600, help="max HID units per report")
    p.add_argument("--rel-home", type=int, default=200,
                   help="no modo rel: reports para ancorar o cursor no canto (0 = desliga)")
    p.add_argument("--min-cutoff", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.012)
    p.add_argument("--watchdog-ms", type=int, default=300)
    p.add_argument("--max-age-ms", type=int, default=250)
    p.add_argument("--dry-run", action="store_true", help="nao escreve em /dev/hidg0")
    p.add_argument("--print-every", type=int, default=25, help="com --dry-run, imprime 1 a cada N reports")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--duration", type=float, default=0.0, help="sai depois de N segundos (testes)")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "auto":
        args.mode = detect_report_mode()
        print(f"modo {args.mode} (lido do report_length do gadget)", file=sys.stderr)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.setblocking(False)

    actuator = Actuator(args)
    if args.mode == "rel" and args.rel_home > 0:
        actuator.home()
    period = 1.0 / args.hz
    started = time.monotonic()
    deadline = started + args.duration if args.duration else None
    next_tick = started
    printed = 0

    print(
        f"j5-hidmouse: udp {args.bind}:{args.port} -> {args.hidg} ({args.mode}), "
        f"{args.hz:.0f} Hz, watchdog {args.watchdog_ms}ms",
        file=sys.stderr,
        flush=True,
    )

    try:
        while True:
            now = time.monotonic()
            frame = drain(sock)
            if frame is not None:
                actuator.feed(frame, now)

            before = actuator.writes
            actuator.tick(now)
            if args.dry_run and actuator.writes != before:
                printed += 1
                if printed % max(1, args.print_every) == 0:
                    x, y = actuator.sent
                    print(f"{now - started:8.3f}s  x={x:5d} y={y:5d}  ({x / 327.67:5.1f}% , {y / 327.67:5.1f}%)")
            actuator.report(now)

            if deadline is not None and now >= deadline:
                break

            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass

    if args.dry_run:
        print(
            f"-- frames={actuator.frames} writes={actuator.writes} "
            f"stale={actuator.dropped_age} reorder={actuator.dropped_seq} "
            f"final={actuator.sent}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())