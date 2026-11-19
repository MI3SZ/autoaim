"""Turns screen coordinates into datagrams for the J5.

Also carries the standalone test patterns, because the first thing you want after
wiring the gadget is to prove the actuator works before any vision is involved:
if `--pattern circle` does not make the cursor draw a circle, the problem is not
the camera.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol  # noqa: E402


class Sender:
    def __init__(self, host: str, port: int = 5555):
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def send(self, x: int, y: int, valid: bool = True, click: bool = False, age_ms: int = 0) -> None:
        self.seq = (self.seq + 1) & 0xFF
        self.socket.sendto(protocol.pack(x, y, valid, click, age_ms, self.seq), self.address)
        self.sent += 1

    def send_screen(self, px: float, py: float, rect, click: bool = False, age_ms: int = 0) -> None:
        ax, ay = protocol.to_abs(px, py, rect)
        self.send(ax, ay, True, click, age_ms)

    def park(self) -> None:
        """Explicit 'no target': the actuator stops moving and the watchdog agrees."""
        self.send(0, 0, False, False, 0)

    def close(self) -> None:
        self.socket.close()


def pattern_circle(t: float, rect) -> tuple[float, float]:
    ox, oy, w, h = rect
    phase = t * 2 * math.pi / 6.0
    radius = min(w, h) * 0.32
    return ox + w / 2 + radius * math.cos(phase), oy + h / 2 + radius * math.sin(phase)


def pattern_sweep(t: float, rect) -> tuple[float, float]:
    ox, oy, w, h = rect
    phase = 0.5 - 0.5 * math.cos(t * 2 * math.pi / 4.0)
    return ox + w * (0.05 + 0.90 * phase), oy + h * 0.5


def pattern_corners(t: float, rect) -> tuple[float, float]:
    ox, oy, w, h = rect
    corners = ((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95))
    index = int(t / 1.2) % len(corners)
    fx, fy = corners[index]
    return ox + w * fx, oy + h * fy


PATTERNS = {"circle": pattern_circle, "sweep": pattern_sweep, "corners": pattern_corners}


def parse_rect(text: str):
    parts = [int(float(v)) for v in text.replace("x", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("use x,y,w,h")
    return tuple(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="envia alvos pro J5 (teste do atuador)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="circle")
    parser.add_argument("--rect", type=parse_rect, default=None, help="desktop virtual x,y,w,h")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args(argv)

    rect = args.rect
    if rect is None:
        rect = default_rect()
        print(f"rect do desktop detectado: {rect}", file=sys.stderr)

    sender = Sender(args.host, args.port)
    generate = PATTERNS[args.pattern]
    period = 1.0 / args.hz
    start = time.monotonic()
    next_t = start
    print(f"padrao {args.pattern} por {args.duration}s a {args.hz:.0f} Hz -> {args.host}:{args.port}", file=sys.stderr)

    while True:
        now = time.monotonic() - start
        if now >= args.duration:
            break
        px, py = generate(now, rect)
        sender.send_screen(px, py, rect)
        next_t += period
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.monotonic()

    sender.close()
    print(f"{sender.sent} pacotes enviados", file=sys.stderr)
    return 0


def default_rect():
    """Virtual desktop bounds. X11 is easy; elsewhere fall back to the primary."""
    try:
        from Xlib import display as xdisplay

        d = xdisplay.Display()
        screen = d.screen()
        return (
            int(screen.x),
            int(screen.y),
            int(screen.width_in_pixels),
            int(screen.height_in_pixels),
        )
    except Exception:
        pass
    try:
        import mss

        factory = getattr(mss, "MSS", None) or getattr(mss, "mss", None)
        with factory() as sct:
            m = sct.monitors[0]
            return (int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"]))
    except Exception:
        pass
    print("aviso: nao detectei o desktop, usando 1920x1080 em 0,0; passe --rect", file=sys.stderr)
    return (0, 0, 1920, 1080)


if __name__ == "__main__":
    sys.exit(main())