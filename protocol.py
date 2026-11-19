"""Wire format between the vision side (Fedora) and the J5 actuator.

One datagram, 10 bytes, little endian:

    0  1   magic    A5 5A
    2      flags    bit0 = target valid, bit1 = click
    3  4   x        u16, 0..32767 across the virtual desktop
    5  6   y        u16, 0..32767 across the virtual desktop
    7      age      age of the frame in ms, capped at 255
    8      seq      u8, wraps at 256
    9      crc8     poly 0x07 over bytes 0..8

The age field is what makes a jittery link usable: the actuator drops
anything older than its own threshold instead of replaying a stale point.

Same file is imported by the phone (hidmouse.py) and by the vision side.
"""

from __future__ import annotations

import struct
from typing import NamedTuple, Optional

MAGIC = b"\xA5\x5A"
MAX_COORD = 32767

FLAG_VALID = 0x01
FLAG_CLICK = 0x02

# 6 fixed fields: magic(2s) flags(B) x(H) y(H) age(B) seq(B)
_HEAD = struct.Struct("<2sBHHBB")
FRAME_SIZE = _HEAD.size + 1  # 9 + crc

# HID absolute mouse report, see report_desc.py (buttons, x u16, y u16)
HID_ABS_SIZE = 5
# HID relative mouse report (buttons, dx i8, dy i8)
HID_REL_SIZE = 3


def crc8(data: bytes, crc: int = 0) -> int:
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


class Frame(NamedTuple):
    x: int
    y: int
    valid: bool
    click: bool
    age_ms: int
    seq: int


def _clamp(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def pack(
    x: int,
    y: int,
    valid: bool = True,
    click: bool = False,
    age_ms: int = 0,
    seq: int = 0,
) -> bytes:
    flags = (FLAG_VALID if valid else 0) | (FLAG_CLICK if click else 0)
    head = _HEAD.pack(
        MAGIC,
        flags,
        _clamp(int(x), 0, MAX_COORD),
        _clamp(int(y), 0, MAX_COORD),
        _clamp(int(age_ms), 0, 255),
        int(seq) & 0xFF,
    )
    return head + bytes((crc8(head),))


def unpack(data: bytes) -> Optional[Frame]:
    if len(data) != FRAME_SIZE:
        return None
    if crc8(data[:9]) != data[9]:
        return None
    magic, flags, x, y, age, seq = _HEAD.unpack(data[:9])
    if magic != MAGIC:
        return None
    return Frame(
        x=x,
        y=y,
        valid=bool(flags & FLAG_VALID),
        click=bool(flags & FLAG_CLICK),
        age_ms=age,
        seq=seq,
    )


def seq_is_newer(candidate: int, reference: Optional[int]) -> bool:
    """Modular comparison so a wrap at 255 does not look like a rewind."""
    if reference is None:
        return True
    delta = (candidate - reference) & 0xFF
    return 0 < delta < 128


def to_abs(px: float, py: float, rect) -> tuple[int, int]:
    """Screen pixel -> 0..32767 over the rectangle (x, y, w, h) of the desktop.

    The HID range always covers the whole virtual desktop, so passing a single
    monitor's rectangle here is how you get a cursor that stays on that monitor.
    """
    ox, oy, ow, oh = rect
    ax = (px - ox) / max(1.0, ow) * MAX_COORD
    ay = (py - oy) / max(1.0, oh) * MAX_COORD
    return _clamp(int(round(ax)), 0, MAX_COORD), _clamp(int(round(ay)), 0, MAX_COORD)


def hid_abs(x: int, y: int, buttons: int = 0) -> bytes:
    """5-byte report matching the absolute report descriptor."""
    x = _clamp(int(x), 0, MAX_COORD)
    y = _clamp(int(y), 0, MAX_COORD)
    return bytes((buttons & 0x07, x & 0xFF, (x >> 8) & 0xFF, y & 0xFF, (y >> 8) & 0xFF))


def hid_rel(dx: int, dy: int, buttons: int = 0) -> bytes:
    """3-byte report matching the relative report descriptor (signed bytes)."""
    dx = _clamp(int(dx), -127, 127)
    dy = _clamp(int(dy), -127, 127)
    return bytes((buttons & 0x07, dx & 0xFF, dy & 0xFF))