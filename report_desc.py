#!/usr/bin/env python3
"""Write the HID report descriptor into configfs.

This exists because the descriptor is a binary blob and pushing it through
shell `printf` with hex escapes is where this quietly breaks: BusyBox printf
handles escapes differently from coreutils, and a single wrong byte gives you
a mouse that enumerates and then does nothing.

Absolute (default), 5-byte report:
    byte 0     buttons (3 bits) + padding
    byte 1..2  X, 0..32767
    byte 3..4  Y, 0..32767

Relative (--rel), 3-byte report, for hosts that refuse absolute pointing:
    byte 0     buttons
    byte 1     dX, signed -127..127
    byte 2     dY, signed -127..127

Usage:
    report_desc.py <path-to-report_desc> [--rel]
    report_desc.py --hex [--rel]
"""

from __future__ import annotations

import sys

ABS_DESC = bytes(
    [
        0x05, 0x01,        # Usage Page (Generic Desktop)
        0x09, 0x02,        # Usage (Mouse)
        0xA1, 0x01,        # Collection (Application)
        0x09, 0x01,        #   Usage (Pointer)
        0xA1, 0x00,        #   Collection (Physical)
        0x05, 0x09,        #     Usage Page (Button)
        0x19, 0x01,        #     Usage Minimum (1)
        0x29, 0x03,        #     Usage Maximum (3)
        0x15, 0x00,        #     Logical Minimum (0)
        0x25, 0x01,        #     Logical Maximum (1)
        0x95, 0x03,        #     Report Count (3)
        0x75, 0x01,        #     Report Size (1)
        0x81, 0x02,        #     Input (Data,Var,Abs)
        0x95, 0x01,        #     Report Count (1)
        0x75, 0x05,        #     Report Size (5)
        0x81, 0x03,        #     Input (Const,Var,Abs) padding
        0x05, 0x01,        #     Usage Page (Generic Desktop)
        0x09, 0x30,        #     Usage (X)
        0x09, 0x31,        #     Usage (Y)
        0x16, 0x00, 0x00,  #     Logical Minimum (0)
        0x26, 0xFF, 0x7F,  #     Logical Maximum (32767)
        0x75, 0x10,        #     Report Size (16)
        0x95, 0x02,        #     Report Count (2)
        0x81, 0x02,        #     Input (Data,Var,Abs)
        0xC0,              #   End Collection
        0xC0,              # End Collection
    ]
)

REL_DESC = bytes(
    [
        0x05, 0x01,        # Usage Page (Generic Desktop)
        0x09, 0x02,        # Usage (Mouse)
        0xA1, 0x01,        # Collection (Application)
        0x09, 0x01,        #   Usage (Pointer)
        0xA1, 0x00,        #   Collection (Physical)
        0x05, 0x09,        #     Usage Page (Button)
        0x19, 0x01,        #     Usage Minimum (1)
        0x29, 0x03,        #     Usage Maximum (3)
        0x15, 0x00,        #     Logical Minimum (0)
        0x25, 0x01,        #     Logical Maximum (1)
        0x95, 0x03,        #     Report Count (3)
        0x75, 0x01,        #     Report Size (1)
        0x81, 0x02,        #     Input (Data,Var,Abs)
        0x95, 0x01,        #     Report Count (1)
        0x75, 0x05,        #     Report Size (5)
        0x81, 0x03,        #     Input (Const,Var,Abs) padding
        0x05, 0x01,        #     Usage Page (Generic Desktop)
        0x09, 0x30,        #     Usage (X)
        0x09, 0x31,        #     Usage (Y)
        0x15, 0x81,        #     Logical Minimum (-127)
        0x25, 0x7F,        #     Logical Maximum (127)
        0x75, 0x08,        #     Report Size (8)
        0x95, 0x02,        #     Report Count (2)
        0x81, 0x06,        #     Input (Data,Var,Rel)
        0xC0,              #   End Collection
        0xC0,              # End Collection
    ]
)

DEFAULT_PATH = "/sys/kernel/config/usb_gadget/g1/functions/hid.usb0/report_desc"


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    relative = "--rel" in argv
    desc = REL_DESC if relative else ABS_DESC

    if "--hex" in argv:
        print(desc.hex(" "))
        return 0

    path = args[0] if args else DEFAULT_PATH

    # Single unbuffered write: configfs rejects a descriptor split across writes.
    with open(path, "wb", buffering=0) as handle:
        handle.write(desc)

    kind = "relative" if relative else "absolute"
    print(f"report_desc: {len(desc)} bytes, {kind}, report {3 if relative else 5} bytes -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))