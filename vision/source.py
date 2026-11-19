"""Frame sources. Everything downstream is source-agnostic on purpose.

x11   fast path, screen capture, pixel-exact, no calibration needed
http  the camera path: poll a phone camera snapshot over HTTP (lowest latency
      of the phone options, because there is no encoder buffer to sit in)
mjpeg the classic MJPEG stream, higher and more variable latency
test  synthetic frames, so the detector can be exercised without hardware
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from typing import NamedTuple, Optional

import cv2
import numpy as np

# (left, top, width, height) of what the sensor sees, in screen pixels.
# None for a camera: its pixels only mean something after calibration.
Rect = tuple[int, int, int, int]


class Grab(NamedTuple):
    image: np.ndarray  # BGR
    ts: float  # time.monotonic() at the moment the frame was complete
    region: Optional[Rect]  # screen rect covered by the image, if it is a screen


def _fourcc_value(code: str) -> int:
    """cv2 renamed VideoWriter_fourcc across versions; accept either."""
    code = code[:4]
    for owner in (cv2, cv2.VideoWriter):
        fn = getattr(owner, "fourcc", None) or getattr(owner, "VideoWriter_fourcc", None)
        if callable(fn):
            return fn(*code)
    return int.from_bytes(code.encode("ascii"), "little")


class Source:
    region: Optional[Rect] = None

    def grab(self) -> Grab:
        raise NotImplementedError

    def close(self) -> None:
        pass


class X11Source(Source):
    """Screen capture through X11. Wayland sessions return black frames, so we
    refuse up front instead of letting it look like the detector is broken."""

    def __init__(self, monitor: int = 1, scale: float = 1.0):
        import mss as mss_module  # imported here so --source http works without it

        session = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session == "wayland":
            raise RuntimeError(
                "sessao Wayland: XGetImage devolve preto. Entre em 'GNOME on Xorg' "
                "ou use --source http com a camera."
            )
        # mss 10 renamed the class; older releases only have the lowercase alias.
        factory = getattr(mss_module, "MSS", None) or getattr(mss_module, "mss", None)
        self._mss = factory()
        monitors = self._mss.monitors
        if monitor >= len(monitors):
            raise RuntimeError(f"monitor {monitor} nao existe, tenho {len(monitors) - 1}")
        self._monitor = monitors[monitor]
        self._scale = max(0.05, float(scale))
        self.region = (
            int(self._monitor["left"]),
            int(self._monitor["top"]),
            int(self._monitor["width"]),
            int(self._monitor["height"]),
        )

    def virtual_rect(self) -> Rect:
        """Union of all monitors: the HID range always spans this, not one screen."""
        m = self._mss.monitors[0]
        return (int(m["left"]), int(m["top"]), int(m["width"]), int(m["height"]))

    def grab(self) -> Grab:
        shot = self._mss.grab(self._monitor)
        frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        image = np.ascontiguousarray(frame[:, :, :3])  # BGR, no copy of the alpha
        ts = time.monotonic()
        if not self._validated:
            self._validated = True
            if float(image.std()) < 1.0:
                raise RuntimeError(
                    "frame totalmente preto: nesta sessao o XGetImage nao ve a tela "
                    "(Wayland). Use 'GNOME on Xorg', um Xephyr aninhado, ou --source v4l2 "
                    "com a camera virtual do OBS"
                )
        if self._scale != 1.0:
            image = cv2.resize(image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        return Grab(image, ts, self.region)

    def close(self) -> None:
        self._mss.close()


class HttpJpegSource(Source):
    """Poll a single snapshot endpoint.

    A snapshot endpoint beats the MJPEG stream for *latency*: the stream keeps a
    buffer between the sensor and you, so a 25 Hz stream can hand you a frame
    that is 300 ms old while your socket read is instant. Polling a fresh JPEG
    costs a TCP round trip and gives you the newest frame every time.
    """

    CANDIDATES = ("/shot.jpg", "/photo.jpg", "/image.jpg", "/jpeg", "/?action=snapshot")

    def __init__(self, url: str, username: str = "", password: str = "", timeout: float = 2.0):
        self.timeout = timeout
        self._auth = ""
        if username:
            import base64

            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self._auth = f"Basic {token}"

        if url.rstrip("/").endswith((".jpg", ".jpeg", ".cgi")) or "snapshot" in url:
            self.url = url
        else:
            self.url = self._probe(url.rstrip("/"))
        self.dropped = 0

    def _fetch(self, url: str) -> bytes:
        headers = {"User-Agent": "j5-vision/1"}
        if self._auth:
            headers["Authorization"] = self._auth
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _probe(self, base: str) -> str:
        errors = []
        for suffix in self.CANDIDATES:
            candidate = base + suffix
            try:
                data = self._fetch(candidate)
            except (urllib.error.URLError, OSError) as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            if data[:2] == b"\xff\xd8":
                print(f"source: snapshot em {candidate}", file=sys.stderr)
                return candidate
            errors.append(f"{candidate}: nao e JPEG")
        raise RuntimeError("nenhum endpoint de snapshot respondeu:\n  " + "\n  ".join(errors))

    def grab(self) -> Grab:
        data = self._fetch(self.url)
        ts = time.monotonic()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self.dropped += 1
            raise RuntimeError("JPEG invalido (frame descartado)")
        return Grab(image, ts, None)


class MjpegSource(Source):
    def __init__(self, url: str):
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"nao abri o stream MJPEG em {url}")
        self.dropped = 0

    def grab(self) -> Grab:
        ok, image = self._cap.read()
        ts = time.monotonic()
        if not ok or image is None:
            self.dropped += 1
            raise RuntimeError("frame MJPEG perdido")
        return Grab(image, ts, None)

    def close(self) -> None:
        self._cap.release()


class TestSource(Source):
    """Synthetic moving square: proves detect -> calibrate -> send without hardware."""

    def __init__(self, width: int = 640, height: int = 360, color: tuple[int, int, int] = (208, 0, 208)):
        self.width = width
        self.height = height
        self.color = color
        self._t0 = time.monotonic()

    def grab(self) -> Grab:
        t = time.monotonic() - self._t0
        image = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
        cx = int(self.width / 2 + self.width * 0.30 * np.cos(t * 1.3))
        cy = int(self.height / 2 + self.height * 0.30 * np.sin(t * 1.9))
        size = 28
        image[cy - size : cy + size, cx - size : cx + size] = self.color
        return Grab(image, time.monotonic(), None)


class V4L2Source(Source):
    """Any V4L2 capture device: a real webcam, a phone presenting itself as a UVC
    camera, or a v4l2loopback device fed by ffmpeg (a fake camera with no camera).

    BUFFERSIZE=1 matters more than it looks: the default driver queue holds
    several frames, and VideoCapture.read() then hands you frames that are 3-5
    periods old with a perfectly instant return. Chasing "why is my latency 200ms
    when capture takes 5ms" ends here.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        fourcc: str = "",
    ):
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"nao abri {device} (v4l2-ctl --list-devices mostra o que existe)")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, _fourcc_value(fourcc))
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.dropped = 0

    def grab(self) -> Grab:
        ok, image = self._cap.read()
        ts = time.monotonic()
        if not ok or image is None:
            self.dropped += 1
            raise RuntimeError("frame v4l2 perdido")
        return Grab(image, ts, None)

    def close(self) -> None:
        self._cap.release()


class ReplaySource(Source):
    """Replays a record.py dump at its original pacing.

    This is what removes the camera from the development loop: record 20 seconds
    once, then tune thresholds, calibration and filters offline, reproducibly,
    forever. Pacing follows the original timestamps so dropped frames and jitter
    are still there instead of being averaged away.
    """

    def __init__(self, directory: str, loop: bool = True, fps: float = 0.0):
        import csv
        import json

        index_path = os.path.join(directory, "index.csv")
        if not os.path.exists(index_path):
            raise RuntimeError(f"{index_path} nao existe (grave antes com vision/record.py)")
        self.dir = directory
        self.loop = loop
        self.fps = fps
        self.frames: list[tuple[float, str]] = []
        with open(index_path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.frames.append((float(row["epoch"]), row["file"]))
        if not self.frames:
            raise RuntimeError("gravacao vazia")

        meta_path = os.path.join(directory, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            region = meta.get("region")
            self.region = tuple(region) if region else None

        self._index = 0
        self._base_epoch: float | None = None
        self._start_mono = 0.0

    def grab(self) -> Grab:
        if self._index >= len(self.frames):
            if not self.loop:
                raise StopIteration("fim da gravacao")
            self._index = 0
            self._base_epoch = None

        epoch, name = self.frames[self._index]
        if self._base_epoch is None:
            self._base_epoch = epoch
            self._start_mono = time.monotonic()

        offset = self._index / self.fps if self.fps else (epoch - self._base_epoch)
        target = self._start_mono + offset
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        image = cv2.imread(os.path.join(self.dir, name), cv2.IMREAD_COLOR)
        self._index += 1
        if image is None:
            raise RuntimeError(f"frame ilegivel: {name}")
        return Grab(image, time.monotonic(), self.region)


def make_source(spec: str, **kwargs) -> Source:
    if spec == "x11":
        return X11Source(monitor=kwargs.get("monitor", 1), scale=kwargs.get("scale", 1.0))
    if spec == "http":
        return HttpJpegSource(
            kwargs["url"],
            username=kwargs.get("username", ""),
            password=kwargs.get("password", ""),
            timeout=kwargs.get("timeout", 2.0),
        )
    if spec == "mjpeg":
        return MjpegSource(kwargs["url"])
    if spec == "v4l2":
        return V4L2Source(
            kwargs.get("device", "/dev/video0"),
            width=kwargs.get("width", 0),
            height=kwargs.get("height", 0),
            fps=kwargs.get("fps", 0.0),
            fourcc=kwargs.get("fourcc", ""),
        )
    if spec == "replay":
        return ReplaySource(
            kwargs["replay_dir"],
            loop=not kwargs.get("no_loop", False),
            fps=kwargs.get("replay_fps", 0.0),
        )
    if spec == "test":
        return TestSource()
    raise ValueError(f"fonte desconhecida: {spec}")