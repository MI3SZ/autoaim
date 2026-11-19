#!/usr/bin/env python3
"""Main loop: capture -> detect -> map to screen -> send to the J5.

    python3 vision/run.py --source x11 --color d000d0 --host 169.254.1.21
    python3 vision/run.py --source http --url http://192.168.42.1:8080 \
        --model color.json --calib calib.json --rect 0,0,1920,1080 --preview
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import source as vsource  # noqa: E402
from calibrate import Calibration  # noqa: E402
from detect import ColorDetector, Detection  # noqa: E402
from sender import Sender, default_rect, parse_rect  # noqa: E402


def parse_hex_color(text: str) -> tuple[int, int, int]:
    value = text.strip().lstrip("#")
    if len(value) != 6:
        raise argparse.ArgumentTypeError("cor no formato rrggbb")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)  # BGR


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.src = vsource.make_source(
            args.source,
            url=args.url,
            username=args.user,
            password=args.password,
            monitor=args.monitor,
            scale=args.scale,
            device=args.device,
            fourcc=args.fourcc,
            width=args.device_width,
            height=args.device_height,
            fps=args.device_fps,
            replay_dir=args.replay_dir,
            replay_fps=args.replay_fps,
            no_loop=args.no_loop,
        )
        self.rect_explicit = args.rect is not None
        self.rect = self._resolve_rect()
        self.detector = self._resolve_detector()
        self.calibration = Calibration.load(args.calib) if args.calib else None
        self.servo = args.servo
        self.servo_gain = args.servo_gain
        self.cursor_detector = self._resolve_cursor_detector() if args.servo else None
        self.servo_error = None
        self.sender = Sender(args.host, args.port)

        if self.calibration:
            print(f"calibracao: {self.calibration.summary()}")
        elif self.src.region is None and not self.rect_explicit:
            raise SystemExit(
                "uma camera nao sabe onde o pixel cai: informe --calib, ou --rect x,y,w,h "
                "para mapear linearmente (so serve pra debug, sem perspectiva)."
            )

        self.frames = 0
        self.detections = 0
        self.errors = 0
        self.last_target = None
        self.last_stats = time.monotonic()
        self.last_park = 0.0
        self.fps = 0.0

    def _resolve_rect(self):
        if self.args.rect:
            return self.args.rect
        virtual = getattr(self.src, "virtual_rect", None)
        if callable(virtual):
            return virtual()
        print("aviso: --rect nao informado, detectando o desktop local", file=sys.stderr)
        return default_rect()

    # O cursor do trainer e desenhado por ele em ciano (#38d6e8). Isso da a visao
    # uma posicao VERDADEIRA do ponteiro na tela, o que fecha o laco: em vez de
    # confiar no modelo do atuador relativo, medimos o erro e corrigimos.
    CURSOR_BGR = (232, 214, 56)  # #38d6e8 em BGR

    def _resolve_cursor_detector(self) -> ColorDetector:
        return ColorDetector.from_bgr(
            self.CURSOR_BGR,
            tol=45,
            l_tol=85,
            min_area=12,
            downsample=1,
            adapt=0.0,
            reject_bands=False,
            open_size=0,     # a cruz tem 2px de espessura: abertura e blur a apagam
            close_size=7,   # fecha o vao central da cruz num blob so
            blur=0,
        )

    def _resolve_detector(self) -> ColorDetector:
        if self.args.model:
            detector = ColorDetector.load(self.args.model)
            print(f"modelo de cor: {detector.describe()}")
            return detector
        if self.args.color:
            detector = ColorDetector.from_bgr(self.args.color)
            print(f"modelo de cor: {detector.describe()}")
            return detector
        grab = self.src.grab()
        detector = ColorDetector.from_click(grab.image, *self._ask_click(grab.image))
        return detector

    @staticmethod
    def _ask_click(image):
        grabbed = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                grabbed.append((x, y))

        window = "clique no alvo (q cancela)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, on_mouse)
        cv2.imshow(window, image)
        while not grabbed:
            if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                cv2.destroyWindow(window)
                raise SystemExit("nenhuma cor escolhida; passe --color rrggbb")
        cv2.destroyWindow(window)
        return grabbed[0]

    def map_to_screen(self, detection: Detection, image: np.ndarray) -> tuple[float, float]:
        if self.calibration:
            return self.calibration.apply(detection.x, detection.y)
        if self.src.region is not None:
            rx, ry, rw, rh = self.src.region
            h, w = image.shape[:2]
            return rx + detection.x * rw / w, ry + detection.y * rh / h
        h, w = image.shape[:2]
        ox, oy, ow, oh = self.rect
        return ox + detection.x * ow / w, oy + detection.y * oh / h

    def step(self) -> None:
        try:
            grab = self.src.grab()
        except RuntimeError as exc:
            self.errors += 1
            print(f"frame perdido: {exc}", file=sys.stderr)
            if self.errors > 50:
                raise SystemExit("fonte falhando demais, abortando")
            return
        self.frames += 1

        now = time.monotonic()
        detection = self.detector.detect(grab.image)
        age_ms = int(min(255, (now - grab.ts) * 1000))

        if detection is None:
            self._park(now)
        else:
            self.detections += 1
            sx, sy = self.map_to_screen(detection, grab.image)
            if self.servo:
                cursor = self.cursor_detector.detect(grab.image)
                if cursor is not None:
                    cx, cy = self.map_to_screen(cursor, grab.image)
                    ex, ey = sx - cx, sy - cy
                    self.servo_error = (ex, ey)
                    # Manda um alvo deslocado pelo erro medido: o atuador relativo
                    # calcula delta = alvo - modelo, e o modelo dele esta errado.
                    # Empurrar pelo erro observado corrige a cada frame, entao o
                    # laco converge mesmo com aceleracao e escala fracionaria.
                    sx += self.servo_gain * ex
                    sy += self.servo_gain * ey
            self.last_target = (sx, sy)
            if not (np.isfinite(sx) and np.isfinite(sy)):
                self._park(now)
            else:
                self.sender.send_screen(sx, sy, self.rect, age_ms=age_ms)

        if self.args.preview:
            self._preview(grab.image, detection)

    def _park(self, now: float) -> None:
        """Tell the actuator there is no target, but at 10 Hz: repeating it every
        frame changes nothing and the phone already has a watchdog."""
        if now - self.last_park < 0.1:
            return
        self.last_park = now
        self.sender.park()

    def _preview(self, image, detection) -> None:
        view = image.copy()
        if detection is not None:
            x0, y0, w, h = detection.bbox
            cv2.rectangle(view, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 1)
            cv2.drawMarker(view, (int(detection.x), int(detection.y)), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        if self.last_target:
            sx, sy = self.last_target
            ox, oy, ow, oh = self.rect
            cv2.putText(
                view,
                f"tela {int(sx)},{int(sy)}",
                (8, view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow("j5-vision", view)
        if self.detector.last_mask is not None:
            cv2.imshow("mask", self.detector.last_mask)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    def report(self, now: float) -> None:
        if now - self.last_stats < 1.0:
            return
        elapsed = now - self.last_stats
        self.fps = self.frames / elapsed
        print(
            f"{self.fps:5.1f} fps  detects {self.detections:5d}  sends {self.sender.sent:6d}  "
            f"erros {self.errors:3d}  alvo {self.last_target}"
            + (f"  servo {self.servo_error[0]:+.0f},{self.servo_error[1]:+.0f}px"
               if self.servo_error else ""),
            file=sys.stderr,
            flush=True,
        )
        self.frames = 0
        self.last_stats = now

    def run(self) -> None:
        period = 1.0 / self.args.hz
        started = time.monotonic()
        deadline = started + self.args.duration if self.args.duration else None
        next_tick = started
        try:
            while True:
                now = time.monotonic()
                try:
                    self.step()
                except StopIteration:
                    print("fim da fonte", file=sys.stderr)
                    break
                self.report(now)
                if deadline and now >= deadline:
                    break
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        finally:
            self.sender.park()
            self.sender.close()
            self.src.close()
            if self.args.preview:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # opencv-headless: no highgui, nothing to tear down
            if self.args.save_model:
                self.detector.save(self.args.save_model)


def build_parser():
    p = argparse.ArgumentParser(description="loop de visao: captura -> detecta -> J5")
    p.add_argument("--source", choices=("x11", "http", "mjpeg", "v4l2", "replay", "test"), default="x11")
    p.add_argument("--url", default="http://192.168.42.1:8080")
    p.add_argument("--user", default="")
    p.add_argument("--password", default="")
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0, help="downscale da captura 1.0 = cru")
    p.add_argument("--device", default="/dev/video0", help="fonte v4l2: webcam, UVC do J5, v4l2loopback")
    p.add_argument("--fourcc", default="", help="forca MJPG/YUYV no v4l2")
    p.add_argument("--device-width", type=int, default=0)
    p.add_argument("--device-height", type=int, default=0)
    p.add_argument("--device-fps", type=float, default=0.0)
    p.add_argument("--replay-dir", default="rec", help="diretorio de vision/record.py")
    p.add_argument("--replay-fps", type=float, default=0.0, help="replay com taxa fixa em vez da original")
    p.add_argument("--no-loop", action="store_true")
    p.add_argument("--color", type=parse_hex_color, default=None, help="cor do alvo, ex: d000d0")
    p.add_argument("--model", default=None, help="carrega/salva modelo de cor (JSON)")
    p.add_argument("--save-model", default=None)
    p.add_argument("--calib", default=None)
    p.add_argument("--rect", type=parse_rect, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--servo", action="store_true",
                   help="corrige o alvo pela posicao real do cursor (cruz ciana do trainer)")
    p.add_argument("--servo-gain", type=float, default=0.45)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.source == "test" and args.rect is None:
        args.rect = (0, 0, 1920, 1080)
    try:
        Pipeline(args).run()
    except KeyboardInterrupt:
        print("\nparado")
    except (RuntimeError, ValueError, OSError) as exc:
        # A dead camera, a bad --calib path or a missing model is a message,
        # not a traceback.
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())