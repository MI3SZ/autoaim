#!/usr/bin/env python3
"""aim.py - mira em laco fechado, medindo o cursor na tela.

Por que delta e nao posicao: o atuador relativo anti-go controla posicao por um
modelo interno (`delta = alvo - modelo`). Esse modelo satura no fim do espaco de
coordenadas, e depois disso `delta` vale zero mesmo com o cursor longe do alvo -
o cursor fica devendo e nao ha como mandar mais movimento. Aconteceu na pratica:
`cmd` preso em 32767 e erro de 26 mil unidades.

Aqui o PC manda o deslocamento que falta, em contagens de mouse, e o atuador
apenas escreve no HID. Quem sabe onde o cursor esta e a cruz ciana que o trainer
desenha; quem sabe onde o alvo esta e o detector de cor. O laco mede o erro,
estima quantas unidades de tela cada contagem vale (aceleracao de ponteiro e
escala fracionaria mudam isso) e manda so o que falta.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import protocol  # noqa: E402

TARGET_HEX = "d000d0"
CURSOR_HEX = "38d6e8"  # a cruz que o trainer desenha no lugar do cursor


def hex_to_bgr(text: str) -> tuple[int, int, int]:
    value = text.strip().lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def lab_reference(bgr) -> np.ndarray:
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


class Finder:
    """Maior blob de uma cor, em Lab.

    Linhas finas (a cruz do cursor) precisam de open=0 e blur=0: qualquer um dos
    dois apaga um traco de 2 px. E a cruz e quatro segmentos soltos com vao no
    meio, entao o fechamento precisa ser largo o bastante para uni-los - senao o
    centro sai deslocado em meia duzia de pixels.
    """

    def __init__(self, bgr, tol=30.0, l_tol=48.0, min_area=40, downsample=2,
                 open_size=3, close_size=7, blur=5, reject_bands=True):
        self.ref = lab_reference(bgr)
        self.tol = tol
        self.l_tol = l_tol
        self.min_area = min_area
        self.ds = max(1, int(downsample))
        self.open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)) if open_size > 1 else None
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)) if close_size > 1 else None
        self.blur = int(blur) | 1
        self.reject_bands = reject_bands

    def find(self, image):
        """Maior candidato, para o alvo (o cubo e o blob grande da cena)."""
        found = self.candidates(image)
        if not found:
            return None
        best = max(found, key=lambda item: item[2])
        return best[0], best[1]

    def candidates(self, image):
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // self.ds, h // self.ds), interpolation=cv2.INTER_AREA) if self.ds > 1 else image
        if self.blur > 1:
            small = cv2.GaussianBlur(small, (self.blur, self.blur), 0)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
        dl = lab[:, :, 0] - self.ref[0]
        da = lab[:, :, 1] - self.ref[1]
        db = lab[:, :, 2] - self.ref[2]
        mask = ((da * da + db * db <= self.tol * self.tol) & (dl * dl <= self.l_tol * self.l_tol)).astype(np.uint8) * 255
        if self.open_k is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_k)
        if self.close_k is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_k)

        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        min_area = max(4, self.min_area // (self.ds * self.ds))
        shrunk_w = w // self.ds
        shrunk_h = h // self.ds
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            box_w = int(stats[index, cv2.CC_STAT_WIDTH])
            box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
            if self.reject_bands and box_w > 0.60 * shrunk_w and box_h < max(8, 0.03 * shrunk_h):
                continue
            out.append((
                float(centroids[index][0]) * self.ds,
                float(centroids[index][1]) * self.ds,
                area * self.ds * self.ds,
            ))
        return out


def pick_cursor(candidates, previous, max_jump=260.0, area_range=(25, 4000)):
    """Escolhe a cruz entre os candidatos cianos.

    Pegar o maior blob e o que quebra: a tela tem outros objetos cianos e o
    detector alterna entre eles, o que envenena a estimativa de k e faz o cursor
    pular. A cruz e pequena e, principalmente, nao da saltos grandes entre dois
    frames - entao proximidade e o criterio certo.
    """
    plausible = [c for c in candidates if area_range[0] <= c[2] <= area_range[1]] or candidates
    if not plausible:
        return None
    if previous is None:
        return min(plausible, key=lambda c: c[2])  # a cruz e o blob pequeno
    near = [c for c in plausible if math.hypot(c[0] - previous[0], c[1] - previous[1]) <= max_jump]
    if not near:
        return None
    return min(near, key=lambda c: math.hypot(c[0] - previous[0], c[1] - previous[1]))


class Servo:
    """Controle proporcional em contagens de mouse, sobre erro medido.

    `k` e quantas unidades de tela o cursor anda por contagem enviada. Vale 1.0
    sem aceleracao e sem escala, mas o GNOME costuma entregar 0.5 a 1.6. Estimar
    isso e o que impede o laco de ficar lento ou de dar tranco.
    """

    def __init__(self, gain: float = 0.55, deadband: float = 8.0, max_step: float = 450.0):
        self.gain = gain
        self.deadband = deadband
        self.max_step = max_step
        self.k = 1.0
        self.error = None
        self._sent = None
        self._cursor_at_send = None

    def update(self, target, cursor):
        if target is None or cursor is None:
            self.error = None
            self._sent = None
            return None

        # Calibra k com o que foi pedido no frame anterior. Exige passo
        # relevante e movimento NO MESMO SENTIDO: sem isso um palpite de deteccao
        # faz k divergir (ja aconteceu de ir de 1.0 a 8.0 em dois segundos).
        if self._sent is not None and self._cursor_at_send is not None:
            asked = self._sent[0]
            moved = cursor[0] - self._cursor_at_send[0]
            if abs(asked) >= 40 and moved * asked > 0:
                observed = abs(moved / asked)
                if 0.05 < observed < 20.0:
                    self.k += 0.25 * (observed - self.k)

        ex = target[0] - cursor[0]
        ey = target[1] - cursor[1]
        self.error = (ex, ey)

        if abs(ex) <= self.deadband and abs(ey) <= self.deadband:
            self._sent = None
            return None

        scale = 1.0 / max(0.05, min(20.0, self.k))
        ux = max(-self.max_step, min(self.max_step, self.gain * ex * scale))
        uy = max(-self.max_step, min(self.max_step, self.gain * ey * scale))
        step = (int(round(ux)), int(round(uy)))
        if step == (0, 0):
            return None
        self._sent = step
        self._cursor_at_send = cursor
        return step


class Sender:
    def __init__(self, host, port=5555):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def delta(self, dx, dy):
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendto(protocol.pack_delta(dx, dy, self.seq), self.addr)
        self.sent += 1

    def park(self):
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendto(protocol.pack(0, 0, valid=False, seq=self.seq), self.addr)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="mira em laco fechado (cursor medido na tela)")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--color", default=TARGET_HEX, help="cor do alvo")
    parser.add_argument("--cursor-color", default=CURSOR_HEX, help="cor da cruz do cursor")
    parser.add_argument("--host", default="172.16.42.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--gain", type=float, default=0.55)
    parser.add_argument("--deadband", type=float, default=8.0, help="unidades de tela (~0.5px)")
    parser.add_argument("--max-step", type=float, default=450.0,
                        help="contagens maximas por frame (o atuador drena ~127 a 125Hz)")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)

    target_finder = Finder(hex_to_bgr(args.color), tol=30, l_tol=48, min_area=40, downsample=2)
    cursor_finder = Finder(hex_to_bgr(args.cursor_color), tol=45, l_tol=85, min_area=12,
                           downsample=1, open_size=0, close_size=7, blur=0, reject_bands=False)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"nao abri {args.device}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    servo = Servo(args.gain, args.deadband, args.max_step)
    sender = Sender(args.host, args.port)
    rect = (0, 0, args.width, args.height)

    period = 1.0 / args.hz
    start = time.monotonic()
    deadline = start + args.duration if args.duration else None
    next_tick = start
    frames = targetless = cursorless = 0
    last_cursor = None
    lost = 0
    last_stats = start
    warned = False

    print(f"aim: {args.device} {args.width}x{args.height} -> {args.host}:{args.port} "
          f"ganho {args.gain} deadband {args.deadband} max {args.max_step}", file=sys.stderr)

    try:
        while True:
            now = time.monotonic()
            ok, image = cap.read()
            if not ok or image is None:
                time.sleep(period)
                continue
            frames += 1

            target = target_finder.find(image)
            cursor = pick_cursor(cursor_finder.candidates(image), last_cursor)
            if cursor is not None:
                last_cursor = cursor
            if target is not None:
                target = protocol.to_abs(target[0], target[1], rect)
            if cursor is not None:
                cursor = protocol.to_abs(cursor[0], cursor[1], rect)

            if cursor is None and last_cursor is not None:
                lost += 1
                if lost == 45:
                    print("aviso: perdi a cruz ha 45 frames. Ela so existe onde o "
                          "trainer recebe mousemove: deixe-o em tela cheia (tecla f) "
                          "e evite o canto superior esquerdo, que abre a visao de "
                          "atividades do GNOME.", file=sys.stderr)
            elif cursor is not None:
                lost = 0

            if target is None:
                targetless += 1
                if targetless == 30 and not warned:
                    print("aviso: alvo ausente ha 30 frames (janela coberta?)", file=sys.stderr)
                    warned = True
                sender.park()
                step = None
            else:
                targetless = 0
                warned = False
                cursorless = 0 if cursor is not None else cursorless + 1
                step = servo.update(target, cursor)
                if step is not None:
                    sender.delta(step[0], step[1])

            if args.preview:
                if target:
                    cv2.drawMarker(image, (target[0] * args.width // protocol.MAX_COORD,
                                           target[1] * args.height // protocol.MAX_COORD),
                                   (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                if cursor:
                    cv2.drawMarker(image, (cursor[0] * args.width // protocol.MAX_COORD,
                                           cursor[1] * args.height // protocol.MAX_COORD),
                                   (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.imshow("aim", cv2.resize(image, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt

            if now - last_stats >= 1.0:
                err = servo.error
                print(
                    f"frames {frames:5d}  alvo {'-' if target is None else 'ok'}  "
                    f"cursor {'-' if cursor is None else 'ok'}  "
                    f"erro {('-' if err is None else f'{err[0]:+.0f},{err[1]:+.0f}')}  "
                    f"k {servo.k:4.2f}  passo {step if step else '-'}"
                    f"  cruz {('-' if cursor is None else f'{cursor[0]},{cursor[1]}')}",
                    file=sys.stderr, flush=True,
                )
                last_stats = now

            if deadline and now >= deadline:
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"\nfim: {frames} frames, {sender.sent} pacotes, "
              f"alvo ausente em {targetless} seguidos, cursor ausente em {cursorless}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())