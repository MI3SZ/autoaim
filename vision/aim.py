#!/usr/bin/env python3
"""aim.py - laco fechado: mede onde o cursor esta, mede onde o alvo esta, corrige.

O problema que isso resolve: no modo relativo o atuador calcula
`delta = alvo - modelo`, e o modelo dele comeca num chute (o centro da tela).
Enquanto o modelo estiver errado, todo comando sai deslocado, e o limite de passo
por report transforma o erro em movimento forte e torto.

Duas coisas consertam isso, e nenhuma exige mexer no celular:

  ancora   no inicio mandamos o alvo (0,0) por meio segundo. O atuador leva o
           cursor para a esquerda ate o SO travar na borda; a partir dali o
           modelo dele e a posicao real concordam, e mirar em absoluto vale.
  servo    cada frame compara a posicao do cursor com a do alvo e integra a
           correcao. A aceleracao do ponteiro e a escala fracionaria do GNOME
           deixam de importar: o que importa e o erro medido, nao a matematica.

Sem a cruz visivel (jogo sem HUD), cai em malha aberta: manda o alvo e deixa o
modelo seguir, que e o comportamento antigo.
"""

from __future__ import annotations

import argparse
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
    """Maior blob de uma cor, em Lab, com morfologia ajustavel.

    Linhas finas (a cruz do cursor) precisam de open=0 e blur=0: qualquer um dos
    dois apaga um traco de 2 px.
    """

    def __init__(self, bgr, tol=30.0, l_tol=48.0, min_area=40, downsample=2,
                 open_size=3, close_size=7, blur=5):
        self.ref = lab_reference(bgr)
        self.tol = tol
        self.l_tol = l_tol
        self.min_area = min_area
        self.ds = max(1, int(downsample))
        self.open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)) if open_size > 1 else None
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)) if close_size > 1 else None
        self.blur = int(blur) | 1

    def find(self, image):
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
        best, best_area = -1, 0
        min_area = max(4, self.min_area // (self.ds * self.ds))
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            name = int(stats[index, cv2.CC_STAT_WIDTH])
            tall = int(stats[index, cv2.CC_STAT_HEIGHT])
            if name > 0.6 * (w // self.ds) and tall < 10:
                continue  # banda de rolling shutter / linha de UI
            if area > best_area:
                best, best_area = index, area
        if best < 0:
            return None
        return float(centroids[best][0]) * self.ds, float(centroids[best][1]) * self.ds


class Servo:
    """Integrador no espaco do modelo do atuador.

    O atuador faz `modelo = comando` a cada pacote e move o cursor por
    `k * (comando - modelo)`. Integrando a correcao a gente obtem, no cursor,
    um laco de primeira ordem: sem overshoot e convergente com qualquer k > 0.
    """

    def __init__(self, gain: float = 0.6, deadband: float = 6.0):
        self.base_gain = gain
        self.deadband = deadband
        self.cmd = None
        self.anchored = False
        self.error = None
        self.k = 1.0          # pixels andados por unidade comandada
        self._prev_cmd = None     # comando de duas leituras atras
        self._last_cursor = None  # cursor observado quando o ultimo foi calculado

    @staticmethod
    def _clamp(value: float) -> float:
        return min(protocol.MAX_COORD, max(0.0, value))

    def update(self, target_hid, cursor_hid):
        if cursor_hid is None:
            # malha aberta: sem referencia do cursor, so aponta
            self.cmd = target_hid
            self.anchored = False
            self.error = None
            self._last_cmd = self._last_cursor = None
            return self.cmd

        if not self.anchored:
            # Semeia o modelo do atuador com a posicao REAL do cursor (o atuador
            # ancorado entende o primeiro pacote como "voce esta aqui" e nao move).
            self.cmd = (float(cursor_hid[0]), float(cursor_hid[1]))
            self.anchored = True
            self._last_cmd = self.cmd
            self._last_cursor = (float(cursor_hid[0]), float(cursor_hid[1]))
            return int(round(self.cmd[0])), int(round(self.cmd[1]))

        # Quantos pixels o cursor andou por unidade comandada? Aceleracao de
        # ponteiro e escala fracionaria fazem isso valer 0.5 ou 1.6, e sem medir o
        # laco ou fica lento ou bate no teto do espaco de coordenadas.
        if self._prev_cmd is not None and self._last_cursor is not None:
            asked = self.cmd[0] - self._prev_cmd[0]
            moved = cursor_hid[0] - self._last_cursor[0]
            if abs(asked) > 100:
                k_obs = moved / asked
                if 0.2 < k_obs < 3.0:
                    self.k += 0.3 * (k_obs - self.k)

        ex = target_hid[0] - cursor_hid[0]
        ey = target_hid[1] - cursor_hid[1]
        self.error = (ex, ey)

        self._prev_cmd = self.cmd
        self._last_cursor = (float(cursor_hid[0]), float(cursor_hid[1]))

        effective = min(1.5, self.base_gain / max(0.3, min(2.0, self.k)))
        if abs(ex) > self.deadband or abs(ey) > self.deadband:
            self.cmd = (self._clamp(self.cmd[0] + effective * ex),
                        self._clamp(self.cmd[1] + effective * ey))
        return int(round(self.cmd[0])), int(round(self.cmd[1]))


def home(sender, hold: float = 0.6, rate: float = 40.0) -> None:
    """Leva o cursor a borda superior esquerda antes de comecar.

    O atuador relativo parte de um modelo chutado (o centro da tela). Mandar o
    alvo (0,0) faz ele mover o cursor para a esquerda; o sistema operacional
    trava em 0 e o modelo acompanha ate 0. Depois disso os dois concordam, e o
    unico erro restante e de escala, que o servo mede e corrige.
    """
    count = max(1, int(hold * rate))
    for _ in range(count):
        sender.send(0, 0)
        time.sleep(1.0 / rate)
    print(f"ancorado no canto ({count} pacotes)", file=sys.stderr)


class Sender:
    def __init__(self, host, port=5555):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def send(self, x, y, valid=True):
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendto(protocol.pack(x, y, valid, False, 0, self.seq), self.addr)
        self.sent += 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="laco fechado de mira (cursor medido na tela)")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--color", default=TARGET_HEX, help="cor do alvo")
    parser.add_argument("--cursor-color", default=CURSOR_HEX, help="cor da cruz do cursor")
    parser.add_argument("--host", default="172.16.42.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--deadband", type=float, default=6.0, help="HID units, ~0.4px")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--no-home", action="store_true", help="nao ancora no canto no inicio")
    args = parser.parse_args(argv)

    target_finder = Finder(hex_to_bgr(args.color), tol=30, l_tol=48, min_area=40, downsample=2)
    cursor_finder = Finder(hex_to_bgr(args.cursor_color), tol=45, l_tol=85, min_area=12,
                           downsample=1, open_size=0, close_size=7, blur=0)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"nao abri {args.device}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    servo = Servo(args.gain, args.deadband)
    sender = Sender(args.host, args.port)
    rect = (0, 0, args.width, args.height)

    period = 1.0 / args.hz
    start = time.monotonic()
    deadline = start + args.duration if args.duration else None
    next_tick = start
    frames = targetless = cursorless = 0
    last_stats = start

    print(f"aim: {args.device} {args.width}x{args.height} -> {args.host}:{args.port} "
          f"ganho {args.gain} deadband {args.deadband}", file=sys.stderr)

    if not args.no_home:
        home(sender)

    try:
        while True:
            now = time.monotonic()
            ok, image = cap.read()
            if not ok or image is None:
                time.sleep(period)
                continue
            frames += 1

            target = target_finder.find(image)
            cursor = cursor_finder.find(image)

            if target is None:
                targetless += 1
                if targetless == 30:
                    print("aviso: alvo ausente ha 30 frames (janela coberta?)", file=sys.stderr)
                sender.send(0, 0, valid=False)
            else:
                targetless = 0
                if cursor is None:
                    cursorless += 1
                else:
                    cursorless = 0
                t_hid = protocol.to_abs(target[0], target[1], rect)
                c_hid = protocol.to_abs(cursor[0], cursor[1], rect) if cursor else None
                cmd = servo.update(t_hid, c_hid)
                sender.send(cmd[0], cmd[1])

            if args.preview:
                if target:
                    cv2.drawMarker(image, (int(target[0]), int(target[1])), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
                if cursor:
                    cv2.drawMarker(image, (int(cursor[0]), int(cursor[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
                cv2.imshow("aim", cv2.resize(image, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt

            if now - last_stats >= 1.0:
                err = servo.error
                print(
                    f"frames {frames:5d}  alvo {'-' if target is None else 'ok'}  "
                    f"cursor {'-' if cursor is None else 'ok'}  "
                    f"erro {('-' if err is None else f'{err[0]:+.0f},{err[1]:+.0f}')}  "
                    f"modo {'fechado' if servo.anchored else 'aberto'}  "
                    f"cmd {servo.cmd}",
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