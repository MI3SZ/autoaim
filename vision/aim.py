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
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

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


class CursorServer:
    """Escuta a posicao do ponteiro que o trainer.html publica.

    Deduzir o cursor de um blob ciano na tela e fragil por natureza: qualquer
    objeto azulado vira candidato e, quando a cruz some (cursor fora da janela),
    o laco continua empurrando as cegas - foi assim que o cursor foi parar no
    canto e nao saiu. Aqui a posicao vem de quem sabe: a pagina que recebe o
    evento de ponteiro. O alvo continua sendo detectado por visao.
    """

    def __init__(self, port: int = 8765):
        self.port = port
        self._lock = threading.Lock()
        self._pos = None
        self._ts = 0.0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path != "/pos":
                    self.send_error(404)
                    return
                query = parse_qs(parsed.query)
                try:
                    x = float(query["x"][0])
                    y = float(query["y"][0])
                    w = float(query["w"][0])
                    h = float(query["h"][0])
                    d = float(query["d"][0])
                except (KeyError, IndexError, ValueError):
                    self.send_error(400)
                    return
                with server._lock:
                    server._pos = (x, y, w, h, d)
                    server._ts = time.monotonic()
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self._httpd = HTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def latest(self, max_age: float = 0.30):
        with self._lock:
            if self._pos is None or time.monotonic() - self._ts > max_age:
                return None
            return self._pos

    def close(self) -> None:
        self._httpd.shutdown()


class Servo:
    """Controle proporcional em contagens de mouse, sobre erro medido.

    `k` e quantas unidades de tela o cursor anda por contagem enviada. Vale 1.0
    sem aceleracao e sem escala, mas o GNOME costuma entregar 0.5 a 1.6. Estimar
    isso e o que impede o laco de ficar lento ou de dar tranco.
    """

    def __init__(self, gain: float = 0.25, deadband: float = 30.0, max_step: float = 300.0):
        self.gain = gain
        self.deadband = deadband
        self.max_x = max_step
        self.max_y = max_step
        # Sensibilidade POR EIXO: medido no hardware, y anda 1.78x mais que x
        # (exatamente a proporcao 1920/1080). Um k unico deixava x mandando
        # 1.4x a menos e y 1.4x a mais.
        self.kx = 1.0
        self.ky = 1.0
        self.error = None
        self._sent = None
        self._cursor_at_send = None
        self.adapt = True

    def update(self, target, cursor):
        if target is None or cursor is None:
            self.error = None
            self._sent = None
            return None

        # Calibra k com o que foi pedido no frame anterior. Exige passo
        # relevante e movimento NO MESMO SENTIDO: sem isso um palpite de deteccao
        # faz k divergir (ja aconteceu de ir de 1.0 a 8.0 em dois segundos).
        if self._sent is not None and self._cursor_at_send is not None and self.adapt:
            for axis_index, attribute in ((0, "kx"), (1, "ky")):
                asked = self._sent[axis_index]
                moved = cursor[axis_index] - self._cursor_at_send[axis_index]
                if abs(asked) >= 40 and moved * asked > 0:
                    observed = abs(moved / asked)
                    if 0.005 < observed < 50.0:
                        current = getattr(self, attribute)
                        setattr(self, attribute, current + 0.25 * (observed - current))

        ex = target[0] - cursor[0]
        ey = target[1] - cursor[1]
        self.error = (ex, ey)

        if abs(ex) <= self.deadband and abs(ey) <= self.deadband:
            self._sent = None
            return None

        ux = max(-self.max_x, min(self.max_x, self.gain * ex / max(0.005, self.kx)))
        uy = max(-self.max_y, min(self.max_y, self.gain * ey / max(0.005, self.ky)))
        step = (int(round(ux)), int(round(uy)))
        if step == (0, 0):
            return None
        self._sent = step
        self._cursor_at_send = cursor
        return step


def probe_axes(sender, measure, hz: float = 30.0, step: int = 120, log=print):
    """Mede cada eixo, cada sentido, separadamente.

    Serve para descobrir duas coisas que confundem tudo de uma vez: se um eixo
    responde pouco (fica preso na borda) e se o sentido e o esperado. Foi
    escrito depois de ver o eixo Y imovel com comando negativo e se mover com
    comando positivo, o que nao fecha com nenhuma outra explicacao.
    """
    position = None
    for _ in range(90):
        position = measure()
        if position:
            break
        time.sleep(1.0 / hz)
    if not position:
        log("sem posicao publicada; nao da para medir")
        return None

    results = {}
    for axis, name in ((0, "x"), (1, "y")):
        for sign in (-1, +1):
            before = measure() or position
            # se estiver colado na borda nesse sentido, o teste nao vale
            at_edge = (before[axis] < step * 2 and sign < 0) or \
                      (before[axis] > protocol.MAX_COORD - step * 2 and sign > 0)
            delta = [0, 0]
            delta[axis] = sign * step
            sender.delta(*delta)
            after = before
            for _ in range(3):
                time.sleep(2.0 / hz)
                fetched = measure()
                if fetched:
                    after = fetched
            moved = after[axis] - before[axis]
            key = f"{name}{'+' if sign > 0 else '-'}"
            results[key] = (moved, at_edge)
            log(f"  {key}: mandei {sign*step:+4d} contagens -> moveu {moved:+7.0f} unidades"
                + ("   [colado na borda, teste invalido]" if at_edge else ""))
            # volta ao ponto de partida para o proximo teste
            back = [0, 0]
            back[axis] = -sign * step
            sender.delta(*back)
            time.sleep(6.0 / hz)
    return results


def calibrate_k(sender, measure, hz: float = 30.0, step: int = 150):
    """Mede quantas unidades de tela o cursor anda por contagem enviada.

    Estimar isso de dentro do laco nao funciona: o cursor passa boa parte do
    tempo preso na borda, onde o deslocamento medido e truncado, e a media
    exponencial nunca converge. Aqui e um experimento controlado - manda um
    passo conhecido na direcao do centro da tela (onde ha espaco para andar),
    espera assentar e mede o deslocamento total.
    """
    position = None
    for _ in range(90):
        position = measure()
        if position:
            break
        time.sleep(1.0 / hz)
    if not position:
        return None

    ratios = []
    for axis in (0, 1):
        start = position[axis]
        sign = 1 if start < protocol.MAX_COORD / 2 else -1
        delta = [0, 0]
        delta[axis] = sign * step
        # Um envio so: o atuador acumula e drena, entao mandar duas vezes
        # comanda o dobro e a medicao sai dobrada.
        sender.delta(*delta)
        latest = position
        for _ in range(3):        # espera o movimento assentar e mede o total
            time.sleep(2.0 / hz)
            fetched = measure()
            if fetched:
                latest = fetched
        moved = (latest[axis] - start) * sign
        if moved > step * 0.2:
            ratios.append(moved / step)
        position = latest

    if len(ratios) < 2:
        return None
    return ratios[0], ratios[1]


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
    parser.add_argument("--gain", type=float, default=0.25,
                        help="fracao do erro por frame; acima de ~0.3 bate lata com a latencia")
    parser.add_argument("--deadband", type=float, default=30.0, help="unidades de tela (~2px)")
    parser.add_argument("--max-step", type=float, default=0.0,
                        help="teto em contagens por frame (0 = calcula pela velocidade alvo)")
    parser.add_argument("--max-px-frame", type=float, default=40.0,
                        help="velocidade alvo do cursor, em px por frame")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--cursor-source", choices=("auto", "http", "vision"), default="auto",
                        help="http = posicao publicada pelo trainer; vision = cruz ciana")
    parser.add_argument("--cursor-port", type=int, default=8765)
    parser.add_argument("--stats-ms", type=int, default=250,
                        help="intervalo das linhas de estatistica (1s esconde a convergencia)")
    parser.add_argument("--probe-axes", action="store_true",
                        help="mede cada eixo e cada sentido, imprime e sai")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="pula a medicao de k no inicio e usa a estimativa do laco")
    args = parser.parse_args(argv)

    target_finder = Finder(hex_to_bgr(args.color), tol=30, l_tol=48, min_area=40, downsample=2)
    cursor_finder = Finder(hex_to_bgr(args.cursor_color), tol=45, l_tol=85, min_area=12,
                           downsample=1, open_size=0, close_size=7, blur=0, reject_bands=False)

    def current_cursor():
        """Posicao do cursor em unidades de tela, da fonte que estiver ativa."""
        if cursor_server is not None:
            published = cursor_server.latest()
            if published:
                px, py, vw, vh, dpr = published
                if vw > 0 and vh > 0:
                    return protocol.to_abs(px * args.width / vw, py * args.height / vh, rect)
        return None

    rect = (0, 0, args.width, args.height)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"nao abri {args.device}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    servo = Servo(args.gain, args.deadband, args.max_step or 300.0)
    sender = Sender(args.host, args.port)
    cursor_server = None
    if args.cursor_source in ("auto", "http"):
        try:
            cursor_server = CursorServer(args.cursor_port)
            print(f"cursor: esperando o trainer em http://127.0.0.1:{args.cursor_port}/pos",
                  file=sys.stderr)
        except OSError as exc:
            print(f"cursor: porta {args.cursor_port} indisponivel ({exc}); usando visao",
                  file=sys.stderr)
    period = 1.0 / args.hz
    start = time.monotonic()
    deadline = start + args.duration if args.duration else None
    next_tick = start
    if args.probe_axes:
        if cursor_server is None:
            print("--probe-axes precisa da fonte http (o trainer publicando)", file=sys.stderr)
            return 1
        print("sondando cada eixo (leva ~5s)...", file=sys.stderr)
        probe_axes(sender, current_cursor, args.hz,
                   log=lambda msg: print(msg, file=sys.stderr, flush=True))
        return 0

    if not args.no_calibrate and cursor_server is not None:
        measured = calibrate_k(sender, current_cursor, args.hz)
        if measured:
            servo.kx, servo.ky = measured
            servo.adapt = False   # calibrado vale mais que a estimativa do laco
            units_per_px = protocol.MAX_COORD / args.width
            px_x = servo.kx / units_per_px
            px_y = servo.ky / units_per_px
            print(f"calibrado: x {servo.kx:.2f} unidades/contagem ({px_x:.2f} px), "
                  f"y {servo.ky:.2f} ({px_y:.2f} px)", file=sys.stderr)
            if not args.max_step:
                # O atuador drena 127 contagens por tick a 125 Hz: sem teto em
                # pixels ele faz 19 telas por segundo e o laco nunca acompanha.
                servo.max_x = max(1.0, args.max_px_frame / max(0.01, px_x))
                servo.max_y = max(1.0, args.max_px_frame / max(0.01, px_y))
                print(f"teto de velocidade: {args.max_px_frame:.0f} px/frame "
                      f"= x {servo.max_x:.0f} / y {servo.max_y:.0f} contagens", file=sys.stderr)
        else:
            print("calibracao falhou (cursor nao respondeu); usando estimativa do laco",
                  file=sys.stderr)

    frames = targetless = cursorless = 0
    last_cursor = None
    frozen_seen = None
    frozen_count = 0
    frozen_warned = False
    lost = 0
    viewport_checked = False
    last_stats = start
    warned = False

    print(f"aim: {args.device} {args.width}x{args.height} -> {args.host}:{args.port} "
          f"ganho {args.gain} deadband {args.deadband} max {args.max_step}", file=sys.stderr)
    print("dica: ganho alto demais faz o cursor bater de canto a canto; "
          "se oscilar, baixe --gain (0.08 ~ 0.25)", file=sys.stderr)

    try:
        while True:
            now = time.monotonic()
            ok, image = cap.read()
            if not ok or image is None:
                time.sleep(period)
                continue
            frames += 1

            target = target_finder.find(image)

            cursor = None
            if cursor_server is not None:
                published = cursor_server.latest()
                if published is not None:
                    px, py, vw, vh, dpr = published
                    if not viewport_checked:
                        viewport_checked = True
                        physical = (round(vw * dpr), round(vh * dpr))
                        print(f"cursor: trainer publicando viewport {physical[0]}x{physical[1]} "
                              f"(dpr {dpr})", file=sys.stderr)
                        if physical != (args.width, args.height):
                            print("aviso: o viewport nao cobre a tela. Aperte 'f' para "
                                  "tela cheia - sem isso clientX/Y nao sao coordenadas de tela.",
                                  file=sys.stderr)
                    if vw > 0 and vh > 0:
                        # viewport em px de CSS -> px de tela (tela cheia, 0,0)
                        cursor = protocol.to_abs(px * dpr * args.width / (vw * dpr),
                                                 py * dpr * args.height / (vh * dpr),
                                                 rect)
            if cursor is None and args.cursor_source != "http":
                cursor = pick_cursor(cursor_finder.candidates(image), last_cursor)
            if cursor is not None:
                last_cursor = cursor
            if target is not None:
                target = protocol.to_abs(target[0], target[1], rect)
            if cursor is not None:
                cursor = protocol.to_abs(cursor[0], cursor[1], rect)

            # A posicao publicada congela quando o ponteiro sai da janela do
            # trainer (outra janela por cima). Sem isso o laco empurra as cegas.
            if cursor is not None:
                if frozen_seen is not None and cursor == frozen_seen:
                    frozen_count += 1
                else:
                    frozen_count = 0
                    frozen_seen = cursor
                    frozen_warned = False
                if frozen_count == 45 and not frozen_warned:
                    frozen_warned = True
                    print("aviso: a posicao publicada esta congelada ha 45 frames. "
                          "O ponteiro saiu da janela do trainer (outra janela por cima?). "
                          "Deixe o trainer em tela cheia e CLIQUE nele para dar foco.",
                          file=sys.stderr)

            if cursor is None and last_cursor is not None and args.cursor_source == "vision":
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

            if now - last_stats >= args.stats_ms / 1000.0:
                err = servo.error
                err_px = (int(err[0] / (protocol.MAX_COORD / args.width)),
                          int(err[1] / (protocol.MAX_COORD / args.height))) if err else None
                print(
                    f"erro {('-' if err is None else f'{err_px[0]:+5d},{err_px[1]:+5d}px')}  "
                    f"passo {str(step) if step else '-':>12s}  "
                    f"cruz {('-' if cursor is None else f'{int(cursor[0]*args.width/protocol.MAX_COORD)},{int(cursor[1]*args.height/protocol.MAX_COORD)}')}"
                    f"  k {servo.kx:4.1f}/{servo.ky:4.1f}",
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
        if cursor_server is not None:
            cursor_server.close()
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        print(f"\nfim: {frames} frames, {sender.sent} pacotes, "
              f"alvo ausente em {targetless} seguidos, cursor ausente em {cursorless}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())