#!/usr/bin/env python3
"""doctor.py - roda todos os diagnosticos de uma vez e diz onde esta o problema.

Foi escrito depois de uma sessao inteira em que cada sintoma parecia ser outro
problema: cursor deduzido de um blob errado, posicao publicada congelada, k medio
entre eixos com sensibilidades diferentes, latencia subestimada em 4x e captura
parada por janela coberta. Este script mede cada camada separadamente, na ordem
em que ela aparece no sistema, e no fim imprime um veredito.

Como usar: deixe o trainer visivel na tela (janela, nao tela cheia, com o OBS a
vista), pare o aim.py para nao brigarem pelo mesmo cursor, e rode:

    python3 vision/doctor.py --host 172.16.42.1
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
from aim import Finder, hex_to_bgr  # noqa: E402

TARGET_HEX = "d000d0"
CURSOR_HEX = "ffffff"


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def item(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))
        icon = {"ok": "ok  ", "fail": "FALHA", "warn": "aviso", "info": "info"}[status]
        print(f"{icon}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)

    def ok(self, name: str, detail: str = "") -> None:
        self.item("ok", name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.item("fail", name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.item("warn", name, detail)

    def info(self, name: str, detail: str = "") -> None:
        self.item("info", name, detail)

    def failures(self) -> list[tuple[str, str, str]]:
        return [line for line in self.lines if line[0] == "fail"]

    def warnings(self) -> list[tuple[str, str, str]]:
        return [line for line in self.lines if line[0] == "warn"]


class Sender:
    def __init__(self, host: str, port: int = 5555) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.sent = 0

    def delta(self, dx: int, dy: int) -> None:
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendto(protocol.pack_delta(dx, dy, self.seq), self.addr)
        self.sent += 1

    def park(self) -> None:
        self.seq = (self.seq + 1) & 0xFF
        self.sock.sendto(protocol.pack(0, 0, valid=False, seq=self.seq), self.addr)


class Vision:
    """Captura e localizacao do alvo e do cursor."""

    def __init__(self, device: str, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"nao abri {device} (v4l2-ctl --list-devices mostra o que existe)")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        self.target = Finder(hex_to_bgr(TARGET_HEX), tol=30, l_tol=48, min_area=40, downsample=2)
        self.cursor = Finder(hex_to_bgr(CURSOR_HEX), tol=22, l_tol=70, min_area=400,
                             downsample=1, open_size=3, close_size=5, blur=0, reject_bands=False)

    def frame(self):
        ok, image = self.cap.read()
        return image if ok else None

    def cursor_px(self, frames: int = 3):
        """Mediana de algumas leituras: reduz o efeito de um frame ruim."""
        found = []
        for _ in range(frames):
            image = self.frame()
            if image is None:
                continue
            position = self.cursor.find(image)
            if position:
                found.append(position)
        if not found:
            return None
        xs = sorted(f[0] for f in found)
        ys = sorted(f[1] for f in found)
        return (xs[len(xs) // 2], ys[len(ys) // 2])

    def target_px(self):
        image = self.frame()
        return self.target.find(image) if image is not None else None

    def close(self) -> None:
        self.cap.release()


def to_units(position, width, height):
    """Pixels de tela -> unidades de HID (0..32767)."""
    if position is None:
        return None
    return (position[0] / width * protocol.MAX_COORD, position[1] / height * protocol.MAX_COORD)


def from_units(units, width, height):
    if units is None:
        return None
    return (units[0] / protocol.MAX_COORD * width, units[1] / protocol.MAX_COORD * height)


def settle(vision: Vision, quiet_frames: int = 3, timeout: float = 1.2):
    """Espera a imagem parar de mudar e devolve a posicao final do cursor."""
    last = None
    stable = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = vision.cursor_px(1)
        if current is not None and last is not None and abs(current[0] - last[0]) < 2 and abs(current[1] - last[1]) < 2:
            stable += 1
            if stable >= quiet_frames:
                return current
        else:
            stable = 0
        last = current
        time.sleep(0.02)
    return last


def measure_step(vision: Vision, sender: Sender, dx: int, dy: int, settle_frames: int = 5):
    """Manda um delta e devolve (deslocamento medido em px, posicao antes, depois)."""
    before = settle(vision)
    if before is None:
        return None, None, None
    sender.delta(dx, dy)
    time.sleep(settle_frames / 30.0)
    after = settle(vision)
    if after is None:
        return None, before, None
    return (after[0] - before[0], after[1] - before[1]), before, after


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="diagnostico completo do laco visao -> J5")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--host", default="172.16.42.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--quick", action="store_true", help="pula os testes de linearidade e borda")
    args = parser.parse_args(argv)

    report = Report()
    print("=" * 72)
    print("doctor: diagnostico do laco visao -> USB -> cursor")
    print("=" * 72)
    print("premissas: trainer VISIVEL (janela, nao tela cheia), OBS a vista,")
    print("           aim.py parado. Comecando em 2s...")
    time.sleep(2.0)

    # ---- 1. captura ------------------------------------------------------
    try:
        vision = Vision(args.device, args.width, args.height)
    except RuntimeError as exc:
        report.fail("captura: dispositivo", str(exc))
        return 1

    frames = []
    started = time.monotonic()
    for _ in range(15):
        image = vision.frame()
        if image is not None:
            frames.append(image)
    elapsed = time.monotonic() - started
    if not frames:
        report.fail("captura: nenhum frame")
        vision.close()
        return 1
    fps = len(frames) / elapsed if elapsed > 0 else 0
    shape = frames[-1].shape
    report.ok("captura: frames", f"{fps:.0f} fps, {shape[1]}x{shape[0]}")
    if shape[1] != args.width or shape[0] != args.height:
        report.warn("captura: resolucao diferente do --rect",
                    f"frame {shape[1]}x{shape[0]} vs pedido {args.width}x{args.height}")

    identical = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))
    if identical >= len(frames) - 2:
        report.fail("captura: congelada", f"{identical}/{len(frames)-1} frames identicos - "
                    "OBS coberto ou camera virtual parada")
    else:
        report.ok("captura: viva", f"{identical} frames repetidos de {len(frames)-1}")

    # ---- 2. alvo ---------------------------------------------------------
    for _ in range(20):
        target = vision.target_px()
        if target:
            break
        time.sleep(0.05)
    if target:
        report.ok("alvo magenta achado", f"{int(target[0])},{int(target[1])}")
    else:
        report.fail("alvo magenta NAO achado", "trainer no modo 1/2 com cubo na tela?")

    # ---- 3. cursor -------------------------------------------------------
    cursor = vision.cursor_px()
    if cursor:
        report.ok("cursor (bola branca) achado", f"{int(cursor[0])},{int(cursor[1])}")
    else:
        report.fail("cursor NAO achado", "o trainer desenhou a bola? esta com foco?")
        vision.close()
        return 1

    sender = Sender(args.host, args.port)

    # ---- 4. canal: um passo pequeno move o cursor? -----------------------
    moved, before, after = measure_step(vision, sender, -40, 0)
    if moved is None:
        report.fail("canal USB: sem leitura do cursor")
    elif abs(moved[0]) < 4:
        report.fail("canal USB: mandei -40 e o cursor nao andou",
                    "J5 nao esta recebendo/aplicando; veja o hidmouse no celular")
    else:
        report.ok("canal USB responde", f"-40 contagens -> {moved[0]:+.0f} px")

    # ---- 5. centraliza em malha fechada ---------------------------------
    print("\n-- centralizando o cursor (malha fechada) --")
    center = (args.width / 2, args.height / 2)
    k_est = 1.3
    history = []
    latency_frames = None
    for step in range(24):
        position = vision.cursor_px(1)
        if position is None:
            break
        ex = center[0] - position[0]
        ey = center[1] - position[1]
        history.append((position[0], position[1]))
        if abs(ex) < 25 and abs(ey) < 25:
            break
        ux = int(max(-120, min(120, ex / k_est * 0.35)))
        uy = int(max(-120, min(120, ey / k_est * 0.35)))
        if ux == 0 and uy == 0:
            break
        sender.delta(ux, uy)
        time.sleep(1 / 30)
    position = vision.cursor_px(3)
    if position is None:
        report.fail("centralizacao: perdi o cursor")
    else:
        error = (center[0] - position[0], center[1] - position[1])
        if abs(error[0]) <= 60 and abs(error[1]) <= 60:
            report.ok("centralizacao: convergiu", f"erro {error[0]:+.0f},{error[1]:+.0f} px em {len(history)} passos")
        else:
            report.fail("centralizacao: NAO convergiu",
                        f"erro {error[0]:+.0f},{error[1]:+.0f} px depois de {len(history)} passos")

    # ---- 6. sensibilidade por eixo --------------------------------------
    print("\n-- sensibilidade por eixo --")
    k_measured = {}
    for axis, name in ((0, "x"), (1, "y")):
        deltas = [0, 0]
        deltas[axis] = 60 if (vision.cursor_px(1) or center)[axis] < (args.width if axis == 0 else args.height) / 2 else -60
        moved, _, _ = measure_step(vision, sender, deltas[0], deltas[1])
        if moved is None:
            report.warn(f"k {name}: sem leitura")
            continue
        if abs(moved[axis]) < 8:
            report.fail(f"k {name}: sem movimento", f"mandei {deltas[axis]:+d} contagens")
            continue
        k = abs(moved[axis] / deltas[axis])
        k_measured[name] = k
        report.ok(f"k {name}", f"{k:.2f} px por contagem")
    if len(k_measured) == 2:
        ratio = max(k_measured.values()) / max(0.01, min(k_measured.values()))
        if ratio > 1.25:
            report.warn("eixos com sensibilidades diferentes",
                        f"y/x = {ratio:.2f} - o aim.py ja compensa por eixo")

    # ---- 7. linearidade --------------------------------------------------
    if not args.quick:
        print("\n-- linearidade (passos de tamanhos diferentes) --")
        speeds = []
        for magnitude in (30, 90, 180):
            moved, _, _ = measure_step(vision, sender, magnitude, 0)
            if moved is not None and abs(moved[0]) > 5:
                speeds.append((magnitude, abs(moved[0]) / magnitude))
        if len(speeds) >= 2:
            factor = max(s for _, s in speeds) / max(0.01, min(s for _, s in speeds))
            detail = ", ".join(f"{m}->{s:.2f}px" for m, s in speeds)
            if factor > 1.4:
                report.warn("resposta NAO linear", detail)
            else:
                report.ok("resposta linear", detail)

    # ---- 8. latencia -----------------------------------------------------
    print("\n-- latencia (degrau e deteccao de movimento) --")
    before = vision.cursor_px(3)
    if before is not None:
        sender.delta(90, 0)
        moved_at = None
        start = time.monotonic()
        for _ in range(40):
            current = vision.cursor_px(1)
            if current is not None and abs(current[0] - before[0]) > 6:
                moved_at = time.monotonic() - start
                break
            time.sleep(0.01)
        if moved_at is None:
            report.warn("latencia: nao detectei o inicio do movimento")
        else:
            frames_of_delay = moved_at * 30
            report.info("latencia de resposta", f"{moved_at*1000:.0f} ms (~{frames_of_delay:.1f} frames)")
            stable_gain = 1.0 / max(1.0, frames_of_delay + 1.0) * 0.9
            report.info("ganho estavel sugerido", f"--gain {max(0.05, min(0.4, stable_gain)):.2f}")
    sender.park()
    vision.close()

    # ---- veredito --------------------------------------------------------
    print("\n" + "=" * 72)
    failures = report.failures()
    warnings = report.warnings()
    if not failures:
        print("VEREDITO: o laco esta sao. Se ainda nao acerta, o problema e ajuste,")
        print("          nao defeito. Comece com o ganho sugerido acima.")
    else:
        print(f"VEREDITO: {len(failures)} falha(s) - a primeira da lista e a causa raiz:")
        for status, name, detail in failures:
            print(f"   - {name}" + (f" ({detail})" if detail else ""))
    if warnings:
        print(f"\n{len(warnings)} aviso(s) (nao impedem funcionar):")
        for status, name, detail in warnings:
            print(f"   - {name}" + (f" ({detail})" if detail else ""))
    print("=" * 72)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())