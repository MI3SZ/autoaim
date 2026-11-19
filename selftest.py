#!/usr/bin/env python3
"""One command, everything checked.

    python3 selftest.py          # todas as checagens
    python3 selftest.py --bench  # mais o custo por frame do detector

Nothing here needs the phone, the gadget or a camera: every piece is exercised
against its own contract, so a regression shows up as a failing line instead of
as a cursor that behaves strangely on a Tuesday.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))

import numpy as np  # noqa: E402

import protocol  # noqa: E402
import report_desc  # noqa: E402

try:
    import cv2  # noqa: E402
except ImportError:  # pragma: no cover
    print("falta opencv: pip install opencv-python (ou python3-opencv no Fedora)")
    raise SystemExit(1)

import calibrate  # noqa: E402
import detect  # noqa: E402
import hidmouse  # noqa: E402
import source  # noqa: E402

TARGET_BGR = (208, 0, 208)  # #d000d0
TARGET_HEX = "d000d0"

CHECKS: list[tuple[str, object]] = []


def check(name):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


def frame(width=640, height=360, value=24):
    return np.full((height, width, 3), value, np.uint8)


# --- protocol ---------------------------------------------------------------


@check("protocol: 10 bytes, roundtrip, crc e magic")
def _():
    data = protocol.pack(1234, 30000, valid=True, click=False, age_ms=15, seq=7)
    assert len(data) == protocol.FRAME_SIZE == 10, len(data)
    parsed = protocol.unpack(data)
    assert parsed == protocol.Frame(1234, 30000, True, False, 15, 7), parsed
    assert protocol.unpack(data[:9] + bytes((data[9] ^ 0xFF,))) is None, "crc nao pegou corrupcao"
    assert protocol.unpack(b"\x00" + data[1:]) is None, "magic invalido aceito"
    assert protocol.unpack(data[:-1]) is None, "tamanho errado aceito"


@check("protocol: limites e clamp de coordenada")
def _():
    data = protocol.pack(-50, 99999)
    assert data[3:5] == b"\x00\x00", f"clamp inferior falhou: {data.hex()}"
    assert data[5:7] == b"\xff\x7f", f"clamp superior falhou: {data.hex()}"
    x, y = protocol.unpack(protocol.pack(-50, 99999))[:2]
    assert (x, y) == (0, protocol.MAX_COORD), (x, y)
    assert protocol.to_abs(0, 0, (0, 0, 1920, 1080)) == (0, 0)
    assert protocol.to_abs(1920, 1080, (0, 0, 1920, 1080)) == (32767, 32767)
    # multi-monitor: o retangulo virtual inteiro, com offset
    assert protocol.to_abs(1920, 0, (-1920, 0, 3840, 1080)) == (32767, 0)
    centre = protocol.to_abs(0, 540, (-1920, 0, 3840, 1080))
    assert all(abs(v - protocol.MAX_COORD / 2) <= 1 for v in centre), centre


@check("protocol: pacote de delta assinado")
def _():
    data = protocol.pack_delta(-1500, 900, seq=3)
    assert len(data) == protocol.FRAME_SIZE
    parsed = protocol.unpack(data)
    assert parsed is not None and parsed.delta is True
    assert (parsed.x, parsed.y) == (-1500, 900), parsed
    extremes = protocol.unpack(protocol.pack_delta(-32768, 32767, seq=4))
    assert (extremes.x, extremes.y) == (-32768, 32767), extremes
    absolute = protocol.unpack(protocol.pack(1234, 30000, seq=5))
    assert absolute.delta is False and (absolute.x, absolute.y) == (1234, 30000)


@check("hidmouse: delta relativo e escrito direto, sem modelo")
def _():
    act = make_actuator()
    reports = []
    act._write = lambda report: reports.append(report) or True
    act.feed(delta_packet(-1500, 900, seq=1), now=0.0)
    assert len(reports) == 12, f"{len(reports)} reports (esperado 12 para 1500 em passos de 127)"
    assert all(len(r) == 3 for r in reports), {len(r) for r in reports}
    total_x = sum(((r[1] + 128) % 256) - 128 for r in reports)
    total_y = sum(((r[2] + 128) % 256) - 128 for r in reports)
    assert total_x == -1500 and total_y == 900, (total_x, total_y)
    # delta nao passa pelo filtro nem pelo deadband: e aplicado na hora
    assert act.target is None


@check("protocol: seq modular e resync")
def _():
    assert protocol.seq_is_newer(1, None)
    assert protocol.seq_is_newer(3, 255), "wrap 255 -> 3 deveria ser mais novo"
    assert not protocol.seq_is_newer(255, 3)
    assert not protocol.seq_is_newer(7, 7), "duplicata aceita"


@check("protocol: reports HID absoluto e relativo")
def _():
    assert protocol.hid_abs(32767, 0) == bytes((0, 0xFF, 0x7F, 0x00, 0x00))
    assert protocol.hid_abs(0, 32767) == bytes((0, 0x00, 0x00, 0xFF, 0x7F))
    assert protocol.hid_rel(-1, 1) == bytes((0, 0xFF, 0x01)), "signed byte errado"
    assert protocol.hid_rel(-500, 500) == bytes((0, 0x81, 0x7F)), "clamp relativo errado"


# --- descriptor -------------------------------------------------------------


def descriptor_report_bytes(desc: bytes) -> int:
    """Walk the HID items and add up the Input report size."""
    i = 0
    size_bits, count, total = 1, 1, 0
    while i < len(desc):
        prefix = desc[i]
        i += 1
        if prefix == 0xFE:  # long item
            i += 2 + desc[i]
            continue
        length = prefix & 0x03
        length = 4 if length == 3 else length
        item_type = (prefix >> 2) & 0x03  # 0=Main 1=Global 2=Local
        tag = prefix >> 4
        data = desc[i : i + length]
        i += length
        if item_type == 0 and tag == 8:  # Input
            total += count * size_bits
        elif item_type == 1 and tag == 7:  # Report Size
            size_bits = int.from_bytes(data, "little")
        elif item_type == 1 and tag == 9:  # Report Count
            count = int.from_bytes(data, "little")
    return total // 8


@check("report_desc: report declarado bate com o descriptor")
def _():
    abs_bytes = descriptor_report_bytes(report_desc.ABS_DESC)
    rel_bytes = descriptor_report_bytes(report_desc.REL_DESC)
    assert abs_bytes == protocol.HID_ABS_SIZE == 5, abs_bytes
    assert rel_bytes == protocol.HID_REL_SIZE == 3, rel_bytes
    for desc in (report_desc.ABS_DESC, report_desc.REL_DESC):
        assert desc.endswith(b"\xC0\xC0"), "collection nao fechada"
        assert desc[0] == 0x05 and desc[1] == 0x01, "usage page errada"


@check("report_desc: --hex imprime o descritor e nao escreve arquivo")
def _():
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "report_desc.py"), "--hex", "--rel"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out.split() == [f"{b:02x}" for b in report_desc.REL_DESC], "hex divergente"


# --- detect -----------------------------------------------------------------


@check("detect: banda de rolling shutter e rejeitada sozinha")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR)
    band = frame()
    band[180:188, :] = TARGET_BGR
    assert det.detect(band) is None, "banda virou alvo"


@check("detect: banda nao rouba o alvo real")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR)
    img = frame()
    img[180:188, :] = TARGET_BGR
    img[60:112, 400:452] = TARGET_BGR
    found = det.detect(img)
    assert found is not None, "nao achou o quadrado"
    assert abs(found.x - 426) <= 3 and abs(found.y - 86) <= 3, (found.x, found.y)


@check("detect: sobrevive a moire + banding + jpeg q60")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR)
    img = frame()
    img[150:206, 200:256] = TARGET_BGR
    rng = np.random.default_rng(7)
    noisy = np.clip(img.astype(np.int16) + rng.normal(0, 26, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    for y in range(0, 360, 4):
        noisy[y : y + 2, :] = np.clip(noisy[y : y + 2, :].astype(np.int16) + 40, 0, 255)
    ok, encoded = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 60])
    assert ok
    found = det.detect(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    assert found is not None, "perdeu o alvo com jpeg"
    assert abs(found.x - 228) <= 4 and abs(found.y - 178) <= 4, (found.x, found.y)


@check("detect: cor errada nao vira alvo")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR)
    for name, colour in (("vermelho", (0, 0, 230)), ("azul", (230, 0, 0)),
                         ("verde", (0, 220, 0)), ("ciano", (230, 230, 0))):
        img = frame()
        img[100:160, 300:360] = colour
        assert det.detect(img) is None, f"{name} foi confundido com o alvo"
    img = frame()
    img[100:160, 300:360] = (150, 0, 150)   # magenta dessaturado do mesmo matiz
    found = det.detect(img)
    assert found is not None, "magenta escuro deveria continuar valendo"


@check("detect: cena vazia devolve None")
def _():
    assert detect.ColorDetector.from_bgr(TARGET_BGR).detect(frame()) is None


@check("detect: modelo nao se adapta a blob implausivel")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR, adapt=0.5)
    before = det.lab.copy()
    slash = frame()
    slash[180:182, :] = TARGET_BGR  # largura total, 2px de altura
    det.detect(slash)
    assert np.allclose(det.lab, before), f"modelo deriva com blob falso: {before} -> {det.lab}"
    square = frame()
    square[100:160, 100:160] = TARGET_BGR
    det.detect(square)
    assert not np.allclose(det.lab, before), "modelo deveria se adaptar a um alvo plausivel"


@check("detect: cruz ciana do cursor e achada no centro")
def _():
    img = frame(1920, 1080)
    img[500:560, 400:460] = TARGET_BGR
    cyan = (232, 214, 56)  # #38d6e8
    for d in range(-13, 14):
        if abs(d) > 3:
            img[300 + d, 699:701] = cyan
            img[299:301, 700 + d] = cyan
    img[299:301, 699:701] = cyan
    detector = detect.ColorDetector.from_bgr(
        cyan, tol=45, l_tol=85, min_area=12, downsample=1,
        adapt=0.0, reject_bands=False, open_size=0, close_size=7, blur=0,
    )
    found = detector.detect(img)
    assert found is not None, "cruz nao detectada"
    assert abs(found.x - 700) <= 2 and abs(found.y - 300) <= 2, (found.x, found.y)
    # nao pode confundir a cruz com o alvo magenta
    assert detect.ColorDetector.from_bgr(TARGET_BGR).detect(img) is not None


@check("detect: alvo no canto ainda e encontrado")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR)
    img = frame()
    img[0:40, 0:40] = TARGET_BGR
    found = det.detect(img)
    assert found is not None and found.x < 30 and found.y < 30, found


@check("detect: downsample 1 e 3 concordam sobre o centro")
def _():
    img = frame()
    img[100:160, 300:360] = TARGET_BGR
    centres = []
    for ds in (1, 2, 3):
        found = detect.ColorDetector.from_bgr(TARGET_BGR, downsample=ds).detect(img)
        assert found is not None, ds
        centres.append((found.x, found.y))
    for x, y in centres:
        assert abs(x - 330) <= 3 and abs(y - 130) <= 3, centres


@check("detect: save/load preserva o modelo")
def _():
    det = detect.ColorDetector.from_bgr(TARGET_BGR, tol=21, downsample=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "color.json")
        det.save(path)
        back = detect.ColorDetector.load(path)
    assert back.tol == 21 and back.ds == 3
    assert np.allclose(back.lab, det.lab), (back.lab, det.lab)


# --- calibracao -------------------------------------------------------------


def make_projection(view):
    def project(point):
        v = view @ np.array([point[0], point[1], 1.0])
        return (v[0] / v[2], v[1] / v[2])

    return project


@check("calib: homografia exata com 4 pontos")
def _():
    screen = np.array([[100, 80], [1820, 120], [1800, 1000], [140, 940]], float)
    view = np.array([[0.9, 0.05, 0.02], [0.05, 0.85, 0.01], [0.0002, 0.0001, 1.0]])
    cam = np.array([make_projection(view)(p) for p in screen])
    calib = calibrate.Calibration.fit(cam, screen)
    assert calib.model == "homography", calib.model
    assert calib.rms < 0.01, calib.rms
    for (cx, cy), (sx, sy) in zip(cam, screen):
        px, py = calib.apply(cx, cy)
        assert abs(px - sx) < 0.5 and abs(py - sy) < 0.5, (px, py, sx, sy)


@check("calib: polinomio ganha quando ha distorcao de lente")
def _():
    screen = np.array(
        [[100, 80], [1820, 120], [1800, 1000], [140, 940], [960, 540],
         [960, 120], [1820, 560], [960, 980], [140, 560]], float
    )
    view = np.array([[0.9, 0.05, 0.02], [0.05, 0.85, 0.01], [0.0002, 0.0001, 1.0]])
    cam = np.array([make_projection(view)(p) for p in screen])
    centre = cam.mean(axis=0)
    radius = np.linalg.norm(cam - centre, axis=1)
    barrel = cam + (cam - centre) * (0.06 * (radius / radius.max()) ** 2)[:, None]
    calib = calibrate.Calibration.fit(barrel, screen)
    assert calib.model == "poly2", calib.model
    assert calib.rms < 10.0, calib.rms


@check("calib: afinidade pura tem caminho de apply proprio")
def _():
    screen = np.array([[100, 100], [1800, 100], [100, 1000], [1800, 1000], [950, 550]], float)
    calib = calibrate.Calibration._fit_affine(screen.copy(), screen)
    assert calib.model == "affine"
    px, py = calib.apply(950.0, 550.0)
    assert abs(px - 950) < 0.5 and abs(py - 550) < 0.5, (px, py)


@check("calib: save/load e resumo")
def _():
    screen = np.array([[100, 80], [1820, 120], [1800, 1000], [140, 940], [960, 540]], float)
    view = np.array([[0.9, 0.05, 0.02], [0.05, 0.85, 0.01], [0.0002, 0.0001, 1.0]])
    cam = np.array([make_projection(view)(p) for p in screen])
    calib = calibrate.Calibration.fit(cam, screen)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calib.json")
        calib.save(path)
        back = calibrate.Calibration.load(path)
    px, py = back.apply(*cam[4])
    assert abs(px - 960) < 0.5 and abs(py - 540) < 0.5, (px, py)
    assert "rms" in calib.summary()


@check("calib: menos de 3 pontos e erro claro")
def _():
    try:
        calibrate.Calibration.fit([(0, 0)], [(0, 0)])
    except ValueError as exc:
        assert "3 pontos" in str(exc), exc
    else:
        raise AssertionError("deveria recusar 1 ponto")


@check("calib: ponto branco encontrado")
def _():
    img = np.zeros((360, 640, 3), np.uint8)
    cv2.circle(img, (500, 120), 22, (255, 255, 255), -1)
    found = calibrate.find_white_dot(img)
    assert found is not None and abs(found[0] - 500) < 2 and abs(found[1] - 120) < 2, found
    assert calibrate.find_white_dot(np.zeros((360, 640, 3), np.uint8)) is None


# --- fontes -----------------------------------------------------------------


@check("source: test source produz alvo detectavel")
def _():
    src = source.make_source("test")
    grab = src.grab()
    assert grab.image.shape[2] == 3
    found = detect.ColorDetector.from_bgr(TARGET_BGR).detect(grab.image)
    assert found is not None, "quadrado sintetico nao detectado"


@check("source: replay respeita o pacing gravado e o meta.region")
def _():
    import csv
    import json

    with tempfile.TemporaryDirectory() as tmp:
        det = detect.ColorDetector.from_bgr(TARGET_BGR)
        rows = []
        for i in range(6):
            img = frame()
            img[100:160, 40 + i * 20 : 100 + i * 20] = TARGET_BGR
            name = f"f{i:05d}.jpg"
            cv2.imwrite(os.path.join(tmp, name), img)
            rows.append([f"{1000.0 + i * 0.05:.6f}", f"{float(i):.6f}", name])
        with open(os.path.join(tmp, "index.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "mono", "file"])
            writer.writerows(rows)
        with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"region": [10, 20, 1920, 1080]}, handle)

        src = source.make_source("replay", replay_dir=tmp, no_loop=True)
        assert src.region == (10, 20, 1920, 1080), src.region
        started = time.monotonic()
        seen = 0
        while True:
            try:
                grab = src.grab()
            except StopIteration:
                break
            assert det.detect(grab.image) is not None
            seen += 1
        elapsed = time.monotonic() - started
        assert seen == 6, seen
        assert 0.20 <= elapsed <= 0.45, f"pacing errado: {elapsed:.3f}s para 6 frames a 20 fps"


@check("source: replay com --replay-fps ignora o pacing original")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i in range(5):
            name = f"f{i:05d}.jpg"
            cv2.imwrite(os.path.join(tmp, name), frame())
            rows.append([f"{1000.0 + i:.6f}", f"{float(i):.6f}", name])
        import csv

        with open(os.path.join(tmp, "index.csv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "mono", "file"])
            writer.writerows(rows)
        src = source.make_source("replay", replay_dir=tmp, no_loop=True, replay_fps=100.0)
        started = time.monotonic()
        n = 0
        while True:
            try:
                src.grab()
            except StopIteration:
                break
            n += 1
        elapsed = time.monotonic() - started
        assert n == 5 and elapsed < 0.2, (n, elapsed)


@check("source: v4l2 inexistente da erro acionavel")
def _():
    try:
        source.make_source("v4l2", device="/dev/video99")
    except RuntimeError as exc:
        assert "video99" in str(exc), exc
        assert "list-devices" in str(exc), exc
    else:
        raise AssertionError("deveria falhar sem device")


@check("source: replay sem gravacao da erro claro")
def _():
    try:
        source.make_source("replay", replay_dir="/tmp/j5-nao-existe-mesmo")
    except RuntimeError as exc:
        assert "index.csv" in str(exc), exc
    else:
        raise AssertionError("deveria falhar sem index.csv")


@check("record.py: fonte invalida sai 1 com mensagem, sem traceback")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "vision", "record.py"),
             "--source", "v4l2", "--device", "/dev/video99", "--duration", "0.2",
             "--out", os.path.join(tmp, "rec")],
            capture_output=True, text=True, timeout=60,
        )
    assert result.returncode == 1, result.returncode
    assert "Traceback" not in result.stderr, result.stderr
    assert "video99" in result.stderr, result.stderr


@check("run.py: calibracao ausente sai 1 com mensagem, sem traceback")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "vision", "run.py"),
             "--source", "test", "--color", "d000d0", "--calib", os.path.join(tmp, "nao.json"),
             "--rect", "0,0,1920,1080", "--duration", "0.1"],
            capture_output=True, text=True, timeout=60,
        )
    assert result.returncode == 1, result.returncode
    assert "Traceback" not in result.stderr, result.stderr


# --- atuador ----------------------------------------------------------------


def packet(x, y, **kwargs) -> protocol.Frame:
    """feed() takes a decoded frame, like the real loop after drain()."""
    return protocol.unpack(protocol.pack(x, y, **kwargs))


def delta_packet(dx, dy, **kwargs) -> protocol.Frame:
    return protocol.unpack(protocol.pack_delta(dx, dy, **kwargs))


def make_actuator(**overrides):
    argv = ["--dry-run", "--hz", "125"]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return hidmouse.Actuator(hidmouse.build_parser().parse_args(argv))


@check("hidmouse: 1-euro converge sem overshoot")
def _():
    f = hidmouse.OneEuro()
    assert f(0.0, 1 / 125) == 0.0
    value = 0.0
    for _ in range(200):
        value = f(1000.0, 1 / 125)
        assert value <= 1000.0 + 1e-6, f"overshoot: {value}"
    assert value > 995.0, f"nao convergiu: {value}"


@check("hidmouse: reaquisicao emite mesmo no centro exato")
def _():
    act = make_actuator(deadband=50)
    assert act.sent == (16383, 16383)
    act.feed(packet(16383, 16383, seq=1), now=0.0)
    act.tick(0.0)
    assert act.writes == 1, "alvo no centro nao gerou nenhum report"


@check("hidmouse: modo auto le o report_length do gadget")
def _():
    import json
    with tempfile.TemporaryDirectory() as tmp:
        for length, expected in ((5, "abs"), (3, "rel")):
            func = os.path.join(tmp, "usb_gadget", "g1", "functions", "hid.usb0")
            os.makedirs(func, exist_ok=True)
            with open(os.path.join(func, "report_length"), "w", encoding="utf-8") as handle:
                handle.write(str(length))
            os.environ["J5_GADGET_DIR"] = os.path.join(tmp, "usb_gadget")
            hidmouse.GADGET_DIR = os.path.join(tmp, "usb_gadget")
            assert hidmouse.detect_report_mode() == expected, (length, hidmouse.detect_report_mode())
    # sem configfs legivel, cai no absoluto em vez de explodir
    hidmouse.GADGET_DIR = "/nao/existe"
    assert hidmouse.detect_report_mode() == "abs"
    del os.environ["J5_GADGET_DIR"]


@check("hidmouse: deadband evita reports redundantes")
def _():
    act = make_actuator(deadband=5)
    for i in range(20):
        act.feed(packet(16383, 16383, seq=i), now=0.0 + i * 0.008)
        act.tick(0.0 + i * 0.008)
    assert act.writes == 1, f"escreveu {act.writes} reports para um alvo parado"


@check("hidmouse: teto de passo limita o teleporte")
def _():
    act = make_actuator(max_step=1000, deadband=0)
    act.feed(packet(32767, 32767, seq=1), now=0.0)
    act.tick(0.0)
    assert act.sent[0] <= 16383 + 1000 + 1, act.sent
    assert act.sent[1] <= 16383 + 1000 + 1, act.sent


@check("hidmouse: watchdog para de escrever sem pacote")
def _():
    act = make_actuator(watchdog_ms=100)
    act.feed(packet(20000, 20000, seq=1), now=0.0)
    for step in range(0, 6):        # dentro da janela do watchdog
        act.tick(step * 0.02)
    settled = act.writes
    assert settled > 0, "deveria ter se movido na janela do watchdog"
    for step in range(6, 40):       # 0.7s ja sem pacote algum
        act.tick(step * 0.02)
    assert act.writes == settled, f"continuou escrevendo: {act.writes} > {settled}"


@check("hidmouse: age alta descarta o frame")
def _():
    act = make_actuator(max_age_ms=100)
    act.feed(packet(20000, 20000, age_ms=200, seq=1), now=0.0)
    assert act.dropped_age == 1 and act.target is None, (act.dropped_age, act.target)


@check("hidmouse: reinicio do sender e ressincronizado")
def _():
    act = make_actuator(watchdog_ms=100)
    act.feed(packet(10000, 10000, seq=200), now=0.0)
    assert act.target is not None
    # silencio maior que o watchdog, depois o sender volta do seq 1
    act.feed(packet(20000, 21000, seq=1), now=1.0)
    assert act.dropped_seq == 0, f"descartou {act.dropped_seq} pacotes apos restart"
    assert act.target == (20000.0, 21000.0), act.target


@check("hidmouse: pacote invalido estaciona o alvo")
def _():
    act = make_actuator()
    act.feed(packet(10000, 10000, seq=1), now=0.0)
    assert act.target is not None
    act.feed(packet(0, 0, valid=False, seq=2), now=0.01)
    assert act.target is None


@check("hidmouse: ancoragem poe o modelo no canto")
def _():
    act = make_actuator(mode="rel", rel_home=50)
    act.home()
    assert act.sent == (0, 0), act.sent
    assert act.writes == 50, act.writes


@check("hidmouse: reabre o hidg quando o no e recriado")
def _():
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        path = handle.name
    try:
        args = hidmouse.build_parser().parse_args(
            ["--hidg", path, "--mode", "rel", "--deadband", "0", "--rel-home", "0"]
        )
        act = hidmouse.Actuator(args)
        assert act.fd is not None
        act.feed(packet(20000, 20000, seq=1), now=0.0)
        act.tick(0.0)
        first = act.writes
        assert first > 0

        # o kernel destroi e recria o /dev/hidg0 nas costas do processo
        dead = act.fd
        os.close(dead)
        act.fd = dead
        before = act.writes
        act.tick(0.01)          # detecta o fd morto, solta
        assert act.fd is None, "deveria ter soltado o fd morto"
        act.tick(0.02)          # reabre
        assert act.fd is not None, "deveria ter reaberto"
        act.tick(0.03)
        assert act.writes > before, "parou de escrever depois de reabrir"
    finally:
        os.unlink(path)


@check("hidmouse: modo relativo emite reports de 3 bytes")
def _():
    act = make_actuator(mode="rel", deadband=0, max_step=1600)
    reports = []
    act._write = lambda report: reports.append(report) or True
    act.feed(packet(32767, 32767, seq=1), now=0.0)
    for step in range(60):
        act.tick(step / 125)
    assert reports, "nenhum report relativo"
    assert all(len(r) == 3 for r in reports), {len(r) for r in reports}
    assert sum(((r[1] + 128) % 256) - 128 for r in reports) > 0, "nao andou em x"


@check("aim.py: o parser expoe todas as flags que o codigo usa")
def _():
    import ast
    source = (pathlib.Path(ROOT) / "vision" / "aim.py").read_text()
    tree = ast.parse(source)
    args_used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args"
    }
    flags = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value.lstrip("-").replace("-", "_"))
    missing = {name for name in args_used - flags if not name.startswith("_")}
    assert not missing, f"aim.py usa args que o parser nao define: {sorted(missing)}"


# --- gadget.sh --------------------------------------------------------------


def fake_gadget_tree(root: str) -> str:
    gadget = os.path.join(root, "usb_gadget", "g1")
    os.makedirs(os.path.join(gadget, "configs", "c.1"))
    os.makedirs(os.path.join(gadget, "functions"))
    for name, value in (("UDC", "ci_hdrc.0"), ("idVendor", "0x04e8"), ("idProduct", "0x6860")):
        with open(os.path.join(gadget, name), "w", encoding="utf-8") as handle:
            handle.write(value)
    return gadget


def run_gadget(args, cfs: str, hidg: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, CFS=cfs, GADGET="g1", UDC="ci_hdrc.0", HIDG=hidg)
    return subprocess.run(["sh", os.path.join(ROOT, "gadget.sh"), *args],
                          capture_output=True, text=True, env=env, timeout=60)


@check("gadget.sh: add cria a funcao, liga no config e reata o UDC")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        gadget = fake_gadget_tree(tmp)
        hidg = os.path.join(tmp, "hidg0")
        open(hidg, "w").close()

        result = run_gadget(["add"], os.path.join(tmp, "usb_gadget"), hidg)
        assert result.returncode == 0, result.stderr

        function = os.path.join(gadget, "functions", "hid.usb0")
        assert os.path.isdir(function), "funcao nao criada"
        assert open(os.path.join(function, "report_length")).read() == "5"
        assert open(os.path.join(function, "subclass")).read() == "0"
        assert open(os.path.join(function, "protocol")).read() == "0"
        descriptor = open(os.path.join(function, "report_desc"), "rb").read()
        assert descriptor == report_desc.ABS_DESC, "descriptor gravado errado"
        link = os.path.join(gadget, "configs", "c.1", "hid.usb0")
        assert os.path.islink(link), "funcao nao ligada na config"
        assert open(os.path.join(gadget, "UDC")).read() == "ci_hdrc.0", "UDC nao reatado"


@check("gadget.sh: add e idempotente e --rel troca o descriptor")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        gadget = fake_gadget_tree(tmp)
        hidg = os.path.join(tmp, "hidg0")
        open(hidg, "w").close()
        cfs = os.path.join(tmp, "usb_gadget")

        assert run_gadget(["add"], cfs, hidg).returncode == 0
        again = run_gadget(["add"], cfs, hidg)
        assert again.returncode == 0, again.stderr
        assert "recriando a funcao" in again.stderr, again.stderr
        # idempotente: mesmo modo, mesmo descriptor
        function = os.path.join(gadget, "functions", "hid.usb0")
        assert open(os.path.join(function, "report_length")).read() == "5"
        assert open(os.path.join(function, "report_desc"), "rb").read() == report_desc.ABS_DESC

        rel = run_gadget(["add", "--rel"], cfs, hidg)
        assert rel.returncode == 0, rel.stderr
        function = os.path.join(gadget, "functions", "hid.usb0")
        assert open(os.path.join(function, "report_length")).read() == "3"
        assert open(os.path.join(function, "report_desc"), "rb").read() == report_desc.REL_DESC


@check("gadget.sh: status sai 0 e del desliga sem derrubar o UDC")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        gadget = fake_gadget_tree(tmp)
        hidg = os.path.join(tmp, "hidg0")
        open(hidg, "w").close()
        cfs = os.path.join(tmp, "usb_gadget")

        assert run_gadget(["add"], cfs, hidg).returncode == 0
        status = run_gadget(["status"], cfs, hidg)
        assert status.returncode == 0, status.stderr
        assert "presente" in status.stdout
        assert "hid.usb0" in status.stdout

        removed = run_gadget(["del"], cfs, hidg)
        assert removed.returncode == 0, removed.stderr
        assert not os.path.islink(os.path.join(gadget, "configs", "c.1", "hid.usb0"))
        assert open(os.path.join(gadget, "UDC")).read() == "ci_hdrc.0", "del desligou o gadget inteiro"


@check("gadget.sh: falha no meio nao deixa o usb0 morto")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        gadget = fake_gadget_tree(tmp)
        cfs = os.path.join(tmp, "usb_gadget")
        missing = os.path.join(tmp, "nao-existe-hidg")

        result = run_gadget(["add"], cfs, missing)
        assert result.returncode != 0, "deveria falhar sem /dev/hidg"
        assert open(os.path.join(gadget, "UDC")).read() == "ci_hdrc.0", \
            "gadget ficou desligado apos a falha: usb0 morto ate rebootar"


@check("gadget.sh: gadget inexistente da erro explicito")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_gadget(["add"], os.path.join(tmp, "vazio"), "/dev/null")
        assert result.returncode != 0
        assert "nao existe" in result.stderr, result.stderr


# --- loop completo ----------------------------------------------------------

INTEGRATION = '''
import os, sys, time
sys.path.insert(0, {root!r})
sys.path.insert(0, {vision!r})
import protocol, source, detect
from sender import Sender
from calibrate import Calibration
import io, contextlib

src = source.make_source("test")
det = detect.ColorDetector.from_bgr({colour!r})
sender = Sender("127.0.0.1", {port})
start = time.monotonic()
sent = 0
while time.monotonic() - start < 1.5:
    grab = src.grab()
    found = det.detect(grab.image)
    if found is None:
        sender.park()
    else:
        sender.send(int(found.x / grab.image.shape[1] * 32767),
                    int(found.y / grab.image.shape[0] * 32767), age_ms=1)
        sent += 1
    time.sleep(0.005)
print(sent)
'''


@check("integracao: detector -> UDP -> atuador")
def _():
    port = 5591
    script = INTEGRATION.format(root=ROOT, vision=os.path.join(ROOT, "vision"),
                                colour=TARGET_BGR, port=port)
    proc = subprocess.Popen([sys.executable, "-c", script],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.25)

    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.setblocking(True)
    sock.settimeout(2.0)

    act = make_actuator()
    frames = 0
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            break
        parsed = protocol.unpack(data)
        assert parsed is not None, "pacote invalido chegou"
        frames += 1
        act.feed(parsed, now=frames * 0.005)
        act.tick(frames * 0.005)
    sock.close()
    proc.wait(timeout=5)
    assert proc.returncode == 0, proc.stderr.read()
    assert frames > 50, f"poucos pacotes: {frames}"
    assert act.writes > 0, "atuador nunca escreveu"
    assert act.dropped_age == 0 and act.dropped_seq == 0, (act.dropped_age, act.dropped_seq)


# --- runner -----------------------------------------------------------------


def main(argv) -> int:
    bench = "--bench" in argv
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")

    if bench:
        print("\n--- custo por frame ---")
        det = detect.ColorDetector.from_bgr(TARGET_BGR)
        for label, size in (("720p", (720, 1280)), ("1080p", (1080, 1920))):
            img = np.full((*size, 3), 24, np.uint8)
            img[size[0] // 3 : size[0] // 3 + 120, size[1] // 3 : size[1] // 3 + 120] = TARGET_BGR
            for _ in range(20):
                det.detect(img)
            started = time.perf_counter()
            for _ in range(200):
                det.detect(img)
            per_frame = (time.perf_counter() - started) / 200 * 1000
            print(f"detect {label:6s} {per_frame:6.2f} ms  ({1000 / per_frame:6.0f} fps possiveis)")

    total = len(CHECKS)
    print(f"\n{total - failures}/{total} checagens passaram")
    if failures:
        print("houve falha: nao empacote nada antes de resolver")
        return 1
    print("tudo verde")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))