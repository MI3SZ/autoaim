"""Camera pixel -> screen pixel mapping.

Two models, and the code picks between them by measured error instead of taste:

  poly2       least squares over [1, x, y, x^2, x*y, y^2] per axis.
              Absorbs perspective AND a good chunk of lens barrel distortion,
              needs >= 6 points. This is what usually wins on a phone lens.
  homography  classic DLT, 4+ points. Exact for a perfect pinhole camera, so it
              loses the corners on a wide phone lens, but needs almost no setup.

Both are fit, both are scored by reprojection error on the calibration points,
the better one is stored in calib.json along with its residual so you can see
how much to trust it.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

MIN_RMS_TARGET = 3.0  # screen pixels; above this the calibration is suspect


@dataclass
class Calibration:
    model: str  # "poly2" | "homography" | "affine"
    params: list[float]
    rms: float
    max_error: float
    n_points: int

    # --- fitting ---------------------------------------------------------

    @staticmethod
    def _poly2_terms(x: float, y: float) -> np.ndarray:
        return np.array([1.0, x, y, x * x, x * y, y * y], dtype=np.float64)

    @classmethod
    def _fit_poly2(cls, cam: np.ndarray, screen: np.ndarray) -> "Calibration":
        design = np.array([cls._poly2_terms(x, y) for x, y in cam], dtype=np.float64)
        cx, *_ = np.linalg.lstsq(design, screen[:, 0], rcond=None)
        cy, *_ = np.linalg.lstsq(design, screen[:, 1], rcond=None)
        calib = cls("poly2", list(cx) + list(cy), 0.0, 0.0, len(cam))
        return calib.with_score(cam, screen)

    @classmethod
    def _fit_homography(cls, cam: np.ndarray, screen: np.ndarray) -> "Calibration":
        matrix, _ = cv2.findHomography(cam.reshape(-1, 1, 2), screen.reshape(-1, 1, 2), 0)
        if matrix is None:
            raise RuntimeError("homografia falhou")
        calib = cls("homography", [float(v) for v in matrix.reshape(-1)], 0.0, 0.0, len(cam))
        return calib.with_score(cam, screen)

    @classmethod
    def _fit_affine(cls, cam: np.ndarray, screen: np.ndarray) -> "Calibration":
        matrix, _ = cv2.estimateAffine2D(cam, screen)
        if matrix is None:
            raise RuntimeError("afim falhou")
        calib = cls("affine", [float(v) for v in matrix.reshape(-1)], 0.0, 0.0, len(cam))
        return calib.with_score(cam, screen)

    def with_score(self, cam: np.ndarray, screen: np.ndarray) -> "Calibration":
        errors = []
        for (x, y), (sx, sy) in zip(cam, screen):
            px, py = self.apply(float(x), float(y))
            errors.append(float(np.hypot(px - sx, py - sy)))
        self.rms = float(np.sqrt(np.mean(np.square(errors))))
        self.max_error = float(np.max(errors))
        return self

    @classmethod
    def fit(cls, cam: Sequence, screen: Sequence) -> "Calibration":
        cam = np.asarray(cam, dtype=np.float64)
        screen = np.asarray(screen, dtype=np.float64)
        if len(cam) != len(screen):
            raise ValueError("listas de pontos com tamanhos diferentes")
        if len(cam) < 3:
            raise ValueError("preciso de pelo menos 3 pontos")
        if len(cam) < 6:
            print(f"aviso: {len(cam)} pontos, o polinomio de 2a ordem precisa de 6+; cai pra homografia/afim")

        candidates: list[Calibration] = []
        for factory, minimum in ((cls._fit_poly2, 6), (cls._fit_homography, 4), (cls._fit_affine, 3)):
            if len(cam) < minimum:
                continue
            try:
                candidates.append(factory(cam, screen))
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                print(f"  {factory.__name__}: {exc}")

        if not candidates:
            raise RuntimeError("nenhum modelo pode ser ajustado com esses pontos")
        return min(candidates, key=lambda c: c.rms)

    # --- use -------------------------------------------------------------

    def apply(self, x: float, y: float) -> tuple[float, float]:
        if self.model == "poly2":
            terms = self._poly2_terms(x, y)
            cx = np.asarray(self.params[:6])
            cy = np.asarray(self.params[6:12])
            return float(terms @ cx), float(terms @ cy)
        if self.model == "affine":
            matrix = np.asarray(self.params, dtype=np.float64).reshape(2, 3)
            return (
                float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
                float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
            )
        matrix = np.asarray(self.params, dtype=np.float64).reshape(3, 3)
        vec = matrix @ np.array([x, y, 1.0])
        if abs(vec[2]) < 1e-9:
            return float("nan"), float("nan")
        return float(vec[0] / vec[2]), float(vec[1] / vec[2])

    # --- persistence -----------------------------------------------------

    def save(self, path: str) -> None:
        blob = {
            "model": self.model,
            "params": [float(v) for v in self.params],
            "rms": self.rms,
            "max_error": self.max_error,
            "n_points": self.n_points,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        return cls(**blob)

    def summary(self) -> str:
        verdict = "ok" if self.rms <= MIN_RMS_TARGET else "SUSPEITA"
        return (
            f"{self.model}: rms {self.rms:.2f}px  max {self.max_error:.2f}px  "
            f"({self.n_points} pontos) -> {verdict}"
        )


# --- pairing a recording with the trainer's calibration windows -------------


def find_white_dot(image_bgr: np.ndarray) -> tuple[float, float] | None:
    """The calibration dot is pure white on a black screen: the one thing that is
    trivially separable no matter what the phone's white balance is doing."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    mask = (
        (lab[:, :, 0] >= 215)
        & (np.abs(lab[:, :, 1] - 128) <= 14)
        & (np.abs(lab[:, :, 2] - 128) <= 14)
    ).astype(np.uint8) * 255
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if stats[index, cv2.CC_STAT_AREA] < 20:
        return None
    return float(centroids[index][0]), float(centroids[index][1])


def pairs_from_recording(recording_dir: str, trainer_json: str) -> tuple[list, list]:
    """Frames are matched to trainer windows by wall clock, which is why the
    trainer exports epoch milliseconds alongside its monotonic ones."""
    with open(trainer_json, "r", encoding="utf-8") as handle:
        trainer = json.load(handle)
    windows = trainer.get("calibration") or []
    if not windows:
        raise RuntimeError("o JSON do trainer nao tem 'calibration' (rode o modo calib com 'e' no fim)")

    index_path = os.path.join(recording_dir, "index.csv")
    frames = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frames.append((float(row["epoch"]), row["file"]))

    cam_points, screen_points, skipped = [], [], []
    for window in windows:
        t0, t1 = float(window["tw0"]) / 1000.0, float(window["tw1"]) / 1000.0
        observations = []
        for epoch, name in frames:
            if not (t0 <= epoch <= t1):
                continue
            image = cv2.imread(os.path.join(recording_dir, name), cv2.IMREAD_COLOR)
            if image is None:
                continue
            found = find_white_dot(image)
            if found:
                observations.append(found)
        if len(observations) < 2:
            skipped.append(window["i"])
            continue
        cx = float(np.median([o[0] for o in observations]))
        cy = float(np.median([o[1] for o in observations]))
        cam_points.append((cx, cy))
        screen_points.append((float(window["x"]), float(window["y"])))

    if skipped:
        print(f"aviso: ponto(s) {skipped} sem deteccao suficiente, ignorados")
    if len(cam_points) < 3:
        raise RuntimeError("poucos pontos pareados, a camera nao viu o ponto branco")
    return cam_points, screen_points


def main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="calibracao camera -> tela")
    parser.add_argument("--pairs", help="CSV com camx,camy,screenx,screeny")
    parser.add_argument("--from", dest="recording", help="diretorio de gravacao (record.py)")
    parser.add_argument("--trainer-json", help="JSON exportado pelo trainer no modo calib")
    parser.add_argument("--out", default="calib.json")
    args = parser.parse_args(argv[1:])

    if args.pairs:
        cam, screen = [], []
        with open(args.pairs, "r", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0].startswith("#"):
                    continue
                cam.append((float(row[0]), float(row[1])))
                screen.append((float(row[2]), float(row[3])))
    elif args.recording and args.trainer_json:
        cam, screen = pairs_from_recording(args.recording, args.trainer_json)
    else:
        parser.error("use --pairs, ou --from DIR --trainer-json ARQ")

    print(f"{len(cam)} pontos pareados")
    calibration = Calibration.fit(cam, screen)
    print(calibration.summary())
    calibration.save(args.out)
    print(f"salvo em {args.out}")
    return 0 if calibration.rms <= MIN_RMS_TARGET else 1


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))