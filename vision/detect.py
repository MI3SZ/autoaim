"""Target detection by colour, built for a camera looking at a monitor.

Screen capture would make this a one-liner. A camera is what makes it a file:

  moire          sensor grid x monitor grid = colour that flickers per frame
  rolling shutter horizontal bands of the target colour sweep the screen
  jpeg 4:2:0     chroma is subsampled, so small saturated regions desaturate
  auto WB        the phone re-balances without asking, invalidating a fixed HSL window

The design answers, in order: detect on a downscaled image (downscale is the best
anti-moire filter there is), distance in Lab instead of an HSL box (exposure
moves L, not chroma), reject full-width/thin components as bands, and keep
re-referencing the chroma target slowly so WB drift is tracked, not fought.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    x: float  # camera pixel, full resolution
    y: float
    area: int
    bbox: tuple[int, int, int, int]
    lab: tuple[float, float, float]


class ColorDetector:
    def __init__(
        self,
        ref_lab: tuple[float, float, float],
        tol: float = 30.0,
        l_tol: float = 48.0,
        min_area: int = 40,
        max_area_frac: float = 0.25,
        downsample: int = 2,
        adapt: float = 0.02,
        blur: int = 5,
        reject_bands: bool = True,
        open_size: int = 3,
        close_size: int = 7,
    ):
        self.lab = np.array(ref_lab, dtype=np.float32)
        self.tol = float(tol)
        self.l_tol = float(l_tol)
        self.min_area = int(min_area)
        self.max_area_frac = float(max_area_frac)
        self.ds = max(1, int(downsample))
        self.adapt = float(adapt)
        self.blur = int(blur) | 1  # odd kernel or GaussianBlur complains
        self.reject_bands = reject_bands

        # A 3x3 opening erases anything thinner than 3px, which is fatal for a
        # drawn cursor crosshair. Thin targets set open_size=0.
        self._open = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            if open_size > 1 else None
        )
        self._close = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            if close_size > 1 else None
        )

        self.last_mask: Optional[np.ndarray] = None
        self.rejected_bands = 0

    # --- construction ----------------------------------------------------

    @classmethod
    def from_bgr(cls, sample_bgr, **kwargs) -> "ColorDetector":
        patch = np.uint8([[sample_bgr]])
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0, 0].tolist()
        return cls(tuple(round(v, 1) for v in lab), **kwargs)

    @classmethod
    def from_click(cls, image_bgr: np.ndarray, x: int, y: int, radius: int = 4, **kwargs) -> "ColorDetector":
        h, w = image_bgr.shape[:2]
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        patch = image_bgr[y0:y1, x0:x1].reshape(-1, 3)
        median = np.median(patch, axis=0)
        return cls.from_bgr(median, **kwargs)

    @classmethod
    def load(cls, path: str, **overrides) -> "ColorDetector":
        with open(path, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
        blob.update(overrides)
        return cls(**blob)

    def save(self, path: str) -> None:
        blob = {
            "ref_lab": [float(v) for v in self.lab],
            "tol": self.tol,
            "l_tol": self.l_tol,
            "min_area": self.min_area,
            "max_area_frac": self.max_area_frac,
            "downsample": self.ds,
            "adapt": self.adapt,
            "blur": self.blur,
            "reject_bands": self.reject_bands,
            "open_size": int(self._open.shape[0]) if self._open is not None else 0,
            "close_size": int(self._close.shape[0]) if self._close is not None else 0,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(blob, handle, indent=2)
        print(f"modelo de cor salvo em {path}")

    # --- detection -------------------------------------------------------

    def _mask(self, lab: np.ndarray) -> np.ndarray:
        """Box tolerance in Lab, evaluated by one C call.

        The spherical version (subtract, square, add, compare, and) built four
        full-frame float32 temporaries per frame to answer the same question and
        cost about ten times more. A box per axis is also easier to reason about:
        L gets its own width, chroma its own.
        """
        ref = self.lab
        low = (
            max(0, int(ref[0] - self.l_tol)),
            max(0, int(ref[1] - self.tol)),
            max(0, int(ref[2] - self.tol)),
        )
        high = (
            min(255, int(ref[0] + self.l_tol)),
            min(255, int(ref[1] + self.tol)),
            min(255, int(ref[2] + self.tol)),
        )
        return cv2.inRange(lab, np.array(low, np.uint8), np.array(high, np.uint8))

    def detect(self, image_bgr: np.ndarray) -> Optional[Detection]:
        h, w = image_bgr.shape[:2]
        if self.ds > 1:
            small = cv2.resize(image_bgr, (w // self.ds, h // self.ds), interpolation=cv2.INTER_AREA)
        else:
            small = image_bgr

        if self.blur > 1:
            small = cv2.GaussianBlur(small, (self.blur, self.blur), 0)

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        mask = self._mask(lab)
        if self._open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        if self._close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)
        self.last_mask = mask

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count < 2:
            return None

        sh, sw = small.shape[:2]
        min_area_small = max(4, self.min_area // (self.ds * self.ds))
        max_area_small = self.max_area_frac * sw * sh

        best_index = -1
        best_area = 0
        for index in range(1, count):
            _, _, bw, bh, area = (int(v) for v in stats[index])
            if area < min_area_small or area > max_area_small:
                continue
            if self.reject_bands and bw > 0.60 * sw and bh < max(8, 0.03 * sh):
                # A target is never a full-width sliver: that is a rolling
                # shutter band, and it is the single most convincing false
                # positive this pipeline produces.
                self.rejected_bands += 1
                continue
            if area > best_area:
                best_area, best_index = area, index

        if best_index < 0:
            return None

        cx, cy = centroids[best_index]
        x0, y0, bw, bh, area = (int(v) for v in stats[best_index])

        # Mean colour inside the blob, computed only over its bounding box. The
        # full-frame version copies a 200k-element boolean mask per frame just to
        # average a 60x60 patch.
        inside = labels[y0 : y0 + bh, x0 : x0 + bw] == best_index
        if not inside.any():
            return None
        mean_lab = lab[y0 : y0 + bh, x0 : x0 + bw][inside].mean(axis=0)

        # Re-reference the colour model only on a plausible target: roughly
        # square, comfortably above the noise floor. Adapting to every winner
        # lets one false blob drag the reference away for the rest of the run.
        aspect = bw / max(1, bh)
        if self.adapt > 0 and 0.4 <= aspect <= 2.5 and area >= 2 * min_area_small:
            # Chroma only: following L would make the detector chase its own
            # exposure instead of tracking the target's colour.
            self.lab[1] += self.adapt * (mean_lab[1] - self.lab[1])
            self.lab[2] += self.adapt * (mean_lab[2] - self.lab[2])

        return Detection(
            x=float(cx) * self.ds,
            y=float(cy) * self.ds,
            area=int(area * self.ds * self.ds),
            bbox=(x0 * self.ds, y0 * self.ds, bw * self.ds, bh * self.ds),
            lab=(float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])),
        )

    def describe(self) -> str:
        return (
            f"lab=({self.lab[0]:.0f},{self.lab[1]:.0f},{self.lab[2]:.0f}) "
            f"tol={self.tol:.0f} l_tol={self.l_tol:.0f} ds={self.ds}"
        )