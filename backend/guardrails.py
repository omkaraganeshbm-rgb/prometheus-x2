"""Distribution-shift guardrails and lightweight human-in-the-loop memory."""
from __future__ import annotations

import threading
import cv2
import numpy as np


class ShiftGuard:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.baseline: tuple[float, float] | None = None

    def assess(self, image: np.ndarray, heatmap: np.ndarray, embedding_drift: float, auto_stabilize: bool = True) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        luminance = float(np.mean(gray))
        contrast = float(np.std(gray))
        with self.lock:
            if self.baseline is None:
                self.baseline = (luminance, contrast)
            base_lum, base_contrast = self.baseline
        global_shift = abs(luminance - base_lum) / 255.0 > 0.13 or abs(contrast - base_contrast) / 255.0 > 0.12
        localized_peak = float(np.percentile(heatmap, 99.5))
        shifted = bool(global_shift and localized_peak < 0.84 and embedding_drift < 0.30)
        adjusted = image
        if shifted and auto_stabilize:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            adjusted = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        return {
            "distribution_shift_warning": shifted,
            "auto_stabilized": bool(shifted and auto_stabilize),
            "luminance": round(luminance, 2), "contrast": round(contrast, 2),
        }


shift_guard = ShiftGuard()


def inject_normal_memory(engine, embedding: np.ndarray, confidence: float) -> int:
    if not 45.0 <= float(confidence) <= 65.0:
        raise ValueError("Only borderline 45%-65% confidence inspections can be approved.")
    return engine.remember_normal(embedding)
