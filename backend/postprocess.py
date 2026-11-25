"""Interpretable anomaly post-processing, overlays, routing, and calibration."""
from __future__ import annotations

import base64
import cv2
import numpy as np


def _encode_jpeg(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def calibrated_confidence(score: float) -> float:
    """Monotonic logistic calibration; production calibration requires labeled data."""
    x = float(np.clip(score, 0, 1))
    probability = 1.0 / (1.0 + np.exp(-10.0 * (x - 0.48)))
    return round(float(probability * 100.0), 1)


def classify_defect(gray: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int]) -> str:
    x, y, w, h = box
    crop = gray[y:y + h, x:x + w]
    if crop.size == 0:
        return "Surface Contamination"
    aspect = max(w, h) / max(1, min(w, h))
    perimeter = max(cv2.arcLength(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True), 1.0) if cv2.countNonZero(mask) else 1.0
    area = max(cv2.countNonZero(mask), 1)
    circularity = min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
    variance = float(np.std(crop))
    if aspect >= 4.2 and h <= max(18, int(gray.shape[0] * 0.15)):
        return "Micro-Scratch"
    if aspect >= 2.8:
        return "Structural Crack"
    if circularity >= 0.48:
        return "Dent/Puncture"
    if variance > 48:
        return "Surface Contamination"
    return "Dent/Puncture"


def postprocess(image: np.ndarray, raw_heatmap: np.ndarray) -> dict:
    h, w = image.shape[:2]
    heat = cv2.resize(raw_heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.clip(heat, 0, 1)
    # Robust thresholding avoids treating low-level texture across the whole frame as a defect.
    threshold = max(0.58, float(np.percentile(heat, 99.2)))
    binary = (heat >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    accepted = np.zeros_like(binary)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        cv2.drawContours(accepted, [contour], -1, 255, -1)
        boxes.append({"x": int(x), "y": int(y), "width": int(bw), "height": int(bh), "area": int(area)})
    peak = float(np.max(heat)) if heat.size else 0.0
    affected_fraction = float(cv2.countNonZero(accepted)) / float(max(1, h * w))
    severity = float(np.clip((0.6 * peak + 0.4 * affected_fraction) * 100.0, 0, 100))
    # A quiet, uniform map is treated as normal instead of classifying ordinary texture.
    if not boxes or float(np.percentile(heat, 99.5)) < 0.62:
        severity = min(severity, 24.9)
        defect_type = "No defect detected"
        verdict = "PASS"
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        largest = max(boxes, key=lambda b: b["area"])
        defect_type = classify_defect(gray, accepted, (largest["x"], largest["y"], largest["width"], largest["height"]))
        verdict = "PASS" if severity < 25 else "REWORK" if severity <= 65 else "SCRAP"
    confidence = calibrated_confidence(peak)
    color = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image, 0.58, color, 0.42, 0)
    annotated = overlay.copy()
    for box in boxes:
        x, y, bw, bh = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (40, 220, 255), 2)
    return {
        "heatmap": _encode_jpeg(overlay), "annotated_image": _encode_jpeg(annotated),
        "boxes": boxes, "severity": round(severity, 1), "confidence": confidence,
        "defect_type": defect_type, "verdict": verdict,
        "affected_area_pct": round(affected_fraction * 100, 3), "peak_intensity": round(peak, 4),
    }
