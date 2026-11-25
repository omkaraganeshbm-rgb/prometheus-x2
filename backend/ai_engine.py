"""ONNX inference plus a dependency-light OpenCV scratch-sensitive fallback."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_PATH = Path(os.getenv("PROMETHEUS_MODEL", Path(__file__).parent / "model.onnx"))


class VisionEngine:
    def __init__(self) -> None:
        self.session = None
        self.input_name = None
        self.lock = threading.RLock()
        self.normal_bank: list[np.ndarray] = []
        self._load_model()

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            print(f"ONNX model could not be loaded ({exc}); using OpenCV fallback.")
            self.session = None

    @staticmethod
    def _embedding(gray: np.ndarray) -> np.ndarray:
        small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        hist /= max(float(hist.sum()), 1.0)
        return np.concatenate([small.flatten(), hist]).astype(np.float32)

    @staticmethod
    def _directional_map(gray: np.ndarray) -> np.ndarray:
        f = gray.astype(np.float32) / 255.0
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        # Thin, long edges are emphasized using directional gradients and Gabor responses.
        responses = [np.abs(gx), np.abs(gy)]
        for angle in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            kernel = cv2.getGaborKernel((15, 15), 3.2, angle, 7.0, 0.65, 0, ktype=cv2.CV_32F)
            responses.append(np.abs(cv2.filter2D(f, cv2.CV_32F, kernel)))
        stack = np.stack(responses)
        oriented = np.max(stack, axis=0)
        local_mean = cv2.GaussianBlur(f, (0, 0), 5)
        local_detail = np.abs(f - local_mean)
        score = 0.72 * oriented + 0.28 * local_detail
        score = cv2.GaussianBlur(score, (0, 0), 1.1)
        lo, hi = np.percentile(score, (45, 99.5))
        return np.clip((score - lo) / max(float(hi - lo), 1e-5), 0, 1).astype(np.float32)

    def inspect(self, image: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        if image is None or image.size == 0:
            raise ValueError("The uploaded image is empty or invalid.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        embedding = self._embedding(gray)
        edge_map = self._directional_map(gray)
        model_map = None
        if self.session is not None:
            try:
                rgb = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = np.transpose((rgb - np.array([0.485, 0.456, 0.406], np.float32)) /
                                      np.array([0.229, 0.224, 0.225], np.float32), (2, 0, 1))[None]
                outputs = self.session.run(None, {self.input_name: tensor})
                for output in outputs:
                    arr = np.asarray(output)
                    if arr.ndim >= 3 and arr.size > 256:
                        candidate = np.squeeze(arr)
                        if candidate.ndim == 2:
                            model_map = cv2.resize(candidate.astype(np.float32), (256, 256))
                            break
                if model_map is not None:
                    lo, hi = np.percentile(model_map, (1, 99))
                    model_map = np.clip((model_map - lo) / max(float(hi - lo), 1e-5), 0, 1)
            except Exception as exc:
                print(f"ONNX inference failed; using OpenCV fallback for this frame: {exc}")
        heatmap = edge_map if model_map is None else 0.65 * model_map + 0.35 * edge_map
        with self.lock:
            if self.normal_bank:
                distances = [float(np.mean(np.abs(embedding - item))) for item in self.normal_bank[-64:]]
                drift = min(distances)
            else:
                drift = 0.0
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "heatmap": np.clip(heatmap, 0, 1).astype(np.float32),
            "embedding": embedding,
            "drift": drift,
            "latency_ms": round(latency_ms, 2),
            "engine": "onnx+opencv" if model_map is not None else "opencv-fallback",
        }

    def remember_normal(self, embedding: np.ndarray) -> int:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self.lock:
            self.normal_bank.append(vector.copy())
            if len(self.normal_bank) > 256:
                self.normal_bank = self.normal_bank[-256:]
            return len(self.normal_bank)


engine = VisionEngine()
