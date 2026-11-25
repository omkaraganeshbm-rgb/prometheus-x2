"""PROMETHEUS-X API server."""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import database
from ai_engine import engine
from guardrails import inject_normal_memory, shift_guard
from postprocess import postprocess

app = FastAPI(title="PROMETHEUS-X Quality Assurance API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


class Approval(BaseModel):
    embedding: list[float]
    confidence: float


@app.on_event("startup")
def startup() -> None:
    database.initialize()


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "onnx+opencv" if engine.session else "opencv-fallback", "model_loaded": engine.session is not None}


async def inspect_bytes(contents: bytes, serial_id: Optional[str] = None, batch_id: Optional[str] = None, auto_stabilize: bool = True) -> dict:
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise HTTPException(400, "Upload must be a non-empty image smaller than 20 MB.")
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Unsupported image. Upload a JPEG, PNG, BMP, or WebP image.")
    inference = engine.inspect(image)
    guard = shift_guard.assess(image, inference["heatmap"], inference["drift"], auto_stabilize)
    if guard["auto_stabilized"]:
        stable = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), cv2.COLOR_LAB2BGR)
        # Apply CLAHE to the image actually used to render overlays when shift is detected.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    result = postprocess(image, inference["heatmap"])
    result.update({
        "latency_ms": inference["latency_ms"], "engine": inference["engine"],
        "embedding": inference["embedding"].tolist(), "embedding_drift": round(inference["drift"], 4),
        **guard,
    })
    database.add_inspection(result, serial_id, batch_id)
    return result


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...), serial_id: str | None = Form(None), batch_id: str | None = Form(None), auto_stabilize: bool = Form(True)):
    return await inspect_bytes(await file.read(), serial_id, batch_id, auto_stabilize)


@app.post("/api/hitl/approve")
def approve(payload: Approval):
    try:
        size = inject_normal_memory(engine, np.asarray(payload.embedding, dtype=np.float32), payload.confidence)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"approved": True, "memory_bank_size": size, "message": "Normal reference added to the active in-memory bank."}


@app.get("/api/batches")
def batches():
    return {"batches": database.list_batches(), "recent_inspections": database.recent_inspections()}


@app.get("/api/certificate/{serial_id}")
def certificate(serial_id: str):
    item = database.certificate(serial_id)
    if not item:
        raise HTTPException(404, "No verified certificate was found for this serial ID.")
    return {"certificate": item, "verified": True, "issuer": "PROMETHEUS-X Factory QA"}


@app.post("/api/customer/verify")
async def customer_verify(file: UploadFile = File(...)):
    result = await inspect_bytes(await file.read(), auto_stabilize=True)
    result["claim_recommendation"] = "Review with the seller" if result["verdict"] != "PASS" else "No visible post-delivery surface anomaly detected"
    return result


@app.get("/api/video_feed")
def video_feed():
    def frames():
        camera = cv2.VideoCapture(0)
        try:
            while True:
                ok, frame = camera.read() if camera.isOpened() else (False, None)
                if not ok:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera unavailable - use browser camera or upload", (18, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2)
                    ok, encoded = cv2.imencode(".jpg", frame)
                else:
                    ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
                time.sleep(0.08)
        finally:
            camera.release()
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if FRONTEND.exists():
    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(FRONTEND / "index.html")
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
