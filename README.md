# PROMETHEUS-X | Industrial Quality Assurance

A runnable full-stack demonstration of image-based surface anomaly inspection, human review, batch audit, and serial certificate workflows. It runs immediately in OpenCV fallback mode; placing a trained model at `backend/model.onnx` enables ONNX inference. The fallback is a general-purpose Sobel/Gabor heuristic, **not a validated production inspection model**. Validate thresholds, calibrate confidence against labeled factory data, and perform safety/quality qualification before using decisions operationally.

## Requirements

- Python 3.10+ and pip
- Optional: a webcam connected to the machine running the API
- Optional training: a GPU-enabled Colab/runtime for Anomalib and the MVTec AD dataset

## Run locally

From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000>. API docs: <http://127.0.0.1:8000/docs>. The SQLite database is created and seeded on first startup at `backend/data/prometheus.db` (override with `PROMETHEUS_DB`). No frontend build step is required. Internet access is needed for Tailwind and Chart.js CDN assets; the API itself does not depend on them.

## Train/export PatchCore (optional)

Download and arrange MVTec AD so `metal_nut` is available under `./datasets/MVTec`, then run:

```bash
pip install anomalib==1.2.0 lightning
python train_colab.py --data-root ./datasets/MVTec --output backend/model.onnx
```

The training script fits PatchCore using the MVTec `metal_nut` category with the normal training split (`train/good`), `wide_resnet50_2`, `layer2`/`layer3`, and a 0.1 coreset sampling ratio. The resulting ONNX file is loaded automatically on the next server start. Export APIs can vary between Anomalib releases; the script targets Anomalib 1.2.0. For deployment, verify the exported model's input preprocessing and output map against your installed Anomalib version.

## Workflows

- **Factory Console:** upload an image or use camera capture, inspect the heatmap/score, adjust simulated image conditions, and approve a borderline normal embedding (45–65% confidence) into a bounded in-memory reference bank.
- **B2B Audit:** review seeded batches and latest inspection records.
- **Customer:** look up seeded serial IDs (`PX-2026-000184`, `PX-2026-000185`, `PX-2026-000276`) or run a post-delivery self-check upload.

## API

- `GET /api/health`
- `POST /api/inspect` multipart fields: `file`, optional `serial_id`, `batch_id`, `auto_stabilize`
- `POST /api/customer/verify` multipart `file`
- `POST /api/hitl/approve` JSON `{ "embedding": [ ... ], "confidence": 52 }`
- `GET /api/batches`
- `GET /api/certificate/{serial_id}`
- `GET /api/video_feed` MJPEG from the server's local webcam (if present)

Inspection outputs include verdict, severity, confidence, heatmap/annotated JPEG data URLs, boxes, timings, drift indicators, and the feature vector needed for HITL approval. Calibration is a transparent logistic transform, not Platt scaling fitted on a labeled CutPaste dataset; true calibration requires representative labeled examples. HITL references are process-memory only and reset on server restart. Seeded certificates and batch numbers are demo data.
