"""Train a PatchCore model on the good training split of MVTec AD metal_nut.

Install the training extras separately from the lightweight inference requirements:
  pip install anomalib==1.2.0 lightning
  python train_colab.py --data-root ./datasets/MVTec

The exported ONNX model is written to backend/model.onnx. The backend remains
runnable without this model and uses its OpenCV fallback detector instead.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("./datasets/MVTec"))
    parser.add_argument("--output", type=Path, default=Path("backend/model.onnx"))
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    try:
        from anomalib.data import MVTec
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        from anomalib.deploy import ExportType
    except ImportError as exc:
        raise SystemExit("Training requires anomalib and lightning. Install anomalib==1.2.0 first.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_kwargs = {"root": str(args.data_root), "category": "metal_nut"}
    # Keep only train/good for fitting. MVTec's test split remains available for
    # evaluation, but no anomalous/test images are used to construct the memory bank.
    if "train_batch_size" in inspect.signature(MVTec).parameters:
        data_kwargs["train_batch_size"] = 16
    datamodule = MVTec(**data_kwargs)
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )
    engine = Engine(default_root_dir=str(args.output.parent / "training"))
    engine.fit(model=model, datamodule=datamodule)

    # Anomalib 1.2 exports the fitted model via Engine.export. ExportType is
    # used explicitly so the resulting artifact is consumable by ONNX Runtime.
    try:
        result = engine.export(model=model, export_type=ExportType.ONNX)
    except TypeError:
        result = engine.export(model=model, export_type="onnx")
    candidates = [Path(result)] if result else []
    candidates += list((args.output.parent / "training").rglob("*.onnx"))
    candidates += list(args.output.parent.rglob("*.onnx"))
    source = next((p for p in candidates if p.exists() and p.resolve() != args.output.resolve()), None)
    if source is None:
        raise RuntimeError("Anomalib completed training but did not produce an ONNX file.")
    if source.resolve() != args.output.resolve():
        args.output.write_bytes(source.read_bytes())
    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()
