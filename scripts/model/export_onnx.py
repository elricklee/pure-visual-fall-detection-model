from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX with various configurations.")
    parser.add_argument("--model", required=True, help="Input PyTorch model path (.pt).")
    parser.add_argument("--output-dir", default="artifacts/onnx", help="Output directory for ONNX models.")
    parser.add_argument("--imgsz", type=int, default=384, help="Input image size (square).")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes.")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--nms", action="store_true", help="Include NMS in the model.")
    parser.add_argument("--half", action="store_true", help="Export in FP16 precision.")
    parser.add_argument("--device", default=None, help="Device to use for export.")
    parser.add_argument("--config", help="Optional JSON config file for multiple export variants.")
    return parser.parse_args()


def export_single(
    model_path: Path,
    output_path: Path,
    imgsz: int,
    batch: int,
    dynamic: bool,
    simplify: bool,
    opset: int,
    nms: bool,
    half: bool,
    device: str | None,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
        nms=nms,
        half=half,
        device=device,
        name=output_path.stem,
        project=str(output_path.parent),
        exist_ok=True,
    )

    exported_path = Path(str(results))
    target_path = output_path.with_suffix(".onnx")
    if exported_path.exists() and exported_path != target_path:
        exported_path.replace(target_path)

    return {
        "model_path": str(target_path),
        "imgsz": imgsz,
        "batch": batch,
        "dynamic": dynamic,
        "simplify": simplify,
        "opset": opset,
        "nms": nms,
        "half": half,
        "device": device,
        "file_size_mb": round(target_path.stat().st_size / 1_000_000, 4) if target_path.exists() else None,
    }


def load_config(config_path: str) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        configs = load_config(args.config)
        export_logs = []
        for idx, cfg in enumerate(configs):
            model_name = cfg.get("name", f"model_{idx}")
            output_path = output_dir / model_name
            log = export_single(
                model_path=Path(cfg["model"]),
                output_path=output_path,
                imgsz=cfg.get("imgsz", args.imgsz),
                batch=cfg.get("batch", args.batch),
                dynamic=cfg.get("dynamic", args.dynamic),
                simplify=cfg.get("simplify", args.simplify),
                opset=cfg.get("opset", args.opset),
                nms=cfg.get("nms", args.nms),
                half=cfg.get("half", args.half),
                device=cfg.get("device", args.device),
            )
            export_logs.append(log)
            print(f"Exported: {log['model_path']} ({log['file_size_mb']} MB)")

        log_path = output_dir.parent / "model_export_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(export_logs, f, ensure_ascii=False, indent=2)
        print(f"\nExport log saved to: {log_path}")
    else:
        base_name = Path(args.model).stem
        suffix_parts = []
        if args.half:
            suffix_parts.append("fp16")
        else:
            suffix_parts.append("fp32")
        if args.dynamic:
            suffix_parts.append("dynamic")
        suffix = "_".join(suffix_parts)
        output_path = output_dir / f"{base_name}_{args.imgsz}_{suffix}"

        log = export_single(
            model_path=Path(args.model),
            output_path=output_path,
            imgsz=args.imgsz,
            batch=args.batch,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=args.opset,
            nms=args.nms,
            half=args.half,
            device=args.device,
        )
        print(f"Exported: {log['model_path']} ({log['file_size_mb']} MB)")

        log_path = output_dir.parent / "model_export_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([log], f, ensure_ascii=False, indent=2)
        print(f"Export log saved to: {log_path}")


if __name__ == "__main__":
    main()
