# -*- coding: utf-8 -*-
"""
Экспорт детектора YOLOv11-keypoints в TensorRT FP16.

На RTX 3070 (sm_86) engine обычно в 1.5–2 раза быстрее .pt+fp16 при том же recall.
Готовый .engine кладётся рядом с .pt — yolo_kp_detector подхватит его сам.

Требует TensorRT (пакет tensorrt + ultralytics export). Без него скрипт
завершится с понятным сообщением, детектор продолжит работать на .pt.

    python tools/export_yolo_trt.py
    python tools/export_yolo_trt.py --model yolov11m --imgsz 640
    python tools/export_yolo_trt.py --weights D:\\path\\yolov11x-keypoints.pt

Альтернатива без TensorRT: set NOMEROFF_YOLO=yolov11m  (легче, почти тот же recall
на кадре регистратора) и NOMEROFF_FP16=1 (уже по умолчанию).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "torch_models"))
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")


def _find_local_pt(model_name: str) -> Path | None:
    """Ищем уже скачанный .pt, чтобы не ходить в сеть на офлайн-ПК."""
    data = ROOT / "data" / "models" / "Detector"
    if not data.is_dir():
        return None
    for path in data.rglob("*.pt"):
        if model_name.lower() in path.name.lower():
            return path
    pts = list(data.rglob("*.pt"))
    return pts[0] if pts else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("NOMEROFF_YOLO", "yolov11x"),
                    help="имя модели modelhub (yolov11x/yolov11m/yolov11l)")
    ap.add_argument("--weights", default="", help="явный путь к .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else _find_local_pt(args.model)
    if weights is None or not weights.is_file():
        try:
            from nomeroff_net.tools.mcm import modelhub
            info = modelhub.download_model_by_name(args.model)
            weights = Path(info["path"])
        except Exception as ex:  # noqa: BLE001
            print(f"[X] Нет весов {args.model}: {ex}", file=sys.stderr)
            print("    Положите .pt в data/models/Detector или укажите --weights", file=sys.stderr)
            return 1

    out = weights.with_suffix(".engine")
    print(f"[>] export {weights} -> {out}  (FP16, imgsz={args.imgsz})")

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights))
        exported = model.export(
            format="engine",
            half=True,
            imgsz=args.imgsz,
            device=args.device,
            simplify=True,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"[X] TensorRT-экспорт не удался: {ex}", file=sys.stderr)
        print("    Нужен пакет tensorrt (или docker/tensorrt/Dockerfile_convert_ultralytics).", file=sys.stderr)
        print("    Без engine детектор работает на .pt + FP16 — это уже включено.", file=sys.stderr)
        return 2

    exported_path = Path(str(exported))
    if exported_path.is_file() and exported_path.resolve() != out.resolve():
        if out.exists():
            out.unlink()
        exported_path.replace(out)
    if out.is_file():
        print(f"[OK] {out}  ({out.stat().st_size / 1e6:.1f} МБ)")
        print("     При следующем старте OCR подхватит engine автоматически.")
        return 0
    print(f"[X] файл engine не появился: {exported}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
