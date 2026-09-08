import os
import torch
import numpy as np
from typing import List
from nomeroff_net.tools.mcm import (modelhub, get_device_torch)
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect


class Detector:
    """

    """
    @classmethod
    def get_classname(cls: object) -> str:
        return cls.__name__

    def __init__(self, numberplate_classes=None, yolo_model_type=None) -> None:
        self.model = None
        self.numberplate_classes = ["numberplate"]
        if numberplate_classes is not None:
            self.numberplate_classes = numberplate_classes
        self.device = get_device_torch()
        # NOMEROFF_YOLO=yolov11m|yolov11l|yolov11x — x самый тяжёлый в линейке (112 МБ);
        # на кадре регистратора m/l часто дают тот же recall в разы быстрее.
        self.yolo_model_type = yolo_model_type or os.environ.get("NOMEROFF_YOLO", "yolov11x")
        self.half = False

    def load_model(self, weights: str, device: str = '') -> None:
        from ultralytics import YOLO

        device = device or self.device
        # TensorRT FP16 (.engine рядом с .pt или NOMEROFF_YOLO_ENGINE) — если есть.
        # Иначе обычный .pt + half=True. Экспорт: python tools/export_yolo_trt.py
        engine = (os.environ.get("NOMEROFF_YOLO_ENGINE") or "").strip()
        if not engine:
            candidate = os.path.splitext(weights)[0] + ".engine"
            if os.path.isfile(candidate):
                engine = candidate
        load_path = engine if engine and os.path.isfile(engine) else weights
        if load_path != weights:
            print(f"[Detector] TensorRT engine: {load_path}")
        model = YOLO(load_path)
        if not str(load_path).endswith(".engine"):
            model.to(device)
        # FP16 поддерживается только на CUDA; на sm_86 даёт примерно двукратную
        # пропускную способность детектора. NOMEROFF_FP16=0 — выключить.
        self.half = (
            device != "cpu"
            and os.environ.get("NOMEROFF_FP16", "1").strip().lower() not in ("0", "false", "no", "off")
        )
        self.model = model
        self.device = device

    def load(self, path_to_model: str = "latest") -> None:
        if path_to_model == "latest":
            model_info = modelhub.download_model_by_name(self.yolo_model_type)
            path_to_model = model_info["path"]
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
        elif path_to_model.startswith("http"):
            model_info = modelhub.download_model_by_url(path_to_model, self.get_classname(), "numberplate_options")
            path_to_model = model_info["path"]
        elif path_to_model.startswith("modelhub://"):
            path_to_model = path_to_model.split("modelhub://")[1]
            model_info = modelhub.download_model_by_name(path_to_model)
            self.numberplate_classes = model_info.get("classes", self.numberplate_classes)
            path_to_model = model_info["path"]
        self.load_model(path_to_model)

    def convert_model_outputs_to_array(self, model_outputs):
        return [self.convert_model_output_to_array(model_output) for model_output in model_outputs]

    @staticmethod
    def convert_model_output_to_array(result):
        model_output = []
        for item, cls, conf, kps in zip(result.boxes.xyxy.cpu().numpy(),
                                        result.boxes.cls.cpu().numpy(),
                                        result.boxes.conf.cpu().numpy(),
                                        result.keypoints.xy.cpu().numpy()):
            model_output.append([item[0], item[1], item[2], item[3], conf, int(cls), normalize_rect(kps)])
        return model_output

    @torch.inference_mode()
    def predict(self, imgs: List[np.ndarray], min_accuracy: float = 0.4) -> np.ndarray or List:
        model_outputs = self.model(imgs, conf=min_accuracy, verbose=False, save=False, save_txt=False, show=False,
                                   iou=0.7, agnostic_nms=True, half=self.half)
        return self.convert_model_outputs_to_array(model_outputs)
