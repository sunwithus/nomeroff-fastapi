# -*- coding: utf-8 -*-
"""
Подготовка кадра регистратора для OCR: ROI дороги, контраст, инверсия.

Раньше это жило только в C# (Nomeroff.Video.Api/PlateFramePrep.cs) и работало
только на видео-пути, поэтому живая камера читала номера заметно хуже.
Здесь один код на оба пути; варианты применяются к уже декодированному кадру
(BGR ndarray), без промежуточной перекодировки в JPEG.

Каждый вариант возвращает не только картинку, но и аффинное преобразование
(scale_x, scale_y, offset_x, offset_y), чтобы bbox можно было отобразить
обратно в координаты исходного кадра — иначе в БД уезжает кроп негатива
или растянутого ROI вместо настоящего номера.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameVariant:
    """Кадр после предобработки + обратное отображение bbox в исходный кадр."""

    name: str
    image: np.ndarray
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: int = 0
    offset_y: int = 0
    # Инверсия/контраст меняют яркость: кроп для БД надо брать из оригинала
    inverted: bool = False

    def bbox_to_source(self, bbox) -> list[int]:
        """[x1,y1,x2,y2] в координатах варианта -> координаты исходного кадра."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        return [
            int(round(x1 / self.scale_x)) + self.offset_x,
            int(round(y1 / self.scale_y)) + self.offset_y,
            int(round(x2 / self.scale_x)) + self.offset_x,
            int(round(y2 / self.scale_y)) + self.offset_y,
        ]


FRAME_VARIANTS = ("full", "crop", "roi", "roi_contrast", "roi_invert")


def crop_bottom(frame: np.ndarray, bottom_ratio: float = 0.12) -> FrameVariant:
    """Отрезать нижнюю полосу с OSD/капотом."""
    h, w = frame.shape[:2]
    cut = min(h - 40, max(0, int(h * bottom_ratio)))
    if cut <= 0:
        return FrameVariant("crop", frame)
    return FrameVariant("crop", frame[: h - cut, :])


def road_roi_upscaled(
    frame: np.ndarray,
    top_ratio: float = 0.22,
    bottom_crop_ratio: float = 0.14,
    side_crop_ratio: float = 0.12,
    scale: float = 1.5,
    name: str = "roi",
) -> FrameVariant:
    """Центрально-нижняя зона дороги без капота/OSD, увеличенная — мелкие номера вдалеке."""
    h, w = frame.shape[:2]
    y0 = int(np.clip(h * top_ratio, 0, max(0, h - 80)))
    y1 = int(np.clip(h - h * bottom_crop_ratio, y0 + 40, h))
    x0 = int(np.clip(w * side_crop_ratio, 0, w // 2))
    x1 = int(np.clip(w - w * side_crop_ratio, x0 + 40, w))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return FrameVariant(name, frame)

    roi = frame[y0:y1, x0:x1]
    if scale > 1.01:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
    return FrameVariant(name, roi, scale_x=scale, scale_y=scale, offset_x=x0, offset_y=y0)


def contrast_boost(variant: FrameVariant, contrast: float = 1.25, brightness: float = 6.0) -> FrameVariant:
    """Лёгкий контраст для тёмных/военных номеров."""
    boosted = cv2.convertScaleAbs(variant.image, alpha=contrast, beta=brightness)
    return FrameVariant(
        f"{variant.name}_contrast", boosted,
        variant.scale_x, variant.scale_y, variant.offset_x, variant.offset_y,
    )


def invert(variant: FrameVariant) -> FrameVariant:
    """Негатив — военные номера (белый на чёрном)."""
    return FrameVariant(
        f"{variant.name}_invert", cv2.bitwise_not(variant.image),
        variant.scale_x, variant.scale_y, variant.offset_x, variant.offset_y,
        inverted=True,
    )


def build_variants(
    frame: np.ndarray,
    names: list[str] | tuple[str, ...] = ("full",),
    *,
    bottom_crop_ratio: float = 0.12,
) -> list[FrameVariant]:
    """Собрать запрошенные варианты кадра. Неизвестные имена игнорируются."""
    wanted = [n.strip().lower() for n in names if n and n.strip()]
    if not wanted:
        wanted = ["full"]

    roi: FrameVariant | None = None

    def get_roi() -> FrameVariant:
        nonlocal roi
        if roi is None:
            roi = road_roi_upscaled(frame, bottom_crop_ratio=max(bottom_crop_ratio, 0.12))
        return roi

    out: list[FrameVariant] = []
    for name in wanted:
        if name == "full":
            out.append(FrameVariant("full", frame))
        elif name == "crop":
            out.append(crop_bottom(frame, bottom_crop_ratio))
        elif name == "roi":
            out.append(get_roi())
        elif name == "roi_contrast":
            out.append(contrast_boost(get_roi()))
        elif name == "roi_invert":
            out.append(invert(get_roi()))
    return out or [FrameVariant("full", frame)]


def crop_bbox(frame: np.ndarray, bbox, pad: int = 8) -> np.ndarray | None:
    """Кроп по bbox исходного кадра с небольшим padding."""
    if frame is None or bbox is None or len(bbox) < 4:
        return None
    h, w = frame.shape[:2]
    x1, x2 = sorted((int(bbox[0]), int(bbox[2])))
    y1, y2 = sorted((int(bbox[1]), int(bbox[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
