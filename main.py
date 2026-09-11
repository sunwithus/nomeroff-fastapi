# -*- coding: utf-8 -*-
# main.py - only plate OCR. Watchlist/dedup/alerts live in C#.
import os
import re
import sys
import subprocess
import tempfile
import time
import base64
import logging
from contextlib import asynccontextmanager, contextmanager

# === Env до импорта Nomeroff / GitPython / torch ===
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_HOME", os.path.join(_ROOT, "torch_models"))
# На офлайн-ПК часто нет git.exe — modelhub_client тянет GitPython при импорте
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
# Не дергать albumentations за версией (офлайн → warning/ошибка соединения)
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# NOMEROFF_DEVICE=auto|cuda|cpu  (auto = CUDA если доступна и поддерживается)
os.environ.setdefault("NOMEROFF_DEVICE", "auto")
# NOMEROFF_TWO_LINE=1 — дочитывать квадратные/двухстрочные номера (eu_2lines)
os.environ.setdefault("NOMEROFF_TWO_LINE", "1")

# PyTorch cu118: минимум sm_37. GT 710 = 3.5 → CUDNN_STATUS_NOT_SUPPORTED_ARCH_MISMATCH.
# Nomeroff берёт device через torch.cuda.is_available() при импорте — прячем GPU ДО импорта.
_MIN_CUDA_CAP = (3, 7)


def _probe_cuda_usable() -> tuple[bool, str, str]:
    """Подпроцесс: (cuda_ok, gpu_name, note). Не инициализирует CUDA в этом процессе."""
    code = (
        "import sys\n"
        "try:\n"
        " import torch\n"
        " if not torch.cuda.is_available():\n"
        "  print('0||no_cuda')\n"
        "  sys.exit(0)\n"
        " cap = torch.cuda.get_device_capability(0)\n"
        " name = torch.cuda.get_device_name(0)\n"
        " ok = 1 if cap >= (3, 7) else 0\n"
        " print(f'{ok}|{name}|sm_{cap[0]}{cap[1]}')\n"
        "except Exception as e:\n"
        " print(f'0||err:{e}')\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ},
        )
        line = (r.stdout or "").strip().splitlines()
        line = line[-1] if line else "0||"
        parts = line.split("|", 2)
        ok = parts[0].strip() == "1"
        name = parts[1].strip() if len(parts) > 1 else ""
        note = parts[2].strip() if len(parts) > 2 else ""
        return ok, name, note
    except Exception as e:
        return False, "", f"probe_failed:{e}"


def _apply_device_policy_before_torch() -> str:
    """
    Возвращает 'cuda'|'cpu'. При CPU выставляет CUDA_VISIBLE_DEVICES='' до импорта torch.
    """
    prefer = (os.environ.get("NOMEROFF_DEVICE") or "auto").strip().lower()
    if prefer == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    cuda_ok, gpu_name, note = _probe_cuda_usable()
    if cuda_ok and prefer in ("auto", "cuda", ""):
        return "cuda"

    # CUDA нет или GPU слишком старый (например GT 710 sm_35)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if prefer == "cuda":
        # иначе Nomeroff всё равно возьмёт cuda через is_available()
        os.environ["NOMEROFF_DEVICE"] = "cpu"
    # сохраним причину для лога после basicConfig
    reason = gpu_name or note or "cuda_unavailable"
    os.environ["NOMEROFF_DEVICE_FALLBACK_REASON"] = reason
    return "cpu"


_DEVICE_POLICY = _apply_device_policy_before_torch()

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nomeroff_net import pipeline
from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import normalize_rect_new
from nomeroff_net.tools.image_processing import (crop_number_plate_roi_zones_from_images,
                                                crop_number_plate_zones_from_images)

import plate_ru
import vlm_arbiter
from frame_prep import FRAME_VARIANTS, bbox_looks_two_line, build_variants, crop_bbox
from infer_batch import yolo_chunk_size

# === Логирование ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
_fb = os.environ.get("NOMEROFF_DEVICE_FALLBACK_REASON")
if _DEVICE_POLICY == "cpu" and _fb:
    logger.info(
        "GPU не используется (%s) — PyTorch требует CUDA capability >= %s.%s. Работа на CPU.",
        _fb, _MIN_CUDA_CAP[0], _MIN_CUDA_CAP[1],
    )

# === Глобальное состояние ===
pipeline_instance = None
_runtime_device = "cpu"  # "cuda" | "cpu"
_runtime_device_name = ""
# После CUDA OOM уменьшаем чанк до размера, который прошёл.
_yolo_chunk_cap: int | None = None

# Квадратные РФ-номера: высота/ширина кропа заметно больше, чем у длинных 520×112
_SQUARE_ASPECT_MIN = 0.42
_RU_LETTERS = plate_ru.RU_LETTERS
_RU_CIV = plate_ru.RU_CIVILIAN
_RU_MIL = plate_ru.RU_MILITARY

# --- Гейт по геометрии bbox (до OCR) ---
# Отсекаем детекции, которые физически не могут быть читаемым номером:
# выше линии горизонта, слишком мелкие, с неправдоподобным аспектом,
# и попадающие в полосу OSD/капота внизу кадра.
_GATE_MIN_HEIGHT_PX = int(os.environ.get("NOMEROFF_GATE_MIN_HEIGHT", "18"))
_GATE_MIN_ASPECT = float(os.environ.get("NOMEROFF_GATE_MIN_ASPECT", "1.2"))
_GATE_MAX_ASPECT = float(os.environ.get("NOMEROFF_GATE_MAX_ASPECT", "7.0"))
_GATE_HORIZON_RATIO = float(os.environ.get("NOMEROFF_GATE_HORIZON", "0.28"))
_GATE_OSD_RATIO = float(os.environ.get("NOMEROFF_GATE_OSD", "0.94"))


def bbox_gate(bbox, frame_shape) -> str | None:
    """
    Проверить геометрию детекции. Возвращает причину отбраковки или None, если бокс годный.

    Аспект проверяется только для «длинных» боксов: квадратные/двухстрочные номера
    имеют аспект ~1.5 и ниже, их пропускает отдельная ветка _zone_is_square.
    """
    if bbox is None or len(bbox) < 4:
        return "no_bbox"
    try:
        h_img, w_img = int(frame_shape[0]), int(frame_shape[1])
        x1, x2 = sorted((float(bbox[0]), float(bbox[2])))
        y1, y2 = sorted((float(bbox[1]), float(bbox[3])))
    except (TypeError, ValueError, IndexError):
        return "bad_bbox"

    bw, bh = x2 - x1, y2 - y1
    if bh < _GATE_MIN_HEIGHT_PX or bw < _GATE_MIN_HEIGHT_PX:
        return f"too_small({bw:.0f}x{bh:.0f})"
    if h_img > 0 and (y1 + y2) / 2.0 < h_img * _GATE_HORIZON_RATIO:
        return "above_horizon"
    if h_img > 0 and (y1 + y2) / 2.0 > h_img * _GATE_OSD_RATIO:
        return "osd_or_hood"
    aspect = bw / bh if bh > 0 else 0.0
    # 0.3..1.2 — квадратный/двухстрочный, это допустимо
    if aspect > _GATE_MAX_ASPECT or (0.3 < aspect < _GATE_MIN_ASPECT and aspect < 0.9):
        return f"bad_aspect({aspect:.2f})"
    if w_img > 0 and (x2 < 0 or x1 > w_img):
        return "outside_frame"
    return None


def _resolve_torch_device() -> tuple[str, str]:
    """
    Фактическое устройство после политики CUDA_VISIBLE_DEVICES.
    Возвращает (device, human_name), device = 'cuda' | 'cpu'.
    """
    prefer = (os.environ.get("NOMEROFF_DEVICE") or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        logger.warning("torch не установлен")
        return "cpu", ""

    cuda_ok = bool(torch.cuda.is_available())
    if prefer == "cpu" or not cuda_ok:
        if prefer in ("auto", "cuda") and not cuda_ok:
            logger.info(
                "CUDA скрыта/недоступна (старый GPU или NOMEROFF_DEVICE=cpu) — работа на CPU"
            )
        return "cpu", ""

    # cuda доступна и разрешена
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        name, cap = "CUDA", (0, 0)
    if cap < _MIN_CUDA_CAP:
        logger.warning(
            "GPU %s capability %s.%s < %s.%s — CPU",
            name, cap[0], cap[1], _MIN_CUDA_CAP[0], _MIN_CUDA_CAP[1],
        )
        return "cpu", ""
    try:
        torch.cuda.set_device(0)
    except Exception as ex:
        logger.warning("torch.cuda.set_device(0) failed: %s — CPU", ex)
        return "cpu", ""
    return "cuda", name

# Алфавит российских номеров (глифы как латиница в OCR Nomeroff → кириллица для БД/UI)
_LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
})


def latin_to_cyrillic(plate: str) -> str:
    """Nomeroff RU OCR выдаёт латиницу; для хранения/отображения — кириллица."""
    if not plate:
        return plate
    return plate.upper().translate(_LATIN_TO_CYRILLIC)


def _normalize_plate_text(plate: str) -> str:
    # двухстрочный OCR иногда склеивает с пробелом/переносом
    return plate_ru.normalize_plate(plate)


def _looks_like_ru_plate(plate: str) -> bool:
    return plate_ru.looks_like_ru_plate(plate)


def _two_line_enabled() -> bool:
    return (os.environ.get("NOMEROFF_TWO_LINE") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _zone_is_square(zone) -> bool:
    """Квадратный / двухстрочный номер — относительно высокий кроп."""
    try:
        h, w = int(zone.shape[0]), int(zone.shape[1])
    except Exception:
        return False
    if w <= 0 or h <= 0:
        return False
    return (h / float(w)) >= _SQUARE_ASPECT_MIN


def _needs_two_line(zone, bbox=None) -> bool:
    """Две строки: высокий кроп или высокий bbox до сплющивания warp'ом."""
    if not _two_line_enabled():
        return False
    return _zone_is_square(zone) or bbox_looks_two_line(bbox)


def _fp16_enabled() -> bool:
    """fp16 только на CUDA и только если явно не выключен."""
    if _runtime_device != "cuda":
        return False
    return (os.environ.get("NOMEROFF_FP16") or "1").strip().lower() not in ("0", "false", "no", "off")


@contextmanager
def _inference_ctx():
    """
    inference_mode + autocast fp16 вокруг инференса.

    inference_mode дешевле no_grad (не ведёт version counter), autocast на sm_86
    примерно вдвое поднимает пропускную способность и детектора, и OCR.
    """
    try:
        import torch
    except ImportError:
        yield
        return

    with torch.inference_mode():
        if _fp16_enabled():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield


def _bgr_to_jpeg_b64(img, quality: int = 92) -> str | None:
    """JPEG base64 произвольного BGR-кропа."""
    if img is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
    except Exception:
        return None


def _zone_to_jpeg_b64(zone) -> str | None:
    """JPEG base64 кропа номера (зона детектора) для сохранения в БД / просмотрщик."""
    if zone is None:
        return None
    try:
        img = zone
        if hasattr(zone, "shape") and len(zone.shape) == 3 and zone.shape[2] == 3:
            # Nomeroff zones обычно RGB
            img = cv2.cvtColor(zone, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _bbox_to_jpeg_b64(frame_bgr, bbox, pad: int = 8) -> str | None:
    """Fallback-кроп по bbox той же детекции, если zone пустая."""
    if frame_bgr is None or bbox is None or len(bbox) < 4:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _zone_pad(zone, frac: float = 0.14):
    """Расширить кроп — у квадратных часто обрезается правый край региона."""
    try:
        h, w = zone.shape[:2]
        py, px = max(1, int(h * frac)), max(1, int(w * frac))
        return cv2.copyMakeBorder(zone, py, py, px, px, cv2.BORDER_REPLICATE)
    except Exception:
        return zone


def _zone_upscale(zone, scale: float = 2.0):
    try:
        return cv2.resize(
            zone, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    except Exception:
        return zone


def _zone_contrast(zone):
    try:
        lab = cv2.cvtColor(zone, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
    except Exception:
        return zone


def _zone_unsqueeze_2line(zone, target_aspect: float = 0.75, min_width: int = 280):
    """
    Детектор часто сплющивает квадратный номер в широкую полосу (aspect~0.2).
    Вертикально разжимаем до ~квадрата и увеличиваем для OCR.
    """
    try:
        h, w = int(zone.shape[0]), int(zone.shape[1])
    except Exception:
        return zone
    if w <= 0 or h <= 0:
        return zone
    aspect = h / float(w)
    new_w = w
    new_h = h
    if aspect < _SQUARE_ASPECT_MIN:
        new_h = max(h + 1, int(round(w * target_aspect)))
    if new_w < min_width:
        scale = min_width / float(new_w)
        new_w = min_width
        new_h = max(1, int(round(new_h * scale)))
    if new_w == w and new_h == h:
        return zone
    try:
        return cv2.resize(zone, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return zone


def _assemble_rapid_2line(ocr_lines) -> str:
    """Собрать РФ двухстрочный номер из фрагментов RapidOCR (T, 314, XC, 125)."""
    items = []
    for line in ocr_lines or []:
        if not line or len(line) < 2:
            continue
        box, text = line[0], str(line[1] or "").strip()
        if not text or not box:
            continue
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        except Exception:
            continue
        items.append((cy, cx, text))
    if not items:
        return ""
    ys = [t[0] for t in items]
    mid = (min(ys) + max(ys)) / 2.0
    top = sorted([t for t in items if t[0] <= mid], key=lambda t: t[1])
    bot = sorted([t for t in items if t[0] > mid], key=lambda t: t[1])
    # если всё в одной полосе — просто слева направо
    if not top or not bot:
        ordered = sorted(items, key=lambda t: (t[0], t[1]))
        return _normalize_plate_text("".join(t[2] for t in ordered))
    return _normalize_plate_text(
        "".join(t[2] for t in top) + "".join(t[2] for t in bot)
    )


def _rapidocr_2line_plate(zone) -> str:
    """Fallback: RapidOCR по разжатому кропу квадратного номера."""
    try:
        ocr = _get_rapidocr()
    except Exception as ex:
        logger.warning("RapidOCR 2-line unavailable: %s", ex)
        return ""
    img = _zone_unsqueeze_2line(zone)
    # RapidOCR ждёт BGR/файл; зоны Nomeroff — RGB
    try:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        bgr = img
    try:
        result, _ = ocr(bgr)
    except Exception as ex:
        logger.warning("RapidOCR 2-line failed: %s", ex)
        return ""
    plate = _assemble_rapid_2line(result)
    if plate:
        logger.info("RapidOCR 2-line: %r from %s fragments", plate, len(result or []))
    return plate


def _ocr_zones_lines_once(
    detector,
    zones: list,
    lines_count: int,
    label: str = "ru",
) -> list[tuple[str, list[float]]]:
    """
    Проход OCR с заданным числом строк; postprocess склеивает multiline.
    Возвращает (текст, вероятности по символам) — уверенность берётся из CTC-головы.
    """
    n = len(zones)
    if n == 0:
        return []
    labels = [label] * n
    lines = [int(lines_count)] * n
    model_inputs = detector.preprocess(
        zones, [None] * n, labels=labels, lines=lines
    )
    model_outputs = detector.forward(model_inputs)
    texts = detector.postprocess(model_outputs)
    meta = list(getattr(detector, "last_ocr_meta", []) or [])
    if not isinstance(texts, list):
        return [("", [])] * n

    out: list[tuple[str, list[float]]] = []
    for i, t in enumerate(texts):
        text = t if isinstance(t, str) else (str(t) if t is not None else "")
        probs = meta[i].get("char_probs", []) if i < len(meta) else []
        out.append((text, list(probs)))
    while len(out) < n:
        out.append(("", []))
    return out[:n]


def _ocr_zone_halves_1line(detector, zone) -> tuple[str, list[float]]:
    """
    Квадратный номер: верх/низ по отдельности однострочной RU-моделью, затем склейка.
    Часто точнее, чем eu_2lines на регионе (…125).
    """
    try:
        from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (
            split_numberplate,
        )
    except Exception:
        return "", []
    try:
        parts = split_numberplate(zone, 2)
    except Exception:
        return "", []
    if len(parts) < 2:
        return "", []
    # чуть растянуть каждую половину по ширине — регион на нижней строке
    stretched = []
    for part in parts:
        try:
            h, w = part.shape[:2]
            stretched.append(
                cv2.resize(part, (max(w, int(w * 1.35)), h), interpolation=cv2.INTER_CUBIC)
            )
        except Exception:
            stretched.append(part)
    try:
        halves = _ocr_zones_lines_once(detector, stretched, 1)
    except Exception:
        return "", []
    if len(halves) < 2:
        return "", []
    (top_text, top_probs), (bottom_text, bottom_probs) = halves[0], halves[1]
    joined = _normalize_plate_text(top_text) + _normalize_plate_text(bottom_text)
    return joined, list(top_probs) + list(bottom_probs)


def _score_plate_candidate(text: str) -> tuple:
    """Выше — лучше. Формат > существующий код региона > длина региона (125 vs 12)."""
    p = _normalize_plate_text(text)
    if not p:
        return (0, 0, 0, 0)
    ok = 1 if _looks_like_ru_plate(p) else 0
    region_ok = 1 if plate_ru.region_is_valid(p) else 0
    # гражданский с 3 цифрами региона чуть предпочтительнее, чем с 2
    region3 = 1 if (_RU_CIV.match(p) and len(p) >= 9) else 0
    return (ok, region_ok, region3, len(p))


def _reread_zones_2line(zones: list) -> list[tuple[str, list[float]]]:
    """OCR кропов двухстрочной моделью (eu_2lines для label=ru, lines=2)."""
    if not zones or pipeline_instance is None:
        return []
    try:
        detector = pipeline_instance.number_plate_text_reading.detector
    except Exception as ex:
        logger.warning("2-line OCR: нет text detector: %s", ex)
        return [("", [])] * len(zones)

    # Варианты: оригинал / разжатый / pad / contrast (eu_2lines)
    variant_lists: list[list] = []
    for zone in zones:
        unsqueezed = _zone_unsqueeze_2line(zone)
        variants = [zone, unsqueezed, _zone_pad(unsqueezed), _zone_contrast(unsqueezed)]
        variant_lists.append(variants)

    flat_zones = []
    owners: list[int] = []
    for i, variants in enumerate(variant_lists):
        for v in variants:
            flat_zones.append(v)
            owners.append(i)

    best: list[tuple[str, list[float]]] = [("", [])] * len(zones)
    best_score = [(-1, -1, -1, -1)] * len(zones)

    try:
        for (text, probs), owner in zip(_ocr_zones_lines_once(detector, flat_zones, 2), owners):
            sc = _score_plate_candidate(text)
            if sc > best_score[owner]:
                best_score[owner] = sc
                best[owner] = (text, probs)
    except Exception as ex:
        logger.warning("2-line OCR failed: %s", ex)

    # RapidOCR на разжатом кропе — лучше читает регион 125 у квадратных
    # при равном score предпочитаем Rapid (eu_2lines часто даёт …12 / …122).
    # У RapidOCR своей CTC-уверенности нет — отдаём пустые веса (нейтральный голос).
    for i, zone in enumerate(zones):
        rapid = _rapidocr_2line_plate(zone)
        sc = _score_plate_candidate(rapid)
        if sc > best_score[i] or (sc == best_score[i] and sc[0] == 1 and rapid):
            best_score[i] = sc
            best[i] = (rapid, [])

    # Если всё ещё нет валидного кода региона — половины 1-line
    for i, zone in enumerate(zones):
        if best_score[i][0] == 1 and best_score[i][1] == 1:
            continue
        try:
            joined, probs = _ocr_zone_halves_1line(detector, _zone_unsqueeze_2line(zone))
        except Exception:
            joined, probs = "", []
        sc = _score_plate_candidate(joined)
        if sc > best_score[i]:
            best_score[i] = sc
            best[i] = (joined, probs)
            logger.info("2-line halves OCR zone#%s: %r score=%s", i, joined, sc)

    return best


def _pick_better_plate(
    reading_1line: tuple[str, list[float]],
    reading_2line: tuple[str, list[float]],
    *,
    prefer_2line: bool,
) -> tuple[str, list[float]]:
    a_text, a_probs = _normalize_plate_text(reading_1line[0]), reading_1line[1]
    b_text, b_probs = _normalize_plate_text(reading_2line[0]), reading_2line[1]
    a, b = (a_text, a_probs), (b_text, b_probs)
    ok_a, ok_b = _looks_like_ru_plate(a_text), _looks_like_ru_plate(b_text)
    if ok_a and ok_b:
        # оба валидны — длиннее (часто полный регион 125) и 2-line на квадратных
        sa, sb = _score_plate_candidate(a_text), _score_plate_candidate(b_text)
        if prefer_2line and sb >= sa:
            return b
        return a if sa >= sb else b
    if ok_b and not ok_a:
        return b
    if ok_a and not ok_b:
        return a
    if prefer_2line and b_text:
        return b if _score_plate_candidate(b_text) >= _score_plate_candidate(a_text) else a
    return a if len(a_text) >= len(b_text) else b


def _apply_two_line_pass(
    zones_flat: list,
    readings: list[tuple[str, list[float]]],
    source_bboxes: list | None = None,
):
    """
    Повтор OCR с lines=2 для квадратных / двухстрочных номеров.

    Гейт: высокий кроп или высокий bbox в исходном кадре. Warp детектора часто
    сплющивает квадрат в полосу ~0.2 (Н909НР125 → однострочный мусор Н094РТ25),
    поэтому одного aspect кропа мало.
    """
    need_idx: list[int] = []
    need_zones: list = []
    for i, zone in enumerate(zones_flat):
        bbox = source_bboxes[i] if source_bboxes and i < len(source_bboxes) else None
        if zone is not None and _needs_two_line(zone, bbox):
            need_idx.append(i)
            need_zones.append(zone)

    if not need_zones:
        return readings

    out = list(readings)
    texts2 = _reread_zones_2line(need_zones)
    for j, i in enumerate(need_idx):
        before = _normalize_plate_text(out[i][0])
        second = texts2[j] if j < len(texts2) else ("", [])
        after_text, after_probs = _pick_better_plate(out[i], second, prefer_2line=True)
        out[i] = (after_text, after_probs)
        if second[0] and after_text != before:
            logger.info(
                "2-line OCR zone#%s: %r → %r (2line=%r)",
                i, before, after_text, _normalize_plate_text(second[0]),
            )
        bbox = source_bboxes[i] if source_bboxes and i < len(source_bboxes) else None
        out[i] = _reject_mashed_oneline_on_twoline_bbox(out[i], second, bbox)
    return out


def _reject_mashed_oneline_on_twoline_bbox(
    picked: tuple[str, list[float]],
    reading_2line: tuple[str, list[float]],
    bbox,
) -> tuple[str, list[float]]:
    """
    Сплющенный двухстрочный: однострочная голова уверенно читает 8 символов
    (Н094РТ25 вместо Н909НР125). Если bbox квадратный, а 2-line не подтвердил
    эту 8-символьную маску — чтение выбрасываем.
    """
    if not bbox_looks_two_line(bbox):
        return picked
    text = _normalize_plate_text(picked[0])
    if not (_RU_CIV.match(text) and len(text) == 8):
        return picked
    two = _normalize_plate_text(reading_2line[0] if reading_2line else "")
    if two and two == text:
        return picked
    if two and _looks_like_ru_plate(two) and len(two) >= 9:
        return reading_2line
    logger.info("2-line bbox: отброшен однострочный %r (2line=%r)", text, two)
    return "", []


def _ru_presets(with_military: bool) -> dict:
    """Пресеты OCR: гражданский ru, двухстрочный eu_2lines и (если есть) ru_military."""
    from nomeroff_net.pipelines.number_plate_text_reading import DEFAULT_PRESETS

    keep = ("ru", "eu_2lines_efficientnet_b2")
    presets = {name: dict(cfg) for name, cfg in DEFAULT_PRESETS.items() if name in keep}
    if with_military:
        presets["ru_military"] = {
            "for_regions": ["military"],
            "for_count_lines": [1],
            "model_path": "latest",
        }
    return presets


def _build_pipeline():
    """
    Собрать пайплайн. Военная голова ru_military подключается, если её вес есть
    локально (или доступна сеть) — иначе пишем предупреждение и работаем без неё:
    на офлайн-ПК падать из-за отсутствующего .ckpt нельзя.
    """
    want_military = (os.environ.get("NOMEROFF_RU_MILITARY") or "1").strip().lower() not in (
        "0", "false", "no", "off",
    )

    # Классификатор числа строк/региона — украинский options-detector; на РФ он
    # часто врёт, поэтому по умолчанию выключен. Геометрия зоны (_zone_is_square)
    # выбирает 2-line надёжнее. NOMEROFF_CLASSIFY=1 — попробовать нейросеть.
    classify = (os.environ.get("NOMEROFF_CLASSIFY") or "0").strip().lower() not in (
        "0", "false", "no", "off", "",
    )

    def _make(with_military: bool, off_cls: bool):
        return pipeline(
            "number_plate_detection_and_reading",
            image_loader="opencv",
            presets=_ru_presets(with_military),
            off_number_plate_classification=off_cls,
            default_label="ru",
            default_lines_count=1,
        )

    if want_military:
        try:
            instance = _make(True, off_cls=not classify)
            logger.info(
                "OCR-головы: ru + eu_2lines + ru_military (роутинг по формату), classify=%s",
                classify,
            )
            return instance
        except Exception as ex:
            logger.warning(
                "Военная голова ru_military недоступна (%s) — военные номера читает "
                "гражданская ru. Положите .ckpt в data/models/TextDetector/ru_military "
                "или дообучите через tutorials/ju/train/ocr/ru-military.ipynb.",
                ex,
            )
    try:
        instance = _make(False, off_cls=not classify)
    except Exception as ex:
        logger.warning("Классификатор не загрузился (%s) — работаем без него", ex)
        instance = _make(False, off_cls=True)
    logger.info("OCR-головы: ru + eu_2lines, classify=%s", classify)
    return instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, _runtime_device, _runtime_device_name
    _runtime_device, _runtime_device_name = _resolve_torch_device()
    if _runtime_device == "cuda":
        # На случай, если кто-то выставил CUDA_VISIBLE_DEVICES="" — уже обработано is_available
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        try:
            import torch
            # Размеры входов у нас стабильные (1920x1080 кадр, 200x50 кроп) —
            # автотюнер cudnn подбирает ядра один раз и дальше только выигрывает.
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as ex:
            logger.warning("Не удалось включить cudnn.benchmark: %s", ex)
        logger.info("Устройство: CUDA (%s)", _runtime_device_name or "gpu")
    else:
        # Явно не форсим CUDA, если её нет
        logger.info("Устройство: CPU")

    logger.info(
        "Загрузка модели nomeroff-net (RU-only, device=%s, two_line=%s)...",
        _runtime_device,
        _two_line_enabled(),
    )
    pipeline_instance = _build_pipeline()
    # Nomeroff внутри берёт cuda через torch.cuda.is_available(); логируем факт
    try:
        import torch
        from nomeroff_net.tools.mcm import get_device_torch, get_device_name
        actual = get_device_torch()
        _runtime_device = actual
        _runtime_device_name = get_device_name() if actual == "cuda" else ""
        if actual == "cuda":
            logger.info(
                "Модель загружена на CUDA: %s | torch=%s | cuda=%s",
                _runtime_device_name,
                torch.__version__,
                torch.version.cuda,
            )
        else:
            logger.info("Модель загружена на CPU | torch=%s", torch.__version__)
        logger.info(
            "YOLO-батч: %s картинок за forward (device=%s, NOMEROFF_YOLO_BATCH=%s)",
            _yolo_chunk_size(),
            _runtime_device,
            os.environ.get("NOMEROFF_YOLO_BATCH") or "auto",
        )
    except Exception as ex:
        logger.warning("Не удалось уточнить device после загрузки: %s", ex)
        logger.info("Модель загружена (default_label=ru)")
    yield
    logger.info("Очистка ресурсов...")


app = FastAPI(title="Nomeroff API", version="1.0", lifespan=lifespan)

# CORS: браузер отправляет OPTIONS перед POST — без этого 405 Method Not Allowed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Модели ===
class ProcessFrameRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 JPG/PNG")
    variants: list[str] | None = Field(
        None,
        description=f"Варианты предобработки кадра. Доступны: {', '.join(FRAME_VARIANTS)}. "
                    "По умолчанию только full.",
    )
    min_ocr_confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Отбросить чтения с уверенностью CTC-головы ниже порога",
    )
    include_crop: bool = Field(True, description="Возвращать кроп номера в base64")


class BatchFrame(BaseModel):
    image_base64: str
    time_sec: float = 0.0


class ProcessFramesRequest(BaseModel):
    frames: list[BatchFrame] = Field(..., description="Кадры одного проезда, 8-16 за вызов")
    variants: list[str] | None = None
    min_ocr_confidence: float = 0.0
    include_crop: bool = True
    min_frame_hits: int = Field(
        2, ge=1,
        description="Номер попадает в consensus, только если встречен в N разных кадрах",
    )


class ProcessFramesMeta(BaseModel):
    """Метаданные к multipart /api/process_frames_raw (JPEG без base64)."""
    variants: list[str] | None = None
    times_sec: list[float] = Field(default_factory=list)
    min_ocr_confidence: float = 0.0
    include_crop: bool = True
    min_frame_hits: int = 1


class PlateResult(BaseModel):
    plate: str
    # score детектора YOLO: насколько это вообще номер (не насколько верно прочитан)
    confidence: float = 0.95
    det_confidence: float = 0.0
    # уверенность CTC-головы OCR: минимум по символам распознанного текста
    ocr_confidence: float = 0.0
    # вероятности по символам — веса для межкадрового голосования
    char_probs: list[float] = []
    bbox: list[int] = [0, 0, 0, 0]
    # площадь bbox в исходном кадре: по ней выбирается лучший кадр для фото
    bbox_area: int = 0
    variant: str = "full"
    plate_kind: str = ""
    plate_image_base64: str | None = None


class ProcessFrameResponse(BaseModel):
    success: bool
    plates: list[PlateResult]
    processing_time_ms: float
    message: str | None = None


class FrameResult(BaseModel):
    time_sec: float = 0.0
    plates: list[PlateResult] = []


class ConsensusPlate(BaseModel):
    plate: str
    ocr_confidence: float = 0.0
    frame_hits: int = 0
    times_sec: list[float] = []
    best_time_sec: float = 0.0
    best_bbox_area: int = 0
    plate_image_base64: str | None = None
    # 'vote' | 'vlm_candidate' | 'vlm_new' — кто вынес окончательное решение
    decided_by: str = "vote"


class ProcessFramesResponse(BaseModel):
    success: bool
    frames: list[FrameResult]
    consensus: list[ConsensusPlate]
    processing_time_ms: float
    message: str | None = None


class ArbitratePlateRequest(BaseModel):
    plate_image_base64: str = Field(..., description="Кроп номера (JPEG/PNG base64)")
    candidates: list[str] = Field(
        default_factory=list,
        description="Варианты, между которыми не сошлось голосование",
    )


class ArbitratePlateResponse(BaseModel):
    enabled: bool = False
    plate: str = ""
    # 'vlm_candidate' — ответ совпал с кандидатом, 'vlm_new' — другой валидный номер
    decided_by: str = ""
    message: str | None = None


class OcrOverlayRequest(BaseModel):
    gps_base64: str = Field("", description="PNG/JPEG правой полосы (GPS); пусто — не читать")
    date_base64: str = Field("", description="PNG/JPEG левой полосы (дата/время); пусто — не читать")


class OcrOverlayResponse(BaseModel):
    gps_text: str = ""
    date_text: str = ""


_overlay_rapidocr = None
_overlay_easyocr_reader = None


def _get_rapidocr():
    global _overlay_rapidocr
    if _overlay_rapidocr is None:
        from rapidocr_onnxruntime import RapidOCR
        use_cuda = _runtime_device == "cuda"
        logger.info("Загрузка RapidOCR для ocr_overlay (cuda=%s)...", use_cuda)
        if use_cuda:
            # Без onnxruntime-gpu RapidOCR молча уезжает на CPU и тянет за собой
            # весь OSD и двухстрочный проход.
            try:
                _overlay_rapidocr = RapidOCR(
                    det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True
                )
                return _overlay_rapidocr
            except Exception as ex:
                logger.warning("RapidOCR на CUDA не стартовал (%s) — CPU. "
                               "Проверьте, что установлен onnxruntime-gpu.", ex)
        _overlay_rapidocr = RapidOCR()
    return _overlay_rapidocr


def _get_easyocr_reader():
    global _overlay_easyocr_reader
    if _overlay_easyocr_reader is None:
        import easyocr
        use_gpu = _runtime_device == "cuda"
        logger.info(
            "Загрузка EasyOCR для ocr_overlay (gpu=%s, первый вызов может занять минуту)...",
            use_gpu,
        )
        try:
            _overlay_easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as ex:
            if use_gpu:
                logger.warning("EasyOCR на CUDA не стартовал (%s) — fallback CPU", ex)
                _overlay_easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            else:
                raise
    return _overlay_easyocr_reader


def _ocr_strip_rapidocr(img_bgr) -> str:
    ocr = _get_rapidocr()
    result, _ = ocr(img_bgr)
    if not result:
        return ""
    parts = []
    for line in result:
        if line and len(line) > 1 and line[1]:
            parts.append(str(line[1]))
    return " ".join(parts)


def _ocr_strip_easyocr(img_bgr) -> str:
    reader = _get_easyocr_reader()
    parts = reader.readtext(img_bgr, detail=0, paragraph=True)
    if isinstance(parts, list):
        return " ".join(str(p) for p in parts)
    return str(parts) if parts else ""


def _ocr_strip_pytesseract(img_bgr) -> str:
    import pytesseract
    from PIL import Image

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return pytesseract.image_to_string(
        pil,
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,;:/-NSEWnsewkmh ",
    ).strip()


def _ocr_strip_text(img_bgr) -> str:
    if img_bgr is None:
        return ""
    # RapidOCR лучше читает OSD регистратора (131°56.944'E, 43°12.989'N)
    try:
        text = _ocr_strip_rapidocr(img_bgr)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as ex:
        logger.warning("RapidOCR failed: %s", ex)
    try:
        return _ocr_strip_easyocr(img_bgr)
    except ImportError:
        return _ocr_strip_pytesseract(img_bgr)


# === Эндпоинты ===
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        "<html><head><meta charset='utf-8'><title>Nomeroff API</title></head>"
        "<body style='color:#ccc;'>"
        "<h2>Nomeroff API</h2>"
        "<ul>"
        "<li><a href='/test/image' style='color:lightblue;'>Тест: загрузка изображения</a></li>"
        "<li><a href='/test/video' style='color:lightblue;'>Тест: загрузка видео (первый кадр)</a></li>"
        "<li><a href='/docs' style='color:lightblue;'>Swagger UI</a></li>"
        "</ul>"
        "</body></html>"
    )


@app.get("/health")
async def health_check():
    cuda_available = False
    vram: dict = {}
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram = {
                "vram_used_mb": round((total_b - free_b) / (1024 * 1024), 1),
                "vram_free_mb": round(free_b / (1024 * 1024), 1),
                "vram_total_mb": round(total_b / (1024 * 1024), 1),
                "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1),
                "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1),
            }
    except Exception:
        pass
    return {
        "status": "ok",
        "model_loaded": pipeline_instance is not None,
        "gpu_available": cuda_available,
        "device": _runtime_device,
        "device_name": _runtime_device_name or None,
        "device_policy": os.environ.get("NOMEROFF_DEVICE", "auto"),
        "two_line": _two_line_enabled(),
        # Сети — на device; нарезка кадров / JPEG / ROI — всегда CPU.
        "inference": _runtime_device,
        "preprocess": "cpu",
        "yolo_batch": _yolo_chunk_size(),
        **vram,
    }


@app.post("/api/ocr_overlay", response_model=OcrOverlayResponse)
async def ocr_overlay(request: OcrOverlayRequest):
    """
    OCR полос оверлея (GPS справа, дата слева). RapidOCR, иначе EasyOCR/pytesseract.

    Пустая полоса пропускается: вызывающая сторона просит дату лишь несколько раз
    за ролик (время считается от начала записи), а каждая полоса на CPU стоит
    больше секунды.
    """
    try:
        def _decode(b64: str):
            if not b64:
                return None
            data = base64.b64decode(b64)
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gps_img = _decode(request.gps_base64)
        date_img = _decode(request.date_base64)
        return OcrOverlayResponse(
            gps_text=_ocr_strip_text(gps_img) if gps_img is not None else "",
            date_text=_ocr_strip_text(date_img) if date_img is not None else "",
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Установите rapidocr-onnxruntime (или easyocr / pytesseract)",
        )
    except Exception as ex:
        logger.exception("ocr_overlay failed")
        raise HTTPException(status_code=500, detail=str(ex))


def _decode_jpeg_bytes(data: bytes):
    """JPEG/PNG bytes -> BGR. Общий путь для JSON (после base64) и multipart."""
    if not data:
        raise ValueError("Пустой кадр")
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Не удалось декодировать изображение")
    return frame


def _decode_frame(image_base64: str):
    """base64 -> BGR ndarray. Без промежуточного файла на диске."""
    return _decode_jpeg_bytes(base64.b64decode(image_base64))


def _yolo_chunk_size() -> int:
    free_mb = None
    if _runtime_device == "cuda":
        try:
            import torch
            free_b, _total = torch.cuda.mem_get_info(0)
            free_mb = free_b / (1024 * 1024)
        except Exception:
            free_mb = None
    n = yolo_chunk_size(
        _runtime_device,
        free_vram_mb=free_mb,
        env_batch=os.environ.get("NOMEROFF_YOLO_BATCH"),
    )
    if _yolo_chunk_cap is not None:
        n = min(n, _yolo_chunk_cap)
    return max(1, n)


def _is_cuda_oom(ex: BaseException) -> bool:
    msg = str(ex).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _clear_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _localize_once(images_rgb: list) -> list[list]:
    """Один forward детектора. При OOM режем батч пополам — слабая карта не падает."""
    if not images_rgb:
        return []
    detector = pipeline_instance.number_plate_localization.detector
    try:
        return detector.predict(images_rgb)
    except Exception as ex:
        if not (_runtime_device == "cuda" and _is_cuda_oom(ex) and len(images_rgb) > 1):
            raise
        global _yolo_chunk_cap
        _clear_cuda_cache()
        mid = max(1, len(images_rgb) // 2)
        _yolo_chunk_cap = mid if _yolo_chunk_cap is None else min(_yolo_chunk_cap, mid)
        logger.warning(
            "CUDA OOM на YOLO-батче %s — режем до %s+%s",
            len(images_rgb), mid, len(images_rgb) - mid,
        )
        return _localize_once(images_rgb[:mid]) + _localize_once(images_rgb[mid:])


def _localize(images_rgb: list) -> list[list]:
    """Детекция номеров. На CPU чанк = один кадр; на CUDA — по свободной VRAM."""
    if not images_rgb:
        return []
    chunk = _yolo_chunk_size()
    if len(images_rgb) <= chunk:
        return _localize_once(images_rgb)
    out: list[list] = []
    for i in range(0, len(images_rgb), chunk):
        out.extend(_localize_once(images_rgb[i:i + chunk]))
    return out


def _zones_from_bboxes(images_rgb: list, images_bboxs: list):
    """Кропы зон номеров по отфильтрованным bbox (тот же порядок, что у боксов)."""
    zones, image_ids, images_points = crop_number_plate_roi_zones_from_images(images_rgb, images_bboxs)
    images_points = [normalize_rect_new(points) for points in images_points]
    zones, image_ids = crop_number_plate_zones_from_images(zones, image_ids, images_points)
    return zones, image_ids


def _military_ocr_available(detector) -> bool:
    """Есть ли отдельная голова под военный формат (пресет ru_military)."""
    try:
        return (1, "military") in detector.detectors_map
    except Exception:
        return False


def _apply_military_pass(detector, zones: list, readings: list[tuple[str, list[float]]]):
    """
    Перечитать военной моделью зоны, чьё чтение похоже на военный номер.

    Гражданская голова `ru` обучена на маске «буква + 3 цифры», поэтому первую
    цифру военного 9036СС45 она регулярно принимает за букву (У036СС45).
    Отсюда и весь набор C#-костылей вокруг военного формата.
    """
    candidates = [
        i for i, (text, _) in enumerate(readings)
        if plate_ru.is_military(plate_ru.normalize_plate(text))
        or (plate_ru.decode_constrained(text, require_region=False) or ("", ""))[1] == "military"
    ]
    if not candidates:
        return readings

    out = list(readings)
    military = _ocr_zones_lines_once(detector, [zones[i] for i in candidates], 1, label="military")
    for j, i in enumerate(candidates):
        if j >= len(military):
            break
        before, after = out[i], military[j]
        if _score_plate_candidate(after[0]) > _score_plate_candidate(before[0]):
            out[i] = after
            logger.info("military OCR zone#%s: %r → %r", i, before[0], after[0])
    return out


def _promote_military_on_negative(
    readings: list[tuple[str, list[float]]],
    inverted: list[bool],
) -> list[tuple[str, list[float]]]:
    """
    Зоны с негатива: гражданское чтение, похожее на военный номер, поднимаем
    до военного формата. Дальше военная голова перечитает такую зону и либо
    подтвердит, либо даст свой вариант.
    """
    out = list(readings)
    for i, (text, probs) in enumerate(out):
        if i >= len(inverted) or not inverted[i]:
            continue
        fixed = plate_ru.try_military_from_civilian_lookalike(text)
        if fixed:
            logger.info("military lookalike (негатив) zone#%s: %r → %r", i, text, fixed)
            out[i] = (fixed, probs)
    return out


def _ocr_zones(
    zones: list,
    inverted: list[bool] | None = None,
    source_bboxes: list | None = None,
) -> list[tuple[str, list[float]]]:
    """OCR зон: однострочные — ru, квадратные/двухстрочные — сразу 2-line."""
    if not zones:
        return []
    detector = pipeline_instance.number_plate_text_reading.detector
    readings: list[tuple[str, list[float]]] = [("", [])] * len(zones)

    def _bbox_at(i: int):
        return source_bboxes[i] if source_bboxes and i < len(source_bboxes) else None

    idx_1line = [
        i for i, z in enumerate(zones)
        if not _needs_two_line(z, _bbox_at(i))
    ]
    idx_2line = [i for i, z in enumerate(zones) if i not in idx_1line]

    if idx_1line:
        r1 = _ocr_zones_lines_once(detector, [zones[i] for i in idx_1line], 1)
        for j, i in enumerate(idx_1line):
            if j < len(r1):
                readings[i] = r1[j]
    if idx_2line:
        r2 = _ocr_zones_lines_once(detector, [zones[i] for i in idx_2line], 2)
        for j, i in enumerate(idx_2line):
            if j < len(r2):
                readings[i] = r2[j]

    if inverted:
        readings = _promote_military_on_negative(readings, inverted)
    if _military_ocr_available(detector):
        readings = _apply_military_pass(detector, zones, readings)
    if _two_line_enabled():
        # eu_2lines + RapidOCR поверх геометрического 2-line: квадратный Т314ХС125
        # часто читается только этой веткой, а не однострочной гражданской головой.
        readings = _apply_two_line_pass(zones, readings, source_bboxes)
    return readings


def _collect_kept(variants, images_rgb, detections):
    """Гейт bbox по каждому варианту. owners — индекс в variants/images_rgb."""
    kept_boxes: list[list] = [[] for _ in variants]
    owners: list[int] = []
    source_bboxes: list[list[int]] = []
    det_scores: list[float] = []
    for vi, (variant, boxes) in enumerate(zip(variants, detections)):
        shape = images_rgb[vi].shape
        for box in boxes or []:
            reason = bbox_gate(box, shape)
            if reason is not None:
                logger.debug("bbox gate [%s]: %s", variant.name, reason)
                continue
            kept_boxes[vi].append(box)
            owners.append(vi)
            source_bboxes.append(variant.bbox_to_source(box))
            det_scores.append(float(box[4]) if len(box) > 4 else 0.0)
    return kept_boxes, owners, source_bboxes, det_scores


def _emit_plates(
    variants,
    owners: list[int],
    source_bboxes: list[list[int]],
    det_scores: list[float],
    zones,
    readings: list[tuple[str, list[float]]],
    *,
    min_ocr_confidence: float,
    include_crop: bool,
    frame_bgr=None,
    frame_per_owner: list | None = None,
) -> list[PlateResult]:
    plates: list[PlateResult] = []
    for i, (text, char_probs) in enumerate(readings):
        decoded = plate_ru.decode_constrained(text)
        if decoded is None:
            continue
        plate, kind = decoded

        ocr_conf = min(char_probs) if char_probs else 0.0
        if ocr_conf < min_ocr_confidence:
            logger.debug("ocr gate: %r ocr_conf=%.2f < %.2f", plate, ocr_conf, min_ocr_confidence)
            continue

        variant = variants[owners[i]] if i < len(owners) else variants[0]
        bbox = source_bboxes[i] if i < len(source_bboxes) else [0, 0, 0, 0]
        area = max(0, (bbox[2] - bbox[0])) * max(0, (bbox[3] - bbox[1]))
        src_frame = frame_per_owner[i] if frame_per_owner is not None else frame_bgr

        plate_b64 = None
        if include_crop:
            # Кроп всегда из ОРИГИНАЛЬНОГО кадра по обратно отображённому bbox:
            # иначе в БД уезжает негатив или растянутый ROI вместо номера.
            if src_frame is not None:
                crop = crop_bbox(src_frame, bbox)
                if crop is not None and crop.size:
                    plate_b64 = _bgr_to_jpeg_b64(crop)
            if plate_b64 is None and i < len(zones):
                plate_b64 = _zone_to_jpeg_b64(zones[i])

        det_conf = det_scores[i] if i < len(det_scores) else 0.0
        plates.append(
            PlateResult(
                plate=plate,
                confidence=det_conf,
                det_confidence=det_conf,
                ocr_confidence=round(float(ocr_conf), 4),
                char_probs=[round(float(p), 4) for p in char_probs],
                bbox=bbox,
                bbox_area=int(area),
                variant=variant.name,
                plate_kind=kind,
                plate_image_base64=plate_b64,
            )
        )
    return plates


def _recognize_one_frame(
    frame_bgr,
    variant_names: list[str] | tuple[str, ...],
    *,
    min_ocr_confidence: float = 0.0,
    include_crop: bool = True,
) -> list[PlateResult]:
    """Один кадр: 5 вариантов в один YOLO-forward. Путь CPU и камеры."""
    variants = build_variants(frame_bgr, variant_names)
    images_rgb = [cv2.cvtColor(v.image, cv2.COLOR_BGR2RGB) for v in variants]

    with _inference_ctx():
        detections = _localize(images_rgb)
        kept_boxes, owners, source_bboxes, det_scores = _collect_kept(
            variants, images_rgb, detections
        )
        if not owners:
            return []
        zones, _ = _zones_from_bboxes(images_rgb, kept_boxes)
        readings = _ocr_zones(
            zones,
            [variants[o].inverted for o in owners],
            source_bboxes,
        )

    return _emit_plates(
        variants, owners, source_bboxes, det_scores, zones, readings,
        min_ocr_confidence=min_ocr_confidence,
        include_crop=include_crop,
        frame_bgr=frame_bgr,
    )


def _recognize_flat_batch(
    frames_bgr: list,
    variant_names: list[str] | tuple[str, ...],
    *,
    min_ocr_confidence: float,
    include_crop: bool,
) -> list[list[PlateResult]]:
    """Все кадры HTTP-батча — один (чанканутый) YOLO + один OCR. Только CUDA."""
    all_variants = []
    all_rgb = []
    variant_frame: list = []
    frame_of_variant: list[int] = []

    for fi, frame in enumerate(frames_bgr):
        variants = build_variants(frame, variant_names)
        for v in variants:
            all_variants.append(v)
            all_rgb.append(cv2.cvtColor(v.image, cv2.COLOR_BGR2RGB))
            variant_frame.append(frame)
            frame_of_variant.append(fi)

    empty: list[list[PlateResult]] = [[] for _ in frames_bgr]
    if not all_rgb:
        return empty

    with _inference_ctx():
        detections = _localize(all_rgb)
        kept_boxes, owners, source_bboxes, det_scores = _collect_kept(
            all_variants, all_rgb, detections
        )
        if not owners:
            return empty
        zones, _ = _zones_from_bboxes(all_rgb, kept_boxes)
        readings = _ocr_zones(
            zones,
            [all_variants[o].inverted for o in owners],
            source_bboxes,
        )

    return _split_plates_by_frame(
        frames_bgr, all_variants, owners, source_bboxes, det_scores,
        zones, readings, frame_of_variant, variant_frame,
        min_ocr_confidence=min_ocr_confidence,
        include_crop=include_crop,
    )


def _split_plates_by_frame(
    frames_bgr,
    all_variants,
    owners,
    source_bboxes,
    det_scores,
    zones,
    readings,
    frame_of_variant: list[int],
    variant_frame: list,
    *,
    min_ocr_confidence: float,
    include_crop: bool,
) -> list[list[PlateResult]]:
    """Разложить плоские чтения по исходным кадрам."""
    by_frame: list[list[PlateResult]] = [[] for _ in frames_bgr]
    for i, owner in enumerate(owners):
        one = _emit_plates(
            all_variants,
            [owner],
            [source_bboxes[i]],
            [det_scores[i]],
            [zones[i]] if i < len(zones) else [None],
            [readings[i]] if i < len(readings) else [("", [])],
            min_ocr_confidence=min_ocr_confidence,
            include_crop=include_crop,
            frame_bgr=variant_frame[owner],
        )
        if one:
            by_frame[frame_of_variant[owner]].extend(one)
    return by_frame


def _recognize_many_frames(
    frames_bgr: list,
    variant_names: list[str] | tuple[str, ...],
    *,
    min_ocr_confidence: float = 0.0,
    include_crop: bool = True,
) -> list[list[PlateResult]]:
    """
    CUDA: все кадры запроса в один детектор (чанки по VRAM).
    CPU / один кадр: как раньше, по кадру — i3 не раздувает RAM.
    CUDA OOM: откат на по-кадровый путь, качество то же.
    """
    if not frames_bgr:
        return []
    if _runtime_device != "cuda" or len(frames_bgr) == 1:
        return [
            _recognize_one_frame(
                frame, variant_names,
                min_ocr_confidence=min_ocr_confidence,
                include_crop=include_crop,
            )
            for frame in frames_bgr
        ]
    try:
        return _recognize_flat_batch(
            frames_bgr, variant_names,
            min_ocr_confidence=min_ocr_confidence,
            include_crop=include_crop,
        )
    except Exception as ex:
        if not _is_cuda_oom(ex):
            raise
        logger.warning(
            "CUDA OOM на батче из %s кадров — считаем по кадру", len(frames_bgr)
        )
        _clear_cuda_cache()
        return [
            _recognize_one_frame(
                frame, variant_names,
                min_ocr_confidence=min_ocr_confidence,
                include_crop=include_crop,
            )
            for frame in frames_bgr
        ]


def _recognize_frame(
    frame_bgr,
    variant_names: list[str] | tuple[str, ...],
    *,
    min_ocr_confidence: float = 0.0,
    include_crop: bool = True,
) -> list[PlateResult]:
    """
    Полный проход по одному кадру: варианты предобработки -> детекция -> гейт bbox
    -> OCR с уверенностью CTC -> декодирование по маске формата РФ.
    """
    return _recognize_one_frame(
        frame_bgr, variant_names,
        min_ocr_confidence=min_ocr_confidence,
        include_crop=include_crop,
    )


@app.post("/api/process_frame", response_model=ProcessFrameResponse)
async def process_frame(request: ProcessFrameRequest):
    """Только распознавание номеров по кадру. Без watchlist и dedup."""
    start = time.time()
    try:
        frame = _decode_frame(request.image_base64)
        plates = _recognize_frame(
            frame,
            request.variants or ("full",),
            min_ocr_confidence=request.min_ocr_confidence,
            include_crop=request.include_crop,
        )
        elapsed_ms = (time.time() - start) * 1000
        return ProcessFrameResponse(
            success=True,
            plates=plates,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Найдено {len(plates)} номеров" if plates else "Номера не обнаружены",
        )
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    except Exception as e:
        logger.error("Ошибка обработки: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/arbitrate_plate", response_model=ArbitratePlateResponse)
async def arbitrate_plate(request: ArbitratePlateRequest):
    """
    Второй каскад для спорного трека: локальный VLM выбирает из кандидатов.

    Вызывать только когда голосование не сошлось. Выключено по умолчанию —
    без NOMEROFF_VLM=1 и локальной модели отвечает enabled=false, и вызывающая
    сторона просто остаётся с результатом голосования.
    """
    if not vlm_arbiter.is_enabled():
        return ArbitratePlateResponse(enabled=False, message="VLM-арбитр выключен")

    try:
        crop = _decode_frame(request.plate_image_base64)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    verdict = vlm_arbiter.arbitrate(crop, request.candidates)
    if verdict is None:
        return ArbitratePlateResponse(
            enabled=True, message="Арбитр не дал валидного ответа")

    plate, source = verdict
    return ArbitratePlateResponse(enabled=True, plate=plate, decided_by=source)


def _process_decoded_frames(
    frames_bgr: list,
    times_sec: list[float],
    variant_names: list[str] | tuple[str, ...],
    min_ocr_confidence: float,
    include_crop: bool,
    min_frame_hits: int,
) -> ProcessFramesResponse:
    start = time.time()
    per_frame = _recognize_many_frames(
        frames_bgr,
        variant_names,
        min_ocr_confidence=min_ocr_confidence,
        include_crop=include_crop,
    )
    frames_out: list[FrameResult] = []
    votes: dict[str, list[tuple[float, PlateResult]]] = {}
    for time_sec, plates in zip(times_sec, per_frame):
        frames_out.append(FrameResult(time_sec=time_sec, plates=plates))
        best_per_plate: dict[str, PlateResult] = {}
        for p in plates:
            prev = best_per_plate.get(p.plate)
            if prev is None or p.ocr_confidence > prev.ocr_confidence:
                best_per_plate[p.plate] = p
        for plate, p in best_per_plate.items():
            votes.setdefault(plate, []).append((time_sec, p))
    consensus = _build_consensus(votes, min_frame_hits)
    elapsed_ms = (time.time() - start) * 1000
    return ProcessFramesResponse(
        success=True,
        frames=frames_out,
        consensus=consensus,
        processing_time_ms=round(elapsed_ms, 2),
        message=f"Кадров {len(frames_out)}, консенсус по {len(consensus)} номерам",
    )


@app.post("/api/process_frames", response_model=ProcessFramesResponse)
async def process_frames(request: ProcessFramesRequest):
    """
    Батч кадров одного проезда за один вызов + межкадровый консенсус.

    На CUDA детектор берёт все кадры запроса сразу (чанки по VRAM).
    На CPU — по кадру, как раньше, чтобы i3 не упирался в RAM.
    """
    if not request.frames:
        raise HTTPException(status_code=400, detail="frames пуст")
    try:
        frames_bgr = [_decode_frame(item.image_base64) for item in request.frames]
        times = [item.time_sec for item in request.frames]
        return _process_decoded_frames(
            frames_bgr,
            times,
            request.variants or ("full", "crop", "roi"),
            request.min_ocr_confidence,
            request.include_crop,
            request.min_frame_hits,
        )
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    except Exception as ex:
        logger.error("Ошибка батча: %s", ex, exc_info=True)
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/api/process_frames_raw", response_model=ProcessFramesResponse)
async def process_frames_raw(
    meta: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Тот же разбор, что /api/process_frames, но JPEG файлами — без base64.
    Старый клиент без этого маршрута остаётся на JSON.
    """
    if not files:
        raise HTTPException(status_code=400, detail="files пуст")
    try:
        spec = ProcessFramesMeta.model_validate_json(meta)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"meta: {ex}")
    try:
        frames_bgr = []
        times: list[float] = []
        for i, upload in enumerate(files):
            data = await upload.read()
            frames_bgr.append(_decode_jpeg_bytes(data))
            times.append(spec.times_sec[i] if i < len(spec.times_sec) else 0.0)
        return _process_decoded_frames(
            frames_bgr,
            times,
            spec.variants or ("full", "crop", "roi"),
            spec.min_ocr_confidence,
            spec.include_crop,
            spec.min_frame_hits,
        )
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))
    except Exception as ex:
        logger.error("Ошибка raw-батча: %s", ex, exc_info=True)
        raise HTTPException(status_code=500, detail=str(ex))


def _crop_for_arbiter(plate: "PlateResult"):
    """Кроп номера для VLM: обратно из base64, который уже посчитан для БД."""
    if not plate.plate_image_base64:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(plate.plate_image_base64), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _build_consensus(
    votes: dict[str, list[tuple[float, "PlateResult"]]],
    min_frame_hits: int,
) -> list[ConsensusPlate]:
    """
    Слить чтения в номера по треку текста и отсечь то, что видно меньше min_frame_hits раз.

    Сначала голосование по позициям символов внутри группы близких чтений
    (В713НВ129 + 3x В713ВВ125 -> В713ВВ125), затем порог по числу разных кадров.
    """
    # группируем чтения по «стволу» — первые 6 символов задают машину
    groups: dict[str, list[tuple[float, PlateResult]]] = {}
    for plate, items in votes.items():
        groups.setdefault(plate[:6], []).extend(items)

    out: list[ConsensusPlate] = []
    for stem, items in groups.items():
        times = sorted({round(t, 2) for t, _ in items})
        if len(times) < min_frame_hits:
            logger.info(
                "consensus: %s отброшен — кадров %s < %s", stem, len(times), min_frame_hits
            )
            continue

        voted = plate_ru.vote_plate([(p.plate, p.char_probs) for _, p in items])
        if voted is None:
            continue
        plate, conf, _ = voted
        decoded = plate_ru.decode_constrained(plate)
        if decoded is None:
            logger.info("consensus: %s отброшен — голосование дало невалидный %r", stem, plate)
            continue
        plate = decoded[0]

        # кадр для фото — с максимальной площадью bbox: машина ближе всего к камере
        best_time, best = max(items, key=lambda pair: pair[1].bbox_area)

        decided_by = "vote"
        if conf < vlm_arbiter.min_confidence() and vlm_arbiter.is_enabled():
            candidates = sorted({p.plate for _, p in items})
            verdict = vlm_arbiter.arbitrate(_crop_for_arbiter(best), candidates)
            if verdict is not None:
                arbitrated, decided_by = verdict
                if arbitrated != plate:
                    logger.info(
                        "VLM-арбитр: %s -> %s (голосование дало %.2f)", plate, arbitrated, conf
                    )
                plate = arbitrated

        out.append(
            ConsensusPlate(
                plate=plate,
                ocr_confidence=round(float(conf), 4),
                frame_hits=len(times),
                times_sec=times,
                best_time_sec=best_time,
                best_bbox_area=best.bbox_area,
                plate_image_base64=best.plate_image_base64,
                decided_by=decided_by,
            )
        )
    out.sort(key=lambda c: c.best_time_sec)
    return out


@app.get("/test/image", include_in_schema=False)
async def test_image_form():
    html = """
    <html><head><meta charset='utf-8'><title>Test image</title></head>
    <body style='color:#ccc;'>
    <h2>Тест: загрузка изображения</h2>
    <form action="/test/image" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" />
      <button type="submit">Отправить</button>
    </form>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/test/image", include_in_schema=False)
async def test_image_upload(file: UploadFile):
    data = await file.read()
    b64 = base64.b64encode(data).decode("ascii")
    resp = await process_frame(ProcessFrameRequest(image_base64=b64))
    items = "".join(f"<li>{p.plate} (conf={p.confidence:.2f})</li>" for p in resp.plates)
    html = (
        "<html><head><meta charset='utf-8'><title>Result</title></head><body>"
        "<h2>Результат</h2>"
        f"<p>{resp.message}</p>"
        f"<ul>{items or '<li>Ничего не найдено</li>'}</ul>"
        "</body></html>"
    )
    return HTMLResponse(html)


@app.get("/test/video", include_in_schema=False)
async def test_video_form():
    html = """
    <html><head><meta charset='utf-8'><title>Test video</title></head>
    <body style='color:#ccc;'>
    <h2>Тест: загрузка видео (будет обработан первый кадр)</h2>
    <form action="/test/video" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="video/*" />
      <button type="submit">Отправить</button>
    </form>
    </body></html>
    """
    return HTMLResponse(html)


@app.post("/test/video", include_in_schema=False)
async def test_video_upload(file: UploadFile):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            data = await file.read()
            tmp.write(data)

        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать первый кадр")

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="Не удалось перекодировать кадр")

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        resp = await process_frame(ProcessFrameRequest(image_base64=b64))
        items = "".join(f"<li>{p.plate} (conf={p.confidence:.2f})</li>" for p in resp.plates)
        html = (
            "<html><head><meta charset='utf-8'><title>Result</title></head><body>"
            "<h2>Результат по первому кадру видео</h2>"
            f"<p>{resp.message}</p>"
            f"<ul>{items or '<li>Ничего не найдено</li>'}</ul>"
            "</body></html>"
        )
        return HTMLResponse(html)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Не удалось удалить временный видеофайл %s: %s", tmp_path, e)

if __name__ == "__main__":
    import uvicorn
    host = (os.environ.get("NOMEROFF_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int((os.environ.get("NOMEROFF_PORT") or "8000").strip())
    except ValueError:
        port = 8000
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")
