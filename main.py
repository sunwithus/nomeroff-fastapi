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
from contextlib import asynccontextmanager

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
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

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

# Квадратные РФ-номера: высота/ширина кропа заметно больше, чем у длинных 520×112
_SQUARE_ASPECT_MIN = 0.42
_RU_LETTERS = "АВЕКМНОРСТУХ"
_RU_CIV = re.compile(rf"^[{_RU_LETTERS}]\d{{3}}[{_RU_LETTERS}]{{2}}\d{{2,3}}$")
_RU_MIL = re.compile(rf"^\d{{4}}[{_RU_LETTERS}]{{2}}\d{{2,3}}$")


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
    if not plate:
        return ""
    # двухстрочный OCR иногда склеивает с пробелом/переносом
    return latin_to_cyrillic(re.sub(r"[\s\-]+", "", plate.strip()))


def _looks_like_ru_plate(plate: str) -> bool:
    p = _normalize_plate_text(plate)
    return bool(p and (_RU_CIV.match(p) or _RU_MIL.match(p)))


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


def _flatten_zones_texts(images_zones, raw_texts):
    """Выравнивает зоны и тексты в плоские списки одинаковой длины."""
    zones_flat: list = []
    texts_flat: list[str] = []

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, np.ndarray) and x.ndim >= 2 and not isinstance(x, list):
            # одиночная зона-картинка
            return [x]
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # Батч из одного изображения: [[z1,z2], ...] / [["T1","T2"], ...]
    # или уже плоский список.
    zones_items = _as_list(images_zones)
    texts_items = _as_list(raw_texts)

    # Если первый элемент — список зон/строк (per-image), разворачиваем
    if zones_items and isinstance(zones_items[0], (list, tuple)):
        for img_zones in zones_items:
            zones_flat.extend([z for z in img_zones if z is not None])
    else:
        zones_flat = [z for z in zones_items if z is not None]

    if texts_items and isinstance(texts_items[0], (list, tuple)):
        for img_texts in texts_items:
            for t in img_texts:
                texts_flat.append(t if isinstance(t, str) else (str(t) if t is not None else ""))
    else:
        for t in texts_items:
            texts_flat.append(t if isinstance(t, str) else (str(t) if t is not None else ""))

    # выровнять длины
    n = max(len(zones_flat), len(texts_flat))
    while len(zones_flat) < n:
        zones_flat.append(None)
    while len(texts_flat) < n:
        texts_flat.append("")
    return zones_flat, texts_flat


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


def _ocr_zones_lines_once(detector, zones: list, lines_count: int) -> list[str]:
    """Проход OCR с заданным числом строк; postprocess склеивает multiline."""
    n = len(zones)
    if n == 0:
        return []
    labels = ["ru"] * n
    lines = [int(lines_count)] * n
    model_inputs = detector.preprocess(
        zones, [None] * n, labels=labels, lines=lines
    )
    model_outputs = detector.forward(model_inputs)
    texts = detector.postprocess(model_outputs)
    if not isinstance(texts, list):
        return [""] * n
    out = []
    for t in texts:
        out.append(t if isinstance(t, str) else (str(t) if t is not None else ""))
    while len(out) < n:
        out.append("")
    return out[:n]


def _ocr_zone_halves_1line(detector, zone) -> str:
    """
    Квадратный номер: верх/низ по отдельности однострочной RU-моделью, затем склейка.
    Часто точнее, чем eu_2lines на регионе (…125).
    """
    try:
        from nomeroff_net.pipes.number_plate_keypoints_detectors.bbox_np_points_tools import (
            split_numberplate,
        )
    except Exception:
        return ""
    try:
        parts = split_numberplate(zone, 2)
    except Exception:
        return ""
    if len(parts) < 2:
        return ""
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
        texts = _ocr_zones_lines_once(detector, stretched, 1)
    except Exception:
        return ""
    if len(texts) < 2:
        return ""
    return _normalize_plate_text(texts[0]) + _normalize_plate_text(texts[1])


def _score_plate_candidate(text: str) -> tuple:
    """Выше — лучше. Предпочитаем валидный формат и более длинный регион (125 vs 12)."""
    p = _normalize_plate_text(text)
    if not p:
        return (0, 0, 0)
    ok = 1 if _looks_like_ru_plate(p) else 0
    # гражданский с 3 цифрами региона чуть предпочтительнее, чем с 2
    region3 = 1 if (_RU_CIV.match(p) and len(p) >= 9) else 0
    return (ok, region3, len(p))


def _reread_zones_2line(zones: list) -> list[str]:
    """OCR кропов двухстрочной моделью (eu_2lines для label=ru, lines=2)."""
    if not zones or pipeline_instance is None:
        return []
    try:
        detector = pipeline_instance.number_plate_text_reading.detector
    except Exception as ex:
        logger.warning("2-line OCR: нет text detector: %s", ex)
        return [""] * len(zones)

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

    best = [""] * len(zones)
    best_score = [(-1, -1, -1)] * len(zones)

    try:
        flat_texts = _ocr_zones_lines_once(detector, flat_zones, 2)
        for text, owner in zip(flat_texts, owners):
            sc = _score_plate_candidate(text)
            if sc > best_score[owner]:
                best_score[owner] = sc
                best[owner] = text
    except Exception as ex:
        logger.warning("2-line OCR failed: %s", ex)

    # RapidOCR на разжатом кропе — лучше читает регион 125 у квадратных
    # при равном score предпочитаем Rapid (eu_2lines часто даёт …12 / …122)
    for i, zone in enumerate(zones):
        rapid = _rapidocr_2line_plate(zone)
        sc = _score_plate_candidate(rapid)
        if sc > best_score[i] or (sc == best_score[i] and sc[0] == 1 and rapid):
            best_score[i] = sc
            best[i] = rapid

    # Если всё ещё нет валидного 3-значного региона — половины 1-line
    for i, zone in enumerate(zones):
        if best_score[i][0] == 1 and best_score[i][1] == 1:
            continue
        try:
            joined = _ocr_zone_halves_1line(detector, _zone_unsqueeze_2line(zone))
        except Exception:
            joined = ""
        sc = _score_plate_candidate(joined)
        if sc > best_score[i]:
            best_score[i] = sc
            best[i] = joined
            logger.info("2-line halves OCR zone#%s: %r score=%s", i, joined, sc)

    return best


def _pick_better_plate(text_1line: str, text_2line: str, *, prefer_2line: bool) -> str:
    a = _normalize_plate_text(text_1line)
    b = _normalize_plate_text(text_2line)
    ok_a, ok_b = _looks_like_ru_plate(a), _looks_like_ru_plate(b)
    if ok_a and ok_b:
        # оба валидны — длиннее (часто полный регион 125) и 2-line на квадратных
        sa, sb = _score_plate_candidate(a), _score_plate_candidate(b)
        if prefer_2line and sb >= sa:
            return b
        return a if sa >= sb else b
    if ok_b and not ok_a:
        return b
    if ok_a and not ok_b:
        return a
    if prefer_2line and b:
        return b if _score_plate_candidate(b) >= _score_plate_candidate(a) else a
    return a if len(a) >= len(b) else b


def _apply_two_line_pass(images_zones, raw_texts) -> list[str]:
    """
    Для квадратных кропов и «битых» однострочных чтений — повтор OCR lines=2.
    Модель eu_2lines уже в DEFAULT_PRESETS пайплайна.
    """
    zones_flat, texts_flat = _flatten_zones_texts(images_zones, raw_texts)
    if not zones_flat:
        return [_normalize_plate_text(t) for t in texts_flat if t]

    need_idx: list[int] = []
    need_zones: list = []
    for i, (zone, text) in enumerate(zip(zones_flat, texts_flat)):
        if zone is None:
            continue
        square = _zone_is_square(zone)
        ok = _looks_like_ru_plate(text)
        if square or not ok:
            need_idx.append(i)
            need_zones.append(zone)

    if not need_zones:
        return [_normalize_plate_text(t) for t in texts_flat]

    texts2 = _reread_zones_2line(need_zones)
    for j, i in enumerate(need_idx):
        before = _normalize_plate_text(texts_flat[i])
        t2 = texts2[j] if j < len(texts2) else ""
        square = zones_flat[i] is not None and _zone_is_square(zones_flat[i])
        after = _pick_better_plate(texts_flat[i], t2, prefer_2line=square)
        texts_flat[i] = after
        if t2 and after != before:
            logger.info(
                "2-line OCR zone#%s square=%s: %r → %r (2line=%r)",
                i, square, before, after, _normalize_plate_text(t2),
            )

    return [_normalize_plate_text(t) for t in texts_flat]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, _runtime_device, _runtime_device_name
    _runtime_device, _runtime_device_name = _resolve_torch_device()
    if _runtime_device == "cuda":
        # На случай, если кто-то выставил CUDA_VISIBLE_DEVICES="" — уже обработано is_available
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        logger.info("Устройство: CUDA (%s)", _runtime_device_name or "gpu")
    else:
        # Явно не форсим CUDA, если её нет
        logger.info("Устройство: CPU")

    logger.info(
        "Загрузка модели nomeroff-net (RU-only, device=%s, two_line=%s)...",
        _runtime_device,
        _two_line_enabled(),
    )
    pipeline_instance = pipeline(
        "number_plate_detection_and_reading",
        image_loader="opencv",
        off_number_plate_classification=True,
        default_label="ru",
        default_lines_count=1,
    )
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


class PlateResult(BaseModel):
    plate: str
    confidence: float = 0.95
    bbox: list[int] = [0, 0, 0, 0]


class ProcessFrameResponse(BaseModel):
    success: bool
    plates: list[PlateResult]
    processing_time_ms: float
    message: str | None = None


class OcrOverlayRequest(BaseModel):
    gps_base64: str = Field(..., description="PNG/JPEG правой полосы (GPS)")
    date_base64: str = Field(..., description="PNG/JPEG левой полосы (дата/время)")


class OcrOverlayResponse(BaseModel):
    gps_text: str = ""
    date_text: str = ""


_overlay_rapidocr = None
_overlay_easyocr_reader = None


def _get_rapidocr():
    global _overlay_rapidocr
    if _overlay_rapidocr is None:
        from rapidocr_onnxruntime import RapidOCR
        logger.info("Загрузка RapidOCR для ocr_overlay...")
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
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
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
    }


@app.post("/api/ocr_overlay", response_model=OcrOverlayResponse)
async def ocr_overlay(request: OcrOverlayRequest):
    """OCR полос оверлея (GPS справа, дата слева). EasyOCR или pytesseract."""
    try:
        def _decode(b64: str):
            data = base64.b64decode(b64)
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        gps_img = _decode(request.gps_base64)
        date_img = _decode(request.date_base64)
        return OcrOverlayResponse(
            gps_text=_ocr_strip_text(gps_img),
            date_text=_ocr_strip_text(date_img),
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Установите rapidocr-onnxruntime (или easyocr / pytesseract)",
        )
    except Exception as ex:
        logger.exception("ocr_overlay failed")
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/api/process_frame", response_model=ProcessFrameResponse)
async def process_frame(request: ProcessFrameRequest):
    """Только распознавание номеров по кадру. Без watchlist и dedup."""
    start = time.time()
    temp_path = None
    
    try:
        # Декодируем base64 → numpy array
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Не удалось декодировать изображение")

        # Создаём временный файл для совместимости с opencv-loader
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Передаём ПУТЬ к файлу (список путей для батч-обработки)
        results = pipeline_instance([temp_path])
        unpacked = unzip(results)

        # (images, images_bboxs, images_points, images_zones, region_ids, region_names, count_lines, confidences, texts)
        images_bboxs = unpacked[1] if len(unpacked) > 1 else []
        images_zones = unpacked[3] if len(unpacked) > 3 else []
        class_confidences = unpacked[7] if len(unpacked) > 7 else []
        raw_texts = unpacked[-1] if unpacked else []

        if _two_line_enabled():
            all_texts = _apply_two_line_pass(images_zones, raw_texts)
        else:
            all_texts = []
            for item in raw_texts:
                if isinstance(item, list):
                    all_texts.extend(
                        _normalize_plate_text(t) for t in item if isinstance(t, str)
                    )
                elif isinstance(item, str) and item.strip():
                    all_texts.append(_normalize_plate_text(item))

        # confidences: classification (может быть -1 при off_classification) или score детекции YOLO
        flat_class_conf = []
        for item in (class_confidences or []):
            if isinstance(item, (list, tuple)):
                flat_class_conf.extend(item)
            else:
                flat_class_conf.append(item)

        flat_det_scores = []
        for img_boxes in (images_bboxs or []):
            if not img_boxes:
                continue
            for box in img_boxes:
                try:
                    # типично [x1,y1,x2,y2,score,...] или объект с conf
                    if hasattr(box, "__len__") and len(box) >= 5:
                        flat_det_scores.append(float(box[4]))
                    elif hasattr(box, "conf"):
                        flat_det_scores.append(float(box.conf))
                except Exception:
                    pass

        plates = []
        for i, plate_text in enumerate(all_texts):
            if not isinstance(plate_text, str):
                continue
            plate_clean = _normalize_plate_text(plate_text)
            if not plate_clean:
                continue

            conf = None
            if i < len(flat_class_conf):
                try:
                    c = float(flat_class_conf[i])
                    if c >= 0:
                        conf = c
                except (TypeError, ValueError):
                    pass
            if conf is None and i < len(flat_det_scores):
                try:
                    conf = float(flat_det_scores[i])
                except (TypeError, ValueError):
                    pass
            if conf is None:
                conf = 0.95

            conf = max(0.0, min(1.0, float(conf)))
            plates.append(PlateResult(plate=plate_clean, confidence=conf))

        elapsed_ms = (time.time() - start) * 1000
        return ProcessFrameResponse(
            success=True,
            plates=plates,
            processing_time_ms=round(elapsed_ms, 2),
            message=f"Найдено {len(plates)} номеров" if plates else "Номера не обнаружены",
        )
        
    except Exception as e:
        logger.error("Ошибка обработки: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning("Не удалось удалить временный файл %s: %s", temp_path, e)


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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
