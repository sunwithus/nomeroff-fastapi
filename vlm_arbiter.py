# -*- coding: utf-8 -*-
"""
Локальный VLM вторым каскадом — только арбитр спорных треков, не детектор.

Почему именно так:
  * Основным проходом VLM не годится. 8 ГБ VRAM делятся с nomeroff, кроп читается
    на порядок медленнее CRNN, а главное — VLM галлюцинирует правдоподобные
    номера, то есть ровно тот режим отказа, с которым мы и боремся.
  * Зато на треках, где голосование не сошлось (5–15% случаев), у нас уже есть
    список кандидатов. Тогда задача VLM сводится к выбору из вариантов, а
    любой ответ вне валидного формата РФ просто отбрасывается.

Выключено по умолчанию (NOMEROFF_VLM=0). Машина готовится офлайн, поэтому
модель никогда не скачивается на ходу: нужен локальный путь в NOMEROFF_VLM_MODEL,
иначе арбитр молча остаётся выключенным.

Включение:
    set NOMEROFF_VLM=1
    set NOMEROFF_VLM_MODEL=D:\\models\\Qwen2.5-VL-7B-Instruct-4bit
    set NOMEROFF_VLM_MIN_CONF=0.80
"""
from __future__ import annotations

import logging
import os
import threading

import cv2
import numpy as np

import plate_ru

logger = logging.getLogger(__name__)

_PROMPT = (
    "На фото — российский автомобильный номер. "
    "Выведи только символы номера без пробелов, дефисов и пояснений. "
    "Если номер не читается, выведи NONE."
)

_lock = threading.Lock()
_model = None
_processor = None
_load_failed = False


def is_enabled() -> bool:
    """Арбитр включён и есть локальный путь к модели."""
    flag = (os.environ.get("NOMEROFF_VLM") or "0").strip().lower()
    if flag in ("0", "false", "no", "off", ""):
        return False
    return bool(model_path())


def model_path() -> str:
    return (os.environ.get("NOMEROFF_VLM_MODEL") or "").strip()


def min_confidence() -> float:
    """Треки с уверенностью голосования ниже порога отдаются арбитру."""
    try:
        return float(os.environ.get("NOMEROFF_VLM_MIN_CONF", "0.80"))
    except ValueError:
        return 0.80


def _load():
    """
    Ленивая загрузка. Любая ошибка — арбитр выключается навсегда до перезапуска:
    на офлайн-ПК падать из-за отсутствующей опциональной модели нельзя.
    """
    global _model, _processor, _load_failed
    if _model is not None or _load_failed:
        return _model, _processor

    with _lock:
        if _model is not None or _load_failed:
            return _model, _processor

        path = model_path()
        if not path or not os.path.isdir(path):
            logger.warning("VLM-арбитр: NOMEROFF_VLM_MODEL не указывает на локальную папку — выключен")
            _load_failed = True
            return None, None
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            _processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            _model = AutoModelForImageTextToText.from_pretrained(
                path,
                local_files_only=True,
                dtype=torch.float16,
                device_map="auto",
            )
            _model.eval()
            logger.info("VLM-арбитр загружен: %s", path)
        except Exception as ex:
            logger.warning("VLM-арбитр недоступен (%s) — работаем без него", ex)
            _model, _processor = None, None
            _load_failed = True
    return _model, _processor


def _read_plate(crop_bgr: np.ndarray) -> str:
    model, processor = _load()
    if model is None or processor is None:
        return ""

    import torch
    from PIL import Image

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
    }]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def arbitrate(crop_bgr: np.ndarray | None, candidates: list[str]) -> tuple[str, str] | None:
    """
    Выбрать номер по кропу.

    Returns:
        (plate, source) где source = 'vlm_candidate' если ответ совпал с одним из
        кандидатов голосования, или 'vlm_new' если это другой валидный номер РФ.
        None — арбитр выключен, не загрузился, или ответ не прошёл валидацию.

    Ответ вне формата РФ или с несуществующим кодом региона отбрасывается: это
    и есть защита от галлюцинаций, из-за которых VLM нельзя ставить основным.
    """
    if crop_bgr is None or crop_bgr.size == 0 or not is_enabled():
        return None

    try:
        raw = _read_plate(crop_bgr)
    except Exception as ex:
        logger.warning("VLM-арбитр: ошибка чтения (%s)", ex)
        return None

    if not raw or "NONE" in raw.upper():
        return None

    decoded = plate_ru.decode_constrained(raw)
    if decoded is None:
        logger.info("VLM-арбитр: ответ %r невалиден для РФ — игнорируем", raw)
        return None
    plate = decoded[0]

    normalized = {plate_ru.normalize_plate(c) for c in candidates}
    if plate in normalized:
        return plate, "vlm_candidate"

    logger.info("VLM-арбитр: %r нет среди кандидатов %s", plate, sorted(normalized))
    return plate, "vlm_new"
