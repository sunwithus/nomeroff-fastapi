# -*- coding: utf-8 -*-
"""Размер батча YOLO: на CUDA больше, на CPU как один кадр (безопасно для i3)."""


def yolo_chunk_size(
    device: str,
    free_vram_mb: float | None = None,
    env_batch: str | None = None,
) -> int:
    """
    Сколько картинок (вариантов кадра) отдать YOLO за один forward.

    CPU: всегда 5 — как full+crop+roi+contrast+invert одного кадра.
    Не склеиваем ролики в один огромный тензор: на i3 это только раздувает RAM.

    CUDA: от свободной VRAM, с запасом. 1.4 ГБ «занято» на RTX 3070 — это веса
    + контекст CUDA, не пик активаций. На 2 ГБ карте после загрузки весов
    свободного почти нет — чанк маленький, иначе OOM.
    """
    raw = (env_batch or "").strip()
    if raw.isdigit():
        return max(1, min(int(raw), 64))

    if (device or "cpu").strip().lower() != "cuda":
        return 5

    free = 4096.0 if free_vram_mb is None else float(free_vram_mb)
    if free < 800:
        return 2
    if free < 1600:
        return 4
    if free < 2800:
        return 10
    if free < 5000:
        return 20
    return 32
