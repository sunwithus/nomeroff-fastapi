# -*- coding: utf-8 -*-
"""
Авторазметка кропов номеров по межкадровому консенсусу — датасет для дообучения OCR.

Зачем: 72 файла в REG_VIDEO — ровно целевой домен (та же камера, тот же угол,
те же номера), а военной головы ru_military на диске нет вообще. Размечать
руками 20 ГБ нереально, но верный ответ обычно уже есть в чтениях: номер,
совпавший в 4+ кадрах одного проезда, можно писать в разметку почти без риска.

Что делает:
  1. Разбирает видео через ffmpeg (raw MJPEG в stdout, без файлов на диске).
  2. Прогоняет кадры через тот же пайплайн, что и API (main._recognize_frame).
  3. Группирует чтения в треки по IoU bbox — один проезд машины.
  4. Голосует по позициям символов (plate_ru.vote_plate) и оставляет только
     треки с достаточным согласием.
  5. Пишет датасет в формате nomeroff (img/*.png + ann/*.json) и review.csv
     для ручной верификации — от слабых к сильным.

Дальше: ручная проверка review.csv (или просмотр img/ подряд), затем обучение
через tutorials/ju/train/ocr/ru.ipynb и ru-military.ipynb.

Примеры:
    python tools/export_ocr_dataset.py --videos D:\\REG_VIDEO --out data/dataset/ru_reg
    python tools/export_ocr_dataset.py --videos D:\\REG_VIDEO --kind military --min-agree 3
    python tools/export_ocr_dataset.py --videos D:\\REG_VIDEO --limit-videos 5 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plate_ru  # noqa: E402
from frame_prep import crop_bbox  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export-dataset")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".mts", ".m2ts", ".ts")
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


# ---------------------------------------------------------------- кадры из ffmpeg

def resolve_ffmpeg() -> str:
    """ffmpeg из PATH или из локальной папки офлайн-пакета."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (
        os.path.join(root, "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.path.dirname(root), "ffmpeg", "bin", "ffmpeg.exe"),
    ):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("ffmpeg не найден: добавьте в PATH или положите в ./ffmpeg/bin")


def iter_frames(ffmpeg: str, video_path: str, fps: float, max_frames: int | None = None):
    """
    Кадры видео как (index, time_sec, BGR ndarray).

    MJPEG в stdout: декодирование идёт потоком, промежуточных файлов нет.
    """
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "auto",
        "-i", video_path,
        "-vf", f"fps={fps:g}",
        "-q:v", "2",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = bytearray()
    index = 0
    try:
        while True:
            chunk = proc.stdout.read(1 << 16)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                start = buffer.find(JPEG_SOI)
                if start < 0:
                    break
                end = buffer.find(JPEG_EOI, start + 2)
                if end < 0:
                    break
                jpeg = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                yield index, index / fps, frame
                index += 1
                if max_frames and index >= max_frames:
                    return
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.stdout.close()
        except Exception:
            pass
        stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()
        if stderr.strip():
            logger.debug("ffmpeg: %s", stderr.strip()[:500])


# ------------------------------------------------------------------- трекинг

def iou(a, b) -> float:
    ax1, ax2 = sorted((a[0], a[2]))
    ay1, ay2 = sorted((a[1], a[3]))
    bx1, bx2 = sorted((b[0], b[2]))
    by1, by2 = sorted((b[1], b[3]))
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Reading:
    plate: str
    char_probs: list[float]
    ocr_confidence: float
    time_sec: float
    bbox: list[int]
    bbox_area: int
    crop: np.ndarray
    plate_kind: str


@dataclass
class Track:
    readings: list[Reading] = field(default_factory=list)
    last_bbox: list[int] | None = None
    last_time: float = 0.0

    def add(self, reading: Reading) -> None:
        self.readings.append(reading)
        self.last_bbox = reading.bbox
        self.last_time = reading.time_sec


class Tracker:
    """Тот же принцип, что в C# PlateTracker: один проезд — один трек."""

    def __init__(self, iou_threshold: float = 0.15, max_gap_sec: float = 3.0):
        self.iou_threshold = iou_threshold
        self.max_gap_sec = max_gap_sec
        self.tracks: list[Track] = []

    def add(self, reading: Reading) -> None:
        best, best_score = None, 0.0
        for track in self.tracks:
            if reading.time_sec - track.last_time > self.max_gap_sec:
                continue
            if not track.last_bbox:
                continue
            score = iou(track.last_bbox, reading.bbox)
            if score > best_score and score >= self.iou_threshold:
                best, best_score = track, score
        if best is None:
            best = Track()
            self.tracks.append(best)
        best.add(reading)


# ---------------------------------------------------------------- распознавание

def load_pipeline():
    """Тот же пайплайн, что у API, но без HTTP-сервера."""
    import main as api

    api.pipeline_instance = api._build_pipeline()
    return api


def recognize_video(
    api,
    ffmpeg: str,
    video_path: str,
    *,
    fps: float,
    variants: list[str],
    min_ocr_confidence: float,
    max_frames: int | None,
) -> list[Track]:
    tracker = Tracker()
    frames_seen = 0
    for _, time_sec, frame in iter_frames(ffmpeg, video_path, fps, max_frames):
        frames_seen += 1
        plates = api._recognize_frame(
            frame,
            variants,
            min_ocr_confidence=min_ocr_confidence,
            include_crop=False,
        )
        for p in plates:
            crop = crop_bbox(frame, p.bbox)
            if crop is None or crop.size == 0:
                continue
            tracker.add(Reading(
                plate=p.plate,
                char_probs=list(p.char_probs or []),
                ocr_confidence=float(p.ocr_confidence),
                time_sec=time_sec,
                bbox=list(p.bbox),
                bbox_area=int(p.bbox_area),
                crop=crop,
                plate_kind=p.plate_kind or "",
            ))
        if frames_seen % 50 == 0:
            logger.info("  %s: %d кадров, треков %d",
                        os.path.basename(video_path), frames_seen, len(tracker.tracks))
    logger.info("  %s: %d кадров, треков %d",
                os.path.basename(video_path), frames_seen, len(tracker.tracks))
    return tracker.tracks


# ------------------------------------------------------------------- экспорт

@dataclass
class Sample:
    name: str
    plate: str
    predicted: str
    confidence: float
    votes: int
    crop: np.ndarray
    video: str
    time_sec: float
    kind: str


def build_samples(
    tracks: list[Track],
    video_name: str,
    *,
    min_agree: int,
    min_confidence: float,
    crops_per_track: int,
    kind_filter: str,
) -> tuple[list[Sample], dict[str, int]]:
    stats = {"tracks": len(tracks), "too_few_frames": 0, "low_confidence": 0,
             "wrong_kind": 0, "no_consensus": 0, "exported": 0}
    samples: list[Sample] = []

    for ti, track in enumerate(tracks):
        distinct_times = {round(r.time_sec, 2) for r in track.readings}
        if len(distinct_times) < min_agree:
            stats["too_few_frames"] += 1
            continue

        voted = plate_ru.vote_plate([(r.plate, r.char_probs) for r in track.readings])
        if voted is None:
            stats["no_consensus"] += 1
            continue
        plate, confidence, _ = voted

        decoded = plate_ru.decode_constrained(plate)
        if decoded is None:
            stats["no_consensus"] += 1
            continue
        plate, plate_kind = decoded

        if kind_filter != "all" and plate_kind != kind_filter:
            stats["wrong_kind"] += 1
            continue
        if confidence < min_confidence:
            stats["low_confidence"] += 1
            continue

        # Крупные кропы информативнее: машина ближе к камере, меньше интерполяции
        best = sorted(track.readings, key=lambda r: -r.bbox_area)[:crops_per_track]
        for ci, reading in enumerate(best):
            samples.append(Sample(
                name=f"{video_name}_t{ti:03d}_{ci}_{plate}",
                plate=plate,
                predicted=reading.plate,
                confidence=confidence,
                votes=len(distinct_times),
                crop=reading.crop,
                video=video_name,
                time_sec=reading.time_sec,
                kind=plate_kind,
            ))
        stats["exported"] += 1
    return samples, stats


def write_dataset(samples: list[Sample], out_dir: str) -> None:
    img_dir = os.path.join(out_dir, "img")
    ann_dir = os.path.join(out_dir, "ann")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    for s in samples:
        height, width = s.crop.shape[:2]
        cv2.imwrite(os.path.join(img_dir, f"{s.name}.png"), s.crop)
        ann = {
            "description": s.plate,
            "name": s.name,
            # 12 = ru в разметке nomeroff; военные обучаются отдельной головой
            "region_id": 12,
            "count_lines": 1,
            "size": {"width": int(width), "height": int(height)},
            "moderation": {
                # predicted — сырое чтение до голосования, description — консенсус.
                # Расхождение этих двух полей и есть список на ручную проверку.
                "predicted": s.predicted,
                "isModerated": 0,
            },
            "auto_label": {
                "source_video": s.video,
                "time_sec": round(s.time_sec, 2),
                "vote_confidence": round(s.confidence, 4),
                "frame_votes": s.votes,
                "plate_kind": s.kind,
            },
        }
        with open(os.path.join(ann_dir, f"{s.name}.json"), "w", encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False, indent=1)


def write_review_csv(samples: list[Sample], out_dir: str) -> str:
    """Список на ручную проверку: слабые и спорные сверху."""
    path = os.path.join(out_dir, "review.csv")
    rows = sorted(
        samples,
        key=lambda s: (s.plate == s.predicted, s.confidence, s.votes),
    )
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "name", "plate", "raw_reading", "agrees",
            "vote_confidence", "frame_votes", "kind", "video", "time_sec",
        ])
        for s in rows:
            writer.writerow([
                s.name, s.plate, s.predicted, int(s.plate == s.predicted),
                f"{s.confidence:.4f}", s.votes, s.kind, s.video, f"{s.time_sec:.2f}",
            ])
    return path


# ---------------------------------------------------------------------- main

def list_videos(path: str, limit: int | None) -> list[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for name in sorted(os.listdir(path)):
        if name.lower().endswith(VIDEO_EXTS):
            files.append(os.path.join(path, name))
    return files[:limit] if limit else files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Авторазметка кропов номеров по консенсусу — датасет для дообучения OCR",
    )
    parser.add_argument("--videos", required=True,
                        help="Папка с видео или отдельный файл")
    parser.add_argument("--out", default="data/dataset/auto_ru",
                        help="Куда писать датасет (img/ + ann/ + review.csv)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Кадров в секунду на разбор (больше кадров — больше голосов)")
    parser.add_argument("--variants", default="full,roi,roi_invert",
                        help="Варианты предобработки кадра через запятую")
    parser.add_argument("--min-agree", type=int, default=4,
                        help="Минимум разных кадров с этим номером в треке")
    parser.add_argument("--min-confidence", type=float, default=0.75,
                        help="Минимальная уверенность голосования (0..1)")
    parser.add_argument("--min-ocr-confidence", type=float, default=0.5,
                        help="Порог CTC-головы для отдельного чтения")
    parser.add_argument("--crops-per-track", type=int, default=3,
                        help="Сколько кропов сохранить с одного трека")
    parser.add_argument("--kind", choices=("all", "civilian", "military"), default="all",
                        help="Экспортировать только гражданские / только военные")
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Ограничить число кадров на видео (для проб)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Посчитать статистику, ничего не записывая")
    args = parser.parse_args()

    if not os.path.exists(args.videos):
        logger.error("Путь не найден: %s", args.videos)
        return 2

    videos = list_videos(args.videos, args.limit_videos)
    if not videos:
        logger.error("Видео не найдены в %s", args.videos)
        return 2

    ffmpeg = resolve_ffmpeg()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    logger.info("Видео: %d, fps=%s, варианты=%s, min_agree=%d",
                len(videos), args.fps, variants, args.min_agree)

    api = load_pipeline()

    all_samples: list[Sample] = []
    totals = {"tracks": 0, "too_few_frames": 0, "low_confidence": 0,
              "wrong_kind": 0, "no_consensus": 0, "exported": 0}

    for i, video in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        logger.info("[%d/%d] %s", i, len(videos), name)
        try:
            tracks = recognize_video(
                api, ffmpeg, video,
                fps=args.fps,
                variants=variants,
                min_ocr_confidence=args.min_ocr_confidence,
                max_frames=args.max_frames,
            )
        except Exception as ex:
            logger.error("  пропуск %s: %s", name, ex)
            continue

        samples, stats = build_samples(
            tracks, name,
            min_agree=args.min_agree,
            min_confidence=args.min_confidence,
            crops_per_track=args.crops_per_track,
            kind_filter=args.kind,
        )
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
        all_samples.extend(samples)
        logger.info("  треков %d -> номеров %d (кропов %d)",
                    stats["tracks"], stats["exported"], len(samples))

    plates = sorted({s.plate for s in all_samples})
    logger.info(
        "Итого: треков %d, принято номеров %d, кропов %d, уникальных номеров %d",
        totals["tracks"], totals["exported"], len(all_samples), len(plates),
    )
    logger.info(
        "Отбраковано: мало кадров %d, низкая уверенность %d, не тот тип %d, нет консенсуса %d",
        totals["too_few_frames"], totals["low_confidence"],
        totals["wrong_kind"], totals["no_consensus"],
    )

    if args.dry_run:
        logger.info("--dry-run: ничего не записано")
        return 0
    if not all_samples:
        logger.warning("Нечего экспортировать")
        return 1

    os.makedirs(args.out, exist_ok=True)
    write_dataset(all_samples, args.out)
    review = write_review_csv(all_samples, args.out)
    logger.info("Датасет: %s (img/ + ann/)", os.path.abspath(args.out))
    logger.info("На проверку: %s — сверху расхождения консенсуса и сырых чтений", review)
    logger.info(
        "Дальше: проверить руками, затем tutorials/ju/train/ocr/%s",
        "ru-military.ipynb" if args.kind == "military" else "ru.ipynb",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
