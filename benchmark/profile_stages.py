# -*- coding: utf-8 -*-
"""
Где именно уходит время: ffmpeg, JSON/base64 транспорт, детектор+OCR, OSD.

Стенд говорит «225 с на минуту видео», но не говорит, что именно тормозит.
Сокращение вариантов предобработки дало всего 4%, значит основной расход
не в детекторе — этот скрипт замеряет этапы по отдельности.

    python benchmark/profile_stages.py --frames 32
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OCR_BASE = "http://127.0.0.1:8000"
VIDEO = Path(r"D:\REG_VIDEO\NO20260707-145101-000033F.MP4")
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def grab_frames(video: Path, fps: float, limit: int) -> tuple[list[bytes], float]:
    ffmpeg = shutil.which("ffmpeg") or str(Path(__file__).parent.parent / "ffmpeg" / "bin" / "ffmpeg.exe")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "auto", "-i", str(video),
        "-vf", f"fps={fps:g}", "-q:v", "2",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames: list[bytes] = []
    buf = bytearray()
    while len(frames) < limit:
        chunk = proc.stdout.read(1 << 16)
        if not chunk:
            break
        buf.extend(chunk)
        while len(frames) < limit:
            start = buf.find(JPEG_SOI)
            if start < 0:
                break
            end = buf.find(JPEG_EOI, start + 2)
            if end < 0:
                break
            frames.append(bytes(buf[start:end + 2]))
            del buf[:end + 2]
    proc.kill()
    proc.wait()
    return frames, time.time() - t0


def post(path: str, payload: dict, timeout: float = 900) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OCR_BASE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--variants", default="full,crop,roi,roi_contrast")
    args = ap.parse_args()

    print(f"Кадры: {args.frames} @ {args.fps} fps из {VIDEO.name}")
    frames, ffmpeg_sec = grab_frames(VIDEO, args.fps, args.frames)
    if not frames:
        print("ffmpeg не отдал кадры", file=sys.stderr)
        return 2
    avg_kb = sum(len(f) for f in frames) / len(frames) / 1024
    print(f"  ffmpeg: {ffmpeg_sec:.1f}с на {len(frames)} кадров "
          f"({ffmpeg_sec / len(frames) * 1000:.0f} мс/кадр), средний JPEG {avg_kb:.0f} КБ")

    t0 = time.time()
    encoded = [base64.b64encode(f).decode("ascii") for f in frames]
    b64_sec = time.time() - t0
    print(f"  base64: {b64_sec:.2f}с ({b64_sec / len(frames) * 1000:.0f} мс/кадр)")

    variants = [v for v in args.variants.split(",") if v]
    total_wall = 0.0
    total_py = 0.0
    for i in range(0, len(encoded), args.batch):
        chunk = encoded[i:i + args.batch]
        body, wall = post("/api/process_frames", {
            "frames": [{"image_base64": b, "time_sec": (i + j) / args.fps}
                       for j, b in enumerate(chunk)],
            "variants": variants,
            "min_ocr_confidence": 0.5,
            "include_crop": True,
            "min_frame_hits": 1,
        })
        py = float(body.get("processing_time_ms") or 0) / 1000
        total_wall += wall
        total_py += py
        print(f"  батч {i // args.batch + 1}: wall {wall:.1f}с, python {py:.1f}с, "
              f"транспорт {wall - py:.1f}с")

    n = len(frames)
    print(f"\nИтого на {n} кадров ({args.variants}):")
    print(f"  ffmpeg           {ffmpeg_sec:7.1f}с")
    print(f"  base64           {b64_sec:7.1f}с")
    print(f"  python (OCR)     {total_py:7.1f}с  {total_py / n * 1000:.0f} мс/кадр")
    print(f"  транспорт+JSON   {total_wall - total_py:7.1f}с  "
          f"{(total_wall - total_py) / n * 1000:.0f} мс/кадр")
    video_sec = n / args.fps
    print(f"\n  на минуту видео: {(ffmpeg_sec + b64_sec + total_wall) / video_sec * 60:.0f}с "
          f"(без OSD и без записи кадров в БД)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
