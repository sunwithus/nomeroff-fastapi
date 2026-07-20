# -*- coding: utf-8 -*-
"""Прогон выборки кадров из D:\\REG_VIDEO через /api/process_frame."""
from __future__ import annotations

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

VIDEO_DIR = Path(r"D:\REG_VIDEO")
API_URL = "http://127.0.0.1:8000/api/process_frame"
HEALTH_URL = "http://127.0.0.1:8000/health"
NUM_VIDEOS = 8
FRAME_INTERVAL_SEC = 30
MAX_FRAMES_PER_VIDEO = 8
REPORT_PATH = Path(r"D:\_ANumberRecognition\logs\reg_video_cyrillic_benchmark.json")


def wait_health(timeout_sec: float = 600) -> dict:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("model_loaded"):
                    return data
        except Exception as ex:  # noqa: BLE001
            last_err = ex
        time.sleep(3)
    raise RuntimeError(f"API not ready: {last_err}")


def process_frame_bgr(frame) -> list[str]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return []
    body = json.dumps({"image_base64": base64.b64encode(buf.tobytes()).decode("ascii")}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [p.get("plate", "") for p in data.get("plates", []) if p.get("plate")]


def sample_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video": path.name, "error": "open_failed", "plates": [], "frames": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    step = int(max(1, round(fps * FRAME_INTERVAL_SEC)))

    plates: list[str] = []
    frames_checked = 0
    idx = 0
    while frames_checked < MAX_FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t0 = time.time()
        found = process_frame_bgr(frame)
        elapsed = round((time.time() - t0) * 1000, 1)
        time_sec = round(idx / fps, 1) if fps else idx
        print(f"  t={time_sec:6.1f}s  {elapsed:7.0f}ms  plates={found or '-'}", flush=True)
        plates.extend(found)
        frames_checked += 1
        idx += step
        if idx >= frame_count:
            break

    cap.release()
    unique = sorted(set(plates))
    return {
        "video": path.name,
        "duration_sec": round(duration, 1),
        "frames_checked": frames_checked,
        "plates_all": plates,
        "plates_unique": unique,
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Waiting for Nomeroff API...", flush=True)
    health = wait_health()
    print(f"API OK: {health}", flush=True)

    videos = sorted(VIDEO_DIR.glob("*.MP4"), key=lambda p: p.stat().st_mtime, reverse=True)[:NUM_VIDEOS]
    if not videos:
        print("No MP4 in REG_VIDEO", file=sys.stderr)
        return 1

    results = []
    all_unique: set[str] = set()
    for i, path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {path.name}", flush=True)
        try:
            row = sample_video(path)
        except Exception as ex:  # noqa: BLE001
            row = {"video": path.name, "error": str(ex), "plates_unique": []}
            print(f"  ERROR: {ex}", flush=True)
        results.append(row)
        all_unique.update(row.get("plates_unique") or [])

    summary = {
        "videos": len(results),
        "frame_interval_sec": FRAME_INTERVAL_SEC,
        "max_frames_per_video": MAX_FRAMES_PER_VIDEO,
        "all_unique_plates": sorted(all_unique),
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(f"videos={len(results)} unique_plates={len(all_unique)}", flush=True)
    for p in sorted(all_unique):
        print(f"  {p}", flush=True)
    print(f"report: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
