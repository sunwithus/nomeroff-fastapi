# -*- coding: utf-8 -*-
"""
Регрессионный стенд ANPR: прогон видео через полный конвейер (.NET /api/process-video-path)
и счёт precision / recall / фантомов / секунд на минуту видео против benchmark/ground_truth.json.

Оба сервиса должны быть запущены (start-app.bat): OCR :8000 и App :5555.

    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --label after-consensus
    python benchmark/run_benchmark.py --engine python   # только Python OCR, без C#-голосования

Отчёт: logs/benchmark_<label>.json + сравнение с предыдущим прогоном в logs/.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
VIDEO_DIR = Path(r"D:\REG_VIDEO")
LOG_DIR = REPO_ROOT / "logs"
GROUND_TRUTH = Path(__file__).resolve().parent / "ground_truth.json"

APP_BASE = "http://127.0.0.1:5555"
OCR_BASE = "http://127.0.0.1:8000"

_LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
})


def normalize(plate: str) -> str:
    if not plate:
        return ""
    s = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]", "", plate).upper()
    return s.translate(_LATIN_TO_CYRILLIC)


def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_services(timeout_sec: float = 900) -> dict:
    """Ждём и OCR :8000 (модель загружена), и App :5555."""
    deadline = time.time() + timeout_sec
    ocr_health: dict = {}
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            if not ocr_health.get("model_loaded"):
                ocr_health = _get_json(f"{OCR_BASE}/health")
            if ocr_health.get("model_loaded"):
                _get_json(f"{APP_BASE}/health")
                return ocr_health
        except Exception as ex:  # noqa: BLE001
            last_err = ex
        time.sleep(3)
    raise RuntimeError(f"Сервисы не поднялись за {timeout_sec:.0f}с: {last_err}")


def run_video_dotnet(path: Path, sample_fps: float) -> tuple[dict, float]:
    """POST /api/process-video-path, читаем NDJSON-стрим. Возвращает (result, elapsed_sec)."""
    body = json.dumps({"path": str(path)}).encode("utf-8")
    url = f"{APP_BASE}/api/process-video-path?sampleFps={sample_fps}"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    result: dict = {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60 * 60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = (evt.get("type") or evt.get("Type") or "").lower()
            if kind == "error":
                raise RuntimeError(evt.get("message") or evt.get("Message") or "pipeline error")
            if kind == "progress":
                msg = evt.get("message") or evt.get("Message") or ""
                print(f"    .. {msg}", end="\r", flush=True)
                continue
            result = evt
    print(" " * 78, end="\r")
    return result, time.time() - t0


def run_video_dotnet_analyzed(path: Path, sample_fps: float) -> tuple[dict, float]:
    """
    POST /api/analyze-video-path: за один прогон и сырые чтения, и итог после
    голосования.

    Сырые чтения показывают работу распознавателя, но пользователь видит
    результат после трекинга, голосования и порога по числу кадров, поэтому
    считать надо оба уровня.
    """
    body = json.dumps({"path": str(path), "sampleFps": sample_fps}).encode("utf-8")
    req = urllib.request.Request(
        f"{APP_BASE}/api/analyze-video-path",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60 * 60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.time() - t0


def collect_tracks(result: dict) -> dict[str, dict]:
    """Итоговые номера после голосования: plate -> {hits, frames, ...}."""
    agg: dict[str, dict] = {}
    for tr in result.get("plates") or []:
        plate = normalize(str(tr.get("plate", "")))
        if not plate:
            continue
        row = agg.setdefault(
            plate, {"hits": 0, "frames": [], "best_conf": 0.0, "best_ocr_conf": 0.0}
        )
        row["hits"] += 1
        row["frames"].append(round(float(tr.get("timeSec") or 0.0), 2))
        row["best_conf"] = max(row["best_conf"], float(tr.get("confidence") or 0.0))
        row["best_ocr_conf"] = row["best_conf"]
    return agg


def run_video_python(path: Path, sample_fps: float) -> tuple[dict, float]:
    """Резервный режим: кадры через OpenCV прямо в /api/process_frame (без C#-голосования)."""
    import cv2  # локальный импорт: нужен только этому режиму

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не открылось: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / max(sample_fps, 0.01))))

    frames: list[dict] = []
    t0 = time.time()
    idx = 0
    while idx < total or total == 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            payload = json.dumps(
                {"image_base64": base64.b64encode(buf.tobytes()).decode("ascii")}
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{OCR_BASE}/api/process_frame",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            frames.append(
                {
                    "timeSec": round(idx / fps, 2),
                    "plates": [
                        {
                            "plate": p.get("plate", ""),
                            "confidence": p.get("confidence", 0.0),
                            "ocrConfidence": p.get("ocr_confidence", 0.0),
                        }
                        for p in data.get("plates", [])
                    ],
                }
            )
        idx += step
    cap.release()
    return {"totalFrames": len(frames), "results": frames}, time.time() - t0


def _frames_of(result: dict) -> list[dict]:
    return result.get("results") or result.get("Results") or []


def _plates_of(frame: dict) -> list[dict]:
    return frame.get("plates") or frame.get("Plates") or []


def _field(d: dict, *names, default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def collect_readings(result: dict) -> dict[str, dict]:
    """plate -> {hits, frames, best_conf, best_ocr_conf}."""
    agg: dict[str, dict] = {}
    for fr in _frames_of(result):
        t = _field(fr, "timeSec", "TimeSec", default=0.0)
        seen_in_frame: set[str] = set()
        for p in _plates_of(fr):
            plate = normalize(str(_field(p, "plate", "Plate", default="")))
            if not plate or plate in seen_in_frame:
                continue
            seen_in_frame.add(plate)
            conf = float(_field(p, "confidence", "Confidence", default=0.0) or 0.0)
            ocr = float(
                _field(p, "ocrConfidence", "OcrConfidence", "ocr_confidence", default=0.0) or 0.0
            )
            row = agg.setdefault(
                plate, {"hits": 0, "frames": [], "best_conf": 0.0, "best_ocr_conf": 0.0}
            )
            row["hits"] += 1
            row["frames"].append(round(float(t), 2))
            row["best_conf"] = max(row["best_conf"], conf)
            row["best_ocr_conf"] = max(row["best_ocr_conf"], ocr)
    return agg


def score(expected: list[str], optional: list[str], found: dict[str, dict]) -> dict:
    exp = {normalize(p) for p in expected}
    opt = {normalize(p) for p in optional}
    got = set(found)

    hit = sorted(exp & got)
    missed = sorted(exp - got)
    phantoms = sorted(got - exp - opt)

    tp, fp = len(hit), len(phantoms)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(exp) if exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "expected": sorted(exp),
        "hit": hit,
        "missed": missed,
        "phantoms": phantoms,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "phantom_count": fp,
    }


def check_timestamps(result: dict, start_local: str | None) -> dict:
    """Дата/время: сколько кадров получили осмысленное время и насколько оно совпадает с эталоном."""
    if not start_local:
        return {"checked": 0}
    try:
        base = datetime.fromisoformat(start_local)
    except ValueError:
        return {"checked": 0, "error": f"bad start_local: {start_local}"}

    checked = 0
    within_2s = 0
    worst = 0.0
    for fr in _frames_of(result):
        raw = _field(fr, "overlayTimeUtc", "OverlayTimeUtc", "timeUtc", "TimeUtc")
        if not raw:
            continue
        try:
            got = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        t = float(_field(fr, "timeSec", "TimeSec", default=0.0) or 0.0)
        want = base + timedelta(seconds=t)
        # эталон в локальном времени; приводим обе точки к naive-локали
        if got.tzinfo is not None:
            got = got.astimezone().replace(tzinfo=None)
        delta = abs((got - want).total_seconds())
        checked += 1
        worst = max(worst, delta)
        if delta <= 2.0:
            within_2s += 1
    return {
        "checked": checked,
        "within_2s": within_2s,
        "worst_delta_sec": round(worst, 1),
        "ok": checked > 0 and within_2s == checked,
    }


def previous_report(label: str) -> dict | None:
    files = sorted(LOG_DIR.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if f.stem == f"benchmark_{label}":
            continue
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M"),
                    help="метка прогона для имени отчёта")
    ap.add_argument("--engine", choices=["dotnet", "python"], default="dotnet",
                    help="dotnet = полный конвейер с голосованием; python = только OCR")
    ap.add_argument("--sample-fps", type=float, default=3.0)
    ap.add_argument("--video", action="append", default=None,
                    help="обработать только указанные файлы (можно повторять)")
    args = ap.parse_args()

    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ждём сервисы (OCR :8000, App :5555)...", flush=True)
    health = wait_services()
    print(f"OCR: device={health.get('device')} {health.get('device_name') or ''}", flush=True)

    runner = run_video_dotnet_analyzed if args.engine == "dotnet" else run_video_python
    rows: list[dict] = []
    tot_tp = tot_fp = tot_exp = 0
    tot_sec = tot_video_sec = 0.0

    wanted = {v.lower() for v in (args.video or [])}
    for spec in gt["videos"]:
        name = spec["file"]
        if wanted and name.lower() not in wanted:
            continue
        path = VIDEO_DIR / name
        if not path.exists():
            print(f"[skip] нет файла {path}", file=sys.stderr)
            continue

        print(f"\n[{name}] engine={args.engine} fps={args.sample_fps}", flush=True)
        try:
            result, elapsed = runner(path, args.sample_fps)
        except Exception as ex:  # noqa: BLE001
            print(f"  ОШИБКА: {ex}", file=sys.stderr)
            rows.append({"video": name, "error": str(ex)})
            continue

        raw = collect_readings(result)
        raw_sc = score(spec.get("plates", []), spec.get("optional", []), raw)

        # Итоговая оценка — по тому, что ушло бы в БД. У python-движка голосования
        # нет, поэтому там оба уровня совпадают.
        tracks = collect_tracks(result)
        found = tracks if tracks else raw
        sc = score(spec.get("plates", []), spec.get("optional", []), found)

        dur = float(spec.get("duration_sec") or 0) or 60.0
        sec_per_min = elapsed / (dur / 60.0)

        row = {
            "video": name,
            "engine": args.engine,
            "sample_fps": args.sample_fps,
            "duration_sec": dur,
            "elapsed_sec": round(elapsed, 1),
            "sec_per_video_minute": round(sec_per_min, 1),
            "realtime_factor": round(elapsed / dur, 2),
            "frames": len(_frames_of(result)),
            "readings": {k: v for k, v in sorted(found.items())},
            "raw_readings": {k: v for k, v in sorted(raw.items())},
            "raw_precision": raw_sc["precision"],
            "raw_recall": raw_sc["recall"],
            "raw_phantom_count": raw_sc["phantom_count"],
            "datetime": check_timestamps(result, spec.get("start_local")),
            **sc,
        }
        rows.append(row)
        tot_tp += len(sc["hit"])
        tot_fp += sc["phantom_count"]
        tot_exp += len(sc["expected"])
        tot_sec += elapsed
        tot_video_sec += dur

        print(f"  precision={sc['precision']:.2%} recall={sc['recall']:.2%} "
              f"фантомов={sc['phantom_count']} {sec_per_min:.0f}с/мин видео")
        print(f"  сырые чтения (до голосования): precision={raw_sc['precision']:.2%} "
              f"recall={raw_sc['recall']:.2%} фантомов={raw_sc['phantom_count']}")
        if sc["hit"]:
            print(f"  + найдено: {', '.join(sc['hit'])}")
        if sc["missed"]:
            print(f"  - пропущено: {', '.join(sc['missed'])}")
        if sc["phantoms"]:
            hits = {p: found[p]["hits"] for p in sc["phantoms"]}
            print(f"  ! фантомы: {hits}")

    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    recall = tot_tp / tot_exp if tot_exp else 0.0
    summary = {
        "label": args.label,
        "when": datetime.now().isoformat(timespec="seconds"),
        "engine": args.engine,
        "sample_fps": args.sample_fps,
        "device": health.get("device"),
        "device_name": health.get("device_name"),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "phantoms": tot_fp,
        "sec_per_video_minute": round(tot_sec / (tot_video_sec / 60.0), 1) if tot_video_sec else 0.0,
        "targets": gt.get("targets", {}),
        "videos": rows,
    }

    out = LOG_DIR / f"benchmark_{args.label}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    tgt = gt.get("targets", {})
    print("\n=== ИТОГ ===")
    print(f"  precision            {precision:.2%}   (цель >= {tgt.get('precision', 0):.0%})")
    print(f"  recall               {recall:.2%}   (цель >= {tgt.get('recall', 0):.0%})")
    print(f"  фантомы              {tot_fp}       (цель {tgt.get('phantoms', 0)})")
    print(f"  сек / мин видео      {summary['sec_per_video_minute']}   (цель <= {tgt.get('sec_per_video_minute', 0)})")

    prev = previous_report(args.label)
    if prev:
        def d(key: str, fmt: str = "+.2%") -> str:
            a, b = prev.get(key), summary.get(key)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                return "n/a"
            return format(b - a, fmt)
        print(f"\n  против «{prev.get('label')}»: "
              f"precision {d('precision')}, recall {d('recall')}, "
              f"фантомы {d('phantoms', '+d')}, время {d('sec_per_video_minute', '+.1f')}с/мин")

    print(f"\n  отчёт: {out}")

    ok = (precision >= tgt.get("precision", 0)
          and recall >= tgt.get("recall", 0)
          and tot_fp <= tgt.get("phantoms", 0))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
