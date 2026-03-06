# main.py — только распознавание номеров. Watchlist, dedup, оповещения — в C#.
import os
import tempfile
import time
import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

# === Настройки ===
os.environ.setdefault('TORCH_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torch_models'))

# === Логирование ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# === Глобальное состояние ===
pipeline_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    logger.info("🚀 Загрузка модели nomeroff-net...")
    pipeline_instance = pipeline("number_plate_detection_and_reading", image_loader="opencv")
    logger.info("✅ Модель загружена")
    yield
    logger.info("🛑 Очистка ресурсов...")


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


# === Эндпоинты ===
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        "<html><head><meta charset='utf-8'><title>Nomeroff API</title></head>"
        "<body>"
        "<h2>Nomeroff API</h2>"
        "<ul>"
        "<li><a href='/test/image'>Тест: загрузка изображения</a></li>"
        "<li><a href='/test/video'>Тест: загрузка видео (первый кадр)</a></li>"
        "<li><a href='/docs'>Swagger UI</a></li>"
        "</ul>"
        "</body></html>"
    )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline_instance is not None,
        "gpu_available": False,  # при необходимости: import torch; torch.cuda.is_available()
    }


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
        
        # 🔧 Правильное извлечение текстов: unpacked[-1] = List[List[str]]
        raw_texts = unpacked[-1] if unpacked else []
        
        # "Плоское" извлечение: для каждого изображения в батче берём его номера
        all_texts = []
        for item in raw_texts:
            if isinstance(item, list):
                all_texts.extend(item)  # Распаковываем вложенный список
            elif isinstance(item, str) and item.strip():
                all_texts.append(item)

        plates = []
        for plate_text in all_texts:
            # Дополнительная защита: пропускаем не-строки
            if not isinstance(plate_text, str):
                continue
            plate_clean = plate_text.strip().upper()
            if plate_clean:
                plates.append(PlateResult(plate=plate_clean))

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
    <body>
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
    <body>
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
