# 📸 Nomeroff Net API

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)](https://fastapi.tiangolo.com/)
[![Nomeroff-Net](https://img.shields.io/badge/Nomeroff--Net-1.3.2-orange)](https://github.com/ria-com/nomeroff-net)

REST API сервис для распознавания автомобильных номеров на базе популярного фреймворка Nomeroff-Net. Этот проект представляет собой готовую к использованию обёртку, которая позволяет легко интегрировать распознавание номеров (ANPR/ALPR) в ваши приложения через простые HTTP-запросы.

## 🚀 Ключевые особенности

- Готовый эндпоинт `/api/process_frame` для распознавания номеров из изображений (base64)
- Поддержка изображений и видео (автоматически берётся первый кадр)
- Автоматическая загрузка моделей при первом запуске
- CORS настроен для работы с браузерами и мобильными приложениями
- Встроенный Swagger UI (`/docs`) для тестирования
- Оптимизирован для Windows (пути к моделям, временные файлы)

## 🛠 Стек технологий

- **Ядро распознавания**: Nomeroff-Net (YOLOv8 + RNN OCR)
- **Фреймворк API**: FastAPI

## Установка

### Предварительные требования

- Python 3.12.4 (или 3.9+)
- Git
- (Опционально) Видеокарта NVIDIA с поддержкой CUDA для ускорения

### Автоматическая установка (для Python 3.12.4, Windows)
bash
git clone https://github.com/sunwithus/nomeroff-fastapi.git
cd nomeroff-fastapi
   
   запустите install.bat, после установки для запуска использовать start.bat

### Пошаговая инструкция

1. **Клонируйте репозиторий:**

bash
git clone https://github.com/sunwithus/nomeroff-fastapi.git
cd nomeroff-fastapi
   
Создайте и активируйте виртуальное окружение:

bash
python -m venv venv
source venv/bin/activate

Обновите pip и установите базовые пакеты:

bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=60.0.0,<70.0.0" wheel
Установите PyTorch:

С поддержкой CUDA 11.8:

bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Установите зависимости проекта:

bash
pip install -r requirements.txt
pip install -r requirements-api.txt
Проверьте установку:

bash
python -c "import torch, nomeroff_net, cv2; print('✅ Всё готово к работе!')"
python test.py
🚀 Запуск сервера
bash
python main.py
Сервер запустится по адресу: http://127.0.0.1:8000

Доступные интерфейсы
Корневой путь / — простая HTML-страница с навигацией

Документация Swagger: /docs

Альтернативная документация ReDoc: /redoc

📡 API Эндпоинты
1. Проверка состояния сервера
GET /health

Ответ:

json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_available": false
}
2. Распознавание номера по кадру
POST /api/process_frame

Основной эндпоинт для интеграции. Принимает изображение в base64.

Тело запроса (JSON):

json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
}
Пример успешного ответа (200 OK):

json
{
  "success": true,
  "plates": [
    {
      "plate": "AC4921CB",
      "confidence": 0.95,
      "bbox": [0, 0, 0, 0]
    }
  ],
  "processing_time_ms": 245.32,
  "message": "Найдено 1 номеров"
}
Примечание: в текущей версии confidence и bbox возвращаются как значения по умолчанию.

3. Тестовые HTML-формы
GET /test/image — форма для загрузки изображения

GET /test/video — форма для загрузки видео (обрабатывается первый кадр)

🗂 Структура проекта
text
nomeroff-fastapi/
│
├── main.py                 # Основной файл FastAPI приложения
├── test.py                 # Простой тест для проверки установки
├── requirements.txt        # Зависимости оригинального Nomeroff-Net
├── requirements-api.txt    # Доп. зависимости для API (fastapi, uvicorn)
│
├── torch_models/           # Директория для скачанных моделей (создаётся автоматически)
├── data/                   # Примеры изображений
│   └── examples/
│
└── venv/                   # Виртуальное окружение
⚙️ Конфигурация
Переменные окружения (можно задать в системе или в коде):

TORCH_HOME — путь для скачивания моделей (по умолчанию: ./torch_models)

📝 Примечания для Windows
Путь к моделям задаётся через os.environ['TORCH_HOME'] для корректной работы

Временные файлы корректно удаляются после обработки

Если OpenCV жалуется на кодеки, установите opencv-python-headless вместо обычного

🧠 Как это работает
Клиент отправляет изображение в формате base64

Сервер декодирует изображение и сохраняет его во временный файл

Nomeroff-Net pipeline загружает предобученные модели (YOLO для детекции, RNN для OCR)

Модели обрабатывают изображение и возвращают распознанные номера

Результат очищается, форматируется и отправляется клиенту

🤝 Вклад в проект
Contributions are welcome! Если вы хотите улучшить проект:

Форкните репозиторий

Создайте ветку для фичи (git checkout -b feature/amazing-feature)

Закоммитьте изменения (git commit -m 'Add some amazing feature')

Запушьте ветку (git push origin feature/amazing-feature)

Откройте Pull Request

📄 Лицензия
Этот проект является форком Nomeroff Net и распространяется под той же лицензией (уточните в оригинальном репозитории). Код API-обёртки предоставляется "AS IS".

🙏 Благодарности
Команде RIA.com за создание и поддержку великолепного фреймворка Nomeroff-Net

Сообществу FastAPI за отличный инструмент для создания API

⭐ Если проект оказался полезным, поставьте звезду на GitHub!
