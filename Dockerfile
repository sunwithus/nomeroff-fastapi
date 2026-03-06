FROM python:3.12.4-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Переменные окружения
ENV TORCH_HOME=/app/torch_models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Копируем зависимости
COPY requirements.txt .
COPY requirements-api.txt .

# Устанавливаем Python-зависимости
RUN pip install --upgrade pip && \
    pip install "setuptools>=60.0.0,<70.0.0" wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-api.txt

# Копируем проект
COPY . .

# Скачиваем модели (это выполнится один раз при сборке)
RUN python -c "from nomeroff_net import pipeline; pipeline('number_plate_detection_and_reading', image_loader='opencv')" || echo "Предзагрузка моделей не удалась, будет выполнена при первом запросе"

# Проверяем, что main.py существует
RUN test -f main.py || echo "main.py не найден!"

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]