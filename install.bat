@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: =============================================
:: Nomeroff Net API - Запуск сервера
:: =============================================
title Nomeroff Net API Server

echo ========================================
echo    Nomeroff Net API Server Launcher
echo ========================================
echo.

:: Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [❌] Python не найден! Установите Python 3.9 или выше.
    echo     Скачать: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Проверка версии Python (должна быть 3.9+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo [✅] Найдена версия Python: %pyver%

:: Проверка наличия виртуального окружения
if not exist "venv\Scripts\activate.bat" (
    echo [⚙️] Виртуальное окружение не найдено. Создаём...
    
    echo [1/5] Создание виртуального окружения...
    python -m venv venv
    if errorlevel 1 (
        echo [❌] Ошибка создания виртуального окружения!
        pause
        exit /b 1
    )
    
    echo [2/5] Активация окружения...
    call venv\Scripts\activate.bat
    
    echo [3/5] Обновление pip и установка базовых пакетов...
    python -m pip install --upgrade pip >nul
    pip install "setuptools>=60.0.0,<70.0.0" wheel >nul
    
    echo [4/5] Установка PyTorch (CUDA 11.8)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    echo [5/5] Установка зависимостей проекта...
    if exist requirements.txt (
        pip install -r requirements.txt
    ) else (
        echo [⚠️] Файл requirements.txt не найден
    )
    
    if exist requirements-api.txt (
        pip install -r requirements-api.txt
    )
    
    echo.
    echo [✅] Установка завершена!
) else (
    echo [✅] Виртуальное окружение найдено
    call venv\Scripts\activate.bat
)

:: Очистка экрана перед запуском
cls

:: Создание директории для моделей, если её нет
if not exist torch_models mkdir torch_models

:: Запуск сервера
echo.
echo ========================================
echo    🚀 Запуск Nomeroff Net API Server
echo ========================================
echo.
echo 📡 Сервер будет доступен по адресу:
echo    http://127.0.0.1:8000
echo    http://localhost:8000
echo.
echo 📚 Документация:
echo    Swagger UI: http://127.0.0.1:8000/docs
echo    ReDoc:      http://127.0.0.1:8000/redoc
echo.
echo 🔧 Для остановки сервера нажмите Ctrl+C
echo ========================================
echo.

:: Запуск main.py
python main.py

:: Если произошла ошибка, показать сообщение и подождать
if errorlevel 1 (
    echo.
    echo [❌] Сервер завершился с ошибкой!
    echo     Проверьте вывод выше для диагностики.
    pause
)

:: Деактивация окружения при выходе
deactivate