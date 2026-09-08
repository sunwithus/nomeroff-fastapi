@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo   Nomeroff OCR API  (port 8000)
echo ========================================
echo.

set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [X] Нет venv: %PY%
    echo.
    echo     1^) setup-python-local.bat   ^(Python в папку python\^)
    echo     2^) install.bat
    pause
    exit /b 1
)

rem После копирования на другой диск/ПК поправить home в pyvenv.cfg
call "%~dp0_fix-venv-home.bat"

"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)" 2>nul
if errorlevel 1 (
    echo [X] venv не запускается. Обычно нужен рядом python\:
    echo       setup-python-local.bat
    echo       install.bat
    pause
    exit /b 1
)

if not exist "torch_models" mkdir torch_models

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem Офлайн-ПК без git.exe
set "GIT_PYTHON_REFRESH=quiet"
rem Старые GPU (GT 710 и т.п.) auto уйдёт на CPU; явно: set NOMEROFF_DEVICE=cpu
if not defined NOMEROFF_DEVICE set "NOMEROFF_DEVICE=auto"
rem Дочитывать квадратные/двухстрочные номера (eu_2lines); 0 — выключить
if not defined NOMEROFF_TWO_LINE set "NOMEROFF_TWO_LINE=1"
rem Детектор: yolov11x (по умолчанию) или yolov11m/l быстрее; TensorRT: python tools/export_yolo_trt.py
if not defined NOMEROFF_YOLO set "NOMEROFF_YOLO=yolov11x"

echo [OK] %PY%
"%PY%" -c "import sys; print('    Python', sys.version.split()[0])"
echo.
echo     http://127.0.0.1:8000
echo     Docs: http://127.0.0.1:8000/docs
echo     NOMEROFF_DEVICE=%NOMEROFF_DEVICE%
echo     Stop: Ctrl+C
echo ========================================
echo.

"%PY%" main.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if %EXIT_CODE% neq 0 (
    echo [X] OCR завершился с кодом %EXIT_CODE%
) else (
    echo [OK] OCR остановлен
)
pause
exit /b %EXIT_CODE%
