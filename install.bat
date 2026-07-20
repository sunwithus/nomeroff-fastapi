@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

echo ========================================
echo   Nomeroff OCR - установка venv
echo ========================================
echo.

call "%~dp0_resolve-python.bat"
if not defined PYEXE if not defined PYLAUNCH (
    echo [X] Python не найден.
    echo.
    echo     Без установки в систему:
    echo       setup-python-local.bat
    echo     ^(скачает Python 3.12 в nomeroff-net\python^)
    echo.
    echo     Или положите готовый Python сюда:
    echo       nomeroff-net\python\python.exe
    echo.
    pause
    exit /b 1
)

if defined PYEXE (
    echo [>] Базовый Python: %PYEXE%
    "%PYEXE%" -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3,9) else 1)"
) else (
    echo [>] Базовый Python: %PYLAUNCH%
    %PYLAUNCH% -c "import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3,9) else 1)"
)
if errorlevel 1 (
    echo [X] Нужен Python 3.9+. Сейчас не подходит.
    pause
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    echo [~] Удаляю старый venv...
    rmdir /s /q venv 2>nul
)

echo [>] Создаю venv...
if defined PYEXE (
    "%PYEXE%" -m venv venv --copies
    if errorlevel 1 "%PYEXE%" -m venv venv
) else (
    %PYLAUNCH% -m venv venv --copies
    if errorlevel 1 %PYLAUNCH% -m venv venv
)
if errorlevel 1 (
    echo [X] Не удалось создать venv
    pause
    exit /b 1
)

call "%~dp0_fix-venv-home.bat"

set "PIP=%~dp0venv\Scripts\python.exe"
"%PIP%" -m pip install --upgrade pip
"%PIP%" -m pip install "setuptools>=60.0.0,<70.0.0" wheel

echo [>] PyTorch (CUDA 11.8). При ошибке пробую CPU...
"%PIP%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [!] CUDA wheel не встал - пробую CPU...
    "%PIP%" -m pip install torch torchvision torchaudio
)

echo [>] requirements...
"%PIP%" -m pip install -r requirements.txt
if errorlevel 1 (echo [X] requirements.txt & pause & exit /b 1)
"%PIP%" -m pip install -r requirements-api.txt
if errorlevel 1 (echo [X] requirements-api.txt & pause & exit /b 1)

if not exist "torch_models" mkdir torch_models

echo.
echo [OK] Готово.
echo     Запуск: start.bat  или  ..\start-app.bat
echo     На другой ПК копируйте вместе: python\ + venv\ + код
pause
exit /b 0
