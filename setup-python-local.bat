@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"

rem Установка Python 3.12 В ПАПКУ nomeroff-net\python (без PATH / без Install for all users).
rem Дальше: install.bat

set "TARGET=%~dp0python"
set "VER=3.12.4"
set "INSTALLER=%TEMP%\python-%VER%-amd64.exe"
set "URL=https://www.python.org/ftp/python/%VER%/python-%VER%-amd64.exe"

echo ========================================
echo   Локальный Python %VER% -^> python\
echo ========================================
echo.
echo Будет установлено в:
echo   %TARGET%
echo   ^(системный PATH не трогаем^)
echo.

if exist "%TARGET%\python.exe" (
    echo [OK] Уже есть: %TARGET%\python.exe
    "%TARGET%\python.exe" -c "import sys; print(sys.version)"
    echo.
    echo Если нужно переустановить - удалите папку python\ и запустите снова.
    pause
    exit /b 0
)

if exist "%TARGET%\python.exe" (
    echo [OK] Уже есть: %TARGET%\python.exe
    "%TARGET%\python.exe" -c "import sys; print(sys.version)"
    echo.
    echo Если нужно переустановить - удалите папку python\ и запустите снова.
    pause
    exit /b 0
)

rem Если установщик лежит в корне репозитория — используем его
set "ROOT_INSTALLER=%~dp0..\python-3.12.4-amd64.exe"
if exist "%ROOT_INSTALLER%" set "INSTALLER=%ROOT_INSTALLER%"

if not exist "%INSTALLER%" (
  echo [>] Скачиваю установщик...
  powershell -NoProfile -Command ^
    "try { Invoke-WebRequest -Uri '%URL%' -OutFile '%INSTALLER%' -UseBasicParsing } catch { exit 1 }"
  if errorlevel 1 (
      echo [X] Не удалось скачать %URL%
      echo     Скопируйте готовый Python в:
      echo     %TARGET%
      pause
      exit /b 1
  )
) else (
  echo [>] Найден установщик: %INSTALLER%
)

echo [>] Тихая установка в папку проекта...
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 SimpleInstall=1 TargetDir="%TARGET%"
if errorlevel 1 (
    echo [X] Установка не удалась
    pause
    exit /b 1
)

if not exist "%TARGET%\python.exe" (
    echo [X] После установки нет %TARGET%\python.exe
    pause
    exit /b 1
)

echo [>] ensurepip / pip
"%TARGET%\python.exe" -m ensurepip --upgrade 2>nul
"%TARGET%\python.exe" -m pip install --upgrade pip

echo.
echo [OK] Локальный Python готов:
"%TARGET%\python.exe" -c "import sys; print('   ', sys.executable); print('   ', sys.version)"
echo.
echo Дальше: install.bat
pause
exit /b 0
