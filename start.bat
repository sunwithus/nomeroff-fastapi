@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

call venv\Scripts\activate.bat

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