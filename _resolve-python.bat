@echo off
rem Prefer folder-local Python. Sets PYEXE (full path) or PYLAUNCH (py -3.x).
rem Usage from nomeroff-net:  call "%~dp0_resolve-python.bat"

set "PYEXE="
set "PYLAUNCH="

if exist "%~dp0python\python.exe" (
    set "PYEXE=%~dp0python\python.exe"
    goto :eof
)

if exist "%~dp0..\python\python.exe" (
    set "PYEXE=%~dp0..\python\python.exe"
    goto :eof
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" 2>nul
    if not errorlevel 1 (
        set "PYLAUNCH=py -3.12"
        goto :eof
    )
    py -3 -c "import sys" 2>nul
    if not errorlevel 1 (
        set "PYLAUNCH=py -3"
        goto :eof
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        set "PYEXE=%%i"
        goto :eof
    )
)

goto :eof
