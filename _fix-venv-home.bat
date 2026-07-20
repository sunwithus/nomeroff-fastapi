@echo off
rem Rewrite venv\pyvenv.cfg to point at local python\ (portable after copy).
rem NOTE: do not use PowerShell $HOME — it is read-only.
setlocal EnableExtensions
cd /d "%~dp0"

set "CFG=%~dp0venv\pyvenv.cfg"
set "PYHOME="
if exist "%~dp0python\python.exe" set "PYHOME=%~dp0python"
if not defined PYHOME if exist "%~dp0..\python\python.exe" set "PYHOME=%~dp0..\python"
if not defined PYHOME goto :eof
if not exist "%CFG%" goto :eof

if "%PYHOME:~-1%"=="\" set "PYHOME=%PYHOME:~0,-1%"

powershell -NoProfile -Command ^
  "$cfg = Join-Path '%~dp0' 'venv\pyvenv.cfg';" ^
  "$pyHome = (Resolve-Path -LiteralPath '%PYHOME%').Path;" ^
  "$pyExe = Join-Path $pyHome 'python.exe';" ^
  "$venvPath = (Resolve-Path -LiteralPath (Join-Path '%~dp0' 'venv')).Path;" ^
  "$lines = Get-Content -LiteralPath $cfg;" ^
  "$out = foreach ($l in $lines) {" ^
  "  if ($l -match '^home\s*=') { 'home = ' + $pyHome }" ^
  "  elseif ($l -match '^executable\s*=') { 'executable = ' + $pyExe }" ^
  "  elseif ($l -match '^command\s*=') { 'command = ' + $pyExe + ' -m venv ' + $venvPath }" ^
  "  else { $l }" ^
  "};" ^
  "Set-Content -LiteralPath $cfg -Value $out -Encoding ASCII"

goto :eof
