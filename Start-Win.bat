@echo off
rem Keep this file ASCII-only: cmd.exe parses batch with the ANSI codepage
rem (GBK on Chinese Windows), so UTF-8 Chinese text breaks the parser.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and check "Add to PATH".
    pause
    exit /b 1
)

python -c "import requests" >nul 2>nul
if errorlevel 1 (
    echo [First run] Installing dependency: requests ...
    python -m pip install -r requirements.txt --disable-pip-version-check -q
)

echo Starting TRAE-llamaCPP-Bridge GUI ...
python gui_app.py
if errorlevel 1 pause
