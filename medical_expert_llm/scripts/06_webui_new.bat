@echo off
REM New UI (FastAPI + SSE) - replaces the Gradio demo interface.
REM Serves http://127.0.0.1:8080 with the redesigned medical chat frontend.
cd /d "%~dp0\.."
set PYEXE=%~dp0..\venv\Scripts\python.exe
if not exist "%PYEXE%" (
    echo [ERROR] venv\Scripts\python.exe not found. See README.md setup first.
    pause
    exit /b 1
)
"%PYEXE%" app\server.py --port 8080
pause
