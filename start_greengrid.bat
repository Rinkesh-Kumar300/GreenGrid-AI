@echo off
setlocal

:: ─────────────────────────────────────────────
:: GreenGrid AI — Windows Startup Script
:: ─────────────────────────────────────────────
:: Starts Ollama (if not already running),
:: FastAPI backend on port 8001,
:: and opens the frontend in the default browser.
:: Each service runs in its own labelled window.
:: ─────────────────────────────────────────────

set "ROOT=%~dp0"
:: Strip trailing backslash
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

echo.
echo  ============================================
echo   GreenGrid AI — Starting Services
echo  ============================================
echo.

:: ── 1. Ollama ────────────────────────────────
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if %ERRORLEVEL% EQU 0 (
    echo [1/3] Ollama is already running — skipping.
) else (
    echo [1/3] Starting Ollama ...
    start "Ollama" cmd /k "ollama serve"
    :: Give Ollama a moment to bind its port before the backend connects
    timeout /t 3 /nobreak >nul
)

:: ── 2. FastAPI backend on port 8001 ──────────
echo [2/3] Starting FastAPI backend on port 8001 ...
start "GreenGrid API (port 8001)" cmd /k ^
    "cd /d "%ROOT%" && uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload"

:: Wait for the API to be ready before opening the browser
echo       Waiting for API to become ready ...
:wait_api
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command ^
    "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8001/health' -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% NEQ 0 goto wait_api
echo       API is up.

:: ── 3. Frontend ───────────────────────────────
echo [3/3] Opening frontend in default browser ...
start "" "%ROOT%\frontend\index.html"

echo.
echo  ============================================
echo   All services started.
echo.
echo   API:       http://127.0.0.1:8001
echo   Docs:      http://127.0.0.1:8001/docs
echo   Frontend:  frontend\index.html
echo  ============================================
echo.
echo  Close this window at any time.
echo  To stop services close their individual windows.
echo.
pause
endlocal
