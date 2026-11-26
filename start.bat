@echo off
rem DreamScape: build and start every container, then open the dream page when it answers.
cd /d "%~dp0"
set PORT=80
if exist .env for /f "usebackq tokens=1,* delims==" %%a in (".env") do if /i "%%a"=="WEB_PORT" set PORT=%%b

echo Starting DreamScape. The first build takes about 10 minutes.
docker compose up -d --build
if errorlevel 1 (
  echo.
  echo Docker could not start DreamScape. Check that Docker Desktop is running, then try again.
  pause
  exit /b 1
)

echo.
echo Waiting for the dream page on port %PORT% ...
:wait
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:%PORT%/healthz | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  timeout /t 3 /nobreak >nul
  goto wait
)
echo The models keep downloading in the background; the dots at the top of the page show their progress.
start "" http://localhost:%PORT%/
