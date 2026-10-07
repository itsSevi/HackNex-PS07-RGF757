@echo off
title HACKNEX 2026 - Vision AI Dashboard Server
echo ========================================================
echo HACKNEX 2026 - AUTONOMOUS VISION & BEHAVIOUR DASHBOARD
echo ========================================================
echo Starting FastAPI Backend and Web UI Server on port 8000...
echo.

cd /d "%~dp0\backend"
start http://127.0.0.1:8000
python main.py

pause
