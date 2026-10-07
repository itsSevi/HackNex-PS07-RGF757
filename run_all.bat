@echo off
title HACKNEX 2026 - Launcher
echo ========================================================
echo HACKNEX 2026 - PS07 FULL SYSTEM LAUNCHER
echo ========================================================
echo 1. Launching Dashboard Server at http://127.0.0.1:8000
echo 2. Launching YOLO Pose + BoT-SORT AI Pipeline
echo.

start "FastAPI Dashboard" cmd /k "cd /d ""%~dp0\backend"" && python main.py"
timeout /t 2 >nul
start http://127.0.0.1:8000
timeout /t 1 >nul
start "YOLO AI Inference" cmd /k "cd /d ""%~dp0"" && python main_AI_FASTAPI.py"

echo All systems launched!
