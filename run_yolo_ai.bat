@echo off
title HACKNEX 2026 - YOLO Pose + BoT-SORT Tracking
echo ========================================================
echo HACKNEX 2026 - YOLO POSE + BOTSORT REID TRACKING
echo ========================================================
echo Starting AI Tracking script...
echo.

cd /d "%~dp0"
python main_AI_FASTAPI.py

pause
