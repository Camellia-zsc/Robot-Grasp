@echo off
cd /d "%~dp0ws\src\cube_task"
python -m cube_task assistant_ui --open-browser %*
pause
