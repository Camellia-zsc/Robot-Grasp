@echo off
setlocal
set "GRASP_PYTHON=python"
if exist "%~dp0.venv-physics\Scripts\python.exe" set "GRASP_PYTHON=%~dp0.venv-physics\Scripts\python.exe"
if defined MUJOCO_PYTHON set "GRASP_PYTHON=%MUJOCO_PYTHON%"
cd /d "%~dp0ws\src\cube_task"
"%GRASP_PYTHON%" -m cube_task assistant_ui --open-browser %*
pause
