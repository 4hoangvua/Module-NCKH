@echo off
setlocal
cd /d "%~dp0..\.."
cls

echo ========================================================
echo           KIEM THU NHAN DANG GIONG NOI (ASR)
echo ========================================================
echo.

REM Uu tien su dung python trong venv neu da cai dat
set PYTHON_EXE=python
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
)

if "%~1"=="" (
    "%PYTHON_EXE%" test_asr.py
) else (
    "%PYTHON_EXE%" test_asr.py "%~1" "%~2" "%~3"
)

echo.
pause
