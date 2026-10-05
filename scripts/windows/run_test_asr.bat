@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
cls
echo ========================================================
echo           KIỂM THỬ NHẬN DẠNG GIỌNG NÓI (ASR)
echo ========================================================
echo.

REM Ưu tiên sử dụng python trong venv nếu đã cài đặt
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
