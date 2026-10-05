@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."
cls
echo ========================================================
echo        KIỂM THỬ TRÍCH XUẤT PHỤ ĐỀ VIDEO (OCR)
echo ========================================================
echo.

REM Ưu tiên sử dụng python trong venv nếu đã cài đặt
set PYTHON_EXE=python
if exist "venv\Scripts\python.exe" (
    set PYTHON_EXE=venv\Scripts\python.exe
)

if "%~1"=="" (
    "%PYTHON_EXE%" test_ocr.py
) else (
    "%PYTHON_EXE%" test_ocr.py "%~1" %*
)

echo.
pause
