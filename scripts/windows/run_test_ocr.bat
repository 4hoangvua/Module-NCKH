@echo off
setlocal
cd /d "%~dp0..\.."
cls

echo ========================================================
echo        KIEM THU TRICH XUAT PHU DE VIDEO (OCR)
echo ========================================================
echo.

REM Uu tien su dung python trong venv neu da cai dat
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
