@echo off
setlocal
cd /d "%~dp0..\.."

echo ========================================================
echo        KHOI TAO MOI TRUONG ASR - OCR (WINDOWS)
echo ========================================================
echo.

REM 1. Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay Python tren he thong.
    echo Vui long cai dat Python va tick chon Add Python to PATH.
    echo.
    pause
    exit /b 1
)

REM 2. Tao moi truong ao neu chua co
if not exist "venv" (
    echo [*] Dang tao moi truong ao Python venv...
    python -m venv venv
)

REM 3. Kich hoat moi truong ao va cai thu vien
echo [*] Dang cai dat thu vien tu requirements.txt...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================================
echo  HOAN TAT! Ban co the chay run_test_asr.bat de kiem thu.
echo ========================================================
pause
