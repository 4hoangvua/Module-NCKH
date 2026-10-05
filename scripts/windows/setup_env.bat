@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

echo ========================================================
echo        KHỞI TẠO MÔI TRƯỜNG CHO DỰ ÁN ASR - OCR (WINDOWS)
echo ========================================================
echo.

REM 1. Kiểm tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [LỖI] Không tìm thấy Python trên máy tính. Vui lòng cài Python trước.
    pause
    exit /b
)

REM 2. Tạo môi trường ảo nếu chưa có
if not exist "venv\" (
    echo [*] Đang tạo môi trường ảo Python (venv)...
    python -m venv venv
)

REM 3. Kích hoạt môi trường ảo và cài thư viện
echo [*] Đang kích hoạt môi trường ảo và cài đặt thư viện từ requirements.txt...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================================
echo  HOÀN TẤT! Bạn có thể chạy 'run_test_asr.bat' để kiểm thử.
echo ========================================================
pause
