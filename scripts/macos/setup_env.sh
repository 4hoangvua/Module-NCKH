#!/bin/bash
# ========================================================
#   KHỞI TẠO MÔI TRƯỜNG CHO DỰ ÁN ASR - OCR (macOS / Linux)
# ========================================================

set -e

# Chuyển về thư mục gốc của dự án
cd "$(dirname "$0")/../.."

echo "========================================================"
echo "       KHỞI TẠO MÔI TRƯỜNG CHO DỰ ÁN ASR - OCR (macOS)"
echo "========================================================"
echo ""

# 1. Kiểm tra Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "[LỖI] Không tìm thấy Python 3. Vui lòng cài đặt Python (ví dụ: brew install python trên Mac)."
    exit 1
fi

echo "[*] Đang dùng: $($PYTHON_CMD --version)"

# 2. Tạo môi trường ảo
if [ ! -d "venv" ]; then
    echo "[*] Đang tạo môi trường ảo Python (venv)..."
    $PYTHON_CMD -m venv venv
fi

# 3. Kích hoạt môi trường ảo
source venv/bin/activate
pip install --upgrade pip

# 4. Cài đặt thư viện theo hệ điều hành
OS_NAME="$(uname -s)"
echo "[*] Đang phát hiện hệ điều hành: $OS_NAME"

if [ "$OS_NAME" = "Darwin" ]; then
    echo "[*] Đang cài đặt thư viện cho macOS (Apple Silicon / Intel)..."
    pip install faster-whisper torch rapidocr onnxruntime opencv-python Pillow numpy psutil jiwer python-Levenshtein
else
    echo "[*] Đang cài đặt thư viện cho Linux..."
    pip install faster-whisper torch rapidocr onnxruntime opencv-python Pillow numpy psutil jiwer python-Levenshtein
fi

echo ""
echo "========================================================"
echo "  HOÀN TẤT! Bạn có thể chạy scripts/macos/run_test_asr.sh"
echo "========================================================"
