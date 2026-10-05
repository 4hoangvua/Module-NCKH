# HƯỚNG DẪN THỰC NGHIỆM: MODULE ASR & OCR (Local)

- **Thành viên:** Võ Thành Công
- **Công nghệ:** Faster-Whisper + Silero-VAD (ASR) | PaddleOCR + OpenCV ROI (OCR)
- **Mục tiêu:** Chạy PoC trên máy cá nhân → đo hiệu năng → xuất JSON bàn giao cho MT (Hoàng) & TTS (Duy)

---

## CHECKLIST

| # | Công việc | Script |
|---|----------|--------|
| 1 | Cài FFmpeg + Python venv | — |
| 2 | Cài thư viện Python | `requirements.txt` |
| 3 | Chuẩn bị thư mục & data test | — |
| 4 | Tách audio + frames | `step1_extract.py` |
| 5 | Chạy ASR | `test_asr.py` |
| 6 | Chạy OCR | `test_ocr.py` |
| 7 | Đo WER | `eval_wer.py` |
| 8 | Ghi chép RAM/VRAM/tốc độ | Task Manager / `nvidia-smi` |
| 9 | Bàn giao JSON cho nhóm | — |

---

## BƯỚC 1: MÔI TRƯỜNG

### 1.1 FFmpeg (bắt buộc)
1. Tải tại: https://www.gyan.dev/ffmpeg/builds/ → chọn `ffmpeg-git-full.7z`
2. Giải nén vào `C:\ffmpeg`, thêm `C:\ffmpeg\bin` vào biến `Path`
3. Kiểm tra:
```bash
ffmpeg -version
```

### 1.2 Python Virtual Environment
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# (Nếu lỗi Execution Policy: Set-ExecutionPolicy Unrestricted -Scope Process)

# Windows CMD:
.\venv\Scripts\activate.bat
```

---

## BƯỚC 2: CÀI THƯ VIỆN

Tạo file `requirements.txt`:
```text
faster-whisper
jiwer
opencv-python
pillow
paddlepaddle
paddleocr
python-Levenshtein
psutil
```

Cài đặt:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý GPU:** Nếu máy có NVIDIA GPU + CUDA, cài `paddlepaddle-gpu` thay cho `paddlepaddle` (xem đúng phiên bản CUDA tại paddlepaddle.org.cn).

---

## BƯỚC 3: CẤU TRÚC THƯ MỤC

```text
module_asr_ocr/
├── data/
│   ├── input_video.mp4          # Video mẫu tiếng Anh ~30-60s
│   ├── ground_truth.txt         # Lời thoại chuẩn (gõ tay) để đo WER
│   └── frames/                  # Frame trích xuất (tự sinh)
├── output/                      # Kết quả (tự sinh)
├── requirements.txt
├── step1_extract.py
├── test_asr.py
├── test_ocr.py
└── eval_wer.py
```

---

## BƯỚC 4: TÁCH AUDIO & FRAMES (`step1_extract.py`)

```python
import subprocess, os

VIDEO  = "data/input_video.mp4"
AUDIO  = "output/audio_16k.wav"
FRAMES = "data/frames"

os.makedirs("output", exist_ok=True)
os.makedirs(FRAMES, exist_ok=True)

# Tách audio 16 kHz mono PCM 16-bit
print("[1/2] Tách audio...")
subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    AUDIO
], check=True)

# Trích xuất 2 frame/giây
print("[2/2] Trích xuất frames (2 fps)...")
subprocess.run([
    "ffmpeg", "-y", "-i", VIDEO,
    "-vf", "fps=2",
    os.path.join(FRAMES, "frame_%04d.jpg")
], check=True)

print("Hoàn tất tiền xử lý.")
```

---

## BƯỚC 5: THỰC NGHIỆM ASR (`test_asr.py`)

```python
import time, json
from faster_whisper import WhisperModel

AUDIO_PATH  = "output/audio_16k.wav"
OUTPUT_JSON = "output/asr_output.json"

MODEL_SIZE   = "small"       # base | small | medium
DEVICE       = "cpu"         # cpu | cuda
COMPUTE_TYPE = "int8"        # int8 (CPU) | float16 (GPU)

def run_asr():
    print(f"--- ASR: {MODEL_SIZE} | {DEVICE} | {COMPUTE_TYPE} ---")

    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"Load model: {time.time() - t0:.2f}s")

    t1 = time.time()
    segments, info = model.transcribe(
        AUDIO_PATH,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    print(f"Ngôn ngữ: {info.language} ({info.language_probability:.2f})")

    results = []
    for i, seg in enumerate(segments, 1):
        dur = round(seg.end - seg.start, 2)
        results.append({
            "id": i,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": dur,
            "source_type": "ASR",
            "source_text": seg.text.strip(),
            "confidence": round(float(seg.avg_logprob), 4)
        })
        print(f"  [{seg.start:.2f} → {seg.end:.2f}] {seg.text.strip()}")

    print(f"Suy luận: {time.time() - t1:.2f}s")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"→ Xuất {len(results)} segments → {OUTPUT_JSON}")

if __name__ == "__main__":
    run_asr()
```

---

## BƯỚC 6: THỰC NGHIỆM OCR (`test_ocr.py`)

```python
import cv2, json, time
from paddleocr import PaddleOCR
from Levenshtein import ratio

VIDEO_PATH  = "data/input_video.mp4"
OUTPUT_JSON = "output/ocr_output.json"

def run_ocr():
    print("--- OCR: PaddleOCR + ROI 20% ---")
    t0 = time.time()

    ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / 2))  # 2 frame/giây

    raw = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ts = round(idx / fps, 2)
            h, w = frame.shape[:2]
            roi = frame[int(h * 0.8):h, 0:w]  # 20% chân màn hình
            result = ocr.ocr(roi, cls=False)
            if result and result[0]:
                lines = [l[1][0] for l in result[0] if l[1][1] > 0.6]
                if lines:
                    raw.append((ts, " ".join(lines).strip()))
        idx += 1
    cap.release()

    # Khử trùng lặp bằng Levenshtein ratio >= 0.8
    deduped = []
    if raw:
        start, text, end = raw[0][0], raw[0][1], raw[0][0]
        sid = 1
        for i in range(1, len(raw)):
            ts, t = raw[i]
            if ratio(text, t) >= 0.8:
                end = ts
            else:
                deduped.append({
                    "id": sid, "start": start,
                    "end": round(end + 0.5, 2),
                    "duration": round(end + 0.5 - start, 2),
                    "source_type": "OCR",
                    "source_text": text, "confidence": 0.85
                })
                sid += 1
                start, text, end = ts, t, ts
        deduped.append({
            "id": sid, "start": start,
            "end": round(end + 0.5, 2),
            "duration": round(end + 0.5 - start, 2),
            "source_type": "OCR",
            "source_text": text, "confidence": 0.85
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"Tổng: {time.time() - t0:.2f}s | {len(deduped)} segments → {OUTPUT_JSON}")

if __name__ == "__main__":
    run_ocr()
```

---

## BƯỚC 7: ĐO WORD ERROR RATE (`eval_wer.py`)

```python
import json, jiwer

GT_PATH  = "data/ground_truth.txt"
ASR_PATH = "output/asr_output.json"

def calc_wer():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        ref = f.read().strip()
    with open(ASR_PATH, "r", encoding="utf-8") as f:
        hyp = " ".join(s["source_text"] for s in json.load(f)).strip()

    tx = jiwer.Compose([
        jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(), jiwer.Strip()
    ])
    ref_c, hyp_c = tx(ref), tx(hyp)

    print("=== ĐO LƯỜNG ASR ===")
    print(f"WER: {jiwer.wer(ref_c, hyp_c) * 100:.2f}%")
    print(f"MER: {jiwer.mer(ref_c, hyp_c) * 100:.2f}%")
    print(f"WIL: {jiwer.wil(ref_c, hyp_c) * 100:.2f}%")

if __name__ == "__main__":
    calc_wer()
```

---

## BƯỚC 8: BẢNG GHI KẾT QUẢ

### Bảng 1: Faster-Whisper (video mẫu ~60s)

| Model | Device | Compute | Load (s) | Infer (s) | RAM/VRAM | WER (%) |
|-------|--------|---------|----------|-----------|----------|---------|
| base  | CPU    | int8    |          |           |          |         |
| small | CPU    | int8    |          |           |          |         |
| medium| CPU    | int8    |          |           |          |         |

### Bảng 2: OCR — so sánh có/không ROI

| Phương pháp    | Vùng quét           | Thời gian (s) | Nhận xét             |
|----------------|---------------------|---------------|----------------------|
| Không ROI      | Toàn bộ 1920×1080   |               | Nhiều nhiễu nền      |
| **Có ROI 20%** | 1920×216 (chân)     |               | Nhanh hơn, ít nhiễu  |

---

## BƯỚC 9: BÀN GIAO JSON (DATA CONTRACT)

Gửi `output/asr_output.json` và/hoặc `output/ocr_output.json` cho Hoàng (MT). Chuẩn format:

```json
[
  {
    "id": 1,
    "start": 0.0,
    "end": 3.42,
    "duration": 3.42,
    "source_type": "ASR",
    "source_text": "In this video, we will learn about local AI architecture.",
    "confidence": -0.28
  }
]
```

| Trường | Ý nghĩa |
|--------|---------|
| `id` | Số thứ tự segment |
| `start` / `end` | Thời điểm bắt đầu / kết thúc (giây) |
| `duration` | Độ dài segment (giây) |
| `source_type` | `"ASR"` hoặc `"OCR"` |
| `source_text` | Văn bản nhận dạng được |
| `confidence` | Điểm tin cậy (avg_logprob cho ASR) |