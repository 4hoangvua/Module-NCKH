import os
import sys
import site
import time
import json
import re

# Tự động nạp toàn bộ thư mục DLL của NVIDIA trên Windows (cuBLAS, cuDNN)
if sys.platform == "win32":
    for site_pkg in site.getsitepackages():
        nvidia_base = os.path.join(site_pkg, "nvidia")
        if os.path.exists(nvidia_base):
            for sub in os.listdir(nvidia_base):
                bin_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.exists(bin_dir):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(bin_dir)
                        except Exception:
                            pass

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font/mã hóa
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from faster_whisper import WhisperModel
from profiler import HardwareProfiler

try:
    from opencc import OpenCC
    t2s_converter = OpenCC("t2s")
except Exception:
    t2s_converter = None


def to_simplified_chinese(text: str) -> str:
    """Chuyển đổi văn bản tiếng Trung Phồn thể sang Giản thể chuẩn."""
    if not text:
        return ""
    if t2s_converter is not None:
        return t2s_converter.convert(text)
    return text


# ============================================================
# 1. ĐO LƯỜNG CHẤT LƯỢNG KHOA HỌC (WER / CER BENCHMARK)
# ============================================================
def calculate_levenshtein(ref: list, hyp: list) -> int:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(ref)][len(hyp)]


def compute_accuracy_metrics(hypothesis_cues: list, ground_truth_file: str) -> dict | None:
    """So sánh kết quả nhận dạng với Ground Truth và tính WER, CER."""
    if not os.path.exists(ground_truth_file):
        return None

    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        ref_text = " ".join([to_simplified_chinese(item.get("text", "")) for item in gt_data if "text" in item])
        hyp_text = " ".join([to_simplified_chinese(item.get("text", "")) for item in hypothesis_cues if "text" in item])

        normalize = lambda s: re.sub(r"[^\w\u4e00-\u9fff]", "", to_simplified_chinese(s).lower()).strip()
        ref_norm = normalize(ref_text)
        hyp_norm = normalize(hyp_text)
        ref_words = list(ref_norm) if any('\u4e00' <= c <= '\u9fff' for c in ref_norm) else ref_norm.split()
        hyp_words = list(hyp_norm) if any('\u4e00' <= c <= '\u9fff' for c in hyp_norm) else hyp_norm.split()

        if not ref_words:
            return None

        try:
            import jiwer
            wer_score = jiwer.wer(ref_norm, hyp_norm)
            cer_score = jiwer.cer(ref_norm, hyp_norm)
        except Exception:
            wer_score = calculate_levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)
            cer_score = calculate_levenshtein(list(ref_norm), list(hyp_norm)) / max(len(ref_norm), 1)

        accuracy = max(0.0, (1.0 - cer_score if any('\u4e00' <= c <= '\u9fff' for c in ref_norm) else 1.0 - wer_score) * 100)
        return {
            "ground_truth": ground_truth_file,
            "total_words_ref": len(ref_words),
            "total_words_hyp": len(hyp_words),
            "wer": round(wer_score, 4),
            "cer": round(cer_score, 4),
            "accuracy": round(accuracy, 2),
        }
    except Exception as e:
        print(f"[!] Lỗi khi đánh giá độ chính xác: {e}")
        return None


# ============================================================
# 2. XUẤT FILE PHỤ ĐỀ SRT
# ============================================================
def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def save_to_srt(cues: list, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for c in cues:
            f.write(f"{c['id']}\n")
            f.write(f"{format_srt_time(c['start'])} --> {format_srt_time(c['end'])}\n")
            f.write(f"{c['text']}\n\n")


# ============================================================
# 3. CHƯƠNG TRÌNH CHÍNH (STANDALONE ASR PIPELINE)
# ============================================================
def main():
    if len(sys.argv) > 1:
        media_path = sys.argv[1]
    else:
        candidates = ["data/input_video.mp4", "data/input_audio.wav", "data/sample.mp3"]
        media_path = next((c for c in candidates if os.path.exists(c)), "data/input_video.mp4")

    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    language_req = sys.argv[3] if len(sys.argv) > 3 else None

    # TỰ ĐỘNG CHỌN GPU NVIDIA VÀ KIỂU TÍNH TOÁN TỐI ƯU
    device = "cpu"
    compute_type = "int8"
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            device = "cuda"
            if "float16" in cuda_types:
                compute_type = "float16"
            elif "int8_float32" in cuda_types:
                compute_type = "int8_float32"  # Dành riêng cho GTX 1060/1070/1080
            elif "float32" in cuda_types:
                compute_type = "float32"
            else:
                compute_type = "int8"
    except Exception:
        device = "cpu"
        compute_type = "int8"

    output_json = "output/asr_output.json"
    output_srt = "output/subtitles-source.srt"
    ground_truth_file = "output/subtitles-source.json"

    print("=" * 65)
    print(f"[*] FILE          : {media_path}")
    print(f"[*] MODEL         : Faster-Whisper {model_size.upper()} ({device.upper()} | {compute_type})")
    print(f"[*] CHIẾN LƯỢC    : CPU Parallel Threads + VAD Filter + Fast Beam Search")
    print("=" * 65)

    if not os.path.exists(media_path):
        print(f"\n[!] Không tìm thấy file đầu vào: {media_path}")
        print("    Vui lòng đặt video/audio vào thư mục data/ rồi thử lại.\n")
        return

    os.makedirs("output", exist_ok=True)

    # Khởi động bộ đo lường tài nguyên phần cứng
    profiler = HardwareProfiler()
    profiler.start()

    # 1. Tải Model (Tối ưu hóa số luồng CPU vật lý)
    print("\n[1/3] Tải model Faster-Whisper...")
    t0 = time.time()
    import psutil
    physical_cores = psutil.cpu_count(logical=False) or 4
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=physical_cores,
        num_workers=1,
    )
    print(f"      Xong ({time.time() - t0:.1f}s)")

    # 2. Nhận dạng âm thanh (beam_size=2 cho tốc độ tối đa trên CPU)
    print("[2/3] Nhận dạng âm thanh...")
    initial_prompt = "以下是普通话简体中文内容。" if language_req in [None, "zh", "chinese"] else None
    beam_size = 5 if device == "cuda" else 2
    segments_iter, info = model.transcribe(
        media_path,
        language=language_req,
        beam_size=beam_size,
        word_timestamps=False,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        initial_prompt=initial_prompt,
    )

    lang = info.language
    print(f"      Ngôn ngữ phát hiện: {lang.upper()} ({info.language_probability*100:.1f}%)")
    print(f"      Thời lượng file   : {info.duration:.2f}s")

    # 3. Trích xuất Cues & Chuyển đổi Giản thể chuẩn
    print("[3/3] Trích xuất danh sách câu...")
    cues = []
    for i, seg in enumerate(segments_iter, start=1):
        raw_text = seg.text.strip()
        if not raw_text:
            continue
        cleaned_text = to_simplified_chinese(raw_text)
        # Bỏ dấu câu dư thừa ở đầu/cuối
        cleaned_text = re.sub(r"^[,\.?!;:—\-_ ]+|[,\.?!;:—\-_ ]+$", "", cleaned_text).strip()
        if not cleaned_text:
            continue
        cues.append({
            "id": i,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": cleaned_text,
            "confidence": round(getattr(seg, "avg_logprob", 0.0), 4),
        })

    # Dừng profiler & lấy số liệu
    hw_stats = profiler.stop()

    # In kết quả dạng bảng
    print("-" * 65)
    print(f"{'#':<4} {'TIMESTAMPS':<22} | {'VĂN BẢN'}")
    print("-" * 65)
    for c in cues:
        ts = f"[{c['start']:6.2f}s → {c['end']:6.2f}s]"
        print(f"{c['id']:<4} {ts:<22} | {c['text']}")
    print("-" * 65)

    last = cues[-1]["end"] if cues else 0
    print(f"\n[✓] Tổng số câu: {len(cues)} | Độ dài: {last:.2f}s / {info.duration:.2f}s")

    # 4. Lưu kết quả
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"cues": cues, "hardware_benchmark": hw_stats}, f, ensure_ascii=False, indent=2)
    save_to_srt(cues, output_srt)
    print(f"[✓] Đã lưu JSON: {output_json}")
    print(f"[✓] Đã lưu SRT : {output_srt}")

    # 5. Đo lường & In bảng báo cáo NCKH toàn diện
    acc_metrics = compute_accuracy_metrics(cues, ground_truth_file)
    profiler.print_academic_report(
        task_name="Nhận dạng giọng nói (ASR)",
        model_name=f"Faster-Whisper {model_size.upper()} ({device.upper()} - {compute_type})",
        media_duration_sec=info.duration,
        accuracy_metrics=acc_metrics,
    )


if __name__ == "__main__":
    main()
