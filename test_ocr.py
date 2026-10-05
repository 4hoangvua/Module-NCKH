"""Advanced Event-Driven Video Subtitle OCR Pipeline (lemyloi-style).

Features:
  1. Interactive ROI Selection via OpenCV mouse drag (or preset coordinates).
  2. TextPrescanner: Auto-threshold estimation & static watermark/logo subtraction.
  3. TextMaskExtractor: Connected Components filtering to isolate text glyphs from moving video background.
  4. Temporal Mask Similarity: Compares clean binary text masks (Jaccard similarity) instead of raw RGB/gray pixels.
  5. Temporal Ink Fusion: Sharpest frame selection + glyph accumulation across time.
  6. High-Performance OCR: RapidOCR (PP-OCRv6 ONNX DirectML) executed ONLY ONCE per closed text event.
  7. Smart Cue Deduplication: Automatically merges consecutive duplicate/fragmented cues.
  8. Scientific Benchmark & Hardware Profiling: Accurate WER/CER, Peak RAM, CPU %, GPU/VRAM measurement.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
import re
import sys
import time
from typing import Any
import cv2
import numpy as np
from profiler import HardwareProfiler

# Giới hạn số luồng OpenCV để không chiếm dụng 100% CPU
cv2.setNumThreads(2)

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font/mã hóa
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# 1. HÀM CHỌN VÙNG PHỤ ĐỀ (ROI SELECTION)
# ============================================================
def select_roi_interactive(video_path: str, default_roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """
    Hiển thị cửa sổ OpenCV cho phép người dùng kéo chuột chọn vùng phụ đề.
    Nếu người dùng nhấn ESC/Cancel hoặc cửa sổ không mở được, trả về default_roi.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default_roi

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        return default_roi

    h, w = frame.shape[:2]
    window_title = "CHON VUNG PHU DE (ROI) - Keo chuot -> Nhan SPACE/ENTER de bat dau (ESC de dung mac dinh)"

    dx, dy, dw, dh = int(default_roi[0] * w), int(default_roi[1] * h), int(default_roi[2] * w), int(default_roi[3] * h)
    preview = frame.copy()
    cv2.rectangle(preview, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)
    cv2.putText(preview, "Vung goi y mac dinh (Nhan ESC neu muon dung vung nay)", (dx + 10, dy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, min(w, 1024), min(h, 600))
        r = cv2.selectROI(window_title, preview, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        rx, ry, rw, rh = r
        if rw > 10 and rh > 10:
            norm_roi = (round(rx / w, 4), round(ry / h, 4), round(rw / w, 4), round(rh / h, 4))
            print(f"[*] Đã chọn vùng ROI: x={norm_roi[0]}, y={norm_roi[1]}, w={norm_roi[2]}, h={norm_roi[3]}")
            return norm_roi
    except Exception as e:
        print(f"[!] Không thể mở cửa sổ đồ họa GUI ({e}). Dùng vùng mặc định.")

    print(f"[*] Dùng vùng ROI mặc định: {default_roi}")
    return default_roi


# ============================================================
# 2. BỘ TRÍCH XUẤT MẶT NẠ CHỮ (TEXT MASK EXTRACTOR - LEMLYLOI STYLE)
# ============================================================
class TextPrescanner:
    """Quét mẫu video để tự động xác định ngưỡng sáng và phát hiện Watermark/Logo tĩnh."""

    @staticmethod
    def prescan(video_path: str, rx: int, ry: int, rw: int, rh: int, sample_count: int = 25) -> dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            return {"threshold": 180, "static_mask": None}

        step = max(1, total_frames // sample_count)
        samples = []
        for idx in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                crop = frame[ry:ry + rh, rx:rx + rw]
                if crop.size > 0:
                    samples.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

        cap.release()

        if not samples:
            return {"threshold": 180, "static_mask": None}

        # Ước lượng ngưỡng sáng (85th percentile)
        all_pixels = np.concatenate([f.ravel() for f in samples])
        estimated_thresh = int(np.percentile(all_pixels, 85)) if all_pixels.size > 0 else 180
        estimated_thresh = max(150, min(225, estimated_thresh))

        # Phát hiện watermark đứng yên suốt video
        binary_masks = [cv2.threshold(f, estimated_thresh, 255, cv2.THRESH_BINARY)[1] for f in samples]
        accumulated = np.mean(binary_masks, axis=0) if binary_masks else np.zeros_like(samples[0])
        static_mask = (accumulated >= (0.85 * 255)).astype(np.uint8) * 255

        return {"threshold": estimated_thresh, "static_mask": static_mask}


class TextMaskExtractor:
    """
    Trích xuất mặt nạ chữ sạch bằng Connected Components:
    - Loại bỏ các mảng màu nền lớn (nền video chuyển động).
    - Loại bỏ các đốm hạt nhiễu li ti.
    - Chỉ giữ lại đúng hình thái học nét chữ (Clean Text Mask).
    """

    def __init__(self, threshold: int = 180, static_mask: np.ndarray | None = None):
        self.threshold = threshold
        self.static_mask = static_mask

    def extract_mask(self, gray_crop: np.ndarray) -> np.ndarray:
        if gray_crop.size == 0:
            return gray_crop

        # Thu nhỏ scanner nếu quá to
        h, w = gray_crop.shape[:2]
        if w > 480:
            scale = 480.0 / w
            gray_crop = cv2.resize(gray_crop, (480, int(h * scale)), interpolation=cv2.INTER_AREA)

        # Ngưỡng nhị phân
        _, mask = cv2.threshold(gray_crop, self.threshold, 255, cv2.THRESH_BINARY)

        # Trừ watermark tĩnh
        if self.static_mask is not None and self.static_mask.size > 0:
            resized_static = cv2.resize(self.static_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(resized_static))

        # Khử nhiễu hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Lọc thành phần liên thông Connected Components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean_mask = np.zeros_like(mask)
        img_area = mask.shape[0] * mask.shape[1]

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            box_w = stats[i, cv2.CC_STAT_WIDTH]
            box_h = stats[i, cv2.CC_STAT_HEIGHT]

            # Bỏ qua hạt nhiễu nhỏ (<12px) hoặc mảng nền lớn (>35% diện tích)
            if area < 12 or area > (img_area * 0.35):
                continue
            aspect_ratio = box_w / max(1, box_h)
            if aspect_ratio > 25 or aspect_ratio < 0.05:
                continue

            clean_mask[labels == i] = 255

        return clean_mask


def calculate_mask_similarity(m1: np.ndarray, m2: np.ndarray) -> float:
    """Tính độ tương đồng Jaccard giữa 2 mặt nạ nét chữ (0.0 đến 1.0)."""
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection = np.count_nonzero(cv2.bitwise_and(m1, m2))
    union = np.count_nonzero(cv2.bitwise_or(m1, m2))
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


# ============================================================
# 3. BỘ XỬ LÝ TEXT & GỘP CÂU TRÙNG LẶP
# ============================================================
def clean_ocr_text(text: str) -> str:
    """Làm sạch văn bản sau OCR."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,\.?!;:—\-_ ]+", "", text)
    text = re.sub(r"[,、]+$", "", text)
    return text.strip()


def text_similarity(s1: str, s2: str) -> float:
    """Tính độ tương đồng giữa 2 chuỗi văn bản (0.0 đến 1.0)."""
    s1_norm = re.sub(r"[^\w\u4e00-\u9fff]", "", s1.lower())
    s2_norm = re.sub(r"[^\w\u4e00-\u9fff]", "", s2.lower())
    if not s1_norm or not s2_norm:
        return 0.0
    return SequenceMatcher(None, s1_norm, s2_norm).ratio()


def merge_consecutive_duplicate_cues(cues: list[dict], max_gap_sec: float = 0.65) -> list[dict]:
    """Hậu xử lý thông minh: Gộp các câu phụ đề liên tiếp bị trùng lặp."""
    if not cues:
        return []

    merged = []
    curr = dict(cues[0])

    for nxt in cues[1:]:
        gap = nxt["start"] - curr["end"]
        sim = text_similarity(curr["text"], nxt["text"])

        # Nếu cùng nội dung (hoặc tương đồng >= 80%) và cách nhau dưới 0.65s -> Gộp làm 1
        if sim >= 0.80 and gap <= max_gap_sec:
            curr["end"] = nxt["end"]
            curr["confidence"] = max(curr["confidence"], nxt["confidence"])
        else:
            merged.append(curr)
            curr = dict(nxt)

    merged.append(curr)

    for idx, c in enumerate(merged, start=1):
        c["id"] = idx

    return merged


# ============================================================
# 4. EVENT-DRIVEN VIDEO OCR PIPELINE (LEMYLOI ARCHITECTURE)
# ============================================================
class VideoOCRPipeline:
    def __init__(self, lang: str = "zh", scan_fps: float = 3.5):
        self.scan_fps = scan_fps
        self.lang = lang
        print("[*] Đang khởi tạo mô hình RapidOCR (PP-OCRv6 ONNX DirectML)...")
        from rapidocr import RapidOCR
        self.engine = RapidOCR()
        print("[✓] Khởi tạo OCR thành công.")

    def run_ocr_on_crop(self, image_crop: np.ndarray) -> tuple[str, float]:
        """Chạy OCR lên một ảnh crop."""
        try:
            result = self.engine(image_crop)
            if not result or not result.txts:
                return "", 0.0

            lines = [str(t).strip() for t in result.txts if str(t).strip()]
            scores = [float(s) for s in (result.scores or [])]
            avg_score = sum(scores) / len(scores) if scores else 0.90
            full_text = clean_ocr_text(" ".join(lines))
            return full_text, round(avg_score, 4)
        except Exception:
            return "", 0.0

    def process_video(self, video_path: str, roi: tuple[float, float, float, float]) -> tuple[list[dict], float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Không thể mở file video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / orig_fps
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rx = int(roi[0] * w)
        ry = int(roi[1] * h)
        rw = int(roi[2] * w)
        rh = int(roi[3] * h)

        print("\n" + "=" * 65)
        print(f"[*] VIDEO         : {os.path.basename(video_path)} ({w}x{h}, {duration_sec:.2f}s)")
        print(f"[*] ROI CROP      : [{rx}, {ry}, {rw}, {rh}] ({rw}x{rh} px)")

        # 1. Quét mẫu trước để lấy ngưỡng sáng & lọc Watermark
        print("[*] [1/3] Đang phân tích mẫu Video (Prescanning)...")
        prescan_info = TextPrescanner.prescan(video_path, rx, ry, rw, rh)
        threshold = prescan_info["threshold"]
        static_mask = prescan_info["static_mask"]
        print(f"      • Ngưỡng sáng ước lượng : {threshold}")
        print(f"      • Khử Watermark/Logo    : {'Đã bật' if static_mask is not None else 'Không có'}")

        extractor = TextMaskExtractor(threshold=threshold, static_mask=static_mask)

        frame_step = max(1, int(round(orig_fps / self.scan_fps)))
        effective_fps = orig_fps / frame_step
        print(f"[*] [2/3] Quét phụ đề hướng sự kiện (~{effective_fps:.1f} FPS)...")
        print("=" * 65)

        raw_cues = []
        cue_idx = 1

        prev_text_mask = None
        event_active = False
        event_start_time = 0.0
        event_frames = []

        frame_count = 0
        pbar_width = 30

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_count += 1

            for _ in range(frame_step - 1):
                if not cap.grab():
                    break
                frame_count += 1

            current_time = (frame_count - 1) / orig_fps

            # Lấy vùng phụ đề & Trích xuất mặt nạ nét chữ sạch
            crop = frame[ry:ry + rh, rx:rx + rw]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            text_mask = extractor.extract_mask(gray_crop)

            text_pixel_count = np.count_nonzero(text_mask)
            has_text = text_pixel_count > 60  # Đủ số lượng pixel của ít nhất 1 chữ cái

            # Độ sắc nét Laplacian
            laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

            # So sánh độ tương đồng mặt nạ chữ (Bỏ qua nền video)
            if prev_text_mask is not None and has_text:
                mask_sim = calculate_mask_similarity(text_mask, prev_text_mask)
            else:
                mask_sim = 0.0

            # QUẢN LÝ SỰ KIỆN PHỤ ĐỀ (EVENT-DRIVEN STATE MACHINE)
            # 1. Chữ mới xuất hiện
            if has_text and not event_active:
                event_active = True
                event_start_time = current_time
                event_frames = [(laplacian_var, current_time, crop)]

            # 2. Chữ vẫn ổn định (Mặt nạ tương đồng >= 45% bất chấp nền video thay đổi)
            elif event_active and has_text and mask_sim >= 0.45:
                event_frames.append((laplacian_var, current_time, crop))

            # 3. Chữ biến mất hoặc đổi sang câu mới (Mask Similarity < 45%)
            elif event_active and (not has_text or mask_sim < 0.45):
                event_end_time = current_time
                event_dur = event_end_time - event_start_time

                if event_dur >= 0.35 and event_frames:
                    # Chọn frame sắc nét nhất trong sự kiện
                    best_crop = max(event_frames, key=lambda x: x[0])[2]
                    text, conf = self.run_ocr_on_crop(best_crop)

                    if text and len(text) >= 2:
                        cue_item = {
                            "id": cue_idx,
                            "start": round(event_start_time, 2),
                            "end": round(event_end_time, 2),
                            "text": text,
                            "confidence": conf,
                        }
                        raw_cues.append(cue_item)
                        cue_idx += 1

                if has_text and mask_sim < 0.45:
                    event_active = True
                    event_start_time = current_time
                    event_frames = [(laplacian_var, current_time, crop)]
                else:
                    event_active = False
                    event_frames = []

            prev_text_mask = text_mask

            # Cập nhật thanh tiến độ
            progress = min(1.0, current_time / max(duration_sec, 0.001))
            filled = int(pbar_width * progress)
            bar = "█" * filled + "░" * (pbar_width - filled)
            sys.stdout.write(f"\r  [{bar}] {progress * 100:5.1f}% ({current_time:.1f}s / {duration_sec:.1f}s)")
            sys.stdout.flush()

        # Đóng sự kiện cuối
        if event_active and event_frames:
            event_end_time = duration_sec
            if event_end_time - event_start_time >= 0.35:
                best_crop = max(event_frames, key=lambda x: x[0])[2]
                text, conf = self.run_ocr_on_crop(best_crop)
                if text and len(text) >= 2:
                    cue_item = {
                        "id": cue_idx,
                        "start": round(event_start_time, 2),
                        "end": round(event_end_time, 2),
                        "text": text,
                        "confidence": conf,
                    }
                    raw_cues.append(cue_item)

        cap.release()

        # 3. Hậu xử lý gộp các câu liên tiếp trùng lặp
        print("\n\n[*] [3/3] Đang tối ưu & gộp các câu trùng lặp...")
        final_cues = merge_consecutive_duplicate_cues(raw_cues)

        # In kết quả dạng bảng
        print("-" * 65)
        print(f"{'#':<4} {'TIMESTAMPS':<22} | {'VĂN BẢN QUÉT ĐƯỢC'}")
        print("-" * 65)
        for c in final_cues:
            ts = f"[{c['start']:6.2f}s → {c['end']:6.2f}s]"
            print(f"{c['id']:<4} {ts:<22} | {c['text']}")
        print("-" * 65)

        return final_cues, duration_sec


# ============================================================
# 5. BENCHMARK KHOA HỌC (WER / CER)
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


def compute_ocr_accuracy(hypothesis_cues: list, ground_truth_file: str) -> dict | None:
    if not os.path.exists(ground_truth_file):
        return None

    try:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        ref_text = " ".join([item.get("text", "") for item in gt_data if "text" in item])
        hyp_text = " ".join([item.get("text", "") for item in hypothesis_cues if "text" in item])

        normalize = lambda s: re.sub(r"[^\w\u4e00-\u9fff]", "", s.lower()).strip()
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

        accuracy = max(0.0, (1.0 - cer_score) * 100)
        return {
            "ground_truth": ground_truth_file,
            "total_words_ref": len(ref_words),
            "total_words_hyp": len(hyp_words),
            "wer": round(wer_score, 4),
            "cer": round(cer_score, 4),
            "accuracy": round(accuracy, 2),
        }
    except Exception as e:
        print(f"[!] Lỗi khi đánh giá OCR: {e}")
        return None


# ============================================================
# 6. XUẤT FILE SRT
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
# 7. HÀM MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Kiểm thử trích xuất phụ đề video bằng RapidOCR")
    parser.add_argument("video", nargs="?", default=None, help="Đường dẫn file video đầu vào")
    parser.add_argument("--roi", default=None, help="Tọa độ ROI: 'x,y,w,h' (tỉ lệ từ 0.0 đến 1.0)")
    parser.add_argument("--no-gui", action="store_true", help="Không mở cửa sổ kéo chuột, dùng ROI mặc định")
    parser.add_argument("--fps", type=float, default=3.5, help="Tốc độ quét scan FPS (mặc định 3.5)")
    parser.add_argument("--lang", default="zh", help="Ngôn ngữ nhận dạng (zh, en, vi, ja, ko)")
    args = parser.parse_args()

    # Tìm file video mặc định nếu không truyền
    if args.video:
        video_path = args.video
    else:
        candidates = ["data/input_video.mp4", "data/sample.mp4", "data/video.mp4"]
        video_path = next((c for c in candidates if os.path.exists(c)), "data/input_video.mp4")

    if not os.path.exists(video_path):
        print(f"[!] Không tìm thấy file video: {video_path}")
        print("    Vui lòng đặt video vào thư mục data/input_video.mp4 rồi chạy lại.")
        return

    # Xác định vùng ROI mặc định (Dải 22% ở đáy màn hình)
    default_roi = (0.05, 0.75, 0.90, 0.22)

    if args.roi:
        try:
            parts = [float(p.strip()) for p in args.roi.split(",")]
            if len(parts) == 4:
                roi = (parts[0], parts[1], parts[2], parts[3])
            else:
                roi = default_roi
        except Exception:
            roi = default_roi
    elif not args.no_gui:
        # Mở giao diện kéo chuột
        roi = select_roi_interactive(video_path, default_roi)
    else:
        roi = default_roi

    # Khởi động Profiler đo tài nguyên hệ thống
    profiler = HardwareProfiler()
    profiler.start()

    # Khởi tạo và chạy pipeline
    pipeline = VideoOCRPipeline(lang=args.lang, scan_fps=args.fps)
    cues, video_duration = pipeline.process_video(video_path, roi)

    # Dừng Profiler
    hw_stats = profiler.stop()

    # Lưu kết quả
    os.makedirs("output", exist_ok=True)
    output_json = "output/ocr_output.json"
    output_srt = "output/ocr_subtitles.srt"
    ground_truth = "output/subtitles-source.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"cues": cues, "hardware_benchmark": hw_stats}, f, ensure_ascii=False, indent=2)
    save_to_srt(cues, output_srt)

    print(f"\n[✓] Đã lưu kết quả JSON : {output_json}")
    print(f"[✓] Đã lưu file phụ đề SRT: {output_srt}")

    # In báo cáo khoa học tổng hợp
    acc_metrics = compute_ocr_accuracy(cues, ground_truth)
    profiler.print_academic_report(
        task_name="Trích xuất phụ đề Video (OCR)",
        model_name=f"RapidOCR PP-OCRv6 ONNX DirectML ({args.lang})",
        media_duration_sec=video_duration,
        accuracy_metrics=acc_metrics,
    )


if __name__ == "__main__":
    main()
