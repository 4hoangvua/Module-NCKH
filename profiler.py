"""Module đo lường hiệu năng phần cứng & tài nguyên hệ thống phục vụ NCKH.

Đo lường:
  - Thời gian xử lý & Tốc độ (RTF, Real-time multiplier).
  - Tiêu thụ RAM (Peak RAM, RAM Delta).
  - Tải CPU (CPU %, Số Cores, giới hạn luồng).
  - Tiêu thụ GPU / VRAM (Đo trực tiếp qua NVIDIA NVML hoặc PyTorch CUDA).
"""

from __future__ import annotations

import os
import sys
import time
import threading
from typing import Any

# Kiểm tra psutil an toàn
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Kiểm tra NVIDIA NVML an toàn
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# Kiểm tra PyTorch CUDA
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH_CUDA = False


class HardwareProfiler:
    """Bộ lấy mẫu tài nguyên phần cứng định kỳ chạy trên Background Thread."""

    def __init__(self, sample_interval: float = 0.05):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

        self.is_running = False
        self._thread: threading.Thread | None = None

        self.start_time: float = 0.0
        self.end_time: float = 0.0

        # Số liệu RAM (Bytes)
        self.init_ram: int = 0
        self.peak_ram: int = 0

        # Số liệu CPU (%)
        self.cpu_samples: list[float] = []

        # Số liệu GPU (Bytes & Tên thiết bị)
        self.gpu_device_name: str | None = None
        self.has_gpu_usage: bool = False
        self.init_vram: int = 0
        self.peak_vram: int = 0
        self.max_vram_observed: int = 0
        self._nvml_handle = None

        if HAS_NVML:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_device_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            except Exception:
                self._nvml_handle = None

        if not self.gpu_device_name and HAS_TORCH_CUDA:
            try:
                self.gpu_device_name = torch.cuda.get_device_name(0)
            except Exception:
                self.gpu_device_name = None

    def _sample_loop(self):
        """Vòng lặp lấy mẫu tài nguyên chạy nền."""
        while self.is_running:
            try:
                # 1. Đo RAM Process (RSS) bao gồm cả tiến trình con
                if self.process:
                    mem_rss = self.process.memory_info().rss
                    try:
                        for child in self.process.children(recursive=True):
                            try:
                                mem_rss += child.memory_info().rss
                            except Exception:
                                pass
                    except Exception:
                        pass

                    if mem_rss > self.peak_ram:
                        self.peak_ram = mem_rss

                    # 2. Đo CPU %
                    cpu_p = self.process.cpu_percent(interval=None)
                    if cpu_p >= 0.0:
                        self.cpu_samples.append(cpu_p)

                # 3. Đo GPU VRAM qua NVML (NVIDIA) hoặc PyTorch CUDA
                if self._nvml_handle:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    used_vram = mem_info.used
                    if used_vram > self.max_vram_observed:
                        self.max_vram_observed = used_vram
                    delta_vram = used_vram - self.init_vram
                    if delta_vram > self.peak_vram:
                        self.peak_vram = delta_vram
                        if delta_vram > 20 * 1024 * 1024:
                            self.has_gpu_usage = True

                elif HAS_TORCH_CUDA:
                    vram_alloc = torch.cuda.memory_allocated()
                    vram_max = torch.cuda.max_memory_allocated()
                    current_vram = max(vram_alloc, vram_max)
                    if current_vram > self.peak_vram:
                        self.peak_vram = current_vram
                        if current_vram > 10 * 1024 * 1024:
                            self.has_gpu_usage = True

            except Exception:
                pass

            time.sleep(self.sample_interval)

    def start(self):
        """Bắt đầu đo lường."""
        if self.process:
            self.process.cpu_percent(interval=None)
            self.init_ram = self.process.memory_info().rss
            self.peak_ram = self.init_ram
        self.cpu_samples = []

        if self._nvml_handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                self.init_vram = mem_info.used
                self.max_vram_observed = self.init_vram
                self.peak_vram = 0
            except Exception:
                pass
        elif HAS_TORCH_CUDA:
            try:
                torch.cuda.reset_peak_memory_stats()
                self.init_vram = torch.cuda.memory_allocated()
                self.peak_vram = self.init_vram
            except Exception:
                pass

        self.is_running = True
        self.start_time = time.time()

        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Dừng đo lường và trả về báo cáo số liệu."""
        self.end_time = time.time()
        self.is_running = False

        if self._thread:
            self._thread.join(timeout=0.5)

        # Lấy mẫu chốt hạ
        try:
            if self.process:
                mem_rss = self.process.memory_info().rss
                try:
                    for child in self.process.children(recursive=True):
                        try:
                            mem_rss += child.memory_info().rss
                        except Exception:
                            pass
                except Exception:
                    pass
                if mem_rss > self.peak_ram:
                    self.peak_ram = mem_rss

            if self._nvml_handle:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                delta_vram = mem_info.used - self.init_vram
                if delta_vram > self.peak_vram:
                    self.peak_vram = delta_vram
                if self.peak_vram > 20 * 1024 * 1024:
                    self.has_gpu_usage = True
        except Exception:
            pass

        # Tính toán các chỉ số thống kê
        elapsed_sec = max(0.001, self.end_time - self.start_time)
        peak_ram_mb = (self.peak_ram / (1024 * 1024)) if self.peak_ram > 0 else 0.0
        ram_delta_mb = max(0.0, (self.peak_ram - self.init_ram) / (1024 * 1024)) if self.peak_ram > 0 else 0.0
        
        cpu_cores = psutil.cpu_count(logical=True) if HAS_PSUTIL else (os.cpu_count() or 4)
        raw_avg_cpu = sum(self.cpu_samples) / max(len(self.cpu_samples), 1) if self.cpu_samples else (psutil.cpu_percent() if HAS_PSUTIL else 0.0)
        normalized_cpu = min(100.0, raw_avg_cpu / max(cpu_cores, 1))

        stats = {
            "elapsed_sec": round(elapsed_sec, 3),
            "peak_ram_mb": round(peak_ram_mb, 2),
            "ram_delta_mb": round(ram_delta_mb, 2),
            "avg_cpu_percent": round(normalized_cpu, 1),
            "raw_cpu_percent": round(raw_avg_cpu, 1),
            "cpu_cores": cpu_cores,
            "has_gpu": self.has_gpu_usage,
            "gpu_name": self.gpu_device_name if self.has_gpu_usage else None,
            "peak_vram_mb": round(self.peak_vram / (1024 * 1024), 2) if self.has_gpu_usage else 0.0,
            "peak_vram_gb": round(self.peak_vram / (1024 * 1024 * 1024), 3) if self.has_gpu_usage else 0.0,
        }
        return stats

    def print_academic_report(
        self,
        task_name: str,
        model_name: str,
        media_duration_sec: float | None = None,
        accuracy_metrics: dict[str, Any] | None = None,
        forced_gpu_name: str | None = None,
    ):
        """In bảng báo cáo tổng hợp chuẩn NCKH (Độ chính xác + Tốc độ + Phần cứng)."""
        stats = self.stop() if self.is_running else self.stop()

        print("\n" + "=" * 65)
        print(f"  📊 BÁO CÁO THỰC NGHIỆM NCKH: {task_name.upper()}")
        print("=" * 65)
        print(f"  • Mô hình AI (Model)                   : {model_name}")

        # 1. ĐỘ CHÍNH XÁC (ACCURACY)
        if accuracy_metrics:
            print("-" * 65)
            print("  1. ĐỘ CHÍNH XÁC NHẬN DẠNG (ACCURACY BENCHMARK)")
            if "ground_truth" in accuracy_metrics:
                print(f"  • File đối sánh (Ground Truth)         : {accuracy_metrics['ground_truth']}")
            if "total_words_ref" in accuracy_metrics:
                print(f"  • Số từ chuẩn / Nhận dạng              : {accuracy_metrics['total_words_ref']} / {accuracy_metrics.get('total_words_hyp', 0)} từ")
            if "wer" in accuracy_metrics:
                print(f"  • Word Error Rate (WER)                : {accuracy_metrics['wer'] * 100:.2f}%")
            if "cer" in accuracy_metrics:
                print(f"  • Character Error Rate (CER)           : {accuracy_metrics['cer'] * 100:.2f}%")
            if "accuracy" in accuracy_metrics:
                print(f"  • ⭐ ĐỘ CHÍNH XÁC (Accuracy)           : {accuracy_metrics['accuracy']:.2f}%")

        # 2. THỜI GIAN & TỐC ĐỘ (SPEED & LATENCY)
        print("-" * 65)
        print("  2. THỜI GIAN & TỐC ĐỘ XỬ LÝ (SPEED & LATENCY)")
        if media_duration_sec and media_duration_sec > 0:
            rtf = stats['elapsed_sec'] / media_duration_sec
            speed_x = media_duration_sec / stats['elapsed_sec']
            print(f"  • Thời lượng Media                     : {media_duration_sec:.2f}s (~{media_duration_sec/60:.1f} phút)")
            print(f"  • Thời gian xử lý (Inference Time)     : {stats['elapsed_sec']:.2f}s")
            print(f"  • Real-Time Factor (RTF)               : {rtf:.4f} (Càng nhỏ càng tốt)")
            print(f"  • Tốc độ xử lý (Speed Multiplier)      : {speed_x:.1f}x realtime")
        else:
            print(f"  • Tổng thời gian thực thi              : {stats['elapsed_sec']:.2f}s")

        # 3. TIÊU THỤ TÀI NGUYÊN HỆ THỐNG (RESOURCE FOOTPRINT)
        print("-" * 65)
        print("  3. TIÊU THỤ TÀI NGUYÊN PHẦN CỨNG (RESOURCE FOOTPRINT)")
        print(f"  • Tải CPU trung bình                   : {stats['avg_cpu_percent']:.1f}% (trên {stats['cpu_cores']} logical cores)")
        print(f"  • Dung lượng RAM đỉnh (Peak RAM)       : {stats['peak_ram_mb']:,.2f} MB")
        print(f"  • Mức tăng RAM thực tế (RAM Delta)     : +{stats['ram_delta_mb']:,.2f} MB")

        detected_gpu = forced_gpu_name or stats["gpu_name"]
        if stats["has_gpu"] and detected_gpu:
            print(f"  • Thiết bị phần cứng (Hardware Mode)   : 🚀 GPU ({detected_gpu})")
            if stats['peak_vram_mb'] > 0:
                print(f"  • Bộ nhớ GPU đỉnh (Peak VRAM)          : {stats['peak_vram_mb']:,.2f} MB ({stats['peak_vram_gb']:.2f} GB)")
        elif sys.platform == "darwin":
            print(f"  • Thiết bị phần cứng (Hardware Mode)   : 💻 Apple Silicon / Unified Memory")
        else:
            print(f"  • Thiết bị phần cứng (Hardware Mode)   : 💻 CPU Only (Không chiếm dụng VRAM GPU)")

        print("=" * 65 + "\n")
