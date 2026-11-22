import sys
import os
import csv
import wave
import time
import tempfile
import subprocess
import shutil
import hashlib
import json
import logging
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable
from pathlib import Path

import cv2
import numpy as np
from proglog import ProgressBarLogger

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QSlider, QListWidget, QLabel, QGroupBox, QListWidgetItem,
    QAbstractItemView, QProgressBar, QSizePolicy, QTabWidget, QSpinBox,
    QCheckBox, QButtonGroup, QRadioButton, QGridLayout, QDialog,
    QDialogButtonBox, QMessageBox, QDoubleSpinBox, QComboBox, QLineEdit, QSplitter,
    QMenu, QStatusBar, QListView, QScrollArea
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QUrl, QThread, Signal, QPoint, QRect, QSize, QTimer, QSettings, QObject, QRunnable, QThreadPool, QLineF
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QShortcut, QKeySequence, QIcon

# ============================================================
# LOGGING & UTILITIES
# ============================================================
log = logging.getLogger("video_editor")

def _configure_logging():
    log.setLevel(logging.DEBUG)
    if log.handlers: return
    try:
        fh = logging.FileHandler("video_editor.log", mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
        log.addHandler(fh)
    except Exception as e: print(f"Failed to init file logger: {e}", file=sys.stderr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(ch)
    log.propagate = False

    def excepthook(exc_type, exc_value, exc_tb):
        log.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = excepthook

HAARCASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".mpg", ".mpeg")
PROJECT_EXTS = "JSON Project (*.json)"
AUTOSAVE_FILE = "video_editor.autosave.json"
AUTOSAVE_INTERVAL_MS = 60_000
PROGRESS_THROTTLE_MS = 100

def format_timestamp(seconds: float) -> str:
    if seconds < 0: seconds = 0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    if hrs > 0: return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
    return f"{mins:02d}:{secs:02d}.{ms:03d}"

def rgb_frame_to_qimage(frame_rgb: np.ndarray) -> QImage:
    h, w = frame_rgb.shape[:2]
    bytes_per_line = w * 3
    return QImage(frame_rgb.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()

def rgb_frame_to_pixmap(frame_rgb: np.ndarray, max_w: Optional[int] = None, max_h: Optional[int] = None) -> QPixmap:
    q_img = rgb_frame_to_qimage(frame_rgb)
    pix = QPixmap.fromImage(q_img)
    if max_w and max_h:
        pix = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pix

def get_color_for_category(category: str) -> QColor:
    if "Action" in category or "Shorts" in category: return QColor(220, 53, 69)
    if "Talking" in category: return QColor(0, 120, 215)
    if "Dead Air" in category or "Silence" in category: return QColor(108, 117, 125)
    if "Highlight" in category or "Montage" in category: return QColor(255, 193, 7)
    if "RL" in category: return QColor(102, 16, 242)
    return QColor(40, 167, 69)

# ============================================================
# MOVIEPY COMPATIBILITY WRAPPERS
# ============================================================
def _get_subclip(clip, start, end):
    if hasattr(clip, "subclipped"): return clip.subclipped(start, end)
    return clip.subclip(start, end)

def _resize_clip(clip, size=None, height=None, width=None):
    if hasattr(clip, "resized"):
        if size: return clip.resized(size)
        if height: return clip.resized(height=height)
        if width: return clip.resized(width=width)
    else:
        if size: return clip.resize(size)
        if height: return clip.resize(height=height)
        if width: return clip.resize(width=width)
    return clip

def _crop_clip(clip, **kwargs):
    if hasattr(clip, "cropped"): return clip.cropped(**kwargs)
    return clip.crop(**kwargs)

def _with_fps(clip, fps):
    if hasattr(clip, "with_fps"): return clip.with_fps(fps)
    return clip.set_fps(fps)

def _with_speed(clip, speed):
    if hasattr(clip, "with_speed_scaled"): return clip.with_speed_scaled(speed)
    return clip.fx(lambda c: c.speedx(speed))

def _with_volume(clip, vol):
    if hasattr(clip, "with_volume_scaled"): return clip.with_volume_scaled(vol)
    return clip.volumex(vol)

def _with_position(clip, pos, relative=False):
    if hasattr(clip, "with_position"): return clip.with_position(pos, relative=relative)
    return clip.set_position(pos, relative=relative)

def _with_duration(clip, dur):
    if hasattr(clip, "with_duration"): return clip.with_duration(dur)
    return clip.set_duration(dur)

def _with_start(clip, t):
    if hasattr(clip, "with_start"): return clip.with_start(t)
    return clip.set_start(t)

# ============================================================
# FILTER & KEN BURNS HELPERS
# ============================================================
def apply_lut_to_clip(clip, filter_type: str):
    """Applies color grading LUTs/filters using OpenCV on the fly."""
    def filter_fn(get_frame, t):
        frame = get_frame(t)
        if filter_type == "B&W":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif filter_type == "Vibrant":
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        elif filter_type == "Cinematic":
            frame = frame.astype(np.float32)
            frame = (frame - 128) * 1.2 + 128
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.05 + 5, 0, 255)
            return np.clip(frame, 0, 255).astype(np.uint8)
        elif filter_type == "Cool":
            frame = frame.astype(np.float32)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.1 + 5, 0, 255)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.95, 0, 255)
            return np.clip(frame, 0, 255).astype(np.uint8)
        elif filter_type == "Warm":
            frame = frame.astype(np.float32)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.1 + 5, 0, 255)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.95, 0, 255)
            return np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    return clip.fl_image(filter_fn)

def apply_ken_burns(clip, effect: str):
    """Applies Pan/Zoom effects using OpenCV resize."""
    duration = max(0.1, clip.duration)
    w, h = clip.size
    
    def filter_fn(get_frame, t):
        frame = get_frame(t)
        if effect == "Zoom In":
            scale = 1.0 + 0.1 * (t / duration)
        elif effect == "Zoom Out":
            scale = 1.1 - 0.1 * (t / duration)
        elif effect == "Pan Left-Right":
            scale = 1.2
        else:
            return frame
            
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= w or new_h <= h:
            return frame
            
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if effect == "Pan Left-Right":
            x_offset = int((t / duration) * (new_w - w))
            y_center = new_h // 2
            return frame_resized[y_center - h//2 : y_center + h//2, x_offset:x_offset+w]
        else:
            x_center = new_w // 2
            y_center = new_h // 2
            return frame_resized[y_center - h//2 : y_center + h//2, x_center - w//2 : x_center + w//2]
            
    return clip.fl_image(filter_fn)

# ============================================================
# ASYNC THUMBNAIL WORKER
# ============================================================
class ThumbnailSignals(QObject):
    finished = Signal(int, QImage)

class ThumbnailTask(QRunnable):
    def __init__(self, media_path: str, t_sec: float, seg_id: int, w: int, h: int):
        super().__init__()
        self.media_path, self.t_sec, self.seg_id, self.w, self.h = media_path, t_sec, seg_id, w, h
        self.signals = ThumbnailSignals()

    def run(self):
        q_img = QImage()
        if not os.path.isfile(self.media_path):
            self.signals.finished.emit(self.seg_id, q_img); return
        cap = cv2.VideoCapture(self.media_path)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, self.t_sec * 1000)
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.mean() > 5:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        q_img = rgb_frame_to_qimage(rgb)
                        break
        except Exception as e:
            log.warning(f"ThumbnailTask error: {e}")
        finally:
            cap.release()
            self.signals.finished.emit(self.seg_id, q_img)

# ============================================================
# CV MODES DATA TABLE
# ============================================================
@dataclass
class CVModeSpec:
    key: str
    label: str
    icon: str
    slider_range: Tuple[int, int]
    slider_default: int
    label_text: str
    threshold_transform: Optional[Callable[[float], float]] = None
    needs_audio: bool = False

CV_MODES: List[CVModeSpec] = [
    CVModeSpec("scene",    "Scene Match",        "🎬", (1, 50),  10, "Change % (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("motion",   "Motion Detection",   "🏃", (1, 50),  10, "Change % (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("face",     "Face Detection",     "👤", (1, 50),  10, "Min Neighbors (1-50):"),
    CVModeSpec("fade",     "Fade to Black",      "⬛", (1, 80),  20, "Brightness (1-80):"),
    CVModeSpec("color",    "Color Scene Cut",    "🌈", (1, 50),  20, "Hist Diff (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("action",   "Action / Flow",      "🏃‍♂️", (1, 50),  5,  "Flow Action (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("sharp",    "Sharpness / Focus",  "🔬", (1, 100), 30, "Sharpness (1-100):", threshold_transform=lambda v: v * 10.0),
    CVModeSpec("person",   "Person Detection",   "🚶", (1, 50),  10, "Confidence (1-50):"),
    CVModeSpec("silence",  "Dead Air (Audio)",   "🤐", (0, 1),   0,  "Uses Audio dB Slider:", needs_audio=True),
    CVModeSpec("template", "Reference Match",    "🎯", (1, 100), 70, "Match Score % (1-100):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("shorts",   "🔥 Shorts/Reels Finder", "📱", (1, 100), 20, "Min Action Energy (1-100):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("talkhead", "🗣️ Talking Head (Long)", "🎙️", (1, 50),  15, "Max Background Motion (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("gameplay", "🎮 Gameplay Action", "🎮", (1, 50),  15, "Min Action (1-50):", threshold_transform=lambda v: v / 100.0),
    CVModeSpec("rl_episode", "🤖 RL Episode Match", "🤖", (1, 100), 75, "Boundary Certainty % (1-100):", threshold_transform=lambda v: v / 100.0),
]

# ============================================================
# FILMSTRIP TIMELINE WIDGET
# ============================================================
class FilmstripGenerator(QThread):
    finished = Signal(list, int, np.ndarray)

    def __init__(self, video_path: str, count: int = 20, thumb_w: int = 160, thumb_h: int = 72):
        super().__init__()
        self.video_path, self.count, self.thumb_w, self.thumb_h = video_path, count, thumb_w, thumb_h
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        thumbs, waveform = [], np.array([])
        cap = cv2.VideoCapture(self.video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            if total <= 0 or fps <= 0:
                if not self._cancelled:
                    self.finished.emit([], 0, np.array([]))
                return
            duration_ms = int((total / fps) * 1000)
            step = max(1, total // self.count)
            for i in range(self.count):
                if self._cancelled: break
                target = i * step
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - 1))
                for _ in range(8):
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= target: break
                    if not cap.grab(): break
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    thumbs.append(rgb_frame_to_pixmap(rgb, self.thumb_w, self.thumb_h))
        finally:
            cap.release()

        if self._cancelled: return

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
            try:
                subprocess.run([ffmpeg_path, "-y", "-i", self.video_path, "-vn", "-ac", "1", "-ar", "8000", "-f", "wav", tmp.name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                with wave.open(tmp.name, "rb") as wf: raw = wf.readframes(wf.getnframes())
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                target_points = 400
                chunk_size = max(1, len(samples) // target_points)
                valid_len = (len(samples) // chunk_size) * chunk_size
                if valid_len > 0:
                    waveform = np.max(np.abs(samples[:valid_len]).reshape(-1, chunk_size), axis=1)
            except Exception: pass
            finally:
                try: os.remove(tmp.name)
                except OSError: pass
        
        if not self._cancelled:
            self.finished.emit(thumbs, duration_ms, waveform)

class FilmstripWidget(QWidget):
    seek_requested = Signal(int)
    hover_time = Signal(int)
    THUMB_TARGET_W = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnails, self.segment_markers, self.waveform = [], [], np.array([])
        self.duration_ms, self.current_pos_ms = 0, 0
        self._generator = None
        self._scaled_thumbs, self._last_width = [], -1
        self._zoom_start, self._zoom_end = 0.0, 1.0
        self.setMinimumHeight(100); self.setMaximumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")

    def set_media(self, video_path: str):
        self.thumbnails.clear(); self.segment_markers.clear()
        self.waveform = np.array([]); self.duration_ms = 0
        self._zoom_start, self._zoom_end = 0.0, 1.0
        self._invalidate_cache(); self.update()
        
        if self._generator and self._generator.isRunning():
            self._generator.cancel()
            if not self._generator.wait(3000):
                log.warning("Filmstrip generator did not stop in time, terminating.")
                self._generator.terminate()
                self._generator.wait(1000)
                
        count = max(10, min(60, self.width() // self.THUMB_TARGET_W))
        self._generator = FilmstripGenerator(video_path, count=count)
        self._generator.finished.connect(self._on_thumbs_ready)
        self._generator.start()

    def _on_thumbs_ready(self, thumbs, duration_ms, waveform):
        self.thumbnails, self.duration_ms, self.waveform = thumbs, duration_ms, waveform
        self._invalidate_cache(); self.update()

    def set_segment_markers(self, markers): self.segment_markers = markers; self.update()
    def update_playhead(self, pos_ms): self.current_pos_ms = pos_ms; self.update()
    
    def clear_media(self):
        self.thumbnails.clear(); self.segment_markers.clear()
        self.waveform = np.array([]); self.duration_ms = 0
        self._invalidate_cache(); self.update()

    def _invalidate_cache(self): self._scaled_thumbs, self._last_width = [], -1

    def _x_for_ms(self, ms: int) -> float:
        if self.duration_ms <= 0: return 0
        t = ms / self.duration_ms
        t = (t - self._zoom_start) / max(1e-6, self._zoom_end - self._zoom_start)
        return max(0.0, min(1.0, t)) * self.width()

    def _ms_for_x(self, x: float) -> int:
        t = x / max(1, self.width())
        t = self._zoom_start + t * (self._zoom_end - self._zoom_start)
        return int(t * self.duration_ms)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            if not self.thumbnails:
                painter.fillRect(self.rect(), QColor(20, 20, 20)); return
            w, h = self.width(), self.height()
            thumb_h = h - 20
            thumb_w = w / max(1, len(self.thumbnails))
            if self._last_width != w or not self._scaled_thumbs:
                self._scaled_thumbs = [pix.scaled(int(thumb_w) + 1, thumb_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation) for pix in self.thumbnails]
                self._last_width = w
            for i, scaled in enumerate(self._scaled_thumbs): painter.drawPixmap(int(i * thumb_w), 0, scaled)
            
            if len(self.waveform) > 0:
                wave_h = 15
                wave_y = h - wave_h - 2
                painter.setPen(QPen(QColor(0, 255, 255, 150), 1))
                mid_y = wave_y + wave_h // 2
                step = w / len(self.waveform)
                
                lines = []
                for i, val in enumerate(self.waveform):
                    bar_h = int((val / 32768.0) * (wave_h / 2))
                    x = int(i * step)
                    lines.append(QLineF(x, mid_y - bar_h, x, mid_y + bar_h))
                painter.drawLines(lines)

            if self.duration_ms > 0:
                for s_ms, e_ms, color in self.segment_markers:
                    x1, x2 = self._x_for_ms(s_ms), self._x_for_ms(e_ms)
                    painter.setBrush(color); painter.setPen(Qt.NoPen)
                    painter.drawRect(int(x1), thumb_h, max(1, int(x2 - x1)), 8)
                x = self._x_for_ms(self.current_pos_ms)
                painter.setPen(QPen(QColor(255, 50, 50), 2))
                painter.drawLine(int(x), 0, int(x), h)
        finally:
            painter.end()

    def mousePressEvent(self, event):
        if self.duration_ms > 0 and event.button() == Qt.LeftButton:
            pos = event.position()
            new_pos = self._ms_for_x(pos.x())
            snap_threshold_ms = 1500
            best_snap, best_dist = None, snap_threshold_ms
            for s_ms, e_ms, _ in self.segment_markers:
                dist_s = abs(new_pos - s_ms)
                if dist_s < best_dist: best_dist, best_snap = dist_s, s_ms
                dist_e = abs(new_pos - e_ms)
                if dist_e < best_dist: best_dist, best_snap = dist_e, e_ms
            if best_snap is not None: new_pos = best_snap
            self.seek_requested.emit(new_pos)
            self.current_pos_ms = new_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.duration_ms > 0:
            pos = event.position()
            ms = self._ms_for_x(pos.x())
            self.hover_time.emit(ms)
            self.setToolTip(format_timestamp(ms / 1000.0))
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._zoom_start, self._zoom_end = 0.0, 1.0
        self._invalidate_cache(); self.update()

    def wheelEvent(self, event):
        if not (event.modifiers() & Qt.ControlModifier) or self.duration_ms <= 0:
            return super().wheelEvent(event)
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 if delta > 0 else 0.85
        span = self._zoom_end - self._zoom_start
        new_span = max(0.05, min(1.0, span * factor))
        pos = event.position().x() / max(1, self.width())
        center = self._zoom_start + pos * span
        center = max(new_span / 2, min(1.0 - new_span / 2, center))
        self._zoom_start = max(0.0, center - new_span / 2)
        self._zoom_end = min(1.0, self._zoom_start + new_span)
        self._zoom_start = max(0.0, self._zoom_end - new_span)
        self._invalidate_cache(); self.update()

    def leaveEvent(self, event):
        self.hover_time.emit(-1); self.setToolTip("")
        super().leaveEvent(event)

# ============================================================
# VISUAL ROI PICKER DIALOG
# ============================================================
class CanvasLabel(QLabel):
    roi_selected = Signal(object)

    def __init__(self, frame_rgb: np.ndarray):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        oh, ow = frame_rgb.shape[:2]
        scale = min(800 / ow, 600 / oh, 1.0)
        self.scale_factor, self.scaled_w, self.scaled_h = scale, int(ow * scale), int(oh * scale)
        self.scaled_pixmap = rgb_frame_to_pixmap(frame_rgb).scaled(self.scaled_w, self.scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(self.scaled_pixmap)
        self.setFixedSize(self.scaled_w, self.scaled_h)
        self.start_pt, self.end_pt, self.is_drawing = QPoint(), QPoint(), False

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.start_pt.isNull() and not self.end_pt.isNull():
            painter = QPainter(self)
            try:
                rect = QRect(self.start_pt, self.end_pt).normalized()
                painter.setBrush(QColor(0, 0, 0, 150)); painter.setPen(Qt.NoPen)
                painter.drawRect(0, 0, self.width(), rect.top())
                painter.drawRect(0, rect.bottom(), self.width(), self.height() - rect.bottom())
                painter.drawRect(0, rect.top(), rect.left(), rect.height())
                painter.drawRect(rect.right(), rect.top(), self.width() - rect.right(), rect.height())
                painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine)); painter.setBrush(Qt.NoBrush)
                painter.drawRect(rect)
            finally:
                painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pt = event.position().toPoint()
            self.start_pt, self.end_pt, self.is_drawing = pt, pt, True
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            pos = event.position()
            self.end_pt = QPoint(max(0, min(int(pos.x()), self.width())), max(0, min(int(pos.y()), self.height())))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            pos = event.position()
            self.end_pt = QPoint(max(0, min(int(pos.x()), self.width())), max(0, min(int(pos.y()), self.height())))
            self.is_drawing = False; self.update()
            rect = QRect(self.start_pt, self.end_pt).normalized()
            if rect.width() > 5 and rect.height() > 5:
                self.roi_selected.emit((int(rect.x()/self.scale_factor), int(rect.y()/self.scale_factor), int(rect.width()/self.scale_factor), int(rect.height()/self.scale_factor)))
            else:
                self.roi_selected.emit(None)

class VisualPickerDialog(QDialog):
    def __init__(self, frame_rgb: np.ndarray, parent=None, title="Select Region of Interest (ROI)"):
        super().__init__(parent)
        self.setWindowTitle(title); self.setModal(True); self.resize(850, 650)
        self.roi_rect = None
        layout = QVBoxLayout()
        self.canvas = CanvasLabel(frame_rgb)
        self.canvas.roi_selected.connect(self.update_roi_info)
        layout.addWidget(self.canvas, 1, Qt.AlignCenter)
        self.info_label = QLabel("Click and drag on the image to select a region.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #aaa; padding: 5px;")
        layout.addWidget(self.info_label)
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Selection")
        self.clear_btn.clicked.connect(self.clear_roi)
        btn_layout.addWidget(self.clear_btn)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout); self.setLayout(layout)

    def update_roi_info(self, roi):
        self.roi_rect = roi
        if roi: self.info_label.setText(f"Selected Region: X={roi[0]}, Y={roi[1]}, W={roi[2]}, H={roi[3]}")
        else: self.info_label.setText("No region selected.")

    def clear_roi(self):
        self.canvas.start_pt, self.canvas.end_pt = QPoint(), QPoint()
        self.canvas.update(); self.update_roi_info(None)

# ============================================================
# DIALOGS
# ============================================================
class EditSegmentDialog(QDialog):
    def __init__(self, start: float, end: float, name: str, category: str = "Default", speed: float = 1.0, track: int = 1, parent=None, filter_type="None", ken_burns="None", jcut=0.0, lcut=0.0):
        super().__init__(parent)
        self.setWindowTitle("Edit Segment"); self.setModal(True)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(name); layout.addWidget(self.name_edit)
        
        layout.addWidget(QLabel("Start (seconds):"))
        self.start_spin = QDoubleSpinBox(); self.start_spin.setRange(0, 1e9); self.start_spin.setDecimals(3); self.start_spin.setSingleStep(0.1); self.start_spin.setValue(start)
        layout.addWidget(self.start_spin)
        
        layout.addWidget(QLabel("End (seconds):"))
        self.end_spin = QDoubleSpinBox(); self.end_spin.setRange(0, 1e9); self.end_spin.setDecimals(3); self.end_spin.setSingleStep(0.1); self.end_spin.setValue(end)
        layout.addWidget(self.end_spin)

        nudge_layout = QHBoxLayout()
        btn_s_minus = QPushButton("-0.1s"); btn_s_plus = QPushButton("+0.1s")
        btn_s_minus.clicked.connect(lambda: self.start_spin.setValue(self.start_spin.value() - 0.1))
        btn_s_plus.clicked.connect(lambda: self.start_spin.setValue(self.start_spin.value() + 0.1))
        btn_e_minus = QPushButton("-0.1s"); btn_e_plus = QPushButton("+0.1s")
        btn_e_minus.clicked.connect(lambda: self.end_spin.setValue(self.end_spin.value() - 0.1))
        btn_e_plus.clicked.connect(lambda: self.end_spin.setValue(self.end_spin.value() + 0.1))
        nudge_layout.addWidget(QLabel("Nudge In:")); nudge_layout.addWidget(btn_s_minus); nudge_layout.addWidget(btn_s_plus)
        nudge_layout.addStretch(); nudge_layout.addWidget(QLabel("Nudge Out:")); nudge_layout.addWidget(btn_e_minus); nudge_layout.addWidget(btn_e_plus)
        layout.addLayout(nudge_layout)

        layout.addWidget(QLabel("Category Tag:"))
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["Default", "🔥 Action", "🗣️ Talking", "💤 Dead Air", "⭐ Highlight", "🤖 RL Episode"])
        self.cat_combo.setCurrentText(category if category in [self.cat_combo.itemText(i) for i in range(self.cat_combo.count())] else "Default")
        layout.addWidget(self.cat_combo)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x", "1.5x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentText(f"{speed}x" if speed != 1.0 else "1.0x")
        speed_layout.addWidget(self.speed_combo); layout.addLayout(speed_layout)
        self.ramp_check = QCheckBox("Smooth speed ramp (ease in/out)")
        layout.addWidget(self.ramp_check)

        track_layout = QHBoxLayout()
        track_layout.addWidget(QLabel("Track:"))
        self.track_combo = QComboBox()
        self.track_combo.addItems(["🎬 Main Track (Full Screen)", "🖼️ B-Roll Track (Overlay)"])
        self.track_combo.setCurrentIndex(0 if track == 1 else 1)
        track_layout.addWidget(self.track_combo); layout.addLayout(track_layout)

        jl_layout = QHBoxLayout()
        jl_layout.addWidget(QLabel("J-Cut (Audio Early s):"))
        self.jcut_spin = QDoubleSpinBox(); self.jcut_spin.setRange(0, 10); self.jcut_spin.setValue(jcut); self.jcut_spin.setSingleStep(0.1)
        jl_layout.addWidget(self.jcut_spin)
        jl_layout.addWidget(QLabel("L-Cut (Audio Late s):"))
        self.lcut_spin = QDoubleSpinBox(); self.lcut_spin.setRange(0, 10); self.lcut_spin.setValue(lcut); self.lcut_spin.setSingleStep(0.1)
        jl_layout.addWidget(self.lcut_spin)
        layout.addLayout(jl_layout)

        fx_layout = QHBoxLayout()
        fx_layout.addWidget(QLabel("🎨 Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "B&W", "Vibrant", "Cinematic", "Cool", "Warm"])
        self.filter_combo.setCurrentText(filter_type if filter_type in ["None", "B&W", "Vibrant", "Cinematic", "Cool", "Warm"] else "None")
        fx_layout.addWidget(self.filter_combo)
        
        fx_layout.addWidget(QLabel("🎥 Ken Burns:"))
        self.ken_burns_combo = QComboBox()
        self.ken_burns_combo.addItems(["None", "Zoom In", "Zoom Out", "Pan Left-Right"])
        self.ken_burns_combo.setCurrentText(ken_burns if ken_burns in ["None", "Zoom In", "Zoom Out", "Pan Left-Right"] else "None")
        fx_layout.addWidget(self.ken_burns_combo)
        layout.addLayout(fx_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box); self.setLayout(layout)

    def values(self):
        speed = float(self.speed_combo.currentText().replace('x', ''))
        return (self.name_edit.text(), float(self.start_spin.value()), float(self.end_spin.value()), 
                self.cat_combo.currentText(), speed, self.ramp_check.isChecked(), 
                (1 if self.track_combo.currentIndex() == 0 else 2),
                self.filter_combo.currentText(), self.ken_burns_combo.currentText(),
                self.jcut_spin.value(), self.lcut_spin.value())

class TextOverlayDialog(QDialog):
    def __init__(self, parent=None, existing_text="", existing_position="bottom", existing_duration_pct=100, existing_style="bar"):
        super().__init__(parent)
        self.setWindowTitle("Text Overlay"); self.setModal(True)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Text:"))
        self.text_edit = QLineEdit(existing_text)
        self.text_edit.setPlaceholderText("Enter title, caption, or lower-third text...")
        layout.addWidget(self.text_edit)

        pos_layout = QHBoxLayout(); pos_layout.addWidget(QLabel("Position:"))
        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["🔝 Top", "🔽 Bottom", "🎯 Center", "↖️ Top-Left", "↙️ Bottom-Left"])
        positions = ["top", "bottom", "center", "top-left", "bottom-left"]
        self.pos_combo.setCurrentIndex(positions.index(existing_position) if existing_position in positions else 1)
        pos_layout.addWidget(self.pos_combo); layout.addLayout(pos_layout)

        dur_layout = QHBoxLayout(); dur_layout.addWidget(QLabel("Visible for:"))
        self.dur_combo = QComboBox()
        self.dur_combo.addItems(["Full duration", "First 3 seconds", "Last 3 seconds", "First half", "Second half"])
        durations = [100, 3, -3, 50, -50]
        self.dur_combo.setCurrentIndex(durations.index(existing_duration_pct) if existing_duration_pct in durations else 0)
        dur_layout.addWidget(self.dur_combo); layout.addLayout(dur_layout)

        style_layout = QHBoxLayout(); style_layout.addWidget(QLabel("Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["📋 Lower-Third Bar", "🎬 Cinematic", "💬 Caption Box", "✨ Minimal Text"])
        styles = ["bar", "cinematic", "box", "minimal"]
        self.style_combo.setCurrentIndex(styles.index(existing_style) if existing_style in styles else 0)
        style_layout.addWidget(self.style_combo); layout.addLayout(style_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box); self.setLayout(layout)

    def values(self):
        pos_map = {"🔝 Top": "top", "🔽 Bottom": "bottom", "🎯 Center": "center", "↖️ Top-Left": "top-left", "↙️ Bottom-Left": "bottom-left"}
        dur_map = {"Full duration": 100, "First 3 seconds": 3, "Last 3 seconds": -3, "First half": 50, "Second half": -50}
        style_map = {"📋 Lower-Third Bar": "bar", "🎬 Cinematic": "cinematic", "💬 Caption Box": "box", "✨ Minimal Text": "minimal"}
        return (self.text_edit.text().strip(), pos_map[self.pos_combo.currentText()], dur_map[self.dur_combo.currentText()], style_map[self.style_combo.currentText()])

class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts"); self.setMinimumWidth(400)
        layout = QVBoxLayout()
        html = """
        <h3>Playback</h3><table>
            <tr><td><b>Space / K</b></td><td>Play / Pause</td></tr>
            <tr><td>J</td><td>Slow down (0.5x)</td></tr>
            <tr><td>L</td><td>Speed up (2.0x)</td></tr>
            <tr><td>← / →</td><td>Seek -5s / +5s</td></tr>
            <tr><td>, / .</td><td>Step 1 frame back / forward</td></tr>
        </table>
        <h3>Editing</h3><table>
            <tr><td><b>I</b></td><td>Set In point</td></tr>
            <tr><td><b>O</b></td><td>Set Out point</td></tr>
            <tr><td>[ / ]</td><td>Snap to previous / next segment boundary</td></tr>
            <tr><td>Del</td><td>Remove selected segment</td></tr>
            <tr><td>Ctrl+Z / Ctrl+Y</td><td>Undo / Redo</td></tr>
            <tr><td>Ctrl+Click Segment</td><td>Edit Properties (J/L Cut, Filters)</td></tr>
        </table>
        <h3>Files</h3><table>
            <tr><td>Ctrl+S</td><td>Save Project</td></tr>
            <tr><td>Ctrl+O</td><td>Load Project</td></tr>
            <tr><td>Ctrl+E</td><td>Export Video</td></tr>
        </table>
        <h3>Filmstrip</h3><table>
            <tr><td>Ctrl + Wheel</td><td>Zoom timeline in/out</td></tr>
            <tr><td>Double-click</td><td>Reset zoom</td></tr>
        </table>
        """
        label = QLabel(html); label.setTextFormat(Qt.RichText); layout.addWidget(label)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok); btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box); self.setLayout(layout)

class YouTubeExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube Export Settings"); self.setModal(True)
        layout = QVBoxLayout()
        res_layout = QHBoxLayout(); res_layout.addWidget(QLabel("Target Resolution:"))
        self.res_combo = QComboBox(); self.res_combo.addItems(["1080p (1920x1080)", "1440p (2560x1440)", "4K (3840x2160)"])
        res_layout.addWidget(self.res_combo); layout.addLayout(res_layout)
        fps_layout = QHBoxLayout(); fps_layout.addWidget(QLabel("Target Framerate:"))
        self.fps_combo = QComboBox(); self.fps_combo.addItems(["30 fps", "60 fps"])
        fps_layout.addWidget(self.fps_combo); layout.addLayout(fps_layout)
        speed_layout = QHBoxLayout(); speed_layout.addWidget(QLabel("CPU Encoder Speed:"))
        self.speed_combo = QComboBox(); self.speed_combo.addItems(["slow", "medium", "fast", "ultrafast"]); self.speed_combo.setCurrentText("medium")
        speed_layout.addWidget(self.speed_combo); layout.addLayout(speed_layout)
        info = QLabel("ℹ️ Bitrate will be set automatically to YouTube's official standards.\nVideos will be scaled and cropped to fit the exact 16:9 aspect ratio.")
        info.setWordWrap(True); info.setStyleSheet("color: #aaa; padding: 5px;"); layout.addWidget(info)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box); self.setLayout(layout)

    def get_settings(self):
        res_text = self.res_combo.currentText()
        if "1080p" in res_text: res = (1920, 1080)
        elif "1440p" in res_text: res = (2560, 1440)
        else: res = (3840, 2160)
        return res, (30 if "30 fps" in self.fps_combo.currentText() else 60), self.speed_combo.currentText()

class ThumbnailGeneratorDialog(QDialog):
    def __init__(self, frame_rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YouTube Thumbnail Generator"); self.setModal(True)
        self.resize(1100, 700)
        self.frame = frame_rgb.copy()
        self.target_w, self.target_h = 1920, 1080
        
        layout = QVBoxLayout()
        self.canvas = QLabel(); self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: black;")
        layout.addWidget(self.canvas, 1)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Headline Text:"))
        self.text_input = QLineEdit("WATCH THIS!")
        self.text_input.textChanged.connect(self.update_preview)
        ctrl_layout.addWidget(self.text_input)
        
        ctrl_layout.addWidget(QLabel("Font Size:"))
        self.size_spin = QSpinBox(); self.size_spin.setRange(50, 300); self.size_spin.setValue(150)
        self.size_spin.valueChanged.connect(self.update_preview)
        ctrl_layout.addWidget(self.size_spin)
        
        self.btn_export = QPushButton("💾 Export 1920x1080 JPG")
        self.btn_export.clicked.connect(self.export_thumb)
        ctrl_layout.addWidget(self.btn_export)
        layout.addLayout(ctrl_layout)
        self.setLayout(layout)
        self.update_preview()

    def render_image(self):
        h, w = self.frame.shape[:2]
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
        y_off = (self.target_h - new_h) // 2
        x_off = (self.target_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        text = self.text_input.text()
        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.size_spin.value() / 50.0
            thickness = int(font_scale * 4)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (self.target_w - tw) // 2
            y = self.target_h - 100
            cv2.putText(canvas, text, (x+5, y+5), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), font, font_scale, (255,255,0), thickness, cv2.LINE_AA)
        return canvas

    def update_preview(self):
        canvas = self.render_image()
        q_img = rgb_frame_to_qimage(canvas)
        self.canvas.setPixmap(QPixmap.fromImage(q_img).scaled(800, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def export_thumb(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Thumbnail", "thumbnail.jpg", "JPEG Image (*.jpg *.jpeg)")
        if not path: return
        canvas = self.render_image()
        cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        self.accept()

class BatchExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export Profiles"); self.setModal(True)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select formats to export simultaneously:"))
        
        self.check_169 = QCheckBox("16:9 Landscape (1920x1080) - YouTube")
        self.check_169.setChecked(True)
        self.check_916 = QCheckBox("9:16 Vertical (1080x1920) - Shorts/TikTok")
        self.check_11 = QCheckBox("1:1 Square (1080x1080) - Instagram")
        
        layout.addWidget(self.check_169)
        layout.addWidget(self.check_916)
        layout.addWidget(self.check_11)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept); btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box); self.setLayout(layout)

    def get_selected(self):
        out = []
        if self.check_169.isChecked(): out.append("16:9")
        if self.check_916.isChecked(): out.append("9:16")
        if self.check_11.isChecked(): out.append("1:1")
        return out

# ============================================================
# MULTI-TRACK VISUAL TIMELINE WIDGET
# ============================================================
class TimelineWidget(QWidget):
    seek_requested = Signal(int)
    segment_moved = Signal()
    
    TRACK_HEIGHT = 50
    HEADER_WIDTH = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments = []
        self.duration_ms = 0
        self.current_pos_ms = 0
        self.setMouseTracking(True)
        self.setMinimumHeight(160)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self._dragging_item = None
        self._drag_mode = None
        self._drag_start_x = 0
        self._original_start_ms = 0
        self._original_end_ms = 0

    def set_segments(self, segments, duration_ms):
        self.segments = segments
        self.duration_ms = duration_ms
        self.update()

    def update_playhead(self, pos_ms):
        self.current_pos_ms = pos_ms
        self.update()

    def _x_for_ms(self, ms, width):
        if self.duration_ms <= 0: return 0
        return self.HEADER_WIDTH + (ms / self.duration_ms) * (width - self.HEADER_WIDTH)

    def _ms_for_x(self, x, width):
        if self.duration_ms <= 0: return 0
        ratio = max(0, min(1, (x - self.HEADER_WIDTH) / max(1, (width - self.HEADER_WIDTH))))
        return ratio * self.duration_ms

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        
        tracks = ["Main", "B-Roll", "Audio"]
        painter.setPen(QColor(255, 255, 255))
        for i, name in enumerate(tracks):
            y = i * self.TRACK_HEIGHT
            painter.fillRect(0, y, self.HEADER_WIDTH, self.TRACK_HEIGHT, QColor(30, 30, 30))
            painter.drawText(10, y + 30, name)
            painter.drawLine(0, y + self.TRACK_HEIGHT, w, y + self.TRACK_HEIGHT)

        if self.duration_ms <= 0: return

        for seg in self.segments:
            x1 = self._x_for_ms(seg['start_ms'], w)
            x2 = self._x_for_ms(seg['end_ms'], w)
            track_y = (seg['track'] - 1) * self.TRACK_HEIGHT
            
            color = seg.get('color', QColor(40, 167, 69))
            painter.setBrush(color); painter.setPen(QPen(QColor(0,0,0), 1))
            painter.drawRect(int(x1), track_y + 2, max(2, int(x2 - x1)), self.TRACK_HEIGHT - 4)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(x1) + 5, track_y + 25, seg['name'][:20])

        px = self._x_for_ms(self.current_pos_ms, w)
        painter.setPen(QPen(QColor(255, 50, 50), 2))
        painter.drawLine(int(px), 0, int(px), h)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton: return
        x = event.position().x()
        y = event.position().y()
        track = (y // self.TRACK_HEIGHT) + 1
        ms = self._ms_for_x(x, self.width())
        
        for seg in self.segments:
            if seg['track'] != track: continue
            x1 = self._x_for_ms(seg['start_ms'], self.width())
            x2 = self._x_for_ms(seg['end_ms'], self.width())
            if x1 <= x <= x2:
                self._dragging_item = seg
                self._drag_start_x = x
                self._original_start_ms = seg['start_ms']
                self._original_end_ms = seg['end_ms']
                if abs(x - x1) < 10: self._drag_mode = "resize_left"
                elif abs(x - x2) < 10: self._drag_mode = "resize_right"
                else: self._drag_mode = "move"
                return
        self.seek_requested.emit(int(ms))

    def mouseMoveEvent(self, event):
        if not self._dragging_item: return
        x = event.position().x()
        delta_ms = self._ms_for_x(x, self.width()) - self._ms_for_x(self._drag_start_x, self.width())
        
        if self._drag_mode == "move":
            self._dragging_item['start_ms'] = max(0, self._original_start_ms + delta_ms)
            self._dragging_item['end_ms'] = max(0, self._original_end_ms + delta_ms)
        elif self._drag_mode == "resize_left":
            self._dragging_item['start_ms'] = max(0, min(self._original_end_ms - 1000, self._original_start_ms + delta_ms))
        elif self._drag_mode == "resize_right":
            self._dragging_item['end_ms'] = max(self._original_start_ms + 1000, self._original_end_ms + delta_ms)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging_item:
            self.segment_moved.emit()
            self._dragging_item = None
            self._drag_mode = None

# ============================================================
# UNDO/REDO STACK
# ============================================================
class SegmentUndoStack:
    def __init__(self, max_size=50):
        self._undo, self._redo = deque(maxlen=max_size), deque(maxlen=max_size)

    def snapshot(self, list_widget: QListWidget):
        self._undo.append(self._snapshot(list_widget)); self._redo.clear()

    def discard_last_snapshot(self):
        if self._undo: self._undo.pop()

    def undo(self, list_widget):
        if not self._undo: return False
        self._redo.append(self._snapshot(list_widget))
        self._restore(list_widget, self._undo.pop()); return True

    def redo(self, list_widget):
        if not self._redo: return False
        self._undo.append(self._snapshot(list_widget))
        self._restore(list_widget, self._redo.pop()); return True

    @staticmethod
    def _snapshot(list_widget):
        snap = []
        for i in range(list_widget.count()):
            it = list_widget.item(i)
            p, s, e = it.data(Qt.UserRole)
            cat = it.data(Qt.UserRole + 1) or "Default"
            spd = it.data(Qt.UserRole + 2) or 1.0
            rmp = it.data(Qt.UserRole + 3) or False
            ovl = it.data(Qt.UserRole + 4)
            trk = it.data(Qt.UserRole + 5) or 1
            name = it.data(Qt.UserRole + 7) or ""
            filt = it.data(Qt.UserRole + 8) or "None"
            kb = it.data(Qt.UserRole + 9) or "None"
            jcut = it.data(Qt.UserRole + 10) or 0.0
            lcut = it.data(Qt.UserRole + 11) or 0.0
            snap.append((p, s, e, name, it.background(), it.icon(), cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut))
        return snap

    @staticmethod
    def _restore(list_widget, snap):
        list_widget.clear()
        for p, s, e, name, bg, icon, cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut in snap:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, (p, s, e))
            it.setData(Qt.UserRole + 1, cat)
            it.setData(Qt.UserRole + 2, spd)
            it.setData(Qt.UserRole + 3, rmp)
            it.setData(Qt.UserRole + 4, ovl)
            it.setData(Qt.UserRole + 5, trk)
            it.setData(Qt.UserRole + 7, name)
            it.setData(Qt.UserRole + 8, filt)
            it.setData(Qt.UserRole + 9, kb)
            it.setData(Qt.UserRole + 10, jcut)
            it.setData(Qt.UserRole + 11, lcut)
            it.setBackground(bg)
            if trk == 2: it.setForeground(QColor(255, 200, 0))
            if not icon.isNull(): it.setIcon(icon)
            list_widget.addItem(it)
            list_widget._apply_item_decorations(it)

# ============================================================
# MOVIEPY EXPORT THREAD
# ============================================================
class VideoExportLogger(ProgressBarLogger):
    def __init__(self, callback):
        super().__init__(); self.callback = callback
    
    def bars(self, *args, **kwargs):
        items = kwargs.get('items', args[3] if len(args) > 3 else None)
        completed = kwargs.get('completed', args[4] if len(args) > 4 else None)
        try:
            if items and items > 0:
                self.callback(int((completed / items) * 100))
        except Exception:
            pass

class ExportWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)
    _codec_cache: Optional[Tuple[str, str]] = None

    def __init__(self, segments, output_path, preset="medium", is_youtube=False, yt_resolution=None, yt_fps=None, yt_speed="medium", target_aspect="16:9", transition_type="hard_cut", transition_duration=0.0, normalize_audio=False, segment_speeds=None, segment_ramps=None, text_overlays=None, segment_tracks=None, segment_filters=None, segment_ken_burns=None, segment_jcuts=None, segment_lcuts=None, bgm_path=None, bgm_volume=0.2, duck_audio=False):
        super().__init__()
        self.segments = segments
        self.output_path = output_path
        self.preset, self.is_youtube = preset, is_youtube
        self.yt_resolution, self.yt_fps, self.yt_speed = yt_resolution, yt_fps, yt_speed
        self.target_aspect = target_aspect
        self.transition_type = transition_type
        self.transition_duration = transition_duration
        self.normalize_audio = normalize_audio
        self.segment_speeds = segment_speeds or {}
        self.segment_ramps = segment_ramps or {}
        self.text_overlays = text_overlays or {}
        self.segment_tracks = segment_tracks or {}
        self.segment_filters = segment_filters or {}
        self.segment_ken_burns = segment_ken_burns or {}
        self.segment_jcuts = segment_jcuts or {}
        self.segment_lcuts = segment_lcuts or {}
        self.bgm_path = bgm_path
        self.bgm_volume = bgm_volume
        self.duck_audio = duck_audio

    def _pick_codec(self):
        if ExportWorker._codec_cache is not None: return ExportWorker._codec_cache
        ffmpeg = shutil.which("ffmpeg")
        chosen = ("libx264", "aac")
        if ffmpeg:
            try:
                res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True, timeout=10)
                priority = ("h264_nvenc", "h264_qsv", "h264_vaapi", "h264_videotoolbox")
                available = set()
                for line in res.stdout.splitlines():
                    s = line.strip()
                    if not s or s.startswith((";", "Codecs:")): continue
                    for hw in priority:
                        if hw in s: available.add(hw)
                for hw in priority:
                    if hw in available and self._probe_encoder(ffmpeg, hw):
                        chosen = (hw, "aac"); break
            except Exception: pass
        ExportWorker._codec_cache = chosen
        return chosen

    @staticmethod
    def _probe_encoder(ffmpeg: str, codec: str) -> bool:
        try:
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1", "-frames:v", "1", "-an", "-c:v", codec, "-f", "null", "-"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
            return True
        except Exception: return False

    def _render_text_overlay(self, sc, overlay_data, target_w, target_h):
        text, position, duration_pct, style = overlay_data
        if not text: return sc
        try:
            try:
                from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
            except ImportError:
                from moviepy import TextClip, ColorClip, CompositeVideoClip
        except ImportError:
            log.warning("TextClip not available, skipping text overlay."); return sc

        clip_duration = sc.duration
        if duration_pct == 100: txt_start, txt_end = 0, clip_duration
        elif duration_pct == 3: txt_start, txt_end = 0, min(3.0, clip_duration)
        elif duration_pct == -3: txt_start, txt_end = max(0, clip_duration - 3.0), clip_duration
        elif duration_pct == 50: txt_start, txt_end = 0, clip_duration / 2
        elif duration_pct == -50: txt_start, txt_end = clip_duration / 2, clip_duration
        else: txt_start, txt_end = 0, clip_duration

        font_size = max(24, int(target_h * 0.05))
        txt_clip = None
        try:
            kwargs = {'txt': text, 'fontsize': font_size, 'color': 'white', 'font': 'Arial-Bold', 'stroke_color': 'black', 'stroke_width': 2, 'method': 'caption', 'size': (int(target_w * 0.85), None)}
            try: txt_clip = TextClip(text=text, font_size=font_size, color='white', font='Arial-Bold', stroke_color='black', stroke_width=2, method='caption', size=(int(target_w * 0.85), None))
            except TypeError: txt_clip = TextClip(**kwargs)
        except Exception:
            try: txt_clip = TextClip(text=text, fontsize=font_size, color='white', font='Arial-Bold', stroke_color='black', stroke_width=2)
            except Exception as e:
                log.warning(f"TextClip creation failed: {e}"); return sc

        txt_clip = _with_duration(txt_clip, txt_end - txt_start)
        txt_clip = _with_start(txt_clip, txt_start)

        if position == "top": txt_clip = _with_position(txt_clip, ('center', 0.05), relative=True)
        elif position == "bottom": txt_clip = _with_position(txt_clip, ('center', 0.85), relative=True)
        elif position == "center": txt_clip = _with_position(txt_clip, 'center', relative=True)
        elif position == "top-left": txt_clip = _with_position(txt_clip, (0.05, 0.05), relative=True)
        elif position == "bottom-left": txt_clip = _with_position(txt_clip, (0.05, 0.85), relative=True)

        if style == "bar":
            bar_h = int(target_h * 0.12)
            bar_clip = ColorClip(size=(target_w, bar_h), color=(0, 0, 0))
            bar_clip = _with_duration(bar_clip, txt_end - txt_start)
            bar_clip = _with_start(bar_clip, txt_start)
            if position == "top": bar_clip = _with_position(bar_clip, (0, 0))
            else: bar_clip = _with_position(bar_clip, (0, target_h - bar_h))
            return CompositeVideoClip([sc, bar_clip, txt_clip])
        return CompositeVideoClip([sc, txt_clip])

    def run(self):
        try:
            try:
                from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, CompositeAudioClip, AudioFileClip
                from moviepy.video.fx import all as vfx
            except ImportError:
                from moviepy import VideoFileClip, concatenate_videoclips, CompositeVideoClip, CompositeAudioClip, AudioFileClip
                try: from moviepy.video.fx import all as vfx
                except: vfx = None
            
        except ImportError as e:
            self.error.emit(f"MoviePy not installed: {e}"); return
        
        clip_cache, main_clips, broll_clips, audio_clips = [], [], [], []
        final_clip = None
        abs_time = 0.0
        
        try:
            first_path = self.segments[0][0]
            probe = VideoFileClip(first_path)
            target_fps = max(1.0, probe.fps or 24.0)
            probe.close()

            for i, (path, start, end) in enumerate(self.segments):
                if end <= start: continue
                if path not in clip_cache: clip_cache[path] = VideoFileClip(path)
                base = clip_cache[path]
                sc = _get_subclip(base, start, end)

                track = self.segment_tracks.get((path, start, end), 1)
                speed = self.segment_speeds.get((path, start, end), 1.0)
                ramp = self.segment_ramps.get((path, start, end), False)

                if speed != 1.0:
                    if ramp:
                        duration = sc.duration
                        def ramp_t(t, dur=duration, sp=speed):
                            if t < 0.2 * dur: f = t / (0.2 * dur); cs = 1.0 + (sp - 1.0) * f
                            elif t > 0.8 * dur: f = (dur - t) / (0.2 * dur); cs = 1.0 + (sp - 1.0) * f
                            else: cs = sp
                            return t + (cs - 1.0) * 0.05
                        try: sc = sc.with_time_transform(ramp_t) if hasattr(sc, "with_time_transform") else sc.time_transform(ramp_t)
                        except Exception: sc = _with_speed(sc, speed)
                    else: sc = _with_speed(sc, speed)

                filt = self.segment_filters.get((path, start, end))
                if filt and filt != "None": sc = apply_lut_to_clip(sc, filt)
                    
                kb = self.segment_ken_burns.get((path, start, end))
                if kb and kb != "None": sc = apply_ken_burns(sc, kb)

                if self.is_youtube and self.yt_resolution:
                    target_w, target_h = self.yt_resolution
                else:
                    if self.target_aspect == "9:16": target_w, target_h = 1080, 1920
                    elif self.target_aspect == "1:1": target_w, target_h = 1080, 1080
                    else: target_w, target_h = 1920, 1080

                cw, ch = sc.size
                if cw / ch > target_w / target_h:
                    sc = _resize_clip(sc, height=target_h)
                    sc = _crop_clip(sc, x_center=sc.size[0] / 2, width=target_w)
                else:
                    sc = _resize_clip(sc, width=target_w)
                    sc = _crop_clip(sc, y_center=sc.size[1] / 2, height=target_h)

                # J/L CUTS
                jcut = self.segment_jcuts.get((path, start, end), 0.0)
                lcut = self.segment_lcuts.get((path, start, end), 0.0)
                if (jcut > 0 or lcut > 0) and base.audio:
                    a_start = max(0, start - jcut)
                    a_end = min(base.duration, end + lcut)
                    a_clip = _get_subclip(base, a_start, a_end)
                    a_clip = _with_start(a_clip, abs_time - jcut)
                    audio_clips.append(a_clip)
                    try: sc = sc.without_audio()
                    except: pass

                if self.normalize_audio and sc.audio:
                    try:
                        max_vol = sc.audio.max_volume()
                        if max_vol > 0.01: sc = _with_volume(sc, 0.9 / max_vol)
                    except: pass

                overlay_data = self.text_overlays.get((path, start, end))
                if overlay_data and overlay_data[0]: sc = self._render_text_overlay(sc, overlay_data, target_w, target_h)

                if track == 2:
                    broll_clips.append((sc.copy(), abs_time))
                    sc = _with_volume(sc, 0.0) 
                else:
                    if i > 0 and self.transition_type != "hard_cut":
                        dur = self.transition_duration
                        if self.transition_type == "crossfade":
                            try: sc = sc.crossfadein(dur)
                            except:
                                if vfx and hasattr(vfx, "CrossFadeIn"): sc = sc.with_effects([vfx.CrossFadeIn(dur)])
                        elif self.transition_type == "slide":
                            w, h = sc.size
                            sc = _with_position(sc, lambda t: (0, 0) if t >= dur else (w * (1 - t / dur), 0))
                        elif self.transition_type == "zoom":
                            try: sc = sc.crossfadein(dur)
                            except:
                                if vfx and hasattr(vfx, "CrossFadeIn"): sc = sc.with_effects([vfx.CrossFadeIn(dur)])
                            w, h = sc.size
                            def zoom_fn(get_frame, t):
                                frame = get_frame(t)
                                if t >= dur: return frame
                                scale = 1.0 + (0.1 * (t / dur))
                                nw, nh = int(w * scale), int(h * scale)
                                r = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
                                return r[nh//2 - h//2 : nh//2 + h//2, nw//2 - w//2 : nw//2 + w//2]
                            sc = sc.fl_image(zoom_fn)

                if self.is_youtube: sc = _with_fps(sc, float(self.yt_fps))
                elif abs((sc.fps or 24.0) - target_fps) > 0.01: sc = _with_fps(sc, target_fps)

                main_clips.append((sc, abs_time))
                abs_time += sc.duration
                if i < len(self.segments) - 1 and self.transition_type != "hard_cut" and track == 1:
                    abs_time -= self.transition_duration

            if not main_clips: self.error.emit("No valid segments to export."); return

            composite_clips = [_with_start(clip, start_time) for clip, start_time in main_clips]
            final_clip = CompositeVideoClip(composite_clips, size=(target_w, target_h))

            if broll_clips:
                composite_layers = [final_clip]
                for broll_clip, start_time in broll_clips:
                    broll_w = int(target_w * 0.4)
                    broll_resized = _resize_clip(broll_clip, width=broll_w)
                    pos_x = target_w - broll_w - int(target_w * 0.02)
                    pos_y = target_h - broll_resized.h - int(target_h * 0.02)
                    broll_resized = _with_start(broll_resized, start_time)
                    broll_resized = _with_position(broll_resized, (pos_x, pos_y))
                    composite_layers.append(broll_resized)
                final_clip = CompositeVideoClip(composite_layers, size=(target_w, target_h))

            # Audio Compositing (BGM + Ducking)
            final_audio_clips = audio_clips + [final_clip.audio]
            
            if self.bgm_path and os.path.isfile(self.bgm_path):
                bgm = AudioFileClip(self.bgm_path)
                if bgm.duration < final_clip.duration:
                    try: 
                        if vfx and hasattr(vfx, "loop"): bgm = bgm.fx(vfx.loop, duration=final_clip.duration)
                    except: pass
                bgm = _with_duration(bgm, final_clip.duration)
                bgm = _with_volume(bgm, self.bgm_volume)
                
                if self.duck_audio:
                    try:
                        arr = final_clip.audio.to_soundarray(fps=22050)
                        if arr.ndim > 1: arr = arr.mean(axis=1)
                        chunk = 2205
                        arr = arr[: (len(arr)//chunk)*chunk]
                        rms = np.sqrt(np.mean((arr[:len(arr)]).reshape(-1, chunk)**2, axis=1))
                        db_arr = 20 * np.log10(rms + 1e-6)
                        
                        def duck_vol(t):
                            idx = int(t * 10)
                            if idx < len(db_arr) and db_arr[idx] > -25.0:
                                return 0.3
                            return 1.0
                            
                        if hasattr(bgm, "with_volume_scaled"):
                            bgm = bgm.with_volume_scaled(duck_vol)
                        else:
                            bgm = bgm.volumex(duck_vol)
                    except Exception as e:
                        log.warning(f"Ducking analysis failed: {e}")
                        
                final_audio_clips.append(bgm)
                
            final_audio = CompositeAudioClip(final_audio_clips)
            final_clip = final_clip.with_audio(final_audio)

            logger = VideoExportLogger(lambda p: self.progress.emit(p))
            vcodec, acodec = self._pick_codec()
            
            write_kwargs = {
                "codec": vcodec, "audio_codec": acodec, "logger": logger,
                "threads": max(2, os.cpu_count() or 4),
                "ffmpeg_params": ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            }
            if self.is_youtube:
                res_w, res_h = self.yt_resolution; fps = self.yt_fps
                if res_h == 1080: bitrate = "12000k" if fps == 60 else "8000k"
                elif res_h == 1440: bitrate = "24000k" if fps == 60 else "16000k"
                else: bitrate = "60000k" if fps == 60 else "40000k"
                write_kwargs["bitrate"] = bitrate; write_kwargs["audio_bitrate"] = "256k"
                write_kwargs["ffmpeg_params"].extend(["-profile:v", "high", "-level", "5.1" if res_h == 2160 else "4.2", "-maxrate", bitrate, "-bufsize", f"{int(bitrate[:-1])*2}k"])
                if vcodec == "libx264": write_kwargs["preset"] = self.yt_speed
            else:
                if vcodec == "libx264": write_kwargs["preset"] = self.preset

            final_clip.write_videofile(self.output_path, **write_kwargs)
            self.finished.emit(self.output_path)
        except Exception as e:
            log.exception("Export failed")
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            for c in clip_cache.values():
                try: c.close()
                except: pass
            if final_clip is not None:
                try: final_clip.close()
                except: pass

# ============================================================
# COMPUTER VISION WORKER THREADS
# ============================================================
class CVWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    status = Signal(str)
    segment_found = Signal(str, float, float, str, object, str)
    frame_preview = Signal(object, bool)

    _face_cascade: Optional["cv2.CascadeClassifier"] = None
    _hog_descriptor: Optional["cv2.HOGDescriptor"] = None

    def __init__(self, video_paths: List[str], mode_key: str, threshold: float, min_seg_len_sec: float,
                 frame_skip: int, use_clahe: bool, roi_rect: Optional[Tuple[int, int, int, int]],
                 audio_filter: bool, audio_thresh_db: float, reference_frame_rgb: Optional[np.ndarray] = None,
                 ignore_rois: Optional[List[Tuple[int, int, int, int]]] = None, invert_match: bool = False,
                 rl_start_frame: Optional[np.ndarray] = None, rl_end_frame: Optional[np.ndarray] = None):
        super().__init__()
        self.video_paths, self.mode_key, self.threshold = video_paths, mode_key, threshold
        self.min_seg_len_sec, self.frame_skip, self.use_clahe = min_seg_len_sec, max(1, frame_skip), use_clahe
        self.roi_rect, self.audio_filter, self.audio_thresh_db = roi_rect, audio_filter, audio_thresh_db
        self.reference_frame_rgb = reference_frame_rgb
        self.ignore_rois = ignore_rois or []
        self.invert_match = invert_match
        self.rl_start_frame, self.rl_end_frame = rl_start_frame, rl_end_frame
        self._cancelled = False
        self.audio_db_per_frame = np.array([])
        self.current_path = ""

    def cancel(self): self._cancelled = True

    @classmethod
    def get_face_cascade(cls):
        if cls._face_cascade is None:
            for p in [HAARCASCADE_PATH, os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'), '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml']:
                if os.path.isfile(p):
                    cascade = cv2.CascadeClassifier(p)
                    if not cascade.empty(): cls._face_cascade = cascade; break
        return cls._face_cascade

    @classmethod
    def get_hog(cls):
        if cls._hog_descriptor is None:
            hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())
            cls._hog_descriptor = hog
        return cls._hog_descriptor

    def _extract_audio_data(self) -> bool:
        if not self.audio_filter: return True
        self.status.emit(f"Extracting audio for {os.path.basename(self.current_path)}...")
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            self.status.emit("ffmpeg not found in PATH; audio filter disabled.")
            self.audio_filter = False; return False
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        try:
            subprocess.run([ffmpeg_path, "-y", "-i", self.current_path, "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", tmp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            cap = cv2.VideoCapture(self.current_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            with wave.open(tmp.name, "rb") as wf:
                framerate, n_frames = wf.getframerate(), wf.getnframes()
                raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            win = max(1, int(framerate / fps))
            n_video = max(total_frames + 1, len(samples) // win + 1)
            db = np.full(n_video, -60.0, dtype=np.float32)
            trim_len = (len(samples) // win) * win
            if trim_len > 0:
                chunks = samples[:trim_len].reshape(-1, win)
                rms = np.sqrt(np.mean((chunks / 32768.0) ** 2, axis=1))
                db[:len(rms)] = 20 * np.log10(rms + 1e-6)
            self.audio_db_per_frame = db; return True
        except Exception as e:
            self.status.emit(f"Audio extraction failed: {e}. Filter disabled.")
            self.audio_filter = False; return False
        finally:
            try: os.remove(tmp.name)
            except OSError: pass

    def _passes_audio_filter(self, frame_idx: int) -> bool:
        if not self.audio_filter: return True
        if frame_idx >= len(self.audio_db_per_frame): return True
        return self.audio_db_per_frame[frame_idx] >= self.audio_thresh_db

    def run(self):
        total_files = len(self.video_paths)
        for i, path in enumerate(self.video_paths):
            if self._cancelled: break
            self.current_path = path
            self.status.emit(f"Processing {i+1}/{total_files}: {os.path.basename(path)}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened(): continue
            self.audio_db_per_frame = np.array([])
            if self.audio_filter: self._extract_audio_data()
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            try: self._run_analysis(cap, fps, total_frames, i, total_files)
            except Exception as e: self.finished.emit(f"Error on {os.path.basename(path)}:\n{e}")
            finally: cap.release()
        if self._cancelled: self.finished.emit("Cancelled by user.")
        else: self.finished.emit(f"Batch CV Analysis complete! Processed {total_files} files.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.ignore_rois:
            fh, fw = frame.shape[:2]
            for mx, my, mw, mh in self.ignore_rois:
                frame[max(0,my):min(fh,my+mh), max(0,mx):min(fw,mx+mw)] = 0
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            fh, fw = frame.shape[:2]
            x, y = max(0, min(x, fw - 1)), max(0, min(y, fh - 1))
            w, h = max(1, min(w, fw - x)), max(1, min(h, fh - y))
            frame = frame[y:y+h, x:x+w]
        return frame

    def _emit_segment(self, start_frame, end_frame, name, frame_thumb_bgr, fps, cat="Default"):
        start_sec, end_sec = start_frame / fps, end_frame / fps
        thumb = self._process_frame(frame_thumb_bgr)
        thumb_rgb = np.ascontiguousarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        self.segment_found.emit(self.current_path, start_sec, end_sec, name, thumb_rgb, cat)

    def _run_analysis(self, cap, fps, total_frames, file_idx, total_files):
        ret, prev_frame = cap.read()
        if not ret: return
        prev_frame_orig = prev_frame.copy()
        prev_frame = self._process_frame(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_count, event_start_frame, event_start_orig, is_in_event, event_count = 0, 0, None, False, 1
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if self.use_clahe else None
        face_cascade = self.get_face_cascade() if self.mode_key in ("face", "talkhead") else None
        hog = self.get_hog() if self.mode_key == "person" else None
        min_seg_len_frames = max(1, int(self.min_seg_len_sec * fps))

        ref_hist, ref_des, orb_detector, bf_matcher = None, None, None, None
        start_hist, start_des, end_hist, end_des = None, None, None, None

        def extract_feats(frame_rgb):
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            roi = self._process_frame(bgr)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            if clahe is not None: gray = clahe.apply(gray)
            _, des = orb_detector.detectAndCompute(gray, None)
            return hist, des

        if self.mode_key == "template" and self.reference_frame_rgb is not None:
            orb_detector = cv2.ORB_create(nfeatures=500); bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            ref_hist, ref_des = extract_feats(self.reference_frame_rgb)
        elif self.mode_key == "rl_episode" and self.rl_start_frame is not None and self.rl_end_frame is not None:
            orb_detector = cv2.ORB_create(nfeatures=500); bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            start_hist, start_des = extract_feats(self.rl_start_frame)
            end_hist, end_des = extract_feats(self.rl_end_frame)
            self.status.emit("🤖 RL Mode: Start/End templates locked.")

        while True:
            if self._cancelled: return
            if self.frame_skip > 1:
                for _ in range(self.frame_skip - 1):
                    if not cap.grab(): break
                    frame_count += 1
            ret, curr_frame = cap.read()
            if not ret: break
            frame_count += 1

            curr_frame_roi = self._process_frame(curr_frame)
            curr_gray = cv2.cvtColor(curr_frame_roi, cv2.COLOR_BGR2GRAY)
            if clahe is not None: curr_gray = clahe.apply(curr_gray)

            triggered, valid_audio = False, self._passes_audio_filter(frame_count)

            if self.mode_key == "scene":
                diff = cv2.absdiff(prev_gray, curr_gray)
                if np.count_nonzero(diff > 30) / diff.size > self.threshold: triggered = True
            elif self.mode_key == "motion":
                diff = cv2.absdiff(cv2.GaussianBlur(prev_frame, (21, 21), 0), cv2.GaussianBlur(curr_frame_roi, (21, 21), 0))
                _, thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                if np.count_nonzero(thresh_img) / diff.size > self.threshold: triggered = True
            elif self.mode_key == "face":
                if face_cascade and len(face_cascade.detectMultiScale(curr_gray, 1.3, 5)) > 0: triggered = True
            elif self.mode_key == "fade":
                if np.mean(curr_gray) < self.threshold: triggered = True
            elif self.mode_key == "color":
                hp = cv2.calcHist([cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hc = cv2.calcHist([cv2.cvtColor(curr_frame_roi, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hp, hp, 0, 1, cv2.NORM_MINMAX); cv2.normalize(hc, hc, 0, 1, cv2.NORM_MINMAX)
                if cv2.compareHist(hp, hc, cv2.HISTCMP_BHATTACHARYYA) > self.threshold: triggered = True
            elif self.mode_key == "action":
                flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev_gray, (160, 90)), cv2.resize(curr_gray, (160, 90)), None, 0.5, 2, 15, 2, 5, 1.1, 0)
                if cv2.magnitude(flow[..., 0], flow[..., 1]).mean() > self.threshold: triggered = True
            elif self.mode_key == "sharp":
                if cv2.Laplacian(curr_gray, cv2.CV_64F).var() > self.threshold: triggered = True
            elif self.mode_key == "person":
                if hog:
                    scale = 0.4; small = cv2.resize(curr_frame_roi, None, fx=scale, fy=scale)
                    bodies, _ = hog.detectMultiScale(small, winStride=(16, 16), padding=(8, 8), scale=1.1, hitThreshold=0.3)
                    if len(bodies) > 0: triggered = True
            elif self.mode_key == "silence":
                triggered = not self._passes_audio_filter(frame_count); valid_audio = True
            elif self.mode_key == "shorts":
                diff = cv2.absdiff(prev_gray, curr_gray)
                change_ratio = np.count_nonzero(diff > 30) / diff.size
                flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev_gray, (160, 90)), cv2.resize(curr_gray, (160, 90)), None, 0.5, 2, 15, 2, 5, 1.1, 0)
                if cv2.magnitude(flow[..., 0], flow[..., 1]).mean() > self.threshold or change_ratio > self.threshold: triggered = True
            elif self.mode_key == "talkhead":
                if face_cascade:
                    diff = cv2.absdiff(cv2.GaussianBlur(prev_frame, (21, 21), 0), cv2.GaussianBlur(curr_frame_roi, (21, 21), 0))
                    _, thresh_img = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    if np.count_nonzero(thresh_img) / diff.size < self.threshold:
                        if len(face_cascade.detectMultiScale(curr_gray, 1.3, 5)) > 0: triggered = True
            elif self.mode_key == "gameplay":
                flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev_gray, (160, 90)), cv2.resize(curr_gray, (160, 90)), None, 0.5, 2, 15, 2, 5, 1.1, 0)
                if cv2.magnitude(flow[..., 0], flow[..., 1]).mean() > self.threshold and self._passes_audio_filter(frame_count): triggered = True
            elif self.mode_key == "template" and ref_hist is not None:
                curr_hsv = cv2.cvtColor(curr_frame_roi, cv2.COLOR_BGR2HSV)
                curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
                hist_score = max(0.0, cv2.compareHist(ref_hist, curr_hist, cv2.HISTCMP_CORREL))
                feature_score = 0.0
                if ref_des is not None and len(ref_des) > 0:
                    _, des2 = orb_detector.detectAndCompute(curr_gray, None)
                    if des2 is not None and len(des2) > 0:
                        good = [m for m in bf_matcher.match(ref_des, des2) if m.distance < 50]
                        feature_score = min(1.0, len(good) / max(1, len(ref_des)))
                combined = (0.4 * hist_score) + (0.6 * feature_score) if ref_des is not None else hist_score
                if self.invert_match:
                    if combined < self.threshold: triggered = True
                else:
                    if combined > self.threshold: triggered = True
            elif self.mode_key == "rl_episode" and start_hist is not None and end_hist is not None:
                curr_hsv = cv2.cvtColor(curr_frame_roi, cv2.COLOR_BGR2HSV)
                curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
                
                def get_score(ref_h, ref_d):
                    h_score = max(0.0, cv2.compareHist(ref_h, curr_hist, cv2.HISTCMP_CORREL))
                    f_score = 0.0
                    if ref_d is not None and len(ref_d) > 0:
                        _, des2 = orb_detector.detectAndCompute(curr_gray, None)
                        if des2 is not None and len(des2) > 0:
                            good = [m for m in bf_matcher.match(ref_d, des2) if m.distance < 50]
                            f_score = min(1.0, len(good) / max(1, len(ref_d)))
                    return (0.4 * h_score) + (0.6 * f_score) if ref_d is not None else h_score

                start_combined = get_score(start_hist, start_des)
                end_combined = get_score(end_hist, end_des)

                if not is_in_event:
                    if start_combined > self.threshold: triggered = True
                else:
                    if end_combined > self.threshold: triggered = False
                    else: triggered = True

            if triggered and valid_audio:
                if not is_in_event:
                    is_in_event = True
                    event_start_frame = frame_count
                    event_start_orig = curr_frame.copy()
            else:
                if is_in_event:
                    duration = frame_count - event_start_frame
                    if duration >= min_seg_len_frames:
                        cat = "Default"
                        if self.mode_key in ("shorts", "action", "gameplay"): cat = "🔥 Action"
                        elif self.mode_key == "talkhead": cat = "🗣️ Talking"
                        elif self.mode_key == "silence": cat = "💤 Dead Air"
                        elif self.mode_key == "rl_episode": cat = "🤖 RL Episode"
                        
                        name = f"Auto {self.mode_key} {event_count}"
                        thumb = event_start_orig if event_start_orig is not None else curr_frame
                        self._emit_segment(event_start_frame, frame_count, name, thumb, fps, cat)
                        event_count += 1
                    is_in_event = False; event_start_orig = None

            if frame_count % 10 == 0:
                preview_frame = cv2.resize(curr_frame_roi, (160, 90))
                preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                self.frame_preview.emit(np.ascontiguousarray(preview_rgb), triggered)

            if frame_count % 5 == 0 and total_frames > 0:
                file_progress = (frame_count / total_frames) * (100 / total_files)
                self.progress.emit(min(int((file_idx * (100 / total_files)) + file_progress), 99))
                self.status.emit(f"Analyzing {os.path.basename(self.current_path)}: {int(file_progress)}%")

            prev_frame = curr_frame_roi.copy()
            prev_frame_orig = curr_frame.copy()
            prev_gray = curr_gray.copy()

        if is_in_event and (frame_count - event_start_frame) >= min_seg_len_frames:
            cat = "Default"
            if self.mode_key in ("shorts", "action", "gameplay"): cat = "🔥 Action"
            elif self.mode_key == "talkhead": cat = "🗣️ Talking"
            elif self.mode_key == "silence": cat = "💤 Dead Air"
            elif self.mode_key == "rl_episode": cat = "🤖 RL Episode"
            name = f"Auto {self.mode_key} {event_count}"
            thumb = event_start_orig if event_start_orig is not None else prev_frame_orig
            self._emit_segment(event_start_frame, frame_count, name, thumb, fps, cat)

class SmartMontageWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    status = Signal(str)
    segment_found = Signal(str, float, float, str, object, str)

    def __init__(self, video_path: str, target_duration_sec: float = 60.0, segment_length_sec: float = 5.0, audio_filter: bool = True, audio_thresh_db: float = -20.0):
        super().__init__()
        self.video_path, self.target_duration_sec, self.segment_length_sec = video_path, target_duration_sec, segment_length_sec
        self.audio_filter, self.audio_thresh_db = audio_filter, audio_thresh_db
        self._cancelled = False
        self.audio_db_per_frame = np.array([])

    def cancel(self): self._cancelled = True

    def _extract_audio_data(self):
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path: return False
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        try:
            subprocess.run([ffmpeg_path, "-y", "-i", self.video_path, "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", tmp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            with wave.open(tmp.name, "rb") as wf:
                framerate, n_frames = wf.getframerate(), wf.getnframes()
                raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            win = max(1, int(framerate / fps))
            n_video = max(total_frames + 1, len(samples) // win + 1)
            db = np.full(n_video, -60.0, dtype=np.float32)
            trim_len = (len(samples) // win) * win
            if trim_len > 0:
                chunks = samples[:trim_len].reshape(-1, win)
                rms = np.sqrt(np.mean((chunks / 32768.0) ** 2, axis=1))
                db[:len(rms)] = 20 * np.log10(rms + 1e-6)
            self.audio_db_per_frame = db; return True
        except Exception: return False
        finally:
            try: os.remove(tmp.name)
            except OSError: pass

    def run(self):
        self.status.emit("🧠 Smart Montage: Extracting audio...")
        if self.audio_filter: self._extract_audio_data()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): self.finished.emit("Error: Cannot open video."); return
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        seg_len_frames = int(self.segment_length_sec * fps)
        num_windows = total_frames // seg_len_frames
        self.status.emit(f"🧠 Analyzing {num_windows} candidate moments...")
        face_cascade = CVWorker.get_face_cascade()
        scores, prev_gray = [], None

        for w_idx in range(num_windows):
            if self._cancelled: self.finished.emit("Cancelled."); cap.release(); return
            start_frame = w_idx * seg_len_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            action_score, face_score, audio_score, color_change_score, frame_count, thumb_frame = 0.0, 0.0, 0.0, 0.0, 0, None

            for _ in range(seg_len_frames):
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev_gray, (80, 45)), cv2.resize(gray, (80, 45)), None, 0.5, 2, 15, 2, 5, 1.1, 0)
                    action_score += cv2.magnitude(flow[..., 0], flow[..., 1]).mean()
                if face_cascade and frame_count % 5 == 0:
                    if len(face_cascade.detectMultiScale(gray, 1.3, 5)) > 0: face_score += 1.0
                if len(self.audio_db_per_frame) > start_frame + frame_count:
                    if self.audio_db_per_frame[start_frame + frame_count] > self.audio_thresh_db: audio_score += 1.0
                if prev_gray is not None and frame_count % 3 == 0:
                    color_change_score += cv2.compareHist(cv2.calcHist([prev_gray], [0], None, [64], [0, 256]), cv2.calcHist([gray], [0], None, [64], [0, 256]), cv2.HISTCMP_BHATTACHARYYA)
                prev_gray = gray
                if thumb_frame is None: thumb_frame = frame.copy()

            if frame_count == 0: continue
            action_score /= frame_count
            face_score /= (frame_count / 5.0) if frame_count > 0 else 1
            audio_score /= frame_count
            color_change_score /= (frame_count / 3.0) if frame_count > 0 else 1
            composite = (face_score * 3.0) + (audio_score * 2.0) + (action_score * 1.5) + (color_change_score * 0.5)
            scores.append((composite, start_frame, start_frame + frame_count, thumb_frame))
            self.progress.emit(int((w_idx / num_windows) * 80))
            self.status.emit(f"🧠 Scored moment {w_idx+1}/{num_windows}: {composite:.2f}")

        cap.release()
        if not scores: self.finished.emit("No segments found."); return
        scores.sort(key=lambda x: x[0], reverse=True)
        selected, total_selected_sec = [], 0.0
        for score, s_frame, e_frame, thumb in scores:
            if total_selected_sec >= self.target_duration_sec: break
            selected.append((score, s_frame, e_frame, thumb))
            total_selected_sec += (e_frame - s_frame) / fps
        selected.sort(key=lambda x: x[1])
        self.status.emit(f"🧠 Generating {len(selected)} highlight segments...")
        for idx, (score, s_frame, e_frame, thumb) in enumerate(selected):
            thumb_rgb = np.ascontiguousarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            self.segment_found.emit(self.video_path, s_frame / fps, e_frame / fps, f"Montage Clip {idx+1}", thumb_rgb, "⭐ Highlight")
            self.progress.emit(80 + int((idx / len(selected)) * 20))
        self.finished.emit(f"✅ Smart Montage complete! Generated {len(selected)} clips ({total_selected_sec:.1f}s).")

# ============================================================
# MAIN APPLICATION WINDOW
# ============================================================
class VideoEditorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Multi-Video AI Editor")
        self.resize(1400, 900)
        self.setAcceptDrops(True)
        
        self.video_path, self.start_time, self.end_time = None, None, None
        self.current_frame_data, self.roi_rect, self.current_fps = None, None, 24.0
        self.ignore_rois = []
        self.rl_start_frame, self.rl_end_frame = None, None
        self.bgm_path = None
        
        self.seek_slider_dragging, self.preview_end_ms = False, 0
        self._last_filmstrip_tick, self._last_label_ms = 0, 0
        
        self.cv_thread, self.montage_thread, self.export_thread = None, None, None
        self.undo = SegmentUndoStack()
        self._bulk_loading, self._dirty = False, False
        self._next_seg_id = 0

        self.settings = QSettings("MyLab", "ProMultiVideoEditor")
        self._preview_queue, self._preview_idx = [], 0
        self.thread_pool = QThreadPool(); self.thread_pool.setMaxThreadCount(4)
        self._latest_qimage = None
        
        self.setup_ui()
        self.setup_shortcuts()
        self._setup_menubar()
        self.apply_theme()
        self._restore_settings()
        self._check_environment()
        self._check_for_autosave()
        
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._autosave_project)
        self.autosave_timer.start(AUTOSAVE_INTERVAL_MS)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0); root_layout.setSpacing(0)

        self.main_h_splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(self.main_h_splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 5, 10); left_layout.setSpacing(10)
        
        left_layout.addWidget(QLabel("📁 Media Pool\n(Double-click to load)"))
        self.media_list = QListWidget()
        self.media_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.media_list.setIconSize(QSize(100, 60))
        self.media_list.itemDoubleClicked.connect(self.load_media_from_pool)
        left_layout.addWidget(self.media_list)
        
        self.btn_import = QPushButton("➕ Import Media")
        self.btn_import.clicked.connect(self.import_media)
        left_layout.addWidget(self.btn_import)

        proj_layout = QHBoxLayout()
        self.btn_save_proj = QPushButton("💾 Save")
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_proj = QPushButton("📂 Load")
        self.btn_load_proj.clicked.connect(self.load_project)
        proj_layout.addWidget(self.btn_save_proj); proj_layout.addWidget(self.btn_load_proj)
        left_layout.addLayout(proj_layout)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_manual_tab(), "✂️ Manual")
        self.tabs.addTab(self._create_cv_tab(), "🤖 Auto-Detect")
        left_layout.addWidget(self.tabs, 1)
        self.main_h_splitter.addWidget(left_widget)

        self.center_v_splitter = QSplitter(Qt.Vertical)
        self.main_h_splitter.addWidget(self.center_v_splitter)

        top_center_widget = QWidget()
        top_center_layout = QVBoxLayout(top_center_widget)
        top_center_layout.setContentsMargins(5, 10, 5, 5); top_center_layout.setSpacing(8)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setStyleSheet("background-color: black; border: 1px solid #111;")
        top_center_layout.addWidget(self.video_widget, 1)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.audio_output.setVolume(0.8)
        self.media_player.errorOccurred.connect(self._on_media_error)

        self._video_sink = self.media_player.videoSink()
        self._video_sink.videoFrameChanged.connect(self._on_video_frame_changed)

        self.active_media_label = QLabel("Active Media: None")
        self.active_media_label.setStyleSheet("font-weight: bold; color: #0078d4; padding: 2px;")
        self.active_media_label.setAlignment(Qt.AlignCenter)
        top_center_layout.addWidget(self.active_media_label)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("font-family: Consolas, monospace; font-size: 14px; color: #ddd;")
        top_center_layout.addWidget(self.time_label)
        
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self.set_position)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        top_center_layout.addWidget(self.seek_slider)

        self.filmstrip = FilmstripWidget()
        self.filmstrip.seek_requested.connect(self.media_player.setPosition)
        self.filmstrip.hover_time.connect(self._on_filmstrip_hover)
        top_center_layout.addWidget(self.filmstrip)

        controls_layout = QHBoxLayout()
        self.btn_frame_back = QPushButton("⏮ 1F"); self.btn_back = QPushButton("⏪ -5s")
        self.btn_play = QPushButton("▶ Play"); self.btn_pause = QPushButton("⏸ Pause")
        self.btn_fwd = QPushButton("+5s ⏩"); self.btn_frame_fwd = QPushButton("1F ⏭")

        self.btn_frame_back.clicked.connect(lambda: self.step_frame(-1))
        self.btn_frame_fwd.clicked.connect(lambda: self.step_frame(1))
        self.btn_back.clicked.connect(lambda: self.skip_time(-5000))
        self.btn_fwd.clicked.connect(lambda: self.skip_time(5000))
        self.btn_play.clicked.connect(self.media_player.play)
        self.btn_pause.clicked.connect(self.media_player.pause)

        controls_layout.addStretch()
        for btn in [self.btn_frame_back, self.btn_back, self.btn_play, self.btn_pause, self.btn_fwd, self.btn_frame_fwd]:
            controls_layout.addWidget(btn)
        controls_layout.addStretch()
        top_center_layout.addLayout(controls_layout)

        vol_speed_layout = QHBoxLayout()
        vol_speed_layout.addWidget(QLabel("🔊 Vol:"))
        self.vol_slider = QSlider(Qt.Horizontal); self.vol_slider.setRange(0, 100); self.vol_slider.setValue(80)
        self.vol_slider.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))
        vol_speed_layout.addWidget(self.vol_slider, 2)
        vol_speed_layout.addSpacing(20)
        vol_speed_layout.addWidget(QLabel("⚡ Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x", "4.0x"]); self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        vol_speed_layout.addWidget(self.speed_combo); vol_speed_layout.addStretch()
        top_center_layout.addLayout(vol_speed_layout)
        
        self.center_v_splitter.addWidget(top_center_widget)

        self.bottom_h_splitter = QSplitter(Qt.Horizontal)
        self.center_v_splitter.addWidget(self.bottom_h_splitter)

        timeline_widget = QWidget()
        timeline_layout = QVBoxLayout(timeline_widget)
        timeline_layout.setContentsMargins(5, 5, 5, 10); timeline_layout.setSpacing(6)
        
        timeline_header = QHBoxLayout()
        timeline_header.addWidget(QLabel("🎬 Master Timeline"))
        self.timeline_stats_label = QLabel("Segments: 0 • Total: 00:00.000")
        self.timeline_stats_label.setAlignment(Qt.AlignRight)
        self.timeline_stats_label.setStyleSheet("color: #aaa; font-weight: bold;")
        timeline_header.addStretch(); timeline_header.addWidget(self.timeline_stats_label)
        timeline_layout.addLayout(timeline_header)

        self.segment_list = QListWidget()
        self.segment_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.segment_list.setIconSize(QSize(120, 70))
        self.segment_list.setMovement(QListView.Movement.Free)
        self.segment_list.setFlow(QListView.Flow.LeftToRight)
        self.segment_list.setWrapping(False)
        self.segment_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.segment_list.itemDoubleClicked.connect(self.on_segment_double_clicked)
        self.segment_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.segment_list.customContextMenuRequested.connect(self._show_segment_menu)
        
        self.visual_timeline = TimelineWidget()
        self.visual_timeline.seek_requested.connect(self.media_player.setPosition)
        self.visual_timeline.segment_moved.connect(self._sync_timeline_to_list)
        timeline_layout.addWidget(self.visual_timeline)

        list_btn_row1 = QHBoxLayout()
        self.btn_delete_segment = QPushButton("❌ Remove"); self.btn_delete_segment.clicked.connect(self.remove_segment)
        self.btn_merge = QPushButton("🔗 Merge"); self.btn_merge.clicked.connect(self.merge_overlapping_segments)
        self.btn_clear_all = QPushButton("🗑️ Clear"); self.btn_clear_all.clicked.connect(self.clear_all_segments)
        self.btn_preview = QPushButton("▶ Preview Selected"); self.btn_preview.clicked.connect(self.preview_segment)
        list_btn_row1.addWidget(self.btn_preview); list_btn_row1.addStretch()
        list_btn_row1.addWidget(self.btn_delete_segment); list_btn_row1.addWidget(self.btn_merge); list_btn_row1.addWidget(self.btn_clear_all)
        timeline_layout.addLayout(list_btn_row1)
        
        self.bottom_h_splitter.addWidget(timeline_widget)

        export_widget = QWidget()
        export_layout = QVBoxLayout(export_widget)
        export_layout.setContentsMargins(5, 5, 10, 10); export_layout.setSpacing(6)
        
        export_group = QGroupBox("Export & Tools")
        exp_layout = QVBoxLayout()
        
        preview_btn_layout = QHBoxLayout()
        self.btn_preview_export = QPushButton("🎞️ Preview Export")
        self.btn_preview_export.clicked.connect(self.preview_export)
        preview_btn_layout.addWidget(self.btn_preview_export)
        exp_layout.addLayout(preview_btn_layout)

        self.btn_export = QPushButton("💾 Export Final Video")
        self.btn_export.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 15px; font-size: 15px;")
        self.btn_export.clicked.connect(self.export_video)
        exp_layout.addWidget(self.btn_export)
        
        self.btn_youtube = QPushButton("📤 Export for YouTube")
        self.btn_youtube.setStyleSheet("background-color: #ff0000; color: white; font-weight: bold; padding: 12px; font-size: 14px;")
        self.btn_youtube.clicked.connect(self.export_for_youtube)
        exp_layout.addWidget(self.btn_youtube)

        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Aspect Ratio:"))
        self.btn_ratio_169 = QPushButton("16:9"); self.btn_ratio_916 = QPushButton("9:16"); self.btn_ratio_11 = QPushButton("1:1")
        for btn in [self.btn_ratio_169, self.btn_ratio_916, self.btn_ratio_11]:
            btn.setCheckable(True)
        self.btn_ratio_169.setChecked(True)
        ratio_group = QButtonGroup(self)
        for btn in [self.btn_ratio_169, self.btn_ratio_916, self.btn_ratio_11]:
            ratio_group.addButton(btn)
        ratio_layout.addWidget(self.btn_ratio_169); ratio_layout.addWidget(self.btn_ratio_916); ratio_layout.addWidget(self.btn_ratio_11)
        exp_layout.addLayout(ratio_layout)

        trans_layout = QHBoxLayout()
        trans_layout.addWidget(QLabel("Transitions:"))
        self.trans_combo = QComboBox()
        self.trans_combo.addItems(["✂️ Hard Cut", "✨ Crossfade (0.2s)", "✨ Crossfade (0.5s)", "➡️ Slide Left (0.2s)", "🔍 Zoom In (0.3s)"])
        trans_layout.addWidget(self.trans_combo)
        exp_layout.addLayout(trans_layout)

        self.norm_check = QCheckBox("🔊 Normalize Audio Levels")
        self.norm_check.setChecked(True)
        exp_layout.addWidget(self.norm_check)

        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Quality:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["ultrafast", "fast", "medium", "slow", "YouTube (1080p)"]); self.preset_combo.setCurrentText("medium")
        preset_layout.addWidget(self.preset_combo)
        exp_layout.addLayout(preset_layout)

        bgm_layout = QHBoxLayout()
        self.btn_bgm = QPushButton("🎵 BGM Track")
        self.btn_bgm.clicked.connect(self.set_bgm_track)
        bgm_layout.addWidget(self.btn_bgm)
        self.lbl_bgm = QLabel("None")
        bgm_layout.addWidget(self.lbl_bgm)
        exp_layout.addLayout(bgm_layout)
        
        self.duck_check = QCheckBox("Enable Audio Ducking")
        exp_layout.addWidget(self.duck_check)

        csv_btn_layout = QHBoxLayout()
        self.btn_csv = QPushButton("📊 CSV"); self.btn_csv.clicked.connect(self.export_csv)
        self.btn_srt = QPushButton("📝 SRT"); self.btn_srt.clicked.connect(self.export_srt)
        csv_btn_layout.addWidget(self.btn_csv); csv_btn_layout.addWidget(self.btn_srt)
        exp_layout.addLayout(csv_btn_layout)
        
        extra_tools_layout = QHBoxLayout()
        self.btn_thumb = QPushButton("🖼️ YouTube Thumbnail")
        self.btn_thumb.clicked.connect(self.generate_thumbnail)
        extra_tools_layout.addWidget(self.btn_thumb)
        self.btn_batch_export = QPushButton("⏩ Batch Export")
        self.btn_batch_export.clicked.connect(self.batch_export_video)
        extra_tools_layout.addWidget(self.btn_batch_export)
        exp_layout.addLayout(extra_tools_layout)
        
        exp_layout.addStretch()
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); self.progress_bar.setVisible(False)
        exp_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8a8; padding: 2px;")
        exp_layout.addWidget(self.status_label)
        
        export_group.setLayout(exp_layout)
        export_layout.addWidget(export_group)
        self.bottom_h_splitter.addWidget(export_widget)

        self.main_h_splitter.setSizes([300, 1100])
        self.center_v_splitter.setSizes([600, 300])
        self.bottom_h_splitter.setSizes([800, 300])
        self.main_h_splitter.setStretchFactor(0, 0)
        self.main_h_splitter.setStretchFactor(1, 1)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self._setup_tooltips()

    def _on_video_frame_changed(self, frame):
        if frame.isValid():
            self._latest_qimage = frame.toImage()

    def _setup_tooltips(self):
        self.btn_set_start.setToolTip("Set In point (I)"); self.btn_set_end.setToolTip("Set Out point (O)")
        self.btn_play.setToolTip("Play (Space)"); self.btn_pause.setToolTip("Pause (Space or K)")
        self.btn_back.setToolTip("Back 5s (←)"); self.btn_fwd.setToolTip("Forward 5s (→)")
        self.btn_frame_back.setToolTip("Previous frame (,)"); self.btn_frame_fwd.setToolTip("Next frame (.)")
        self.btn_delete_segment.setToolTip("Remove selected (Delete)"); self.btn_export.setToolTip("Export timeline as MP4 (Ctrl+E)")

    def _create_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        segment_layout = QHBoxLayout()
        self.btn_set_start = QPushButton("[ In ] (I)"); self.btn_set_end = QPushButton("[ Out ] (O)")
        self.btn_add_segment = QPushButton("➕ Add Segment")
        self.btn_set_start.clicked.connect(self.set_start)
        self.btn_set_end.clicked.connect(self.set_end)
        self.btn_add_segment.clicked.connect(self.add_segment)
        segment_layout.addWidget(self.btn_set_start); segment_layout.addWidget(self.btn_set_end); segment_layout.addWidget(self.btn_add_segment)
        layout.addLayout(segment_layout)

        self.lbl_start = QLabel("In: Not Set"); self.lbl_end = QLabel("Out: Not Set")
        self.lbl_start.setStyleSheet("font-weight: bold; color: #ff4c4c;")
        self.lbl_end.setStyleSheet("font-weight: bold; color: #4ca6ff;")
        layout.addWidget(self.lbl_start); layout.addWidget(self.lbl_end)
        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_cv_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content.setStyleSheet("""
            QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 14px; padding-top: 12px; font-weight: bold; color: #aaa; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QRadioButton { padding: 4px; spacing: 8px; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; }
            QRadioButton::indicator::unchecked { border: 2px solid #555; background: #222; }
            QRadioButton::indicator::checked { border: 2px solid #0078d4; background: #0078d4; }
        """)
        scroll.setWidget(scroll_content)
        
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(12)

        format_group = QGroupBox("📺 Target Format Focus")
        f_layout = QHBoxLayout()
        self.btn_fmt_any = QPushButton("🎬 Any Format")
        self.btn_fmt_shorts = QPushButton("📱 Shorts / Reels")
        self.btn_fmt_long = QPushButton("🎙️ Long-Form / Podcast")
        for btn in [self.btn_fmt_any, self.btn_fmt_shorts, self.btn_fmt_long]:
            btn.setCheckable(True)
        self.btn_fmt_any.setChecked(True)
        self.btn_fmt_any.clicked.connect(lambda: self._set_format_focus("any"))
        self.btn_fmt_shorts.clicked.connect(lambda: self._set_format_focus("shorts"))
        self.btn_fmt_long.clicked.connect(lambda: self._set_format_focus("talkhead"))
        f_layout.addWidget(self.btn_fmt_any); f_layout.addWidget(self.btn_fmt_shorts); f_layout.addWidget(self.btn_fmt_long)
        format_group.setLayout(f_layout)
        layout.addWidget(format_group)

        mode_group = QGroupBox("🤖 AI Detection Mode")
        mode_grid = QGridLayout(); mode_grid.setSpacing(10)
        self.mode_group = QButtonGroup(self)
        for idx, spec in enumerate(CV_MODES):
            rb = QRadioButton(f"{spec.icon}  {spec.label}")
            rb.setProperty("mode_key", spec.key)
            rb.setToolTip(f"Key: {spec.key}")
            if spec.key == "scene": rb.setChecked(True)
            self.mode_group.addButton(rb)
            mode_grid.addWidget(rb, idx // 2, idx % 2)
            rb.toggled.connect(self.update_threshold_slider_range)
        mode_group.setLayout(mode_grid)
        layout.addWidget(mode_group)

        self.mode_help_label = QLabel("")
        self.mode_help_label.setWordWrap(True)
        self.mode_help_label.setStyleSheet("color: #0078d4; font-style: italic; padding: 0 10px; font-size: 12px;")
        layout.addWidget(self.mode_help_label)

        settings_group = QGroupBox("⚙️ Analysis Settings")
        s_layout = QGridLayout(); s_layout.setVerticalSpacing(8); s_layout.setHorizontalSpacing(10)
        self.thresh_label = QLabel("Threshold:"); self.thresh_label.setMinimumWidth(110)
        s_layout.addWidget(self.thresh_label, 0, 0)
        self.thresh_slider = QSlider(Qt.Horizontal); self.thresh_slider.setRange(1, 50); self.thresh_slider.setValue(10)
        self.thresh_slider.valueChanged.connect(self.update_threshold_label)
        s_layout.addWidget(self.thresh_slider, 0, 1)
        self.thresh_value_lbl = QLabel("10"); self.thresh_value_lbl.setMinimumWidth(40)
        self.thresh_value_lbl.setAlignment(Qt.AlignCenter)
        self.thresh_value_lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #28a745;")
        s_layout.addWidget(self.thresh_value_lbl, 0, 2)
        s_layout.addWidget(QLabel("Min Duration (s):"), 1, 0)
        self.min_seg_spin = QDoubleSpinBox(); self.min_seg_spin.setRange(0.1, 3600.0); self.min_seg_spin.setValue(1.0); self.min_seg_spin.setSingleStep(0.5)
        s_layout.addWidget(self.min_seg_spin, 1, 1, 1, 2)
        s_layout.addWidget(QLabel("Frame Skip:"), 2, 0)
        self.frame_skip_spin = QSpinBox(); self.frame_skip_spin.setRange(1, 10); self.frame_skip_spin.setValue(1)
        s_layout.addWidget(self.frame_skip_spin, 2, 1, 1, 2)
        self.clahe_check = QCheckBox("🌗 Enhance Contrast (CLAHE)")
        s_layout.addWidget(self.clahe_check, 3, 0, 1, 3)
        settings_group.setLayout(s_layout)
        layout.addWidget(settings_group)

        roi_group = QGroupBox("🎯 Region of Interest (ROI) & RL States")
        roi_layout = QGridLayout()
        self.btn_grab_frame = QPushButton("🎥 Grab Frame")
        self.btn_grab_frame.clicked.connect(self.grab_current_frame)
        self.btn_pick_roi = QPushButton("✂️ Focus ROI"); self.btn_pick_roi.setEnabled(False)
        self.btn_pick_roi.clicked.connect(self.open_visual_picker)
        self.btn_mask_hud = QPushButton("🚫 Mask HUD (Ignore)"); self.btn_mask_hud.setEnabled(False)
        self.btn_mask_hud.clicked.connect(self.open_hud_picker)
        self.btn_mark_start = QPushButton("▶️ Set Start Pose"); self.btn_mark_start.setEnabled(False)
        self.btn_mark_start.setStyleSheet("background-color: #1b6b3a; color: white; font-weight: bold; padding: 8px;")
        self.btn_mark_start.clicked.connect(self._set_rl_start_frame)
        self.btn_mark_end = QPushButton("⏹️ Set End Pose"); self.btn_mark_end.setEnabled(False)
        self.btn_mark_end.setStyleSheet("background-color: #b8400b; color: white; font-weight: bold; padding: 8px;")
        self.btn_mark_end.clicked.connect(self._set_rl_end_frame)
        self.lbl_rl_status = QLabel("Start Pose: Not Set | End Pose: Not Set")
        self.lbl_rl_status.setStyleSheet("color: #aaa; font-weight: bold; padding: 2px;")
        
        roi_layout.addWidget(self.btn_grab_frame, 0, 0); roi_layout.addWidget(self.btn_pick_roi, 0, 1)
        roi_layout.addWidget(self.btn_mask_hud, 1, 0)
        roi_layout.addWidget(self.btn_mark_start, 2, 0); roi_layout.addWidget(self.btn_mark_end, 2, 1)
        roi_layout.addWidget(self.lbl_rl_status, 3, 0, 1, 2)
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)

        self.invert_check = QCheckBox("🔄 Invert Match (Segment when DIFFERENT from grabbed frame)")
        self.invert_check.setStyleSheet("font-weight: bold; color: #17a2b8; padding: 5px;")
        layout.addWidget(self.invert_check)

        audio_group = QGroupBox("🔊 Audio Filter (Optional)")
        aud_layout = QVBoxLayout()
        self.audio_check = QCheckBox("Require Audio Spike (Dead Air Filter)")
        aud_layout.addWidget(self.audio_check)
        aud_slider_layout = QHBoxLayout()
        aud_slider_layout.addWidget(QLabel("Threshold (dB):"))
        self.audio_thresh_slider = QSlider(Qt.Horizontal); self.audio_thresh_slider.setRange(-60, 0); self.audio_thresh_slider.setValue(-15)
        aud_slider_layout.addWidget(self.audio_thresh_slider)
        self.audio_db_label = QLabel("-15 dB"); self.audio_db_label.setMinimumWidth(45)
        self.audio_db_label.setStyleSheet("font-family: monospace; color: #ffc107;")
        self.audio_thresh_slider.valueChanged.connect(lambda v: self.audio_db_label.setText(f"{v} dB"))
        aud_slider_layout.addWidget(self.audio_db_label)
        aud_layout.addLayout(aud_slider_layout)
        audio_group.setLayout(aud_layout)
        layout.addWidget(audio_group)

        montage_group = QGroupBox("🧠 Smart Auto-Montage")
        m_layout = QGridLayout()
        m_layout.addWidget(QLabel("Target Duration (s):"), 0, 0)
        self.montage_dur_spin = QDoubleSpinBox(); self.montage_dur_spin.setRange(5.0, 600.0); self.montage_dur_spin.setValue(60.0); self.montage_dur_spin.setSingleStep(5.0)
        m_layout.addWidget(self.montage_dur_spin, 0, 1)
        m_layout.addWidget(QLabel("Clip Length (s):"), 1, 0)
        self.montage_clip_spin = QDoubleSpinBox(); self.montage_clip_spin.setRange(1.0, 30.0); self.montage_clip_spin.setValue(5.0); self.montage_clip_spin.setSingleStep(0.5)
        m_layout.addWidget(self.montage_clip_spin, 1, 1)
        self.montage_audio_check = QCheckBox("Require audio activity (speech/sound)"); self.montage_audio_check.setChecked(True)
        m_layout.addWidget(self.montage_audio_check, 2, 0, 1, 2)
        self.btn_run_montage = QPushButton("🧠 Generate Smart Montage")
        self.btn_run_montage.setStyleSheet("background-color: #e83e8c; color: white; font-weight: bold; padding: 12px; border-radius: 4px; font-size: 14px;")
        self.btn_run_montage.clicked.connect(self.run_smart_montage)
        m_layout.addWidget(self.btn_run_montage, 3, 0, 1, 2)
        montage_group.setLayout(m_layout)
        layout.addWidget(montage_group)

        layout.addStretch()
        self.btn_run_cv = QPushButton("▶  Analyze Current Video")
        self.btn_run_cv.setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; padding: 12px; border-radius: 4px; font-size: 14px;")
        self.btn_run_cv.clicked.connect(self.run_single_cv_analysis)
        layout.addWidget(self.btn_run_cv)

        self.btn_run_cv_batch = QPushButton("🤖  Batch Analyze Selected Media")
        self.btn_run_cv_batch.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold; padding: 12px; border-radius: 4px; font-size: 14px;")
        self.btn_run_cv_batch.clicked.connect(self.run_batch_cv_analysis)
        layout.addWidget(self.btn_run_cv_batch)

        self.btn_cancel_cv = QPushButton("⏹  Cancel Analysis")
        self.btn_cancel_cv.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 10px; border-radius: 4px;")
        self.btn_cancel_cv.clicked.connect(self.cancel_cv); self.btn_cancel_cv.setEnabled(False)
        layout.addWidget(self.btn_cancel_cv)

        preview_layout = QHBoxLayout()
        self.lbl_cv_preview = QLabel("Live Preview"); self.lbl_cv_preview.setFixedSize(170, 100)
        self.lbl_cv_preview.setAlignment(Qt.AlignCenter)
        self.lbl_cv_preview.setStyleSheet("background-color: #000; color: #555; border: 1px solid #333; font-size: 10px;")
        preview_info_layout = QVBoxLayout()
        preview_info_layout.addWidget(QLabel("👁️ CV Vision:"))
        self.lbl_cv_status = QLabel("Waiting for analysis..."); self.lbl_cv_status.setWordWrap(True)
        self.lbl_cv_status.setStyleSheet("color: #aaa; font-size: 11px;")
        preview_info_layout.addWidget(self.lbl_cv_status); preview_info_layout.addStretch()
        preview_layout.addLayout(preview_info_layout); preview_layout.addWidget(self.lbl_cv_preview)
        layout.addLayout(preview_layout)

        self.update_threshold_slider_range()
        return scroll

    def _setup_menubar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        file_menu.addAction("Import Media...", self.import_media, "Ctrl+Shift+I")
        file_menu.addAction("Save Project", self.save_project, "Ctrl+S")
        file_menu.addAction("Load Project...", self.load_project, "Ctrl+O")
        file_menu.addSeparator()
        file_menu.addAction("Export Video...", self.export_video, "Ctrl+E")
        file_menu.addAction("Export CSV...", self.export_csv)
        file_menu.addAction("Export SRT...", self.export_srt)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, "Ctrl+Q")

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction("Undo", self._undo, "Ctrl+Z")
        edit_menu.addAction("Redo", self._redo, "Ctrl+Y")
        edit_menu.addSeparator()
        edit_menu.addAction("Remove Segment", self.remove_segment, "Delete")
        edit_menu.addAction("Merge Overlaps", self.merge_overlapping_segments)
        edit_menu.addAction("Clear All", self.clear_all_segments)
        
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("Keyboard Shortcuts", self._show_shortcuts)
        self.setStatusBar(QStatusBar(self)); self.statusBar().showMessage("Ready")

    def _show_shortcuts(self):
        dlg = ShortcutsDialog(self); dlg.exec()

    def setup_shortcuts(self):
        def mk(seq, fn):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut); sc.activated.connect(fn); return sc
        mk(Qt.Key_Space, self.toggle_play_pause); mk(Qt.Key_I, self.set_start); mk(Qt.Key_O, self.set_end)
        mk(Qt.Key_Delete, self.remove_segment); mk(Qt.Key_Right, lambda: self.skip_time(5000)); mk(Qt.Key_Left, lambda: self.skip_time(-5000))
        mk(Qt.Key_Comma, lambda: self.step_frame(-1)); mk(Qt.Key_Period, lambda: self.step_frame(1))
        mk("Ctrl+Z", self._undo); mk("Ctrl+Y", self._redo); mk("Ctrl+S", self.save_project)
        mk("Ctrl+O", self.load_project); mk("Ctrl+E", self.export_video)
        mk("J", lambda: self.change_speed("0.5x")); mk("K", self.toggle_play_pause); mk("L", lambda: self.change_speed("2.0x"))
        mk(Qt.Key_BracketLeft, lambda: self._snap_to_boundary(-1)); mk(Qt.Key_BracketRight, lambda: self._snap_to_boundary(1))

    def _undo(self):
        if self.undo.undo(self.segment_list): self._dirty = True; self._refresh_timeline_stats()

    def _redo(self):
        if self.undo.redo(self.segment_list): self._dirty = True; self._refresh_timeline_stats()

    def _snap_to_boundary(self, direction):
        if not self.video_path: return
        pos = self.media_player.position() / 1000.0
        boundaries = []
        for i in range(self.segment_list.count()):
            p, s, e = self.segment_list.item(i).data(Qt.UserRole)
            if p == self.video_path: boundaries.extend([s, e])
        if not boundaries: return
        boundaries.sort()
        if direction == 1:
            nxt = next((b for b in boundaries if b > pos + 0.05), None)
            if nxt is not None: self.media_player.setPosition(int(nxt*1000))
        else:
            prv = next((b for b in reversed(boundaries) if b < pos - 0.05), None)
            if prv is not None: self.media_player.setPosition(int(prv*1000))

    def _current_mode_spec(self) -> CVModeSpec:
        btn = self.mode_group.checkedButton()
        key = btn.property("mode_key") if btn else "scene"
        return next(s for s in CV_MODES if s.key == key)

    def _set_format_focus(self, focus: str):
        self.btn_fmt_any.setChecked(focus == "any"); self.btn_fmt_shorts.setChecked(focus == "shorts"); self.btn_fmt_long.setChecked(focus == "talkhead")
        if focus == "shorts":
            for btn in self.mode_group.buttons():
                if btn.property("mode_key") == "shorts": btn.setChecked(True); break
            self.min_seg_spin.setValue(3.0); self.frame_skip_spin.setValue(1); self.btn_ratio_916.setChecked(True)
            self.statusBar().showMessage("🎯 Shorts Mode: Optimized for high-action, fast-paced vertical clips.")
        elif focus == "talkhead":
            for btn in self.mode_group.buttons():
                if btn.property("mode_key") == "talkhead": btn.setChecked(True); break
            self.min_seg_spin.setValue(20.0); self.frame_skip_spin.setValue(2); self.btn_ratio_169.setChecked(True)
            self.statusBar().showMessage("🎯 Long-Form Mode: Optimized for stable talking heads and low-motion scenes.")
        else:
            for btn in self.mode_group.buttons():
                if btn.property("mode_key") == "scene": btn.setChecked(True); break
            self.min_seg_spin.setValue(1.0); self.frame_skip_spin.setValue(1); self.btn_ratio_169.setChecked(True)
            self.statusBar().showMessage("🎯 General Mode: Balanced scene detection.")

    def update_threshold_slider_range(self):
        spec = self._current_mode_spec()
        lo, hi = spec.slider_range
        self.thresh_slider.setRange(lo, hi); self.thresh_slider.setValue(spec.slider_default)
        help_texts = {
            "scene": "Detects hard cuts by comparing pixel differences between frames.",
            "motion": "Detects moving objects by isolating background changes.",
            "face": "Uses Haar Cascades to find faces in the frame.",
            "fade": "Triggers when the video fades to black or near-black.",
            "color": "Detects scene changes via color histogram shifting.",
            "action": "Uses optical flow to detect high-action/fast-motion segments.",
            "sharp": "Finds frames with the highest focus/sharpness (Laplacian).",
            "person": "Uses HOG descriptors to detect full human bodies.",
            "silence": "Finds segments where the audio drops below the dB threshold.",
            "template": "🎯 Grab a frame, then find every moment in the video that looks like it. Optionally pick an ROI to focus on a face, logo, or region. Uses HSV histogram + ORB keypoint matching.",
            "shorts": "Finds high-energy segments (fast cuts or high motion) ideal for TikTok/Shorts/Reels.",
            "talkhead": "Finds stable shots with a face present and low background motion, ideal for long-form YouTube/podcasts.",
            "gameplay": "🎮 Requires BOTH high visual motion AND an audio spike (gunfire/explosions). Perfect for finding combat highlights in game VODs.",
            "rl_episode": "🤖 Grab a frame of the START pose and the END pose. The AI will segment the video into individual episodes between these two states. Use HUD Masking to ignore moving background elements."
        }
        self.mode_help_label.setText(f"ℹ️ {help_texts.get(spec.key, '')}")
        self.update_threshold_label()
        self.thresh_slider.setEnabled(not spec.needs_audio)
        if spec.key == "silence":
            self.audio_check.setText("Silence mode: find segments where audio is BELOW this dB")
            self.audio_check.setChecked(True); self.audio_check.setEnabled(False)
        else:
            self.audio_check.setText("Require Audio Spike (Dead Air Filter)"); self.audio_check.setEnabled(True)

    def update_threshold_label(self):
        spec = self._current_mode_spec()
        raw = self.thresh_slider.value()
        if spec.needs_audio:
            self.thresh_label.setText("Audio Mode:"); self.thresh_value_lbl.setText("Auto")
            self.thresh_value_lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #ffc107;")
        elif spec.threshold_transform:
            actual = spec.threshold_transform(raw)
            self.thresh_label.setText(f"{spec.label_text}")
            self.thresh_value_lbl.setText(f"{actual:.3f}")
            self.thresh_value_lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #28a745;")
        else:
            self.thresh_label.setText(f"{spec.label_text}"); self.thresh_value_lbl.setText(f"{raw}")
            self.thresh_value_lbl.setStyleSheet("font-family: monospace; font-weight: bold; color: #28a745;")

    def change_speed(self, text):
        try:
            rate = float(text.replace('x', ''))
            self.media_player.setPlaybackRate(rate)
        except ValueError:
            pass
        self.speed_combo.blockSignals(True); self.speed_combo.setCurrentText(text); self.speed_combo.blockSignals(False)

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState: self.media_player.pause()
        else: self.media_player.play()

    def skip_time(self, ms):
        self.preview_end_ms = 0; self._preview_queue.clear()
        self.media_player.setPosition(max(0, self.media_player.position() + ms))

    def step_frame(self, direction):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        self.preview_end_ms = 0; self._preview_queue.clear()
        frame_ms = int(1000 / max(1.0, self.current_fps))
        self.media_player.setPosition(max(0, self.media_player.position() + (direction * frame_ms)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(VIDEO_EXTS): self._add_media_to_pool(path)

    def import_media(self):
        last_dir = self.settings.value("last_media_dir", "")
        files, _ = QFileDialog.getOpenFileNames(self, "Import Video Files", last_dir, "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.m4v *.mpg *.mpeg)")
        if not files: return
        self.settings.setValue("last_media_dir", os.path.dirname(files[0]))
        for file_name in files: self._add_media_to_pool(file_name)
        self.media_list.setCurrentRow(self.media_list.count() - len(files))

    def _add_media_to_pool(self, path: str):
        if not os.path.isfile(path): self.status_label.setText(f"⚠️ File not found: {path}"); return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status_label.setText(f"⚠️ Cannot open (codec?): {os.path.basename(path)}"); cap.release(); return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            self.status_label.setText(f"⚠️ Invalid frame count: {os.path.basename(path)}"); cap.release(); return
        ret, frame = cap.read(); cap.release()
        if ret: pix = rgb_frame_to_pixmap(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 80, 50)
        else: pix = QPixmap()
        item = QListWidgetItem(os.path.basename(path)); item.setIcon(pix)
        item.setData(Qt.UserRole, path); item.setToolTip(path)
        self.media_list.addItem(item)

    def _safe_set_enabled(self, widget, enabled: bool):
        """Safely set enabled state, surviving deleted C++ objects."""
        try:
            if widget is not None:
                widget.setEnabled(enabled)
        except RuntimeError:
            log.warning(f"Attempted to access deleted widget: {widget}")

    def load_media_from_pool(self, item: QListWidgetItem):
        if self.cv_thread and self.cv_thread.isRunning():
            self.cv_thread.cancel(); self.cv_thread.wait(2000)
        file_name = item.data(Qt.UserRole)
        self.video_path = file_name
        self.media_player.setSource(QUrl.fromLocalFile(file_name))
        self.filmstrip.set_media(file_name)
        cap = cv2.VideoCapture(file_name)
        if cap.isOpened(): self.current_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0; cap.release()
        self.roi_rect = None; self.current_frame_data = None
        
        self._safe_set_enabled(self.btn_pick_roi, False)
        self._safe_set_enabled(self.btn_mask_hud, False)
        self._safe_set_enabled(self.btn_mark_start, False)
        self._safe_set_enabled(self.btn_mark_end, False)
        
        self.preview_end_ms = 0; self._preview_queue.clear()
        self.start_time = None; self.end_time = None
        self.lbl_start.setText("In: Not Set"); self.lbl_end.setText("Out: Not Set")
        self.speed_combo.blockSignals(True); self.speed_combo.setCurrentText("1.0x"); self.speed_combo.blockSignals(False)
        self.media_player.setPlaybackRate(1.0)
        self.active_media_label.setText(f"🎬 {os.path.basename(file_name)}")
        self.statusBar().showMessage(f"Loaded: {os.path.basename(file_name)}")
        self.update_filmstrip_markers()

    def set_start(self):
        if not self.video_path: self.status_label.setText("⚠️ Load a video first."); return
        self.start_time = self.media_player.position() / 1000.0
        self.lbl_start.setText(f"In: {format_timestamp(self.start_time)}")

    def set_end(self):
        if not self.video_path: self.status_label.setText("⚠️ Load a video first."); return
        pos = self.media_player.position() / 1000.0
        if self.start_time is not None and pos <= self.start_time:
            self.end_time = self.start_time; self.start_time = pos
            self.lbl_start.setText(f"In: {format_timestamp(self.start_time)}")
            self.lbl_end.setText(f"Out: {format_timestamp(self.end_time)}")
            self.statusBar().showMessage("ℹ️ Swapped In/Out automatically.")
        else:
            self.end_time = pos; self.lbl_end.setText(f"Out: {format_timestamp(self.end_time)}")

    def _refresh_timeline_stats(self):
        total = 0.0
        for i in range(self.segment_list.count()):
            _, s, e = self.segment_list.item(i).data(Qt.UserRole); total += max(0, e - s)
        self.timeline_stats_label.setText(f"Segments: {self.segment_list.count()}  •  Total: {format_timestamp(total)}")
        self.update_filmstrip_markers(); self._refresh_media_pool_badges()

    def _refresh_media_pool_badges(self):
        counts = {}
        for i in range(self.segment_list.count()):
            p, _, _ = self.segment_list.item(i).data(Qt.UserRole); counts[p] = counts.get(p, 0) + 1
        for i in range(self.media_list.count()):
            it = self.media_list.item(i); p = it.data(Qt.UserRole)
            n = counts.get(p, 0); base = os.path.basename(p)
            it.setText(base if n == 0 else f"{base}  [{'·' * min(n, 8)}{'+' if n > 8 else ''}]")

    def update_filmstrip_markers(self):
        markers = []
        timeline_segs = []
        if self.video_path and self.filmstrip.duration_ms > 0:
            for i in range(self.segment_list.count()):
                it = self.segment_list.item(i)
                p, s, e = it.data(Qt.UserRole)
                track = it.data(Qt.UserRole + 5) or 1
                color = it.background().color()
                if p == self.video_path: markers.append((int(s * 1000), int(e * 1000), color))
                timeline_segs.append({'start_ms': int(s*1000), 'end_ms': int(e*1000), 'track': track, 'color': color, 'name': it.data(Qt.UserRole+7) or ""})
        
        self.filmstrip.set_segment_markers(markers)
        self.visual_timeline.set_segments(timeline_segs, self.media_player.duration())

    def _segment_display_text(self, fname, s, e, name=None):
        duration = e - s; base = name if name else f"[{fname}]"
        return f"{base} • {format_timestamp(s)}→{format_timestamp(e)} ({format_timestamp(duration)})"

    def _apply_item_decorations(self, item: QListWidgetItem):
        path, s, e = item.data(Qt.UserRole)
        clean_name = item.data(Qt.UserRole + 7) or f"[{os.path.basename(path)}]"
        base = self._segment_display_text(os.path.basename(path), s, e, clean_name)
        
        prefix = ""
        if item.data(Qt.UserRole + 5) == 2: prefix += "🖼️ "
        overlay = item.data(Qt.UserRole + 4)
        if overlay and overlay[0]: prefix += "📝 "
        item.setText(prefix + base)

    def add_segment(self, media_path=None, start=None, end=None, name=None, thumb_frame=None, thumb_array=None, category="Default"):
        if not getattr(self, "_bulk_loading", False): self.undo.snapshot(self.segment_list)
        mp = media_path if media_path else self.video_path
        s = start if start is not None else self.start_time
        e = end if end is not None else self.end_time
        if s is None or e is None:
            if not getattr(self, "_bulk_loading", False): self.undo.discard_last_snapshot()
            self.show_error("Please set both In and Out points."); return
        if s >= e:
            if not getattr(self, "_bulk_loading", False): self.undo.discard_last_snapshot()
            self.show_error("Start time must be less than End time."); return

        fname = os.path.basename(mp)
        item = QListWidgetItem()
        seg_id = self._next_seg_id
        self._next_seg_id += 1
        
        item.setData(Qt.UserRole, (mp, s, e))
        item.setData(Qt.UserRole + 1, category)
        item.setData(Qt.UserRole + 2, 1.0)
        item.setData(Qt.UserRole + 3, False)
        item.setData(Qt.UserRole + 4, None)
        item.setData(Qt.UserRole + 5, 1)
        item.setData(Qt.UserRole + 6, seg_id)
        item.setData(Qt.UserRole + 7, name)
        item.setData(Qt.UserRole + 8, "None")
        item.setData(Qt.UserRole + 9, "None")
        item.setData(Qt.UserRole + 10, 0.0)
        item.setData(Qt.UserRole + 11, 0.0)
        item.setToolTip(f"File: {mp}\nIn: {s:.3f}s\nOut: {e:.3f}s\nTag: {category}")
        item.setBackground(get_color_for_category(category))
        
        self._apply_item_decorations(item)
        self.segment_list.addItem(item)

        if thumb_array is not None:
            q_img = rgb_frame_to_qimage(np.ascontiguousarray(thumb_array))
            pix = QPixmap.fromImage(q_img).scaled(100, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(pix)
        else:
            t = thumb_frame if thumb_frame is not None else s
            task = ThumbnailTask(mp, t, seg_id, 100, 60)
            task.signals.finished.connect(self._on_thumb_ready); self.thread_pool.start(task)

        self._dirty = True
        if start is None:
            self.start_time = None; self.end_time = None
            self.lbl_start.setText("In: Not Set"); self.lbl_end.setText("Out: Not Set")
        self._refresh_timeline_stats()

    def _on_thumb_ready(self, seg_id, q_img):
        if q_img.isNull(): return
        pix = QPixmap.fromImage(q_img).scaled(100, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        for i in range(self.segment_list.count()):
            item = self.segment_list.item(i)
            if item and item.data(Qt.UserRole + 6) == seg_id:
                item.setIcon(pix)
                break

    def remove_segment(self):
        if not self.segment_list.selectedItems(): return
        self.undo.snapshot(self.segment_list)
        for item in self.segment_list.selectedItems(): self.segment_list.takeItem(self.segment_list.row(item))
        self._dirty = True; self._refresh_timeline_stats()

    def clear_all_segments(self):
        if self.segment_list.count() == 0: return
        if QMessageBox.question(self, "Clear timeline", f"Remove all {self.segment_list.count()} segments?\nThis can be undone with Ctrl+Z.") != QMessageBox.StandardButton.Yes: return
        self.undo.snapshot(self.segment_list); self.segment_list.clear()
        self._dirty = True; self._refresh_timeline_stats()

    def on_segment_double_clicked(self, item: QListWidgetItem):
        path, s, e = item.data(Qt.UserRole)
        category = item.data(Qt.UserRole + 1) or "Default"
        speed = item.data(Qt.UserRole + 2) or 1.0
        ramp = item.data(Qt.UserRole + 3) or False
        track = item.data(Qt.UserRole + 5) or 1
        clean_name = item.data(Qt.UserRole + 7) or ""
        filter_type = item.data(Qt.UserRole + 8) or "None"
        ken_burns = item.data(Qt.UserRole + 9) or "None"
        jcut = item.data(Qt.UserRole + 10) or 0.0
        lcut = item.data(Qt.UserRole + 11) or 0.0
        
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            dlg = EditSegmentDialog(s, e, clean_name, category, speed, track, self, filter_type, ken_burns, jcut, lcut)
            dlg.ramp_check.setChecked(ramp)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                name, new_s, new_e, new_cat, new_speed, new_ramp, new_track, new_filter, new_kb, new_jcut, new_lcut = dlg.values()
                if new_s < new_e:
                    self.undo.snapshot(self.segment_list)
                    item.setData(Qt.UserRole, (path, new_s, new_e))
                    item.setData(Qt.UserRole + 1, new_cat)
                    item.setData(Qt.UserRole + 2, new_speed)
                    item.setData(Qt.UserRole + 3, new_ramp)
                    item.setData(Qt.UserRole + 5, new_track)
                    item.setData(Qt.UserRole + 7, name)
                    item.setData(Qt.UserRole + 8, new_filter)
                    item.setData(Qt.UserRole + 9, new_kb)
                    item.setData(Qt.UserRole + 10, new_jcut)
                    item.setData(Qt.UserRole + 11, new_lcut)
                    item.setBackground(get_color_for_category(new_cat))
                    if new_track == 2: item.setForeground(QColor(255, 200, 0))
                    else: item.setForeground(QColor(255, 255, 255))
                    self._apply_item_decorations(item)
                    self._dirty = True; self._refresh_timeline_stats()
                else: self.show_error("Start must be less than end.")
        else: self.preview_segment()
            
    def _show_segment_menu(self, pos: QPoint):
        item = self.segment_list.itemAt(pos)
        if not item: return
        menu = QMenu(self)
        menu.addAction("✂️ Split at playhead", self._split_at_playhead)
        menu.addAction("✏️ Edit Segment...", lambda: self.on_segment_double_clicked(item))
        menu.addAction("📝 Add/Edit Text Overlay", lambda: self._edit_text_overlay(item))
        current_track = item.data(Qt.UserRole + 5) or 1
        if current_track == 1: menu.addAction("➡️ Move to B-Roll Track", lambda: self._set_track(item, 2))
        else: menu.addAction("⬅️ Move to Main Track", lambda: self._set_track(item, 1))
        menu.addAction("❌ Remove", self.remove_segment)
        menu.exec(self.segment_list.mapToGlobal(pos))

    def _split_at_playhead(self):
        items = self.segment_list.selectedItems()
        if not items: return
        item = items[0]
        path, s, e = item.data(Qt.UserRole)
        pos = self.media_player.position() / 1000.0
        if not (s < pos < e): self.show_error("Playhead must be inside the segment to split."); return
        self.undo.snapshot(self.segment_list)
        item.setData(Qt.UserRole, (path, s, pos))
        self._apply_item_decorations(item)
        self.add_segment(media_path=path, start=pos, end=e, name=item.data(Qt.UserRole + 7))
        self._dirty = True

    def _set_track(self, item, track):
        self.undo.snapshot(self.segment_list)
        item.setData(Qt.UserRole + 5, track)
        if track == 2: item.setForeground(QColor(255, 200, 0))
        else: item.setForeground(QColor(255, 255, 255))
        self._apply_item_decorations(item)
        self._dirty = True

    def _edit_text_overlay(self, item):
        p, s, e = item.data(Qt.UserRole)
        existing = item.data(Qt.UserRole + 4) or ("", "bottom", 100, "bar")
        dlg = TextOverlayDialog(self, *existing)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            text, position, duration, style = dlg.values()
            if text:
                item.setData(Qt.UserRole + 4, (text, position, duration, style))
                self._apply_item_decorations(item)
                self._dirty = True; self.statusBar().showMessage(f"Text overlay added: '{text}'")
            else:
                item.setData(Qt.UserRole + 4, None)
                self._apply_item_decorations(item)

    def find_media_row(self, path: str) -> int:
        for i in range(self.media_list.count()):
            if self.media_list.item(i).data(Qt.UserRole) == path: return i
        return -1

    def preview_segment(self):
        items = self.segment_list.selectedItems()
        if not items: self.show_error("No segment selected."); return
        path, s, e = items[0].data(Qt.UserRole)
        self._preview_queue.clear()
        if self.video_path != path:
            row = self.find_media_row(path)
            if row >= 0:
                self.media_list.setCurrentRow(row)
                QTimer.singleShot(300, lambda: self._play_preview(s, e)); return
            else: self.show_error("Source media not found in pool!"); return
        self._play_preview(s, e)

    def _play_preview(self, s, e):
        self.media_player.setPosition(int(s * 1000)); self.media_player.play(); self.preview_end_ms = int(e * 1000)

    def preview_export(self):
        if self.segment_list.count() == 0: self.show_error("No segments to preview."); return
        self._preview_queue = [self.segment_list.item(i).data(Qt.UserRole) for i in range(self.segment_list.count())]
        self._preview_idx = 0; self._play_next_preview()

    def _play_next_preview(self):
        if self._preview_idx >= len(self._preview_queue): self.statusBar().showMessage("✅ Preview complete."); return
        path, s, e = self._preview_queue[self._preview_idx]
        if self.video_path != path:
            row = self.find_media_row(path)
            if row < 0: self._preview_idx += 1; QTimer.singleShot(50, self._play_next_preview); return
            self.media_list.setCurrentRow(row)
            QTimer.singleShot(300, lambda: self._actually_play_preview(s, e))
        else: self._actually_play_preview(s, e)

    def _actually_play_preview(self, s, e):
        self.statusBar().showMessage(f"▶ Previewing segment {self._preview_idx + 1}/{len(self._preview_queue)}")
        self.media_player.setPosition(int(s * 1000)); self.media_player.play()
        self.preview_end_ms = int(e * 1000); self._preview_idx += 1

    def merge_overlapping_segments(self):
        if self.segment_list.count() < 2: return
        self.undo.snapshot(self.segment_list)
        segs = []
        for i in range(self.segment_list.count()):
            it = self.segment_list.item(i)
            p, s, e = it.data(Qt.UserRole)
            cat = it.data(Qt.UserRole + 1) or "Default"
            spd = it.data(Qt.UserRole + 2) or 1.0
            rmp = it.data(Qt.UserRole + 3) or False
            ovl = it.data(Qt.UserRole + 4)
            trk = it.data(Qt.UserRole + 5) or 1
            name = it.data(Qt.UserRole + 7) or ""
            filt = it.data(Qt.UserRole + 8) or "None"
            kb = it.data(Qt.UserRole + 9) or "None"
            jcut = it.data(Qt.UserRole + 10) or 0.0
            lcut = it.data(Qt.UserRole + 11) or 0.0
            segs.append((p, s, e, name, it.icon(), it.background(), cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut))
        segs.sort(key=lambda x: (x[0], x[1]))
        merged = [segs[0]]
        for p, s, e, name, icon, bg, cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut in segs[1:]:
            lp, ls, le, lname, licon, lbg, lcat, lspd, lrmp, lovl, ltrk, lfilt, lkb, ljcut, llcut = merged[-1]
            if p == lp and s <= le: merged[-1] = (lp, ls, max(le, e), lname, licon, lbg, lcat, lspd, lrmp, lovl, ltrk, lfilt, lkb, ljcut, llcut)
            else: merged.append((p, s, e, name, icon, bg, cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut))
        self.segment_list.clear()
        for p, s, e, name, icon, bg, cat, spd, rmp, ovl, trk, filt, kb, jcut, lcut in merged:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, (p, s, e))
            it.setData(Qt.UserRole + 1, cat); it.setData(Qt.UserRole + 2, spd); it.setData(Qt.UserRole + 3, rmp)
            it.setData(Qt.UserRole + 4, ovl); it.setData(Qt.UserRole + 5, trk)
            it.setData(Qt.UserRole + 7, name); it.setData(Qt.UserRole + 8, filt); it.setData(Qt.UserRole + 9, kb)
            it.setData(Qt.UserRole + 10, jcut); it.setData(Qt.UserRole + 11, lcut)
            if not icon.isNull(): it.setIcon(icon)
            it.setBackground(bg)
            if trk == 2: it.setForeground(QColor(255, 200, 0))
            self.segment_list.addItem(it)
            self._apply_item_decorations(it)
        self.statusBar().showMessage(f"✅ Merged into {len(merged)} segments.")
        self._dirty = True; self._refresh_timeline_stats()

    def grab_current_frame(self):
        if self._latest_qimage is None or self._latest_qimage.isNull():
            self.statusBar().showMessage("⚠️ No frame available yet — press Play briefly."); return
        
        qimg = self._latest_qimage
        if qimg.format() != QImage.Format_RGB888:
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            
        ptr = qimg.bits()
        if hasattr(ptr, "setsize"):
            ptr.setsize(qimg.sizeInBytes())
            
        try:
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(qimg.height(), qimg.width(), 3)
            self.current_frame_data = arr.copy()
            
            self._safe_set_enabled(self.btn_pick_roi, True)
            self._safe_set_enabled(self.btn_mask_hud, True)
            self._safe_set_enabled(self.btn_mark_start, True)
            self._safe_set_enabled(self.btn_mark_end, True)
            
            self.statusBar().showMessage("✅ Frame grabbed. Set Start/End poses, Mask HUD, or pick ROI.")
            self.update_threshold_slider_range()
        except Exception as e:
            log.error(f"Failed to convert grabbed frame: {e}")
            self.statusBar().showMessage("⚠️ Failed to grab frame data.")

    def open_visual_picker(self):
        if self.current_frame_data is None: return
        dialog = VisualPickerDialog(self.current_frame_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.roi_rect = dialog.roi_rect
            if self.roi_rect: self.statusBar().showMessage(f"✅ ROI selected: {self.roi_rect}")
            else: self.statusBar().showMessage("ROI cleared.")

    def open_hud_picker(self):
        if self.current_frame_data is None: return
        dialog = VisualPickerDialog(self.current_frame_data, self, title="Select HUD Area to Ignore (Black Out)")
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.roi_rect:
            self.ignore_rois.append(dialog.roi_rect)
            self.statusBar().showMessage(f"🚫 HUD masked: {dialog.roi_rect}")

    def _set_rl_start_frame(self):
        if self.current_frame_data is None: return
        self.rl_start_frame = self.current_frame_data
        self.lbl_rl_status.setText(f"Start Pose: ✅ Set | End Pose: {'✅ Set' if self.rl_end_frame is not None else 'Not Set'}")
        self.statusBar().showMessage("✅ Start Pose locked.")

    def _set_rl_end_frame(self):
        if self.current_frame_data is None: return
        self.rl_end_frame = self.current_frame_data
        self.lbl_rl_status.setText(f"Start Pose: {'✅ Set' if self.rl_start_frame is not None else 'Not Set'} | End Pose: ✅ Set")
        self.statusBar().showMessage("✅ End Pose locked.")

    def _missing_deps(self) -> List[str]:
        missing = []
        try: import moviepy
        except ImportError: missing.append("moviepy")
        if shutil.which("ffmpeg") is None: missing.append("ffmpeg (on PATH)")
        return missing

    def _prepare_cv_worker(self, paths: List[str]):
        if not paths: self.show_error("No videos selected or loaded."); return None
        spec = self._current_mode_spec()
        thresh = spec.threshold_transform(self.thresh_slider.value()) if spec.threshold_transform else float(self.thresh_slider.value())
        audio_filter = self.audio_check.isChecked() or spec.needs_audio

        reference_frame_rgb, rl_start_frame, rl_end_frame = None, None, None
        if spec.key == "template":
            if self.current_frame_data is None:
                self.show_error("🎯 Reference Match requires a grabbed frame.\n\n1) Seek to shot\n2) Click '🎥 Grab Frame'\n3) Run Analyze."); return None
            reference_frame_rgb = self.current_frame_data
        elif spec.key == "rl_episode":
            if self.rl_start_frame is None or self.rl_end_frame is None:
                self.show_error("🤖 RL Episode Match requires both Start and End frames."); return None
            rl_start_frame = self.rl_start_frame; rl_end_frame = self.rl_end_frame

        worker = CVWorker(
            video_paths=paths, mode_key=spec.key, threshold=thresh,
            min_seg_len_sec=float(self.min_seg_spin.value()), frame_skip=self.frame_skip_spin.value(),
            use_clahe=self.clahe_check.isChecked(), roi_rect=self.roi_rect,
            audio_filter=audio_filter, audio_thresh_db=float(self.audio_thresh_slider.value()),
            reference_frame_rgb=reference_frame_rgb, ignore_rois=self.ignore_rois,
            invert_match=self.invert_check.isChecked(), rl_start_frame=rl_start_frame, rl_end_frame=rl_end_frame
        )
        return worker

    def run_single_cv_analysis(self):
        if not self.video_path: self.show_error("No video loaded in the player."); return
        self.cv_thread = self._prepare_cv_worker([self.video_path])
        if self.cv_thread:
            self.btn_run_cv.setEnabled(False); self.btn_run_cv_batch.setEnabled(False); self.btn_cancel_cv.setEnabled(True)
            self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
            self.cv_thread.progress.connect(self.progress_bar.setValue)
            self.cv_thread.status.connect(lambda s: self.statusBar().showMessage(s))
            self.cv_thread.segment_found.connect(self.on_cv_segment_found)
            self.cv_thread.frame_preview.connect(self._on_cv_preview)
            self.cv_thread.finished.connect(self.on_cv_finished); self.cv_thread.start()

    def run_batch_cv_analysis(self):
        items = self.media_list.selectedItems()
        if len(items) == 0: self.show_error("Select at least one video in the Media Pool to batch analyze."); return
        paths = [item.data(Qt.UserRole) for item in items]
        self.cv_thread = self._prepare_cv_worker(paths)
        if self.cv_thread:
            self.btn_run_cv.setEnabled(False); self.btn_run_cv_batch.setEnabled(False); self.btn_cancel_cv.setEnabled(True)
            self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
            self.cv_thread.progress.connect(self.progress_bar.setValue)
            self.cv_thread.status.connect(lambda s: self.statusBar().showMessage(s))
            self.cv_thread.segment_found.connect(self.on_cv_segment_found)
            self.cv_thread.frame_preview.connect(self._on_cv_preview)
            self.cv_thread.finished.connect(self.on_cv_finished); self.cv_thread.start()

    def run_smart_montage(self):
        if not self.video_path: self.show_error("No video loaded in the player."); return
        if self.segment_list.count() > 0:
            reply = QMessageBox.question(self, "Smart Montage", "This will add new highlight segments. Clear existing segments first?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Cancel: return
            if reply == QMessageBox.StandardButton.Yes: self.clear_all_segments()

        self.montage_thread = SmartMontageWorker(
            video_path=self.video_path, target_duration_sec=float(self.montage_dur_spin.value()),
            segment_length_sec=float(self.montage_clip_spin.value()), audio_filter=self.montage_audio_check.isChecked(),
            audio_thresh_db=float(self.audio_thresh_slider.value())
        )
        self.btn_run_cv.setEnabled(False); self.btn_run_cv_batch.setEnabled(False); self.btn_run_montage.setEnabled(False)
        self.btn_cancel_cv.setEnabled(True); self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.montage_thread.progress.connect(self.progress_bar.setValue)
        self.montage_thread.status.connect(lambda s: self.statusBar().showMessage(s))
        self.montage_thread.segment_found.connect(self.on_cv_segment_found)
        self.montage_thread.finished.connect(self.on_cv_finished); self.montage_thread.start()

    def cancel_cv(self):
        if self.cv_thread and self.cv_thread.isRunning(): self.cv_thread.cancel()
        if hasattr(self, 'montage_thread') and self.montage_thread and self.montage_thread.isRunning(): self.montage_thread.cancel()

    def _on_cv_preview(self, frame_rgb, triggered):
        pix = rgb_frame_to_pixmap(frame_rgb, 160, 90); self.lbl_cv_preview.setPixmap(pix)
        if triggered:
            self.lbl_cv_preview.setStyleSheet("background-color: #000; border: 2px solid #28a745;")
            self.lbl_cv_status.setText("🟢 TRIGGERED\nSegment active.")
        else:
            self.lbl_cv_preview.setStyleSheet("background-color: #000; border: 1px solid #333;")
            self.lbl_cv_status.setText("🔴 Scanning...\nNo trigger met.")

    def on_cv_segment_found(self, media_path, start, end, name, thumb_array, category):
        self.add_segment(media_path=media_path, start=start, end=end, name=name, thumb_frame=start, thumb_array=thumb_array, category=category)

    def on_cv_finished(self, msg):
        self.statusBar().showMessage(msg)
        self.btn_run_cv.setEnabled(True); self.btn_run_cv_batch.setEnabled(True)
        if hasattr(self, 'btn_run_montage'): self.btn_run_montage.setEnabled(True)
        self.btn_cancel_cv.setEnabled(False)
        QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))

    def _autosave_project(self):
        if self.segment_list.count() == 0 and self.media_list.count() == 0:
            if os.path.exists(AUTOSAVE_FILE):
                try: os.remove(AUTOSAVE_FILE)
                except: pass
            return
        data = {
            "media_pool": [self.media_list.item(i).data(Qt.UserRole) for i in range(self.media_list.count())],
            "segments": [], "active_media": self.video_path,
        }
        for i in range(self.segment_list.count()):
            it = self.segment_list.item(i)
            p, s, e = it.data(Qt.UserRole)
            data["segments"].append({"path": p, "start": s, "end": e, "name": it.data(Qt.UserRole + 7) or "", "cat": it.data(Qt.UserRole + 1) or "Default", "speed": it.data(Qt.UserRole + 2) or 1.0, "ramp": it.data(Qt.UserRole + 3) or False, "overlay": it.data(Qt.UserRole + 4), "track": it.data(Qt.UserRole + 5) or 1, "filter": it.data(Qt.UserRole + 8) or "None", "kb": it.data(Qt.UserRole + 9) or "None", "jcut": it.data(Qt.UserRole + 10) or 0.0, "lcut": it.data(Qt.UserRole + 11) or 0.0})
        try:
            tmp_path = AUTOSAVE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            os.replace(tmp_path, AUTOSAVE_FILE)
        except Exception as e: log.warning(f"Autosave failed: {e}")

    def _check_for_autosave(self):
        if not os.path.exists(AUTOSAVE_FILE): return
        reply = QMessageBox.question(self, "Crash Recovery", "An autosave file from a previous session was found.\nDo you want to restore it?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
        if reply == QMessageBox.StandardButton.Yes: self._load_project_from_path(AUTOSAVE_FILE)
        else:
            try: os.remove(AUTOSAVE_FILE)
            except: pass

    def save_project(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", self._last_proj_dir(), PROJECT_EXTS)
        if not path: return False
        data = {
            "media_pool": [self.media_list.item(i).data(Qt.UserRole) for i in range(self.media_list.count())],
            "segments": [], "active_media": self.video_path,
        }
        for i in range(self.segment_list.count()):
            it = self.segment_list.item(i)
            p, s, e = it.data(Qt.UserRole)
            data["segments"].append({"path": p, "start": s, "end": e, "name": it.data(Qt.UserRole + 7) or "", "cat": it.data(Qt.UserRole + 1) or "Default", "speed": it.data(Qt.UserRole + 2) or 1.0, "ramp": it.data(Qt.UserRole + 3) or False, "overlay": it.data(Qt.UserRole + 4), "track": it.data(Qt.UserRole + 5) or 1, "filter": it.data(Qt.UserRole + 8) or "None", "kb": it.data(Qt.UserRole + 9) or "None", "jcut": it.data(Qt.UserRole + 10) or 0.0, "lcut": it.data(Qt.UserRole + 11) or 0.0})
        try:
            with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            self.settings.setValue("last_proj_dir", os.path.dirname(path))
            self._dirty = False
            self.statusBar().showMessage(f"💾 Project saved: {path}"); return True
        except Exception as e:
            self.show_error(f"Save failed: {e}"); return False

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", self._last_proj_dir(), PROJECT_EXTS)
        if not path: return
        self._load_project_from_path(path)

    def _last_proj_dir(self) -> str: return self.settings.value("last_proj_dir", "")

    def _load_project_from_path(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f: data = json.load(f)
            self.media_list.clear(); self.segment_list.clear()
            missing_media = []
            for mp in data.get("media_pool", []):
                if not os.path.isfile(mp): missing_media.append(mp); continue
                self._add_media_to_pool(mp)
            self._bulk_loading = True; skipped_segs = 0
            try:
                for seg in data.get("segments", []):
                    if not os.path.isfile(seg["path"]): skipped_segs += 1; continue
                    self.add_segment(media_path=seg["path"], start=seg["start"], end=seg["end"], name=seg.get("name", ""), category=seg.get("cat", "Default"))
                    item = self.segment_list.item(self.segment_list.count() - 1)
                    if item:
                        item.setData(Qt.UserRole + 2, seg.get("speed", 1.0))
                        item.setData(Qt.UserRole + 3, seg.get("ramp", False))
                        item.setData(Qt.UserRole + 4, seg.get("overlay"))
                        trk = seg.get("track", 1)
                        item.setData(Qt.UserRole + 5, trk)
                        item.setData(Qt.UserRole + 8, seg.get("filter", "None"))
                        item.setData(Qt.UserRole + 9, seg.get("kb", "None"))
                        item.setData(Qt.UserRole + 10, seg.get("jcut", 0.0))
                        item.setData(Qt.UserRole + 11, seg.get("lcut", 0.0))
                        if trk == 2: item.setForeground(QColor(255, 200, 0))
                        self._apply_item_decorations(item)
            finally: self._bulk_loading = False
            self.undo.snapshot(self.segment_list); self._dirty = False
            if data.get("active_media") and os.path.isfile(data["active_media"]):
                row = self.find_media_row(data["active_media"])
                if row >= 0: self.media_list.setCurrentRow(row)
            self.statusBar().showMessage(f"📂 Project loaded: {path}")
            if missing_media or skipped_segs:
                QMessageBox.warning(self, "Missing files", f"{len(missing_media)} media file(s) and {skipped_segs} segment(s) could not be found and were skipped:\n\n" + "\n".join(missing_media[:5]) + ("..." if len(missing_media) > 5 else ""))
        except Exception as e:
            self.show_error(f"Load failed: {e}"); log.exception("Project load failed")

    def _sync_timeline_to_list(self):
        for i, seg_data in enumerate(self.visual_timeline.segments):
            if i < self.segment_list.count():
                item = self.segment_list.item(i)
                p, _, _ = item.data(Qt.UserRole)
                new_s = seg_data['start_ms'] / 1000.0
                new_e = seg_data['end_ms'] / 1000.0
                item.setData(Qt.UserRole, (p, new_s, new_e))
                self._apply_item_decorations(item)
        self._dirty = True
        self._refresh_timeline_stats()

    def set_bgm_track(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Background Music", "", "Audio Files (*.mp3 *.wav *.aac)")
        if path:
            self.bgm_path = path
            self.lbl_bgm.setText(os.path.basename(path))

    def generate_thumbnail(self):
        if self._latest_qimage is None or self._latest_qimage.isNull():
            self.statusBar().showMessage("⚠️ Play video briefly to grab a frame first.")
            return
        qimg = self._latest_qimage.convertToFormat(QImage.Format_RGB888)
        ptr = qimg.bits(); ptr.setsize(qimg.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(qimg.height(), qimg.width(), 3).copy()
        dlg = ThumbnailGeneratorDialog(arr, self)
        dlg.exec()

    def batch_export_video(self):
        if self.segment_list.count() == 0: self.show_error("No segments to export."); return
        dlg = BatchExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        formats = dlg.get_selected()
        if not formats: return
        
        default_dir = self.settings.value("last_export_dir", "")
        base_name = "batch_export_" + time.strftime("%Y%m%d_%H%M%S")
        
        self.btn_export.setEnabled(False); self.btn_batch_export.setEnabled(False)
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self._batch_queue = []
        
        segments, segment_speeds, segment_ramps, text_overlays, segment_tracks = [], {}, {}, {}, {}
        segment_filters, segment_ken_burns, segment_jcuts, segment_lcuts = {}, {}, {}, {}
        for i in range(self.segment_list.count()):
            item = self.segment_list.item(i)
            p, s, e = item.data(Qt.UserRole)
            segments.append((p, s, e))
            if (item.data(Qt.UserRole + 2) or 1.0) != 1.0: segment_speeds[(p,s,e)] = item.data(Qt.UserRole + 2)
            if item.data(Qt.UserRole + 3): segment_ramps[(p,s,e)] = item.data(Qt.UserRole + 3)
            if item.data(Qt.UserRole + 4): text_overlays[(p,s,e)] = item.data(Qt.UserRole + 4)
            segment_tracks[(p,s,e)] = item.data(Qt.UserRole + 5) or 1
            if item.data(Qt.UserRole + 8) and item.data(Qt.UserRole + 8) != "None": segment_filters[(p,s,e)] = item.data(Qt.UserRole + 8)
            if item.data(Qt.UserRole + 9) and item.data(Qt.UserRole + 9) != "None": segment_ken_burns[(p,s,e)] = item.data(Qt.UserRole + 9)
            if item.data(Qt.UserRole + 10) and item.data(Qt.UserRole + 10) > 0: segment_jcuts[(p,s,e)] = item.data(Qt.UserRole + 10)
            if item.data(Qt.UserRole + 11) and item.data(Qt.UserRole + 11) > 0: segment_lcuts[(p,s,e)] = item.data(Qt.UserRole + 11)

        for fmt in formats:
            save_path = os.path.join(default_dir, f"{base_name}_{fmt.replace(':', 'x')}.mp4")
            self._batch_queue.append((fmt, save_path))
            
        self._run_next_batch_export(segments, segment_speeds, segment_ramps, text_overlays, segment_tracks, segment_filters, segment_ken_burns, segment_jcuts, segment_lcuts)

    def _run_next_batch_export(self, segments, speeds, ramps, overlays, tracks, filters, ken_burns, jcuts, lcuts):
        if not self._batch_queue:
            self.btn_export.setEnabled(True); self.btn_batch_export.setEnabled(True)
            self.statusBar().showMessage("✅ Batch export complete!")
            QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
            return
            
        fmt, save_path = self._batch_queue.pop(0)
        self.statusBar().showMessage(f"Batch Exporting {fmt} to {save_path}...")
        
        self.export_thread = ExportWorker(
            segments, save_path, preset=self.preset_combo.currentText(), target_aspect=fmt,
            transition_type="hard_cut", transition_duration=0.0,
            normalize_audio=self.norm_check.isChecked(),
            segment_speeds=speeds, segment_ramps=ramps, text_overlays=overlays, 
            segment_tracks=tracks, segment_filters=filters, segment_ken_burns=ken_burns,
            segment_jcuts=jcuts, segment_lcuts=lcuts, 
            bgm_path=self.bgm_path, bgm_volume=0.2, duck_audio=self.duck_check.isChecked()
        )
        self.export_thread.progress.connect(lambda p: self.progress_bar.setValue(p))
        self.export_thread.finished.connect(lambda p: (self.statusBar().showMessage(f"Saved {p}"), self._run_next_batch_export(segments, speeds, ramps, overlays, tracks, filters, ken_burns, jcuts, lcuts)))
        self.export_thread.error.connect(self.on_export_error)
        self.export_thread.start()

    def export_video(self):
        if self.segment_list.count() == 0: self.show_error("No segments to export."); return
        missing = self._missing_deps()
        if missing:
            sep = '\n  • '
            self.show_error(f"Cannot export — missing dependencies:\n  • {sep.join(missing)}\n\nInstall them and restart the app.")
            return
        default_name = "edit_" + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        default_dir = self.settings.value("last_export_dir", "")
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Video", os.path.join(default_dir, default_name), "MP4 Video (*.mp4)")
        if not save_path: return
        if not save_path.lower().endswith(".mp4"): save_path += ".mp4"
        self.settings.setValue("last_export_dir", os.path.dirname(save_path))

        segments, segment_speeds, segment_ramps, text_overlays, segment_tracks = [], {}, {}, {}, {}
        segment_filters, segment_ken_burns, segment_jcuts, segment_lcuts = {}, {}, {}, {}
        for i in range(self.segment_list.count()):
            item = self.segment_list.item(i)
            p, s, e = item.data(Qt.UserRole)
            segments.append((p, s, e))
            speed = item.data(Qt.UserRole + 2) or 1.0
            ramp = item.data(Qt.UserRole + 3) or False
            overlay = item.data(Qt.UserRole + 4)
            track = item.data(Qt.UserRole + 5) or 1
            filt = item.data(Qt.UserRole + 8) or "None"
            kb = item.data(Qt.UserRole + 9) or "None"
            jcut = item.data(Qt.UserRole + 10) or 0.0
            lcut = item.data(Qt.UserRole + 11) or 0.0
            if speed != 1.0: segment_speeds[(p, s, e)] = speed; segment_ramps[(p, s, e)] = ramp
            if overlay and overlay[0]: text_overlays[(p, s, e)] = overlay
            if filt != "None": segment_filters[(p, s, e)] = filt
            if kb != "None": segment_ken_burns[(p, s, e)] = kb
            if jcut > 0: segment_jcuts[(p, s, e)] = jcut
            if lcut > 0: segment_lcuts[(p, s, e)] = lcut
            segment_tracks[(p, s, e)] = track

        target_aspect = "16:9"
        if self.btn_ratio_916.isChecked(): target_aspect = "9:16"
        elif self.btn_ratio_11.isChecked(): target_aspect = "1:1"
        
        trans_text = self.trans_combo.currentText()
        transition_type = "hard_cut"
        transition_duration = 0.0
        if "Crossfade (0.2s)" in trans_text: transition_type = "crossfade"; transition_duration = 0.2
        elif "Crossfade (0.5s)" in trans_text: transition_type = "crossfade"; transition_duration = 0.5
        elif "Slide Left (0.2s)" in trans_text: transition_type = "slide"; transition_duration = 0.2
        elif "Zoom In (0.3s)" in trans_text: transition_type = "zoom"; transition_duration = 0.3

        self.btn_export.setEnabled(False); self.btn_youtube.setEnabled(False); self.preset_combo.setEnabled(False)
        self.btn_export.setText("Exporting... Please Wait")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)

        self.export_thread = ExportWorker(
            segments, save_path, preset=self.preset_combo.currentText(), target_aspect=target_aspect,
            transition_type=transition_type, transition_duration=transition_duration,
            normalize_audio=self.norm_check.isChecked(),
            segment_speeds=segment_speeds, segment_ramps=segment_ramps, text_overlays=text_overlays, 
            segment_tracks=segment_tracks, segment_filters=segment_filters, segment_ken_burns=segment_ken_burns,
            segment_jcuts=segment_jcuts, segment_lcuts=segment_lcuts,
            bgm_path=self.bgm_path, bgm_volume=0.2, duck_audio=self.duck_check.isChecked()
        )
        self.export_thread.progress.connect(lambda p: (self.progress_bar.setValue(p), self.statusBar().showMessage(f"Exporting: {p}%")))
        self.export_thread.finished.connect(self.on_export_finished)
        self.export_thread.error.connect(self.on_export_error)
        self.export_thread.start()

    def export_for_youtube(self):
        if self.segment_list.count() == 0: self.show_error("No segments to export."); return
        missing = self._missing_deps()
        if missing:
            sep = '\n  • '
            self.show_error(f"Cannot export — missing dependencies:\n  • {sep.join(missing)}\n\nInstall them and restart the app.")
            return
        dlg = YouTubeExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return
        yt_res, yt_fps, yt_speed = dlg.get_settings()
        default_name = f"youtube_{yt_res[1]}p_{yt_fps}fps_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        default_dir = self.settings.value("last_export_dir", "")
        save_path, _ = QFileDialog.getSaveFileName(self, "Save YouTube Video", os.path.join(default_dir, default_name), "MP4 Video (*.mp4)")
        if not save_path: return
        if not save_path.lower().endswith(".mp4"): save_path += ".mp4"
        self.settings.setValue("last_export_dir", os.path.dirname(save_path))

        segments, segment_speeds, segment_ramps, text_overlays, segment_tracks = [], {}, {}, {}, {}
        segment_filters, segment_ken_burns, segment_jcuts, segment_lcuts = {}, {}, {}, {}
        for i in range(self.segment_list.count()):
            item = self.segment_list.item(i)
            p, s, e = item.data(Qt.UserRole)
            segments.append((p, s, e))
            speed = item.data(Qt.UserRole + 2) or 1.0
            ramp = item.data(Qt.UserRole + 3) or False
            overlay = item.data(Qt.UserRole + 4)
            track = item.data(Qt.UserRole + 5) or 1
            filt = item.data(Qt.UserRole + 8) or "None"
            kb = item.data(Qt.UserRole + 9) or "None"
            jcut = item.data(Qt.UserRole + 10) or 0.0
            lcut = item.data(Qt.UserRole + 11) or 0.0
            if speed != 1.0: segment_speeds[(p, s, e)] = speed; segment_ramps[(p, s, e)] = ramp
            if overlay and overlay[0]: text_overlays[(p, s, e)] = overlay
            if filt != "None": segment_filters[(p, s, e)] = filt
            if kb != "None": segment_ken_burns[(p, s, e)] = kb
            if jcut > 0: segment_jcuts[(p, s, e)] = jcut
            if lcut > 0: segment_lcuts[(p, s, e)] = lcut
            segment_tracks[(p, s, e)] = track

        self.btn_export.setEnabled(False); self.btn_youtube.setEnabled(False); self.preset_combo.setEnabled(False)
        self.btn_youtube.setText("Exporting... Please Wait")
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)

        self.export_thread = ExportWorker(
            segments, save_path, is_youtube=True, yt_resolution=yt_res, yt_fps=yt_fps, yt_speed=yt_speed,
            target_aspect="16:9", transition_type="hard_cut", transition_duration=0.0, normalize_audio=self.norm_check.isChecked(),
            segment_speeds=segment_speeds, segment_ramps=segment_ramps, text_overlays=text_overlays, segment_tracks=segment_tracks,
            segment_filters=segment_filters, segment_ken_burns=segment_ken_burns, segment_jcuts=segment_jcuts, segment_lcuts=segment_lcuts,
            bgm_path=self.bgm_path, bgm_volume=0.2, duck_audio=self.duck_check.isChecked()
        )
        self.export_thread.progress.connect(lambda p: (self.progress_bar.setValue(p), self.statusBar().showMessage(f"Exporting: {p}%")))
        self.export_thread.finished.connect(self.on_export_finished)
        self.export_thread.error.connect(self.on_export_error)
        self.export_thread.start()

    def on_export_finished(self, path):
        self.btn_export.setEnabled(True)
        if hasattr(self, 'btn_youtube'): self.btn_youtube.setEnabled(True); self.btn_youtube.setText("📤 Export for YouTube")
        if hasattr(self, 'preset_combo'): self.preset_combo.setEnabled(True)
        self.btn_export.setText("💾 Export Final Video")
        QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))
        QMessageBox.information(self, "Success!", f"Video successfully exported to:\n{path}")

    def on_export_error(self, err):
        self.btn_export.setEnabled(True)
        if hasattr(self, 'btn_youtube'): self.btn_youtube.setEnabled(True); self.btn_youtube.setText("📤 Export for YouTube")
        if hasattr(self, 'preset_combo'): self.preset_combo.setEnabled(True)
        self.btn_export.setText("💾 Export Final Video")
        self.progress_bar.setVisible(False)
        dlg = QMessageBox(self); dlg.setIcon(QMessageBox.Icon.Critical); dlg.setWindowTitle("Export Error")
        dlg.setText("Video export failed."); dlg.setDetailedText(err); dlg.exec()
        log.error("Export failed: %s", err)

    def export_csv(self):
        if self.segment_list.count() == 0: self.show_error("No segments to export."); return
        default_dir = self.settings.value("last_csv_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", os.path.join(default_dir, "segments.csv"), "CSV Files (*.csv)")
        if not path: return
        self.settings.setValue("last_csv_dir", os.path.dirname(path))
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(["File", "Start (s)", "End (s)", "Name"])
                for i in range(self.segment_list.count()):
                    item = self.segment_list.item(i); p, s, e = item.data(Qt.UserRole)
                    writer.writerow([os.path.basename(p), f"{s:.3f}", f"{e:.3f}", item.data(Qt.UserRole + 7) or os.path.basename(p)])
            self.statusBar().showMessage(f"Exported CSV to {path}")
        except Exception as e: self.show_error(f"Failed to export CSV:\n{e}")

    def export_srt(self):
        if self.segment_list.count() == 0: self.show_error("No segments to export."); return
        default_dir = self.settings.value("last_srt_dir", "")
        path, _ = QFileDialog.getSaveFileName(self, "Export SRT", os.path.join(default_dir, "subtitles.srt"), "Subtitle Files (*.srt)")
        if not path: return
        self.settings.setValue("last_srt_dir", os.path.dirname(path))
        try:
            def fmt(t):
                hrs, mins, secs = int(t // 3600), int((t % 3600) // 60), int(t % 60)
                millis = int((t - int(t)) * 1000); return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(self.segment_list.count()):
                    item = self.segment_list.item(i); p, s, e = item.data(Qt.UserRole)
                    f.write(f"{i + 1}\n{fmt(s)} --> {fmt(e)}\n{item.data(Qt.UserRole + 7) or os.path.basename(p)}\n\n")
            self.statusBar().showMessage(f"Exported SRT to {path}")
        except Exception as e: self.show_error(f"Failed to export SRT:\n{e}")

    def position_changed(self, position):
        if self.preview_end_ms > 0 and position >= self.preview_end_ms:
            self.media_player.pause(); self.preview_end_ms = 0
            if self._preview_queue: self._play_next_preview()
        if not self.seek_slider_dragging: self.seek_slider.setValue(position)
        if position - self._last_filmstrip_tick >= 50 or self._last_filmstrip_tick > position:
            self.filmstrip.update_playhead(position)
            self.visual_timeline.update_playhead(position)
            self._last_filmstrip_tick = position
        if position - self._last_label_ms >= PROGRESS_THROTTLE_MS:
            current_time = position / 1000.0; duration = self.media_player.duration() / 1000.0
            self.time_label.setText(f"{format_timestamp(current_time)} / {format_timestamp(duration)}")
            self._last_label_ms = position

    def duration_changed(self, duration): self.seek_slider.setRange(0, max(1, duration))

    def set_position(self, position):
        self.preview_end_ms = 0; self._preview_queue.clear()
        self.media_player.setPosition(position)

    def _on_slider_pressed(self): self.seek_slider_dragging = True
    def _on_slider_released(self): self.seek_slider_dragging = False

    def _on_filmstrip_hover(self, ms):
        if ms >= 0: self.time_label.setText(f"Hover: {format_timestamp(ms/1000)}")
        else:
            current_time = self.media_player.position() / 1000.0; duration = self.media_player.duration() / 1000.0
            self.time_label.setText(f"{format_timestamp(current_time)} / {format_timestamp(duration)}")

    def _on_media_error(self, *args):
        if not args or args[0] == QMediaPlayer.Error.NoError: return
        msg = args[1] if len(args) > 1 and args[1] else (self.media_player.errorString() or str(args[0]))
        self.statusBar().showMessage(f"⚠️ Media error: {msg}")
        log.error("Media error: %s", msg)

    def show_error(self, message): QMessageBox.critical(self, "Error", message)

    def _check_environment(self):
        missing = self._missing_deps()
        if missing:
            self.statusBar().showMessage("⚠️ Missing optional deps: " + ", ".join(missing) + " — analysis works, but export will fail.")
            self.btn_export.setEnabled(False)
            self.btn_youtube.setEnabled(False)

    def _restore_settings(self):
        self.restoreGeometry(self.settings.value("geometry", b""))
        self.restoreState(self.settings.value("state", b""))
        vol = self.settings.value("volume", 80, type=int)
        self.vol_slider.setValue(vol); self.audio_output.setVolume(vol / 100.0)

    def _wait_for_thread(self, thread: QThread, name: str, timeout_ms: int = 5000) -> bool:
        if not thread or not thread.isRunning(): return True
        if hasattr(thread, "cancel"): thread.cancel()
        elapsed = 0
        while thread.isRunning() and elapsed < timeout_ms:
            QApplication.processEvents(); thread.wait(100); elapsed += 100
            self.statusBar().showMessage(f"Waiting for {name} to finish… ({elapsed/1000:.1f}s)")
        return not thread.isRunning()

    def closeEvent(self, event):
        running = []
        if self.cv_thread and self.cv_thread.isRunning(): running.append("CV analysis")
        if self.export_thread and self.export_thread.isRunning(): running.append("video export")
        if running:
            if QMessageBox.question(self, "Work in progress", f"{' and '.join(running).capitalize()} is still running.\nCancel and quit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                event.ignore(); return
        if self._dirty:
            btn = QMessageBox.question(self, "Unsaved changes", "You have unsaved timeline changes.\n\nSave before quitting?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if btn == QMessageBox.StandardButton.Cancel: event.ignore(); return
            if btn == QMessageBox.StandardButton.Save:
                if not self.save_project(): event.ignore(); return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.settings.setValue("volume", self.vol_slider.value())
        self._wait_for_thread(self.cv_thread, "CV analysis", 5000)
        self._wait_for_thread(self.export_thread, "export", 10000)
        if hasattr(self, "filmstrip") and self.filmstrip._generator:
            if self.filmstrip._generator.isRunning():
                self.filmstrip._generator.cancel()
            self._wait_for_thread(self.filmstrip._generator, "filmstrip", 3000)
        if hasattr(self, "montage_thread") and self.montage_thread:
            self._wait_for_thread(self.montage_thread, "smart montage", 5000)
        self.thread_pool.clear(); self.thread_pool.waitForDone(2000)
        if os.path.exists(AUTOSAVE_FILE):
            try: os.remove(AUTOSAVE_FILE)
            except: pass
        event.accept()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #d0d0d0; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }
            QListWidget { background-color: #252525; border: 1px solid #1a1a1a; border-radius: 4px; font-size: 13px; outline: 0; }
            QListWidget::item { padding: 5px; border-bottom: 1px solid #2a2a2a; border-radius: 2px; }
            QListWidget::item:selected { background-color: #0078d4; color: white; }
            QListWidget::item:hover { background-color: #333333; }
            QPushButton { background-color: #333333; border: 1px solid #404040; padding: 8px 12px; border-radius: 4px; color: #cccccc; }
            QPushButton:hover { background-color: #3d3d3d; border-color: #555555; color: white; }
            QPushButton:pressed { background-color: #222222; }
            QPushButton:disabled { color: #666; background-color: #2a2a2a; border-color: #333; }
            QSlider::groove:horizontal { background: #2a2a2a; height: 6px; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #0078d4; border-radius: 3px; }
            QSlider::add-page:horizontal { background: #2a2a2a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #e0e0e0; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::handle:horizontal:hover { background: #ffffff; border: 1px solid #0078d4; }
            QGroupBox { border: 1px solid #333; border-radius: 4px; margin-top: 14px; padding-top: 14px; font-weight: bold; color: #888; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QProgressBar { text-align: center; border: 1px solid #333; border-radius: 4px; background-color: #222; color: white; height: 22px; font-weight: bold; }
            QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
            QTabWidget::pane { border: 1px solid #333; border-radius: 4px; top: -1px; }
            QTabBar::tab { background: #222; padding: 8px 16px; border: 1px solid #333; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; color: #888; margin-right: 2px; }
            QTabBar::tab:selected { background: #2d2d2d; color: #0078d4; border-bottom: 2px solid #0078d4; }
            QTabBar::tab:hover:!selected { background: #282828; }
            QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox { background-color: #2d2d2d; border: 1px solid #404040; padding: 4px; border-radius: 4px; color: white; selection-background-color: #0078d4; }
            QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 1px solid #0078d4; }
            QComboBox:hover { border: 1px solid #555; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView { background-color: #2d2d2d; border: 1px solid #404040; selection-background-color: #0078d4; color: #e0e0e0; outline: 0; }
            QSplitter::handle { background-color: #111111; }
            QSplitter::handle:hover { background-color: #0078d4; }
            QSplitter::handle:horizontal { width: 2px; }
            QSplitter::handle:vertical { height: 2px; }
            QScrollBar:vertical { background: #1e1e1e; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #3a3a3a; min-height: 30px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #454545; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar:horizontal { background: #1e1e1e; height: 10px; margin: 0; }
            QScrollBar::handle:horizontal { background: #3a3a3a; min-width: 30px; border-radius: 4px; }
            QScrollBar::handle:horizontal:hover { background: #454545; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QLabel { color: #ccc; background: transparent; }
            QStatusBar { background-color: #0078d4; color: white; font-weight: bold; }
            QStatusBar::item { border: none; }
            QMenu { background-color: #2d2d2d; border: 1px solid #404040; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #0078d4; color: white; }
        """)

if __name__ == "__main__":
    _configure_logging()
    log.info("Starting Pro Multi-Video AI Editor...")
    parser = argparse.ArgumentParser(description="Pro Multi-Video AI Editor")
    parser.add_argument("--project", help="Load a project file on startup")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("ProMultiVideoEditor")
    app.setOrganizationName("MyLab")
    
    window = VideoEditorApp()
    if args.project:
        QTimer.singleShot(100, lambda: window._load_project_from_path(args.project))
    window.show()
    sys.exit(app.exec())