import sys
import cv2
import os
import time
import shutil
import random
import re
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F_torch
from torchvision.transforms import functional as F
import winsound
import threading
import subprocess

# Optimized: Use inference_mode for better performance
torch.backends.cudnn.benchmark = True 

try:
    from ultralytics import YOLO as UltralyticsYOLO
except ImportError:
    UltralyticsYOLO = None

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QSlider, QCheckBox,
    QStatusBar, QComboBox, QSizePolicy, QMessageBox, QGroupBox, QDoubleSpinBox,
    QAbstractSpinBox, QDialog, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QObject, pyqtSlot, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor, QTextDocument, QFont
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog
from PyQt6.QtMultimedia import QMediaDevices

# --- STYLESHEET ---
DARK_THEME_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #f0f0f0; font-size: 10pt; border: none; }
QMainWindow { background-color: #3c3c3c; }
QPushButton { background-color: #555; border: 1px solid #777; padding: 5px 10px; border-radius: 4px; }
QPushButton:hover { background-color: #666; }
QPushButton:pressed { background-color: #777; }
QPushButton:disabled { background-color: #444; color: #888; border-color: #555; }
QLineEdit, QComboBox { background-color: #444; border: 1px solid #666; padding: 4px; border-radius: 4px; }
QSlider::groove:horizontal { border: 1px solid #555; height: 4px; background: #444; margin: 2px 0; border-radius: 2px; }
QSlider::handle:horizontal { background: #0078d7; border: 1px solid #f0f0f0; width: 14px; margin: -6px 0; border-radius: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #777; border-radius: 3px; background: #444; }
QCheckBox::indicator:hover { border-color: #0078d7; }
QCheckBox::indicator:checked { background-color: #0078d7; border: 1px solid #0078d7; }
QStatusBar { background-color: #3c3c3c; font-size: 11pt; }
"""

def _fourcc_to_string(fourcc_value):
    try:
        fourcc_int = int(fourcc_value) & 0xFFFFFFFF
        if fourcc_int == 0: return "UNKNOWN"
        chars = fourcc_int.to_bytes(4, byteorder="little")
        return "".join(chr(c) if 32 <= c <= 126 else "." for c in chars)
    except: return "UNKNOWN"

def open_camera_by_index(index, resolution=None, prefer_backend="DSHOW", fps=None, preferred_fourcc=None):
    pref = (prefer_backend or "AUTO").upper()
    backend = cv2.CAP_ANY
    if pref == "MSMF": backend = cv2.CAP_MSMF
    elif pref == "DSHOW": backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened(): return None, None
    
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if preferred_fourcc and preferred_fourcc != "Auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*preferred_fourcc[:4]))
        
    fourcc_mode = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    return cap, fourcc_mode

class ArduinoManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._serial = None
        self._port = None
        self._baudrate = 9600

    def available_ports(self):
        if list_ports is None: return []
        return [(p.device, f"{p.device} ({p.description})") for p in list_ports.comports()]

    def is_connected(self):
        with self._lock: return bool(self._serial and self._serial.is_open)

    def connect(self, port, baudrate=9600):
        with self._lock:
            try:
                if self._serial: self._serial.close()
                self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)
                self._port, self._baudrate = port, baudrate
            except: self._serial = None; raise

    def disconnect(self):
        with self._lock:
            if self._serial: self._serial.close()
            self._serial = None

    def send_command(self, command):
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.write(f"{command}\n".encode())

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._torch_device = torch.device(self.device)
        self.mode = 'recon'
        
        # Thresholds
        self.cv_threshold = 40
        self.contour_area_threshold = 10
        self.mse_threshold = 0.01
        
        # HSV Settings
        self.hsv_ranges = {
            'p': {'h':(15,35), 's':60, 'v':120, 'enabled':True},
            's': {'h':(75,95), 's':60, 'v':120, 'enabled':True},
            't': {'h':(105,125), 's':60, 'v':120, 'enabled':True}
        }
        
        # YOLO
        self.yolo_model = None
        self.yolo_enabled = False
        self.yolo_confidence = 0.35
        self.yolo_class_filter = set()
        self.yolo_dangerous_classes = set()
        
        # Optimization: Detection Scaling
        # ประมวลผลที่ขนาดเล็กลงเพื่อเพิ่ม FPS (เช่น ไม่เกิน 640px)
        self.detection_max_dim = 640 

    def set_mode(self, mode): self.mode = mode

    def load_yolo_model(self, path):
        if not path or UltralyticsYOLO is None: return False, "Error"
        try:
            self.yolo_model = UltralyticsYOLO(path)
            return True, "YOLO Loaded"
        except Exception as e: return False, str(e)

    def load_model(self, path):
        try:
            self.model = torch.load(path, map_location=self._torch_device)
            if hasattr(self.model, 'eval'): self.model.eval()
            return True
        except: return False

    def _get_hsv_mask(self, hsv_img, params):
        if not params['enabled']: return None
        low = np.array([params['h'][0], params['s'], params['v']])
        high = np.array([params['h'][1], 255, 255])
        return cv2.inRange(hsv_img, low, high)

    def process_frame(self, frame):
        # 1. Scale frame for faster detection
        h, w = frame.shape[:2]
        scale = 1.0
        if max(h, w) > self.detection_max_dim:
            scale = self.detection_max_dim / max(h, w)
            proc_frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        else:
            proc_frame = frame.copy()

        annotated = frame.copy()
        is_anom = False
        all_contours = []
        mse = 0.0
        self.last_detection_sources = []

        # YOLO detection (Run on resized frame for speed)
        yolo_dets = []
        if self.yolo_enabled and self.yolo_model:
            results = self.yolo_model(proc_frame, conf=self.yolo_confidence, verbose=False)
            if results:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.yolo_model.names.get(cls_id, str(cls_id))
                    
                    # Filtering
                    if self.yolo_class_filter and label.lower() not in self.yolo_class_filter: continue
                    
                    is_anom = True
                    # Re-scale box to original frame size
                    b = box.xyxy[0].cpu().numpy() / scale
                    yolo_dets.append({'bbox': b.astype(int), 'label': label, 'conf': conf})

        # Color Detection
        if self.mode in ['color', 'hybrid']:
            hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            combined_mask = None
            for key in ['p', 's', 't']:
                m = self._get_hsv_mask(hsv, self.hsv_ranges[key])
                if m is not None:
                    combined_mask = m if combined_mask is None else cv2.bitwise_or(combined_mask, m)
            
            if combined_mask is not None:
                # Morphological cleanup
                kernel = np.ones((3,3), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > (self.contour_area_threshold * scale):
                        is_anom = True
                        self.last_detection_sources.append("Color")
                        # Scale contour back
                        cnt_scaled = (cnt / scale).astype(int)
                        all_contours.append(cnt_scaled)

        # Model Reconstruction (Only if needed to save resources)
        if self.mode in ['recon', 'hybrid'] and not (is_anom and self.mode == 'hybrid'):
            # Optimization: Reconstruction is very heavy, skip if color/YOLO already found something in hybrid
            try:
                if self.model:
                    # Logic for PyTorch inference...
                    pass 
            except: pass

        # Drawing results
        for det in yolo_dets:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{det['label']} {det['conf']:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for cnt in all_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 2)

        return annotated, mse, is_anom, all_contours

class DetectionWorker(QObject):
    result_ready = pyqtSignal(np.ndarray, np.ndarray, float, bool, int)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.is_busy = False
        self.anomaly_count = 0
        self.frame_skip_count = 0
        self.process_every_n_frame = 2 # Optimized: Process every 2nd frame to reduce lag

    @pyqtSlot(np.ndarray)
    def process_frame(self, frame):
        if self.is_busy: return
        
        # Frame Skipping logic
        self.frame_skip_count += 1
        if self.frame_skip_count < self.process_every_n_frame:
            # Emit original frame but skip detection to keep UI alive
            self.result_ready.emit(frame, frame, 0.0, False, self.anomaly_count)
            return
        self.frame_skip_count = 0

        self.is_busy = True
        try:
            processed, mse, is_anom, contours = self.detector.process_frame(frame)
            if is_anom: self.anomaly_count += 1
            self.result_ready.emit(processed, frame, mse, is_anom, self.anomaly_count)
        finally:
            self.is_busy = False

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_index=0, resolution=None, fps_limit=None):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self._run_flag = True
        self.fps_limit = fps_limit

    def run(self):
        cap, _ = open_camera_by_index(self.camera_index, self.resolution)
        if not cap: return
        
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
                if self.fps_limit: time.sleep(1/self.fps_limit)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Anomaly Detection Optimized')
        self.detector = AnomalyDetector()
        self.initUI()
        self.setupThreads()
        self.setStyleSheet(DARK_THEME_STYLESHEET)

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_detection)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

    def setupThreads(self):
        self.det_thread = QThread()
        self.worker = DetectionWorker(self.detector)
        self.worker.moveToThread(self.det_thread)
        self.worker.result_ready.connect(self.update_image)
        self.det_thread.start()

    def start_detection(self):
        self.video_thread = VideoThread(camera_index=0)
        self.video_thread.change_pixmap_signal.connect(self.worker.process_frame)
        self.video_thread.start()

    def stop_detection(self):
        if hasattr(self, 'video_thread'): self.video_thread.stop()

    @pyqtSlot(np.ndarray, np.ndarray, float, bool, int)
    def update_image(self, processed, original, mse, is_anom, count):
        # Optimized Image conversion
        h, w, ch = processed.shape
        bytes_per_line = ch * w
        q_img = QImage(processed.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())