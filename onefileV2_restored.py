import sys
import cv2
import os
import time
import shutil
from datetime import datetime
import numpy as np
import torch
from torchvision.transforms import functional as F
import winsound # Import for beep sound
import threading

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QSlider, QCheckBox,
    QStatusBar, QComboBox, QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtMultimedia import QMediaDevices

# ... (ส่วน THEME และ open_camera_by_index เหมือนเดิม) ...
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

def open_camera_by_index(index, resolution=None, prefer_backend="MSMF"):
    if prefer_backend == "MSMF":
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif prefer_backend == "DSHOW":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW]
    cap = None
    for be in backends:
        tmp = cv2.VideoCapture(index, be)
        if tmp.isOpened():
            cap = tmp
            break
        else:
            tmp.release()
    if cap is None:
        return None
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return cap

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.cv_threshold = 40
        self.contour_area_threshold = 10
        self.mse_threshold = 0.01
        self.h_low, self.h_high = 15, 35
        self.s_min, self.v_min = 60, 120
        self.mode = 'recon'
        self.first_inference = True # <--- LOGGING FLAG

    def set_mode(self, mode: str):
        self.mode = mode
    def set_hsv_thresholds(self, h_low=None, h_high=None, s_min=None, v_min=None):
        if h_low is not None:  self.h_low  = int(max(0, min(179, h_low)))
        if h_high is not None: self.h_high = int(max(0, min(179, h_high)))
        if self.h_low > self.h_high:
            self.h_low, self.h_high = self.h_high, self.h_low
        if s_min is not None:  self.s_min  = int(max(0, min(255, s_min)))
        if v_min is not None:  self.v_min  = int(max(0, min(255, v_min)))
    def set_cv_threshold(self, value):
        self.cv_threshold = int(value)
    def set_contour_threshold(self, value):
        self.contour_area_threshold = int(value)
    def set_threshold(self, value):
        self.mse_threshold = float(value)

    def load_model(self, model_path):
        if model_path and model_path.endswith('.pth'):
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            try:
                device = torch.device(self.device)
                try:
                    self.model = torch.load(model_path, map_location=device, weights_only=True)
                except TypeError:
                    self.model = torch.load(model_path, map_location=device)
                if isinstance(self.model, dict):
                    print("Loaded a state_dict. No model class available -> using dummy recon path.")
                    self.model = "loaded"
                else:
                    if hasattr(self.model, 'to'): self.model.to(device)
                    if hasattr(self.model, 'eval'): self.model.eval()
                print(f"Model loaded: {model_path}")
            except Exception as e:
                print(f"Model load failed: {e} -> using dummy recon")
                self.model = "loaded"
            return True
        return False

    def _contours_from_mask(self, mask):
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > self.contour_area_threshold]

    def _mask_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.h_low, self.s_min, self.v_min], dtype=np.uint8)
        upper1 = np.array([self.h_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(self.h_low-5,0), self.s_min, self.v_min], dtype=np.uint8)
        upper2 = np.array([min(self.h_high+5,179), 255, 255], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    def _mask_recon_or_dummy(self, frame, is_first_frame=False):
        try:
            if self.model is not None and self.model != "loaded" and callable(self.model):
                with torch.no_grad():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inp = F.to_tensor(rgb).unsqueeze(0).to(torch.device(self.device))
                    if hasattr(self.model, 'eval'): self.model.eval()

                    if self.first_inference: # <--- LOGGING
                         print(f"[LOG {time.time():.2f}]   - Performing FIRST model inference (model warm-up)...")
                         start_infer_time = time.time()

                    rec = self.model(inp)

                    if self.first_inference: # <--- LOGGING
                        end_infer_time = time.time()
                        print(f"[LOG {end_infer_time:.2f}]   - FIRST model inference took {end_infer_time - start_infer_time:.4f} seconds.")
                        self.first_inference = False

                    rec = rec.squeeze(0).detach().cpu().numpy()
                    rec = np.transpose(rec, (1,2,0))
                    rec = np.clip(rec*255, 0, 255).astype(np.uint8)
                    reconstructed = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            else:
                reconstructed = cv2.GaussianBlur(frame, (7,7), 0)
        except Exception as e:
            print(f"Inference error: {e} -> fallback to dummy")
            reconstructed = cv2.GaussianBlur(frame, (7,7), 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_rec = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, gray_rec)
        _, mask = cv2.threshold(diff, self.cv_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((11,11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask, float((np.square(diff)).mean())

    def process_frame(self, frame, is_first_frame=False):
        if self.mode == 'color':
            mask = self._mask_color(frame)
            contours = self._contours_from_mask(mask)
            out = frame.copy()
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,255), 2)
            is_anom = len(contours) > 0
            if is_anom:
                cv2.putText(out, 'Color Anomaly (Yellow)', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            mse = float(mask.mean()/255.0)
            return out, mse, is_anom

        if self.mode == 'recon':
            mask, mse = self._mask_recon_or_dummy(frame, is_first_frame)
            contours = self._contours_from_mask(mask)
            out = frame.copy()
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 2)
            is_anom = len(contours) > 0
            if is_anom:
                cv2.putText(out, 'Anomaly (Recon)', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            return out, mse, is_anom

        # HYBRID OR
        mask_recon, mse = self._mask_recon_or_dummy(frame, is_first_frame)
        mask_color = self._mask_color(frame)
        contours_recon = self._contours_from_mask(mask_recon)
        contours_color = self._contours_from_mask(mask_color)
        out = frame.copy()
        for c in contours_recon:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
        for c in contours_color:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,255), 2)
        is_anom = (len(contours_recon) + len(contours_color)) > 0
        if is_anom:
            cv2.putText(out, 'HYBRID: Color OR Recon', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        mse_hybrid = 0.5*mse + 0.5*(mask_color.mean()/255.0)
        return out, float(mse_hybrid), is_anom

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, camera_index=0, resolution=None, fps_limit=None):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self._run_flag = True
        self.sleep_duration_ms = int(1000 / fps_limit) if fps_limit else 0
    def run(self):
        print(f"[LOG {time.time():.2f}] VideoThread entered run() method.")
        print(f"[LOG {time.time():.2f}] Starting to open camera index {self.camera_index}...")
        start_cam_time = time.time()
        cap = open_camera_by_index(self.camera_index, self.resolution, prefer_backend="MSMF")
        end_cam_time = time.time()
        print(f"[LOG {end_cam_time:.2f}] Camera opened. Took {end_cam_time - start_cam_time:.4f} seconds.")
        if cap is None:
            print(f"Error: Cannot open camera {self.camera_index} with any backend")
            return
        print(f"[LOG {time.time():.2f}] Entering main capture loop...")
        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
                if self.sleep_duration_ms > 0:
                    self.msleep(self.sleep_duration_ms)
            else:
                self.msleep(50)
        cap.release()
        print(f"[LOG {time.time():.2f}] VideoThread capture loop finished.")

    def stop(self):
        self._run_flag = False
        self.wait()

class DetectionWorker(QObject):
    result_ready = pyqtSignal(np.ndarray, np.ndarray, float, bool, int)
    status_update = pyqtSignal(str)
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.is_busy = False
        self.auto_save = False
        self.last_save_time = 0
        self.anomaly_count = 0
        self.beep_enabled = False
        self.first_frame_processed = False # <--- LOGGING FLAG

    def _beep_thread_target(self):
        try:
            winsound.Beep(1000, 200)
        except Exception as e:
            print(f"Error playing beep sound: {e}")

    def _play_beep_async(self):
        if not self.beep_enabled:
            return
        threading.Thread(target=self._beep_thread_target, daemon=True).start()

    @pyqtSlot(np.ndarray)
    def process_frame(self, cv_img):
        if self.is_busy: return
        self.is_busy = True

        is_first_frame = not self.first_frame_processed
        if is_first_frame:
            print(f"[LOG {time.time():.2f}] Processing FIRST frame in DetectionWorker...")
            start_process_time = time.time()

        original_frame = cv_img.copy()
        processed_frame, mse, is_anomaly = self.detector.process_frame(cv_img, is_first_frame)

        if is_first_frame:
            end_process_time = time.time()
            print(f"[LOG {end_process_time:.2f}] FIRST frame processed. Took {end_process_time - start_process_time:.4f} seconds.")
            self.first_frame_processed = True

        if is_anomaly:
            self.anomaly_count += 1
            self._play_beep_async()
        if is_anomaly and self.auto_save and time.time() - self.last_save_time > 2:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            fname = f"anomaly_{ts}.jpg"
            det_dir = os.path.join('output','detections')
            ori_dir = os.path.join('output','original')
            os.makedirs(det_dir, exist_ok=True)
            os.makedirs(ori_dir, exist_ok=True)
            cv2.imwrite(os.path.join(det_dir,fname), processed_frame)
            cv2.imwrite(os.path.join(ori_dir,fname), original_frame)
            self.status_update.emit(f"Anomaly saved as {fname}")
            self.last_save_time = time.time()
        self.result_ready.emit(processed_frame, original_frame, mse, is_anomaly, self.anomaly_count)
        self.is_busy = False

    @pyqtSlot(bool)
    def set_beep_enabled(self, status):
        self.beep_enabled = status
    @pyqtSlot(bool)
    def set_auto_save(self, status):
        self.auto_save = status
    def reset_counter(self):
        self.anomaly_count = 0
        self.first_frame_processed = False
        self.detector.first_inference = True

class MainWindow(QMainWindow):
    # ... (ส่วน __init__ และอื่นๆ เหมือนเดิม) ...
    trigger_process = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live Anomaly Detection GUI (Hybrid OR + Folder Clear)')
        self.setGeometry(100,100,1200,800)
        self.detector = AnomalyDetector()
        self.is_detection_running = False
        self.current_frame = None
        self.last_tested_image = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0
        self.central_widget = QWidget(); self.main_layout = QVBoxLayout(self.central_widget); self.setCentralWidget(self.central_widget)
        self._create_top_bar(); self._create_video_display(); self._create_controls(); self._create_status_bar(); self._setup_detection_thread()
        self.setStyleSheet(DARK_THEME_STYLESHEET)
        self._load_settings()

    def _setup_detection_thread(self):
        self.detection_thread = QThread()
        self.detection_worker = DetectionWorker(self.detector)
        self.detection_worker.moveToThread(self.detection_thread)
        self.trigger_process.connect(self.detection_worker.process_frame)
        self.detection_worker.result_ready.connect(self.display_processed_frame)
        self.detection_worker.status_update.connect(lambda msg: self.status_bar.showMessage(msg, 3000))
        self.auto_save_check.toggled.connect(self.detection_worker.set_auto_save)
        self.detection_thread.start()
        self.detection_worker.set_beep_enabled(self.beep_check.isChecked())

    def _create_top_bar(self):
        top = QHBoxLayout()
        top.addWidget(QLabel('Model Path:'))
        self.model_path_edit = QLineEdit(); self.model_path_edit.setReadOnly(True); top.addWidget(self.model_path_edit)
        self.browse_btn = QPushButton('Browse...'); self.browse_btn.clicked.connect(self._browse_model); top.addWidget(self.browse_btn)
        self.main_layout.addLayout(top)

    def _create_video_display(self):
        self.video_container = QWidget(); self.video_container.setStyleSheet('background-color:#000; border:1px solid #555;'); self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay = QVBoxLayout(self.video_container); lay.setContentsMargins(0,0,0,0)
        self.video_label = QLabel("Press 'Start Detection' or 'Test Image' to begin"); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        lay.addWidget(self.video_label); self.main_layout.addWidget(self.video_container, stretch=1)

    def _create_controls(self):
        controls = QHBoxLayout(); left = QVBoxLayout(); right = QVBoxLayout(); buttons = QHBoxLayout()
        # thresholds
        thr = QHBoxLayout(); thr.addWidget(QLabel('MSE Threshold:'))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal); self.thresh_slider.setRange(1,1000); self.thresh_slider.setValue(10); self.thresh_slider.valueChanged.connect(self._update_mse_threshold)
        self.thresh_label = QLabel(f"{self.thresh_slider.value()/1000:.3f}")
        thr.addWidget(self.thresh_slider); thr.addWidget(self.thresh_label); left.addLayout(thr)
        cvthr = QHBoxLayout(); cvthr.addWidget(QLabel('CV Threshold:'))
        self.cv_thresh_slider = QSlider(Qt.Orientation.Horizontal); self.cv_thresh_slider.setRange(5,200); self.cv_thresh_slider.setValue(40); self.cv_thresh_slider.valueChanged.connect(self._update_cv_threshold)
        self.cv_thresh_label = QLabel(str(self.cv_thresh_slider.value()))
        cvthr.addWidget(self.cv_thresh_slider); cvthr.addWidget(self.cv_thresh_label); left.addLayout(cvthr)
        cont = QHBoxLayout(); cont.addWidget(QLabel('Contour Area:'))
        self.contour_slider = QSlider(Qt.Orientation.Horizontal); self.contour_slider.setRange(1,200); self.contour_slider.setValue(10); self.contour_slider.valueChanged.connect(self._update_contour_threshold)
        self.contour_label = QLabel(str(self.contour_slider.value()))
        cont.addWidget(self.contour_slider); cont.addWidget(self.contour_label); left.addLayout(cont)
        # camera & fps
        cr = QHBoxLayout(); cr.addWidget(QLabel('Source:'))
        self.cam_combo = QComboBox(); cr.addWidget(self.cam_combo)
        self.list_cam_btn = QPushButton('List Cameras'); self.list_cam_btn.clicked.connect(self._list_cameras); cr.addWidget(self.list_cam_btn)
        cr.addWidget(QLabel('Resolution:'))

        self.fps_combo = QComboBox(); self.fps_options = ['Uncapped','60','30','15']; self.fps_combo.addItems(self.fps_options); cr.addWidget(self.fps_combo)
        self.res_combo = QComboBox(); self.resolution_options = ['Source/Native','2592x1944', '2560x1440', '2048x1536', '1600x1200', '1280X960', '1024x768', '960X720', '800x600', '848x480', '640x360', '2592x1440', '2304x1296', '1920x1080', '1600x900', '1280x720', '1024x576', '960x540', '800x450', '640x480']; self.res_combo.addItems(self.resolution_options); cr.addWidget(self.res_combo)
        cr.addWidget(QLabel('Frame Rate:'))
        left.addLayout(cr)

        # right panel
        auto_row = QHBoxLayout()
        self.auto_save_check = QCheckBox('Auto-save Detections')
        self.auto_save_status_label = QLabel('Disabled')
        self.auto_save_status_label.setStyleSheet('color: #d35400; font-weight: bold;')
        self.auto_save_check.toggled.connect(self._update_auto_save_status)
        auto_row.addWidget(self.auto_save_check)
        auto_row.addWidget(self.auto_save_status_label)
        auto_row.addStretch()
        right.addLayout(auto_row)

        # Beep sound control
        beep_row = QHBoxLayout()
        self.beep_check = QCheckBox('Enable Beep Sound')
        self.beep_check.toggled.connect(self._update_beep_status)
        beep_row.addWidget(self.beep_check)
        beep_row.addStretch()
        right.addLayout(beep_row)

        mode_row = QHBoxLayout(); mode_row.addWidget(QLabel('Mode:'))
        self.mode_combo = QComboBox(); self.mode_combo.addItems(['Reconstruction/Model','Color (HSV • Yellow)','Hybrid (OR)'])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo); right.addLayout(mode_row)
        # HSV sliders
        h_low_l = QHBoxLayout(); h_low_l.addWidget(QLabel('Hue Low:'))
        self.h_low_slider = QSlider(Qt.Orientation.Horizontal); self.h_low_slider.setRange(0,179); self.h_low_slider.setValue(15); self.h_low_slider.valueChanged.connect(self._update_hsv)
        self.h_low_label = QLabel('15'); h_low_l.addWidget(self.h_low_slider); h_low_l.addWidget(self.h_low_label); right.addLayout(h_low_l)
        h_high_l = QHBoxLayout(); h_high_l.addWidget(QLabel('Hue High:'))
        self.h_high_slider = QSlider(Qt.Orientation.Horizontal); self.h_high_slider.setRange(0,179); self.h_high_slider.setValue(35); self.h_high_slider.valueChanged.connect(self._update_hsv)
        self.h_high_label = QLabel('35'); h_high_l.addWidget(self.h_high_slider); h_high_l.addWidget(self.h_high_label); right.addLayout(h_high_l)
        s_min_l = QHBoxLayout(); s_min_l.addWidget(QLabel('Saturation Min:'))
        self.s_min_slider = QSlider(Qt.Orientation.Horizontal); self.s_min_slider.setRange(0,255); self.s_min_slider.setValue(60); self.s_min_slider.valueChanged.connect(self._update_hsv)
        self.s_min_label = QLabel('60'); s_min_l.addWidget(self.s_min_slider); s_min_l.addWidget(self.s_min_label); right.addLayout(s_min_l)
        v_min_l = QHBoxLayout(); v_min_l.addWidget(QLabel('Value Min:'))
        self.v_min_slider = QSlider(Qt.Orientation.Horizontal); self.v_min_slider.setRange(0,255); self.v_min_slider.setValue(120); self.v_min_slider.valueChanged.connect(self._update_hsv)
        self.v_min_label = QLabel('120'); v_min_l.addWidget(self.v_min_slider); v_min_l.addWidget(self.v_min_label); right.addLayout(v_min_l)

        # buttons row
        self.start_btn = QPushButton('Start Detection'); self.start_btn.clicked.connect(self._start_detection); self.start_btn.setDisabled(True)
        self.stop_btn = QPushButton('Stop Detection'); self.stop_btn.clicked.connect(self._stop_detection); self.stop_btn.setDisabled(True)
        self.test_image_btn = QPushButton('Test Image'); self.test_image_btn.clicked.connect(self._test_image); self.test_image_btn.setDisabled(True)
        self.capture_btn = QPushButton('Capture Image'); self.capture_btn.clicked.connect(self._capture_image); self.capture_btn.setDisabled(True)
        self.clear_det_ori_btn = QPushButton('Clear Detected & Original')
        self.clear_det_ori_btn.clicked.connect(self._clear_detections_and_originals)
        self.clear_all_btn = QPushButton('Clear ALL output')
        self.clear_all_btn.clicked.connect(self._clear_output_folder)

        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.test_image_btn)
        buttons.addWidget(self.capture_btn)
        buttons.addWidget(self.clear_det_ori_btn)
        buttons.addWidget(self.clear_all_btn)
        buttons.addStretch()

        controls.addLayout(left,3); controls.addLayout(right,2)
        self.main_layout.addLayout(controls); self.main_layout.addLayout(buttons)

        self._list_cameras(); self._enable_hsv_controls(False)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready. Load a model or choose Color/Hybrid mode.')

    def _browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Model', '', 'PyTorch Model Files (*.pth)')
        if file_name: self._load_model_action(file_name)

    def _load_model_action(self, file_name):
        if self.detector.load_model(file_name):
            self.model_path_edit.setText(file_name)
            if 'cuda' in self.detector.device:
                try:
                    gpu_index = torch.cuda.current_device()
                except Exception:
                    gpu_index = 0
                disp = f"CUDA:{gpu_index} ({torch.cuda.get_device_name(gpu_index)})"
            else:
                disp = 'CPU'
            self.status_bar.showMessage(f"Model loaded on {disp}")
            self.start_btn.setDisabled(False); self.test_image_btn.setDisabled(False)
        else:
            self.model_path_edit.setText(''); self.status_bar.showMessage('Failed to load model.')
            if self.mode_combo.currentIndex() == 0:
                self.start_btn.setDisabled(True); self.test_image_btn.setDisabled(True)

    def _list_cameras(self):
        self.cam_combo.clear()
        available_cameras = QMediaDevices.videoInputs()
        if not available_cameras:
            self.cam_combo.addItem('No Camera Found')
        else:
            for camera_device in available_cameras:
                self.cam_combo.addItem(camera_device.description())

    def _update_mse_threshold(self, value):
        self.thresh_label.setText(f"{value/1000:.3f}"); self.detector.set_threshold(value/1000.0); self._reprocess_image()
    def _update_cv_threshold(self, value):
        self.cv_thresh_label.setText(str(value)); self.detector.set_cv_threshold(value); self._reprocess_image()
    def _update_contour_threshold(self, value):
        self.contour_label.setText(str(value)); self.detector.set_contour_threshold(value); self._reprocess_image()
    def _update_auto_save_status(self, checked):
        if checked:
            self.auto_save_status_label.setText('Enabled')
            self.auto_save_status_label.setStyleSheet('color: #27ae60; font-weight: bold;')
        else:
            self.auto_save_status_label.setText('Disabled')
            self.auto_save_status_label.setStyleSheet('color: #d35400; font-weight: bold;')
        self.status_bar.showMessage('Auto-save enabled.' if checked else 'Auto-save disabled.', 2000)

    def _update_beep_status(self, checked):
        if checked:
            self.status_bar.showMessage('Beep sound enabled.', 2000)
        else:
            self.status_bar.showMessage('Beep sound disabled.', 2000)
        if hasattr(self, 'detection_worker'):
            self.detection_worker.set_beep_enabled(checked)

    def _enable_hsv_controls(self, enabled: bool):
        for w in [self.h_low_slider, self.h_high_slider, self.s_min_slider, self.v_min_slider, self.h_low_label, self.h_high_label, self.s_min_label, self.v_min_label]:
            w.setEnabled(enabled)

    def _mode_changed(self, idx):
        mode = ['reconstruction/model','color','hybrid'][idx]
        self.detector.set_mode({'reconstruction/model':'recon','color':'color','hybrid':'hybrid'}[mode])
        self._enable_hsv_controls(idx in (1,2))
        if idx in (1,2):
            self.start_btn.setDisabled(False); self.test_image_btn.setDisabled(False)
        else:
            if not self.model_path_edit.text():
                self.start_btn.setDisabled(True); self.test_image_btn.setDisabled(True)
        self._reprocess_image()

    def _update_hsv(self, _):
        h_low = self.h_low_slider.value(); h_high = self.h_high_slider.value()
        if h_low > h_high:
            sender = self.sender()
            if sender is self.h_low_slider:
                self.h_high_slider.setValue(h_low); h_high = h_low
            else:
                self.h_low_slider.setValue(h_high); h_low = h_high
        s_min = self.s_min_slider.value(); v_min = self.v_min_slider.value()
        self.h_low_label.setText(str(h_low)); self.h_high_label.setText(str(h_high))
        self.s_min_label.setText(str(s_min)); self.v_min_label.setText(str(v_min))
        self.detector.set_hsv_thresholds(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min)
        self._reprocess_image()

    def _start_detection(self):
        start_time_total = time.time()
        print(f"\n[LOG {start_time_total:.2f}] 'Start Detection' button clicked.")

        if self.cam_combo.currentText() == 'No Camera Found':
            self.status_bar.showMessage('Error: No camera selected or found.'); return
        self.last_tested_image = None; self.status_bar.showMessage('Starting detection...')
        self.frame_count = 0; self.start_time = time.time(); self.detection_worker.reset_counter()
        res_text = self.res_combo.currentText(); resolution = tuple(map(int, res_text.split('x'))) if res_text != 'Source/Native' else None
        fps_text = self.fps_combo.currentText(); fps_limit = int(fps_text) if fps_text != 'Uncapped' else None

        print(f"[LOG {time.time():.2f}] Creating VideoThread...")
        self.video_thread = VideoThread(camera_index=self.cam_combo.currentIndex(), resolution=resolution, fps_limit=fps_limit)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        
        print(f"[LOG {time.time():.2f}] Starting VideoThread...")
        self.video_thread.start()
        
        self.is_detection_running = True; self._toggle_controls(False)
        print(f"[LOG {time.time():.2f}] _start_detection function finished. Total time: {time.time() - start_time_total:.4f} seconds.")

    def _stop_detection(self):
        if hasattr(self,'video_thread') and self.video_thread.isRunning(): self.video_thread.stop()
        self.is_detection_running = False
        if self.last_tested_image is None and (self.video_label.pixmap() is None or self.video_label.pixmap().isNull()):
            self.video_label.setText("Press 'Start Detection' or 'Test Image' to begin")
        self.status_bar.showMessage('Detection stopped.'); self._toggle_controls(True)

    # ... (ส่วนที่เหลือของ MainWindow เหมือนเดิม) ...
    def _toggle_controls(self, enable):
        self.start_btn.setDisabled(not enable); self.browse_btn.setDisabled(not enable); self.cam_combo.setDisabled(not enable)
        self.res_combo.setDisabled(not enable); self.list_cam_btn.setDisabled(not enable); self.test_image_btn.setDisabled(not enable)
        self.stop_btn.setDisabled(enable); self.capture_btn.setDisabled(not enable); self.fps_combo.setDisabled(not enable)

    def _capture_image(self):
        if not self.is_detection_running or self.current_frame is None:
            self.status_bar.showMessage('Capture only available during live detection.', 3000); return
        original_frame = self.current_frame.copy()
        processed_frame, _, _ = self.detector.process_frame(original_frame.copy())
        anomaly_count = self.detection_worker.anomaly_count
        h,w,_ = processed_frame.shape; margin = 10
        fps_text = f'FPS: {self.fps:.2f}'; cv2.putText(processed_frame, fps_text, (margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        res_text = f'{w}x{h}'; (res_w,_),_ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2); cv2.putText(processed_frame, res_text, (w-res_w-margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        det_text = f'Detections: {anomaly_count}'; (det_w,_),_ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3); cv2.putText(processed_frame, det_text, (w-det_w-margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]; fname = f'capture_{ts}.jpg'
        det_dir = os.path.join('output','captures_detected'); ori_dir = os.path.join('output','captures_original')
        os.makedirs(det_dir, exist_ok=True); os.makedirs(ori_dir, exist_ok=True)
        cv2.imwrite(os.path.join(det_dir,fname), processed_frame); cv2.imwrite(os.path.join(ori_dir,fname), original_frame)
        self.status_bar.showMessage(f'Image captured and saved as {fname}', 4000)

    def _clear_detections_and_originals(self):
        was_running = self.is_detection_running
        if was_running:
            self._stop_detection()
        reply = QMessageBox.question(
            self,
            'Confirm Clear',
            'Delete ALL files inside "output/detections" and "output/original"? (captures_* will be kept)',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            if was_running:
                self._start_detection()
            return
        try:
            for sub in ['detections', 'original']:
                path = os.path.join('output', sub)
                if os.path.isdir(path):
                    for name in os.listdir(path):
                        fpath = os.path.join(path, name)
                        try:
                            if os.path.isfile(fpath) or os.path.islink(fpath):
                                os.remove(fpath)
                            elif os.path.isdir(fpath):
                                shutil.rmtree(fpath, ignore_errors=True)
                        except Exception:
                            pass
                else:
                    os.makedirs(path, exist_ok=True)
            self.status_bar.showMessage('Cleared output/detections and output/original.', 3000)
        except Exception as e:
            self.status_bar.showMessage(f'Failed to clear: {e}', 5000)
        finally:
            if was_running:
                self._start_detection()

    def _clear_output_folder(self):
        was_running = self.is_detection_running
        if was_running:
            self._stop_detection()
        reply = QMessageBox.question(
            self,
            'Confirm Delete',
            'Delete the entire output folder and ALL its contents?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            if was_running:
                self._start_detection()
            return
        try:
            out_dir = 'output'
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(os.path.join(out_dir, 'detections'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'original'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'captures_detected'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'captures_original'), exist_ok=True)
            self.status_bar.showMessage('Cleared ENTIRE output folder.', 3000)
        except Exception as e:
            self.status_bar.showMessage(f'Failed to clear output: {e}', 5000)
        finally:
            if was_running:
                self._start_detection()

    def _test_image(self):
        if self.is_detection_running: self._stop_detection()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not file_name: return
        cv_img = cv2.imread(file_name)
        if cv_img is None:
            self.status_bar.showMessage('Error: Could not read image file', 4000); self.last_tested_image = None; return
        self.status_bar.showMessage(f'Loaded image: {os.path.basename(file_name)}'); self.last_tested_image = cv_img; self._reprocess_image()

    def _reprocess_image(self):
        if self.last_tested_image is None or self.is_detection_running: return
        processed_frame, mse, is_anomaly = self.detector.process_frame(self.last_tested_image.copy())
        if is_anomaly:
            h,w,_ = processed_frame.shape; text='Detections: 1'; (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.putText(processed_frame, text, (w-tw-10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        mode_map = {'recon':'Recon','color':'HSV','hybrid':'Hybrid'}; mode = mode_map[self.detector.mode]
        self.status_bar.showMessage(f"Last Image Test | Mode: {mode} | MSE: {mse:.4f} | Anomaly: {'Yes' if is_anomaly else 'No'}")
        self.video_label.setPixmap(self.convert_cv_qt(processed_frame))

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self.trigger_process.emit(cv_img)

    @pyqtSlot(np.ndarray, np.ndarray, float, bool, int)
    def display_processed_frame(self, processed_frame, original_frame, mse, is_anomaly, anomaly_count):
        self.current_frame = original_frame
        self.frame_count += 1; elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count/elapsed; self.frame_count = 0; self.start_time = time.time()
        h,w,_ = processed_frame.shape; margin = 10
        cv2.putText(processed_frame, f'FPS: {self.fps:.2f}', (margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        res_text = f'{w}x{h}'; (rw,_),_ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(processed_frame, res_text, (w-rw-margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        det_text = f'Detections: {anomaly_count}'; (dw,_),_ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(processed_frame, det_text, (w-dw-margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        self.video_label.setPixmap(self.convert_cv_qt(processed_frame))

    def convert_cv_qt(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape; bytes_per_line = ch*w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(self.video_container.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def _save_settings(self):
        s = QSettings('MyCompany','AnomalyApp')
        s.setValue('model_path', self.model_path_edit.text())
        s.setValue('mse_threshold', self.thresh_slider.value())
        s.setValue('cv_threshold', self.cv_thresh_slider.value())
        s.setValue('contour_area', self.contour_slider.value())
        s.setValue('camera_index', self.cam_combo.currentIndex())
        s.setValue('resolution_text', self.res_combo.currentText())
        s.setValue('fps_limit_text', self.fps_combo.currentText())
        s.setValue('auto_save', self.auto_save_check.isChecked())
        s.setValue('mode_index', self.mode_combo.currentIndex())
        s.setValue('h_low', self.h_low_slider.value())
        s.setValue('h_high', self.h_high_slider.value())
        s.setValue('s_min', self.s_min_slider.value())
        s.setValue('v_min', self.v_min_slider.value())
        s.setValue('beep_enabled', self.beep_check.isChecked())

    def _load_settings(self):
        s = QSettings('MyCompany','AnomalyApp')
        model_path = s.value('model_path','')
        if model_path and os.path.exists(model_path): self._load_model_action(model_path)
        self.thresh_slider.setValue(s.value('mse_threshold',10,type=int))
        self.cv_thresh_slider.setValue(s.value('cv_threshold',40,type=int))
        self.contour_slider.setValue(s.value('contour_area',10,type=int))
        if self.cam_combo.count()>0:
            idx = s.value('camera_index',0,type=int)
            if idx < self.cam_combo.count(): self.cam_combo.setCurrentIndex(idx)
        res_text = s.value('resolution_text','1280x720');
        if res_text in self.resolution_options: self.res_combo.setCurrentText(res_text)
        fps_text = s.value('fps_limit_text','Uncapped');
        if fps_text in self.fps_options: self.fps_combo.setCurrentText(fps_text)
        auto_save = s.value('auto_save',False,type=bool); self.auto_save_check.setChecked(auto_save); self.detection_worker.set_auto_save(auto_save)
        mode_idx = s.value('mode_index',0,type=int); self.mode_combo.setCurrentIndex(mode_idx)
        self.h_low_slider.setValue(s.value('h_low',15,type=int)); self.h_high_slider.setValue(s.value('h_high',35,type=int))
        self.s_min_slider.setValue(s.value('s_min',60,type=int)); self.v_min_slider.setValue(s.value('v_min',120,type=int))
        self.h_low_label.setText(str(self.h_low_slider.value())); self.h_high_label.setText(str(self.h_high_slider.value()))
        self.s_min_label.setText(str(self.s_min_slider.value())); self.v_min_label.setText(str(self.v_min_slider.value()))
        self.detector.set_hsv_thresholds(self.h_low_slider.value(), self.h_high_slider.value(), self.s_min_slider.value(), self.v_min_slider.value())
        self._enable_hsv_controls(self.mode_combo.currentIndex() in (1,2))
        self.beep_check.setChecked(s.value('beep_enabled', False, type=bool))
        self.detection_worker.set_beep_enabled(self.beep_check.isChecked())
        if mode_idx in (1,2) and not model_path:
            self.start_btn.setDisabled(False); self.test_image_btn.setDisabled(False)

    def closeEvent(self, event):
        self._save_settings(); self._stop_detection(); self.detection_thread.quit(); self.detection_thread.wait(); event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())