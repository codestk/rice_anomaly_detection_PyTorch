import sys
import cv2
import os
import time
from datetime import datetime
import numpy as np
import torch
from torchvision.transforms import functional as F  # Import for tensor conversion

from PyQt6.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QSlider,
    QCheckBox,
    QStatusBar,
    QComboBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSettings, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import winsound  # For beep sound
import threading # For non-blocking beep

# --- Theme and Detector are now part of this file ---

# Dark theme stylesheet for the PyQt application
DARK_THEME_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    font-size: 10pt; /* Increased font size */
    border: none;
}
QMainWindow {
    background-color: #3c3c3c;
}
QPushButton {
    background-color: #555;
    border: 1px solid #777;
    padding: 5px 10px;
    border-radius: 4px;
    min-height: 16px;
}
QPushButton:hover {
    background-color: #666;
}
QPushButton:pressed {
    background-color: #777;
}
QPushButton:disabled {
    background-color: #444;
    color: #888;
    border-color: #555;
}
QPushButton:checkable:checked {
    background-color: #0078d7;
    border-color: #005a9e;
}
QLineEdit, QComboBox {
    background-color: #444;
    border: 1px solid #666;
    padding: 4px;
    border-radius: 4px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    image: url(down_arrow.png); /* A real app might need an icon */
}
QSlider::groove:horizontal {
    border: 1px solid #555;
    height: 4px;
    background: #444;
    margin: 2px 0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #0078d7;
    border: 1px solid #f0f0f0;
    width: 14px;
    margin: -6px 0; /* handle is placed vertically centered */
    border-radius: 8px;
}
QStatusBar {
    background-color: #3c3c3c;
    font-size: 11pt; /* Increased font size */
}
QCheckBox {
    spacing: 5px;
}
QCheckBox::indicator {
    width: 13px;
    height: 13px;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background-color: #444;
    border: 1px solid #666;
}
QCheckBox::indicator:checked {
    background-color: #0078d7; /* Bright blue to indicate checked state */
    border: 1px solid #999;
}
QLabel {
    background-color: transparent;
}
"""


class AnomalyDetector:
    """
    An anomaly detector that supports two modes:
    1) Color-based anomaly (HSV) for yellow grains
    2) Reconstruction-based (placeholder/dummy when model is not loaded)
    """

    def __init__(self):
        self.model = None
        self.device = "CPU"
        self.mse_threshold = 0.01
        self.cv_threshold = 40
        self.contour_area_threshold = 10

        # --- Color detection (HSV) params ---
        self.color_mode = False  # toggle for HSV color detection
        # OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
        self.h_low = 15
        self.h_high = 35
        self.s_min = 60
        self.v_min = 120

    # ---------------- Color controls -----------------
    def set_color_mode(self, enabled: bool):
        self.color_mode = bool(enabled)

    def set_hsv_thresholds(self, h_low=None, h_high=None, s_min=None, v_min=None):
        if h_low is not None:
            self.h_low = int(max(0, min(179, h_low)))
        if h_high is not None:
            self.h_high = int(max(0, min(179, h_high)))
        # keep bounds sane
        if self.h_low > self.h_high:
            self.h_low, self.h_high = self.h_high, self.h_low
        if s_min is not None:
            self.s_min = int(max(0, min(255, s_min)))
        if v_min is not None:
            self.v_min = int(max(0, min(255, v_min)))

    # ---------------- General controls -----------------
    def load_model(self, model_path):
        """Load a PyTorch model if available. Color mode can work without a model."""
        if model_path and model_path.endswith(".pth"):
            # Select device
            if torch.cuda.is_available():
                self.device = "cuda:0"  # Explicitly select GPU 0
            else:
                self.device = "cpu"

            # Try to load a real model if provided (kept simple here)
            try:
                device = torch.device(self.device)
                self.model = torch.load(model_path, map_location=device)
                if hasattr(self.model, 'to'):
                    self.model.to(device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Warning: failed to load model ({e}). Fallback to dummy mode.")
                self.model = "loaded"  # still allow GUI to continue

            return True
        return False

    def set_threshold(self, value):
        self.mse_threshold = value

    def set_cv_threshold(self, value):
        self.cv_threshold = value

    def set_contour_threshold(self, value):
        self.contour_area_threshold = value

    # ----------------- Processing ----------------------
    def _process_color_hsv(self, frame):
        # Convert to HSV and threshold for yellow-ish hue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.h_low, self.s_min, self.v_min], dtype=np.uint8)
        upper1 = np.array([self.h_high, 255, 255], dtype=np.uint8)

        # optional slack around bounds to be robust to lighting
        lower2 = np.array([max(self.h_low - 5, 0), self.s_min, self.v_min], dtype=np.uint8)
        upper2 = np.array([min(self.h_high + 5, 179), 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_frame = frame.copy()
        is_anomaly = False
        for c in contours:
            if cv2.contourArea(c) > self.contour_area_threshold:
                is_anomaly = True
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if is_anomaly:
            cv2.putText(
                output_frame,
                "Color Anomaly (Yellow)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

        # pseudo-mse: average mask intensity (0..255) normalized
        mse = float(mask.mean() / 255.0)
        return output_frame, mse, is_anomaly

    def _process_reconstruction_dummy(self, frame):
        # --- Dummy CPU-based Processing Logic (Current Active Code) ---
        reconstructed = cv2.GaussianBlur(frame, (7, 7), 0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame, gray_reconstructed)
        _, thresh = cv2.threshold(diff, self.cv_threshold, 255, cv2.THRESH_BINARY)
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(processed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_frame = frame.copy()
        mse = (np.square(diff)).mean()
        is_anomaly = False

        for contour in contours:
            if cv2.contourArea(contour) > self.contour_area_threshold:
                is_anomaly = True
                (cx, cy, cw, ch) = cv2.boundingRect(contour)
                cv2.rectangle(output_frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)

        if is_anomaly:
            cv2.putText(output_frame, "Anomaly Detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return output_frame, mse, is_anomaly

    def _process_reconstruction_model(self, frame):
        # Example path if a real PyTorch model is available
        # This code will run only when self.model appears to be a real model with callable behavior
        try:
            with torch.no_grad():
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = F.to_tensor(rgb_frame).unsqueeze(0)
                device = torch.device(self.device)
                input_tensor = input_tensor.to(device)

                if callable(self.model):
                    reconstructed_tensor = self.model(input_tensor)
                else:
                    # If model is not callable, fallback to dummy path
                    return self._process_reconstruction_dummy(frame)

                reconstructed_frame_np = reconstructed_tensor.squeeze(0).detach().cpu().numpy()
                reconstructed_frame_np = np.transpose(reconstructed_frame_np, (1, 2, 0))
                reconstructed_frame_np = np.clip(reconstructed_frame_np * 255.0, 0, 255).astype(np.uint8)
                reconstructed = cv2.cvtColor(reconstructed_frame_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Model inference error: {e}. Fallback to dummy.")
            return self._process_reconstruction_dummy(frame)

        # Diff in grayscale as before
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame, gray_reconstructed)
        _, thresh = cv2.threshold(diff, self.cv_threshold, 255, cv2.THRESH_BINARY)
        kernel_size = 11
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(processed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_frame = frame.copy()
        mse = (np.square(diff)).mean()
        is_anomaly = False
        for contour in contours:
            if cv2.contourArea(contour) > self.contour_area_threshold:
                is_anomaly = True
                (cx, cy, cw, ch) = cv2.boundingRect(contour)
                cv2.rectangle(output_frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
        if is_anomaly:
            cv2.putText(output_frame, "Anomaly Detected (Model)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return output_frame, mse, is_anomaly

    def process_frame(self, frame):
        """
        Processes a single frame to detect anomalies.
        Color-mode path takes precedence if enabled.
        """
        if self.color_mode:
            return self._process_color_hsv(frame)

        # If model appears loaded, try to use it; otherwise use dummy
        if self.model is not None and self.model != "loaded":
            return self._process_reconstruction_model(frame)
        else:
            return self._process_reconstruction_dummy(frame)


# --- Thread for camera feed ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0, resolution=None, fps_limit=None):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self._run_flag = True
        self.sleep_duration_ms = 0
        if fps_limit is not None and fps_limit > 0:
            self.sleep_duration_ms = int(1000 / fps_limit)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return

        if self.resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
                if self.sleep_duration_ms > 0:
                    self.msleep(self.sleep_duration_ms)
            else:
                self.msleep(50)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# --- Worker for detection processing ---
class DetectionWorker(QObject):
    result_ready = pyqtSignal(np.ndarray, np.ndarray, float, bool, int)  # processed_frame, original_frame, mse, is_anomaly, anomaly_count
    status_update = pyqtSignal(str)

    def _beep_thread_target(self):
        winsound.Beep(2500, 200)  # Example: 2500 Hz, 200 ms duration

    def play_beep(self):
        # Play a beep sound in a separate thread to avoid blocking
        if self.beep_enabled:
            beep_thread = threading.Thread(target=self._beep_thread_target)
            beep_thread.daemon = True  # Allow the program to exit even if the thread is still running
            beep_thread.start()

    @pyqtSlot(bool)
    def set_beep_enabled(self, status):
        self.beep_enabled = status

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.is_busy = False
        self.auto_save = False
        self.last_save_time = 0
        self.anomaly_count = 0
        self.beep_enabled = True # New: control beep sound

    @pyqtSlot(np.ndarray)
    def process_frame(self, cv_img):
        if self.is_busy:
            return

        self.is_busy = True

        original_frame = cv_img.copy()
        processed_frame, mse, is_anomaly = self.detector.process_frame(cv_img)

        if is_anomaly:
            self.anomaly_count += 1
            self.play_beep()  # Play beep sound when anomaly is detected

        # Auto-save logic is moved here to run in the background thread
        if is_anomaly and self.auto_save and time.time() - self.last_save_time > 2:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"anomaly_{timestamp}.jpg"

            detections_dir = os.path.join("output", "detections")
            original_dir = os.path.join("output", "original")

            os.makedirs(detections_dir, exist_ok=True)
            os.makedirs(original_dir, exist_ok=True)

            detection_save_path = os.path.join(detections_dir, filename)
            original_save_path = os.path.join(original_dir, filename)

            cv2.imwrite(detection_save_path, processed_frame)
            cv2.imwrite(original_save_path, original_frame)

            self.status_update.emit(f"Anomaly saved as {filename}")
            self.last_save_time = time.time()

        self.result_ready.emit(processed_frame, original_frame, mse, is_anomaly, self.anomaly_count)
        self.is_busy = False

    @pyqtSlot(bool)
    def set_auto_save(self, status):
        self.auto_save = status

    def reset_counter(self):
        self.anomaly_count = 0


# --- Main Application Window ---
class MainWindow(QMainWindow):
    trigger_process = pyqtSignal(np.ndarray)  # Signal to trigger the worker

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Anomaly Detection GUI (Multithreaded)")
        self.setGeometry(100, 100, 1200, 800)

        self.detector = AnomalyDetector()

        # --- State variables ---
        self.is_detection_running = False
        self.current_frame = None
        self.last_tested_image = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0

        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        self._create_top_bar()
        self._create_video_display()
        self._create_controls()
        self._create_status_bar()
        self._setup_detection_thread()  # New method to set up worker thread

        self.setStyleSheet(DARK_THEME_STYLESHEET)
        self._load_settings()

    def _setup_detection_thread(self):
        self.detection_thread = QThread()
        self.detection_worker = DetectionWorker(self.detector)
        self.detection_worker.moveToThread(self.detection_thread)

        # Connect signals and slots
        self.trigger_process.connect(self.detection_worker.process_frame)
        self.detection_worker.result_ready.connect(self.display_processed_frame)
        self.detection_worker.status_update.connect(lambda msg: self.status_bar.showMessage(msg, 3000))
        self.auto_save_check.toggled.connect(self.detection_worker.set_auto_save)
        self.beep_check.toggled.connect(self.detection_worker.set_beep_enabled) # New: connect beep toggle

        self.detection_thread.start()

    def _create_top_bar(self):
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Model Path:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        top_layout.addWidget(self.model_path_edit)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_model)
        top_layout.addWidget(self.browse_btn)
        self.main_layout.addLayout(top_layout)

    def _create_video_display(self):
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: #000; border: 1px solid #555;")
        self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("Press 'Start Detection' or 'Test Image' to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        container_layout.addWidget(self.video_label)
        self.main_layout.addWidget(self.video_container, stretch=1)

    def _create_controls(self):
        controls_layout = QHBoxLayout()
        left_controls = QVBoxLayout()
        right_controls = QVBoxLayout()
        button_layout = QHBoxLayout()

        # --- Sliders (reconstruction/diff mode) ---
        # MSE Threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("MSE Threshold:"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(1, 1000)
        self.thresh_slider.setValue(10)
        self.thresh_slider.valueChanged.connect(self._update_mse_threshold)
        self.thresh_label = QLabel(f"{self.thresh_slider.value()/1000:.3f}")
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_label)
        left_controls.addLayout(thresh_layout)

        # CV Threshold
        cv_thresh_layout = QHBoxLayout()
        cv_thresh_layout.addWidget(QLabel("CV Threshold:"))
        self.cv_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.cv_thresh_slider.setRange(10, 200)
        self.cv_thresh_slider.setValue(40)
        self.cv_thresh_slider.valueChanged.connect(self._update_cv_threshold)
        self.cv_thresh_label = QLabel(f"{self.cv_thresh_slider.value()}")
        cv_thresh_layout.addWidget(self.cv_thresh_slider)
        cv_thresh_layout.addWidget(self.cv_thresh_label)
        left_controls.addLayout(cv_thresh_layout)

        # Contour Area
        contour_thresh_layout = QHBoxLayout()
        contour_thresh_layout.addWidget(QLabel("Contour Area:"))
        self.contour_slider = QSlider(Qt.Orientation.Horizontal)
        self.contour_slider.setRange(1, 200)
        self.contour_slider.setValue(10)
        self.contour_slider.valueChanged.connect(self._update_contour_threshold)
        self.contour_label = QLabel(f"{self.contour_slider.value()}")
        contour_thresh_layout.addWidget(self.contour_slider)
        contour_thresh_layout.addWidget(self.contour_label)
        left_controls.addLayout(contour_thresh_layout)

        # --- Camera, Resolution, and Frame Rate ---
        cam_res_layout = QHBoxLayout()
        cam_res_layout.addWidget(QLabel("Source:"))
        self.cam_combo = QComboBox()
        cam_res_layout.addWidget(self.cam_combo)
        self.list_cam_btn = QPushButton("List Cameras")
        self.list_cam_btn.clicked.connect(self._list_cameras)
        cam_res_layout.addWidget(self.list_cam_btn)

        cam_res_layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.resolution_options = [
            "Source/Native",
            "2592x100",
            "2304x100",
            "1920x100",
            "1600x900",
            "1280x720",
            "1024x576",
            "960x540",
            "800x450",
            "640x480",
        ]
        self.res_combo.addItems(self.resolution_options)
        cam_res_layout.addWidget(self.res_combo)

        cam_res_layout.addWidget(QLabel("Frame Rate:"))
        self.fps_combo = QComboBox()
        self.fps_options = ["Uncapped", "60", "30", "15"]
        self.fps_combo.addItems(self.fps_options)
        cam_res_layout.addWidget(self.fps_combo)

        left_controls.addLayout(cam_res_layout)

        # --- Right Controls ---
        self.auto_save_check = QCheckBox("Auto-save Detections")
        self.auto_save_check.toggled.connect(self._update_auto_save_status)
        right_controls.addWidget(self.auto_save_check, alignment=Qt.AlignmentFlag.AlignTop)

        # Color mode toggle
        self.color_mode_check = QCheckBox("Color Mode (HSV • Yellow)")
        self.color_mode_check.toggled.connect(self._toggle_color_mode)
        right_controls.addWidget(self.color_mode_check)

        # Beep sound toggle
        self.beep_check = QCheckBox("Enable Beep Sound")
        self.beep_check.setChecked(True) # Default to enabled
        right_controls.addWidget(self.beep_check)

        # HSV sliders
        # Hue Low
        h_low_layout = QHBoxLayout()
        h_low_layout.addWidget(QLabel("Hue Low:"))
        self.h_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_low_slider.setRange(0, 179)
        self.h_low_slider.setValue(self.detector.h_low)
        self.h_low_slider.valueChanged.connect(self._update_hsv)
        self.h_low_label = QLabel(str(self.detector.h_low))
        h_low_layout.addWidget(self.h_low_slider)
        h_low_layout.addWidget(self.h_low_label)
        right_controls.addLayout(h_low_layout)

        # Hue High
        h_high_layout = QHBoxLayout()
        h_high_layout.addWidget(QLabel("Hue High:"))
        self.h_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_high_slider.setRange(0, 179)
        self.h_high_slider.setValue(self.detector.h_high)
        self.h_high_slider.valueChanged.connect(self._update_hsv)
        self.h_high_label = QLabel(str(self.detector.h_high))
        h_high_layout.addWidget(self.h_high_slider)
        h_high_layout.addWidget(self.h_high_label)
        right_controls.addLayout(h_high_layout)

        # Sat Min
        s_min_layout = QHBoxLayout()
        s_min_layout.addWidget(QLabel("Saturation Min:"))
        self.s_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.s_min_slider.setRange(0, 255)
        self.s_min_slider.setValue(self.detector.s_min)
        self.s_min_slider.valueChanged.connect(self._update_hsv)
        self.s_min_label = QLabel(str(self.detector.s_min))
        s_min_layout.addWidget(self.s_min_slider)
        s_min_layout.addWidget(self.s_min_label)
        right_controls.addLayout(s_min_layout)

        # Val Min
        v_min_layout = QHBoxLayout()
        v_min_layout.addWidget(QLabel("Value Min:"))
        self.v_min_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_min_slider.setRange(0, 255)
        self.v_min_slider.setValue(self.detector.v_min)
        self.v_min_slider.valueChanged.connect(self._update_hsv)
        self.v_min_label = QLabel(str(self.detector.v_min))
        v_min_layout.addWidget(self.v_min_slider)
        v_min_layout.addWidget(self.v_min_label)
        right_controls.addLayout(v_min_layout)

        # buttons row
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._start_detection)
        self.start_btn.setDisabled(True)  # will be enabled when model loaded OR color mode enabled

        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self._stop_detection)
        self.stop_btn.setDisabled(True)

        self.test_image_btn = QPushButton("Test Image")
        self.test_image_btn.clicked.connect(self._test_image)
        self.test_image_btn.setDisabled(True)

        self.capture_btn = QPushButton("Capture Image")
        self.capture_btn.clicked.connect(self._capture_image)
        self.capture_btn.setDisabled(True)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.test_image_btn)
        button_layout.addWidget(self.capture_btn)
        button_layout.addStretch()

        controls_layout.addLayout(left_controls, stretch=3)
        controls_layout.addLayout(right_controls, stretch=2)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addLayout(button_layout)

        self._list_cameras()
        # Initially disable HSV controls until color mode toggled on
        self._enable_hsv_controls(False)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please load a model or enable Color Mode (HSV).")

    def _browse_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "PyTorch Model Files (*.pth)")
        if file_name:
            self._load_model_action(file_name)

    def _load_model_action(self, file_name):
        if self.detector.load_model(file_name):
            self.model_path_edit.setText(file_name)

            display_device_name = ""
            if "cuda" in self.detector.device:
                gpu_index = int(self.detector.device.split(":")[-1])
                gpu_name = torch.cuda.get_device_name(gpu_index)
                display_device_name = f"CUDA:{gpu_index} ({gpu_name})"
            else:
                display_device_name = "CPU"

            self.status_bar.showMessage(f"Model loaded from {os.path.basename(file_name)} on {display_device_name}")
            print(f"--- Anomaly Detector running on: {display_device_name} ---")
            self.start_btn.setDisabled(False)
            self.test_image_btn.setDisabled(False)
        else:
            self.model_path_edit.setText("")
            self.status_bar.showMessage("Failed to load model. Please check the file.")
            # Do not force-disable if color mode is on
            if not self.color_mode_check.isChecked():
                self.start_btn.setDisabled(True)
                self.test_image_btn.setDisabled(True)

    def _list_cameras(self):
        self.cam_combo.clear()
        available_cameras = [f"Camera {i}" for i in range(5) if cv2.VideoCapture(i, cv2.CAP_DSHOW).isOpened()]
        if available_cameras:
            self.cam_combo.addItems(available_cameras)
        else:
            self.cam_combo.addItem("No Camera Found")

    # --- Slider Update Handlers ---
    def _update_mse_threshold(self, value):
        new_thresh = value / 1000.0
        self.thresh_label.setText(f"{new_thresh:.3f}")
        self.detector.set_threshold(new_thresh)
        self._reprocess_image()

    def _update_cv_threshold(self, value):
        self.cv_thresh_label.setText(f"{value}")
        self.detector.set_cv_threshold(value)
        self._reprocess_image()

    def _update_contour_threshold(self, value):
        self.contour_label.setText(f"{value}")
        self.detector.set_contour_threshold(value)
        self._reprocess_image()

    def _update_auto_save_status(self, checked):
        if checked:
            self.status_bar.showMessage("Auto-save enabled.", 2000)
        else:
            self.status_bar.showMessage("Auto-save disabled.", 2000)

    def _toggle_color_mode(self, checked):
        self.detector.set_color_mode(checked)
        self._enable_hsv_controls(checked)
        if checked:
            # Allow running even without model
            self.start_btn.setDisabled(False)
            self.test_image_btn.setDisabled(False)
            self.status_bar.showMessage("Color Mode (HSV) enabled. You can start without a model.")
        else:
            # If no model loaded, disable again
            if not self.model_path_edit.text():
                self.start_btn.setDisabled(True)
                self.test_image_btn.setDisabled(True)
            self.status_bar.showMessage("Color Mode (HSV) disabled.")
        self._reprocess_image()

    def _enable_hsv_controls(self, enabled: bool):
        for w in [
            self.h_low_slider,
            self.h_high_slider,
            self.s_min_slider,
            self.v_min_slider,
            self.h_low_label,
            self.h_high_label,
            self.s_min_label,
            self.v_min_label,
        ]:
            w.setEnabled(enabled)

    def _update_hsv(self, _value):
        # clamp high/low coherently
        h_low = self.h_low_slider.value()
        h_high = self.h_high_slider.value()
        if h_low > h_high:
            # keep sliders consistent: move the other end
            sender = self.sender()
            if sender is self.h_low_slider:
                self.h_high_slider.setValue(h_low)
                h_high = h_low
            else:
                self.h_low_slider.setValue(h_high)
                h_low = h_high
        s_min = self.s_min_slider.value()
        v_min = self.v_min_slider.value()

        self.h_low_label.setText(str(h_low))
        self.h_high_label.setText(str(h_high))
        self.s_min_label.setText(str(s_min))
        self.v_min_label.setText(str(v_min))

        self.detector.set_hsv_thresholds(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min)
        self._reprocess_image()

    # --- Detection Control ---
    def _start_detection(self):
        if self.cam_combo.currentText() == "No Camera Found":
            self.status_bar.showMessage("Error: No camera selected or found.")
            return

        self.last_tested_image = None
        self.status_bar.showMessage("Starting detection...")

        self.frame_count = 0
        self.start_time = time.time()
        self.detection_worker.reset_counter()  # Reset worker's counter

        res_text = self.res_combo.currentText()
        resolution = tuple(map(int, res_text.split('x'))) if res_text != "Source/Native" else None

        fps_text = self.fps_combo.currentText()
        fps_limit = int(fps_text) if fps_text != "Uncapped" else None

        self.video_thread = VideoThread(camera_index=self.cam_combo.currentIndex(), resolution=resolution, fps_limit=fps_limit)
        self.video_thread.change_pixmap_signal.connect(self.update_image)  # lightweight pass-through
        self.video_thread.start()

        self.is_detection_running = True
        self._toggle_controls(enable=False)

    def _stop_detection(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()

        self.is_detection_running = False
        if self.last_tested_image is None and (self.video_label.pixmap() is None or self.video_label.pixmap().isNull()):
            self.video_label.setText("Press 'Start Detection' or 'Test Image' to begin")

        self.status_bar.showMessage("Detection stopped.")
        self._toggle_controls(enable=True)

    def _toggle_controls(self, enable):
        self.start_btn.setDisabled(not enable)
        self.browse_btn.setDisabled(not enable)
        self.cam_combo.setDisabled(not enable)
        self.res_combo.setDisabled(not enable)
        self.list_cam_btn.setDisabled(not enable)
        self.test_image_btn.setDisabled(not enable)
        self.stop_btn.setDisabled(enable)
        self.capture_btn.setDisabled(enable)
        self.fps_combo.setDisabled(not enable)

    def _capture_image(self):
        """Saves the current live frame on button press."""
        if not self.is_detection_running or self.current_frame is None:
            self.status_bar.showMessage("Capture only available during live detection.", 3000)
            return

        original_frame = self.current_frame.copy()

        # Re-run detection on the last frame to draw boxes consistently
        processed_frame, _, is_anomaly = self.detector.process_frame(original_frame.copy())

        anomaly_count = self.detection_worker.anomaly_count

        # --- Draw overlay ---
        h, w, _ = processed_frame.shape
        margin = 10

        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(processed_frame, fps_text, (margin, h - margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        res_text = f"{w}x{h}"
        (res_width, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(processed_frame, res_text, (w - res_width - margin, h - margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        det_text = f"Detections: {anomaly_count}"
        (det_width, _), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(processed_frame, det_text, (w - det_width - margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # --- Save files logic ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp}.jpg"

        detections_dir = os.path.join("output", "captures_detected")
        original_dir = os.path.join("output", "captures_original")

        os.makedirs(detections_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)

        detection_save_path = os.path.join(detections_dir, filename)
        original_save_path = os.path.join(original_dir, filename)

        cv2.imwrite(detection_save_path, processed_frame)
        cv2.imwrite(original_save_path, original_frame)

        self.status_bar.showMessage(f"Image captured and saved as {filename}", 4000)

    # --- Image/Frame Processing ---
    def _test_image(self):
        if self.is_detection_running:
            self._stop_detection()

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not file_name:
            return

        cv_img = cv2.imread(file_name)
        if cv_img is None:
            self.status_bar.showMessage(f"Error: Could not read image file", 4000)
            self.last_tested_image = None
            return

        self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_name)}")
        self.last_tested_image = cv_img
        self._reprocess_image()

    def _reprocess_image(self):
        if self.last_tested_image is None or self.is_detection_running:
            return

        processed_frame, mse, is_anomaly = self.detector.process_frame(self.last_tested_image.copy())

        if is_anomaly:
            h, w, _ = processed_frame.shape
            text = "Detections: 1"
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            (text_width, _), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_x = w - text_width - 10
            cv2.putText(processed_frame, text, (text_x, 50), font_face, font_scale, (0, 0, 255), thickness)

        mode = "HSV" if self.detector.color_mode else ("Model" if self.detector.model not in (None, "loaded") else "Dummy")
        status_text = f"Last Image Test | Mode: {mode} | MSE: {mse:.4f} | Anomaly: {'Yes' if is_anomaly else 'No'}"
        self.status_bar.showMessage(status_text)

        qt_img = self.convert_cv_qt(processed_frame)
        self.video_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        self.trigger_process.emit(cv_img)

    @pyqtSlot(np.ndarray, np.ndarray, float, bool, int)
    def display_processed_frame(self, processed_frame, original_frame, mse, is_anomaly, anomaly_count):
        self.current_frame = original_frame  # Store the original frame for capture

        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        # Draw info on frame
        h, w, _ = processed_frame.shape
        margin = 10

        # Display FPS at the bottom-left
        fps_text = f"FPS: {self.fps:.2f}"
        cv2.putText(processed_frame, fps_text, (margin, h - margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Display Resolution at the bottom-right
        res_text = f"{w}x{h}"
        (res_width, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(processed_frame, res_text, (w - res_width - margin, h - margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Display detection count at the top-right
        det_text = f"Detections: {anomaly_count}"
        (det_width, _), _ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(processed_frame, det_text, (w - det_width - margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Convert and display
        qt_img = self.convert_cv_qt(processed_frame)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.video_container.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    # --- Settings and Close Event ---
    def _save_settings(self):
        settings = QSettings("MyCompany", "AnomalyApp")
        settings.setValue("model_path", self.model_path_edit.text())
        settings.setValue("mse_threshold", self.thresh_slider.value())
        settings.setValue("cv_threshold", self.cv_thresh_slider.value())
        settings.setValue("contour_area", self.contour_slider.value())
        settings.setValue("camera_index", self.cam_combo.currentIndex())
        settings.setValue("resolution_text", self.res_combo.currentText())
        settings.setValue("fps_limit_text", self.fps_combo.currentText())
        settings.setValue("auto_save", self.auto_save_check.isChecked())
        settings.setValue("beep_enabled", self.beep_check.isChecked()) # New: save beep state
        # color settings
        settings.setValue("color_mode", self.color_mode_check.isChecked())
        settings.setValue("h_low", self.h_low_slider.value())
        settings.setValue("h_high", self.h_high_slider.value())
        settings.setValue("s_min", self.s_min_slider.value())
        settings.setValue("v_min", self.v_min_slider.value())

    def _load_settings(self):
        settings = QSettings("MyCompany", "AnomalyApp")
        model_path = settings.value("model_path", "")
        if model_path and os.path.exists(model_path):
            self._load_model_action(model_path)

        self.thresh_slider.setValue(settings.value("mse_threshold", 10, type=int))
        self.cv_thresh_slider.setValue(settings.value("cv_threshold", 40, type=int))
        self.contour_slider.setValue(settings.value("contour_area", 10, type=int))

        if self.cam_combo.count() > 0:
            cam_index = settings.value("camera_index", 0, type=int)
            if cam_index < self.cam_combo.count():
                self.cam_combo.setCurrentIndex(cam_index)

        res_text = settings.value("resolution_text", "1280x720")
        if res_text in self.resolution_options:
            self.res_combo.setCurrentText(res_text)

        fps_text = settings.value("fps_limit_text", "Uncapped")
        if fps_text in self.fps_options:
            self.fps_combo.setCurrentText(fps_text)

        auto_save = settings.value("auto_save", False, type=bool)
        self.auto_save_check.setChecked(auto_save)
        self.detection_worker.set_auto_save(auto_save)  # Also update worker state on load

        beep_enabled = settings.value("beep_enabled", True, type=bool) # New: load beep state
        self.beep_check.setChecked(beep_enabled)
        self.detection_worker.set_beep_enabled(beep_enabled) # Also update worker state on load

        # color settings
        color_mode = settings.value("color_mode", False, type=bool)
        self.color_mode_check.setChecked(color_mode)

        h_low = settings.value("h_low", self.detector.h_low, type=int)
        h_high = settings.value("h_high", self.detector.h_high, type=int)
        s_min = settings.value("s_min", self.detector.s_min, type=int)
        v_min = settings.value("v_min", self.detector.v_min, type=int)

        self.h_low_slider.setValue(h_low)
        self.h_high_slider.setValue(h_high)
        self.s_min_slider.setValue(s_min)
        self.v_min_slider.setValue(v_min)

        self.h_low_label.setText(str(h_low))
        self.h_high_label.setText(str(h_high))
        self.s_min_label.setText(str(s_min))
        self.v_min_label.setText(str(v_min))

        self.detector.set_hsv_thresholds(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min)
        self.detector.set_color_mode(color_mode)
        self._enable_hsv_controls(color_mode)

        # if color mode is on and no model loaded, allow running
        if color_mode and not model_path:
            self.start_btn.setDisabled(False)
            self.test_image_btn.setDisabled(False)

    def closeEvent(self, event):
        self._save_settings()
        self._stop_detection()

        # Properly shut down the worker thread
        self.detection_thread.quit()
        self.detection_thread.wait()

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
