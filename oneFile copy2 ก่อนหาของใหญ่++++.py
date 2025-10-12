import sys
import cv2
import os
import time
from datetime import datetime
import numpy as np

from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit, QSlider, QCheckBox,
                             QStatusBar, QComboBox, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QRect, QSettings, QObject, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

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
    A dummy AnomalyDetector class for demonstration.
    In a real scenario, this would load a PyTorch/TensorFlow model.
    """
    def __init__(self):
        self.model = None
        self.device = "CPU"
        self.mse_threshold = 0.01
        self.cv_threshold = 40
        self.contour_area_threshold = 10

    def load_model(self, model_path):
        """Simulates loading a model."""
        if model_path and model_path.endswith(".pth"):
            self.model = "loaded"
            print(f"Dummy model loaded from {model_path}")
            # In a real app, you would check for CUDA here.
            # import torch
            # self.device = "cuda" if torch.cuda.is_available() else "cpu"
            return True
        return False

    def set_threshold(self, value):
        self.mse_threshold = value

    def set_cv_threshold(self, value):
        self.cv_threshold = value

    def set_contour_threshold(self, value):
        self.contour_area_threshold = value

    def process_frame(self, frame):
        """
        Processes a single frame to detect anomalies.
        """
        # Make a copy to draw on, preserving the original frame for potential saving
        output_frame = frame.copy()
        
        # --- Dummy Processing Logic ---
        # This simulates a reconstruction autoencoder model.
        # It blurs the image and then finds differences.
        reconstructed = cv2.GaussianBlur(frame, (7, 7), 0)
        
        # Calculate difference and threshold
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
        
        # Use absdiff to find the difference
        diff = cv2.absdiff(gray_frame, gray_reconstructed)
        
        # Threshold the difference image
        _, thresh = cv2.threshold(diff, self.cv_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of the differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- Anomaly Detection ---
        mse = (np.square(diff)).mean() # Dummy MSE for status bar
        is_anomaly = False

        for contour in contours:
            if cv2.contourArea(contour) > self.contour_area_threshold:
                is_anomaly = True
                (cx, cy, cw, ch) = cv2.boundingRect(contour)
                # Draw red bounding box for the detected anomaly on the output frame
                cv2.rectangle(output_frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
                
        if is_anomaly:
            cv2.putText(output_frame, "Anomaly Detected", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3) # Increased font size and thickness

        return output_frame, mse, is_anomaly

# --- Thread for camera feed ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0, resolution=None):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self._run_flag = True

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
            else:
                self.msleep(50)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# --- Worker for detection processing ---
class DetectionWorker(QObject):
    result_ready = pyqtSignal(np.ndarray, np.ndarray, float, bool, int) # processed_frame, original_frame, mse, is_anomaly, anomaly_count
    status_update = pyqtSignal(str)
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.is_busy = False
        self.auto_save = False
        self.last_save_time = 0
        self.anomaly_count = 0

    @pyqtSlot(np.ndarray)
    def process_frame(self, cv_img):
        if self.is_busy:
            return
            
        self.is_busy = True
        
        original_frame = cv_img.copy()
        processed_frame, mse, is_anomaly = self.detector.process_frame(cv_img)

        if is_anomaly:
            self.anomaly_count += 1
        
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
    trigger_process = pyqtSignal(np.ndarray) # Signal to trigger the worker

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
        self._setup_detection_thread() # New method to set up worker thread

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

        self.detection_thread.start()

    def _create_top_bar(self):
        # ... (no changes here)
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
        # ... (no changes here)
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
        # ... (no changes here, except connecting auto_save_check to a different slot later)
        controls_layout = QHBoxLayout()
        left_controls = QVBoxLayout()
        right_controls = QVBoxLayout()
        button_layout = QHBoxLayout()

        # --- Sliders ---
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

        # --- Camera and Resolution ---
        cam_res_layout = QHBoxLayout()
        cam_res_layout.addWidget(QLabel("Source:"))
        self.cam_combo = QComboBox()
        cam_res_layout.addWidget(self.cam_combo)
        self.list_cam_btn = QPushButton("List Cameras")
        self.list_cam_btn.clicked.connect(self._list_cameras)
        cam_res_layout.addWidget(self.list_cam_btn)
        
        cam_res_layout.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.resolution_options =  ["Source/Native","2592x1440","2304x1296","1920x1080","1600x900","1280x720","1024x576","960x540","800x450","640x480"]
        self.res_combo.addItems(self.resolution_options)
        cam_res_layout.addWidget(self.res_combo)
        left_controls.addLayout(cam_res_layout)
        
        # --- Right Controls ---
        self.auto_save_check = QCheckBox("Auto-save Detections")
        self.auto_save_check.toggled.connect(self._update_auto_save_status) # Connect signal to slot
        right_controls.addWidget(self.auto_save_check, alignment=Qt.AlignmentFlag.AlignTop)

        # --- Buttons ---
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._start_detection)
        self.start_btn.setDisabled( True)
        
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
        controls_layout.addLayout(right_controls, stretch=1)
        self.main_layout.addLayout(controls_layout)
        self.main_layout.addLayout(button_layout)
        
        self._list_cameras()

    def _create_status_bar(self):
        # ... (no changes here)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Please load a model file.")
    
    def _browse_model(self):
        # ... (no changes here)
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "PyTorch Model Files (*.pth)")
        if file_name:
            self._load_model_action(file_name)
    
    def _load_model_action(self, file_name):
        # ... (no changes here)
        if self.detector.load_model(file_name):
            self.model_path_edit.setText(file_name)
            device_name = str(self.detector.device).upper()
            self.status_bar.showMessage(f"Model loaded from {os.path.basename(file_name)} on {device_name}")
            self.start_btn.setDisabled(False)
            self.test_image_btn.setDisabled(False)
        else:
            self.model_path_edit.setText("")
            self.status_bar.showMessage("Failed to load model. Please check the file.")
            self.start_btn.setDisabled(True)
            self.test_image_btn.setDisabled(True)
            
    def _list_cameras(self):
        # ... (no changes here)
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
        """Updates the status bar and the worker's auto_save flag."""
        # This function now mainly provides user feedback. The actual logic is handled by the worker.
        if checked:
            self.status_bar.showMessage("Auto-save enabled.", 2000)
        else:
            self.status_bar.showMessage("Auto-save disabled.", 2000)
        # The toggled signal is already connected to the worker's set_auto_save slot.

    # --- Detection Control ---
    def _start_detection(self):
        if self.cam_combo.currentText() == "No Camera Found":
            self.status_bar.showMessage("Error: No camera selected or found.")
            return

        self.last_tested_image = None 
        self.status_bar.showMessage("Starting detection...")
        
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_worker.reset_counter() # Reset worker's counter
        
        res_text = self.res_combo.currentText()
        resolution = tuple(map(int, res_text.split('x'))) if res_text != "Source/Native" else None
        
        self.video_thread = VideoThread(camera_index=self.cam_combo.currentIndex(), resolution=resolution)
        self.video_thread.change_pixmap_signal.connect(self.update_image) # Connect to the new lightweight update function
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
        self.capture_btn.setDisabled(enable) # Note: Capture button enabled when running now

    def _capture_image(self):
        """Saves the current live frame on button press."""
        if not self.is_detection_running or self.current_frame is None:
            self.status_bar.showMessage("Capture only available during live detection.", 3000)
            return

        # Use a copy of the current frame for saving the original
        original_frame = self.current_frame.copy()

        # Create the processed frame with detections
        # We need to re-run detection here to get the boxes for the latest frame
        processed_frame, _, is_anomaly = self.detector.process_frame(original_frame.copy())
        
        anomaly_count = self.detection_worker.anomaly_count # Get current count from worker
        
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
        # ... (no changes here)
        if self.is_detection_running:
            self._stop_detection()

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not file_name: return

        cv_img = cv2.imread(file_name)
        if cv_img is None:
            self.status_bar.showMessage(f"Error: Could not read image file", 4000)
            self.last_tested_image = None
            return
        
        self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_name)}")
        self.last_tested_image = cv_img
        self._reprocess_image()

    def _reprocess_image(self):
        # ... (no changes here)
        if self.last_tested_image is None or self.is_detection_running:
            return
        
        processed_frame, mse, is_anomaly = self.detector.process_frame(self.last_tested_image.copy())
        
        if is_anomaly:
            h, w, _ = processed_frame.shape
            text = "Detections: 1"
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            text_x = w - text_width - 10 
            cv2.putText(processed_frame, text, (text_x, 50), font_face, font_scale, (0, 0, 255), thickness)


        status_text = f"Last Image Test | MSE: {mse:.4f} | Anomaly: {'Yes' if is_anomaly else 'No'}"
        self.status_bar.showMessage(status_text)
        
        qt_img = self.convert_cv_qt(processed_frame)
        self.video_label.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """ This function is now very lightweight. It just triggers the worker. """
        self.trigger_process.emit(cv_img)

    @pyqtSlot(np.ndarray, np.ndarray, float, bool, int)
    def display_processed_frame(self, processed_frame, original_frame, mse, is_anomaly, anomaly_count):
        """ This slot receives the finished work from the worker thread and updates the GUI. """
        self.current_frame = original_frame # Store the original frame for capture
        
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
        # ... (no changes here)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.video_container.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    # --- Settings and Close Event ---
    def _save_settings(self):
        # ... (no changes here)
        settings = QSettings("MyCompany", "AnomalyApp")
        settings.setValue("model_path", self.model_path_edit.text())
        settings.setValue("mse_threshold", self.thresh_slider.value())
        settings.setValue("cv_threshold", self.cv_thresh_slider.value())
        settings.setValue("contour_area", self.contour_slider.value())
        settings.setValue("camera_index", self.cam_combo.currentIndex())
        settings.setValue("resolution_text", self.res_combo.currentText())
        settings.setValue("auto_save", self.auto_save_check.isChecked())

    def _load_settings(self):
        # ... (no changes here)
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

        auto_save = settings.value("auto_save", False, type=bool)
        self.auto_save_check.setChecked(auto_save)
        self.detection_worker.set_auto_save(auto_save) # Also update worker state on load

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