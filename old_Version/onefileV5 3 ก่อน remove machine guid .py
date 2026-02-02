import sys
import cv2
import os
import time
import shutil
import random
from datetime import datetime
import numpy as np
import torch
from torchvision.transforms import functional as F
import winsound # Import for beep sound
import threading

from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QSlider, QCheckBox,
    QStatusBar, QComboBox, QSizePolicy, QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QObject, pyqtSlot, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor
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

def _fourcc_to_string(fourcc_value):
    """Convert numeric FOURCC to readable string."""
    try:
        fourcc_int = int(fourcc_value) & 0xFFFFFFFF
        if fourcc_int == 0:
            return "UNKNOWN (0x00000000)"
        chars = fourcc_int.to_bytes(4, byteorder="little")
        printable = "".join(chr(c) if 32 <= c <= 126 else "." for c in chars)
        return f"{printable} (0x{fourcc_int:08X})"
    except Exception:
        return f"UNKNOWN ({fourcc_value})"


def open_camera_by_index(index, resolution=None, prefer_backend="DSHOW", fps=None, preferred_fourcc=None, warmup_frames=10):
    pref = (prefer_backend or "AUTO").upper()
    if pref == "MSMF":
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif pref == "DSHOW":
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
        return None, None
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    if preferred_fourcc and preferred_fourcc.upper() != "AUTO":
        try:
            code = cv2.VideoWriter_fourcc(*preferred_fourcc[:4])
            if code:
                cap.set(cv2.CAP_PROP_FOURCC, code)
                print(f"[LOG {time.time():.2f}] Requested FOURCC {preferred_fourcc} for camera {index}.")
        except Exception as err:
            print(f"[WARN {time.time():.2f}] Unable to apply FOURCC {preferred_fourcc} on camera {index}: {err}")
    fourcc_value = cap.get(cv2.CAP_PROP_FOURCC)
    fourcc_mode = _fourcc_to_string(fourcc_value)
    return cap, fourcc_mode


# def open_camera_by_index(index, resolution=None, prefer_backend="MSMF", fps=None, preferred_fourcc=None, warmup_frames=10):
#     pref = (prefer_backend or "AUTO").upper()
#     if pref == "MSMF":
#         backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
#     elif pref == "DSHOW":
#         backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
#     else:
#         backends = [cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW]

#     fourcc_list = preferred_fourcc or [ "YUY2","MJPG",  "H264", None]

#     def _apply_settings(cap, fourcc_code):
#         if fourcc_code:
#             cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_code))
#         if resolution:
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
#         if fps:
#             cap.set(cv2.CAP_PROP_FPS, fps)

#     for backend in backends:
#         base_cap = cv2.VideoCapture(index, backend)
#         if not base_cap.isOpened():
#             base_cap.release()
#             continue

#         for attempt, fourcc_code in enumerate(fourcc_list):
#             if attempt > 0:
#                 base_cap.release()
#                 base_cap = cv2.VideoCapture(index, backend)
#                 if not base_cap.isOpened():
#                     break

#             _apply_settings(base_cap, fourcc_code)
#             ok = True
#             for _ in range(max(1, warmup_frames)):
#                 ret, _ = base_cap.read()
#                 if not ret:
#                     ok = False
#                     break
#             if ok:
#                 if fourcc_code:
#                     print(f"[LOG {time.time():.2f}] Camera {index}: using FOURCC {fourcc_code} via backend {backend}.")
#                 else:
#                     print(f"[LOG {time.time():.2f}] Camera {index}: using default FOURCC via backend {backend}.")
#                 return base_cap

#         base_cap.release()

#     print(f"[LOG {time.time():.2f}] Unable to configure camera {index} with requested formats.")
#     return None



class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.cv_threshold = 40
        self.contour_area_threshold = 10
        self.mse_threshold = 0.01
        self.h_low, self.h_high = 15, 35
        self.s_min, self.v_min = 60, 120
        self.h2_low, self.h2_high = 75, 95
        self.s2_min, self.v2_min = 60, 120
        self.h3_low, self.h3_high = 105, 125
        self.s3_min, self.v3_min = 60, 120
        self.mode = 'recon'
        self.first_inference = True # <--- LOGGING FLAG
        self.primary_hue_enabled = True
        self.secondary_hue_enabled = True
        self.tertiary_hue_enabled = True

    def set_mode(self, mode: str):
        self.mode = mode
    def set_hsv_thresholds(self, h_low=None, h_high=None, s_min=None, v_min=None):
        if h_low is not None:  self.h_low  = int(max(0, min(179, h_low)))
        if h_high is not None: self.h_high = int(max(0, min(179, h_high)))
        if self.h_low > self.h_high:
            self.h_low, self.h_high = self.h_high, self.h_low
        if s_min is not None:  self.s_min  = int(max(0, min(255, s_min)))
        if v_min is not None:  self.v_min  = int(max(0, min(255, v_min)))

    def set_hsv_secondary(self, h_low=None, h_high=None, s_min=None, v_min=None):
        if h_low is not None:  self.h2_low  = int(max(0, min(179, h_low)))
        if h_high is not None: self.h2_high = int(max(0, min(179, h_high)))
        if self.h2_low > self.h2_high:
            self.h2_low, self.h2_high = self.h2_high, self.h2_low
        if s_min is not None:  self.s2_min  = int(max(0, min(255, s_min)))
        if v_min is not None:  self.v2_min  = int(max(0, min(255, v_min)))
    def set_hsv_tertiary(self, h_low=None, h_high=None, s_min=None, v_min=None):
        if h_low is not None:  self.h3_low  = int(max(0, min(179, h_low)))
        if h_high is not None: self.h3_high = int(max(0, min(179, h_high)))
        if self.h3_low > self.h3_high:
            self.h3_low, self.h3_high = self.h3_high, self.h3_low
        if s_min is not None:  self.s3_min  = int(max(0, min(255, s_min)))
        if v_min is not None:  self.v3_min  = int(max(0, min(255, v_min)))
    def set_cv_threshold(self, value):
        self.cv_threshold = int(value)
    def set_contour_threshold(self, value):
        self.contour_area_threshold = int(value)
    def set_threshold(self, value):
        self.mse_threshold = float(value)
    def set_primary_hue_enabled(self, enabled: bool):
        self.primary_hue_enabled = bool(enabled)
    def set_secondary_hue_enabled(self, enabled: bool):
        self.secondary_hue_enabled = bool(enabled)
    def set_tertiary_hue_enabled(self, enabled: bool):
        self.tertiary_hue_enabled = bool(enabled)

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
        if not self.primary_hue_enabled:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.h_low, self.s_min, self.v_min], dtype=np.uint8)
        upper1 = np.array([self.h_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(self.h_low-5,0), self.s_min, self.v_min], dtype=np.uint8)
        upper2 = np.array([min(self.h_high+5,179), 255, 255], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    def _mask_color_secondary(self, frame):
        if not self.secondary_hue_enabled:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.h2_low, self.s2_min, self.v2_min], dtype=np.uint8)
        upper1 = np.array([self.h2_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(self.h2_low-5,0), self.s2_min, self.v2_min], dtype=np.uint8)
        upper2 = np.array([min(self.h2_high+5,179), 255, 255], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    def _mask_color_tertiary(self, frame):
        if not self.tertiary_hue_enabled:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([self.h3_low, self.s3_min, self.v3_min], dtype=np.uint8)
        upper1 = np.array([self.h3_high, 255, 255], dtype=np.uint8)
        lower2 = np.array([max(self.h3_low-5,0), self.s3_min, self.v3_min], dtype=np.uint8)
        upper2 = np.array([min(self.h3_high+5,179), 255, 255], dtype=np.uint8)
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
            mask_primary = self._mask_color(frame)
            mask_secondary = self._mask_color_secondary(frame)
            mask_tertiary = self._mask_color_tertiary(frame)
            contours_primary = self._contours_from_mask(mask_primary)
            contours_secondary = self._contours_from_mask(mask_secondary)
            contours_tertiary = self._contours_from_mask(mask_tertiary)
            out = frame.copy()
            for c in contours_primary:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,255), 2)
            for c in contours_secondary:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
            for c in contours_tertiary:
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,0), 2)
            all_contours = contours_primary + contours_secondary + contours_tertiary
            is_anom = len(all_contours) > 0
            if is_anom:
                cv2.putText(out, 'Color Anomaly (Hue1/Hue2/Hue3)', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
            mask_means = [mask_primary.mean(), mask_secondary.mean(), mask_tertiary.mean()]
            mse = float(sum(mask_means)/(len(mask_means)*255.0))
            return out, mse, is_anom, all_contours

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
            return out, mse, is_anom, contours

        # HYBRID OR
        mask_recon, mse = self._mask_recon_or_dummy(frame, is_first_frame)
        mask_color_primary = self._mask_color(frame)
        mask_color_secondary = self._mask_color_secondary(frame)
        mask_color_tertiary = self._mask_color_tertiary(frame)
        contours_recon = self._contours_from_mask(mask_recon)
        contours_color_primary = self._contours_from_mask(mask_color_primary)
        contours_color_secondary = self._contours_from_mask(mask_color_secondary)
        contours_color_tertiary = self._contours_from_mask(mask_color_tertiary)
        out = frame.copy()
        for c in contours_recon:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,0,255), 2)
        for c in contours_color_primary:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,255), 2)
        for c in contours_color_secondary:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,255,0), 2)
        for c in contours_color_tertiary:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w,y+h), (255,0,0), 2)
        all_contours = contours_recon + contours_color_primary + contours_color_secondary + contours_color_tertiary
        is_anom = len(all_contours) > 0
        if is_anom:
            cv2.putText(out, 'HYBRID: Color OR Recon', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        mask_mean_component = (mask_color_primary.mean() + mask_color_secondary.mean() + mask_color_tertiary.mean())/(3*255.0)
        mse_hybrid = 0.5*mse + 0.5*mask_mean_component
        return out, float(mse_hybrid), is_anom, all_contours

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fourcc_signal = pyqtSignal(str)
    def __init__(self, camera_index=0, resolution=None, fps_limit=None, prefer_backend="AUTO", preferred_fourcc=None):
        super().__init__()
        self.camera_index = camera_index
        self.resolution = resolution
        self._run_flag = True
        self.target_fps = fps_limit
        self.sleep_duration_ms = int(1000 / fps_limit) if fps_limit else 0
        self.prefer_backend = prefer_backend
        self.preferred_fourcc = preferred_fourcc
        self._paused = False
        self.active_fourcc = None
    def run(self):
        print(f"[LOG {time.time():.2f}] VideoThread entered run() method.")
        print(f"[LOG {time.time():.2f}] Starting to open camera index {self.camera_index} using backend {self.prefer_backend}...")
        start_cam_time = time.time()
        requested_fourcc = self.preferred_fourcc or "Auto"
        print(f"[LOG {time.time():.2f}] Preferred FOURCC request: {requested_fourcc}")
        cap, fourcc_mode = open_camera_by_index(
            self.camera_index,
            self.resolution,
            prefer_backend=self.prefer_backend,
            fps=self.target_fps,
            preferred_fourcc=self.preferred_fourcc
        )
        end_cam_time = time.time()
        print(f"[LOG {end_cam_time:.2f}] Camera opened. Took {end_cam_time - start_cam_time:.4f} seconds.")
        if cap is None:
            print(f"Error: Cannot open camera {self.camera_index} with any backend")
            self.fourcc_signal.emit("Unavailable")
            return
        self.active_fourcc = fourcc_mode
        self.fourcc_signal.emit(fourcc_mode)
        print(f"[LOG {time.time():.2f}] Active FOURCC mode: {fourcc_mode}")
        print(f"[LOG {time.time():.2f}] Entering main capture loop...")
        while self._run_flag:
            if self._paused:
                self.msleep(50)
                continue
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
        self._paused = False
        self.wait()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

class DetectionWorker(QObject):
    result_ready = pyqtSignal(np.ndarray, np.ndarray, float, bool, int)
    status_update = pyqtSignal(str)
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.is_busy = False
        self.auto_save = False
        self.last_save_time = 0
        self.random_save_enabled = False
        self.random_save_probability = 0.05
        self.random_save_min_interval = 2.0
        self.last_random_save_time = 0
        self.anomaly_count = 0
        self.beep_enabled = False
        self.tripwire_enabled = False
        self.recently_detected_centroids = []
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
        processed_frame, mse, is_anomaly_present, contours = self.detector.process_frame(cv_img, is_first_frame)

        now_ts = time.time()
        is_new_anomaly_found = False

        if self.tripwire_enabled and is_anomaly_present:
            # 1. Clean up old centroids
            expiry_time = 2.0 # seconds
            self.recently_detected_centroids = [
                (c, t) for (c, t) in self.recently_detected_centroids if now_ts - t < expiry_time
            ]

            newly_detected_centroids = []
            distance_threshold = 50 # pixels

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)

                # 2. Check if this is a new object
                is_new = True
                for existing_centroid, _ in self.recently_detected_centroids:
                    dist = np.sqrt((centroid[0] - existing_centroid[0])**2 + (centroid[1] - existing_centroid[1])**2)
                    if dist < distance_threshold:
                        is_new = False
                        break
                
                # 3. If it's new, mark it and add to our lists
                if is_new:
                    is_new_anomaly_found = True
                    newly_detected_centroids.append((centroid, now_ts))

            # 4. Add the truly new objects to the main list
            self.recently_detected_centroids.extend(newly_detected_centroids)
            is_anomaly = is_new_anomaly_found
        else:
            # Original behavior when tripwire is off
            is_anomaly = is_anomaly_present
            if is_anomaly:
                is_new_anomaly_found = True


        if is_first_frame:
            end_process_time = time.time()
            print(f"[LOG {end_process_time:.2f}] FIRST frame processed. Took {end_process_time - start_process_time:.4f} seconds.")
            self.first_frame_processed = True

        if is_new_anomaly_found:
            self.anomaly_count += 1
            self._play_beep_async()

        if is_new_anomaly_found and self.auto_save and now_ts - self.last_save_time > 2:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            fname = f"anomaly_{ts}.png"
            det_dir = os.path.join('output','detections')
            ori_dir = os.path.join('output','original')
            os.makedirs(det_dir, exist_ok=True)
            os.makedirs(ori_dir, exist_ok=True)
            cv2.imwrite(os.path.join(det_dir,fname), processed_frame)
            cv2.imwrite(os.path.join(ori_dir,fname), original_frame)
            self.status_update.emit(f"Anomaly saved as {fname}")
            self.last_save_time = now_ts

        if self.random_save_enabled and (now_ts - self.last_random_save_time) >= self.random_save_min_interval:
            if random.random() < self.random_save_probability:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                rand_suffix = random.randint(0, 9999)
                base_name = f"train_{ts}_{rand_suffix:04d}.png"
                det_dir = os.path.join('output', 'training_samples', 'detected')
                ori_dir = os.path.join('output', 'training_samples', 'original')
                os.makedirs(det_dir, exist_ok=True)
                os.makedirs(ori_dir, exist_ok=True)
                cv2.imwrite(os.path.join(det_dir, base_name), processed_frame)
                cv2.imwrite(os.path.join(ori_dir, base_name), original_frame)
                self.status_update.emit(f"Training sample saved as {base_name}")
                self.last_random_save_time = now_ts

        self.result_ready.emit(processed_frame, original_frame, mse, is_anomaly, self.anomaly_count)
        self.is_busy = False

    @pyqtSlot(bool)
    def set_beep_enabled(self, status):
        self.beep_enabled = status
    @pyqtSlot(bool)
    def set_auto_save(self, status):
        self.auto_save = status
    @pyqtSlot(bool)
    def set_random_save_enabled(self, status):
        self.random_save_enabled = status
        if not status:
            self.last_random_save_time = 0
    @pyqtSlot(bool)
    def set_tripwire_enabled(self, status):
        self.tripwire_enabled = status
        if not status:
            self.recently_detected_centroids.clear()

    def reset_counter(self):
        self.anomaly_count = 0
        self.first_frame_processed = False
        self.detector.first_inference = True

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)
    wheel = pyqtSignal(int)
    pan = pyqtSignal(int, int)

    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self._panning = False
        self._pan_start_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.main_window.zoom_level > 1.0:
                self._panning = True
                self._pan_start_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
            else:
                pos = event.position() if hasattr(event, "position") else event.pos()
                point = pos.toPoint() if hasattr(pos, "toPoint") else QPoint(int(pos.x()), int(pos.y()))
                self.clicked.emit(point.x(), point.y())
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start_pos
            self.pan.emit(delta.x(), delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.wheel.emit(delta)
        event.accept()


class VideoWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.setWindowTitle('Live Feed')
        self.resize(960, 720)
        self.setStyleSheet('background-color:#000;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.video_label = ClickableLabel(main_window, "Press 'Start Detection' or 'Test Image' to begin")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.video_label)


class MainWindow(QMainWindow):
    # ... (ส่วน __init__ และอื่นๆ เหมือนเดิม) ...
    trigger_process = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Live Anomaly Detection GUI (Hybrid OR + Folder Clear)')
        self.setGeometry(100,100,1200,800)
        self.detector = AnomalyDetector()
        self.is_detection_running = False
        self.is_paused = False
        self.current_frame = None
        self.last_tested_image = None
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0.0
        self._last_pixmap_size = (0, 0)
        self.focus_measure = 0.0
        self.show_video_labels = True
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self._sample_state = {
            'primary': {'pending': None, 'timestamp': 0.0, 'first': None, 'second': None, 'combined': None},
            'secondary': {'pending': None, 'timestamp': 0.0, 'first': None, 'second': None, 'combined': None},
            'tertiary': {'pending': None, 'timestamp': 0.0, 'first': None, 'second': None, 'combined': None},
        }
        self._current_sample_target = 'primary'
        self.hsv_target_locked = False
        self.machine_guide_enabled = True
        self.center_marker_enabled = True
        self.video_window = VideoWindow(self)
        self.video_window.video_label.clicked.connect(self._handle_video_click)
        self.video_window.video_label.wheel.connect(self._handle_zoom)
        self.video_window.video_label.pan.connect(self._handle_pan)
        self.central_widget = QWidget(); self.main_layout = QVBoxLayout(self.central_widget); self.setCentralWidget(self.central_widget)
        self.main_layout.setContentsMargins(12, 8, 12, 12)
        self.main_layout.setSpacing(8)
        self._create_video_display(); self._create_controls(); self._create_status_bar(); self._setup_detection_thread()
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
        self.tripwire_check.toggled.connect(self.detection_worker.set_tripwire_enabled)
        self.detection_thread.start()
        self.detection_worker.set_beep_enabled(self.beep_check.isChecked())
        self._update_random_save_status(self.random_save_check.isChecked())

    def _create_top_bar(self):
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(QLabel('Model Path:'))
        self.model_path_edit = QLineEdit(); self.model_path_edit.setReadOnly(True); top.addWidget(self.model_path_edit)
        self.browse_btn = QPushButton('Browse...'); self.browse_btn.clicked.connect(self._browse_model); top.addWidget(self.browse_btn)
        return top

    def _create_video_display(self):
        self.video_window.show()
        self.video_window.raise_()
        self.video_window.activateWindow()

    def _create_controls(self):
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(16)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 6, 0, 0)
        buttons.setSpacing(8)
        top_bar_layout = self._create_top_bar()

        # thresholds
        thr = QHBoxLayout(); thr.addWidget(QLabel('MSE Threshold:'))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal); self.thresh_slider.setRange(1,1000); self.thresh_slider.setValue(10); self.thresh_slider.valueChanged.connect(self._update_mse_threshold)
        self.thresh_label = QLabel(f"{self.thresh_slider.value()/1000:.3f}")
        thr.addWidget(self.thresh_slider); thr.addWidget(self.thresh_label)
        cvthr = QHBoxLayout(); cvthr.addWidget(QLabel('CV Threshold:'))
        self.cv_thresh_slider = QSlider(Qt.Orientation.Horizontal); self.cv_thresh_slider.setRange(5,200); self.cv_thresh_slider.setValue(40); self.cv_thresh_slider.valueChanged.connect(self._update_cv_threshold)
        self.cv_thresh_label = QLabel(str(self.cv_thresh_slider.value()))
        cvthr.addWidget(self.cv_thresh_slider); cvthr.addWidget(self.cv_thresh_label)
        cont = QHBoxLayout(); cont.addWidget(QLabel('Contour Area:'))
        self.contour_slider = QSlider(Qt.Orientation.Horizontal); self.contour_slider.setRange(1,200); self.contour_slider.setValue(10); self.contour_slider.valueChanged.connect(self._update_contour_threshold)
        self.contour_label = QLabel(str(self.contour_slider.value()))
        cont.addWidget(self.contour_slider); cont.addWidget(self.contour_label)

        thresholds_group = QGroupBox('Model & Detection Thresholds')
        thresholds_group.setStyleSheet(
            'QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 4px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }'
        )
        thresholds_layout = QVBoxLayout()
        thresholds_layout.setSpacing(8)
        thresholds_layout.setContentsMargins(10, 10, 10, 10)
        thresholds_layout.addLayout(top_bar_layout)
        thresholds_layout.addLayout(thr)
        thresholds_layout.addLayout(cvthr)
        thresholds_layout.addLayout(cont)
        thresholds_group.setLayout(thresholds_layout)
        left.addWidget(thresholds_group)
        # camera & fps
        cr = QHBoxLayout(); cr.addWidget(QLabel('Source:'))
        self.cam_combo = QComboBox(); cr.addWidget(self.cam_combo)
        self.list_cam_btn = QPushButton('List Cameras'); self.list_cam_btn.clicked.connect(self._list_cameras); cr.addWidget(self.list_cam_btn)
        cr.addWidget(QLabel('Backend:'))
        self.backend_combo = QComboBox(); self.backend_options = ['Auto','MSMF','DSHOW']; self.backend_combo.addItems(self.backend_options)
        self.backend_combo.currentIndexChanged.connect(lambda _: self._list_cameras())
        cr.addWidget(self.backend_combo)
        cr.addWidget(QLabel('FourCC:'))
        self.fourcc_combo = QComboBox()
        self.fourcc_options = ['Auto', 'YUY2', 'MJPG', 'NV12', 'I420', 'H264']
        self.fourcc_combo.addItems(self.fourcc_options)
        cr.addWidget(self.fourcc_combo)
        cr.addWidget(QLabel('Frame Rate:'))
        self.fps_combo = QComboBox(); self.fps_options = ['Uncapped','120','60','50','30','15','5','2']; self.fps_combo.addItems(self.fps_options); cr.addWidget(self.fps_combo)
        cr.addWidget(QLabel('Resolution:'))
        self.res_combo = QComboBox(); self.resolution_options = ['Source/Native','2592x1944','2592x1440','2560x1440','2048x1536','2304x1296','1920x1080','1600x1200','1600x900','1280X960','1280x720','1024x768','960X720','1024x576','960x540','800x600','848x480','800x450','640x480','640x360']; self.res_combo.addItems(self.resolution_options); cr.addWidget(self.res_combo)
        left.addLayout(cr)
        left.addStretch()

        # right panel
        # general controls group (moved to left column)
        general_group = QGroupBox('General Controls')
        general_group.setStyleSheet(
            'QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 4px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }'
        )
        general_layout = QVBoxLayout()
        general_layout.setSpacing(6)
        general_layout.setContentsMargins(12, 12, 12, 12)

        auto_row = QHBoxLayout()
        self.auto_save_check = QCheckBox('Auto-save Detections')
        self.auto_save_status_label = QLabel('Disabled')
        self.auto_save_status_label.setStyleSheet('color: #d35400; font-weight: bold;')
        self.auto_save_check.toggled.connect(self._update_auto_save_status)
        auto_row.addWidget(self.auto_save_check)
        auto_row.addWidget(self.auto_save_status_label)
        auto_row.addStretch()
        general_layout.addLayout(auto_row)

        beep_row = QHBoxLayout()
        self.beep_check = QCheckBox('Enable Beep Sound')
        self.beep_check.toggled.connect(self._update_beep_status)
        beep_row.addWidget(self.beep_check)
        beep_row.addStretch()
        general_layout.addLayout(beep_row)

        guide_row = QHBoxLayout()
        self.machine_guide_check = QCheckBox('Show Machine Guide')
        self.machine_guide_check.setChecked(self.machine_guide_enabled)
        self.machine_guide_check.toggled.connect(self._toggle_machine_guide)
        guide_row.addWidget(self.machine_guide_check)
        self.center_marker_check = QCheckBox('Show Center Marker')
        self.center_marker_check.setChecked(self.center_marker_enabled)
        self.center_marker_check.toggled.connect(self._toggle_center_marker)
        guide_row.addWidget(self.center_marker_check)
        guide_row.addStretch()
        general_layout.addLayout(guide_row)

        training_row = QHBoxLayout()
        self.random_save_check = QCheckBox('Random Save for Training')
        self.random_save_check.toggled.connect(self._update_random_save_status)
        training_row.addWidget(self.random_save_check)
        training_row.addStretch()
        general_layout.addLayout(training_row)

        tripwire_row = QHBoxLayout()
        self.tripwire_check = QCheckBox('Tripwire/Once per object')
        self.tripwire_check.toggled.connect(self._update_tripwire_status)
        tripwire_row.addWidget(self.tripwire_check)
        tripwire_row.addStretch()
        general_layout.addLayout(tripwire_row)
 
        labels_row = QHBoxLayout()
        self.video_labels_check = QCheckBox('Show Video Labels')
        self.video_labels_check.setChecked(True)
        self.video_labels_check.toggled.connect(self._toggle_video_labels)
        labels_row.addWidget(self.video_labels_check)
        labels_row.addStretch()
        general_layout.addLayout(labels_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel('Mode:'))
        self.mode_combo = QComboBox(); self.mode_combo.addItems(['Reconstruction/Model','Color (HSV)','Hybrid (OR)'])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        general_layout.addLayout(mode_row)

        general_group.setLayout(general_layout)
        left.addWidget(general_group)

        def build_hue_group(title, enable_attr, enable_slot, slider_specs, summary_attr):
            group = QGroupBox(title)
            group.setStyleSheet(
                'QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 6px; margin-top: 8px; }'
                'QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }'
            )
            group_layout = QVBoxLayout()
            group_layout.setSpacing(8)
            group_layout.setContentsMargins(10, 10, 10, 10)

            toggle_row = QHBoxLayout()
            toggle = QCheckBox('Enable detection')
            toggle.setChecked(True)
            toggle.toggled.connect(enable_slot)
            setattr(self, enable_attr, toggle)
            toggle_row.addWidget(toggle)
            toggle_row.addStretch()
            group_layout.addLayout(toggle_row)

            for label_text, slider_attr, value_attr, minimum, maximum, default, handler in slider_specs:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(8)
                row.addWidget(QLabel(label_text))
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(minimum, maximum)
                slider.setValue(default)
                setattr(self, slider_attr, slider)
                slider.valueChanged.connect(handler)
                value_label = QLabel(str(default))
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                setattr(self, value_attr, value_label)
                row.addWidget(slider, 1)
                row.addWidget(value_label)
                group_layout.addLayout(row)

            summary_label = QLabel()
            summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
            summary_label.setStyleSheet(
                'padding: 6px 8px; border: 1px solid #555; border-radius: 5px; '
                'background-color: #222; font-weight: bold; font-size: 10pt; min-height: 28px; color: #f0f0f0;'
            )
            setattr(self, summary_attr, summary_label)
            group_layout.addWidget(summary_label)
            group_layout.addStretch()

            group.setLayout(group_layout)
            return group

        hue1_group = build_hue_group(
            'Hue1 (Yellow)',
            'hue1_enable_check',
            self._toggle_primary_hue_enabled,
            [
                ('Hue Low:', 'h_low_slider', 'h_low_label', 0, 179, 15, self._update_hsv),
                ('Hue High:', 'h_high_slider', 'h_high_label', 0, 179, 35, self._update_hsv),
                ('Saturation Min:', 's_min_slider', 's_min_label', 0, 255, 60, self._update_hsv),
                ('Value Min:', 'v_min_slider', 'v_min_label', 0, 255, 120, self._update_hsv),
            ],
            'hsv_summary_label'
        )
        right.addWidget(hue1_group)

        hue2_group = build_hue_group(
            'Hue2 (Green)',
            'hue2_enable_check',
            self._toggle_secondary_hue_enabled,
            [
                ('Hue2 Low:', 'h2_low_slider', 'h2_low_label', 0, 179, 75, self._update_hsv_secondary),
                ('Hue2 High:', 'h2_high_slider', 'h2_high_label', 0, 179, 95, self._update_hsv_secondary),
                ('Saturation2 Min:', 's2_min_slider', 's2_min_label', 0, 255, 60, self._update_hsv_secondary),
                ('Value2 Min:', 'v2_min_slider', 'v2_min_label', 0, 255, 120, self._update_hsv_secondary),
            ],
            'hsv2_summary_label'
        )
        right.addWidget(hue2_group)

        hue3_group = build_hue_group(
            'Hue3 (Blue)',
            'hue3_enable_check',
            self._toggle_tertiary_hue_enabled,
            [
                ('Hue3 Low:', 'h3_low_slider', 'h3_low_label', 0, 179, 105, self._update_hsv_tertiary),
                ('Hue3 High:', 'h3_high_slider', 'h3_high_label', 0, 179, 125, self._update_hsv_tertiary),
                ('Saturation3 Min:', 's3_min_slider', 's3_min_label', 0, 255, 60, self._update_hsv_tertiary),
                ('Value3 Min:', 'v3_min_slider', 'v3_min_label', 0, 255, 120, self._update_hsv_tertiary),
            ],
            'hsv3_summary_label'
        )
        right.addWidget(hue3_group)

        right.addSpacing(6)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel('Sample Target:'))
        self.hsv_target_combo = QComboBox(); self.hsv_target_combo.addItems(['Hue1 (Yellow)', 'Hue2 (Green)', 'Hue3 (Blue)'])
        self.hsv_target_combo.currentIndexChanged.connect(self._change_sample_target)
        target_row.addWidget(self.hsv_target_combo)
        self.hsv_lock_check = QCheckBox('Hue Target Lock')
        self.hsv_lock_check.toggled.connect(self._toggle_hsv_target_lock)
        target_row.addWidget(self.hsv_lock_check)
        target_row.addStretch()
        right.addLayout(target_row)

        self.hsv_sample1_label = QLabel('Sample 1: -')
        self.hsv_sample2_label = QLabel('Sample 2: -')
        self.hsv_sample_avg_label = QLabel('Combined: -')

        # Create a separate row for each sample label
        for lbl in (self.hsv_sample1_label, self.hsv_sample2_label, self.hsv_sample_avg_label):
            lbl.setStyleSheet('padding: 4px 8px; border: 1px solid #555; border-radius: 4px; background-color: #222; color: #f0f0f0;')
            row = QHBoxLayout()
            row.addWidget(lbl)
            row.addStretch()
            right.addLayout(row)

        # buttons row
        self.start_btn = QPushButton('Start Detection'); self.start_btn.clicked.connect(self._start_detection); self.start_btn.setDisabled(True)
        self.pause_btn = QPushButton('Pause Detection'); self.pause_btn.clicked.connect(self._pause_detection); self.pause_btn.setDisabled(True)
        self.resume_btn = QPushButton('Resume Detection'); self.resume_btn.clicked.connect(self._resume_detection); self.resume_btn.setDisabled(True)
        self.stop_btn = QPushButton('Stop Detection'); self.stop_btn.clicked.connect(self._stop_detection); self.stop_btn.setDisabled(True)
        self.test_image_btn = QPushButton('Test Image'); self.test_image_btn.clicked.connect(self._test_image); self.test_image_btn.setDisabled(True)
        self.capture_btn = QPushButton('Capture Image'); self.capture_btn.clicked.connect(self._capture_image); self.capture_btn.setDisabled(True)
        self.clear_all_btn = QPushButton('Clear ALL output')
        self.clear_all_btn.clicked.connect(self._clear_output_folder)

        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(self.resume_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addWidget(self.test_image_btn)
        buttons.addWidget(self.capture_btn)
        buttons.addWidget(self.clear_all_btn)
        buttons.addStretch()

        controls.addLayout(left,3); controls.addLayout(right,2)
        self.main_layout.addLayout(controls)
        self.main_layout.setAlignment(controls, Qt.AlignmentFlag.AlignTop)
        self.main_layout.addLayout(buttons)
        self._update_hsv_summary()
        self._update_hsv_sample_labels()

        self._list_cameras(); self._enable_hsv_controls(False)

    def _create_status_bar(self):
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready. Load a model or choose Color/Hybrid mode.')
        self.fourcc_status_label = QLabel('Active FourCC: -')
        self.status_bar.addPermanentWidget(self.fourcc_status_label, 0)

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
        prefer_backend = getattr(self, 'backend_combo', None)
        backend_label = prefer_backend.currentText() if prefer_backend else 'Auto'
        available_cameras = QMediaDevices.videoInputs()
        if not available_cameras:
            self.cam_combo.addItem('No Camera Found')
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f'No cameras found. Selected backend: {backend_label}.', 3000)
            return
        for idx, camera_device in enumerate(available_cameras):
            description = camera_device.description() or f'Camera {idx}'
            self.cam_combo.addItem(description)
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f'Detected {len(available_cameras)} camera(s). Backend: {backend_label}.', 3000)

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

    def _update_random_save_status(self, checked):
        if hasattr(self, 'detection_worker'):
            self.detection_worker.set_random_save_enabled(checked)
        self.status_bar.showMessage(
            'Random training capture enabled.' if checked else 'Random training capture disabled.',
            2000
        )

    def _update_tripwire_status(self, checked):
        if hasattr(self, 'detection_worker'):
            self.detection_worker.set_tripwire_enabled(checked)
        self.status_bar.showMessage(
            'Tripwire enabled.' if checked else 'Tripwire disabled.',
            2000
        )

    def _toggle_machine_guide(self, checked):
        self.machine_guide_enabled = checked
        if hasattr(self, 'status_bar'):
            msg = 'Machine guide overlay enabled.' if checked else 'Machine guide overlay hidden.'
            self.status_bar.showMessage(msg, 2000)
        if not self.is_detection_running and self.last_tested_image is not None:
            self._reprocess_image()

    def _toggle_center_marker(self, checked):
        self.center_marker_enabled = checked
        if hasattr(self, 'status_bar'):
            msg = 'Center marker enabled.' if checked else 'Center marker hidden.'
            self.status_bar.showMessage(msg, 2000)
        if not self.is_detection_running and self.last_tested_image is not None:
            self._reprocess_image()

    def _change_sample_target(self, idx):
        targets = {0: 'primary', 1: 'secondary', 2: 'tertiary'}
        self._current_sample_target = targets.get(idx, 'primary')
        self._update_hsv_sample_labels()

    def _toggle_hsv_target_lock(self, locked):
        self.hsv_target_locked = locked
        if hasattr(self, 'status_bar'):
            msg = 'Hue target locked; video sampling disabled.' if locked else 'Hue target unlocked; video sampling enabled.'
            self.status_bar.showMessage(msg, 2000)

    def _enable_hsv_controls(self, enabled: bool):
        for w in [self.h_low_slider, self.h_high_slider, self.s_min_slider, self.v_min_slider,
                  self.h_low_label, self.h_high_label, self.s_min_label, self.v_min_label,
                  self.h2_low_slider, self.h2_high_slider, self.s2_min_slider, self.v2_min_slider,
                  self.h2_low_label, self.h2_high_label, self.s2_min_label, self.v2_min_label,
                  self.h3_low_slider, self.h3_high_slider, self.s3_min_slider, self.v3_min_slider,
                  self.h3_low_label, self.h3_high_label, self.s3_min_label, self.v3_min_label]:
            w.setEnabled(enabled)
        if hasattr(self, 'hue1_enable_check'):
            self.hue1_enable_check.setEnabled(enabled)
        if hasattr(self, 'hue2_enable_check'):
            self.hue2_enable_check.setEnabled(enabled)
        if hasattr(self, 'hue3_enable_check'):
            self.hue3_enable_check.setEnabled(enabled)
        if hasattr(self, 'hsv_summary_label'):
            self.hsv_summary_label.setEnabled(True)
        if hasattr(self, 'hsv2_summary_label'):
            self.hsv2_summary_label.setEnabled(True)
        if hasattr(self, 'hsv3_summary_label'):
            self.hsv3_summary_label.setEnabled(True)

    def _mode_changed(self, idx):
        mode = ['reconstruction/model','color','hybrid'][idx]
        self.detector.set_mode({'reconstruction/model':'recon','color':'color','hybrid':'hybrid'}[mode])
        self._enable_hsv_controls(idx in (1,2))
        self._update_hsv_summary()
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
        self._update_hsv_summary()
        self._reprocess_image()

    def _update_hsv_secondary(self, _):
        h_low = self.h2_low_slider.value(); h_high = self.h2_high_slider.value()
        if h_low > h_high:
            sender = self.sender()
            if sender is self.h2_low_slider:
                self.h2_high_slider.setValue(h_low); h_high = h_low
            else:
                self.h2_low_slider.setValue(h_high); h_low = h_high
        s_min = self.s2_min_slider.value(); v_min = self.v2_min_slider.value()
        self.h2_low_label.setText(str(h_low)); self.h2_high_label.setText(str(h_high))
        self.s2_min_label.setText(str(s_min)); self.v2_min_label.setText(str(v_min))
        self.detector.set_hsv_secondary(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min)
        self._update_hsv2_summary()
        self._update_hsv3_summary()
        self._reprocess_image()

    def _update_hsv_tertiary(self, _):
        h_low = self.h3_low_slider.value(); h_high = self.h3_high_slider.value()
        if h_low > h_high:
            sender = self.sender()
            if sender is self.h3_low_slider:
                self.h3_high_slider.setValue(h_low); h_high = h_low
            else:
                self.h3_low_slider.setValue(h_high); h_low = h_high
        s_min = self.s3_min_slider.value(); v_min = self.v3_min_slider.value()
        self.h3_low_label.setText(str(h_low)); self.h3_high_label.setText(str(h_high))
        self.s3_min_label.setText(str(s_min)); self.v3_min_label.setText(str(v_min))
        self.detector.set_hsv_tertiary(h_low=h_low, h_high=h_high, s_min=s_min, v_min=v_min)
        self._update_hsv3_summary()
        self._reprocess_image()

    def _toggle_primary_hue_enabled(self, checked: bool):
        self.detector.set_primary_hue_enabled(checked)
        if hasattr(self, 'status_bar'):
            msg = 'Hue1 (Yellow) detection enabled.' if checked else 'Hue1 (Yellow) detection disabled.'
            self.status_bar.showMessage(msg, 2000)
        self._update_hsv_summary()
        self._reprocess_image()

    def _toggle_secondary_hue_enabled(self, checked: bool):
        self.detector.set_secondary_hue_enabled(checked)
        if hasattr(self, 'status_bar'):
            msg = 'Hue2 (Green) detection enabled.' if checked else 'Hue2 (Green) detection disabled.'
            self.status_bar.showMessage(msg, 2000)
        self._update_hsv2_summary()
        self._reprocess_image()

    def _toggle_tertiary_hue_enabled(self, checked: bool):
        self.detector.set_tertiary_hue_enabled(checked)
        if hasattr(self, 'status_bar'):
            msg = 'Hue3 (Blue) detection enabled.' if checked else 'Hue3 (Blue) detection disabled.'
            self.status_bar.showMessage(msg, 2000)
        self._update_hsv3_summary()
        self._reprocess_image()

    def _toggle_video_labels(self, checked: bool):
        self.show_video_labels = bool(checked)
        if hasattr(self, 'status_bar'):
            msg = 'Video labels shown.' if checked else 'Video labels hidden.'
            self.status_bar.showMessage(msg, 2000)
        if not self.is_detection_running and self.last_tested_image is not None:
            self._reprocess_image()

    def _update_hsv_summary(self):
        if not hasattr(self, 'hsv_summary_label'):
            return
        if not self.detector.primary_hue_enabled:
            self.hsv_summary_label.setText('Hue1 Disabled')
            self.hsv_summary_label.setStyleSheet(
                'padding: 6px 8px; border: 1px solid #555; border-radius: 5px; '
                'background-color: #333; color: #888; font-weight: bold; font-size: 10pt; min-height: 28px;'
            )
        else:
            h_low = self.h_low_slider.value(); h_high = self.h_high_slider.value()
            s_min = self.s_min_slider.value(); v_min = self.v_min_slider.value()
            self.hsv_summary_label.setText(f'H {h_low}-{h_high} | S >= {s_min} | V >= {v_min}')
            hue_mid = max(0, min(179, (h_low + h_high) // 2))
            sat_preview = max(s_min, min(255, s_min + (255 - s_min) // 2))
            val_preview = max(v_min, min(255, v_min + (255 - v_min) // 2))
            preview_color = QColor.fromHsv(hue_mid * 2, sat_preview, val_preview)
            text_color = '#000' if preview_color.value() > 160 else '#fff'
            self.hsv_summary_label.setStyleSheet(
                f'padding: 6px 8px; border: 1px solid #555; border-radius: 5px;'
                f'background-color: {preview_color.name()}; color: {text_color}; font-weight: bold; font-size: 10pt; min-height: 28px;'
            )
        self._update_hsv2_summary()
        self._update_hsv3_summary()

    def _update_hsv2_summary(self):
        if not hasattr(self, 'hsv2_summary_label'):
            return
        if not self.detector.secondary_hue_enabled:
            self.hsv2_summary_label.setText('Hue2 Disabled')
            self.hsv2_summary_label.setStyleSheet(
                'padding: 6px 8px; border: 1px solid #555; border-radius: 5px; '
                'background-color: #333; color: #888; font-weight: bold; font-size: 10pt; min-height: 28px;'
            )
            return
        h_low = self.h2_low_slider.value(); h_high = self.h2_high_slider.value()
        s_min = self.s2_min_slider.value(); v_min = self.v2_min_slider.value()
        self.hsv2_summary_label.setText(f'H {h_low}-{h_high} | S >= {s_min} | V >= {v_min}')
        hue_mid = max(0, min(179, (h_low + h_high) // 2))
        sat_preview = max(s_min, min(255, s_min + (255 - s_min) // 2))
        val_preview = max(v_min, min(255, v_min + (255 - v_min) // 2))
        preview_color = QColor.fromHsv(hue_mid * 2, sat_preview, val_preview)
        text_color = '#000' if preview_color.value() > 160 else '#fff'
        self.hsv2_summary_label.setStyleSheet(
            f'padding: 6px 8px; border: 1px solid #555; border-radius: 5px;'
            f'background-color: {preview_color.name()}; color: {text_color}; font-weight: bold; font-size: 10pt; min-height: 28px;'
        )

    def _update_hsv3_summary(self):
        if not hasattr(self, 'hsv3_summary_label'):
            return
        if not self.detector.tertiary_hue_enabled:
            self.hsv3_summary_label.setText('Hue3 Disabled')
            self.hsv3_summary_label.setStyleSheet(
                'padding: 6px 8px; border: 1px solid #555; border-radius: 5px; '
                'background-color: #333; color: #888; font-weight: bold; font-size: 10pt; min-height: 28px;'
            )
            return
        h_low = self.h3_low_slider.value(); h_high = self.h3_high_slider.value()
        s_min = self.s3_min_slider.value(); v_min = self.v3_min_slider.value()
        self.hsv3_summary_label.setText(f'H {h_low}-{h_high} | S >= {s_min} | V >= {v_min}')
        hue_mid = max(0, min(179, (h_low + h_high) // 2))
        sat_preview = max(s_min, min(255, s_min + (255 - s_min) // 2))
        val_preview = max(v_min, min(255, v_min + (255 - v_min) // 2))
        preview_color = QColor.fromHsv(hue_mid * 2, sat_preview, val_preview)
        text_color = '#000' if preview_color.value() > 160 else '#fff'
        self.hsv3_summary_label.setStyleSheet(
            f'padding: 6px 8px; border: 1px solid #555; border-radius: 5px;'
            f'background-color: {preview_color.name()}; color: {text_color}; font-weight: bold; font-size: 10pt; min-height: 28px;'
        )

    def _set_sample_label(self, label, title, sample):
        base_style = 'padding: 4px 8px; border: 1px solid #555; border-radius: 4px; background-color: #222; color: #f0f0f0;'
        if sample is None:
            label.setText(f'{title}: -')
            label.setStyleSheet(base_style)
            return
        h, s, v = (int(sample[0]), int(sample[1]), int(sample[2]))
        hue_qt = max(0, min(359, h * 2))
        qcolor = QColor.fromHsv(hue_qt, max(0, min(255, s)), max(0, min(255, v)))
        text_color = '#000' if qcolor.value() > 160 else '#fff'
        label.setText(f'{title}: H{h} S{s} V{v}')
        label.setStyleSheet(
            f'padding: 4px 8px; border: 1px solid #555; border-radius: 4px; '
            f'background-color: {qcolor.name()}; color: {text_color};'
        )

    def _update_hsv_sample_labels(self):
        if not hasattr(self, 'hsv_sample1_label'):
            return
        target = self._current_sample_target
        state = self._sample_state[target]
        target_name_map = {
            'primary': 'Hue1',
            'secondary': 'Hue2',
            'tertiary': 'Hue3'
        }
        target_name = target_name_map.get(target, 'Hue1')
        self._set_sample_label(self.hsv_sample1_label, f'Sample 1 ({target_name})', state['first'])
        self._set_sample_label(self.hsv_sample2_label, f'Sample 2 ({target_name})', state['second'])
        base_style = 'padding: 4px 8px; border: 1px solid #555; border-radius: 4px; background-color: #222; color: #f0f0f0;'
        if state['combined'] is None:
            self.hsv_sample_avg_label.setText(f'Combined ({target_name}): -')
            self.hsv_sample_avg_label.setStyleSheet(base_style)
        else:
            h_low, h_high, s_thr, v_thr = state['combined']
            hue_mid = max(0, min(179, (h_low + h_high) // 2))
            sat_preview = max(s_thr, min(255, s_thr + (255 - s_thr) // 2))
            val_preview = max(v_thr, min(255, v_thr + (255 - v_thr) // 2))
            preview_color = QColor.fromHsv(hue_mid * 2, sat_preview, val_preview)
            text_color = '#000' if preview_color.value() > 160 else '#fff'
            self.hsv_sample_avg_label.setText(f'Combined ({target_name}): H{h_low}-{h_high} | S>={s_thr} | V>={v_thr}')
            self.hsv_sample_avg_label.setStyleSheet(
                f'padding: 4px 8px; border: 1px solid #555; border-radius: 4px; '
                f'background-color: {preview_color.name()}; color: {text_color};'
            )

    def _apply_sampled_hsv(self, h_cv, s, v, target='primary'):
        h_cv = int(max(0, min(179, h_cv)))
        s = int(max(0, min(255, s)))
        v = int(max(0, min(255, v)))
        if target == 'primary':
            low_slider, high_slider = self.h_low_slider, self.h_high_slider
            s_slider, v_slider = self.s_min_slider, self.v_min_slider
            updater = self._update_hsv
        elif target == 'secondary':
            low_slider, high_slider = self.h2_low_slider, self.h2_high_slider
            s_slider, v_slider = self.s2_min_slider, self.v2_min_slider
            updater = self._update_hsv_secondary
        else:
            low_slider, high_slider = self.h3_low_slider, self.h3_high_slider
            s_slider, v_slider = self.s3_min_slider, self.v3_min_slider
            updater = self._update_hsv_tertiary

        current_span = max(2, high_slider.value() - low_slider.value())
        half_span = current_span // 2
        new_low = max(0, min(179, h_cv - half_span))
        new_high = max(0, min(179, h_cv + half_span))
        updates = (
            (low_slider, new_low),
            (high_slider, new_high),
            (s_slider, s),
            (v_slider, v),
        )
        for slider, value in updates:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        updater(None)

    def _apply_dual_samples(self, sample_a, sample_b, target='primary'):
        h_vals = [int(sample_a[0]), int(sample_b[0])]
        s_vals = [int(sample_a[1]), int(sample_b[1])]
        v_vals = [int(sample_a[2]), int(sample_b[2])]

        h1, h2 = h_vals
        diff = abs(h1 - h2)
        if diff > 90 and ((min(h_vals) < 20 and max(h_vals) > 159) or diff >= 120):
            h_low = 0
            h_high = 179
        else:
            h_low = min(h_vals)
            h_high = max(h_vals)
        margin = 5
        h_low = max(0, h_low - margin)
        h_high = min(179, h_high + margin)
        if h_high < h_low:
            h_low, h_high = h_high, h_low
        if h_high - h_low < 2:
            h_high = min(179, h_low + 2)

        s_thr = max(0, min(s_vals) - 20)
        v_thr = max(0, min(v_vals) - 20)
        s_thr = min(255, s_thr)
        v_thr = min(255, v_thr)

        if target == 'primary':
            updates = (
                (self.h_low_slider, h_low),
                (self.h_high_slider, h_high),
                (self.s_min_slider, s_thr),
                (self.v_min_slider, v_thr),
            )
            updater = self._update_hsv
        elif target == 'secondary':
            updates = (
                (self.h2_low_slider, h_low),
                (self.h2_high_slider, h_high),
                (self.s2_min_slider, s_thr),
                (self.v2_min_slider, v_thr),
            )
            updater = self._update_hsv_secondary
        else:
            updates = (
                (self.h3_low_slider, h_low),
                (self.h3_high_slider, h_high),
                (self.s3_min_slider, s_thr),
                (self.v3_min_slider, v_thr),
            )
            updater = self._update_hsv_tertiary
        for slider, value in updates:
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
        updater(None)
        return (int(h_low), int(h_high), int(s_thr), int(v_thr))

    def _handle_video_click(self, x, y):
        if self.hsv_target_locked:
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage('Hue target lock is enabled; click ignored.', 2000)
            return
        if self.mode_combo.currentIndex() not in (1,2):
            return
        frame = None
        if self.is_detection_running and self.current_frame is not None:
            frame = self.current_frame
        elif not self.is_detection_running and self.last_tested_image is not None:
            frame = self.last_tested_image
        if frame is None:
            return
        pixmap = self.video_window.video_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        pix_w, pix_h = pixmap.width(), pixmap.height()
        label_w, label_h = self.video_window.video_label.width(), self.video_window.video_label.height()
        offset_x = max(0.0, (label_w - pix_w) / 2.0)
        offset_y = max(0.0, (label_h - pix_h) / 2.0)
        img_x = x - offset_x
        img_y = y - offset_y
        if img_x < 0 or img_y < 0 or img_x >= pix_w or img_y >= pix_h:
            return
        frame_h, frame_w = frame.shape[:2]
        sample_x = int(np.clip((img_x / pix_w) * frame_w, 0, frame_w - 1))
        sample_y = int(np.clip((img_y / pix_h) * frame_h, 0, frame_h - 1))
        patch = frame[sample_y:sample_y+1, sample_x:sample_x+1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
        sample = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
        now = time.time()
        target_index = self.hsv_target_combo.currentIndex()
        target_map = {0: 'primary', 1: 'secondary', 2: 'tertiary'}
        target = target_map.get(target_index, 'primary')
        state = self._sample_state[target]
        target_name_lookup = {
            'primary': 'Hue1 (Yellow)',
            'secondary': 'Hue2 (Green)',
            'tertiary': 'Hue3 (Blue)',
        }
        target_name = target_name_lookup.get(target, 'Hue1 (Yellow)')
        self._current_sample_target = target
        if state['pending'] is not None and (now - state['timestamp']) > 6.0:
            state['pending'] = None
            state['timestamp'] = 0.0
            state['first'] = None
            state['second'] = None
            state['combined'] = None
        if state['pending'] is None:
            state['pending'] = sample
            state['timestamp'] = now
            state['first'] = sample
            state['second'] = None
            state['combined'] = None
            self._apply_sampled_hsv(*sample, target=target)
            self._update_hsv_sample_labels()
            self.status_bar.showMessage(
                f'[{target_name}] HSV sample at ({sample_x},{sample_y}) -> H:{sample[0]} S:{sample[1]} V:{sample[2]} (click another point within 6s to average)',
                5000,
            )
        else:
            first_sample = state['pending']
            state['second'] = sample
            combined = self._apply_dual_samples(first_sample, sample, target=target)
            state['pending'] = None
            state['timestamp'] = 0.0
            state['combined'] = combined
            self._update_hsv_sample_labels()
            self.status_bar.showMessage(
                f'[{target_name}] Combined HSV from two samples -> H:{combined[0]}-{combined[1]} | S>={combined[2]} | V>={combined[3]}',
                5000,
            )

    def _start_detection(self):
        start_time_total = time.time()
        print(f"\n[LOG {start_time_total:.2f}] 'Start Detection' button clicked.")

        if self.cam_combo.currentText() == 'No Camera Found':
            self.status_bar.showMessage('Error: No camera selected or found.'); return
        self.last_tested_image = None; self.status_bar.showMessage('Starting detection...')
        self.frame_count = 0; self.start_time = time.time(); self.detection_worker.reset_counter()
        res_text = self.res_combo.currentText()
        resolution = tuple(map(int, res_text.lower().split('x'))) if res_text != 'Source/Native' else None
        fps_text = self.fps_combo.currentText(); fps_limit = int(fps_text) if fps_text != 'Uncapped' else None
        backend_choice = self.backend_combo.currentText() if hasattr(self, 'backend_combo') else 'Auto'
        fourcc_choice = self.fourcc_combo.currentText() if hasattr(self, 'fourcc_combo') else 'Auto'
        preferred_fourcc = None if not fourcc_choice or fourcc_choice == 'Auto' else fourcc_choice

        print(f"[LOG {time.time():.2f}] Creating VideoThread with backend {backend_choice}...")
        print(f"[LOG {time.time():.2f}] Preferred FOURCC selection: {fourcc_choice}")
        if hasattr(self, 'fourcc_status_label'):
            self.fourcc_status_label.setText('Active FourCC: Connecting...')
        self.video_thread = VideoThread(
            camera_index=self.cam_combo.currentIndex(),
            resolution=resolution,
            fps_limit=fps_limit,
            prefer_backend=backend_choice,
            preferred_fourcc=preferred_fourcc
        )
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.fourcc_signal.connect(self._update_fourcc_label)
        
        print(f"[LOG {time.time():.2f}] Starting VideoThread...")
        self.video_thread.start()
        
        self.is_detection_running = True
        self.is_paused = False
        self.start_time = time.time()
        self.frame_count = 0
        self._toggle_controls(False)
        print(f"[LOG {time.time():.2f}] _start_detection function finished. Total time: {time.time() - start_time_total:.4f} seconds.")

    def _pause_detection(self):
        if not self.is_detection_running or self.is_paused:
            return
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.pause()
        self.is_paused = True
        self.frame_count = 0
        self.start_time = time.time()
        self.status_bar.showMessage('Detection paused.')
        self._toggle_controls(False)

    def _resume_detection(self):
        if not self.is_detection_running or not self.is_paused:
            return
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.resume()
        self.is_paused = False
        self.frame_count = 0
        self.start_time = time.time()
        self.status_bar.showMessage('Detection resumed.')
        self._toggle_controls(False)

    def _stop_detection(self):
        if hasattr(self,'video_thread') and self.video_thread.isRunning(): self.video_thread.stop()
        self.is_detection_running = False
        self.is_paused = False
        if self.last_tested_image is None and (self.video_window.video_label.pixmap() is None or self.video_window.video_label.pixmap().isNull()):
            self.video_window.video_label.setText("Press 'Start Detection' or 'Test Image' to begin")
        if hasattr(self, 'fourcc_status_label'):
            self.fourcc_status_label.setText('Active FourCC: -')
        self.status_bar.showMessage('Detection stopped.'); self._toggle_controls(True)

    # ... (ส่วนที่เหลือของ MainWindow เหมือนเดิม) ...
    def _toggle_controls(self, enable):
        running = not enable
        paused = self.is_paused if running else False

        self.start_btn.setDisabled(running)
        self.pause_btn.setDisabled((not running) or paused)
        self.resume_btn.setDisabled((not running) or (not paused))
        self.stop_btn.setDisabled(not running)

        allow_config = enable
        self.browse_btn.setDisabled(not allow_config); self.cam_combo.setDisabled(not allow_config)
        self.res_combo.setDisabled(not allow_config); self.list_cam_btn.setDisabled(not allow_config); self.test_image_btn.setDisabled(not allow_config)
        if hasattr(self, 'backend_combo'):
            self.backend_combo.setDisabled(not allow_config)
        if hasattr(self, 'fourcc_combo'):
            self.fourcc_combo.setDisabled(not allow_config)
        self.capture_btn.setDisabled((not running) or paused)
        self.fps_combo.setDisabled(not allow_config)
        hue_controls_allowed = allow_config and (self.mode_combo.currentIndex() in (1,2) if hasattr(self, 'mode_combo') else False)
        if hasattr(self, 'hue1_enable_check'):
            self.hue1_enable_check.setDisabled(not hue_controls_allowed)
        if hasattr(self, 'hue2_enable_check'):
            self.hue2_enable_check.setDisabled(not hue_controls_allowed)
        if hasattr(self, 'hue3_enable_check'):
            self.hue3_enable_check.setDisabled(not hue_controls_allowed)

    @pyqtSlot(str)
    def _update_fourcc_label(self, mode_text):
        if hasattr(self, 'fourcc_status_label'):
            self.fourcc_status_label.setText(f'Active FourCC: {mode_text}')

    def _capture_image(self):
        if not self.is_detection_running or self.is_paused or self.current_frame is None:
            self.status_bar.showMessage('Capture only available during live detection.', 3000); return
        original_frame = self.current_frame.copy()
        processed_frame, _, _, _ = self.detector.process_frame(original_frame.copy())
        anomaly_count = self.detection_worker.anomaly_count
        h,w,_ = processed_frame.shape; margin = 10
        fps_text = f'FPS: {self.fps:.2f}'; cv2.putText(processed_frame, fps_text, (margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        res_text = f'{w}x{h}'; (res_w,_),_ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2); cv2.putText(processed_frame, res_text, (w-res_w-margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        det_text = f'Detections: {anomaly_count}'; (det_w,_),_ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3); cv2.putText(processed_frame, det_text, (w-det_w-margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]; fname = f'capture_{ts}.png'
        det_dir = os.path.join('output','captures_detected'); ori_dir = os.path.join('output','captures_original')
        os.makedirs(det_dir, exist_ok=True); os.makedirs(ori_dir, exist_ok=True)
        cv2.imwrite(os.path.join(det_dir,fname), processed_frame); cv2.imwrite(os.path.join(ori_dir,fname), original_frame)
        self.status_bar.showMessage(f'Image captured and saved as {fname}', 4000)

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

    def _compute_focus_measure(self, frame, roi_fraction: float = 0.25):
        """Return focus value and ROI rectangle (x1,y1,x2,y2) using variance of Laplacian."""
        if frame is None or frame.size == 0:
            return 0.0, None
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return 0.0, None
        roi_fraction = max(0.05, min(0.5, roi_fraction))
        roi_w = max(16, int(w * roi_fraction))
        roi_h = max(16, int(h * roi_fraction))
        x1 = max(0, (w - roi_w) // 2)
        y1 = max(0, (h - roi_h) // 2)
        x2 = min(w, x1 + roi_w)
        y2 = min(h, y1 + roi_h)
        if x2 <= x1 or y2 <= y1:
            return 0.0, None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        focus_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(focus_value), (x1, y1, x2, y2)

    def _draw_focus_overlay(self, frame, focus_value: float, roi_rect):
        if frame is None or frame.size == 0:
            return
        if not getattr(self, 'show_video_labels', True):
            return
        h, w = frame.shape[:2]
        margin = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text = f'Focus (center): {focus_value:6.1f}'
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = margin
        y = margin + text_h
        cv2.rectangle(
            frame,
            (x - 6, y - text_h - 6),
            (x + text_w + 6, y + baseline + 6),
            (0, 0, 0),
            -1
        )
        cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
        if roi_rect:
            cv2.rectangle(frame, (roi_rect[0], roi_rect[1]), (roi_rect[2], roi_rect[3]), (0, 255, 255), 2)

    def _test_image(self):
        if self.is_detection_running: self._stop_detection()
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if not file_name: return
        cv_img = cv2.imread(file_name)
        if cv_img is None:
            self.status_bar.showMessage('Error: Could not read image file', 4000); self.last_tested_image = None; return
        self.status_bar.showMessage(f'Loaded image: {os.path.basename(file_name)}'); self.last_tested_image = cv_img; self._reprocess_image()

    def _apply_visual_guides(self, frame):
        if frame is None or frame.size == 0:
            return
        if not (self.machine_guide_enabled or self.center_marker_enabled):
            return
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return
        thickness = max(1, min(w, h) // 200)
        guide_color = (0, 165, 255)
        marker_color = (0, 255, 255)

        if self.machine_guide_enabled:
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            top_left = (margin_x, margin_y)
            bottom_right = (w - margin_x, h - margin_y)
            cv2.rectangle(frame, top_left, bottom_right, guide_color, thickness)

            tick = max(thickness * 4, int(min(w, h) * 0.04))
            corner_points = [
                top_left,
                (bottom_right[0], top_left[1]),
                (top_left[0], bottom_right[1]),
                bottom_right
            ]
            offsets = [(tick, 0), (-tick, 0), (tick, 0), (-tick, 0)]
            vertical_offsets = [(0, tick), (0, tick), (0, -tick), (0, -tick)]
            for (x, y), (dx, dy), (vx, vy) in zip(corner_points, offsets, vertical_offsets):
                cv2.line(frame, (x, y), (x + dx, y + dy), guide_color, thickness)
                cv2.line(frame, (x, y), (x + vx, y + vy), guide_color, thickness)

        if self.center_marker_enabled:
            cx, cy = w // 2, h // 2
            arm = max(thickness * 8, int(min(w, h) * 0.06))
            cv2.line(frame, (cx - arm, cy), (cx + arm, cy), marker_color, thickness)
            cv2.line(frame, (cx, cy - arm), (cx, cy + arm), marker_color, thickness)
            cv2.circle(frame, (cx, cy), max(2, thickness * 2), marker_color, -1)

    def _reprocess_image(self):
        if self.last_tested_image is None or self.is_detection_running: return
        self.current_frame = self.last_tested_image.copy()
        processed_frame, mse, is_anomaly, _ = self.detector.process_frame(self.last_tested_image.copy())
        show_labels = getattr(self, 'show_video_labels', True)
        if is_anomaly and show_labels:
            h,w,_ = processed_frame.shape; text='Detections: 1'; (tw,_),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.putText(processed_frame, text, (w-tw-10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        mode_map = {'recon':'Recon','color':'HSV','hybrid':'Hybrid'}; mode = mode_map[self.detector.mode]
        self.status_bar.showMessage(f"Last Image Test | Mode: {mode} | MSE: {mse:.4f} | Anomaly: {'Yes' if is_anomaly else 'No'}")
        focus_value, roi_rect = self._compute_focus_measure(self.current_frame)
        self.focus_measure = focus_value
        self._apply_visual_guides(processed_frame)
        if show_labels:
            self._draw_focus_overlay(processed_frame, focus_value, None)
        pixmap = self.convert_cv_qt(processed_frame)
        self.video_window.video_label.setPixmap(pixmap)
        self._last_pixmap_size = (pixmap.width(), pixmap.height())

    def _handle_zoom(self, delta):
        if delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        self.zoom_level = max(1.0, min(self.zoom_level, 10.0)) # Clamp zoom level
        if self.zoom_level == 1.0:
            self.pan_offset = QPoint(0,0) # Reset pan on zoom out
        if not self.is_detection_running:
            self._reprocess_image()

    def _handle_pan(self, dx, dy):
        # Introduce a sensitivity factor for smoother panning
        sensitivity = 0.5
        self.pan_offset.setX(self.pan_offset.x() + int(dx * sensitivity))
        self.pan_offset.setY(self.pan_offset.y() + int(dy * sensitivity))
        if not self.is_detection_running:
            self._reprocess_image()


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        if self.is_paused:
            return
        self.trigger_process.emit(cv_img)

    @pyqtSlot(np.ndarray, np.ndarray, float, bool, int)
    def display_processed_frame(self, processed_frame, original_frame, mse, is_anomaly, anomaly_count):
        self.current_frame = original_frame
        self.frame_count += 1; elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count/elapsed; self.frame_count = 0; self.start_time = time.time()
        show_labels = getattr(self, 'show_video_labels', True)
        if show_labels:
            h,w,_ = processed_frame.shape; margin = 10
            cv2.putText(processed_frame, f'FPS: {self.fps:.2f}', (margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            res_text = f'{w}x{h}'; (rw,_),_ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(processed_frame, res_text, (w-rw-margin, h-margin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            det_text = f'Detections: {anomaly_count}'; (dw,_),_ = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            cv2.putText(processed_frame, det_text, (w-dw-margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        focus_value, roi_rect = self._compute_focus_measure(original_frame)
        self.focus_measure = focus_value
        self._apply_visual_guides(processed_frame)
        if show_labels:
            self._draw_focus_overlay(processed_frame, focus_value, None)
        pixmap = self.convert_cv_qt(processed_frame)
        self.video_window.video_label.setPixmap(pixmap)
        self._last_pixmap_size = (pixmap.width(), pixmap.height())

    def convert_cv_qt(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape; bytes_per_line = ch*w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        target_size = self.video_window.video_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.video_window.size()
        pixmap = QPixmap.fromImage(qimg)
        if self.zoom_level > 1.0:
            w, h = pixmap.width(), pixmap.height()
            zoom_w, zoom_h = int(w / self.zoom_level), int(h / self.zoom_level)

            # Apply panning
            center_x = w // 2 - self.pan_offset.x()
            center_y = h // 2 - self.pan_offset.y()

            # Calculate crop coordinates
            crop_x = center_x - zoom_w // 2
            crop_y = center_y - zoom_h // 2

            # Clamp crop coordinates to be within image bounds
            crop_x = max(0, min(crop_x, w - zoom_w))
            crop_y = max(0, min(crop_y, h - zoom_h))

            # Update pan offset to reflect clamping
            self.pan_offset.setX(w // 2 - (crop_x + zoom_w // 2))
            self.pan_offset.setY(h // 2 - (crop_y + zoom_h // 2))


            pixmap = pixmap.copy(crop_x, crop_y, zoom_w, zoom_h)
        return pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def _save_settings(self):
        s = QSettings('config.ini', QSettings.Format.IniFormat)
        s.setValue('model_path', self.model_path_edit.text())
        s.setValue('mse_threshold', self.thresh_slider.value())
        s.setValue('cv_threshold', self.cv_thresh_slider.value())
        s.setValue('contour_area', self.contour_slider.value())
        if hasattr(self, 'backend_combo'):
            s.setValue('backend_index', self.backend_combo.currentIndex())
        if hasattr(self, 'fourcc_combo'):
            s.setValue('fourcc_text', self.fourcc_combo.currentText())
        if hasattr(self, 'hue1_enable_check'):
            s.setValue('hue1_enabled', self.hue1_enable_check.isChecked())
        if hasattr(self, 'hue2_enable_check'):
            s.setValue('hue2_enabled', self.hue2_enable_check.isChecked())
        if hasattr(self, 'hue3_enable_check'):
            s.setValue('hue3_enabled', self.hue3_enable_check.isChecked())
        s.setValue('camera_index', self.cam_combo.currentIndex())
        s.setValue('resolution_text', self.res_combo.currentText())
        s.setValue('fps_limit_text', self.fps_combo.currentText())
        s.setValue('auto_save', self.auto_save_check.isChecked())
        if hasattr(self, 'video_labels_check'):
            s.setValue('video_labels_enabled', self.video_labels_check.isChecked())
        s.setValue('mode_index', self.mode_combo.currentIndex())
        s.setValue('h_low', self.h_low_slider.value())
        s.setValue('h_high', self.h_high_slider.value())
        s.setValue('s_min', self.s_min_slider.value())
        s.setValue('v_min', self.v_min_slider.value())
        s.setValue('h2_low', self.h2_low_slider.value())
        s.setValue('h2_high', self.h2_high_slider.value())
        s.setValue('s2_min', self.s2_min_slider.value())
        s.setValue('v2_min', self.v2_min_slider.value())
        s.setValue('h3_low', self.h3_low_slider.value())
        s.setValue('h3_high', self.h3_high_slider.value())
        s.setValue('s3_min', self.s3_min_slider.value())
        s.setValue('v3_min', self.v3_min_slider.value())
        s.setValue('beep_enabled', self.beep_check.isChecked())
        s.setValue('random_save', self.random_save_check.isChecked())
        s.setValue('hsv_target_lock', self.hsv_lock_check.isChecked())
        s.setValue('machine_guide', self.machine_guide_check.isChecked())
        s.setValue('center_marker', self.center_marker_check.isChecked())
        if hasattr(self, 'tripwire_check'):
            s.setValue('tripwire_enabled', self.tripwire_check.isChecked())

    def _load_settings(self):
        s = QSettings('config.ini', QSettings.Format.IniFormat)
        model_path = s.value('model_path','')
        if model_path and os.path.exists(model_path): self._load_model_action(model_path)
        self.thresh_slider.setValue(s.value('mse_threshold',10,type=int))
        self.cv_thresh_slider.setValue(s.value('cv_threshold',40,type=int))
        self.contour_slider.setValue(s.value('contour_area',10,type=int))
        if hasattr(self, 'backend_combo'):
            backend_idx = s.value('backend_index', 0, type=int)
            if 0 <= backend_idx < self.backend_combo.count():
                self.backend_combo.setCurrentIndex(backend_idx)
            else:
                self.backend_combo.setCurrentIndex(0)
        if hasattr(self, 'fourcc_combo'):
            fourcc_text = s.value('fourcc_text', 'Auto')
            if fourcc_text in self.fourcc_options:
                self.fourcc_combo.setCurrentText(fourcc_text)
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
        self.h2_low_slider.setValue(s.value('h2_low',75,type=int)); self.h2_high_slider.setValue(s.value('h2_high',95,type=int))
        self.s2_min_slider.setValue(s.value('s2_min',60,type=int)); self.v2_min_slider.setValue(s.value('v2_min',120,type=int))
        self.h3_low_slider.setValue(s.value('h3_low',105,type=int)); self.h3_high_slider.setValue(s.value('h3_high',125,type=int))
        self.s3_min_slider.setValue(s.value('s3_min',60,type=int)); self.v3_min_slider.setValue(s.value('v3_min',120,type=int))
        self.h_low_label.setText(str(self.h_low_slider.value())); self.h_high_label.setText(str(self.h_high_slider.value()))
        self.s_min_label.setText(str(self.s_min_slider.value())); self.v_min_label.setText(str(self.v_min_slider.value()))
        self.h2_low_label.setText(str(self.h2_low_slider.value())); self.h2_high_label.setText(str(self.h2_high_slider.value()))
        self.s2_min_label.setText(str(self.s2_min_slider.value())); self.v2_min_label.setText(str(self.v2_min_slider.value()))
        self.h3_low_label.setText(str(self.h3_low_slider.value())); self.h3_high_label.setText(str(self.h3_high_slider.value()))
        self.s3_min_label.setText(str(self.s3_min_slider.value())); self.v3_min_label.setText(str(self.v3_min_slider.value()))
        self.detector.set_hsv_thresholds(self.h_low_slider.value(), self.h_high_slider.value(), self.s_min_slider.value(), self.v_min_slider.value())
        self.detector.set_hsv_secondary(self.h2_low_slider.value(), self.h2_high_slider.value(), self.s2_min_slider.value(), self.v2_min_slider.value())
        self.detector.set_hsv_tertiary(self.h3_low_slider.value(), self.h3_high_slider.value(), self.s3_min_slider.value(), self.v3_min_slider.value())
        self._enable_hsv_controls(self.mode_combo.currentIndex() in (1,2))
        self._update_hsv_summary()
        self.beep_check.setChecked(s.value('beep_enabled', False, type=bool))
        self.machine_guide_check.setChecked(s.value('machine_guide', True, type=bool))
        self.center_marker_check.setChecked(s.value('center_marker', True, type=bool))
        self.detection_worker.set_beep_enabled(self.beep_check.isChecked())
        random_save = s.value('random_save', False, type=bool)
        self.random_save_check.setChecked(random_save)
        self.detection_worker.set_random_save_enabled(random_save)
        if hasattr(self, 'video_labels_check'):
            labels_enabled = s.value('video_labels_enabled', True, type=bool)
            self.video_labels_check.blockSignals(True)
            self.video_labels_check.setChecked(labels_enabled)
            self.video_labels_check.blockSignals(False)
            self.show_video_labels = labels_enabled
        if hasattr(self, 'hue1_enable_check'):
            hue1_enabled = s.value('hue1_enabled', True, type=bool)
            self.hue1_enable_check.blockSignals(True)
            self.hue1_enable_check.setChecked(hue1_enabled)
            self.hue1_enable_check.blockSignals(False)
            self.detector.set_primary_hue_enabled(hue1_enabled)
        if hasattr(self, 'hue2_enable_check'):
            hue2_enabled = s.value('hue2_enabled', True, type=bool)
            self.hue2_enable_check.blockSignals(True)
            self.hue2_enable_check.setChecked(hue2_enabled)
            self.hue2_enable_check.blockSignals(False)
            self.detector.set_secondary_hue_enabled(hue2_enabled)
        if hasattr(self, 'hue3_enable_check'):
            hue3_enabled = s.value('hue3_enabled', True, type=bool)
            self.hue3_enable_check.blockSignals(True)
            self.hue3_enable_check.setChecked(hue3_enabled)
            self.hue3_enable_check.blockSignals(False)
            self.detector.set_tertiary_hue_enabled(hue3_enabled)
        self._update_hsv_summary()
        lock_hsv = s.value('hsv_target_lock', False, type=bool)
        self.hsv_lock_check.setChecked(lock_hsv)
        self.hsv_target_locked = lock_hsv

        if hasattr(self, 'tripwire_check'):
            tripwire_enabled = s.value('tripwire_enabled', False, type=bool)
            self.tripwire_check.setChecked(tripwire_enabled)
            self.detection_worker.set_tripwire_enabled(tripwire_enabled)

        if mode_idx in (1,2) and not model_path:
            self.start_btn.setDisabled(False); self.test_image_btn.setDisabled(False)

    def closeEvent(self, event):
        self._save_settings(); self._stop_detection(); self.video_window.close(); self.detection_thread.quit(); self.detection_thread.wait(); event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
