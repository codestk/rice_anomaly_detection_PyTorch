"""cv2.VideoCapture-compatible adapter for HuaTeng/mvsdk industrial cameras.

The HuaTeng camera does not register as a Windows multimedia capture device,
so it is invisible to cv2.VideoCapture / QMediaDevices. This module talks to
it directly through the vendor's mvsdk API (bundled under
NewCamera/python_demo/mvsdk.py) and exposes just enough of the
cv2.VideoCapture surface (isOpened/read/set/get/release) that the existing
VideoThread and open_camera_by_index code can use it unchanged.
"""
import math
import os
import sys
import time
import platform

import cv2
import numpy as np

_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NewCamera", "python_demo")
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

try:
    import mvsdk
except ImportError:
    mvsdk = None

# Exposure values coming from the app's UI slider use the same log2-ish
# DirectShow scale as the webcam path (e.g. -13..0). mvsdk wants an absolute
# exposure time in microseconds, so we remap the slider value onto the
# HT-SUA134GC/M's documented exposure range (4.5us - 584.4ms) instead of
# reworking the slider UI.
_EXPOSURE_US_MIN = 5
_EXPOSURE_US_MAX = 584_400


def _slider_value_to_exposure_us(value):
    value = max(-13, min(0, int(value)))
    # value=-13 -> fastest/darkest, value=0 -> slowest/brightest.
    # The sensor's exposure range spans 5us-584ms (5 orders of magnitude),
    # so interpolate in log space or most slider steps would collapse into
    # the top end of the range.
    frac = (value + 13) / 13.0
    log_min, log_max = math.log10(_EXPOSURE_US_MIN), math.log10(_EXPOSURE_US_MAX)
    return int(round(10 ** (log_min + frac * (log_max - log_min))))


def is_available():
    return mvsdk is not None


def list_devices():
    """Return [(index, friendly_name), ...] for connected mvsdk cameras."""
    if mvsdk is None:
        return []
    try:
        dev_list = mvsdk.CameraEnumerateDevice()
    except Exception:
        return []
    return [(i, dev.GetFriendlyName()) for i, dev in enumerate(dev_list)]


class MvsdkCapture:
    """Minimal cv2.VideoCapture-like wrapper around one mvsdk camera."""

    def __init__(self, index, resolution=None, exposure_value=None):
        self._hcamera = 0
        self._frame_buffer = 0
        self._mono = False
        self._opened = False
        self._open(index, resolution, exposure_value)

    def _open(self, index, resolution, exposure_value):
        dev_list = mvsdk.CameraEnumerateDevice()
        if not (0 <= index < len(dev_list)):
            return
        dev_info = dev_list[index]
        try:
            hcamera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as err:
            print(f"[WARN {time.time():.2f}] mvsdk CameraInit failed({err.error_code}): {err.message}")
            return

        cap = mvsdk.CameraGetCapability(hcamera)
        self._mono = (cap.sIspCapacity.bMonoSensor != 0)
        mvsdk.CameraSetIspOutFormat(
            hcamera,
            mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self._mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8,
        )
        mvsdk.CameraSetTriggerMode(hcamera, 0)  # continuous acquisition

        # The camera boots into its slowest speed mode (Normal) unless told
        # otherwise; the datasheet FPS figure only applies at the fastest of
        # the Normal/High/Super speed levels reported by iFrameSpeedDesc.
        if cap.iFrameSpeedDesc > 0:
            try:
                mvsdk.CameraSetFrameSpeed(hcamera, cap.iFrameSpeedDesc - 1)
            except mvsdk.CameraException as err:
                print(f"[WARN {time.time():.2f}] Unable to set camera frame speed({err.error_code}): {err.message}")

        mvsdk.CameraSetAeState(hcamera, 0)  # manual exposure
        exposure_us = _slider_value_to_exposure_us(exposure_value) if exposure_value is not None else 30 * 1000
        mvsdk.CameraSetExposureTime(hcamera, exposure_us)

        if not self._mono:
            # Keep white balance fixed/manual so hue-based thresholds stay
            # stable frame-to-frame; color is corrected on demand via
            # set_white_balance_once() (one-shot calibration against a
            # neutral gray/white target) instead of continuous auto-WB,
            # which would drift the hue readings while detection is running.
            try:
                mvsdk.CameraSetWbMode(hcamera, 0)
            except mvsdk.CameraException as err:
                print(f"[WARN {time.time():.2f}] Unable to disable auto white balance({err.error_code}): {err.message}")

        mvsdk.CameraPlay(hcamera)

        buffer_size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if self._mono else 3)
        self._frame_buffer = mvsdk.CameraAlignMalloc(buffer_size, 16)
        self._hcamera = hcamera
        self._opened = True
        # resolution is left at the camera's default/native setting; mvsdk
        # resolution changes go through CameraSetImageResolution and are not
        # wired up here since the app runs "Source/Native" for this backend.
        _ = resolution

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        try:
            praw, frame_head = mvsdk.CameraGetImageBuffer(self._hcamera, 200)
            mvsdk.CameraImageProcess(self._hcamera, praw, self._frame_buffer, frame_head)
            mvsdk.CameraReleaseImageBuffer(self._hcamera, praw)
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self._frame_buffer, frame_head, 1)
            data = (mvsdk.c_ubyte * frame_head.uBytes).from_address(self._frame_buffer)
            frame = np.frombuffer(data, dtype=np.uint8)
            channels = 1 if frame_head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            frame = frame.reshape((frame_head.iHeight, frame_head.iWidth, channels)).copy()
            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return True, frame
        except mvsdk.CameraException as err:
            if err.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print(f"[WARN {time.time():.2f}] mvsdk CameraGetImageBuffer failed({err.error_code}): {err.message}")
            return False, None

    def set_white_balance_once(self):
        """Trigger a single white-balance calibration against whatever the
        camera currently sees. Point it at a neutral gray/white area first."""
        if not self._opened or self._mono:
            return False
        try:
            mvsdk.CameraSetOnceWB(self._hcamera)
            return True
        except mvsdk.CameraException as err:
            print(f"[WARN {time.time():.2f}] CameraSetOnceWB failed({err.error_code}): {err.message}")
            return False

    def set(self, prop_id, value):
        if prop_id == cv2.CAP_PROP_EXPOSURE and self._opened:
            try:
                mvsdk.CameraSetExposureTime(self._hcamera, _slider_value_to_exposure_us(value))
                return True
            except mvsdk.CameraException:
                return False
        return False

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FOURCC:
            return 0
        return 0

    def release(self):
        if self._hcamera:
            mvsdk.CameraUnInit(self._hcamera)
            self._hcamera = 0
        if self._frame_buffer:
            mvsdk.CameraAlignFree(self._frame_buffer)
            self._frame_buffer = 0
        self._opened = False


def open_mvsdk_camera(index, resolution=None, exposure_value=None):
    """Mirror open_camera_by_index's (cap, fourcc_label) return signature."""
    if mvsdk is None:
        print(f"[WARN {time.time():.2f}] mvsdk module not importable; is the HuaTeng SDK installed?")
        return None, None
    cap = MvsdkCapture(index, resolution=resolution, exposure_value=exposure_value)
    if not cap.isOpened():
        return None, None
    label = "MVSDK MONO8" if cap._mono else "MVSDK BGR8"
    return cap, label
