import cv2
import numpy as np
import os
from datetime import datetime

# ค่าช่วงสีปัจจุบัน (ตัวอย่าง)
lower = np.array([20, 100, 100], dtype=np.uint8)  # H,S,V
upper = np.array([35, 255, 255], dtype=np.uint8)

# ระยะขอบที่ถือว่า "ใกล้" (ควรเริ่มเล็กแล้วค่อยปรับ)
dH, dS, dV = 5, 20, 20

def clamp_luv(x, lo, hi):  # helper กันค่าติดลบ/เกิน 255
    return np.clip(x, lo, hi).astype(np.uint8)

# คำนวณขอบชั้นใน/ชั้นนอก
inner_lower = clamp_luv(lower + np.array([dH, dS, dV]), 0, 255)
inner_upper = clamp_luv(upper - np.array([dH, dS, dV]), 0, 255)
outer_lower = clamp_luv(lower - np.array([dH, dS, dV]), 0, 255)
outer_upper = clamp_luv(upper + np.array([dH, dS, dV]), 0, 255)

def log_near_border_if_needed(frame_bgr, save_dir="logs/hsv_border", ratio_threshold=0.02, min_pixels=2000):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # หน้ากากตามช่วงจริง
    mask = cv2.inRange(hsv, lower, upper)

    # ชั้นใน/ชั้นนอก
    mask_inner = cv2.inRange(hsv, inner_lower, inner_upper)
    mask_outer = cv2.inRange(hsv, outer_lower, outer_upper)

    # แถบ “ใกล้ขอบ” = พื้นที่ที่อยู่ใน outer แต่ไม่อยู่ใน inner (exclusive region)
    near_band = cv2.bitwise_xor(mask_outer, mask_inner)

    # เกณฑ์ตัดสินว่าจะเซฟไหม
    nb_pixels = int(np.count_nonzero(near_band))
    h, w = near_band.shape[:2]
    ratio = nb_pixels / float(h * w)

    if (ratio >= ratio_threshold) or (nb_pixels >= min_pixels):
        # เตรียมไฟล์/โฟลเดอร์
        date_dir = datetime.now().strftime("%Y%m%d")
        folder = os.path.join(save_dir, date_dir)
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # ทำ overlay เพื่อรีวิวง่าย: ขีดเส้น/เทสีบริเวณ near-band เป็นสีเด่น
        overlay = frame_bgr.copy()
        overlay[near_band > 0] = (0, 0, 255)  # แต่งสี (แดง) เฉพาะบริเวณใกล้ขอบ

        # สถิติ HSV ของบริเวณ near-band (เอาไว้ปรับช่วง)
        nb_idx = near_band > 0
        h_vals = hsv[...,0][nb_idx].astype(np.int32)
        s_vals = hsv[...,1][nb_idx].astype(np.int32)
        v_vals = hsv[...,2][nb_idx].astype(np.int32)
        def stats(arr):
            if arr.size == 0:
                return (None, None, None, None)
            return (int(np.mean(arr)), int(np.std(arr)),
                    int(np.percentile(arr, 10)), int(np.percentile(arr, 90)))

        h_mean,h_std,h_p10,h_p90 = stats(h_vals)
        s_mean,s_std,s_p10,s_p90 = stats(s_vals)
        v_mean,v_std,v_p10,v_p90 = stats(v_vals)

        # เซฟรูป
        cv2.imwrite(os.path.join(folder, f"{ts}_frame.jpg"), frame_bgr)
        cv2.imwrite(os.path.join(folder, f"{ts}_mask.png"), mask)
        cv2.imwrite(os.path.join(folder, f"{ts}_near_band.png"), near_band)
        cv2.imwrite(os.path.join(folder, f"{ts}_overlay.jpg"), overlay)

        # เซฟ meta (อ่านง่าย)
        meta = (
            f"time={ts}\n"
            f"lower={tuple(int(x) for x in lower)}, upper={tuple(int(x) for x in upper)}\n"
            f"delta=(dH={dH}, dS={dS}, dV={dV})\n"
            f"near_pixels={nb_pixels}, ratio={ratio:.4f}\n"
            f"H(mean={h_mean}, std={h_std}, p10={h_p10}, p90={h_p90})\n"
            f"S(mean={s_mean}, std={s_std}, p10={s_p10}, p90={s_p90})\n"
            f"V(mean={v_mean}, std={v_std}, p10={v_p10}, p90={v_p90})\n"
        )
        with open(os.path.join(folder, f"{ts}_meta.txt"), "w", encoding="utf-8") as f:
            f.write(meta)

        return True, meta  # เพื่อขึ้น log ใน UI ถ้าต้องการ

    return False, None
