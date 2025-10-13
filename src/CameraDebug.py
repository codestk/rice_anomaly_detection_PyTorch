import cv2, numpy as np, math, os

def fourcc_str(v):
    try:
        iv = int(v)
        if iv == 0 or math.isnan(v):   # MSMF มักคืน 0
            return "UNKNOWN"
        return "".join([chr((iv >> (8*i)) & 0xFF) for i in range(4)])
    except Exception:
        return "UNKNOWN"

api = cv2.CAP_MSMF  # หรือ CAP_DSHOW ลองเทียบ
cap = cv2.VideoCapture(0, api)

# ขอ YUY2 ก่อน (MSMF จะรับไว้เองหรือแปลงเป็น BGR)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# (ทางเลือก) ขอ "ไม่ต้องแปลงเป็น RGB" ถ้า backend รองรับ
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

# อ่านเฟรมทดสอบ
ok, frame = cap.read()
print("ok:", ok)
print("frame shape:", None if not ok else frame.shape)  # เช่น (720,1280,3) = BGR 3 ช่อง
print("dtype:", None if not ok else frame.dtype)

# ลองดูค่าที่รายงานกลับ
fourcc_got = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
fmt = cap.get(cv2.CAP_PROP_FORMAT)         # รหัสชนิด Mat (เช่น CV_8UC3 ~ 16)
backend = cap.get(cv2.CAP_PROP_BACKEND)    # backend id

print("Negotiated FOURCC reported:", fourcc_got)  # MSMF อาจเป็น UNKNOWN
print("CAP_PROP_FORMAT:", fmt)  # ถ้าเป็น 16 (CV_8UC3) = BGR 8-bit 3 ช่อง
print("BACKEND id:", backend)

cap.release()
