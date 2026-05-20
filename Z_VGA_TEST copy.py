<<<<<<< HEAD
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
=======
import cv2
import os

# ===== CONFIG =====
IMAGE_FOLDER = r"D:\rice_anomaly_detection_PyTorch\data\train"
THRESHOLD = 20   # ⭐ ปรับตรงนี้
# ==================

def is_blurry(image_path, threshold=20):
    image = cv2.imread(image_path)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance, variance < threshold


def scan_folder(folder):
    blurry_files = []

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(root, file)

                result = is_blurry(path, THRESHOLD)
                if result is None:
                    continue

                variance, is_blur = result

                if is_blur:
                    #filename = os.path.basename(path)
                    filename = path
                    blurry_files.append(filename)

    return blurry_files


if __name__ == "__main__":
    blurry = scan_folder(IMAGE_FOLDER)

    print("\n=== BLURRY IMAGES (sharpness < 20) ===")
    for name in blurry:
        print(name)

    print(f"\nTotal blurry: {len(blurry)}")




# import cv2
# import os

# # ===== CONFIG =====
# IMAGE_FOLDER = r"C:\yoloTrain\custom_data\images"
# THRESHOLD = 100
# # ==================

# def is_blurry(image_path, threshold=100):
#     image = cv2.imread(image_path)

#     if image is None:
#         return None

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()

#     return variance, variance < threshold


# def scan_folder(folder):
#     blurry_files = []

#     for root, dirs, files in os.walk(folder):
#         for file in files:
#             if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
#                 path = os.path.join(root, file)

#                 result = is_blurry(path, THRESHOLD)
#                 if result is None:
#                     continue

#                 variance, is_blur = result

#                 if is_blur:
#                     # ⭐ เอาแค่ชื่อไฟล์
#                     filename = os.path.basename(path)
#                     blurry_files.append((filename, variance))

#     return blurry_files


# if __name__ == "__main__":
#     blurry = scan_folder(IMAGE_FOLDER)

#     print("\n=== BLURRY IMAGES (FILENAME ONLY) ===")
#     for name, v in blurry:
#         print(f"{name} | sharpness={v:.2f}")

#     print(f"\nTotal blurry: {len(blurry)}")
>>>>>>> 52b1e18a5c32dc75886255b5c56dbdc23115dc1a
