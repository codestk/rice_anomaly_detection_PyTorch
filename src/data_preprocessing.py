# src/data_preprocessing.py
# (ใช้โค้ดเดิมจากคำตอบก่อนหน้าได้เลย ไม่มีการเปลี่ยนแปลง)
import os
import cv2
import numpy as np

def load_and_preprocess_images(folder_path, img_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
    
    images = np.array(images).astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1) # Shape: (N, 128, 128, 1)
    return images

if __name__ == '__main__':
    TRAIN_DIR = 'data/train'
    print("Processing training images...")
    train_images = load_and_preprocess_images(TRAIN_DIR)
    np.save('data/train_data.npy', train_images)
    print(f"Preprocessing complete. Shape: {train_images.shape}")