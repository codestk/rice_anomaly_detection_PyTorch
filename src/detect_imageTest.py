# src/detect.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from train import Autoencoder # Import a class from another file.

# --- การตั้งค่า ---
MODEL_PATH = 'models/rice_anomaly_detector_pytorch.pth'
IMG_SIZE = (128, 128)
TEST_IMAGE_PATH = 'data/test/09b094ac-capture_1757494850.png'

# --- กำหนดอุปกรณ์ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image_pytorch(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    resized_img = cv2.resize(img, IMG_SIZE)
    processed_img = resized_img.astype('float32') / 255.0
    
    # แปลงเป็น PyTorch Tensor และปรับมิติเป็น (1, C, H, W)
    processed_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0)
    
    return original_image, processed_tensor

def main():
    # โหลดโมเดล
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # ตั้งเป็น evaluation mode
    print(f"Model loaded from {MODEL_PATH}")

    original_image, processed_tensor = preprocess_image_pytorch(TEST_IMAGE_PATH)
    if original_image is None:
        print(f"Error reading image: {TEST_IMAGE_PATH}")
        return

    processed_tensor = processed_tensor.to(device)

    # ทำนายโดยไม่ต้องคำนวณ gradient
    with torch.no_grad():
        reconstructed_tensor = model(processed_tensor)

    # คำนวณค่า MSE
    mse = torch.mean((processed_tensor - reconstructed_tensor) ** 2).item()
    print(f"Image Path: {TEST_IMAGE_PATH}")
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # --- แสดงผล ---
    # แปลง Tensor กลับเป็น NumPy array เพื่อแสดงผล
    reconstructed_img = reconstructed_tensor.cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_img, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    diff = np.abs(processed_tensor.cpu().squeeze().numpy() - reconstructed_img)
    heatmap = axes[2].imshow(diff, cmap='jet')
    axes[2].set_title(f'Difference (MSE: {mse:.4f})')
    axes[2].axis('off')
    fig.colorbar(heatmap, ax=axes[2])

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()