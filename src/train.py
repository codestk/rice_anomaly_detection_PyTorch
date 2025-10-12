# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- การตั้งค่า ---
DATA_PATH = 'data/train_data.npy'
MODEL_SAVE_PATH = 'models/rice_anomaly_detector_pytorch.pth'
NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# --- กำหนดอุปกรณ์ (GPU หรือ CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- ส่วนที่แก้ไข ---
if str(device) == "cuda":
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print(f"Using device: CPU")
# --------------------

# --- สร้างโมเดล Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (B, 1, 128, 128) -> (B, 32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> (B, 32, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # -> (B, 64, 32, 32)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # -> (B, 32, 32, 32)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # -> (B, 32, 64, 64)
            nn.Conv2d(32, 1, kernel_size=3, padding=1), # -> (B, 1, 64, 64)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # -> (B, 1, 128, 128)
            nn.Sigmoid() # ให้ค่า output อยู่ระหว่าง 0-1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model():
    # โหลดและเตรียมข้อมูล
    train_data_np = np.load(DATA_PATH) # Shape: (N, 128, 128, 1)
    # PyTorch ต้องการ (N, C, H, W)
    train_data_np = np.transpose(train_data_np, (0, 3, 1, 2))
    train_tensor = torch.from_numpy(train_data_np)
    
    dataset = TensorDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # สร้างโมเดล, Loss function, และ Optimizer
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        for data in dataloader:
            img = data[0].to(device)
            
            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}')

    # บันทึกโมเดล
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_model()