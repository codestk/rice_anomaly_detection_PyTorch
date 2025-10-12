import torch

def check_cuda():
    """
    Checks the PyTorch and CUDA installation and provides detailed feedback.
    """
    print("--- PyTorch CUDA Installation Checker ---")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    is_available = torch.cuda.is_available()
    print(f"Is CUDA available? -> {is_available}")
    
    if not is_available:
        print("\n[ERROR] CUDA is not available for PyTorch.")
        print("Please check the following:")
        print("1. Do you have an NVIDIA GPU in your system?")
        print("2. Are your NVIDIA drivers correctly installed and up-to-date?")
        print("3. Did you install the PyTorch version that includes CUDA support?")
        print("   You can get the correct command from the official PyTorch website:")
        print("   https://pytorch.org/get-started/locally/")
        return

    # If CUDA is available, print more details
    try:
        print(f"\nCUDA version used by PyTorch: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}")
        
        if device_count > 0:
            for i in range(device_count):
                print(f"\n--- Details for GPU {i} ---")
                print(f"Name: {torch.cuda.get_device_name(i)}")
                
                # Simple test: create a tensor and move it to the GPU
                tensor = torch.tensor([1.0, 2.0]).to(f"cuda:{i}")
                print(f"Successfully moved a test tensor to GPU {i} (Device: {tensor.device})")
        
        print("\n[SUCCESS] Your PyTorch and CUDA setup appears to be working correctly!")

    except Exception as e:
        print(f"\n[ERROR] An error occurred while checking CUDA devices: {e}")
        print("This might indicate an issue with your NVIDIA drivers or a mismatch with the CUDA toolkit.")

if __name__ == "__main__":
    check_cuda()
