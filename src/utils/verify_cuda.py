import torch

def verify_cuda_setup():
    print("\n=== CUDA Setup Verification ===")
    
    # Check CUDA availability
    print("\n1. CUDA Availability:")
    print(f"   CUDA is available: {torch.cuda.is_available()}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        # Get device properties
        current_device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(current_device)
        
        print("\n2. GPU Information:")
        print(f"   Device name: {device_props.name}")
        print(f"   Compute capability: {device_props.major}.{device_props.minor}")
        print(f"   Total memory: {device_props.total_memory / 1024**3:.2f} GB")
        
        # Check current memory usage
        print("\n3. Current Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Test basic CUDA operations
        print("\n4. Testing CUDA Operations:")
        try:
            # Create and move a tensor to GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)  # Perform a matrix multiplication
            print("   ✓ Basic CUDA operations successful")
            
            # Clean up test tensors
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ✗ CUDA operation test failed: {str(e)}")
    
    print("\n=== Verification Complete ===\n")

if __name__ == "__main__":
    verify_cuda_setup()

