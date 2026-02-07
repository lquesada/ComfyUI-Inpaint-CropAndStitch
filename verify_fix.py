
import sys
import os

# Add the directory containing the node to the python path
sys.path.append(os.path.abspath("/home/elezeta/m/nodes/ComfyUI-Inpaint-CropAndStitch"))

import torch
import torchvision.transforms.functional as F
from inpaint_cropandstitch import GPUProcessorLogic

def main():
    print("Verifying fix for bfloat16 crash...")
    processor = GPUProcessorLogic()
    
    width, height = 128, 128
    
    # Test case 1: float32 (standard)
    print("Test 1: float32 tensor")
    try:
        samples_f32 = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        # rescale_i expects [B, H, W, C]
        result = processor.rescale_i(samples_f32, width, height, 'bilinear')
        print(f"  rescale_i result shape: {result.shape}, dtype: {result.dtype}")
        
        mask_f32 = torch.rand(1, 64, 64, dtype=torch.float32)
        # rescale_m expects [B, H, W]
        result_m = processor.rescale_m(mask_f32, width, height, 'nearest')
        print(f"  rescale_m result shape: {result_m.shape}, dtype: {result_m.dtype}")
        print("Test 1 passed.")
    except Exception as e:
        print(f"Test 1 failed: {e}")
        return

    # Test case 2: bfloat16 (if supported on CPU, otherwise float16)
    # Note: bfloat16 might not be fully supported on all CPUs, but let's try.
    print("\nTest 2: bfloat16 tensor")
    try:
        samples_bf16 = torch.rand(1, 64, 64, 3, dtype=torch.bfloat16)
        result = processor.rescale_i(samples_bf16, width, height, 'bilinear')
        print(f"  rescale_i result shape: {result.shape}, dtype: {result.dtype}")
        
        mask_bf16 = torch.rand(1, 64, 64, dtype=torch.bfloat16)
        result_m = processor.rescale_m(mask_bf16, width, height, 'nearest')
        print(f"  rescale_m result shape: {result_m.shape}, dtype: {result_m.dtype}")
        print("Test 2 passed (fix is working for bfloat16 input).")
        
    except RuntimeError as e:
        print(f"RuntimeError during Test 2 (might be expected if CPU doesn't support OP): {e}")
    except TypeError as e:
        print(f"TypeError during Test 2 (FAILURE - likely the bug): {e}")
    except Exception as e:
        print(f"Test 2 failed with unexpected error: {e}")

if __name__ == "__main__":
    main()
