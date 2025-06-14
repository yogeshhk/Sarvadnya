import sys
import torch
import torchvision
import torchaudio

print("python version:", sys.version)
print("python version info:", sys.version_info)
print("torch version:", torch.__version__)
print("cuda version (torch):", torch.version.cuda)
print("torchvision version:", torchvision.__version__)
print("torchaudio version:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())

try:
    import flash_attn
    print("flash-attention version:", flash_attn.__version__)
except ImportError:
    print("flash-attention is not installed or cannot be imported")

try:
    import triton
    print("triton version:", triton.__version__)
except ImportError:
    print("triton is not installed or cannot be imported")

try:
    import sageattention
    print("sageattention version:", sageattention.__version__)
except ImportError:
    print("sageattention is not installed or cannot be imported")
except AttributeError:
    print("sageattention is installed but has no __version__ attribute")