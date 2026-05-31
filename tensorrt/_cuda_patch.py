"""
_cuda_patch.py — Pre-load CUDA and cuDNN libraries to prevent runtime link errors.
================================================================================
Imports of onnxruntime-gpu or torch may cause segmentation faults or driver issues
if system library search paths (LD_LIBRARY_PATH) or library caches are not primed.
This helper resolves library paths dynamically from the active python environment and
preloads them.
"""

import os
import sys
import ctypes

def _patch_library_paths():
    """
    Prepend all known CUDA/cuDNN library directories to LD_LIBRARY_PATH and
    pre-load critical shared objects via ctypes (RTLD_GLOBAL) so that the OS
    linker cache is primed before onnxruntime's dlopen() calls fire.
    """
    # Detect SP dynamically
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    SP = os.path.join(sys.prefix, "lib", f"python{python_ver}", "site-packages", "nvidia")

    NVIDIA_LIBS = [
        os.path.join(SP, "cuda_runtime", "lib"),    # libcudart.so.12
        os.path.join(SP, "cublas", "lib"),          # libcublasLt.so.12, libcublas.so.12
        os.path.join(SP, "cudnn", "lib"),           # libcudnn.so.9
        os.path.join(SP, "curand", "lib"),          # libcurand.so.10
        os.path.join(SP, "cufft", "lib"),           # libcufft.so.11
        os.path.join(SP, "cusparse", "lib"),        # libcusparse.so.12
        os.path.join(SP, "cusolver", "lib"),        # libcusolver.so.11
        os.path.join(SP, "nccl", "lib"),            # libnccl.so.2
        os.path.join(SP, "cuda_nvrtc", "lib"),      # libnvrtc.so
        os.path.join(SP, "cuda_cupti", "lib"),      # libcupti.so
        os.path.join(SP, "nvtx", "lib"),            # libnvToolsExt.so
    ]

    # System CUDA toolkit dirs
    SYSTEM_CUDA = [
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
    ]

    all_dirs = [d for d in NVIDIA_LIBS + SYSTEM_CUDA if os.path.isdir(d)]

    # Prepend to LD_LIBRARY_PATH
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld = ":".join(all_dirs)
    os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{current_ld}" if current_ld else new_ld

    # Pre-load critical libs via ctypes (RTLD_GLOBAL primes the linker)
    PRELOAD = [
        "libcuda.so.1",
        "libcudart.so.12",          # cuda runtime — must come before cublas
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcurand.so.10",
        "libcufft.so.11",
    ]
    
    loaded, failed = [], []
    for lib_name in PRELOAD:
        found = None
        for d in all_dirs:
            candidate = os.path.join(d, lib_name)
            if os.path.isfile(candidate):
                found = candidate
                break
        if found is None:
            found = lib_name

        try:
            ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
            loaded.append(lib_name)
        except OSError:
            failed.append(lib_name)

    # Print status quietly (using environment variable to suppress if needed)
    if os.environ.get("DEBUG_CUDA_PATCH", "1") == "1":
        print(f"[cuda_patch] Pre-loaded libs : {loaded}")
        if failed:
            print(f"[cuda_patch] Not pre-loaded (ok if GPU is still active): {failed}")

_patch_library_paths()
