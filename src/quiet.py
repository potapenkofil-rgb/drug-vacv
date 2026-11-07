from __future__ import annotations

import os

CORES = int(os.environ.get("CORES", 3))
IDLE = 0x00000040

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = str(CORES)


def step_aside(cores=None):
    import ctypes
    from ctypes import wintypes

    cores = CORES if cores is None else cores
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    lowered = bool(kernel.SetPriorityClass(handle, IDLE))
    pinned = bool(kernel.SetProcessAffinityMask(handle, (1 << cores) - 1))
    return lowered and pinned
