"""Lightweight runtime hardware report."""

from __future__ import annotations

import os
import platform
from typing import Any

from app.models.ncf import require_tensorflow


def hardware_info() -> dict[str, Any]:
    tensorflow = require_tensorflow()
    gpus = tensorflow.config.list_physical_devices("GPU")
    available_memory: int | None = None
    try:
        import psutil

        available_memory = int(psutil.virtual_memory().available)
    except ModuleNotFoundError:
        if hasattr(os, "sysconf"):
            try:
                available_memory = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError):
                pass
    return {
        "device": "GPU" if gpus else "CPU",
        "gpu_count": len(gpus),
        "available_ram_bytes": available_memory,
        "tensorflow_version": tensorflow.__version__,
        "python_version": platform.python_version(),
    }
