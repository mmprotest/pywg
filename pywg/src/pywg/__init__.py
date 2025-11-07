"""Top-level package for pywg."""

from .api import DeviceInfo, get_device_info, is_webgpu_available, kernel, map, matmul, reduce, scan
from .version import __version__

__all__ = [
    "DeviceInfo",
    "__version__",
    "get_device_info",
    "is_webgpu_available",
    "kernel",
    "map",
    "matmul",
    "reduce",
    "scan",
]
