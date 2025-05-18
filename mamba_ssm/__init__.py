__version__ = "2.2.4"

import platform
import sys
import warnings

# Check for MacOS Apple Silicon and provide a warning
if sys.platform == "darwin" and platform.machine() == "arm64":
    warnings.warn(
        "Running on macOS Apple Silicon. Some functionality may be limited as CUDA extensions "
        "and Triton are not fully supported on this platform. Using fallback implementations where available."
    )

# Import basic functions that should work with fallbacks
from mamba_ssm.ops.selective_scan_interface import (mamba_inner_fn,
                                                    selective_scan_fn)

# Import modules conditionally
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.modules.mamba_simple import Mamba
    __all__ = ["selective_scan_fn", "mamba_inner_fn", "Mamba", "Mamba2", "MambaLMHeadModel"]
except ImportError as e:
    warnings.warn(
        f"Some modules could not be imported: {str(e)}. "
        "This is expected on platforms without CUDA/Triton support. "
        "Core functionality is still available."
    )
    __all__ = ["selective_scan_fn", "mamba_inner_fn"]
