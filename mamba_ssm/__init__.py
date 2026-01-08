__version__ = "2.2.4"

import platform
import sys
import warnings

# Check for MacOS Apple Silicon and provide information
if sys.platform == "darwin" and platform.machine() == "arm64":
    print(
        "🍎 Mamba2MacOS: Running on macOS Apple Silicon with MPS acceleration support. "
        "This implementation is optimized for M1/M2/M3/M4 chips."
    )

# Import basic functions that should work with fallbacks
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn

# Import modules conditionally
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.modules.mamba2 import Mamba2

    # Import our optimized Mamba2MacOS module
    from mamba_ssm.modules.mamba2_macos import Mamba2MacOS
    from mamba_ssm.modules.mamba_simple import Mamba

    __all__ = [
        "selective_scan_fn",
        "mamba_inner_fn",
        "Mamba",
        "Mamba2",
        "Mamba2MacOS",
        "MambaLMHeadModel",
    ]
except ImportError as e:
    warnings.warn(
        f"Some modules could not be imported: {str(e)}. "
        "Core functionality is still available with CPU fallbacks."
    )
    __all__ = ["selective_scan_fn", "mamba_inner_fn"]
