__version__ = "1.0.0"

import platform
import sys

if sys.platform == "darwin" and platform.machine() == "arm64":
    print("Mamba SSM macOS: Running on Apple Silicon with MPS acceleration")

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
from mamba_ssm.utils.macos import (
    create_tokenizer,
    generate_text_with_model,
    get_device,
    load_and_prepare_model,
)

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
    "Mamba",
    "Mamba2",
    "MambaLMHeadModel",
    "get_device",
    "create_tokenizer",
    "generate_text_with_model",
    "load_and_prepare_model",
]
