"""
Mamba SSM for macOS Apple Silicon

A high-performance implementation of Mamba (State Space Models) optimized for macOS with MPS acceleration.
Supports both Mamba1 and Mamba2 architectures with fallback implementations.
"""

__version__ = "1.0.0"
__author__ = "Mamba SSM macOS Team"

from .models import load_and_prepare_model
from .utils import create_tokenizer, generate_text_with_model, get_device

__all__ = [
    "get_device",
    "create_tokenizer",
    "generate_text_with_model",
    "load_and_prepare_model",
]
