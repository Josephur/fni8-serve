# SPDX-License-Identifier: MIT
"""fni8-serve — W8A8 dp4a inference server for Volta / CMP 100-210, over `fni8`."""
from .config import ServeConfig
from .loader import load_fni8_checkpoint

__all__ = ["ServeConfig", "load_fni8_checkpoint"]
__version__ = "0.0.1"
