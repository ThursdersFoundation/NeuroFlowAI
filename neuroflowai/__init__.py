"""
NeuroFlowAI
===========

A lightweight and modular AI pipeline framework for fast machine learning workflows.

Author: Thursders Foundation
License: MIT
"""

# Version
__version__ = "0.1.0"

# Core imports (biar user bisa langsung import dari root)
from .core import NeuroFlow

# Optional utilities (kalau ada)
try:
    from .utils import *
except ImportError:
    pass

# Public API
__all__ = [
    "NeuroFlow",
]

# Metadata
__author__ = "Thursders Foundation"
__email__ = "thursdersfoundation@gmail.com"

# Simple startup check (optional, bisa kamu hapus kalau mau clean)
def _check_environment():
    try:
        import numpy
        import sklearn
    except ImportError as e:
        raise ImportError(
            "Missing required dependencies. Install with: pip install neuroflowai[full]"
        ) from e

_check_environment()
