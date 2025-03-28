"""
Inference adapter for tests.

This module provides adapter functions that expose the inference functionality
that tests expect, without modifying the original code.
"""

import sys
import os

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the actual implementation
from inference import query_model, curr_cost_est

# Create any additional functionality or aliases needed by tests
__all__ = ['query_model', 'curr_cost_est']