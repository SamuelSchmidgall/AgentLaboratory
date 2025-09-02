"""
Laboratory workflow adapter for tests.

This module provides adapter functions that expose the laboratory workflow functionality
that tests expect, without modifying the original code.
"""

import sys
import os

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import classes from ai_lab_repo.py
from ai_lab_repo import LaboratoryWorkflow

# Create any additional functionality or aliases needed by tests
__all__ = ['LaboratoryWorkflow']