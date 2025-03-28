"""
Token utilities adapter for tests.

This module provides adapter functions that expose the token utility functionality
that tests expect, without modifying the original code.
"""

import sys
import os
import tiktoken

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the inference module
from inference import TOKENS_IN, TOKENS_OUT

def get_token_count(text, model="cl100k_base"):
    """Get the number of tokens in a string using tiktoken."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

# Create any additional functionality or aliases needed by tests
__all__ = ['get_token_count', 'TOKENS_IN', 'TOKENS_OUT']