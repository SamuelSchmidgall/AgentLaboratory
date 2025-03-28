import pytest
import os
import sys
import json

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from agents module
from agents import extract_json_between_markers

class TestExtractJSON:
    """Tests for the JSON extraction functionality."""
    
    def test_extract_valid_json_with_markers(self):
        """Test extracting valid JSON from text with markers."""
        test_input = """
        Some text before the JSON.
        ```json
        {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3]
        }
        ```
        Some text after the JSON.
        """
        
        expected_output = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3]
        }
        
        result = extract_json_between_markers(test_input)
        
        assert result == expected_output
    
    def test_extract_valid_json_without_markers(self):
        """Test extracting valid JSON from text without markers."""
        test_input = """
        Some text before the JSON.
        {
            "key1": "value1",
            "key2": 42
        }
        Some text after the JSON.
        """
        
        expected_output = {
            "key1": "value1",
            "key2": 42
        }
        
        result = extract_json_between_markers(test_input)
        
        assert result == expected_output
    
    def test_extract_invalid_json(self):
        """Test extracting invalid JSON."""
        test_input = """
        Some text with invalid JSON.
        ```json
        {
            "key1": "value1",
            "key2": 42,
            this is not valid
        }
        ```
        """
        
        # Should return None for invalid JSON
        result = extract_json_between_markers(test_input)
        
        assert result is None
    
    def test_extract_no_json(self):
        """Test extracting from text with no JSON."""
        test_input = "This text contains no JSON."
        
        result = extract_json_between_markers(test_input)
        
        assert result is None