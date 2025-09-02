import pytest
import sys
import os

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def test_simple():
    """A simple test that always passes."""
    assert 1 + 1 == 2

def test_import_main_modules():
    """Test that main modules can be imported."""
    try:
        import agents
        import inference
        import tools
        import utils
        import mlesolver
        import ai_lab_repo
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import module: {e}")