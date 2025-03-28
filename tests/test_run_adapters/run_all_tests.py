#!/usr/bin/env python3
"""
Script to run all unit tests using adapters.

This script runs all available test runners for different modules.
"""

import os
import sys
import subprocess

# Add project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def run_script(script_path):
    """Run a test script and return the result."""
    print(f"\n{'='*80}\n")
    print(f"Running: {os.path.basename(script_path)}\n")
    result = subprocess.run(["python", script_path])
    return result.returncode

def run_pytest(test_path):
    """Run pytest on a specific test file or directory."""
    print(f"\n{'='*80}\n")
    print(f"Running pytest on: {test_path}\n")
    result = subprocess.run(["python", "-m", "pytest", test_path, "-v"])
    return result.returncode

def main():
    """Main function to run all tests."""
    # Get the directory paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    test_dir = os.path.abspath(os.path.join(project_root, "tests"))
    unit_test_dir = os.path.abspath(os.path.join(test_dir, "unit_tests"))
    integration_test_dir = os.path.abspath(os.path.join(test_dir, "integration_tests"))
    
    # Define test paths to run with pytest
    test_paths = [
        os.path.join(unit_test_dir, "simple_test.py"),
        os.path.join(unit_test_dir, "test_inference.py"),
        os.path.join(unit_test_dir, "test_extract_json.py"),
        os.path.join(unit_test_dir, "test_base_agent.py"),
        os.path.join(unit_test_dir, "test_professor_agent.py"),
        os.path.join(integration_test_dir, "test_agent_collaboration.py"),
    ]
    
    # Track results
    results = {}
    all_passed = True
    
    # Run each test path
    for path in test_paths:
        if not os.path.exists(path):
            print(f"\nWarning: Test path not found: {path}")
            results[os.path.basename(path)] = False
            all_passed = False
            continue
            
        path_name = os.path.basename(path)
        returncode = run_pytest(path)
        results[path_name] = returncode == 0
        if returncode != 0:
            all_passed = False
    
    # Print summary
    print("\n" + "="*40)
    print("Test Summary:")
    print("="*40)
    for path_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{path_name}: {status}")
    print("="*40)
    
    if all_passed:
        print("\n✅ All tests passed successfully!\n")
        return 0
    else:
        print("\n❌ Some tests failed. See details above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())