# Agent Laboratory Test Suite

## Overview
This test suite provides comprehensive testing for the Agent Laboratory project. The tests are organized into the following categories:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing interactions between components

## Test Structure
```
tests/
├── conftest.py                       # Common test fixtures and configuration
├── unit_tests/                       # Tests for individual components
│   ├── simple_test.py                # Basic sanity checks
│   ├── test_base_agent.py            # Tests for agent base classes
│   ├── test_extract_json.py          # Tests for JSON extraction functionality
│   ├── test_inference.py             # Tests for inference system
│   └── test_professor_agent.py       # Tests for professor agent
├── integration_tests/                # Tests for component interactions
│   └── test_agent_collaboration.py   # Tests for agent collaboration
└── test_run_adapters/                # Test runners
    └── run_all_tests.py              # Script to run all tests
```

## Running Tests

### Running all unit tests
```bash
python -m pytest tests/unit_tests/ -v
```

### Running tests with coverage
```bash
python -m pytest --cov=./ tests/unit_tests/ -v
```

### Running specific test categories
```bash
# Run integration tests
python -m pytest tests/integration_tests/ -v
```

### Running a specific test file
```bash
python -m pytest tests/unit_tests/test_token_utils.py -v
```

## Using Test Adapters

The test suite includes adapter files that allow tests to run without modifying the original codebase. These adapters are located in the `test_adapters/` directory and provide compatibility interfaces for tests.

### Available adapters:
```
test_adapters/
├── __init__.py                       # Adapter package initialization
├── inference_adapter.py              # Adapter for inference functionality
├── laboratory_adapter.py             # Adapter for laboratory workflow
└── token_adapter.py                  # Adapter for token utilities
```

To use the adapters in tests, import from the adapter modules instead of directly from the application:

```python
# Instead of: from inference import query_model
from test_adapters.inference_adapter import query_model

# Instead of: from ai_lab_repo import LaboratoryWorkflow
from test_adapters.laboratory_adapter import LaboratoryWorkflow
```

## Test Coverage
The project aims for high test coverage. Current coverage can be viewed by running:
```bash
python -m pytest --cov=./ --cov-report=html tests/
```

This will generate an HTML coverage report in the `htmlcov/` directory.

## Adding New Tests
When adding new functionality to the project, please follow these guidelines:
1. Create unit tests for all new components
2. Ensure tests are isolated and don't rely on external services
3. Use mocks for external dependencies
4. Keep tests fast and deterministic
5. Follow the naming conventions established in the project
6. Use test adapters instead of modifying the original code