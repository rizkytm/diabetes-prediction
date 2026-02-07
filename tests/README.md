# Tests for Diabetes Prediction Project

## Test Structure

This directory contains tests for the diabetes prediction MLOps project.

### Current Tests

- `test_placeholder.py` - Basic placeholder tests to verify CI/CD infrastructure
  - Test imports of core modules
  - Test preprocessing pipeline creation
  - Model loading test (skipped until models are trained)

### Running Tests

```bash
# Run all tests
make test

# Or directly with pytest
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term
```

### Test Categories to Implement

1. **Unit Tests** (`test_*.py`)
   - Data preprocessing tests
   - Model training tests
   - Feature engineering tests
   - Schema validation tests

2. **Integration Tests** (`test_integration_*.py`)
   - End-to-end pipeline tests
   - API endpoint tests
   - Docker container tests

3. **Performance Tests** (`test_performance_*.py`)
   - Model accuracy tests
   - Inference time tests
   - Memory usage tests

### TODO

- [ ] Add unit tests for `ZeroToNanTransformer`
- [ ] Add unit tests for preprocessing pipeline
- [ ] Add integration tests for FastAPI endpoints
- [ ] Add model evaluation tests
- [ ] Add data validation tests
- [ ] Add edge case tests (missing values, outliers)
- [ ] Add API request/response validation tests
- [ ] Add Docker container health checks
