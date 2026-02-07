"""
Placeholder tests for diabetes-prediction project.

This file contains basic placeholder tests that will be expanded
as the project grows. For now, it ensures the CI pipeline has
something to test.
"""

import pytest


def test_placeholder() -> None:
    """Placeholder test to verify test infrastructure works."""
    assert True


def test_imports() -> None:
    """Test that core modules can be imported."""
    try:
        import src.processing
        import src.training
        import src.schemas
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_processing_pipeline_exists() -> None:
    """Test that preprocessing pipeline can be created."""
    from src.processing import create_preprocessing_pipeline

    pipeline = create_preprocessing_pipeline()
    assert pipeline is not None
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "transform")


@pytest.mark.skipif(
    True, reason="Models need to be trained first - run 'make train'"
)
def test_model_loading():
    """Test that trained models can be loaded."""
    import joblib
    import os

    if not os.path.exists("models/best_model.pkl"):
        pytest.skip("No trained model found")

    model = joblib.load("models/best_model.pkl")
    assert model is not None
