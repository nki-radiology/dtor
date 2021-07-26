"""
pytest tests for the Trainer class.
"""

import pytest
import os
import sys
from unittest.mock import patch


# Fixtures
@pytest.fixture
def mock_init(monkeypatch):
    monkeypatch.setenv("DTORROOT", os.getcwd())


def test_trainer_instantiate(mock_init):
    testargs = ["prog", "--load_json", "config/mnist.json"]
    with patch.object(sys, 'argv', testargs):
        from dtor.trainer import Trainer
        t = Trainer()
        assert t is not None
