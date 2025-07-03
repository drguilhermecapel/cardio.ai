"""Mock pyedflib module for tests to avoid import errors."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import sys
from unittest.mock import MagicMock

# Create mock for pyedflib
mock_pyedflib = MagicMock()
mock_pyedflib.EdfReader = MagicMock
mock_pyedflib.EdfWriter = MagicMock
mock_pyedflib.highlevel = MagicMock()
mock_pyedflib.highlevel.read_edf = MagicMock(return_value=([], None, None))

# Insert mock into sys.modules before any imports
sys.modules["pyedflib"] = mock_pyedflib
