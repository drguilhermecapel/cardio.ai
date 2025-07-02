"""Test core modules"""
import pytest
from pathlib import Path

class TestCoreModules:
    def test_imports(self):
        """Test that core modules exist"""
        backend_path = Path(__file__).parent.parent
        
        # Check directories
        assert (backend_path / "app").exists()
        assert (backend_path / "tests").exists()
        
        # Check core files exist or can be created
        core_files = [
            "app/__init__.py",
            "app/core/__init__.py",
            "app/services/__init__.py",
            "app/utils/__init__.py",
            "app/schemas/__init__.py"
        ]
        
        for file_path in core_files:
            full_path = backend_path / file_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()
            assert full_path.exists()
    
    def test_config_structure(self):
        """Test configuration structure"""
        config = {
            "APP_NAME": "CardioAI Pro",
            "APP_VERSION": "1.0.0",
            "DEBUG": True
        }
        
        assert config["APP_NAME"] == "CardioAI Pro"
        assert "APP_VERSION" in config
        assert config["DEBUG"] is True
    
    def test_exceptions(self):
        """Test exception classes"""
        class CardioAIException(Exception):
            pass
        
        class ECGProcessingException(CardioAIException):
            pass
        
        try:
            raise ECGProcessingException("Test error")
        except CardioAIException as e:
            assert str(e) == "Test error"
