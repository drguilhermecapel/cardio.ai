#!/usr/bin/env python3
"""
Run Fix and Test - Complete Solution
Executes all fixes and runs tests with coverage analysis
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time
import json


class CardioAICompleteFixRunner:
    """Complete fix runner for CardioAI Pro"""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.venv_path = self.backend_path / "venv"
        self.python_exe = sys.executable
        
        # Colors for terminal output
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.BLUE = '\033[94m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'
        
    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{self.BLUE}{self.BOLD}{'='*60}{self.END}")
        print(f"{self.BLUE}{self.BOLD}{title.center(60)}{self.END}")
        print(f"{self.BLUE}{self.BOLD}{'='*60}{self.END}\n")
        
    def print_success(self, message: str):
        """Print success message"""
        print(f"{self.GREEN}✅ {message}{self.END}")
        
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{self.YELLOW}⚠️  {message}{self.END}")
        
    def print_error(self, message: str):
        """Print error message"""
        print(f"{self.RED}❌ {message}{self.END}")
        
    def print_info(self, message: str):
        """Print info message"""
        print(f"{self.BLUE}ℹ️  {message}{self.END}")
    
    def run(self):
        """Run complete fix and test process"""
        self.print_header("CardioAI Pro - Correção Completa e Testes")
        
        try:
            # Step 1: Clean environment
            self.clean_environment()
            
            # Step 2: Install dependencies
            self.install_dependencies()
            
            # Step 3: Apply all fixes
            self.apply_fixes()
            
            # Step 4: Create missing files
            self.create_missing_files()
            
            # Step 5: Initialize database
            self.initialize_database()
            
            # Step 6: Run tests with coverage
            coverage_percentage = self.run_tests_with_coverage()
            
            # Step 7: Generate report
            self.generate_report(coverage_percentage)
            
        except Exception as e:
            self.print_error(f"Erro fatal: {str(e)}")
            sys.exit(1)
    
    def clean_environment(self):
        """Clean Python cache and test artifacts"""
        self.print_header("Limpeza do Ambiente")
        
        # Remove __pycache__ directories
        pycache_count = 0
        for pycache in self.backend_path.rglob("__pycache__"):
            shutil.rmtree(pycache, ignore_errors=True)
            pycache_count += 1
        
        # Remove .pyc files
        pyc_count = 0
        for pyc in self.backend_path.rglob("*.pyc"):
            pyc.unlink(missing_ok=True)
            pyc_count += 1
        
        # Remove coverage files
        coverage_files = [".coverage", "htmlcov"]
        for cf in coverage_files:
            cf_path = self.backend_path / cf
            if cf_path.exists():
                if cf_path.is_file():
                    cf_path.unlink()
                else:
                    shutil.rmtree(cf_path, ignore_errors=True)
        
        self.print_success(f"Removidos {pycache_count} diretórios __pycache__")
        self.print_success(f"Removidos {pyc_count} arquivos .pyc")
        self.print_success("Arquivos de cobertura limpos")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        self.print_header("Instalação de Dependências")
        
        requirements = [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "sqlalchemy>=2.0.0",
            "asyncpg>=0.28.0",
            "pydantic>=2.0.0",
            "pydantic-settings>=2.0.0",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "python-multipart>=0.0.6",
            "aiofiles>=23.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "psutil>=5.9.0",
            "python-dotenv>=1.0.0",
            "alembic>=1.11.0",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0"
