#!/usr/bin/env python3
"""
Correção específica para ValidationException aceitar múltiplos parâmetros
"""

import os
from pathlib import Path

BACKEND_DIR = Path.cwd() / "backend" if (Path.cwd() / "backend").exists() else Path.cwd()

def fix_validation_exception():
    """Corrige ValidationException para aceitar parâmetros flexíveis."""
    
    exceptions_file = BACKEND_DIR / "app" / "core" / "exceptions.py"
    
    if not exceptions_file.exists():
        print("[ERRO] Arquivo exceptions.py não encontrado!")
        return False
    
    print("Corrigindo ValidationException...")
    
    with open(exceptions_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Novo código para ValidationException
    validation_exception_code = '''class ValidationException(CardioAIException):
    """Validation exception with flexible parameters."""

    def __init__(
        self,
        message: str = "Validation error",
        validation_errors: list[dict] = None,
        errors: list[dict] = None,  # Alias para validation_errors
        field: str = None,
        details: dict = None,
        **kwargs
    ) -> None:
        """Initialize validation exception with flexible parameters.
        
        Args:
            message: Error message
            validation_errors: List of validation errors
            errors: Alias for validation_errors
            field: Field that failed validation
            details: Additional error details
            **kwargs: Additional arguments
        """
        # Determinar detalhes
        error_details = details or {}
        
        # Usar validation_errors ou errors
        self.validation_errors = validation_errors or errors or []
        self.errors = self.validation_errors  # Alias
        
        if field:
            error_details['field'] = field
            
        if self.validation_errors:
            error_details['validation_errors'] = self.validation_errors
            
        # Adicionar kwargs aos detalhes
        for key, value in kwargs.items():
            if key not in ['validation_errors', 'errors', 'field', 'details']:
                error_details[key] = value
                setattr(self, key, value)
        
        super().__init__(message, "VALIDATION_ERROR", 422, error_details)
        self.field = field'''
    
    # Substituir a definição existente de ValidationException
    import re
    pattern = r'class ValidationException\(CardioAIException\):.*?(?=\n\nclass|\n\n#|\Z)'
    
    if "class ValidationException" in content:
        content = re.sub(pattern, validation_exception_code, content, flags=re.DOTALL)
    else:
        # Se não existir, adicionar após CardioAIException
        content = content.replace(
            "class CardioAIException(Exception):",
            f"class CardioAIException(Exception):"
        )
        # Adicionar no final do arquivo
        content += "\n\n" + validation_exception_code
    
    with open(exceptions_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] ValidationException corrigida com sucesso!")
    return True


if __name__ == "__main__":
    os.chdir(BACKEND_DIR)
    fix_validation_exception()
