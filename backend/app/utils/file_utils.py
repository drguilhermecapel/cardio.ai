"""File utilities for handling uploads"""

import os
import shutil
from pathlib import Path
from typing import Optional
import hashlib


def save_upload_file(file_content: bytes, filename: str, upload_dir: str = "uploads") -> str:
    """Save uploaded file and return path"""
    # Create upload directory
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{file_hash}{ext}"
    
    # Save file
    file_path = upload_path / unique_filename
    with open(file_path, 'wb') as f:
        f.write(file_content)
    
    return str(file_path)


def delete_file(file_path: str) -> bool:
    """Delete file if exists"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False



def save_upload_file_tmp(upload_file, upload_dir: str = "uploads") -> Path:
    """
    Salva um arquivo de upload temporariamente e retorna o caminho.
    
    Args:
        upload_file: Arquivo de upload do FastAPI
        upload_dir: Diretório para salvar o arquivo
        
    Returns:
        Path do arquivo salvo
    """
    # Criar diretório de upload
    upload_path = Path(upload_dir)
    upload_path.mkdir(exist_ok=True)
    
    # Gerar nome único para o arquivo
    file_hash = hashlib.md5(upload_file.filename.encode()).hexdigest()[:8]
    name, ext = os.path.splitext(upload_file.filename)
    unique_filename = f"{name}_{file_hash}{ext}"
    
    # Salvar arquivo
    file_path = upload_path / unique_filename
    with open(file_path, 'wb') as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path

