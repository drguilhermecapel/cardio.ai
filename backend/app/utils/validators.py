"""
Validadores para o sistema CardioAI
"""
import re
from typing import Optional, Union, Dict, Any
from datetime import datetime


def validate_email(email: str) -> bool:
    """
    Valida formato de email.
    
    Args:
        email: Email a ser validado
        
    Returns:
        bool: True se o email for válido, False caso contrário
    """
    if not email or not isinstance(email, str):
        return False
    
    # Padrão regex para validação de email
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))


def validate_cpf(cpf: str) -> bool:
    """
    Valida CPF brasileiro.
    
    Args:
        cpf: CPF a ser validado (com ou sem formatação)
        
    Returns:
        bool: True se o CPF for válido, False caso contrário
    """
    if not cpf or not isinstance(cpf, str):
        return False
    
    # Remove formatação
    cpf = re.sub(r'[^0-9]', '', cpf)
    
    # Verifica se tem 11 dígitos
    if len(cpf) != 11:
        return False
    
    # Verifica se todos os dígitos são iguais
    if cpf == cpf[0] * 11:
        return False
    
    # Calcula primeiro dígito verificador
    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
    resto = soma % 11
    digito1 = 0 if resto < 2 else 11 - resto
    
    # Calcula segundo dígito verificador
    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
    resto = soma % 11
    digito2 = 0 if resto < 2 else 11 - resto
    
    # Verifica se os dígitos calculados conferem
    return cpf[-2:] == f"{digito1}{digito2}"


def validate_phone(phone: str) -> bool:
    """
    Valida número de telefone brasileiro.
    
    Args:
        phone: Telefone a ser validado
        
    Returns:
        bool: True se o telefone for válido, False caso contrário
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Remove formatação
    phone = re.sub(r'[^0-9]', '', phone)
    
    # Verifica se tem 10 ou 11 dígitos (com DDD)
    if len(phone) not in [10, 11]:
        return False
    
    # Verifica se o DDD é válido (11-99)
    ddd = int(phone[:2])
    if ddd < 11 or ddd > 99:
        return False
    
    return True


def validate_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
    """
    Valida formato de data.
    
    Args:
        date_str: String de data
        format: Formato esperado
        
    Returns:
        bool: True se a data for válida
    """
    try:
        datetime.strptime(date_str, format)
        return True
    except (ValueError, TypeError):
        return False


def validate_ecg_file(file_path: str) -> bool:
    """
    Valida arquivo de ECG.
    
    Args:
        file_path: Caminho do arquivo
        
    Returns:
        bool: True se o arquivo for válido
    """
    if not file_path:
        return False
    
    # Extensões válidas para ECG
    valid_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.dicom', '.edf', '.xml'}
    
    # Verifica extensão
    import os
    _, ext = os.path.splitext(file_path.lower())
    return ext in valid_extensions


def validate_patient_data(data: Dict[str, Any]) -> bool:
    """
    Valida dados do paciente.
    
    Args:
        data: Dicionário com dados do paciente
        
    Returns:
        bool: True se os dados forem válidos
    """
    if not data or not isinstance(data, dict):
        return False
    
    # Campos obrigatórios
    required_fields = ['name', 'birth_date']
    
    # Verifica campos obrigatórios
    for field in required_fields:
        if field not in data or not data[field]:
            return False
    
    # Valida nome (mínimo 2 caracteres)
    if len(data['name']) < 2:
        return False
    
    # Valida data de nascimento
    if not validate_date(str(data['birth_date'])):
        return False
    
    # Valida CPF se presente
    if 'cpf' in data and data['cpf']:
        if not validate_cpf(data['cpf']):
            return False
    
    # Valida email se presente
    if 'email' in data and data['email']:
        if not validate_email(data['email']):
            return False
    
    return True


def validate_ecg_signal(signal: Any) -> bool:
    """
    Valida sinal de ECG.
    
    Args:
        signal: Dados do sinal de ECG
        
    Returns:
        bool: True se o sinal for válido
    """
    if signal is None:
        return False
    
    # Se for numpy array
    try:
        import numpy as np
        if isinstance(signal, np.ndarray):
            # Verifica se tem dados
            if signal.size == 0:
                return False
            # Verifica se não tem NaN
            if np.isnan(signal).any():
                return False
            return True
    except ImportError:
        pass
    
    # Se for lista
    if isinstance(signal, list):
        return len(signal) > 0
    
    return False


def validate_phone_number(phone: str) -> bool:
    """Alias para validate_phone para compatibilidade."""
    return validate_phone(phone)


def validate_medical_record_number(mrn: str) -> bool:
    """
    Valida número de prontuário médico.
    
    Args:
        mrn: Número do prontuário
        
    Returns:
        bool: True se válido
    """
    if not mrn or not isinstance(mrn, str):
        return False
    
    # Remove espaços e caracteres especiais
    mrn_clean = re.sub(r'[^A-Za-z0-9]', '', mrn)
    
    # Deve ter pelo menos 4 caracteres
    return len(mrn_clean) >= 4


def validate_heart_rate(rate: Union[int, float]) -> bool:
    """
    Valida frequência cardíaca.
    
    Args:
        rate: Frequência cardíaca em bpm
        
    Returns:
        bool: True se a frequência for válida
    """
    try:
        rate = float(rate)
        # Frequência normal: 30-250 bpm
        return 30 <= rate <= 250
    except (ValueError, TypeError):
        return False


def validate_blood_pressure(systolic: Union[int, float], diastolic: Union[int, float]) -> bool:
    """
    Valida pressão arterial.
    
    Args:
        systolic: Pressão sistólica
        diastolic: Pressão diastólica
        
    Returns:
        bool: True se os valores forem válidos
    """
    try:
        sys = float(systolic)
        dia = float(diastolic)
        
        # Validações básicas
        if sys <= dia:  # Sistólica deve ser maior que diastólica
            return False
        if sys < 50 or sys > 300:  # Limites razoáveis
            return False
        if dia < 30 or dia > 200:  # Limites razoáveis
            return False
            
        return True
    except (ValueError, TypeError):
        return False
