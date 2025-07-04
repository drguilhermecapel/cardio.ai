#!/usr/bin/env python3
"""
Script para corrigir o formato do modelo e mapeamento de classes
"""

import os
import json
import numpy as np
import h5py
from pathlib import Path

# Diretório de modelos
models_dir = Path("models")

# 1. Corrigir o arquivo model_architecture.json
print("Corrigindo arquitetura do modelo...")
model_arch_path = models_dir / "model_architecture.json"

# Carregar a arquitetura atual
with open(model_arch_path, 'r') as f:
    arch = json.load(f)

# Verificar se a arquitetura já está correta
input_shape = arch["config"]["layers"][0]["config"]["batch_input_shape"]
if input_shape[1] == 1000 and input_shape[2] == 12:
    # Formato incorreto: [batch, 1000, 12]
    # Corrigir para: [batch, 12, 1000]
    arch["config"]["layers"][0]["config"]["batch_input_shape"] = [None, 12, 1000]
    print(f"Formato de entrada corrigido: {input_shape} -> [None, 12, 1000]")

    # Salvar arquitetura corrigida
    with open(model_arch_path, 'w') as f:
        json.dump(arch, f, indent=2)
else:
    print(f"Formato de entrada já está correto: {input_shape}")

# 2. Criar mapeamento correto de classes PTB-XL para diagnósticos clínicos
print("Criando mapeamento de classes...")
ptbxl_classes_path = models_dir / "ptbxl_classes.json"

# Carregar classes PTB-XL
with open(ptbxl_classes_path, 'r') as f:
    ptbxl_classes = json.load(f)

# Criar mapeamento para diagnósticos clínicos
clinical_mapping = {
    # Mapeamento de classes PTB-XL para diagnósticos clínicos simplificados
    "0": "Normal",  # NORM - Normal ECG
    "7": "Fibrilação Atrial",  # AFIB - Atrial Fibrillation
    "49": "Bradicardia",  # SBRAD - Sinus Bradycardia
    "50": "Taquicardia",  # STACH - Sinus Tachycardia
    "6": "Arritmia Ventricular",  # PVC - Premature Ventricular Contraction
    "63": "Bloqueio AV",  # HEART_BLOCK - Heart Block
    "32": "Isquemia",  # ISCAL - Ischemia in Anterolateral
    "1": "Infarto do Miocárdio",  # MI - Myocardial Infarction
    "12": "Hipertrofia Ventricular",  # LVH - Left Ventricular Hypertrophy
    "70": "Anormalidade Inespecífica"  # OTHER - Other Abnormality
}

# Criar mapeamento completo
class_mapping = {}
for idx, label in ptbxl_classes["classes"].items():
    # Verificar se a classe está no mapeamento clínico
    if idx in clinical_mapping:
        # Usar o diagnóstico clínico
        class_mapping[idx] = clinical_mapping[idx]
    else:
        # Usar o nome original da classe
        class_name = label.split(" - ")[1] if " - " in label else label
        class_mapping[idx] = class_name

# Salvar mapeamento
mapping_path = models_dir / "clinical_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump({
        "class_mapping": class_mapping,
        "clinical_mapping": clinical_mapping,
        "severity": {
            "normal": [0],
            "mild": [49, 50, 51],
            "moderate": [7, 6, 12, 63],
            "severe": [1, 32, 33, 34, 35, 36, 37],
            "critical": [14, 15, 25]
        },
        "clinical_priority": {
            "routine": [0, 49, 50, 51],
            "urgent": [7, 6, 12, 63],
            "immediate": [1, 32, 33, 34, 35, 36, 37, 14, 15, 25]
        }
    }, f, indent=2)

print(f"Mapeamento de classes salvo em {mapping_path}")

# 3. Atualizar a função de pré-processamento
preprocess_path = models_dir / "preprocess_functions.py"

# Verificar se a função já está correta
with open(preprocess_path, 'r') as f:
    preprocess_code = f.read()

if "sig = sig.T" in preprocess_code:
    # Corrigir a função de pré-processamento
    new_preprocess_code = preprocess_code.replace(
        "    # Transpor para formato (amostras, derivações) - formato PTB-XL\n    sig = sig.T",
        "    # Manter formato (derivações, amostras) - formato PTB-XL\n    # Não transpor aqui, pois o modelo espera (batch, derivações, amostras)"
    )
    
    with open(preprocess_path, 'w') as f:
        f.write(new_preprocess_code)
    
    print("Função de pré-processamento corrigida")
else:
    print("Função de pré-processamento já está correta")

print("Correções concluídas com sucesso!")