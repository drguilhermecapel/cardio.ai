#!/usr/bin/env python3
"""
Script para carregar os labels reais do PTB-XL
Corrige o problema de ter apenas 1 classe (labels dummy)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import ast

def load_ptbxl_labels(ptbxl_path, output_path):
    """
    Carrega os labels reais do dataset PTB-XL
    
    Args:
        ptbxl_path: Caminho raiz do PTB-XL (onde estão os CSVs)
        output_path: Onde salvar o Y.npy corrigido
    """
    print("🔍 Carregando metadados do PTB-XL...")
    
    # Carregar o arquivo principal de metadados
    metadata_file = Path(ptbxl_path) / "ptbxl_database.csv"
    if not metadata_file.exists():
        print(f"❌ Erro: {metadata_file} não encontrado!")
        print("Certifique-se de que você tem o dataset PTB-XL completo")
        return None
    
    # Ler metadados
    df = pd.read_csv(metadata_file)
    print(f"✓ Carregados {len(df)} registros")
    
    # Carregar códigos SCP
    scp_file = Path(ptbxl_path) / "scp_statements.csv"
    if scp_file.exists():
        scp_codes = pd.read_csv(scp_file, index_col=0)
        print(f"✓ Carregados {len(scp_codes)} códigos SCP")
    else:
        print("⚠️  Arquivo scp_statements.csv não encontrado")
        scp_codes = None
    
    # Extrair labels diagnósticos
    print("\n📊 Processando labels...")
    
    # Converter string de dicionário para dicionário real
    df['scp_codes_dict'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
    
    # Opção 1: Multi-label (múltiplas patologias por ECG)
    # Criar matriz binária para cada condição
    all_conditions = set()
    for codes in df['scp_codes_dict']:
        all_conditions.update(codes.keys())
    
    condition_list = sorted(list(all_conditions))
    print(f"✓ Encontradas {len(condition_list)} condições únicas")
    
    # Criar matriz de labels multi-label
    n_samples = len(df)
    n_conditions = len(condition_list)
    Y_multilabel = np.zeros((n_samples, n_conditions), dtype=np.float32)
    
    for i, codes in enumerate(df['scp_codes_dict']):
        for condition, confidence in codes.items():
            if condition in condition_list:
                idx = condition_list.index(condition)
                Y_multilabel[i, idx] = confidence / 100.0  # Normalizar confiança
    
    # Opção 2: Single-label (patologia principal)
    # Usar a condição com maior confiança
    Y_single = np.zeros(n_samples, dtype=np.int64)
    
    for i, codes in enumerate(df['scp_codes_dict']):
        if codes:
            # Pegar a condição com maior confiança
            main_condition = max(codes.items(), key=lambda x: x[1])[0]
            if main_condition in condition_list:
                Y_single[i] = condition_list.index(main_condition)
    
    # Estatísticas
    print("\n📈 Estatísticas dos labels:")
    unique_labels = np.unique(Y_single)
    print(f"  - Classes únicas (single-label): {len(unique_labels)}")
    print(f"  - Distribuição:")
    
    label_counts = pd.Series(Y_single).value_counts()
    for idx in label_counts.head(10).index:
        condition = condition_list[idx] if idx < len(condition_list) else "Unknown"
        count = label_counts[idx]
        pct = count / n_samples * 100
        print(f"    {condition}: {count} ({pct:.1f}%)")
    
    # Filtrar apenas registros que existem no npy_lr
    print("\n🔗 Sincronizando com arquivos .npy...")
    npy_path = Path(output_path).parent
    existing_files = []
    
    for i, row in df.iterrows():
        ecg_id = row['ecg_id']
        npy_file = npy_path / f"{ecg_id:05d}.npy"
        if npy_file.exists():
            existing_files.append(i)
    
    print(f"✓ Encontrados {len(existing_files)} arquivos correspondentes")
    
    # Filtrar labels
    Y_single_filtered = Y_single[existing_files]
    Y_multilabel_filtered = Y_multilabel[existing_files]
    
    # Salvar
    output_dir = Path(output_path).parent
    
    # Salvar single-label (para uso imediato)
    np.save(output_dir / "Y.npy", Y_single_filtered)
    print(f"\n✅ Y.npy (single-label) salvo com {len(np.unique(Y_single_filtered))} classes")
    
    # Salvar multi-label (para uso futuro)
    np.save(output_dir / "Y_multilabel.npy", Y_multilabel_filtered)
    print(f"✅ Y_multilabel.npy salvo com {n_conditions} condições")
    
    # Salvar mapeamento de classes
    class_mapping = {i: cond for i, cond in enumerate(condition_list)}
    import json
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print("✅ class_mapping.json salvo")
    
    # Criar arquivo de informações
    info_text = f"""
PTB-XL Labels Processados
========================
Total de amostras: {len(Y_single_filtered)}
Classes únicas: {len(np.unique(Y_single_filtered))}
Condições totais: {len(condition_list)}

Arquivos criados:
- Y.npy: Labels single-label (classe principal)
- Y_multilabel.npy: Labels multi-label (todas as condições)
- class_mapping.json: Mapeamento índice->condição

Top 5 condições mais comuns:
"""
    for idx in label_counts.head(5).index:
        condition = condition_list[idx] if idx < len(condition_list) else "Unknown"
        info_text += f"- {condition}: {label_counts[idx]} casos\n"
    
    with open(output_dir / "labels_info.txt", "w") as f:
        f.write(info_text)
    
    return Y_single_filtered

def main():
    # Caminhos - AJUSTE CONFORME NECESSÁRIO
    ptbxl_root = r"D:\ptb-xl"  # Diretório raiz do PTB-XL
    npy_dir = r"D:\ptb-xl\npy_lr"  # Onde estão os arquivos .npy
    
    print("=" * 60)
    print("🏥 CARREGADOR DE LABELS REAIS PTB-XL")
    print("=" * 60)
    
    # Verificar se o diretório PTB-XL existe
    if not Path(ptbxl_root).exists():
        print(f"❌ Erro: Diretório PTB-XL não encontrado em {ptbxl_root}")
        print("\nVocê precisa do dataset completo PTB-XL com os arquivos CSV")
        print("Download: https://physionet.org/content/ptb-xl/1.0.3/")
        return
    
    # Carregar labels
    try:
        labels = load_ptbxl_labels(ptbxl_root, Path(npy_dir) / "Y.npy")
        
        if labels is not None:
            print("\n" + "=" * 60)
            print("✅ SUCESSO! Labels reais carregados")
            print("=" * 60)
            print("\nPróximos passos:")
            print("1. Execute novamente o auto-fix-and-train.py")
            print("2. O sistema detectará múltiplas classes agora")
            print("3. O treinamento produzirá um modelo útil!")
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
