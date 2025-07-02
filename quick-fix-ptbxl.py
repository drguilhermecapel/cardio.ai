#!/usr/bin/env python3
"""
Script de correção rápida para preparar dados PTB-XL para ensemble_ecg_training.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def quick_fix_ptbxl(npy_dir):
    """
    Consolida arquivos .npy individuais em X.npy e Y.npy
    Versão simplificada sem dependências externas
    """
    npy_path = Path(npy_dir)
    
    # Listar arquivos
    npy_files = sorted([f for f in npy_path.glob("*.npy") if f.name[0].isdigit()])
    print(f"Encontrados {len(npy_files)} arquivos ECG")
    
    if len(npy_files) == 0:
        print("❌ Nenhum arquivo .npy encontrado!")
        return
    
    # Carregar primeiro arquivo para verificar dimensões
    first = np.load(npy_files[0])
    print(f"Shape do primeiro arquivo: {first.shape}")
    print(f"Tipo de dados: {first.dtype}")
    
    # Determinar formato
    if first.ndim == 1:
        # Assumir 12 derivações
        total_points = len(first)
        if total_points % 12 == 0:
            n_points = total_points // 12
            n_leads = 12
            print(f"Detectado: sinal unidimensional com {n_leads} derivações, {n_points} pontos cada")
        else:
            # Tentar 8 derivações
            if total_points % 8 == 0:
                n_points = total_points // 8
                n_leads = 8
                print(f"Detectado: sinal unidimensional com {n_leads} derivações, {n_points} pontos cada")
            else:
                print(f"❌ Erro: Tamanho {total_points} não é divisível por 12 ou 8")
                return
    elif first.ndim == 2:
        n_leads, n_points = first.shape
        print(f"Detectado: {n_leads} derivações, {n_points} pontos cada")
    else:
        print(f"❌ Formato não suportado: {first.shape}")
        return
    
    # Criar arrays para todos os dados
    n_samples = len(npy_files)
    X = np.zeros((n_samples, n_leads, n_points), dtype=np.float32)
    
    # Por enquanto, criar labels dummy (todos normais = 0)
    # Em produção, você deve usar os labels reais do PTB-XL
    Y = np.zeros(n_samples, dtype=np.int64)
    
    print("\nCarregando e processando arquivos...")
    errors = 0
    
    for i, f in enumerate(tqdm(npy_files)):
        try:
            data = np.load(f)
            
            # Reformatar se necessário
            if data.ndim == 1:
                data = data.reshape(n_leads, n_points)
            
            # Verificar tamanho
            if data.shape == (n_leads, n_points):
                X[i] = data
            else:
                # Tentar ajustar
                if data.shape[0] == n_leads:
                    if data.shape[1] > n_points:
                        X[i] = data[:, :n_points]
                    else:
                        X[i, :, :data.shape[1]] = data
                else:
                    errors += 1
                    print(f"\n⚠️  Shape incorreto em {f.name}: {data.shape}")
        except Exception as e:
            errors += 1
            print(f"\n❌ Erro em {f.name}: {e}")
    
    if errors > 0:
        print(f"\n⚠️  {errors} arquivos com erro foram ignorados")
    
    # Salvar resultados
    output_dir = npy_path
    print(f"\nSalvando X.npy: {X.shape}")
    np.save(output_dir / "X.npy", X)
    
    print(f"Salvando Y.npy: {Y.shape}")
    np.save(output_dir / "Y.npy", Y)
    
    # Criar arquivo de informações
    info_text = f"""Dataset PTB-XL Processado
========================
Amostras: {n_samples}
Derivações: {n_leads}
Pontos por derivação: {n_points}
Taxa de amostragem: 100 Hz (assumido)
Shape X: {X.shape}
Shape Y: {Y.shape}

ATENÇÃO: Os labels (Y) são dummy! 
Para resultados reais, você precisa usar os labels verdadeiros do PTB-XL.
"""
    
    with open(output_dir / "README.txt", "w") as f:
        f.write(info_text)
    
    print("\n✅ Concluído!")
    print(f"📁 Arquivos criados em: {output_dir}")
    print("   - X.npy")
    print("   - Y.npy")
    print("   - README.txt")
    print("\n⚠️  IMPORTANTE: Os labels são dummy (todos 0)!")
    print("   Para treino real, você precisa dos labels verdadeiros do PTB-XL")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python quick_fix_ptbxl.py <diretório_com_npys>")
        print("Exemplo: python quick_fix_ptbxl.py D:/ptb-xl/npy_lr")
        sys.exit(1)
    
    quick_fix_ptbxl(sys.argv[1])
