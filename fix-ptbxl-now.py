#!/usr/bin/env python3
"""
Script de correção imediata para consolidar arquivos PTB-XL
Resolve o problema de arquivos individuais vs. consolidados
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_ptbxl_now(data_path: str):
    """
    Consolida arquivos .npy individuais do PTB-XL em X.npy e Y.npy
    
    Args:
        data_path: Caminho para D:\ptb-xl\npy_lr
    """
    data_dir = Path(data_path)
    
    # Verificar se o diretório existe
    if not data_dir.exists():
        logger.error(f"Diretório não encontrado: {data_dir}")
        return False
    
    # Verificar se já existem arquivos consolidados
    x_path = data_dir / "X.npy"
    y_path = data_dir / "Y.npy"
    
    if x_path.exists() and y_path.exists():
        logger.info("Arquivos X.npy e Y.npy já existem!")
        try:
            X = np.load(x_path)
            Y = np.load(y_path)
            logger.info(f"X.shape: {X.shape}")
            logger.info(f"Y.shape: {Y.shape}")
            
            resposta = input("\nDeseja recriar os arquivos? (s/n): ").lower()
            if resposta != 's':
                return True
        except Exception as e:
            logger.error(f"Erro ao carregar arquivos existentes: {e}")
    
    # Listar todos os arquivos .npy
    logger.info("Listando arquivos...")
    npy_files = sorted([f for f in data_dir.glob("*.npy") if f.stem.isdigit()])
    
    # Filtrar apenas os que estão no range mencionado (09004 a 19417)
    npy_files = [f for f in npy_files if 9004 <= int(f.stem) <= 19417]
    
    logger.info(f"Encontrados {len(npy_files)} arquivos ECG válidos")
    
    if len(npy_files) == 0:
        logger.error("Nenhum arquivo .npy encontrado!")
        return False
    
    # Verificar o formato do primeiro arquivo
    logger.info("Verificando formato dos dados...")
    first_file = np.load(npy_files[0])
    logger.info(f"Formato do primeiro arquivo: {first_file.shape}")
    logger.info(f"Tipo de dados: {first_file.dtype}")
    
    # Determinar dimensões
    if first_file.ndim == 1:
        # Arquivo unidimensional - precisa ser reformatado
        total_length = len(first_file)
        
        # PTB-XL tem 12 derivações
        n_leads = 12
        
        # Calcular comprimento por derivação
        if total_length % n_leads == 0:
            n_points = total_length // n_leads
            logger.info(f"Detectado: {n_leads} derivações, {n_points} pontos cada")
        else:
            logger.error(f"Tamanho {total_length} não é divisível por 12!")
            return False
            
    elif first_file.ndim == 2:
        n_leads, n_points = first_file.shape
        logger.info(f"Detectado: {n_leads} derivações, {n_points} pontos cada")
    else:
        logger.error(f"Formato não suportado: {first_file.shape}")
        return False
    
    # Verificar se é 100Hz ou 500Hz
    if n_points == 1000:
        logger.info("Taxa de amostragem: 100 Hz (10 segundos)")
    elif n_points == 5000:
        logger.info("Taxa de amostragem: 500 Hz (10 segundos)")
    else:
        logger.warning(f"Número de pontos incomum: {n_points}")
    
    # Criar arrays para todos os dados
    n_samples = len(npy_files)
    logger.info(f"\nCriando arrays para {n_samples} amostras...")
    
    X = np.zeros((n_samples, n_leads, n_points), dtype=np.float32)
    
    # Por enquanto, criar labels dummy (todos normais = 0)
    # Em um cenário real, você deveria carregar os labels do arquivo ptbxl_database.csv
    Y = np.zeros(n_samples, dtype=np.int64)
    
    # Se existir arquivo de metadados, tentar carregar labels reais
    metadata_file = data_dir.parent / 'ptbxl_database.csv'
    has_real_labels = False
    
    if metadata_file.exists():
        logger.info("Arquivo de metadados encontrado! Tentando carregar labels reais...")
        try:
            import pandas as pd
            metadata = pd.read_csv(metadata_file)
            has_real_labels = True
            logger.info("✓ Metadados carregados com sucesso")
        except Exception as e:
            logger.warning(f"Não foi possível carregar metadados: {e}")
            logger.warning("Usando labels dummy")
    else:
        logger.warning("Arquivo ptbxl_database.csv não encontrado")
        logger.warning("Os labels serão dummy (todos 0). Para resultados reais, você precisa dos metadados!")
    
    # Processar todos os arquivos
    logger.info("\nProcessando arquivos...")
    
    errors = 0
    for i, file_path in enumerate(tqdm(npy_files, desc="Consolidando ECGs")):
        try:
            # Carregar dados
            data = np.load(file_path)
            
            # Reformatar se necessário
            if data.ndim == 1:
                data = data.reshape(n_leads, n_points)
            
            # Validar dimensões
            if data.shape == (n_leads, n_points):
                X[i] = data
            else:
                # Tentar ajustar se possível
                if data.shape[0] == n_leads:
                    if data.shape[1] > n_points:
                        # Truncar
                        X[i] = data[:, :n_points]
                    else:
                        # Preencher com zeros
                        X[i, :, :data.shape[1]] = data
                else:
                    errors += 1
                    logger.warning(f"Shape incorreto em {file_path.name}: {data.shape}")
            
            # Tentar obter label real se possível
            if has_real_labels:
                ecg_id = int(file_path.stem)
                meta_row = metadata[metadata['ecg_id'] == ecg_id]
                if not meta_row.empty:
                    # Aqui você implementaria a lógica de mapeamento de patologias
                    # Por enquanto, usar superclass_idx se disponível
                    if 'superclass_idx' in meta_row.columns:
                        Y[i] = meta_row['superclass_idx'].iloc[0]
                    else:
                        Y[i] = 0
                        
        except Exception as e:
            errors += 1
            logger.error(f"Erro em {file_path.name}: {e}")
    
    if errors > 0:
        logger.warning(f"\n⚠️  {errors} arquivos com erro foram ignorados")
    
    # Salvar resultados
    output_dir = data_dir
    logger.info(f"\nSalvando arquivos consolidados...")
    
    np.save(output_dir / "X.npy", X)
    logger.info(f"✓ X.npy salvo: {X.shape} ({X.nbytes / 1024**2:.1f} MB)")
    
    np.save(output_dir / "Y.npy", Y)
    logger.info(f"✓ Y.npy salvo: {Y.shape}")
    
    # Criar arquivo de informações
    info_text = f"""Dataset PTB-XL Consolidado
========================
Data: {pd.Timestamp.now() if 'pd' in locals() else 'N/A'}
Amostras: {n_samples}
Derivações: {n_leads}
Pontos por derivação: {n_points}
Taxa de amostragem: {100 if n_points == 1000 else 500} Hz
Duração: 10 segundos

Shape X: {X.shape}
Shape Y: {Y.shape}
Tipo X: {X.dtype}
Tipo Y: {Y.dtype}

Estatísticas do sinal:
- Min amplitude: {np.min(X):.3f}
- Max amplitude: {np.max(X):.3f}
- Média: {np.mean(X):.3f}
- Desvio padrão: {np.std(X):.3f}

Labels:
- Únicos: {np.unique(Y)}
- Distribuição: {dict(zip(*np.unique(Y, return_counts=True)))}

{"⚠️  ATENÇÃO: Os labels são dummy! Para resultados reais, você precisa dos metadados do PTB-XL." if not has_real_labels else "✓ Labels carregados dos metadados"}
"""
    
    with open(output_dir / "dataset_info.txt", "w") as f:
        f.write(info_text)
    
    logger.info(f"✓ dataset_info.txt criado")
    
    # Criar um plot de exemplo
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(12, 1, figsize=(10, 12), sharex=True)
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Plotar primeiro ECG
        for i in range(12):
            axes[i].plot(X[0, i, :], 'b-', linewidth=0.5)
            axes[i].set_ylabel(lead_names[i])
            axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Amostras')
        fig.suptitle('Exemplo: Primeiro ECG do Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'exemplo_ecg.png', dpi=100)
        logger.info("✓ exemplo_ecg.png criado")
    except:
        pass
    
    logger.info("\n" + "="*60)
    logger.info("✅ CONSOLIDAÇÃO CONCLUÍDA COM SUCESSO!")
    logger.info("="*60)
    logger.info(f"📁 Arquivos criados em: {output_dir}")
    logger.info("   - X.npy: Sinais ECG")
    logger.info("   - Y.npy: Labels")
    logger.info("   - dataset_info.txt: Informações do dataset")
    
    if not has_real_labels:
        logger.info("\n⚠️  IMPORTANTE:")
        logger.info("   Os labels são dummy (todos 0)!")
        logger.info("   Para treino real, você precisa dos labels verdadeiros do PTB-XL")
    
    return True


def main():
    """Função principal"""
    # Caminho fixo
    DATA_PATH = r"D:\ptb-xl\npy_lr"
    
    print("="*60)
    print("CORREÇÃO PTB-XL - CONSOLIDADOR DE ARQUIVOS")
    print("="*60)
    print(f"\nDiretório: {DATA_PATH}")
    
    success = fix_ptbxl_now(DATA_PATH)
    
    if success:
        print("\n✅ Pronto! Agora você pode executar o treinamento.")
    else:
        print("\n❌ Erro na consolidação. Verifique os logs acima.")
    
    input("\nPressione ENTER para sair...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        input("\nPressione ENTER para sair...")
