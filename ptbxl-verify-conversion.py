#!/usr/bin/env python3
"""
Verificador de Conversão PTB-XL
================================

Verifica e visualiza os dados convertidos para garantir qualidade.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter


class PTBXLVerifier:
    """Classe para verificar dados convertidos do PTB-XL"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = {}
        self.metadata = {}
    
    def load_data(self):
        """Carrega os dados convertidos"""
        print("📂 Carregando dados...")
        
        # Arquivos essenciais
        files_to_load = {
            'X.npy': 'Sinais ECG',
            'Y.npy': 'Labels single-label',
            'Y_multilabel.npy': 'Labels multi-label',
            'ecg_ids.npy': 'IDs dos ECGs',
            'metadata.json': 'Metadados'
        }
        
        for filename, description in files_to_load.items():
            filepath = self.data_path / filename
            if filepath.exists():
                if filename.endswith('.json'):
                    with open(filepath, 'r') as f:
                        self.metadata = json.load(f)
                    print(f"✓ {description} carregado")
                else:
                    # Usar mmap_mode para arquivos grandes
                    self.data[filename.replace('.npy', '')] = np.load(
                        filepath, mmap_mode='r'
                    )
                    print(f"✓ {description} carregado: {self.data[filename.replace('.npy', '')].shape}")
            else:
                print(f"⚠️  {description} não encontrado: {filename}")
    
    def verify_data_integrity(self):
        """Verifica a integridade dos dados"""
        print("\n🔍 Verificando integridade dos dados...")
        
        issues = []
        
        # Verificar dimensões
        if 'X' in self.data:
            X = self.data['X']
            n_samples = X.shape[0]
            
            # Verificar formato esperado
            if X.ndim != 3:
                issues.append(f"X tem {X.ndim} dimensões, esperado 3")
            elif X.shape[1] != 12:
                issues.append(f"X tem {X.shape[1]} derivações, esperado 12")
            
            # Verificar valores
            print(f"  - Range de valores: [{np.min(X):.3f}, {np.max(X):.3f}] mV")
            
            # Verificar NaN/Inf
            if np.any(np.isnan(X)):
                issues.append("X contém valores NaN")
            if np.any(np.isinf(X)):
                issues.append("X contém valores infinitos")
            
            # Verificar consistência com labels
            if 'Y' in self.data and self.data['Y'].shape[0] != n_samples:
                issues.append(f"Inconsistência: X tem {n_samples} amostras, Y tem {self.data['Y'].shape[0]}")
        
        # Verificar labels
        if 'Y' in self.data:
            Y = self.data['Y']
            unique_labels = np.unique(Y)
            print(f"  - Classes únicas: {len(unique_labels)}")
            print(f"  - Range de labels: [{np.min(Y)}, {np.max(Y)}]")
            
            if self.metadata and 'n_conditions' in self.metadata:
                if np.max(Y) >= self.metadata['n_conditions']:
                    issues.append("Labels fora do range esperado")
        
        # Mostrar resultado
        if issues:
            print("\n❌ Problemas encontrados:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ Dados íntegros!")
    
    def show_statistics(self):
        """Mostra estatísticas detalhadas do dataset"""
        print("\n📊 ESTATÍSTICAS DO DATASET")
        print("="*50)
        
        if 'X' in self.data:
            X = self.data['X']
            print(f"Total de amostras: {X.shape[0]:,}")
            print(f"Derivações: {X.shape[1]}")
            print(f"Pontos por sinal: {X.shape[2]:,}")
            
            if self.metadata and 'sampling_rate' in self.metadata:
                duration = X.shape[2] / self.metadata['sampling_rate']
                print(f"Duração do sinal: {duration:.1f} segundos")
                print(f"Taxa de amostragem: {self.metadata['sampling_rate']} Hz")
        
        if 'Y' in self.data:
            Y = self.data['Y']
            print(f"\nDistribuição de classes (top 10):")
            label_counts = Counter(Y)
            total = len(Y)
            
            # Usar metadados para nomes das condições
            if self.metadata and 'idx_to_condition' in self.metadata:
                idx_to_cond = {int(k): v for k, v in self.metadata['idx_to_condition'].items()}
                
                for label, count in label_counts.most_common(10):
                    cond_name = idx_to_cond.get(label, f"Classe {label}")
                    percentage = count / total * 100
                    print(f"  {cond_name}: {count:,} ({percentage:.1f}%)")
            else:
                for label, count in label_counts.most_common(10):
                    percentage = count / total * 100
                    print(f"  Classe {label}: {count:,} ({percentage:.1f}%)")
    
    def plot_sample_ecg(self, sample_idx: int = 0):
        """Plota um ECG de exemplo"""
        if 'X' not in self.data:
            print("⚠️  Dados de ECG não carregados")
            return
        
        X = self.data['X']
        ecg = X[sample_idx]
        
        # Nomes das derivações
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Criar figura
        fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
        fig.suptitle(f'ECG Exemplo #{sample_idx}', fontsize=16)
        
        # Tempo em segundos
        if self.metadata and 'sampling_rate' in self.metadata:
            sr = self.metadata['sampling_rate']
            time = np.arange(ecg.shape[1]) / sr
            xlabel = 'Tempo (s)'
        else:
            time = np.arange(ecg.shape[1])
            xlabel = 'Amostras'
        
        # Plotar cada derivação
        for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
            ax.plot(time, ecg[i], 'b-', linewidth=0.5)
            ax.set_ylabel(f'{lead_name}\n(mV)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2, 2)  # Range típico em mV
            
            # Adicionar label se disponível
            if i == 0 and 'Y' in self.data:
                label = self.data['Y'][sample_idx]
                if self.metadata and 'idx_to_condition' in self.metadata:
                    idx_to_cond = {int(k): v for k, v in self.metadata['idx_to_condition'].items()}
                    cond_name = idx_to_cond.get(label, f"Classe {label}")
                    ax.set_title(f'Diagnóstico: {cond_name}', fontsize=10)
        
        axes[-1].set_xlabel(xlabel)
        plt.tight_layout()
        
        # Salvar figura
        output_file = self.data_path / f'sample_ecg_{sample_idx}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n📈 ECG de exemplo salvo em: {output_file}")
        plt.close()
    
    def plot_signal_quality(self, n_samples: int = 100):
        """Analisa a qualidade dos sinais"""
        if 'X' not in self.data:
            print("⚠️  Dados de ECG não carregados")
            return
        
        print("\n🔬 Analisando qualidade dos sinais...")
        
        X = self.data['X']
        n_samples = min(n_samples, X.shape[0])
        
        # Métricas de qualidade
        snr_values = []
        baseline_drift = []
        
        for i in range(n_samples):
            ecg = X[i]
            
            # Estimar SNR (simplificado)
            for lead in ecg:
                signal_power = np.var(lead)
                noise_estimate = np.var(np.diff(lead)) / 2
                if noise_estimate > 0:
                    snr = 10 * np.log10(signal_power / noise_estimate)
                    snr_values.append(snr)
            
            # Estimar drift de linha de base
            for lead in ecg:
                # Usar média móvel para estimar linha de base
                window = min(len(lead) // 10, 500)
                baseline = np.convolve(lead, np.ones(window)/window, mode='same')
                drift = np.std(baseline)
                baseline_drift.append(drift)
        
        # Plotar histogramas
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # SNR
        ax1.hist(snr_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Distribuição de SNR')
        ax1.axvline(np.median(snr_values), color='red', linestyle='--', 
                   label=f'Mediana: {np.median(snr_values):.1f} dB')
        ax1.legend()
        
        # Drift
        ax2.hist(baseline_drift, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Drift de Linha de Base (mV)')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição de Drift')
        ax2.axvline(np.median(baseline_drift), color='red', linestyle='--',
                   label=f'Mediana: {np.median(baseline_drift):.3f} mV')
        ax2.legend()
        
        plt.tight_layout()
        output_file = self.data_path / 'signal_quality_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Análise de qualidade salva em: {output_file}")
        plt.close()
        
        # Estatísticas
        print(f"\nEstatísticas de Qualidade:")
        print(f"  SNR médio: {np.mean(snr_values):.1f} dB")
        print(f"  SNR mediano: {np.median(snr_values):.1f} dB")
        print(f"  Drift médio: {np.mean(baseline_drift):.3f} mV")
        print(f"  Drift mediano: {np.median(baseline_drift):.3f} mV")
    
    def generate_report(self):
        """Gera relatório completo da verificação"""
        report_file = self.data_path / 'verification_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE VERIFICAÇÃO - PTB-XL\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data: {Path.ctime(self.data_path)}\n")
            f.write(f"Diretório: {self.data_path}\n\n")
            
            # Arquivos
            f.write("ARQUIVOS ENCONTRADOS:\n")
            for file in self.data_path.glob("*"):
                if file.is_file():
                    size_mb = file.stat().st_size / (1024**2)
                    f.write(f"  - {file.name}: {size_mb:.1f} MB\n")
            
            # Dados carregados
            f.write("\nDADOS CARREGADOS:\n")
            for key, value in self.data.items():
                f.write(f"  - {key}: {value.shape}\n")
            
            # Metadados
            if self.metadata:
                f.write("\nMETADADOS:\n")
                f.write(f"  - Taxa de amostragem: {self.metadata.get('sampling_rate')} Hz\n")
                f.write(f"  - Número de condições: {self.metadata.get('n_conditions')}\n")
                f.write(f"  - Data de conversão: {self.metadata.get('conversion_date')}\n")
            
            f.write("\n✅ Verificação concluída\n")
        
        print(f"\n📄 Relatório salvo em: {report_file}")


def main():
    """Função principal"""
    print("="*70)
    print("   🔍 VERIFICADOR DE CONVERSÃO PTB-XL")
    print("="*70)
    print()
    
    # Caminho padrão
    base_path = r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    processed_path = Path(base_path) / 'processed_npy'
    
    if not processed_path.exists():
        print(f"❌ Diretório de dados processados não encontrado: {processed_path}")
        print("\nExecute primeiro o script de conversão!")
        input("\nPressione ENTER para sair...")
        return
    
    # Listar versões disponíveis
    print("📁 Versões disponíveis:")
    versions = [d for d in processed_path.iterdir() if d.is_dir()]
    
    if not versions:
        print("❌ Nenhuma versão convertida encontrada!")
        input("\nPressione ENTER para sair...")
        return
    
    for i, version in enumerate(versions, 1):
        print(f"{i}. {version.name}")
    
    # Escolher versão
    if len(versions) == 1:
        chosen = versions[0]
    else:
        choice = input(f"\nEscolha a versão (1-{len(versions)}): ").strip()
        try:
            idx = int(choice) - 1
            chosen = versions[idx]
        except:
            print("Escolha inválida")
            return
    
    print(f"\n✓ Verificando: {chosen}")
    
    # Criar verificador
    verifier = PTBXLVerifier(chosen)
    
    # Executar verificações
    try:
        # 1. Carregar dados
        verifier.load_data()
        
        # 2. Verificar integridade
        verifier.verify_data_integrity()
        
        # 3. Mostrar estatísticas
        verifier.show_statistics()
        
        # 4. Perguntar sobre visualizações
        visualizar = input("\n📊 Gerar visualizações? (S/n): ").strip().lower()
        if visualizar != 'n':
            print("\nGerando visualizações...")
            
            # ECG de exemplo
            verifier.plot_sample_ecg(sample_idx=0)
            
            # Análise de qualidade
            verifier.plot_signal_quality(n_samples=100)
        
        # 5. Gerar relatório
        verifier.generate_report()
        
        print("\n✅ Verificação concluída com sucesso!")
        
    except Exception as e:
        print(f"\n❌ Erro durante verificação: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPressione ENTER para sair...")


if __name__ == "__main__":
    main()
