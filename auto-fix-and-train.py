#!/usr/bin/env python3
"""
Sistema Automatizado PTB-XL: Correção + Treinamento
Executa todo o pipeline de forma automática e robusta
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Configurações do sistema
DATA_PATH = r"D:\ptb-xl\npy_lr"
OUTPUT_DIR = "./ensemble_ptbxl_results"

class PTBXLAutomation:
    """Sistema automatizado para processar e treinar com PTB-XL"""
    
    def __init__(self):
        self.data_path = Path(DATA_PATH)
        self.output_dir = Path(OUTPUT_DIR)
        self.start_time = time.time()
        
    def print_header(self, title):
        """Imprime cabeçalho formatado"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70 + "\n")
    
    def check_environment(self):
        """Verifica o ambiente e dependências"""
        self.print_header("VERIFICAÇÃO DO AMBIENTE")
        
        # Verificar Python
        print(f"✓ Python {sys.version.split()[0]}")
        
        # Verificar GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ GPU disponível: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA: {torch.version.cuda}")
            else:
                print("⚠️  GPU não disponível - treinamento será mais lento")
        except:
            print("❌ PyTorch não instalado!")
            return False
        
        # Verificar diretório de dados
        if not self.data_path.exists():
            print(f"❌ Diretório de dados não encontrado: {self.data_path}")
            return False
        else:
            print(f"✓ Diretório de dados: {self.data_path}")
        
        # Contar arquivos
        npy_files = list(self.data_path.glob("*.npy"))
        print(f"✓ Arquivos .npy encontrados: {len(npy_files)}")
        
        return True
    
    def consolidate_data(self):
        """Consolida os arquivos individuais em X.npy e Y.npy"""
        self.print_header("CONSOLIDAÇÃO DOS DADOS")
        
        x_path = self.data_path / "X.npy"
        y_path = self.data_path / "Y.npy"
        
        # Verificar se já existe
        if x_path.exists() and y_path.exists():
            print("📦 Arquivos consolidados já existem!")
            try:
                X = np.load(x_path, mmap_mode='r')
                Y = np.load(y_path)
                print(f"   Shape X: {X.shape}")
                print(f"   Shape Y: {Y.shape}")
                
                # Validar dimensões
                if len(X.shape) == 3 and X.shape[1] == 12:
                    print("✓ Formato correto detectado!")
                    return True
                else:
                    print("⚠️  Formato incorreto, recriando...")
            except Exception as e:
                print(f"⚠️  Erro ao verificar: {e}")
        
        # Executar consolidação
        print("🔧 Iniciando consolidação dos arquivos...")
        
        # Salvar script de consolidação
        consolidation_script = '''
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

data_path = Path(sys.argv[1])
npy_files = sorted([f for f in data_path.glob("*.npy") if f.stem.isdigit()])

# Filtrar range válido
npy_files = [f for f in npy_files if 9004 <= int(f.stem) <= 19417]
print(f"Processando {len(npy_files)} arquivos...")

if len(npy_files) == 0:
    print("Erro: Nenhum arquivo encontrado!")
    sys.exit(1)

# Verificar formato
first = np.load(npy_files[0])
print(f"Formato detectado: {first.shape}")

if first.ndim == 1:
    n_leads = 12
    n_points = len(first) // n_leads
else:
    n_leads, n_points = first.shape

print(f"Configuração: {n_leads} derivações, {n_points} pontos")

# Criar arrays
X = np.zeros((len(npy_files), n_leads, n_points), dtype=np.float32)
Y = np.zeros(len(npy_files), dtype=np.int64)

# Processar
errors = 0
for i, f in enumerate(tqdm(npy_files)):
    try:
        data = np.load(f)
        if data.ndim == 1:
            data = data.reshape(n_leads, n_points)
        X[i] = data
    except:
        errors += 1

print(f"\\nProcessados: {len(npy_files) - errors}")
print(f"Erros: {errors}")

# Salvar
np.save(data_path / "X.npy", X)
np.save(data_path / "Y.npy", Y)

print(f"\\n✓ X.npy: {X.shape}")
print(f"✓ Y.npy: {Y.shape}")
'''
        
        # Executar script
        with open("_temp_consolidate.py", "w") as f:
            f.write(consolidation_script)
        
        try:
            result = subprocess.run(
                [sys.executable, "_temp_consolidate.py", str(self.data_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("\n✅ Consolidação concluída com sucesso!")
                return True
            else:
                print(f"\n❌ Erro na consolidação: {result.stderr}")
                return False
                
        finally:
            # Limpar arquivo temporário
            if Path("_temp_consolidate.py").exists():
                os.remove("_temp_consolidate.py")
    
    def configure_training(self):
        """Configura parâmetros de treinamento"""
        self.print_header("CONFIGURAÇÃO DO TREINAMENTO")
        
        # Configurações padrão otimizadas
        config = {
            "models": ["cnn", "resnet", "inception", "attention"],
            "ensemble_method": "weighted_voting",
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": True,
            "patience": 10
        }
        
        print("🔧 Configurações do Ensemble:")
        print(f"   Modelos: {', '.join(config['models'])}")
        print(f"   Método: {config['ensemble_method']}")
        print(f"   Épocas: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        
        # Perguntar se deseja modificar
        modify = input("\n💡 Usar configurações padrão? (S/n): ").strip().lower()
        
        if modify == 'n':
            # Épocas
            try:
                epochs = int(input("Número de épocas (padrão 50): ") or "50")
                config['epochs'] = epochs
            except:
                pass
            
            # Batch size
            try:
                batch = int(input("Batch size (padrão 32): ") or "32")
                config['batch_size'] = batch
            except:
                pass
        
        return config
    
    def train_ensemble(self, config):
        """Executa o treinamento do ensemble"""
        self.print_header("TREINAMENTO DO ENSEMBLE")
        
        print("🚀 Iniciando treinamento...")
        print("⏱️  Isso pode levar várias horas dependendo do hardware\n")
        
        # Criar diretório de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comando de treinamento
        cmd = [
            sys.executable,
            "ensemble_ecg_training.py",
            "--data-path", str(self.data_path),
            "--models"] + config['models'] + [
            "--ensemble-method", config['ensemble_method'],
            "--epochs", str(config['epochs']),
            "--batch-size", str(config['batch_size']),
            "--output-dir", str(self.output_dir)
        ]
        
        # Log do comando
        with open(self.output_dir / "training_command.txt", "w") as f:
            f.write(" ".join(cmd))
        
        # Executar treinamento
        start_train = time.time()
        
        try:
            # Usar Popen para saída em tempo real
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Capturar saída linha por linha
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                train_time = time.time() - start_train
                print(f"\n✅ Treinamento concluído em {train_time/3600:.1f} horas!")
                return True
            else:
                print("\n❌ Erro durante o treinamento!")
                return False
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Treinamento interrompido pelo usuário!")
            process.terminate()
            return False
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            return False
    
    def analyze_results(self):
        """Analisa os resultados do treinamento"""
        self.print_header("ANÁLISE DOS RESULTADOS")
        
        # Verificar arquivos de resultado
        results_file = self.output_dir / "results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("📊 Métricas Finais:")
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"   Acurácia: {metrics.get('accuracy', 0)*100:.2f}%")
                print(f"   F1-Score: {metrics.get('f1_score', 0)*100:.2f}%")
                print(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
                
                # Métricas por modelo
                for key, value in metrics.items():
                    if 'model_' in key:
                        print(f"   {key}: {value:.4f}")
        
        # Verificar modelos salvos
        print("\n📦 Modelos Salvos:")
        model_files = list(self.output_dir.glob("*.pth"))
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024*1024)
            print(f"   {model_file.name} ({size_mb:.1f} MB)")
        
        # Tempo total
        total_time = time.time() - self.start_time
        print(f"\n⏱️  Tempo total: {total_time/60:.1f} minutos")
    
    def create_inference_script(self):
        """Cria script para inferência"""
        self.print_header("CRIANDO SCRIPT DE INFERÊNCIA")
        
        inference_script = f'''#!/usr/bin/env python3
"""
Script de Inferência para Ensemble PTB-XL
Gerado automaticamente em {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import numpy as np
import torch
from pathlib import Path

# Configurações
MODEL_PATH = "{self.output_dir}/ensemble_best.pth"
DATA_PATH = "{self.data_path}"

def load_model():
    """Carrega o modelo treinado"""
    # Implementar carregamento do modelo
    pass

def preprocess_ecg(ecg_signal):
    """Pré-processa o sinal ECG"""
    # Implementar pré-processamento
    pass

def predict(ecg_signal, model):
    """Faz predição para um sinal ECG"""
    # Implementar predição
    pass

def main():
    # Exemplo de uso
    print("Carregando modelo...")
    model = load_model()
    
    # Carregar um ECG de exemplo
    X = np.load(Path(DATA_PATH) / "X.npy")
    ecg_example = X[0]  # Primeiro ECG
    
    # Fazer predição
    prediction = predict(ecg_example, model)
    print(f"Predição: {{prediction}}")

if __name__ == "__main__":
    main()
'''
        
        inference_file = self.output_dir / "inference_example.py"
        with open(inference_file, "w") as f:
            f.write(inference_script)
        
        print(f"✓ Script de inferência criado: {inference_file}")
    
    def run_complete_pipeline(self):
        """Executa o pipeline completo"""
        print("\n" + "🚀 "*20)
        self.print_header("SISTEMA AUTOMATIZADO PTB-XL")
        print("Bem-vindo ao sistema automatizado de treinamento de ensemble para ECG!")
        print("Este sistema irá:")
        print("  1. Verificar o ambiente")
        print("  2. Consolidar os dados")
        print("  3. Configurar o treinamento")
        print("  4. Treinar o ensemble")
        print("  5. Analisar os resultados")
        
        input("\nPressione ENTER para começar...")
        
        # 1. Verificar ambiente
        if not self.check_environment():
            print("\n❌ Falha na verificação do ambiente!")
            return False
        
        # 2. Consolidar dados
        if not self.consolidate_data():
            print("\n❌ Falha na consolidação dos dados!")
            return False
        
        # 3. Configurar treinamento
        config = self.configure_training()
        
        # Confirmar
        print("\n" + "-"*70)
        print("📋 RESUMO DA CONFIGURAÇÃO:")
        print(f"   Dados: {self.data_path}")
        print(f"   Saída: {self.output_dir}")
        print(f"   Modelos: {', '.join(config['models'])}")
        print(f"   Épocas: {config['epochs']}")
        print("-"*70)
        
        confirm = input("\n🚀 Iniciar treinamento? (S/n): ").strip().lower()
        if confirm == 'n':
            print("Treinamento cancelado.")
            return False
        
        # 4. Treinar
        if not self.train_ensemble(config):
            print("\n❌ Falha no treinamento!")
            return False
        
        # 5. Analisar resultados
        self.analyze_results()
        
        # 6. Criar script de inferência
        self.create_inference_script()
        
        # Finalização
        self.print_header("🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
        print(f"📁 Resultados salvos em: {self.output_dir.absolute()}")
        print("\nPróximos passos:")
        print("  1. Analise os resultados em results.json")
        print("  2. Use ensemble_best.pth para inferência")
        print("  3. Veja inference_example.py para começar")
        
        return True


def main():
    """Função principal"""
    automation = PTBXLAutomation()
    
    try:
        success = automation.run_complete_pipeline()
        
        if not success:
            print("\n⚠️  Pipeline interrompido!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processo interrompido pelo usuário!")
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n\nPressione ENTER para sair...")


if __name__ == "__main__":
    main()
