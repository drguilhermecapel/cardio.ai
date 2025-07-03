import os
import sys
import time
import logging
import psutil
import numpy as np
import pandas as pd
import wfdb
import ast
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import signal as scipy_signal

# Configurar TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

class ECGInterpretationSystem:
    def __init__(self, data_path='data/ptb-xl/', project_name='ECG-AI'):
        self.project_name = project_name
        self.data_path = Path(data_path)
        self.start_time = datetime.now()
        self.sampling_rate = 100  # Hz (usar 100Hz para eficiência)
        self.signal_length = 1000  # 10 segundos a 100Hz
        self.num_leads = 12
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Classes diagnósticas principais
        self.diagnostic_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
        self.setup_logging()
        self.log_system_info()
        
    def setup_logging(self):
        """Configura sistema de logging"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        self.logger = logging.getLogger(self.project_name)
        self.logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        log_file = log_dir / f'ecg_training_{self.start_time.strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Sistema de logging configurado. Log salvo em: {log_file}")
        
    def log_system_info(self):
        """Registra informações do sistema"""
        self.logger.info("="*60)
        self.logger.info(f"SISTEMA DE INTERPRETACAO DE ECG - {self.project_name}")
        self.logger.info(f"Dataset: PTB-XL")
        self.logger.info(f"Iniciado em: {self.start_time}")
        self.logger.info("="*60)
        
        self.logger.info(f"Python versao: {sys.version.split()[0]}")
        self.logger.info(f"TensorFlow versao: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self.logger.info(f"GPUs encontradas: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                self.logger.info(f"  GPU {i}: {gpu.name}")
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
        else:
            self.logger.warning("Nenhuma GPU encontrada - usando CPU")
            
        memory = psutil.virtual_memory()
        self.logger.info(f"RAM total: {memory.total/1e9:.2f} GB")
        self.logger.info(f"RAM disponivel: {memory.available/1e9:.2f} GB")
        self.logger.info(f"CPUs disponiveis: {psutil.cpu_count()}")
        
    def load_ptbxl_data(self):
        """Carrega dataset PTB-XL"""
        self.logger.info("\nCARREGANDO DATASET PTB-XL")
        self.logger.info("-"*40)
        
        # Verificar se o dataset existe
        metadata_path = self.data_path / 'ptbxl_database.csv'
        scp_path = self.data_path / 'scp_statements.csv'
        
        if not metadata_path.exists():
            self.logger.error(f"Dataset PTB-XL nao encontrado em {self.data_path}")
            self.logger.error("Baixe o dataset de: https://physionet.org/content/ptb-xl/1.0.1/")
            return False
            
        # Carregar metadados
        self.logger.info("Carregando metadados...")
        self.df = pd.read_csv(metadata_path, index_col='ecg_id')
        self.logger.info(f"Total de registros: {len(self.df)}")
        
        # Converter strings de diagnóstico
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        # Carregar mapeamento de diagnósticos
        self.agg_df = pd.read_csv(scp_path, index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
        
        # Criar labels para superclasses
        self.logger.info("Processando diagnosticos...")
        self.create_diagnostic_labels()
        
        # Estatísticas do dataset
        self.logger.info("\nEstatisticas do dataset:")
        self.logger.info(f"Idade media: {self.df.age.mean():.1f} anos")
        self.logger.info(f"Distribuicao por sexo: M={len(self.df[self.df.sex==0])}, F={len(self.df[self.df.sex==1])}")
        
        for diag in self.diagnostic_classes:
            count = self.df[diag].sum()
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"{diag}: {count} ({percentage:.1f}%)")
            
        return True
        
    def create_diagnostic_labels(self):
        """Cria labels binárias para cada superclasse diagnóstica"""
        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in self.agg_df.index:
                    tmp.append(self.agg_df.loc[key].diagnostic_class)
            return list(set(tmp))
            
        # Aplicar agregação
        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)
        
        # Criar colunas binárias para cada superclasse
        for diag in self.diagnostic_classes:
            self.df[diag] = self.df.diagnostic_superclass.apply(lambda x: diag in x).astype(int)
            
    def load_ecg_signal(self, filename):
        """Carrega sinal de ECG individual"""
        try:
            # PTB-XL salva em subpastas por faixa de ID
            subfolder = filename[:2]
            filepath = self.data_path / 'records100' / subfolder / filename
            
            # Carregar usando wfdb
            record = wfdb.rdrecord(str(filepath))
            signal = record.p_signal
            
            return signal
        except Exception as e:
            self.logger.error(f"Erro ao carregar {filename}: {str(e)}")
            return None
            
    def preprocess_signal(self, signal):
        """Preprocessa sinal de ECG"""
        # Verificar shape
        if signal.shape != (self.signal_length, self.num_leads):
            self.logger.warning(f"Shape incorreto: {signal.shape}")
            return None
            
        # Remover baseline wander com filtro passa-alta
        sos = scipy_signal.butter(2, 0.5, 'high', fs=self.sampling_rate, output='sos')
        signal_filtered = scipy_signal.sosfiltfilt(sos, signal, axis=0)
        
        # Normalizar por lead (Z-score)
        scaler = StandardScaler()
        signal_normalized = scaler.fit_transform(signal_filtered)
        
        # Clip valores extremos
        signal_clipped = np.clip(signal_normalized, -5, 5)
        
        return signal_clipped
        
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Prepara dados para treinamento"""
        self.logger.info("\nPREPARANDO DADOS PARA TREINAMENTO")
        self.logger.info("-"*40)
        
        # Usar subset para teste rápido (remover em produção)
        subset_size = 5000
        self.logger.info(f"Usando subset de {subset_size} amostras para teste")
        df_subset = self.df.head(subset_size)
        
        # Carregar sinais
        X = []
        y = []
        failed = 0
        
        self.logger.info("Carregando sinais de ECG...")
        for idx, (ecg_id, row) in enumerate(df_subset.iterrows()):
            if idx % 500 == 0:
                self.logger.info(f"Progresso: {idx}/{len(df_subset)}")
                self.monitor_resources()
                
            # Carregar sinal
            signal = self.load_ecg_signal(row.filename_lr)
            if signal is None:
                failed += 1
                continue
                
            # Preprocessar
            signal_processed = self.preprocess_signal(signal)
            if signal_processed is None:
                failed += 1
                continue
                
            X.append(signal_processed)
            y.append([row[diag] for diag in self.diagnostic_classes])
            
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Sinais carregados: {len(X)}")
        self.logger.info(f"Falhas: {failed}")
        self.logger.info(f"Shape dos dados: X={X.shape}, y={y.shape}")
        
        # Dividir dados
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y[:, 0]
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp[:, 0]
        )
        
        self.logger.info(f"Divisao dos dados:")
        self.logger.info(f"  Treino: {len(X_train)} amostras")
        self.logger.info(f"  Validacao: {len(X_val)} amostras")
        self.logger.info(f"  Teste: {len(X_test)} amostras")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    def create_ecg_model(self):
        """Cria modelo CNN-LSTM para interpretação de ECG"""
        self.logger.info("\nCRIANDO MODELO DE DEEP LEARNING PARA ECG")
        self.logger.info("-"*40)
        
        inputs = tf.keras.Input(shape=(self.signal_length, self.num_leads))
        
        # Processar cada lead independentemente
        lead_outputs = []
        for i in range(self.num_leads):
            # Extrair lead individual
            lead = tf.keras.layers.Lambda(lambda x: x[:, :, i:i+1])(inputs)
            
            # CNN para extração de features
            x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(lead)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            
            x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            
            x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # LSTM para padrões temporais
            x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
            x = tf.keras.layers.LSTM(32)(x)
            
            lead_outputs.append(x)
            
        # Combinar todas as leads
        combined = tf.keras.layers.concatenate(lead_outputs)
        
        # Camadas densas finais
        x = tf.keras.layers.Dense(256, activation='relu')(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Output multi-label
        outputs = tf.keras.layers.Dense(len(self.diagnostic_classes), activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar com métricas médicas
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', multi_label=True),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='sensitivity')  # Sensibilidade em cardiologia
            ]
        )
        
        self.logger.info(f"Modelo criado com {model.count_params():,} parametros")
        self.logger.info(f"Input shape: {(self.signal_length, self.num_leads)}")
        self.logger.info(f"Output shape: {len(self.diagnostic_classes)} classes")
        
        return model
        
    def monitor_resources(self):
        """Monitora uso de recursos"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        self.logger.debug(f"CPU: {cpu_percent}% | RAM: {memory.percent}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
        
    def calculate_medical_metrics(self, y_true, y_pred, threshold=0.5):
        """Calcula métricas específicas para cardiologia"""
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics = {}
        for i, diag in enumerate(self.diagnostic_classes):
            # Apenas calcular se houver casos positivos
            if y_true[:, i].sum() == 0:
                continue
                
            TP = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 1))
            TN = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 0))
            FP = np.sum((y_true[:, i] == 0) & (y_pred_binary[:, i] == 1))
            FN = np.sum((y_true[:, i] == 1) & (y_pred_binary[:, i] == 0))
            
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0
            
            metrics[diag] = {
                'sensibilidade': sensitivity,
                'especificidade': specificity,
                'VPP': ppv,  # Valor Preditivo Positivo
                'VPN': npv,  # Valor Preditivo Negativo
                'prevalencia': y_true[:, i].mean()
            }
            
        return metrics
        
    def train_model(self, model, train_data, val_data, epochs=30, batch_size=32):
        """Treina modelo com monitoramento detalhado"""
        self.logger.info("\nINICIANDO TREINAMENTO DO MODELO")
        self.logger.info(f"Epocas: {epochs}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info("-"*40)
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_ecg_model.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Callback customizado para métricas médicas
        class MedicalMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger, X_val, y_val, diagnostic_classes):
                self.logger = logger
                self.X_val = X_val
                self.y_val = y_val
                self.diagnostic_classes = diagnostic_classes
                
            def on_epoch_end(self, epoch, logs=None):
                # Log métricas padrão
                metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in logs.items() if not k.startswith('val_')])
                val_metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in logs.items() if k.startswith('val_')])
                
                self.logger.info(f"Epoca {epoch+1}:")
                self.logger.info(f"  Treino: {metrics_str}")
                self.logger.info(f"  Valid.: {val_metrics_str}")
                
                # Calcular métricas médicas a cada 5 épocas
                if (epoch + 1) % 5 == 0:
                    y_pred = self.model.predict(self.X_val, verbose=0)
                    parent = self.model.history.model  # Acessar instância pai
                    metrics = parent.trainer.calculate_medical_metrics(self.y_val, y_pred)
                    
                    self.logger.info("  Metricas medicas por diagnostico:")
                    for diag, m in metrics.items():
                        self.logger.info(f"    {diag}: Sens={m['sensibilidade']:.3f}, Spec={m['especificidade']:.3f}")
                        
        # Adicionar callback customizado
        medical_callback = MedicalMetricsCallback(self.logger, X_val, y_val, self.diagnostic_classes)
        medical_callback.model = model
        model.trainer = self  # Referência para calcular métricas
        callbacks.append(medical_callback)
        
        # Treinar
        try:
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("\nTreinamento concluido com sucesso")
            return history
            
        except KeyboardInterrupt:
            self.logger.warning("Treinamento interrompido pelo usuario")
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {str(e)}", exc_info=True)
            
    def evaluate_model(self, model, test_data):
        """Avalia modelo no conjunto de teste"""
        self.logger.info("\nAVALIACAO FINAL NO CONJUNTO DE TESTE")
        self.logger.info("-"*40)
        
        X_test, y_test = test_data
        
        # Predições
        y_pred = model.predict(X_test, verbose=1)
        
        # Métricas gerais
        test_loss, test_acc, test_auc, test_prec, test_sens = model.evaluate(X_test, y_test, verbose=0)
        
        self.logger.info(f"Loss: {test_loss:.4f}")
        self.logger.info(f"Acuracia: {test_acc:.4f}")
        self.logger.info(f"AUC-ROC: {test_auc:.4f}")
        self.logger.info(f"Precisao: {test_prec:.4f}")
        self.logger.info(f"Sensibilidade: {test_sens:.4f}")
        
        # Métricas médicas detalhadas
        metrics = self.calculate_medical_metrics(y_test, y_pred)
        
        self.logger.info("\nMETRICAS POR DIAGNOSTICO:")
        for diag, m in metrics.items():
            self.logger.info(f"\n{diag}:")
            self.logger.info(f"  Prevalencia: {m['prevalencia']:.3f}")
            self.logger.info(f"  Sensibilidade: {m['sensibilidade']:.3f}")
            self.logger.info(f"  Especificidade: {m['especificidade']:.3f}")
            self.logger.info(f"  VPP: {m['VPP']:.3f}")
            self.logger.info(f"  VPN: {m['VPN']:.3f}")
            
    def generate_report(self):
        """Gera relatório final"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("RELATORIO FINAL - SISTEMA DE INTERPRETACAO DE ECG")
        self.logger.info("="*60)
        self.logger.info(f"Dataset: PTB-XL")
        self.logger.info(f"Duracao total: {duration}")
        self.logger.info(f"Finalizado em: {end_time}")
        self.logger.info(f"Modelo salvo em: best_ecg_model.h5")
        self.logger.info(f"Logs salvos em: logs/")
        self.logger.info("\nPROXIMOS PASSOS:")
        self.logger.info("1. Validar com cardiologistas")
        self.logger.info("2. Testar em dados externos")
        self.logger.info("3. Implementar explicabilidade (Grad-CAM)")
        self.logger.info("4. Otimizar thresholds por patologia")
        
    def run(self):
        """Executa pipeline completo"""
        try:
            # Carregar dados
            if not self.load_ptbxl_data():
                return
                
            # Preparar dados
            train_data, val_data, test_data = self.prepare_data()
            
            # Criar modelo
            model = self.create_ecg_model()
            
            # Treinar
            history = self.train_model(model, train_data, val_data, epochs=20)
            
            # Avaliar
            self.evaluate_model(model, test_data)
            
            # Relatório final
            self.generate_report()
            
        except Exception as e:
            self.logger.error(f"Erro fatal: {str(e)}", exc_info=True)
        finally:
            self.logger.info("\nSistema finalizado")

def main():
    print("Sistema de Interpretacao de ECG com Deep Learning")
    print("Dataset: PTB-XL")
    print("Pressione Ctrl+C para interromper\n")
    
    # Ajuste o caminho para onde você baixou o PTB-XL
    system = ECGInterpretationSystem(data_path='data/ptb-xl/')
    system.run()

if __name__ == "__main__":
    main()