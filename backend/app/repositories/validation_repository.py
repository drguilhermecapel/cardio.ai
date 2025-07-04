# backend/app/repositories/validation_repository.py
import pandas as pd
from datetime import datetime
import os
import threading
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationRepository:
    """
    Repositório para persistir as validações dos cardiologistas.
    Para simplicidade, usaremos um arquivo CSV. Em produção, isso seria
    uma tabela em um banco de dados (SQL ou NoSQL).
    """
    def __init__(self, storage_path: str = 'validations_feedback.csv', 
                 backup_dir: str = 'data/validations'):
        self.storage_path = storage_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.lock = threading.Lock()  # Para garantir a segurança em escritas concorrentes
        self._initialize_storage()
        
        logger.info(f"✅ ValidationRepository inicializado: {storage_path}")

    def _initialize_storage(self):
        """Cria o arquivo de armazenamento se ele não existir."""
        try:
            if not os.path.exists(self.storage_path):
                df = pd.DataFrame(columns=[
                    'timestamp', 'ecg_id', 'ai_diagnosis', 'ai_probability',
                    'validator_id', 'is_correct', 'corrected_diagnosis', 'comments',
                    'validation_session_id', 'model_version', 'confidence_score',
                    'processing_time_ms', 'additional_metadata'
                ])
                df.to_csv(self.storage_path, index=False)
                logger.info(f"📄 Arquivo de validação criado: {self.storage_path}")
            else:
                # Verificar integridade do arquivo existente
                try:
                    df = pd.read_csv(self.storage_path)
                    logger.info(f"📄 Arquivo de validação carregado: {len(df)} registros existentes")
                except Exception as e:
                    logger.error(f"❌ Erro ao carregar arquivo existente: {e}")
                    # Fazer backup do arquivo corrompido
                    backup_path = self.backup_dir / f"corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    if os.path.exists(self.storage_path):
                        os.rename(self.storage_path, backup_path)
                        logger.warning(f"⚠️ Arquivo corrompido movido para: {backup_path}")
                    # Criar novo arquivo
                    self._create_new_file()
        except Exception as e:
            logger.error(f"❌ Erro na inicialização do storage: {e}")
            raise

    def _create_new_file(self):
        """Cria um novo arquivo de validação."""
        df = pd.DataFrame(columns=[
            'timestamp', 'ecg_id', 'ai_diagnosis', 'ai_probability',
            'validator_id', 'is_correct', 'corrected_diagnosis', 'comments',
            'validation_session_id', 'model_version', 'confidence_score',
            'processing_time_ms', 'additional_metadata'
        ])
        df.to_csv(self.storage_path, index=False)
        logger.info(f"📄 Novo arquivo de validação criado: {self.storage_path}")

    def save_validation(self, validation_data: Dict[str, Any]) -> bool:
        """
        Salva uma nova entrada de validação.

        Args:
            validation_data (dict): Dicionário contendo os dados da validação.
            
        Returns:
            bool: True se salvou com sucesso, False caso contrário
        """
        with self.lock:
            try:
                # Validar dados obrigatórios
                required_fields = ['ecg_id', 'ai_diagnosis', 'validator_id', 'is_correct']
                for field in required_fields:
                    if field not in validation_data:
                        raise ValueError(f"Campo obrigatório ausente: {field}")

                # Adicionar timestamp se não fornecido
                if 'timestamp' not in validation_data:
                    validation_data['timestamp'] = datetime.utcnow().isoformat()

                # Adicionar ID da sessão de validação se não fornecido
                if 'validation_session_id' not in validation_data:
                    validation_data['validation_session_id'] = f"val_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{validation_data['ecg_id']}"

                # Carregar dados existentes
                df = pd.read_csv(self.storage_path)
                
                # Criar nova entrada
                new_entry = pd.DataFrame([validation_data])
                
                # Concatenar com dados existentes
                df = pd.concat([df, new_entry], ignore_index=True)
                
                # Salvar de volta
                df.to_csv(self.storage_path, index=False)
                
                # Criar backup periódico
                self._create_backup_if_needed(df)
                
                logger.info(f"✅ Validação para ECG ID {validation_data.get('ecg_id')} salva com sucesso.")
                return True
                
            except Exception as e:
                logger.error(f"❌ ERRO ao salvar validação: {e}")
                return False

    def get_validations(self, limit: Optional[int] = None, 
                       validator_id: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recupera validações com filtros opcionais.
        
        Args:
            limit: Número máximo de registros a retornar
            validator_id: Filtrar por ID do validador
            start_date: Data de início (ISO format)
            end_date: Data de fim (ISO format)
            
        Returns:
            Lista de dicionários com as validações
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            # Aplicar filtros
            if validator_id:
                df = df[df['validator_id'] == validator_id]
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
                
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            # Ordenar por timestamp (mais recente primeiro)
            df = df.sort_values('timestamp', ascending=False)
            
            # Aplicar limite
            if limit:
                df = df.head(limit)
            
            # Converter para lista de dicionários
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Erro ao recuperar validações: {e}")
            return []

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        Calcula estatísticas das validações.
        
        Returns:
            Dicionário com estatísticas
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            if df.empty:
                return {
                    "total_validations": 0,
                    "accuracy_rate": 0.0,
                    "most_common_corrections": [],
                    "validator_stats": {},
                    "temporal_stats": {}
                }
            
            # Estatísticas básicas
            total_validations = len(df)
            correct_predictions = len(df[df['is_correct'] == True])
            accuracy_rate = correct_predictions / total_validations if total_validations > 0 else 0.0
            
            # Correções mais comuns
            incorrect_df = df[df['is_correct'] == False]
            most_common_corrections = []
            if not incorrect_df.empty:
                corrections = incorrect_df['corrected_diagnosis'].value_counts().head(5)
                most_common_corrections = [
                    {"diagnosis": diag, "count": int(count)} 
                    for diag, count in corrections.items() if pd.notna(diag)
                ]
            
            # Estatísticas por validador
            validator_stats = {}
            for validator in df['validator_id'].unique():
                if pd.notna(validator):
                    validator_df = df[df['validator_id'] == validator]
                    validator_correct = len(validator_df[validator_df['is_correct'] == True])
                    validator_total = len(validator_df)
                    validator_stats[validator] = {
                        "total_validations": validator_total,
                        "accuracy_rate": validator_correct / validator_total if validator_total > 0 else 0.0
                    }
            
            # Estatísticas temporais (últimos 30 dias)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            recent_df = df[df['timestamp'] >= (datetime.now() - pd.Timedelta(days=30))]
            
            temporal_stats = {
                "last_30_days": {
                    "total_validations": len(recent_df),
                    "accuracy_rate": len(recent_df[recent_df['is_correct'] == True]) / len(recent_df) if len(recent_df) > 0 else 0.0
                }
            }
            
            return {
                "total_validations": total_validations,
                "accuracy_rate": accuracy_rate,
                "most_common_corrections": most_common_corrections,
                "validator_stats": validator_stats,
                "temporal_stats": temporal_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao calcular estatísticas: {e}")
            return {"error": str(e)}

    def get_feedback_for_retraining(self, only_incorrect: bool = True) -> List[Dict[str, Any]]:
        """
        Recupera feedback para re-treinamento do modelo.
        
        Args:
            only_incorrect: Se True, retorna apenas validações incorretas
            
        Returns:
            Lista de validações para re-treinamento
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            if only_incorrect:
                df = df[df['is_correct'] == False]
            
            # Filtrar apenas registros com diagnóstico corrigido
            df = df[df['corrected_diagnosis'].notna()]
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"❌ Erro ao recuperar feedback para re-treinamento: {e}")
            return []

    def _create_backup_if_needed(self, df: pd.DataFrame):
        """Cria backup periódico dos dados."""
        try:
            # Criar backup a cada 100 validações
            if len(df) % 100 == 0:
                backup_filename = f"validations_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_path = self.backup_dir / backup_filename
                df.to_csv(backup_path, index=False)
                logger.info(f"💾 Backup criado: {backup_path}")
                
                # Manter apenas os últimos 10 backups
                self._cleanup_old_backups()
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao criar backup: {e}")

    def _cleanup_old_backups(self):
        """Remove backups antigos, mantendo apenas os 10 mais recentes."""
        try:
            backup_files = list(self.backup_dir.glob("validations_backup_*.csv"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remover backups antigos (manter apenas 10)
            for old_backup in backup_files[10:]:
                old_backup.unlink()
                logger.info(f"🗑️ Backup antigo removido: {old_backup}")
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao limpar backups antigos: {e}")

    def export_validations(self, format: str = 'csv', 
                          output_path: Optional[str] = None) -> str:
        """
        Exporta validações em diferentes formatos.
        
        Args:
            format: Formato de exportação ('csv', 'json', 'excel')
            output_path: Caminho de saída (opcional)
            
        Returns:
            Caminho do arquivo exportado
        """
        try:
            df = pd.read_csv(self.storage_path)
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"validations_export_{timestamp}.{format}"
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Formato não suportado: {format}")
            
            logger.info(f"📤 Validações exportadas para: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erro ao exportar validações: {e}")
            raise

    def clear_validations(self, confirm: bool = False) -> bool:
        """
        Limpa todas as validações (usar com cuidado).
        
        Args:
            confirm: Confirmação de que deseja limpar os dados
            
        Returns:
            True se limpou com sucesso
        """
        if not confirm:
            logger.warning("⚠️ Limpeza cancelada: confirmação necessária")
            return False
        
        try:
            # Criar backup antes de limpar
            backup_path = self.export_validations('csv', 
                f"backup_before_clear_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            
            # Criar novo arquivo vazio
            self._create_new_file()
            
            logger.warning(f"🗑️ Todas as validações foram limpas. Backup salvo em: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao limpar validações: {e}")
            return False

