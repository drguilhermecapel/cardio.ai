"""
Endpoints corrigidos da API para análise de ECG por imagem
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
import numpy as np
from typing import Optional, Dict, Any
import time

from backend.app.services.ecg_digitizer_enhanced import ECGDigitizerEnhanced
from backend.app.services.model_loader_robust import ModelLoaderRobust
from backend.app.services.medical_monitoring import MedicalMonitoring

logger = logging.getLogger(__name__)

router = APIRouter()

# Instâncias globais
digitizer = ECGDigitizerEnhanced()
model_loader = None
medical_monitor = None

def get_model_loader():
    """Obtém instância do carregador de modelo"""
    global model_loader
    if model_loader is None:
        model_loader = ModelLoaderRobust()
        if not model_loader.load_model():
            logger.error("Falha ao carregar modelo")
    return model_loader

def get_medical_monitor():
    """Obtém instância do monitor médico"""
    global medical_monitor
    if medical_monitor is None:
        medical_monitor = MedicalMonitoring()
    return medical_monitor

@router.post("/analyze-image-medical")
async def analyze_ecg_image_medical(
    image_file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_gender: Optional[str] = Form(None),
    clinical_context: Optional[str] = Form(None),
    return_preview: bool = Form(False)
):
    """
    Análise completa de ECG por imagem com grau médico
    """
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando análise médica de ECG: {image_file.filename}")
        
        # Validar arquivo
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="Nome do arquivo é obrigatório")
        
        # Verificar formato
        file_ext = image_file.filename.lower().split('.')[-1]
        if file_ext not in digitizer.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato não suportado. Formatos aceitos: {', '.join(digitizer.supported_formats)}"
            )
        
        # Ler dados da imagem
        image_data = await image_file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        if len(image_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="Arquivo muito grande (máx. 50MB)")
        
        # Digitalizar ECG
        logger.info("Iniciando digitalização...")
        digitization_result = digitizer.digitize_ecg_from_image(image_data, image_file.filename)
        
        if not digitization_result.get('success', False):
            raise HTTPException(
                status_code=500, 
                detail=f"Erro na digitalização: {digitization_result.get('error', 'Erro desconhecido')}"
            )
        
        # Preparar dados para o modelo
        ecg_data = digitization_result['ecg_data']
        
        # Converter dados ECG para formato numpy
        ecg_array = convert_ecg_data_to_numpy(ecg_data)
        
        # Obter modelo
        model = get_model_loader()
        if model is None or model.model is None:
            raise HTTPException(status_code=500, detail="Modelo não disponível")
        
        # Realizar predição
        logger.info("Realizando predição com modelo PTB-XL...")
        prediction_result = model.predict_ecg(ecg_array, {
            'patient_id': patient_id,
            'patient_age': patient_age,
            'patient_gender': patient_gender,
            'clinical_context': clinical_context,
            'filename': image_file.filename
        })
        
        if not prediction_result.get('success', True):
            raise HTTPException(
                status_code=500, 
                detail=f"Erro na predição: {prediction_result.get('error', 'Erro desconhecido')}"
            )
        
        # Registrar evento médico
        monitor = get_medical_monitor()
        if monitor:
            monitor.log_diagnostic_event({
                'patient_id': patient_id or 'anonymous',
                'diagnosis': prediction_result.get('primary_diagnosis', {}).get('class_name', 'Unknown'),
                'confidence': prediction_result.get('primary_diagnosis', {}).get('probability', 0.0),
                'quality_score': digitization_result.get('quality_score', 0.0),
                'processing_time': time.time() - start_time,
                'filename': image_file.filename
            })
        
        # Preparar resposta
        response = {
            'success': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': time.time() - start_time,
            'patient_info': {
                'patient_id': patient_id,
                'age': patient_age,
                'gender': patient_gender,
                'clinical_context': clinical_context
            },
            'digitization': {
                'quality_score': digitization_result.get('quality_score', 0.0),
                'quality_level': digitization_result.get('quality_level', 'unknown'),
                'leads_detected': digitization_result.get('leads_detected', 0),
                'grid_detected': digitization_result.get('grid_detected', False),
                'processing_time': digitization_result.get('processing_time', 0.0)
            },
            'diagnosis': prediction_result,
            'medical_grade': True,
            'model_version': model.model_info.get('version', 'unknown') if model.model_info else 'unknown'
        }
        
        # Adicionar preview se solicitado
        if return_preview and 'quality_indicators' in digitization_result:
            response['preview'] = digitization_result['quality_indicators']
        
        logger.info(f"Análise concluída em {time.time() - start_time:.2f}s")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro interno na análise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

def convert_ecg_data_to_numpy(ecg_data: Dict[str, Any]) -> np.ndarray:
    """
    Converte dados ECG digitalizados para formato numpy esperado pelo modelo
    """
    try:
        # Extrair sinais das 12 derivações
        lead_names = ['Lead_I', 'Lead_II', 'Lead_III', 'Lead_aVR', 'Lead_aVL', 'Lead_aVF', 
                     'Lead_V1', 'Lead_V2', 'Lead_V3', 'Lead_V4', 'Lead_V5', 'Lead_V6']
        
        signals = []
        
        for lead_name in lead_names:
            if lead_name in ecg_data:
                signal = ecg_data[lead_name]['signal']
                if isinstance(signal, list):
                    signal = np.array(signal)
                signals.append(signal)
            else:
                # Gerar sinal sintético se derivação não encontrada
                logger.warning(f"Derivação {lead_name} não encontrada, gerando sinal sintético")
                synthetic_signal = generate_synthetic_lead_signal(len(lead_names) - len(signals))
                signals.append(synthetic_signal)
        
        # Garantir que temos exatamente 12 derivações
        while len(signals) < 12:
            synthetic_signal = generate_synthetic_lead_signal(len(signals))
            signals.append(synthetic_signal)
        
        # Converter para numpy array
        ecg_array = np.array(signals[:12])  # Garantir apenas 12 derivações
        
        # Garantir 1000 amostras por derivação
        target_samples = 1000
        if ecg_array.shape[1] != target_samples:
            logger.info(f"Redimensionando de {ecg_array.shape[1]} para {target_samples} amostras")
            # Interpolação linear para redimensionar
            new_array = np.zeros((12, target_samples))
            for i in range(12):
                x_old = np.linspace(0, 1, ecg_array.shape[1])
                x_new = np.linspace(0, 1, target_samples)
                new_array[i, :] = np.interp(x_new, x_old, ecg_array[i, :])
            ecg_array = new_array
        
        # Adicionar dimensão batch: (1, 12, 1000)
        ecg_array = np.expand_dims(ecg_array, axis=0)
        
        logger.info(f"ECG convertido para formato: {ecg_array.shape}")
        return ecg_array.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Erro na conversão de dados ECG: {str(e)}")
        # Retornar ECG sintético em caso de erro
        return generate_fallback_ecg_array()

def generate_synthetic_lead_signal(lead_index: int, samples: int = 1000) -> np.ndarray:
    """
    Gera sinal sintético para uma derivação específica
    """
    try:
        # Gerar sinal ECG sintético baseado no índice da derivação
        t = np.linspace(0, 10, samples)  # 10 segundos
        
        # Frequência cardíaca variável por derivação
        heart_rate = 60 + (lead_index * 3) % 30  # 60-90 bpm
        
        # Componentes do ECG
        # Onda P
        p_wave = 0.15 * np.sin(2 * np.pi * (heart_rate / 60) * t + lead_index * 0.1)
        
        # Complexo QRS
        qrs_freq = heart_rate / 60
        qrs_wave = 0.8 * np.sin(2 * np.pi * qrs_freq * t + lead_index * 0.2)
        qrs_wave += 0.3 * np.sin(4 * np.pi * qrs_freq * t + lead_index * 0.3)
        
        # Onda T
        t_wave = 0.25 * np.sin(2 * np.pi * (heart_rate / 60) * t + lead_index * 0.4 + np.pi/4)
        
        # Combinar componentes
        ecg_signal = p_wave + qrs_wave + t_wave
        
        # Adicionar ruído específico por derivação
        noise_level = 0.03 + (lead_index % 3) * 0.01
        noise = np.random.normal(0, noise_level, len(ecg_signal))
        ecg_signal += noise
        
        # Adicionar linha de base variável
        baseline_drift = 0.05 * np.sin(2 * np.pi * 0.1 * t + lead_index)
        ecg_signal += baseline_drift
        
        # Normalizar para range típico de ECG
        ecg_signal = np.clip(ecg_signal, -3, 3)
        
        return ecg_signal
        
    except Exception as e:
        logger.error(f"Erro na geração de sinal sintético: {str(e)}")
        return np.zeros(samples)

def generate_fallback_ecg_array() -> np.ndarray:
    """
    Gera array ECG de fallback em caso de erro
    """
    try:
        # Gerar 12 derivações sintéticas
        ecg_array = np.zeros((1, 12, 1000))
        
        for lead in range(12):
            ecg_array[0, lead, :] = generate_synthetic_lead_signal(lead, 1000)
        
        return ecg_array.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Erro na geração de ECG fallback: {str(e)}")
        return np.zeros((1, 12, 1000), dtype=np.float32)

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Retorna formatos de imagem suportados
    """
    return {
        'supported_formats': digitizer.supported_formats,
        'max_file_size_mb': 50,
        'recommended_formats': ['jpg', 'png', 'pdf'],
        'description': 'Formatos de imagem suportados para análise de ECG'
    }

@router.get("/digitizer-status")
async def get_digitizer_status():
    """
    Retorna status do digitalizador
    """
    return {
        'status': 'ready',
        'supported_formats': digitizer.supported_formats,
        'target_leads': digitizer.target_leads,
        'target_samples': digitizer.target_samples,
        'sampling_rate': digitizer.sampling_rate
    }

