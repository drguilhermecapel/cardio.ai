"""
Endpoints da API para análise de ECG a partir de imagens
Suporte a JPG, PNG, PDF e outros formatos de imagem
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import io
import base64

# Imports dos serviços
from app.services.ecg_digitizer import process_ecg_image_file
from app.services.model_service_enhanced import get_enhanced_model_service
from app.schemas.fhir import create_ecg_observation

logger = logging.getLogger(__name__)

# Router para endpoints de imagem ECG
router = APIRouter(prefix="/ecg/image", tags=["ECG Image Analysis"])


@router.post("/analyze")
async def analyze_ecg_image(
    patient_id: str = Form(..., description="ID único do paciente"),
    image_file: UploadFile = File(..., description="Arquivo de imagem ECG (JPG, PNG, PDF)"),
    model_name: Optional[str] = Form("demo_ecg_classifier", description="Nome do modelo a usar"),
    create_fhir: bool = Form(True, description="Criar observação FHIR"),
    quality_threshold: float = Form(0.3, description="Threshold mínimo de qualidade (0-1)")
):
    """
    Analisa ECG a partir de arquivo de imagem.
    
    Processo:
    1. Upload da imagem
    2. Digitalização e extração de dados ECG
    3. Análise com modelo de IA
    4. Geração de diagnóstico e recomendações
    5. Criação opcional de observação FHIR
    """
    try:
        logger.info(f"Iniciando análise de imagem ECG para paciente {patient_id}")
        
        # Validar arquivo
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="Nome do arquivo é obrigatório")
        
        # Verificar tipo de arquivo
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']
        file_ext = '.' + image_file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato não suportado. Use: {', '.join(allowed_extensions)}"
            )
        
        # Ler dados da imagem
        image_data = await image_file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        if len(image_data) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=400, detail="Arquivo muito grande (máximo 50MB)")
        
        # Etapa 1: Digitalização da imagem
        logger.info("Iniciando digitalização da imagem ECG")
        digitization_result = process_ecg_image_file(image_data, image_file.filename)
        
        if not digitization_result['success']:
            raise HTTPException(
                status_code=422, 
                detail=f"Erro na digitalização: {digitization_result['error']}"
            )
        
        # Verificar qualidade da extração
        quality_score = digitization_result['metadata']['quality_score']
        if quality_score < quality_threshold:
            logger.warning(f"Qualidade baixa detectada: {quality_score}")
        
        # Etapa 2: Análise com modelo de IA
        logger.info(f"Iniciando análise com modelo {model_name}")
        model_service = get_enhanced_model_service()
        
        # Verificar se modelo existe
        if model_name not in model_service.list_models():
            available_models = model_service.list_models()
            raise HTTPException(
                status_code=404,
                detail=f"Modelo '{model_name}' não encontrado. Disponíveis: {available_models}"
            )
        
        # Analisar cada derivação extraída
        analysis_results = {}
        combined_diagnosis = None
        max_confidence = 0.0
        
        ecg_data = digitization_result['ecg_data']
        
        if not ecg_data:
            raise HTTPException(
                status_code=422,
                detail="Nenhum sinal ECG foi extraído da imagem"
            )
        
        for lead_name, lead_data in ecg_data.items():
            try:
                # Extrair sinal da derivação
                signal = lead_data['signal']
                
                if len(signal) < 100:  # Sinal muito curto
                    continue
                
                # Análise com modelo
                prediction = model_service.predict_ecg(
                    model_name, 
                    signal, 
                    metadata={
                        'lead_name': lead_name,
                        'sampling_rate': lead_data.get('sampling_rate', 500),
                        'patient_id': patient_id,
                        'source': 'image_digitization'
                    }
                )
                
                analysis_results[lead_name] = prediction
                
                # Manter diagnóstico com maior confiança
                if prediction.get('confidence', 0) > max_confidence:
                    max_confidence = prediction.get('confidence', 0)
                    combined_diagnosis = prediction
                    
            except Exception as e:
                logger.error(f"Erro na análise da derivação {lead_name}: {str(e)}")
                analysis_results[lead_name] = {'error': str(e)}
        
        if not combined_diagnosis:
            raise HTTPException(
                status_code=422,
                detail="Não foi possível analisar nenhuma derivação extraída"
            )
        
        # Etapa 3: Consolidar resultados
        consolidated_result = {
            'patient_id': patient_id,
            'analysis_id': f"img_analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            
            # Informações da imagem
            'image_info': {
                'filename': image_file.filename,
                'size_bytes': len(image_data),
                'format': file_ext,
                'dimensions': digitization_result['metadata']['image_size']
            },
            
            # Resultados da digitalização
            'digitization': {
                'success': True,
                'leads_extracted': digitization_result['metadata']['leads_found'],
                'quality_score': quality_score,
                'quality_level': _get_quality_level(quality_score),
                'grid_detected': digitization_result['metadata']['grid_detected'],
                'sampling_rate_estimated': digitization_result['metadata']['sampling_rate_estimated'],
                'calibration_applied': any(
                    lead.get('calibration_applied', False) 
                    for lead in ecg_data.values()
                )
            },
            
            # Diagnóstico principal
            'primary_diagnosis': {
                'diagnosis': combined_diagnosis.get('diagnosis', 'Indeterminado'),
                'confidence': combined_diagnosis.get('confidence', 0.0),
                'confidence_level': combined_diagnosis.get('confidence_level', 'baixa'),
                'predicted_class': combined_diagnosis.get('predicted_class', 0)
            },
            
            # Análise por derivação
            'lead_analysis': analysis_results,
            
            # Recomendações clínicas
            'clinical_recommendations': combined_diagnosis.get('recommendations', {}),
            
            # Modelo usado
            'model_info': {
                'model_name': model_name,
                'model_metadata': model_service.get_model_info(model_name)
            },
            
            # Alertas de qualidade
            'quality_alerts': _generate_quality_alerts(quality_score, digitization_result),
            
            # Preview da digitalização
            'preview_available': bool(digitization_result.get('preview_image'))
        }
        
        # Etapa 4: Criar observação FHIR se solicitado
        fhir_observation = None
        if create_fhir:
            try:
                # Preparar dados para FHIR
                fhir_data = {
                    'signal': str(ecg_data),  # Serializar dados ECG
                    'analysis_results': combined_diagnosis,
                    'source': 'image_digitization'
                }
                
                fhir_observation = create_ecg_observation(
                    patient_id,
                    fhir_data,
                    digitization_result['metadata']['sampling_rate_estimated'],
                    combined_diagnosis
                )
                
                consolidated_result['fhir_observation'] = {
                    'id': fhir_observation.id,
                    'status': fhir_observation.status.value,
                    'resource_type': fhir_observation.resourceType,
                    'created': True
                }
                
            except Exception as e:
                logger.error(f"Erro ao criar observação FHIR: {str(e)}")
                consolidated_result['fhir_observation'] = {
                    'created': False,
                    'error': str(e)
                }
        
        # Log do resultado
        logger.info(
            f"Análise concluída - Paciente: {patient_id}, "
            f"Diagnóstico: {combined_diagnosis.get('diagnosis')}, "
            f"Confiança: {combined_diagnosis.get('confidence', 0):.2f}, "
            f"Qualidade: {quality_score:.2f}"
        )
        
        return JSONResponse(content=consolidated_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de imagem ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.post("/digitize-only")
async def digitize_ecg_image(
    image_file: UploadFile = File(..., description="Arquivo de imagem ECG"),
    return_preview: bool = Form(True, description="Incluir preview da digitalização")
):
    """
    Apenas digitaliza imagem ECG sem análise de IA.
    Útil para verificar qualidade da extração antes da análise.
    """
    try:
        logger.info("Iniciando digitalização apenas")
        
        # Validar arquivo
        if not image_file.filename:
            raise HTTPException(status_code=400, detail="Nome do arquivo é obrigatório")
        
        # Ler dados
        image_data = await image_file.read()
        
        # Digitalizar
        result = process_ecg_image_file(image_data, image_file.filename)
        
        if not result['success']:
            raise HTTPException(status_code=422, detail=result['error'])
        
        # Preparar resposta
        response = {
            'success': True,
            'filename': image_file.filename,
            'metadata': result['metadata'],
            'leads_data': result['ecg_data'],
            'quality_assessment': {
                'score': result['metadata']['quality_score'],
                'level': _get_quality_level(result['metadata']['quality_score']),
                'recommendations': _get_quality_recommendations(result['metadata']['quality_score'])
            }
        }
        
        # Incluir preview se solicitado
        if return_preview and result.get('preview_image'):
            response['preview_image'] = result['preview_image']
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na digitalização: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/supported-formats")
async def get_supported_formats():
    """Retorna formatos de imagem suportados."""
    return {
        'supported_formats': [
            {
                'extension': '.jpg',
                'description': 'JPEG Image',
                'recommended': True
            },
            {
                'extension': '.jpeg',
                'description': 'JPEG Image',
                'recommended': True
            },
            {
                'extension': '.png',
                'description': 'PNG Image',
                'recommended': True
            },
            {
                'extension': '.bmp',
                'description': 'Bitmap Image',
                'recommended': False
            },
            {
                'extension': '.tiff',
                'description': 'TIFF Image',
                'recommended': False
            },
            {
                'extension': '.pdf',
                'description': 'PDF Document',
                'recommended': False,
                'note': 'Requer biblioteca adicional'
            }
        ],
        'max_file_size': '50MB',
        'recommendations': [
            'Use imagens de alta resolução para melhor digitalização',
            'Certifique-se de que o ECG está bem visível e sem cortes',
            'Evite imagens com muito ruído ou baixo contraste',
            'JPG e PNG são os formatos mais recomendados'
        ]
    }


@router.post("/batch-analyze")
async def batch_analyze_ecg_images(
    patient_id: str = Form(...),
    files: list[UploadFile] = File(..., description="Múltiplos arquivos de imagem ECG"),
    model_name: Optional[str] = Form("demo_ecg_classifier")
):
    """
    Análise em lote de múltiplas imagens ECG.
    Útil para análise de ECG seriados ou múltiplas derivações.
    """
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Máximo 10 arquivos por lote")
        
        results = []
        
        for i, file in enumerate(files):
            try:
                logger.info(f"Processando arquivo {i+1}/{len(files)}: {file.filename}")
                
                # Processar cada arquivo individualmente
                image_data = await file.read()
                
                # Digitalizar
                digitization_result = process_ecg_image_file(image_data, file.filename)
                
                if digitization_result['success']:
                    # Analisar com modelo
                    model_service = get_enhanced_model_service()
                    
                    file_results = {
                        'filename': file.filename,
                        'file_index': i,
                        'digitization_success': True,
                        'quality_score': digitization_result['metadata']['quality_score'],
                        'leads_found': digitization_result['metadata']['leads_found'],
                        'analysis_results': {}
                    }
                    
                    # Analisar primeira derivação encontrada
                    ecg_data = digitization_result['ecg_data']
                    if ecg_data:
                        first_lead = list(ecg_data.keys())[0]
                        signal = ecg_data[first_lead]['signal']
                        
                        prediction = model_service.predict_ecg(model_name, signal)
                        file_results['primary_diagnosis'] = prediction.get('diagnosis', 'Indeterminado')
                        file_results['confidence'] = prediction.get('confidence', 0.0)
                    
                else:
                    file_results = {
                        'filename': file.filename,
                        'file_index': i,
                        'digitization_success': False,
                        'error': digitization_result['error']
                    }
                
                results.append(file_results)
                
            except Exception as e:
                logger.error(f"Erro no arquivo {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'file_index': i,
                    'error': str(e)
                })
        
        # Consolidar resultados do lote
        successful_analyses = [r for r in results if r.get('digitization_success', False)]
        
        batch_summary = {
            'patient_id': patient_id,
            'batch_id': f"batch_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_files': len(files),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(files) - len(successful_analyses),
            'average_quality': sum(r.get('quality_score', 0) for r in successful_analyses) / max(len(successful_analyses), 1),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        return JSONResponse(content=batch_summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise em lote: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_quality_level(score: float) -> str:
    """Converte score numérico em nível de qualidade."""
    if score >= 0.8:
        return 'excelente'
    elif score >= 0.6:
        return 'boa'
    elif score >= 0.4:
        return 'moderada'
    elif score >= 0.2:
        return 'baixa'
    else:
        return 'muito_baixa'


def _get_quality_recommendations(score: float) -> list[str]:
    """Gera recomendações baseadas na qualidade."""
    recommendations = []
    
    if score < 0.3:
        recommendations.extend([
            'Qualidade muito baixa - considere nova imagem',
            'Verifique se o ECG está bem visível',
            'Melhore a iluminação e resolução da imagem'
        ])
    elif score < 0.6:
        recommendations.extend([
            'Qualidade moderada - resultados podem ser imprecisos',
            'Considere usar imagem de melhor qualidade se disponível'
        ])
    else:
        recommendations.append('Qualidade adequada para análise')
    
    return recommendations


def _generate_quality_alerts(score: float, digitization_result: dict) -> list[str]:
    """Gera alertas baseados na qualidade da digitalização."""
    alerts = []
    
    if score < 0.3:
        alerts.append('ALERTA: Qualidade muito baixa - resultados não confiáveis')
    
    if not digitization_result['metadata']['grid_detected']:
        alerts.append('AVISO: Grade do ECG não detectada - calibração pode estar incorreta')
    
    if digitization_result['metadata']['leads_found'] < 2:
        alerts.append('AVISO: Poucas derivações detectadas - análise limitada')
    
    return alerts

