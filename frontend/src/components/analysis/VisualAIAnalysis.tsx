// Advanced AI Visual Analysis Component with SHAP Explanations
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  Typography, 
  Button, 
  Badge,
  AIGlow,
  Box,
  CircularProgress,
  Alert
} from '../ui/BasicComponents'
import { AIProcessingAnimation, PulseAnimation } from '../animations/MedicalAnimations'

interface AnalysisLayer {
  id: string
  name: string
  description: string
  confidence: number
  status: 'pending' | 'processing' | 'completed' | 'error'
  findings: string[]
  color: string
}

interface Explanation {
  shap_plot_base64?: { [key: string]: string | null }
  force_plots?: { [key: string]: string | null }
  waterfall_plots?: { [key: string]: string | null }
  feature_importance?: { [key: string]: { [key: string]: number } }
  success?: boolean
  error?: string
}

interface DiagnosticInfo {
  class: string
  probability: number
}

interface AIAnalysisResult {
  id: string
  patientId: string
  timestamp: string
  overallConfidence: number
  riskScore: number
  layers: AnalysisLayer[]
  recommendations: string[]
  criticalFindings: string[]
  diagnostics?: DiagnosticInfo[]
  explanation?: Explanation
}

interface VisualAIAnalysisProps {
  patientData?: any
  ecgData?: any
  diagnostics?: DiagnosticInfo[] | null
  explanation?: Explanation | null
  isLoading?: boolean
  error?: string | null
  onAnalysisComplete?: (result: AIAnalysisResult) => void
  autoStart?: boolean
  className?: string
}

export const VisualAIAnalysis: React.FC<VisualAIAnalysisProps> = ({
  patientData,
  ecgData,
  diagnostics,
  explanation,
  isLoading = false,
  error = null,
  onAnalysisComplete,
  autoStart = false,
  className = ''
}) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [analysisResult, setAnalysisResult] = useState<AIAnalysisResult | null>(null)
  const [progress, setProgress] = useState(0)
  const [showExplanations, setShowExplanations] = useState(false)

  // Analysis layers configuration
  const analysisLayers: AnalysisLayer[] = [
    {
      id: 'signal_quality',
      name: 'Qualidade do Sinal',
      description: 'Avaliando qualidade e integridade dos dados ECG',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#3b82f6'
    },
    {
      id: 'rhythm_analysis',
      name: 'Análise de Ritmo',
      description: 'Detectando padrões de ritmo cardíaco e irregularidades',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#10b981'
    },
    {
      id: 'morphology',
      name: 'Morfologia das Ondas',
      description: 'Analisando formato e características das ondas P, QRS, T',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#f59e0b'
    },
    {
      id: 'intervals',
      name: 'Intervalos e Segmentos',
      description: 'Medindo intervalos PR, QRS, QT e segmentos ST',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#ef4444'
    },
    {
      id: 'arrhythmia',
      name: 'Detecção de Arritmias',
      description: 'Identificando arritmias e distúrbios de condução',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#8b5cf6'
    },
    {
      id: 'risk_assessment',
      name: 'Avaliação de Risco',
      description: 'Calculando scores de risco e prognóstico',
      confidence: 0,
      status: 'pending',
      findings: [],
      color: '#06b6d4'
    }
  ]

  const [layers, setLayers] = useState<AnalysisLayer[]>(analysisLayers)

  // Handle loading state
  if (isLoading) {
    return (
      <Card variant="ai">
        <CardContent className="p-6 text-center">
          <CircularProgress />
          <Typography variant="h6" className="mt-4">
            Analisando ECG com IA...
          </Typography>
        </CardContent>
      </Card>
    )
  }

  // Handle error state
  if (error) {
    return (
      <Alert severity="error" className="mb-4">
        {error}
      </Alert>
    )
  }

  // Simulate AI analysis process
  const runAnalysis = async () => {
    setIsAnalyzing(true)
    setProgress(0)
    setCurrentLayer(0)
    
    // Reset layers
    setLayers(analysisLayers.map(layer => ({ ...layer, status: 'pending', confidence: 0, findings: [] })))

    for (let i = 0; i < layers.length; i++) {
      setCurrentLayer(i)
      
      // Update layer status to processing
      setLayers(prev => prev.map((layer, index) => 
        index === i ? { ...layer, status: 'processing' } : layer
      ))

      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Generate mock results for this layer
      const confidence = 0.75 + Math.random() * 0.24
      const findings = generateLayerFindings(analysisLayers[i].id)
      
      // Update layer with results
      setLayers(prev => prev.map((layer, index) => 
        index === i ? { 
          ...layer, 
          status: 'completed', 
          confidence,
          findings 
        } : layer
      ))

      // Update progress
      setProgress(((i + 1) / layers.length) * 100)
    }

    // Generate final analysis result
    const result: AIAnalysisResult = {
      id: Date.now().toString(),
      patientId: patientData?.id || 'unknown',
      timestamp: new Date().toISOString(),
      overallConfidence: 0.87,
      riskScore: Math.random() * 100,
      layers: layers,
      recommendations: [
        'Monitoramento contínuo recomendado',
        'Considerar ajuste de medicação',
        'Acompanhamento em 30 dias'
      ],
      criticalFindings: Math.random() > 0.7 ? ['Prolongamento QT detectado'] : [],
      diagnostics: diagnostics || [],
      explanation: explanation || null
    }

    setAnalysisResult(result)
    setIsAnalyzing(false)
    onAnalysisComplete?.(result)
  }

  // Generate mock findings for each layer
  const generateLayerFindings = (layerId: string): string[] => {
    const findings: { [key: string]: string[] } = {
      signal_quality: [
        'Sinal de boa qualidade',
        'Ruído mínimo detectado',
        'Baseline estável'
      ],
      rhythm_analysis: [
        'Ritmo sinusal regular',
        'Frequência cardíaca: 75 bpm',
        'Variabilidade normal'
      ],
      morphology: [
        'Ondas P normais',
        'Complexo QRS estreito',
        'Ondas T positivas'
      ],
      intervals: [
        'Intervalo PR: 160ms (normal)',
        'QRS: 90ms (normal)',
        'QT corrigido: 420ms'
      ],
      arrhythmia: [
        'Nenhuma arritmia significativa',
        'Condução AV normal',
        'Sem bloqueios detectados'
      ],
      risk_assessment: [
        'Risco cardiovascular baixo',
        'Score TIMI: 2',
        'Prognóstico favorável'
      ]
    }

    return findings[layerId] || ['Análise concluída']
  }

  // Auto-start analysis
  useEffect(() => {
    if (autoStart && ecgData) {
      runAnalysis()
    }
  }, [autoStart, ecgData])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return '⏳'
      case 'processing': return '🔄'
      case 'completed': return '✅'
      case 'error': return '❌'
      default: return '⏳'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600'
    if (confidence >= 0.7) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <div className={`space-y-6 ${className}`}>
      
      {/* Analysis Control */}
      <Card variant="ai">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <Typography variant="h5" className="font-bold text-purple-700 mb-2">
                Análise Visual de IA
              </Typography>
              <Typography variant="body2" className="text-purple-600">
                Processamento multicamadas com inteligência artificial e explicabilidade
              </Typography>
            </div>
            
            {!isAnalyzing && !analysisResult && (
              <Button
                variant="contained"
                color="ai"
                onClick={runAnalysis}
                disabled={!ecgData}
              >
                🧠 Iniciar Análise
              </Button>
            )}
          </div>

          {/* Progress Bar */}
          {isAnalyzing && (
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <Typography variant="body2" className="text-purple-600">
                  Progresso da Análise
                </Typography>
                <Typography variant="body2" className="text-purple-600">
                  {progress.toFixed(0)}%
                </Typography>
              </div>
              <div className="w-full bg-purple-200 rounded-full h-3">
                <div 
                  className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Diagnostics Results */}
      {diagnostics && diagnostics.length > 0 && (
        <Card variant="medical">
          <CardContent className="p-6">
            <Typography variant="h6" className="font-bold text-gray-900 mb-4">
              🔍 Diagnósticos da IA
            </Typography>
            
            {diagnostics.map((diag, index) => (
              <Box key={index} className="mb-4 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <Typography variant="h6" className="font-bold text-blue-800">
                    {diag.class}
                  </Typography>
                  <Badge 
                    variant={diag.probability > 0.8 ? "success" : diag.probability > 0.6 ? "warning" : "error"}
                    className="text-sm"
                  >
                    {(diag.probability * 100).toFixed(1)}%
                  </Badge>
                </div>
                
                {/* NOVO BLOCO PARA EXIBIR A EXPLICAÇÃO SHAP */}
                {explanation?.force_plots?.[diag.class] && (
                  <Box className="mt-4 border border-gray-300 rounded-lg p-3 bg-white">
                    <Typography variant="subtitle2" className="font-bold text-gray-700 mb-2">
                      🧠 Justificativa do Modelo (SHAP Force Plot) para "{diag.class}"
                    </Typography>
                    <img 
                      src={explanation.force_plots[diag.class]!} 
                      alt={`Explicação SHAP para ${diag.class}`}
                      className="w-full h-auto rounded border"
                    />
                  </Box>
                )}

                {/* Waterfall Plot */}
                {explanation?.waterfall_plots?.[diag.class] && (
                  <Box className="mt-4 border border-gray-300 rounded-lg p-3 bg-white">
                    <Typography variant="subtitle2" className="font-bold text-gray-700 mb-2">
                      📊 Análise de Contribuição (SHAP Waterfall) para "{diag.class}"
                    </Typography>
                    <img 
                      src={explanation.waterfall_plots[diag.class]!} 
                      alt={`Waterfall SHAP para ${diag.class}`}
                      className="w-full h-auto rounded border"
                    />
                  </Box>
                )}

                {/* Feature Importance */}
                {explanation?.feature_importance?.[diag.class] && (
                  <Box className="mt-4 border border-gray-300 rounded-lg p-3 bg-white">
                    <Typography variant="subtitle2" className="font-bold text-gray-700 mb-2">
                      📈 Importância das Features para "{diag.class}"
                    </Typography>
                    <div className="space-y-2">
                      {Object.entries(explanation.feature_importance[diag.class])
                        .slice(0, 5) // Mostrar apenas top 5
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center justify-between">
                            <Typography variant="caption" className="text-gray-600">
                              {feature}
                            </Typography>
                            <div className="flex items-center space-x-2">
                              <div className="w-20 bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-blue-500 h-2 rounded-full"
                                  style={{ width: `${(importance * 100)}%` }}
                                />
                              </div>
                              <Typography variant="caption" className="text-gray-700 font-medium">
                                {importance.toFixed(3)}
                              </Typography>
                            </div>
                          </div>
                        ))}
                    </div>
                  </Box>
                )}
                {/* FIM DO NOVO BLOCO */}
              </Box>
            ))}

            {/* Explanation Controls */}
            {explanation && (
              <div className="mt-4 flex space-x-3">
                <Button 
                  variant="outlined" 
                  color="primary"
                  onClick={() => setShowExplanations(!showExplanations)}
                >
                  {showExplanations ? '🔽 Ocultar' : '🔼 Mostrar'} Explicações Detalhadas
                </Button>
                
                {explanation.success === false && (
                  <Alert severity="warning" className="flex-1">
                    Explicação não disponível: {explanation.error}
                  </Alert>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Analysis Layers */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {layers.map((layer, index) => (
          <AIGlow key={layer.id} active={layer.status === 'processing'}>
            <Card 
              variant={layer.status === 'completed' ? 'medical' : 'default'}
              className={`
                transition-all duration-500 
                ${layer.status === 'processing' ? 'scale-105 shadow-lg' : ''}
                ${index === currentLayer && isAnalyzing ? 'ring-2 ring-purple-400' : ''}
              `}
            >
              <CardContent className="p-4">
                
                {/* Layer Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: layer.color }}
                    />
                    <Typography variant="body2" className="font-medium text-gray-700">
                      {layer.name}
                    </Typography>
                  </div>
                  <span className="text-lg">
                    {getStatusIcon(layer.status)}
                  </span>
                </div>

                {/* Layer Description */}
                <Typography variant="caption" className="text-gray-600 mb-3 block">
                  {layer.description}
                </Typography>

                {/* Processing Animation */}
                {layer.status === 'processing' && (
                  <div className="flex items-center justify-center py-4">
                    <AIProcessingAnimation 
                      stage="analyzing"
                      progress={50}
                    />
                  </div>
                )}

                {/* Results */}
                {layer.status === 'completed' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Typography variant="caption" className="text-gray-600">
                        Confiança
                      </Typography>
                      <Typography 
                        variant="body2" 
                        className={`font-bold ${getConfidenceColor(layer.confidence)}`}
                      >
                        {(layer.confidence * 100).toFixed(1)}%
                      </Typography>
                    </div>
                    
                    <div className="space-y-1">
                      {layer.findings.slice(0, 2).map((finding, i) => (
                        <Typography key={i} variant="caption" className="text-gray-700 block">
                          • {finding}
                        </Typography>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </AIGlow>
        ))}
      </div>

      {/* Final Results */}
      {analysisResult && (
        <Card variant="medical">
          <CardContent className="p-6">
            <Typography variant="h5" className="font-bold text-gray-900 mb-4">
              Resultado da Análise de IA
            </Typography>
            
            {/* Summary Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4 text-center">
                <Typography variant="caption" className="text-purple-600">
                  Confiança Geral
                </Typography>
                <Typography variant="h4" className="font-bold text-purple-700">
                  {(analysisResult.overallConfidence * 100).toFixed(1)}%
                </Typography>
              </div>
              
              <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-4 text-center">
                <Typography variant="caption" className="text-blue-600">
                  Score de Risco
                </Typography>
                <Typography variant="h4" className="font-bold text-blue-700">
                  {analysisResult.riskScore.toFixed(0)}
                </Typography>
              </div>
              
              <div className="bg-gradient-to-r from-green-50 to-yellow-50 rounded-lg p-4 text-center">
                <Typography variant="caption" className="text-green-600">
                  Camadas Analisadas
                </Typography>
                <Typography variant="h4" className="font-bold text-green-700">
                  {layers.filter(l => l.status === 'completed').length}/{layers.length}
                </Typography>
              </div>
            </div>

            {/* Critical Findings */}
            {analysisResult.criticalFindings.length > 0 && (
              <div className="mb-6 p-4 bg-red-50 rounded-lg border border-red-200">
                <Typography variant="h6" className="font-bold text-red-700 mb-2">
                  ⚠️ Achados Críticos
                </Typography>
                {analysisResult.criticalFindings.map((finding, index) => (
                  <Badge key={index} variant="critical" className="mr-2 mb-2">
                    {finding}
                  </Badge>
                ))}
              </div>
            )}

            {/* Recommendations */}
            <div className="mb-6">
              <Typography variant="h6" className="font-bold text-gray-900 mb-3">
                💡 Recomendações da IA
              </Typography>
              <div className="space-y-2">
                {analysisResult.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <span className="text-blue-500">•</span>
                    <Typography variant="body2" className="text-gray-700">
                      {rec}
                    </Typography>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-3">
              <Button variant="contained" color="primary">
                📄 Gerar Relatório
              </Button>
              <Button variant="outlined" color="secondary">
                📤 Exportar Dados
              </Button>
              <Button variant="text" color="ai">
                🔄 Nova Análise
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default VisualAIAnalysis

