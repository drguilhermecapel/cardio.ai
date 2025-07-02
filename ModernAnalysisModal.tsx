// Modern Analysis Modal Component with Advanced Visual Effects
// CardioAI Pro - Fusão entre cardiologia e inteligência artificial

import React, { useState, useEffect } from 'react'
import { Typography, Button, Badge, Card, CardContent, CircularProgress, AIGlow } from '../ui/BasicComponents'

interface AnalysisData {
  id: string
  patientName: string
  type: 'ecg' | 'ai-diagnosis' | 'risk-assessment'
  status: 'analyzing' | 'completed' | 'error'
  progress: number
  results?: {
    diagnosis: string
    confidence: number
    riskLevel: 'low' | 'medium' | 'high' | 'critical'
    recommendations: string[]
    technicalDetails: {
      heartRate: number
      rhythm: string
      intervals: {
        pr: number
        qrs: number
        qt: number
      }
    }
  }
  aiInsights?: string[]
  timestamp: string
}

interface AnalysisModalProps {
  isOpen: boolean
  onClose: () => void
  analysis: AnalysisData
  onExport?: () => void
  onShare?: () => void
}

export const ModernAnalysisModal: React.FC<AnalysisModalProps> = ({
  isOpen,
  onClose,
  analysis,
  onExport,
  onShare
}) => {
  const [animationStep, setAnimationStep] = useState(0)

  useEffect(() => {
    if (isOpen && analysis.status === 'analyzing') {
      const interval = setInterval(() => {
        setAnimationStep(prev => (prev + 1) % 4)
      }, 500)
      return () => clearInterval(interval)
    }
  }, [isOpen, analysis.status])

  if (!isOpen) return null

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'ecg': return '💓'
      case 'ai-diagnosis': return '🧠'
      case 'risk-assessment': return '⚠️'
      default: return '📊'
    }
  }

  const getTypeLabel = (type: string) => {
    switch (type) {
      case 'ecg': return 'Análise ECG'
      case 'ai-diagnosis': return 'Diagnóstico IA'
      case 'risk-assessment': return 'Avaliação de Risco'
      default: return 'Análise'
    }
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'success'
      case 'medium': return 'warning'
      case 'high': return 'warning'
      case 'critical': return 'critical'
      default: return 'info'
    }
  }

  return (
    <div className="modal-overlay animate-fade-in">
      <div className="relative w-full max-w-4xl mx-4 max-h-[90vh] overflow-hidden">
        
        {/* Glassmorphism Background */}
        <div className="absolute inset-0 bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/20"></div>
        
        {/* Modal Content */}
        <div className="relative p-8 overflow-y-auto max-h-[90vh]">
          
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-ai rounded-xl flex items-center justify-center shadow-ai">
                <span className="text-2xl">{getTypeIcon(analysis.type)}</span>
              </div>
              <div>
                <Typography variant="h4" className="font-bold text-gray-900">
                  {getTypeLabel(analysis.type)}
                </Typography>
                <Typography variant="body2" className="text-gray-600">
                  Paciente: {analysis.patientName} • {analysis.timestamp}
                </Typography>
              </div>
            </div>
            
            <Button
              variant="text"
              color="primary"
              className="p-2 rounded-full hover:bg-white/50"
              onClick={onClose}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </Button>
          </div>

          {/* Analysis Status */}
          {analysis.status === 'analyzing' && (
            <AIGlow active={true} className="mb-6">
              <Card variant="ai" className="text-center">
                <CardContent className="p-8">
                  <div className="flex flex-col items-center space-y-4">
                    <CircularProgress size={60} color="ai" />
                    <Typography variant="h6" className="font-bold text-purple-700">
                      Analisando com IA...
                    </Typography>
                    <Typography variant="body2" className="text-purple-600">
                      Processando dados médicos • {analysis.progress}% concluído
                    </Typography>
                    
                    {/* Progress Bar */}
                    <div className="w-full max-w-md bg-purple-100 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${analysis.progress}%` }}
                      ></div>
                    </div>
                    
                    {/* Animated Steps */}
                    <div className="flex space-x-4 mt-4">
                      {['Leitura', 'Análise', 'IA', 'Resultado'].map((step, index) => (
                        <div key={step} className="flex flex-col items-center space-y-1">
                          <div className={`
                            w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold
                            ${index <= animationStep 
                              ? 'bg-purple-500 text-white animate-pulse' 
                              : 'bg-gray-200 text-gray-500'
                            }
                          `}>
                            {index + 1}
                          </div>
                          <Typography variant="caption" className="text-gray-600">
                            {step}
                          </Typography>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </AIGlow>
          )}

          {/* Analysis Results */}
          {analysis.status === 'completed' && analysis.results && (
            <div className="space-y-6">
              
              {/* Main Results */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {/* Diagnosis */}
                <Card variant="medical">
                  <CardContent>
                    <div className="flex items-center space-x-2 mb-4">
                      <span className="text-2xl">🩺</span>
                      <Typography variant="h6" className="font-bold text-gray-900">
                        Diagnóstico
                      </Typography>
                    </div>
                    <Typography variant="body1" className="text-gray-700 mb-4">
                      {analysis.results.diagnosis}
                    </Typography>
                    
                    {/* Confidence */}
                    <div className="mb-4">
                      <div className="flex items-center justify-between mb-2">
                        <Typography variant="body2" className="text-gray-600">
                          Confiança da IA:
                        </Typography>
                        <Typography variant="body2" className="font-bold text-purple-600">
                          {(analysis.results.confidence * 100).toFixed(1)}%
                        </Typography>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-1000"
                          style={{ width: `${analysis.results.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    {/* Risk Level */}
                    <div className="flex items-center justify-between">
                      <Typography variant="body2" className="text-gray-600">
                        Nível de Risco:
                      </Typography>
                      <Badge variant={getRiskColor(analysis.results.riskLevel)} className="capitalize">
                        {analysis.results.riskLevel === 'low' ? 'Baixo' : 
                         analysis.results.riskLevel === 'medium' ? 'Médio' : 
                         analysis.results.riskLevel === 'high' ? 'Alto' : 'Crítico'}
                      </Badge>
                    </div>
                  </CardContent>
                </Card>

                {/* Technical Details */}
                <Card variant="default">
                  <CardContent>
                    <div className="flex items-center space-x-2 mb-4">
                      <span className="text-2xl">📊</span>
                      <Typography variant="h6" className="font-bold text-gray-900">
                        Detalhes Técnicos
                      </Typography>
                    </div>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <Typography variant="body2" className="text-gray-600">
                          Freq. Cardíaca:
                        </Typography>
                        <Typography variant="body2" className="font-bold">
                          {analysis.results.technicalDetails.heartRate} bpm
                        </Typography>
                      </div>
                      
                      <div className="flex justify-between">
                        <Typography variant="body2" className="text-gray-600">
                          Ritmo:
                        </Typography>
                        <Typography variant="body2" className="font-bold">
                          {analysis.results.technicalDetails.rhythm}
                        </Typography>
                      </div>
                      
                      <div className="border-t pt-3">
                        <Typography variant="body2" className="text-gray-600 mb-2">
                          Intervalos:
                        </Typography>
                        <div className="grid grid-cols-3 gap-2 text-sm">
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-bold">PR</div>
                            <div>{analysis.results.technicalDetails.intervals.pr}ms</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-bold">QRS</div>
                            <div>{analysis.results.technicalDetails.intervals.qrs}ms</div>
                          </div>
                          <div className="text-center p-2 bg-gray-50 rounded">
                            <div className="font-bold">QT</div>
                            <div>{analysis.results.technicalDetails.intervals.qt}ms</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Recommendations */}
              <Card variant="medical">
                <CardContent>
                  <div className="flex items-center space-x-2 mb-4">
                    <span className="text-2xl">💡</span>
                    <Typography variant="h6" className="font-bold text-gray-900">
                      Recomendações
                    </Typography>
                  </div>
                  <div className="space-y-2">
                    {analysis.results.recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
                        <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm font-bold mt-0.5">
                          {index + 1}
                        </div>
                        <Typography variant="body2" className="text-green-700 flex-1">
                          {rec}
                        </Typography>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* AI Insights */}
              {analysis.aiInsights && analysis.aiInsights.length > 0 && (
                <AIGlow active={true}>
                  <Card variant="ai">
                    <CardContent>
                      <div className="flex items-center space-x-2 mb-4">
                        <span className="text-2xl">🧠</span>
                        <Typography variant="h6" className="font-bold text-purple-700">
                          Insights da IA
                        </Typography>
                      </div>
                      <div className="space-y-3">
                        {analysis.aiInsights.map((insight, index) => (
                          <div key={index} className="p-3 bg-white/50 rounded-lg border border-purple-200">
                            <Typography variant="body2" className="text-purple-700">
                              {insight}
                            </Typography>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </AIGlow>
              )}
            </div>
          )}

          {/* Error State */}
          {analysis.status === 'error' && (
            <Card variant="critical">
              <CardContent className="text-center p-8">
                <div className="flex flex-col items-center space-y-4">
                  <span className="text-6xl">⚠️</span>
                  <Typography variant="h6" className="font-bold text-red-700">
                    Erro na Análise
                  </Typography>
                  <Typography variant="body2" className="text-red-600">
                    Ocorreu um erro durante o processamento. Tente novamente.
                  </Typography>
                  <Button variant="contained" color="critical">
                    Tentar Novamente
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Actions */}
          {analysis.status === 'completed' && (
            <div className="flex justify-end space-x-3 mt-6 pt-6 border-t border-gray-200">
              <Button variant="outlined" color="primary" onClick={onShare}>
                🔗 Compartilhar
              </Button>
              <Button variant="outlined" color="secondary" onClick={onExport}>
                📄 Exportar PDF
              </Button>
              <Button variant="contained" color="primary" onClick={onClose}>
                Fechar
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ModernAnalysisModal

