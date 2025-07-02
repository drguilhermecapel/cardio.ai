import React, { useState } from 'react'
import {
  BrainCircuit,
  AlertTriangle,
  CheckCircle,
  Info,
  Sparkles,
  TrendingUp,
  Heart,
  Activity,
} from 'lucide-react'

interface AIInsight {
  id: string
  type: 'diagnosis' | 'recommendation' | 'alert' | 'info'
  severity: 'low' | 'medium' | 'high' | 'critical'
  title: string
  description: string
  confidence: number
  timestamp: string
  details?: string[]
}

interface AIInsightPanelProps {
  insights?: AIInsight[]
  isProcessing?: boolean
  className?: string
}

const AIInsightPanel: React.FC<AIInsightPanelProps> = ({
  insights,
  isProcessing = false,
  className = '',
}) => {
  const [selectedInsight, setSelectedInsight] = useState<string | null>(null)

  const defaultInsights: AIInsight[] = [
    {
      id: '1',
      type: 'diagnosis',
      severity: 'medium',
      title: 'Possível Arritmia Sinusal',
      description:
        'Detectada variação irregular no ritmo cardíaco. Recomenda-se monitoramento contínuo.',
      confidence: 87,
      timestamp: '2 min atrás',
      details: [
        'Intervalo RR variável detectado',
        'Frequência cardíaca: 68-82 BPM',
        'Padrão consistente com arritmia sinusal respiratória',
      ],
    },
    {
      id: '2',
      type: 'recommendation',
      severity: 'low',
      title: 'Otimização de Medicação',
      description: 'IA sugere ajuste na dosagem do betabloqueador baseado no perfil do paciente.',
      confidence: 92,
      timestamp: '5 min atrás',
      details: [
        'Resposta atual ao tratamento: Boa',
        'Sugestão: Reduzir dose em 25%',
        'Monitorar pressão arterial por 1 semana',
      ],
    },
    {
      id: '3',
      type: 'alert',
      severity: 'high',
      title: 'Alteração no Segmento ST',
      description:
        'Elevação significativa detectada em V2-V4. Atenção médica imediata recomendada.',
      confidence: 95,
      timestamp: '1 min atrás',
      details: [
        'Elevação ST: 2.5mm em V2, 3.1mm em V3',
        'Padrão sugestivo de IAMCSST',
        'Protocolo de emergência ativado',
      ],
    },
  ]

  const displayInsights = insights || defaultInsights

  const getInsightIcon = (type: string): React.ComponentType<{ className?: string }> => {
    switch (type) {
      case 'diagnosis':
        return BrainCircuit
      case 'recommendation':
        return Sparkles
      case 'alert':
        return AlertTriangle
      default:
        return Info
    }
  }

  const getSeverityColors = (
    severity: string
  ): { bg: string; border: string; text: string; glow: string } => {
    switch (severity) {
      case 'critical':
        return {
          bg: 'bg-red-500/10',
          border: 'border-red-500/30',
          text: 'text-red-400',
          glow: 'shadow-red-500/25',
        }
      case 'high':
        return {
          bg: 'bg-orange-500/10',
          border: 'border-orange-500/30',
          text: 'text-orange-400',
          glow: 'shadow-orange-500/25',
        }
      case 'medium':
        return {
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/30',
          text: 'text-yellow-400',
          glow: 'shadow-yellow-500/25',
        }
      default:
        return {
          bg: 'bg-blue-500/10',
          border: 'border-blue-500/30',
          text: 'text-blue-400',
          glow: 'shadow-blue-500/25',
        }
    }
  }

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 90) return 'text-green-400'
    if (confidence >= 70) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div
      className={`bg-gray-900/40 backdrop-blur-xl rounded-2xl border border-gray-700/50 shadow-2xl ${className}`}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 shadow-lg shadow-purple-500/25">
              <BrainCircuit className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">IA Diagnóstica</h3>
              <p className="text-sm text-gray-400">Insights e recomendações em tempo real</p>
            </div>
          </div>

          {isProcessing && (
            <div className="flex items-center space-x-2 px-3 py-2 bg-purple-500/20 rounded-xl border border-purple-500/30">
              <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-purple-400 font-medium">Analisando...</span>
            </div>
          )}
        </div>
      </div>

      {/* Insights List */}
      <div className="p-6 space-y-4 max-h-96 overflow-y-auto">
        {displayInsights.map(insight => {
          const Icon = getInsightIcon(insight.type)
          const colors = getSeverityColors(insight.severity)
          const isSelected = selectedInsight === insight.id

          return (
            <div
              key={insight.id}
              className={`
                ${colors.bg} ${colors.border} backdrop-blur-sm rounded-xl border 
                p-4 transition-all duration-300 cursor-pointer hover:scale-[1.02]
                ${isSelected ? `${colors.glow} shadow-lg` : ''}
              `}
              onClick={() => setSelectedInsight(isSelected ? null : insight.id)}
            >
              <div className="flex items-start space-x-4">
                <div className={`p-2 rounded-lg ${colors.bg} ${colors.border} border`}>
                  <Icon className={`w-5 h-5 ${colors.text}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-white font-semibold truncate">{insight.title}</h4>
                    <div className="flex items-center space-x-2 ml-4">
                      <div
                        className={`text-xs font-medium ${getConfidenceColor(insight.confidence)}`}
                      >
                        {insight.confidence}%
                      </div>
                      <div className="text-xs text-gray-400">{insight.timestamp}</div>
                    </div>
                  </div>

                  <p className="text-gray-300 text-sm mb-3">{insight.description}</p>

                  {/* Confidence bar */}
                  <div className="w-full bg-gray-700 rounded-full h-2 mb-3">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        insight.confidence >= 90
                          ? 'bg-green-400'
                          : insight.confidence >= 70
                            ? 'bg-yellow-400'
                            : 'bg-red-400'
                      }`}
                      style={{ width: `${insight.confidence}%` }}
                    />
                  </div>

                  {/* Expanded details */}
                  {isSelected && insight.details && (
                    <div className="mt-4 pt-4 border-t border-gray-700/50">
                      <h5 className="text-sm font-medium text-gray-300 mb-2">Detalhes:</h5>
                      <ul className="space-y-1">
                        {insight.details.map((detail, index) => (
                          <li
                            key={index}
                            className="text-sm text-gray-400 flex items-start space-x-2"
                          >
                            <CheckCircle className="w-3 h-3 text-green-400 mt-0.5 flex-shrink-0" />
                            <span>{detail}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer with AI stats */}
      <div className="p-6 border-t border-gray-700/50">
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 text-green-400 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm font-semibold">98.2%</span>
            </div>
            <p className="text-xs text-gray-400">Precisão</p>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 text-blue-400 mb-1">
              <Heart className="w-4 h-4" />
              <span className="text-sm font-semibold">1,247</span>
            </div>
            <p className="text-xs text-gray-400">Análises</p>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-1 text-purple-400 mb-1">
              <Activity className="w-4 h-4" />
              <span className="text-sm font-semibold">0.3s</span>
            </div>
            <p className="text-xs text-gray-400">Tempo médio</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AIInsightPanel
